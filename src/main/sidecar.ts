import { ChildProcess, spawn } from 'child_process'
import { createServer } from 'net'
import { existsSync } from 'fs'
import { join } from 'path'
import { app } from 'electron'
import type { EchoResult, SidecarStatus } from '../shared/types'

const HEALTH_TIMEOUT_MS = 30_000
const HEALTH_POLL_INTERVAL_MS = 250
const REQUEST_TIMEOUT_MS = 5_000
const STDERR_TAIL_LINES = 20

/**
 * Owns the Python sidecar process: picks a free port, spawns the FastAPI
 * server bound to 127.0.0.1, waits for /health, and proxies requests to it.
 * The renderer never talks to the sidecar directly — everything goes through
 * this class via IPC.
 */
export class Sidecar {
  private child: ChildProcess | null = null
  private currentStatus: SidecarStatus = { state: 'stopped' }
  private listeners = new Set<(status: SidecarStatus) => void>()
  private stderrTail: string[] = []
  private stopping = false

  get status(): SidecarStatus {
    return this.currentStatus
  }

  onStatusChange(listener: (status: SidecarStatus) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  async start(): Promise<void> {
    if (this.child) return
    this.stopping = false
    this.stderrTail = []
    this.setStatus({ state: 'starting' })

    try {
      const port = await getFreePort()
      const pythonPath = this.resolvePython()
      const child = spawn(pythonPath, ['-u', 'server.py', '--port', String(port)], {
        cwd: this.sidecarDir(),
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        windowsHide: true
      })
      this.child = child

      child.stdout?.on('data', (chunk) => console.log('[sidecar]', String(chunk).trimEnd()))
      child.stderr?.on('data', (chunk) => {
        const text = String(chunk).trimEnd()
        console.error('[sidecar:err]', text)
        this.stderrTail.push(text)
        if (this.stderrTail.length > STDERR_TAIL_LINES) this.stderrTail.shift()
      })
      child.on('error', (err) => {
        this.child = null
        this.setStatus({ state: 'error', error: `Failed to spawn Python (${pythonPath}): ${err.message}` })
      })
      child.on('exit', (code) => {
        this.child = null
        if (!this.stopping) {
          const tail = this.stderrTail.join('\n')
          this.setStatus({
            state: 'error',
            error: `Sidecar exited unexpectedly (code ${code}).${tail ? `\n${tail}` : ''}`
          })
        }
      })

      const health = await this.waitForHealth(port)
      this.setStatus({ state: 'online', port, pid: child.pid, python: health.python })
    } catch (err) {
      this.setStatus({ state: 'error', error: err instanceof Error ? err.message : String(err) })
    }
  }

  async ping(message: string): Promise<EchoResult> {
    const { state, port } = this.currentStatus
    if (state !== 'online' || !port) {
      throw new Error(`Sidecar is not online (state: ${state})`)
    }
    const startedAt = performance.now()
    const res = await fetch(`http://127.0.0.1:${port}/echo`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
    })
    if (!res.ok) throw new Error(`Sidecar /echo returned HTTP ${res.status}`)
    const data = (await res.json()) as Omit<EchoResult, 'mainLatencyMs'>
    return { ...data, mainLatencyMs: Math.round((performance.now() - startedAt) * 10) / 10 }
  }

  stop(): void {
    this.stopping = true
    if (this.child) {
      this.child.kill()
      this.child = null
    }
    this.setStatus({ state: 'stopped' })
  }

  private async waitForHealth(port: number): Promise<{ python: string }> {
    const deadline = Date.now() + HEALTH_TIMEOUT_MS
    while (Date.now() < deadline) {
      // Spawn failure or early exit already produced a better error message.
      if (!this.child) throw new Error(this.currentStatus.error ?? 'Sidecar process exited during startup')
      try {
        const res = await fetch(`http://127.0.0.1:${port}/health`, {
          signal: AbortSignal.timeout(1_000)
        })
        if (res.ok) return (await res.json()) as { python: string }
      } catch {
        // Not accepting connections yet — keep polling.
      }
      await delay(HEALTH_POLL_INTERVAL_MS)
    }
    throw new Error(`Sidecar did not become healthy within ${HEALTH_TIMEOUT_MS / 1000}s`)
  }

  private sidecarDir(): string {
    return app.isPackaged
      ? join(process.resourcesPath, 'sidecar')
      : join(app.getAppPath(), 'sidecar')
  }

  private resolvePython(): string {
    const venvPython =
      process.platform === 'win32'
        ? join(this.sidecarDir(), '.venv', 'Scripts', 'python.exe')
        : join(this.sidecarDir(), '.venv', 'bin', 'python')
    if (existsSync(venvPython)) return venvPython
    // Fall back to a system Python so `npm run dev` still works before
    // `npm run sidecar:setup` — deps may be missing, which surfaces as an
    // early-exit error with the pip hint in stderr.
    return process.platform === 'win32' ? 'python' : 'python3'
  }

  private setStatus(status: SidecarStatus): void {
    this.currentStatus = status
    for (const listener of this.listeners) listener(status)
  }
}

function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 0
      server.close(() => (port ? resolve(port) : reject(new Error('Failed to allocate a port'))))
    })
  })
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
