import { ChildProcess, spawn } from 'child_process'
import { createServer } from 'net'
import { existsSync, mkdirSync } from 'fs'
import { join } from 'path'
import { app } from 'electron'
import type { SidecarStatus } from '../shared/types'

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
      const pythonPath = await this.ensurePython()
      const dataDir = join(app.getPath('userData'), 'tek-data')
      mkdirSync(dataDir, { recursive: true })
      const child = spawn(
        pythonPath,
        ['-u', 'server.py', '--port', String(port), '--data-dir', dataDir],
        {
          cwd: this.sidecarDir(),
          stdio: ['ignore', 'pipe', 'pipe'],
          env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
          windowsHide: true
        }
      )
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

  get baseUrl(): string {
    const { state, port } = this.currentStatus
    if (state !== 'online' || !port) {
      throw new Error(`Sidecar is not online (state: ${state})`)
    }
    return `http://127.0.0.1:${port}`
  }

  /** JSON request to the sidecar; throws with the API's error detail on 4xx/5xx. */
  async request<T>(path: string, init?: { method?: string; body?: unknown; timeoutMs?: number }): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: init?.method ?? (init?.body !== undefined ? 'POST' : 'GET'),
      headers: { 'content-type': 'application/json' },
      body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
      signal: AbortSignal.timeout(init?.timeoutMs ?? REQUEST_TIMEOUT_MS)
    })
    if (!res.ok) {
      let detail = `HTTP ${res.status}`
      try {
        const data = (await res.json()) as { detail?: string }
        if (data.detail) detail = data.detail
      } catch {
        // non-JSON error body — keep the status text
      }
      throw new Error(detail)
    }
    return (await res.json()) as T
  }

  /** Raw streaming POST (NDJSON endpoints like /chat). Caller owns the body. */
  async stream(path: string, body: unknown, signal: AbortSignal): Promise<Response> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal
    })
    if (!res.ok || !res.body) {
      throw new Error(`Sidecar ${path} returned HTTP ${res.status}`)
    }
    return res
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

  /** The venv lives in the repo during dev, in userData when packaged
   * (resources/ is read-only on installed apps). */
  private venvDir(): string {
    return app.isPackaged
      ? join(app.getPath('userData'), 'sidecar-venv')
      : join(this.sidecarDir(), '.venv')
  }

  private venvPython(): string {
    return process.platform === 'win32'
      ? join(this.venvDir(), 'Scripts', 'python.exe')
      : join(this.venvDir(), 'bin', 'python')
  }

  private async ensurePython(): Promise<string> {
    const venvPython = this.venvPython()
    if (existsSync(venvPython)) return venvPython
    if (!app.isPackaged) {
      // Dev convenience: fall back to system Python; missing deps surface as
      // an early-exit error pointing at `npm run sidecar:setup`.
      return process.platform === 'win32' ? 'python' : 'python3'
    }
    // Packaged first run: build the environment once. Requires a system
    // Python 3.10+ — installer-bundled runtime is a future improvement.
    const systemPython = process.platform === 'win32' ? 'python' : 'python3'
    console.log('[tek] first run — creating Python environment (one time, ~1 min)')
    try {
      await runOnce(systemPython, ['-m', 'venv', this.venvDir()])
      await runOnce(venvPython, [
        '-m', 'pip', 'install', '--no-input', '--quiet',
        '-r', join(this.sidecarDir(), 'requirements.txt')
      ])
    } catch (err) {
      throw new Error(
        'Tek could not set up its Python engine. Install Python 3.10+ from python.org ' +
          `and relaunch. (${err instanceof Error ? err.message : String(err)})`
      )
    }
    return venvPython
  }

  private setStatus(status: SidecarStatus): void {
    this.currentStatus = status
    for (const listener of this.listeners) listener(status)
  }
}

function runOnce(command: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true })
    let stderr = ''
    child.stderr?.on('data', (chunk) => {
      stderr += String(chunk)
    })
    child.on('error', reject)
    child.on('exit', (code) =>
      code === 0 ? resolve() : reject(new Error(stderr.trim().slice(-400) || `exit code ${code}`))
    )
  })
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
