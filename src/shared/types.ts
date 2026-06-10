// Types shared between the Electron main process, preload, and renderer.

export type SidecarState = 'stopped' | 'starting' | 'online' | 'error'

export interface SidecarStatus {
  state: SidecarState
  /** Port the sidecar is listening on (127.0.0.1), when online. */
  port?: number
  pid?: number
  /** Python version reported by the sidecar's /health endpoint. */
  python?: string
  error?: string
}

export interface EchoResult {
  reply: string
  length: number
  python: string
  /** Time spent on the main -> sidecar HTTP leg, in milliseconds. */
  mainLatencyMs: number
}
