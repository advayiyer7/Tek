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

export interface TekSettings {
  folders: string[]
  embed_model: string
  llm_model: string
  watch_enabled: boolean
  cloud_enabled: boolean
}

export interface IndexStatus {
  state: 'idle' | 'loading-model' | 'scanning' | 'indexing' | 'done' | 'error'
  running: boolean
  totalFiles: number
  processedFiles: number
  indexedFiles: number
  skippedFiles: number
  removedFiles: number
  totalChunks: number
  currentPath: string
  error: string
  elapsedS: number
  stats: { files: number; chunks: number }
}

export interface SearchResult {
  path: string
  name: string
  chunkIndex: number
  text: string
  score: number
}

export interface SearchResponse {
  results: SearchResult[]
  tookMs: number
}

export interface Citation {
  ref: number
  path: string
  name: string
  score: number
  preview: string
}

export type ChatEvent =
  | { type: 'citations'; citations: Citation[] }
  | { type: 'token'; text: string }
  | { type: 'fallback'; reason: 'ollama-offline' | 'model-missing' | 'no-results'; text: string }
  | { type: 'done' }
  | { type: 'error'; error: string }
  | { type: 'closed' }

export interface OllamaStatus {
  available: boolean
  version: string | null
  models: string[]
  configuredModel: string
}

export type ActionKind = 'move' | 'rename' | 'trash'

export interface ActionOperation {
  kind: ActionKind
  src: string
  dest?: string
  reason?: string
}

export interface DedupeResult {
  groups: { hash: string; size: number; keep: string; duplicates: string[] }[]
  operations: ActionOperation[]
  wastedBytes: number
}

export interface PlanResult {
  operations: ActionOperation[]
  errors?: string[]
  error?: string
}

export interface ExecResult {
  op: ActionOperation
  ok: boolean
  error?: string
}

export interface SidecarHealth {
  status: string
  version: string
  python: string
  embedModel: { name: string; ready: boolean; loading: boolean; error: string | null }
  index: { files: number; chunks: number }
}
