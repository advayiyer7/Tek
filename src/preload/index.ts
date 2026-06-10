import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron'
import type {
  ActionOperation,
  ChatEvent,
  ChatTurn,
  DedupeResult,
  ExecResult,
  IndexStatus,
  OllamaStatus,
  PlanResult,
  SearchResponse,
  SidecarHealth,
  SidecarStatus,
  TekSettings
} from '../shared/types'

const api = {
  getStatus: (): Promise<SidecarStatus> => ipcRenderer.invoke('sidecar:get-status'),
  health: (): Promise<SidecarHealth> => ipcRenderer.invoke('sidecar:health'),
  onStatus: (callback: (status: SidecarStatus) => void): (() => void) => {
    const listener = (_e: IpcRendererEvent, status: SidecarStatus): void => callback(status)
    ipcRenderer.on('sidecar:status', listener)
    return () => {
      ipcRenderer.removeListener('sidecar:status', listener)
    }
  },

  settings: {
    get: (): Promise<TekSettings> => ipcRenderer.invoke('settings:get'),
    set: (update: Partial<TekSettings>): Promise<TekSettings> =>
      ipcRenderer.invoke('settings:set', update)
  },

  pickFolder: (): Promise<string | null> => ipcRenderer.invoke('dialog:pick-folder'),

  index: {
    start: (): Promise<{ started: boolean; alreadyRunning: boolean }> =>
      ipcRenderer.invoke('index:start'),
    status: (): Promise<IndexStatus> => ipcRenderer.invoke('index:status')
  },

  search: (query: string, k?: number): Promise<SearchResponse> =>
    ipcRenderer.invoke('search:query', query, k),

  ollamaStatus: (): Promise<OllamaStatus> => ipcRenderer.invoke('ollama:status'),

  chat: {
    start: (question: string, history?: ChatTurn[]): Promise<string> =>
      ipcRenderer.invoke('chat:start', question, history ?? []),
    cancel: (id: string): Promise<void> => ipcRenderer.invoke('chat:cancel', id),
    onEvent: (callback: (payload: { id: string; event: ChatEvent }) => void): (() => void) => {
      const listener = (_e: IpcRendererEvent, payload: { id: string; event: ChatEvent }): void =>
        callback(payload)
      ipcRenderer.on('chat:event', listener)
      return () => {
        ipcRenderer.removeListener('chat:event', listener)
      }
    }
  },

  actions: {
    dedupe: (folder: string): Promise<DedupeResult> => ipcRenderer.invoke('actions:dedupe', folder),
    organize: (folder: string, strategy: 'by-type' | 'by-date'): Promise<PlanResult> =>
      ipcRenderer.invoke('actions:organize', folder, strategy),
    rename: (paths: string[]): Promise<PlanResult> => ipcRenderer.invoke('actions:rename', paths),
    summarize: (path: string): Promise<{ summary?: string; error?: string }> =>
      ipcRenderer.invoke('actions:summarize', path),
    execute: (operations: ActionOperation[]): Promise<ExecResult[]> =>
      ipcRenderer.invoke('actions:execute', operations)
  },

  file: {
    open: (path: string): Promise<string> => ipcRenderer.invoke('file:open', path),
    reveal: (path: string): Promise<void> => ipcRenderer.invoke('file:reveal', path)
  },

  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node
  }
}

export type TekApi = typeof api

contextBridge.exposeInMainWorld('tek', api)
