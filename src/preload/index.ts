import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron'
import type { EchoResult, SidecarStatus } from '../shared/types'

const api = {
  getStatus: (): Promise<SidecarStatus> => ipcRenderer.invoke('sidecar:get-status'),
  ping: (message: string): Promise<EchoResult> => ipcRenderer.invoke('sidecar:ping', message),
  onStatus: (callback: (status: SidecarStatus) => void): (() => void) => {
    const listener = (_event: IpcRendererEvent, status: SidecarStatus): void => callback(status)
    ipcRenderer.on('sidecar:status', listener)
    return () => {
      ipcRenderer.removeListener('sidecar:status', listener)
    }
  },
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node
  }
}

export type TekApi = typeof api

contextBridge.exposeInMainWorld('tek', api)
