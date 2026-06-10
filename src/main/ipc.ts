import { dialog, ipcMain, shell, BrowserWindow } from 'electron'
import { randomUUID } from 'crypto'
import { promises as fs } from 'fs'
import { dirname } from 'path'
import type { ActionOperation, ChatEvent, ExecResult } from '../shared/types'
import type { Sidecar } from './sidecar'

const activeChats = new Map<string, AbortController>()

export function registerIpc(sidecar: Sidecar): void {
  ipcMain.handle('sidecar:get-status', () => sidecar.status)
  ipcMain.handle('sidecar:health', () => sidecar.request('/health'))

  // -- settings ------------------------------------------------------------
  ipcMain.handle('settings:get', () => sidecar.request('/settings'))
  ipcMain.handle('settings:set', (_e, update: unknown) =>
    sidecar.request('/settings', { method: 'PUT', body: update })
  )

  ipcMain.handle('dialog:pick-folder', async (event) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    if (!win) return null
    const result = await dialog.showOpenDialog(win, {
      properties: ['openDirectory'],
      title: 'Choose a folder for Tek to index'
    })
    return result.canceled ? null : result.filePaths[0]
  })

  // -- indexing --------------------------------------------------------------
  ipcMain.handle('index:start', () => sidecar.request('/index/start', { method: 'POST', body: {} }))
  ipcMain.handle('index:status', () => sidecar.request('/index/status'))

  // -- retrieval -------------------------------------------------------------
  ipcMain.handle('search:query', (_e, query: unknown, k: unknown) => {
    if (typeof query !== 'string' || !query.trim()) throw new Error('Invalid query')
    // First search after launch may lazy-load the embedding model.
    return sidecar.request('/search', {
      body: { query, k: typeof k === 'number' ? k : 10 },
      timeoutMs: 120_000
    })
  })

  ipcMain.handle('ollama:status', () => sidecar.request('/ollama/status', { timeoutMs: 8_000 }))

  // -- chat (streamed over webContents events) -------------------------------
  ipcMain.handle('chat:start', (event, question: unknown) => {
    if (typeof question !== 'string' || !question.trim() || question.length > 4000) {
      throw new Error('Invalid question')
    }
    const id = randomUUID()
    const controller = new AbortController()
    activeChats.set(id, controller)
    void pumpChat(sidecar, id, question, controller, event.sender)
    return id
  })

  ipcMain.handle('chat:cancel', (_e, id: unknown) => {
    if (typeof id === 'string') activeChats.get(id)?.abort()
  })

  // -- file actions -----------------------------------------------------------
  ipcMain.handle('actions:dedupe', (_e, folder: unknown) =>
    sidecar.request('/actions/dedupe', { body: { folder }, timeoutMs: 600_000 })
  )
  ipcMain.handle('actions:organize', (_e, folder: unknown, strategy: unknown) =>
    sidecar.request('/actions/organize', { body: { folder, strategy }, timeoutMs: 60_000 })
  )
  ipcMain.handle('actions:rename', (_e, paths: unknown) =>
    sidecar.request('/actions/rename', { body: { paths }, timeoutMs: 600_000 })
  )
  ipcMain.handle('actions:summarize', (_e, path: unknown) =>
    sidecar.request('/actions/summarize', { body: { path }, timeoutMs: 300_000 })
  )

  // THE mutation gate. Only reachable from the renderer's confirm dialog —
  // the sidecar itself can never touch the filesystem destructively.
  ipcMain.handle('actions:execute', (_e, ops: unknown) => executeOperations(ops))

  // -- opening files ----------------------------------------------------------
  ipcMain.handle('file:open', async (_e, path: unknown) => {
    if (typeof path !== 'string') return 'invalid path'
    return shell.openPath(path) // returns '' on success, error string otherwise
  })
  ipcMain.handle('file:reveal', (_e, path: unknown) => {
    if (typeof path === 'string') shell.showItemInFolder(path)
  })
}

async function pumpChat(
  sidecar: Sidecar,
  id: string,
  question: string,
  controller: AbortController,
  sender: Electron.WebContents
): Promise<void> {
  const send = (event: ChatEvent): void => {
    if (!sender.isDestroyed()) sender.send('chat:event', { id, event })
  }
  try {
    const res = await sidecar.stream('/chat', { question }, controller.signal)
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let newline: number
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (line) send(JSON.parse(line) as ChatEvent)
      }
    }
    if (buffer.trim()) send(JSON.parse(buffer) as ChatEvent)
  } catch (err) {
    if (!controller.signal.aborted) {
      send({ type: 'error', error: err instanceof Error ? err.message : String(err) })
    }
  } finally {
    activeChats.delete(id)
    send({ type: 'closed' })
  }
}

async function executeOperations(raw: unknown): Promise<ExecResult[]> {
  if (!Array.isArray(raw) || raw.length === 0 || raw.length > 1000) {
    throw new Error('Invalid operations list')
  }
  const results: ExecResult[] = []
  for (const item of raw as ActionOperation[]) {
    const op: ActionOperation = {
      kind: item.kind,
      src: String(item.src ?? ''),
      dest: item.dest === undefined ? undefined : String(item.dest),
      reason: item.reason
    }
    try {
      await executeOne(op)
      results.push({ op, ok: true })
    } catch (err) {
      results.push({ op, ok: false, error: err instanceof Error ? err.message : String(err) })
    }
  }
  return results
}

async function executeOne(op: ActionOperation): Promise<void> {
  const stat = await fs.stat(op.src).catch(() => null)
  if (!stat?.isFile()) throw new Error('source is not a file (was it already moved?)')

  if (op.kind === 'trash') {
    // Recycle bin, never a hard delete — recoverable by design.
    await shell.trashItem(op.src)
    return
  }
  if (op.kind !== 'move' && op.kind !== 'rename') throw new Error(`unknown operation: ${op.kind}`)
  if (!op.dest) throw new Error('missing destination')
  if (await fs.stat(op.dest).then(() => true, () => false)) {
    throw new Error('destination already exists')
  }
  await fs.mkdir(dirname(op.dest), { recursive: true })
  try {
    await fs.rename(op.src, op.dest)
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'EXDEV') {
      // Cross-volume move: copy + remove.
      await fs.copyFile(op.src, op.dest)
      await fs.unlink(op.src)
    } else {
      throw err
    }
  }
}
