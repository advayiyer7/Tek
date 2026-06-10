import { app, BrowserWindow, ipcMain, shell } from 'electron'
import { join } from 'path'
import { Sidecar } from './sidecar'

const sidecar = new Sidecar()
let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 720,
    minWidth: 760,
    minHeight: 540,
    show: false,
    title: 'Tek',
    backgroundColor: '#0b0e14',
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })

  mainWindow.on('ready-to-show', () => mainWindow?.show())
  mainWindow.on('closed', () => {
    mainWindow = null
  })

  // Open external links in the system browser, never inside the app.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })

  if (process.env['ELECTRON_RENDERER_URL']) {
    void mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    void mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

ipcMain.handle('sidecar:get-status', () => sidecar.status)
ipcMain.handle('sidecar:ping', (_event, message: unknown) => {
  if (typeof message !== 'string' || message.length === 0 || message.length > 10_000) {
    throw new Error('Invalid message')
  }
  return sidecar.ping(message)
})

sidecar.onStatusChange((status) => {
  mainWindow?.webContents.send('sidecar:status', status)

  // Startup self-test: prove the main -> sidecar leg works and log it, so the
  // round-trip is verifiable from the terminal without touching the UI.
  if (status.state === 'online') {
    sidecar
      .ping('startup self-test')
      .then((result) =>
        console.log(`[tek] sidecar round-trip ok in ${result.mainLatencyMs}ms — ${result.reply}`)
      )
      .catch((err) => console.error('[tek] startup self-test failed:', err))
  }
})

app.whenReady().then(() => {
  void sidecar.start()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => sidecar.stop())
