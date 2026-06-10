// Drives the INSTALLED Tek app (packaged build): verifies the first-run venv
// bootstrap, sidecar health, and search over the user's existing index.
import { _electron as electron } from 'playwright'
import { join } from 'path'
import { mkdirSync } from 'fs'
import os from 'os'

const exe = join(process.env.LOCALAPPDATA, 'Programs', 'Tek', 'Tek.exe')
const shots = join(os.tmpdir(), 'tek-e2e-shots')
mkdirSync(shots, { recursive: true })
const log = (...a) => console.log('[drive]', ...a)

log('launching', exe)
const app = await electron.launch({ executablePath: exe })
const page = await app.firstWindow()
page.on('console', (m) => { if (/error|fail/i.test(m.text())) console.log('[renderer]', m.text()) })

// First packaged run creates the venv + pip installs — allow up to 8 minutes.
let port = null
let lastState = ''
for (let i = 0; i < 960; i++) {
  const s = await page.evaluate(() => window.tek.getStatus()).catch(() => ({ state: 'starting' }))
  if (s.state !== lastState) { lastState = s.state; log('sidecar state:', s.state) }
  if (s.state === 'online') { port = s.port; break }
  if (s.state === 'error') { console.error('SIDECAR ERROR:', s.error); await page.screenshot({ path: join(shots, '7-installed-error.png') }); await app.close(); process.exit(1) }
  await page.waitForTimeout(500)
}
if (!port) { console.error('sidecar never came online'); await app.close(); process.exit(1) }

const base = `http://127.0.0.1:${port}`
const health = await (await fetch(base + '/health')).json()
log('health:', JSON.stringify({ version: health.version, python: health.python, reranker: health.reranker, index: health.index }))

// Search through the real UI against whatever is already indexed.
await page.getByRole('button', { name: 'Search' }).first().click()
const box = page.getByPlaceholder(/Search your files/)
await box.fill('lecture about trusted operating system design')
await box.press('Enter')
await page.waitForSelector('text=/result/', { timeout: 180_000 })
await page.waitForTimeout(600)
const top = await page.locator('li button').first().textContent().catch(() => '(no results)')
log('installed-app search top hit:', top)
await page.screenshot({ path: join(shots, '8-installed-search.png') })

await app.close()
log('INSTALLED E2E DONE')
