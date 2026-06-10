// Focused chat E2E: verifies grounded streaming answers + multi-turn follow-up.
import { _electron as electron } from 'playwright'
import { existsSync, mkdirSync, writeFileSync, rmSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import os from 'os'

const repo = dirname(dirname(fileURLToPath(import.meta.url)))
const shots = join(os.tmpdir(), 'tek-e2e-shots')
mkdirSync(shots, { recursive: true })

const demo = join(os.tmpdir(), 'tek-demo-notes')
rmSync(demo, { recursive: true, force: true })
mkdirSync(demo, { recursive: true })
writeFileSync(
  join(demo, 'car.md'),
  '# Car maintenance\nThe Subaru needs an oil change every 6,000 miles. Last done at 48,200. Winter tires are stored at the shop on Fern Street.'
)
writeFileSync(
  join(demo, 'lease.md'),
  '# Apartment lease notes\nLease renews September 1. Rent is $2,140/month. 60-day notice required to leave.'
)

const log = (...a) => console.log('[drive]', ...a)
const app = await electron.launch({ args: [repo], cwd: repo })
const page = await app.firstWindow()
page.on('console', (m) => console.log('[renderer]', m.type(), m.text()))
page.on('pageerror', (e) => console.log('[pageerror]', e.message))

let port = null
for (let i = 0; i < 360; i++) {
  const s = await page.evaluate(() => window.tek.getStatus())
  if (s.state === 'online') { port = s.port; break }
  if (s.state === 'error') { console.error('SIDECAR ERROR:', s.error); await app.close(); process.exit(1) }
  await page.waitForTimeout(500)
}
log('sidecar port', port)
const base = `http://127.0.0.1:${port}`
const j = async (r) => { if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`); return r.json() }

const settings = await j(await fetch(base + '/settings'))
const keep = settings.folders.filter((f) => existsSync(f) && f !== demo)
await j(await fetch(base + '/settings', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ folders: [...keep, demo] }) }))
await j(await fetch(base + '/index/start', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }))
for (let i = 0; i < 600; i++) {
  const s = await j(await fetch(base + '/index/status'))
  if (!s.running && (s.state === 'done' || s.state === 'error')) { log('index:', s.state, s.stats.files, 'files'); break }
  await new Promise((r) => setTimeout(r, 1000))
}

// sanity: retrieval ranking over HTTP, no UI interference
const s1 = await j(await fetch(base + '/search', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ query: 'when does my lease renew and how much is rent', k: 5 }) }))
log('HTTP search "lease renew/rent":', s1.results.map((r) => `${r.name}:${r.score}`).join(', '), `(${s1.tookMs}ms)`)

// chat through the real UI
await page.getByRole('button', { name: 'Chat' }).first().click()
const chatBox = page.getByPlaceholder(/Ask anything/)
const askBtn = page.getByRole('button', { name: 'Ask' })

const askAndWait = async (q, shot) => {
  await chatBox.click()
  await chatBox.fill(q)
  await askBtn.click()
  await page.waitForTimeout(800)
  const bubbles = await page.locator('.whitespace-pre-wrap').count()
  log(`after asking ${JSON.stringify(q)}: ${bubbles} answer bubble(s) present`)
  // wait until streaming finishes (Ask button returns) and bubble has text
  await page.waitForSelector('button:has-text("Ask")', { timeout: 240_000 })
  await page.waitForTimeout(500)
  const text = await page.locator('.whitespace-pre-wrap').last().textContent().catch(() => '(none)')
  const cites = await page.locator('button:has(span:text("["))').allTextContents().catch(() => [])
  log('answer:', JSON.stringify(text))
  log('citations:', JSON.stringify(cites))
  await page.screenshot({ path: join(shots, shot) })
}

await askAndWait('what mileage was the last oil change done at?', '5-chat-answer.png')
await askAndWait('and where are the winter tires kept?', '6-chat-followup.png')

// cleanup
await j(await fetch(base + '/settings', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ folders: keep }) }))
if (keep.length) {
  await j(await fetch(base + '/index/start', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }))
  for (let i = 0; i < 600; i++) {
    const s = await j(await fetch(base + '/index/status'))
    if (!s.running && (s.state === 'done' || s.state === 'error')) break
    await new Promise((r) => setTimeout(r, 1000))
  }
}
await app.close()
rmSync(demo, { recursive: true, force: true })
log('DONE')
