// Temporary E2E driver: launches the built app with Playwright, indexes a
// demo corpus alongside any existing folders, then drives Search and Chat
// through the real UI. Not part of the product; safe to delete.
import { _electron as electron } from 'playwright'
import { existsSync, mkdirSync, writeFileSync, rmSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import os from 'os'

const repo = dirname(dirname(fileURLToPath(import.meta.url)))
const shots = join(os.tmpdir(), 'tek-e2e-shots')
mkdirSync(shots, { recursive: true })

// --- demo corpus with distinctive, verifiable facts -------------------------
const demo = join(os.tmpdir(), 'tek-demo-notes')
rmSync(demo, { recursive: true, force: true })
mkdirSync(demo, { recursive: true })
const corpus = {
  'lease.md':
    '# Apartment lease notes\nLease renews September 1. Rent is $2,140/month. Landlord contact: Marisol, unit 4B. 60-day notice required to leave.',
  'car.md':
    '# Car maintenance\nThe Subaru needs an oil change every 6,000 miles. Last done at 48,200. Winter tires are stored at the shop on Fern Street.',
  'thesis.md':
    '# Thesis snippets\nThe ablation shows retrieval quality drops 14% without hard negatives. Reviewer 2 wants a comparison against ColBERT-v2.',
  'passwords_hint.txt':
    'Router admin lives at 10.0.0.138. The NAS backup job runs Sundays at 03:00.',
  'recipes.md':
    '# Weeknight recipes\nMiso salmon: marinate 20 minutes, broil 8. Chickpea curry needs coconut milk and garam masala.'
}
for (const [name, text] of Object.entries(corpus)) writeFileSync(join(demo, name), text)

const log = (...a) => console.log('[drive]', ...a)

// --- launch ------------------------------------------------------------------
// Launch with the project dir (not the main .js) so app.getAppPath() is the
// repo root and the sidecar venv resolves, same as `npm run dev`.
const app = await electron.launch({ args: [repo], cwd: repo })
const page = await app.firstWindow()
page.on('console', (m) => {
  const t = m.text()
  if (/error|fail/i.test(t)) console.log('[renderer]', t)
})

// --- wait for sidecar ---------------------------------------------------------
let port = null
for (let i = 0; i < 360; i++) {
  const s = await page.evaluate(() => window.tek.getStatus())
  if (s.state === 'online') { port = s.port; break }
  if (s.state === 'error') { console.error('SIDECAR ERROR:', s.error); await app.close(); process.exit(1) }
  await page.waitForTimeout(500)
}
if (!port) { console.error('sidecar never came online'); await app.close(); process.exit(1) }
log('sidecar online on port', port)
const base = `http://127.0.0.1:${port}`
const j = async (r) => { if (!r.ok) throw new Error(`${r.url} -> HTTP ${r.status}: ${await r.text()}`); return r.json() }
const put = (path, body) => fetch(base + path, { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }).then(j)
const post = (path, body = {}) => fetch(base + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }).then(j)

const health = await j(await fetch(base + '/health'))
log('health:', JSON.stringify({ version: health.version, reranker: health.reranker, index: health.index }))

// --- configure folders (keep existing ones that still exist) -------------------
const settings = await j(await fetch(base + '/settings'))
const keep = settings.folders.filter((f) => existsSync(f))
await put('/settings', { folders: [...keep, demo] })
log('folders set:', [...keep, demo].join(' | '))

// --- index ---------------------------------------------------------------------
await post('/index/start')
let st
for (let i = 0; i < 1200; i++) {
  st = await j(await fetch(base + '/index/status'))
  if (!st.running && (st.state === 'done' || st.state === 'error')) break
  await new Promise((r) => setTimeout(r, 1000))
}
log('index:', JSON.stringify(st))
if (st.state !== 'done') { console.error('INDEX FAILED'); await app.close(); process.exit(1) }

// --- drive Search ---------------------------------------------------------------
await page.getByRole('button', { name: 'Search' }).first().click()
const searchBox = page.getByPlaceholder(/Search your files/)
await searchBox.fill('when does my lease renew and how much is rent')
await searchBox.press('Enter')
await page.waitForSelector('text=/result/', { timeout: 120_000 }) // first query may download the reranker
await page.waitForTimeout(400)
await page.screenshot({ path: join(shots, '1-search-semantic.png') })
const top1 = await page.locator('li button').first().textContent()
log('search#1 top hit:', top1)

await searchBox.fill('10.0.0.138')
await searchBox.press('Enter')
await page.waitForTimeout(1500)
await page.screenshot({ path: join(shots, '2-search-keyword.png') })
const top2 = await page.locator('li button').first().textContent()
log('search#2 (exact keyword) top hit:', top2)

// --- drive Chat (incl. multi-turn follow-up) -------------------------------------
await page.getByRole('button', { name: 'Chat' }).first().click()
const chatBox = page.getByPlaceholder(/Ask anything|Index a folder/)
const askAndWait = async (q) => {
  await chatBox.fill(q)
  await chatBox.press('Enter')
  await page.waitForSelector('button:has-text("Stop")', { timeout: 15_000 }).catch(() => {})
  await page.waitForSelector('button:has-text("Ask")', { timeout: 180_000 })
  await page.waitForTimeout(300)
}
await askAndWait('what mileage was the last oil change done at?')
await page.screenshot({ path: join(shots, '3-chat-answer.png') })
await askAndWait('and where are the winter tires?') // follow-up: needs history rewrite
await page.screenshot({ path: join(shots, '4-chat-followup.png') })
const answers = await page.locator('.whitespace-pre-wrap').allTextContents()
log('chat answers:', JSON.stringify(answers))

// --- cleanup: restore folders, purge demo chunks ----------------------------------
await put('/settings', { folders: keep })
if (keep.length) {
  await post('/index/start')
  for (let i = 0; i < 600; i++) {
    const s = await j(await fetch(base + '/index/status'))
    if (!s.running && (s.state === 'done' || s.state === 'error')) break
    await new Promise((r) => setTimeout(r, 1000))
  }
}
const finalHealth = await j(await fetch(base + '/health'))
log('final index stats:', JSON.stringify(finalHealth.index))

await app.close()
rmSync(demo, { recursive: true, force: true })
log('screenshots in', shots)
log('E2E COMPLETE')
