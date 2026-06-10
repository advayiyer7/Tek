// Cross-platform CI smoke test: launches the real Electron app (built out/),
// indexes a demo corpus through the sidecar, and asserts hybrid retrieval
// (semantic + exact-keyword) returns the right files. Exits non-zero on any
// failure. Linux CI: run under `xvfb-run -a`.
import { _electron as electron } from 'playwright'
import { mkdirSync, writeFileSync, rmSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import os from 'os'

const repo = dirname(dirname(fileURLToPath(import.meta.url)))
const demo = join(os.tmpdir(), 'tek-ci-corpus')
rmSync(demo, { recursive: true, force: true })
mkdirSync(demo, { recursive: true })
writeFileSync(
  join(demo, 'lease.md'),
  '# Apartment lease notes\nLease renews September 1. Rent is $2,140/month. 60-day notice required to leave.'
)
writeFileSync(
  join(demo, 'network.txt'),
  'Router admin lives at 10.0.0.138. The NAS backup job runs Sundays at 03:00.'
)
writeFileSync(
  join(demo, 'recipes.md'),
  '# Weeknight recipes\nMiso salmon: marinate 20 minutes, broil 8. Chickpea curry needs coconut milk and garam masala.'
)

const log = (...a) => console.log('[smoke]', ...a)
const fail = async (msg) => {
  console.error('[smoke] FAIL:', msg)
  process.exit(1)
}

const app = await electron.launch({
  args: [repo],
  cwd: repo,
  env: { ...process.env, ELECTRON_DISABLE_SANDBOX: '1' }
})
const page = await app.firstWindow()
page.on('console', (m) => {
  if (/error|fail/i.test(m.text())) console.log('[renderer]', m.text())
})

let port = null
for (let i = 0; i < 360; i++) {
  const s = await page.evaluate(() => window.tek.getStatus()).catch(() => ({ state: 'starting' }))
  if (s.state === 'online') { port = s.port; break }
  if (s.state === 'error') await fail(`sidecar error: ${s.error}`)
  await page.waitForTimeout(500)
}
if (!port) await fail('sidecar never came online')
log('sidecar online, port', port)

const base = `http://127.0.0.1:${port}`
const j = async (r) => {
  if (!r.ok) throw new Error(`${r.url} -> HTTP ${r.status}: ${await r.text()}`)
  return r.json()
}
const post = (p, body = {}) =>
  fetch(base + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }).then(j)

await fetch(base + '/settings', {
  method: 'PUT',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ folders: [demo] })
}).then(j)
await post('/index/start')
let st
for (let i = 0; i < 600; i++) {
  st = await j(await fetch(base + '/index/status'))
  if (!st.running && (st.state === 'done' || st.state === 'error')) break
  await new Promise((r) => setTimeout(r, 1000))
}
log('index:', st.state, JSON.stringify(st.stats))
if (st.state !== 'done') await fail(`index ended in state ${st.state}: ${st.error}`)
if (st.stats.files !== 3) await fail(`expected 3 indexed files, got ${st.stats.files}`)

// Semantic probe (no shared keywords beyond topic) and exact-keyword probe
// (exercises the BM25/FTS path) — both must rank the right file first.
const PROBES = [
  { query: 'when does my lease renew and how much is rent', expect: 'lease.md' },
  { query: '10.0.0.138', expect: 'network.txt' },
  { query: 'what goes in the chickpea curry', expect: 'recipes.md' }
]
for (const { query, expect } of PROBES) {
  const res = await post('/search', { query, k: 5 })
  const top = res.results[0]?.name ?? '(none)'
  log(`search ${JSON.stringify(query)} -> ${top} (${res.tookMs}ms)`)
  if (top !== expect) await fail(`expected ${expect}, got ${top}`)
}

const shot = join(repo, 'smoke-screenshot.png')
await page.screenshot({ path: shot })
log('screenshot:', shot)

await app.close()
rmSync(demo, { recursive: true, force: true })
log('SMOKE PASSED')
