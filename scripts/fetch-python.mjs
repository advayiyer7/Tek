// Downloads a standalone CPython runtime (python-build-standalone) for
// bundling into the installer, so installed apps never need system Python.
// Output: build-python/<platform>/python/ — gitignored, fetched at dist time.
import { existsSync, mkdirSync, rmSync, createWriteStream } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { execFileSync } from 'child_process'
import { Readable } from 'stream'
import { pipeline } from 'stream/promises'

const PBS_TAG = '20260602'
const PY_VER = '3.12.13'

const TARGETS = {
  'win32-x64': `cpython-${PY_VER}+${PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz`,
  'darwin-arm64': `cpython-${PY_VER}+${PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz`,
  'darwin-x64': `cpython-${PY_VER}+${PBS_TAG}-x86_64-apple-darwin-install_only.tar.gz`,
  'linux-x64': `cpython-${PY_VER}+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz`
}

const key = `${process.platform}-${process.arch}`
const asset = TARGETS[key]
if (!asset) {
  console.error(`no standalone python mapping for ${key}`)
  process.exit(1)
}

const repo = dirname(dirname(fileURLToPath(import.meta.url)))
const outDir = join(repo, 'build-python', key)
const marker = join(outDir, 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python3')
if (existsSync(marker)) {
  console.log(`standalone python already present: ${marker}`)
  process.exit(0)
}

rmSync(outDir, { recursive: true, force: true })
mkdirSync(outDir, { recursive: true })
const url = `https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${encodeURIComponent(asset)}`
const tarPath = join(outDir, asset)
console.log(`downloading ${asset} ...`)
const res = await fetch(url, { redirect: 'follow' })
if (!res.ok) {
  console.error(`download failed: HTTP ${res.status}`)
  process.exit(1)
}
await pipeline(Readable.fromWeb(res.body), createWriteStream(tarPath))
console.log('extracting ...')
execFileSync('tar', ['-xzf', tarPath, '-C', outDir]) // bsdtar ships with Win10+
rmSync(tarPath)
if (!existsSync(marker)) {
  console.error('extraction did not produce expected layout')
  process.exit(1)
}
console.log(`standalone python ready: ${marker}`)
