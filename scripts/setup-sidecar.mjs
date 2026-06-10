// Cross-platform setup for the Python sidecar: creates sidecar/.venv and
// installs requirements. Run via `npm run sidecar:setup`.
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const sidecarDir = join(root, 'sidecar')
const venvDir = join(sidecarDir, '.venv')
const venvPython =
  process.platform === 'win32'
    ? join(venvDir, 'Scripts', 'python.exe')
    : join(venvDir, 'bin', 'python')

function run(cmd, args) {
  console.log(`> ${cmd} ${args.join(' ')}`)
  const result = spawnSync(cmd, args, { stdio: 'inherit' })
  if (result.error) {
    console.error(`Failed to run ${cmd}: ${result.error.message}`)
    process.exit(1)
  }
  if (result.status !== 0) process.exit(result.status ?? 1)
}

if (!existsSync(venvPython)) {
  const systemPython = process.platform === 'win32' ? 'python' : 'python3'
  run(systemPython, ['-m', 'venv', venvDir])
}
run(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'])
run(venvPython, ['-m', 'pip', 'install', '-r', join(sidecarDir, 'requirements.txt')])
console.log('\nSidecar environment ready at sidecar/.venv')
