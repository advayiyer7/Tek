import { useState } from 'react'
import type { IndexStatus, TekSettings } from '../../../shared/types'

const STATE_LABEL: Record<IndexStatus['state'], string> = {
  idle: 'Not indexed yet',
  'loading-model': 'Loading embedding model (first run downloads ~130MB)…',
  scanning: 'Scanning folders…',
  indexing: 'Indexing',
  done: 'Index up to date',
  error: 'Indexing failed'
}

export function LibraryPage({
  online,
  settings,
  updateSettings,
  indexStatus
}: {
  online: boolean
  settings: TekSettings | null
  updateSettings: (changes: Partial<TekSettings>) => Promise<void>
  indexStatus: IndexStatus | null
}): React.JSX.Element {
  const [error, setError] = useState('')
  const folders = settings?.folders ?? []

  async function addFolder(): Promise<void> {
    setError('')
    try {
      const picked = await window.tek.pickFolder()
      if (!picked) return
      if (folders.includes(picked)) return
      await updateSettings({ folders: [...folders, picked] })
      await window.tek.index.start()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function removeFolder(folder: string): Promise<void> {
    setError('')
    try {
      await updateSettings({ folders: folders.filter((f) => f !== folder) })
      await window.tek.index.start() // prunes the removed folder's files
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function reindex(): Promise<void> {
    setError('')
    try {
      await window.tek.index.start()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const running = indexStatus?.running ?? false
  const progress =
    indexStatus && indexStatus.totalFiles > 0
      ? Math.round((indexStatus.processedFiles / indexStatus.totalFiles) * 100)
      : 0

  return (
    <div className="mx-auto h-full max-w-2xl space-y-5 overflow-y-auto px-6 py-6">
      <section>
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h2 className="text-[15px] font-semibold text-zinc-100">Indexed folders</h2>
            <p className="mt-0.5 text-xs text-zinc-500">
              Tek extracts text on-device and embeds it locally. Media, binaries, and archives are
              skipped automatically.
            </p>
          </div>
          <button
            onClick={() => void addFolder()}
            disabled={!online}
            className="h-9 shrink-0 rounded-lg bg-teal-500 px-4 text-sm font-medium text-ink-950 transition-all hover:bg-teal-400 disabled:cursor-not-allowed disabled:bg-ink-700 disabled:text-zinc-500"
          >
            + Add folder
          </button>
        </div>

        {folders.length === 0 ? (
          <div className="rounded-xl border border-dashed border-ink-700 px-5 py-8 text-center text-sm text-zinc-600">
            No folders yet. Add your Documents, notes, or a projects folder to get started.
          </div>
        ) : (
          <ul className="space-y-1.5">
            {folders.map((folder) => (
              <li
                key={folder}
                className="group flex items-center gap-3 rounded-xl border border-ink-700 bg-ink-900/50 px-4 py-2.5"
              >
                <svg viewBox="0 0 24 24" className="size-4 shrink-0 text-teal-500/70" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                </svg>
                <span className="min-w-0 flex-1 truncate font-mono text-xs text-zinc-300" title={folder}>
                  {folder}
                </span>
                <button
                  onClick={() => void removeFolder(folder)}
                  className="shrink-0 text-[11px] text-zinc-600 opacity-0 transition-opacity hover:text-rose-400 group-hover:opacity-100"
                >
                  remove
                </button>
              </li>
            ))}
          </ul>
        )}
        {error ? (
          <p className="mt-2 font-mono text-xs text-rose-400">{error}</p>
        ) : null}
      </section>

      <section className="rounded-2xl border border-ink-700/70 bg-ink-900/40 px-5 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-[13px] font-medium text-zinc-200">
              {indexStatus ? STATE_LABEL[indexStatus.state] : 'Waiting for engine…'}
            </h3>
            {running && indexStatus ? (
              <p className="mt-0.5 truncate font-mono text-[10px] text-zinc-600" title={indexStatus.currentPath}>
                {indexStatus.currentPath || 'preparing…'}
              </p>
            ) : null}
          </div>
          <button
            onClick={() => void reindex()}
            disabled={!online || running || folders.length === 0}
            className="h-8 shrink-0 rounded-lg border border-ink-700 px-3 text-xs text-zinc-300 transition-colors hover:bg-ink-800 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running ? 'Indexing…' : 'Re-index now'}
          </button>
        </div>

        {running && indexStatus ? (
          <div className="mt-3">
            <div className="h-1.5 overflow-hidden rounded-full bg-ink-700">
              <div
                className="h-full rounded-full bg-gradient-to-r from-teal-500 to-emerald-400 transition-all duration-300"
                style={{ width: `${indexStatus.state === 'indexing' ? progress : 4}%` }}
              />
            </div>
            <p className="mt-1.5 font-mono text-[10px] text-zinc-600">
              {indexStatus.processedFiles}/{indexStatus.totalFiles} files · {indexStatus.indexedFiles} embedded ·{' '}
              {indexStatus.skippedFiles} unchanged/skipped
            </p>
          </div>
        ) : null}

        {indexStatus?.state === 'error' ? (
          <pre className="mt-3 select-text overflow-auto whitespace-pre-wrap rounded-lg border border-rose-500/20 bg-rose-500/5 p-3 font-mono text-[11px] text-rose-300">
            {indexStatus.error}
          </pre>
        ) : null}

        <div className="mt-4 flex gap-6 border-t border-ink-700/50 pt-3">
          <Stat label="files indexed" value={indexStatus?.stats.files ?? 0} />
          <Stat label="text chunks" value={indexStatus?.stats.chunks ?? 0} />
          <Stat
            label="last run"
            value={indexStatus && indexStatus.elapsedS > 0 ? `${indexStatus.elapsedS}s` : '—'}
          />
          <label className="ml-auto flex cursor-pointer items-center gap-2 text-[11px] text-zinc-500">
            <input
              type="checkbox"
              checked={settings?.watch_enabled ?? true}
              onChange={(e) => void updateSettings({ watch_enabled: e.target.checked })}
              className="size-3.5 accent-teal-500"
            />
            watch for changes
          </label>
        </div>
      </section>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number | string }): React.JSX.Element {
  return (
    <div>
      <div className="font-mono text-lg text-zinc-200">{typeof value === 'number' ? value.toLocaleString() : value}</div>
      <div className="text-[10px] uppercase tracking-wider text-zinc-600">{label}</div>
    </div>
  )
}
