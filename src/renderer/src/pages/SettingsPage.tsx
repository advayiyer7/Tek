import type { OllamaStatus, SidecarHealth, TekSettings } from '../../../shared/types'
import { useEffect, useState } from 'react'

export function SettingsPage({
  online,
  settings,
  updateSettings,
  ollama,
  refreshOllama
}: {
  online: boolean
  settings: TekSettings | null
  updateSettings: (changes: Partial<TekSettings>) => Promise<void>
  ollama: OllamaStatus | null
  refreshOllama: () => void
}): React.JSX.Element {
  const [health, setHealth] = useState<SidecarHealth | null>(null)

  useEffect(() => {
    if (!online) return
    void window.tek
      .health()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [online])

  return (
    <div className="mx-auto h-full max-w-2xl space-y-5 overflow-y-auto px-6 py-6">
      <section className="rounded-2xl border border-ink-700/70 bg-ink-900/40 px-5 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-[14px] font-semibold text-zinc-100">Local AI (Ollama)</h2>
            <p className="mt-0.5 text-xs text-zinc-500">
              Powers chat answers, AI rename, and summaries. Search and indexing work without it.
            </p>
          </div>
          <span
            className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${
              ollama?.available
                ? 'bg-emerald-500/10 text-emerald-300'
                : 'bg-amber-500/10 text-amber-300'
            }`}
          >
            {ollama?.available ? `connected · v${ollama.version}` : 'not detected'}
          </span>
        </div>

        {ollama?.available ? (
          <div className="mt-4">
            <label className="text-[11px] uppercase tracking-wider text-zinc-600">
              Chat model
            </label>
            {ollama.models.length > 0 ? (
              <select
                value={settings?.llm_model ?? ''}
                onChange={(e) => void updateSettings({ llm_model: e.target.value })}
                className="mt-1.5 h-9 w-full rounded-lg border border-ink-700 bg-ink-950 px-3 text-sm text-zinc-200 outline-none focus:border-teal-500/50"
              >
                {!ollama.models.includes(settings?.llm_model ?? '') ? (
                  <option value={settings?.llm_model}>
                    {settings?.llm_model} (not pulled)
                  </option>
                ) : null}
                {ollama.models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            ) : (
              <p className="mt-1.5 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-200/90">
                Ollama is running but has no models. Pull one in a terminal:{' '}
                <code className="select-text font-mono text-amber-100">ollama pull llama3.2:3b</code>
              </p>
            )}
          </div>
        ) : (
          <div className="mt-4 space-y-2 text-xs leading-relaxed text-zinc-400">
            <p>
              Install Ollama from{' '}
              <a
                href="https://ollama.com/download"
                target="_blank"
                rel="noreferrer"
                className="text-teal-400 hover:underline"
              >
                ollama.com/download
              </a>
              , then pull a small model:
            </p>
            <pre className="select-text rounded-lg border border-ink-700 bg-ink-950 px-3 py-2 font-mono text-[11px] text-zinc-300">
              ollama pull llama3.2:3b
            </pre>
            <button onClick={refreshOllama} className="text-teal-400 hover:underline">
              Re-check connection
            </button>
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-ink-700/70 bg-ink-900/40 px-5 py-4">
        <h2 className="text-[14px] font-semibold text-zinc-100">Privacy</h2>
        <div className="mt-3 space-y-3">
          <div className="flex items-start gap-3">
            <span className="mt-1 size-2 shrink-0 rounded-full bg-emerald-400" />
            <p className="text-xs leading-relaxed text-zinc-400">
              <span className="text-zinc-200">Everything stays on this machine.</span> Text
              extraction, embeddings ({health?.embedModel.name ?? 'bge-small'}), the vector index,
              and LLM inference all run locally. Tek makes no network calls with your file
              contents.
            </p>
          </div>
          <div className="flex items-start gap-3 opacity-55">
            <span className="mt-1 size-2 shrink-0 rounded-full bg-zinc-600" />
            <div className="text-xs leading-relaxed text-zinc-400">
              <span className="text-zinc-200">Cloud models (opt-in)</span> — coming later. Off by
              default; keys will live in the OS keychain, never on disk.
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-ink-700/70 bg-ink-900/40 px-5 py-4">
        <h2 className="text-[14px] font-semibold text-zinc-100">Engine</h2>
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 font-mono text-[11px]">
          <Row k="sidecar" v={health ? `v${health.version} · python ${health.python}` : '—'} />
          <Row
            k="embedding model"
            v={
              health
                ? `${health.embedModel.name.split('/').pop()} ${health.embedModel.ready ? '(loaded)' : '(loads on first index)'}`
                : '—'
            }
          />
          <Row k="vector store" v="LanceDB (on-disk, local)" />
          <Row
            k="index"
            v={health ? `${health.index.files.toLocaleString()} files · ${health.index.chunks.toLocaleString()} chunks` : '—'}
          />
          <Row k="electron" v={window.tek.versions.electron} />
          <Row k="chromium" v={window.tek.versions.chrome} />
        </dl>
      </section>
    </div>
  )
}

function Row({ k, v }: { k: string; v: string }): React.JSX.Element {
  return (
    <>
      <dt className="text-zinc-600">{k}</dt>
      <dd className="select-text text-right text-zinc-300">{v}</dd>
    </>
  )
}
