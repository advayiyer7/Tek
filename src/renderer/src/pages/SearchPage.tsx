import { useRef, useState } from 'react'
import type { SearchResult } from '../../../shared/types'

export function SearchPage({ online }: { online: boolean }): React.JSX.Element {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[] | null>(null)
  const [tookMs, setTookMs] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [summaries, setSummaries] = useState<Record<string, string>>({})
  const requestSeq = useRef(0)

  async function run(): Promise<void> {
    const q = query.trim()
    if (!q || !online) return
    const seq = ++requestSeq.current
    setBusy(true)
    setError('')
    try {
      const res = await window.tek.search(q, 12)
      if (seq !== requestSeq.current) return
      setResults(res.results)
      setTookMs(res.tookMs)
    } catch (err) {
      if (seq !== requestSeq.current) return
      setError(err instanceof Error ? err.message : String(err))
      setResults(null)
    } finally {
      if (seq === requestSeq.current) setBusy(false)
    }
  }

  async function summarize(path: string): Promise<void> {
    setSummaries((prev) => ({ ...prev, [path]: '…' }))
    const res = await window.tek.actions.summarize(path)
    setSummaries((prev) => ({ ...prev, [path]: res.summary ?? `(${res.error})` }))
  }

  return (
    <div className="flex h-full flex-col">
      <form
        className="flex items-center gap-2 border-b border-ink-700/50 p-4"
        onSubmit={(e) => {
          e.preventDefault()
          void run()
        }}
      >
        <div className="relative min-w-0 flex-1">
          <svg
            viewBox="0 0 24 24"
            className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-zinc-600"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={!online}
            placeholder="Search your files by meaning — “that pdf about quarterly taxes”…"
            maxLength={2000}
            autoFocus
            className="h-11 w-full select-text rounded-xl border border-ink-700 bg-ink-950/80 pl-10 pr-4 text-sm text-zinc-200 placeholder-zinc-600 outline-none transition-colors focus:border-teal-500/50 disabled:opacity-50"
          />
        </div>
        <button
          type="submit"
          disabled={!online || busy || query.trim().length === 0}
          className="h-11 rounded-xl bg-teal-500 px-6 text-sm font-medium text-ink-950 transition-all hover:bg-teal-400 active:scale-95 disabled:cursor-not-allowed disabled:bg-ink-700 disabled:text-zinc-500"
        >
          {busy ? 'Searching…' : 'Search'}
        </button>
      </form>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        {error ? (
          <div className="rounded-xl border border-rose-500/25 bg-rose-500/5 px-4 py-3 font-mono text-xs text-rose-300">
            {error}
          </div>
        ) : results === null ? (
          <p className="px-1 pt-10 text-center text-sm text-zinc-600">
            Semantic search over everything you&apos;ve indexed — works fully offline, no LLM
            needed.
          </p>
        ) : results.length === 0 ? (
          <p className="px-1 pt-10 text-center text-sm text-zinc-500">
            No matches. Try different wording, or index more folders in the Library.
          </p>
        ) : (
          <>
            <p className="mb-3 px-1 font-mono text-[11px] text-zinc-600">
              {results.length} result{results.length === 1 ? '' : 's'} · {tookMs}ms
            </p>
            <ul className="space-y-2.5">
              {results.map((r, i) => (
                <li
                  key={`${r.path}#${r.chunkIndex}-${i}`}
                  className="group rounded-xl border border-ink-700 bg-ink-900/50 px-4 py-3 transition-colors hover:border-ink-700/30 hover:bg-ink-900"
                >
                  <div className="mb-1.5 flex items-center gap-3">
                    <button
                      onClick={() => void window.tek.file.open(r.path)}
                      className="truncate font-mono text-[12px] font-medium text-teal-400 hover:underline"
                      title={`Open ${r.path}`}
                    >
                      {r.name}
                    </button>
                    <ScoreBar score={r.score} />
                    <div className="ml-auto flex shrink-0 gap-2 opacity-0 transition-opacity group-hover:opacity-100">
                      <button
                        onClick={() => void summarize(r.path)}
                        className="text-[10px] text-zinc-500 hover:text-teal-300"
                        title="Summarize with the local LLM (needs Ollama)"
                      >
                        summarize
                      </button>
                      <button
                        onClick={() => void window.tek.file.reveal(r.path)}
                        className="text-[10px] text-zinc-500 hover:text-teal-300"
                      >
                        reveal
                      </button>
                    </div>
                  </div>
                  <p className="truncate font-mono text-[10px] text-zinc-700" title={r.path}>
                    {r.path}
                  </p>
                  <p className="mt-1.5 line-clamp-3 select-text text-xs leading-relaxed text-zinc-400">
                    {r.text}
                  </p>
                  {summaries[r.path] ? (
                    <div className="mt-2 select-text whitespace-pre-wrap rounded-lg border border-teal-500/20 bg-teal-500/5 px-3 py-2 text-xs leading-relaxed text-teal-100/90">
                      {summaries[r.path]}
                    </div>
                  ) : null}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}

function ScoreBar({ score }: { score: number }): React.JSX.Element {
  return (
    <div className="flex shrink-0 items-center gap-1.5" title={`similarity ${(score * 100).toFixed(1)}%`}>
      <div className="h-1 w-14 overflow-hidden rounded-full bg-ink-700">
        <div
          className="h-full rounded-full bg-gradient-to-r from-teal-500 to-emerald-400"
          style={{ width: `${Math.min(100, score * 100)}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-zinc-600">{(score * 100).toFixed(0)}%</span>
    </div>
  )
}
