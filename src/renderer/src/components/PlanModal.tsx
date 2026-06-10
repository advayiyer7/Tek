import { useMemo, useState } from 'react'
import type { ActionOperation, ExecResult } from '../../../shared/types'

const KIND_BADGE: Record<ActionOperation['kind'], { label: string; cls: string }> = {
  move: { label: 'MOVE', cls: 'bg-sky-500/15 text-sky-300' },
  rename: { label: 'RENAME', cls: 'bg-violet-500/15 text-violet-300' },
  trash: { label: 'RECYCLE', cls: 'bg-rose-500/15 text-rose-300' }
}

function shorten(path: string, max = 58): string {
  if (path.length <= max) return path
  return path.slice(0, 22) + '…' + path.slice(-(max - 23))
}

/**
 * The safety gate: every mutating file action passes through this preview.
 * Nothing touches the filesystem until the user clicks Apply here, and
 * deletions only ever go to the recycle bin.
 */
export function PlanModal({
  title,
  subtitle,
  operations,
  onClose,
  onApplied
}: {
  title: string
  subtitle?: string
  operations: ActionOperation[]
  onClose: () => void
  onApplied?: () => void
}): React.JSX.Element {
  const [checked, setChecked] = useState<boolean[]>(() => operations.map(() => true))
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState<ExecResult[] | null>(null)

  const selectedCount = useMemo(() => checked.filter(Boolean).length, [checked])

  async function apply(): Promise<void> {
    const selected = operations.filter((_, i) => checked[i])
    if (selected.length === 0) return
    setBusy(true)
    try {
      const res = await window.tek.actions.execute(selected)
      setResults(res)
      onApplied?.()
    } catch (err) {
      setResults([
        {
          op: selected[0],
          ok: false,
          error: err instanceof Error ? err.message : String(err)
        }
      ])
    } finally {
      setBusy(false)
    }
  }

  const failures = results?.filter((r) => !r.ok) ?? []

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-8">
      <div className="flex max-h-full w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-ink-700 bg-ink-900 shadow-2xl">
        <div className="border-b border-ink-700/60 px-6 py-4">
          <h2 className="text-[15px] font-semibold text-zinc-100">{title}</h2>
          <p className="mt-0.5 text-xs text-zinc-500">
            {subtitle ?? 'Review every change below. Nothing happens until you click Apply.'}
          </p>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
          {results === null ? (
            operations.length === 0 ? (
              <p className="px-2 py-8 text-center text-sm text-zinc-500">
                Nothing to do — this folder is already in shape.
              </p>
            ) : (
              <ul className="space-y-1">
                {operations.map((op, i) => (
                  <li key={i}>
                    <label className="flex cursor-pointer items-start gap-3 rounded-lg px-2 py-2 hover:bg-ink-800/60">
                      <input
                        type="checkbox"
                        checked={checked[i]}
                        onChange={() =>
                          setChecked((prev) => prev.map((v, j) => (j === i ? !v : v)))
                        }
                        className="mt-1 size-3.5 accent-teal-500"
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span
                            className={`rounded px-1.5 py-0.5 font-mono text-[9px] font-semibold tracking-wider ${KIND_BADGE[op.kind].cls}`}
                          >
                            {KIND_BADGE[op.kind].label}
                          </span>
                          <span
                            className="truncate font-mono text-xs text-zinc-300"
                            title={op.src}
                          >
                            {shorten(op.src)}
                          </span>
                        </div>
                        {op.dest ? (
                          <div className="mt-0.5 truncate pl-1 font-mono text-xs text-teal-400/80" title={op.dest}>
                            → {shorten(op.dest)}
                          </div>
                        ) : null}
                        {op.reason ? (
                          <div className="mt-0.5 pl-1 text-[11px] text-zinc-600">{op.reason}</div>
                        ) : null}
                      </div>
                    </label>
                  </li>
                ))}
              </ul>
            )
          ) : (
            <div className="space-y-3 px-2 py-2">
              <p className="text-sm text-zinc-300">
                Applied {results.filter((r) => r.ok).length} of {results.length} change
                {results.length === 1 ? '' : 's'}
                {failures.length > 0 ? ` — ${failures.length} failed:` : '.'}
              </p>
              {failures.map((f, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-rose-500/25 bg-rose-500/5 px-3 py-2 font-mono text-xs text-rose-300"
                >
                  {shorten(f.op.src)}: {f.error}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-ink-700/60 px-6 py-4">
          <span className="text-xs text-zinc-600">
            {results === null
              ? `${selectedCount} of ${operations.length} selected · deletions go to the recycle bin`
              : 'Done.'}
          </span>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="h-9 rounded-lg border border-ink-700 px-4 text-sm text-zinc-300 transition-colors hover:bg-ink-800"
            >
              {results === null ? 'Cancel' : 'Close'}
            </button>
            {results === null && operations.length > 0 ? (
              <button
                onClick={() => void apply()}
                disabled={busy || selectedCount === 0}
                className="h-9 rounded-lg bg-teal-500 px-5 text-sm font-medium text-ink-950 transition-all hover:bg-teal-400 disabled:cursor-not-allowed disabled:bg-ink-700 disabled:text-zinc-500"
              >
                {busy ? 'Applying…' : `Apply ${selectedCount} change${selectedCount === 1 ? '' : 's'}`}
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}
