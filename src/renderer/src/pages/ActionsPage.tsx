import { useState } from 'react'
import type { ActionOperation, OllamaStatus } from '../../../shared/types'
import { PlanModal } from '../components/PlanModal'
import { formatBytes } from '../lib/hooks'

interface Plan {
  title: string
  subtitle?: string
  operations: ActionOperation[]
}

type Tool = 'dedupe' | 'organize-type' | 'organize-date' | 'rename'

const TOOLS: { id: Tool; title: string; desc: string; needsLlm: boolean }[] = [
  {
    id: 'dedupe',
    title: 'Find duplicates',
    desc: 'Content-hash scan; keeps the oldest copy, recycles the rest.',
    needsLlm: false
  },
  {
    id: 'organize-type',
    title: 'Organize by type',
    desc: 'Sorts a folder’s loose files into Documents/, Images/, Code/…',
    needsLlm: false
  },
  {
    id: 'organize-date',
    title: 'Organize by date',
    desc: 'Sorts a folder’s loose files into YYYY-MM/ by modified date.',
    needsLlm: false
  },
  {
    id: 'rename',
    title: 'AI rename',
    desc: 'Suggests descriptive names from file contents. Needs Ollama.',
    needsLlm: true
  }
]

export function ActionsPage({
  online,
  ollama
}: {
  online: boolean
  ollama: OllamaStatus | null
}): React.JSX.Element {
  const [folder, setFolder] = useState('')
  const [busyTool, setBusyTool] = useState<Tool | null>(null)
  const [plan, setPlan] = useState<Plan | null>(null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')

  async function pick(): Promise<void> {
    const picked = await window.tek.pickFolder()
    if (picked) {
      setFolder(picked)
      setError('')
      setNote('')
    }
  }

  async function run(tool: Tool): Promise<void> {
    if (!folder || busyTool) return
    setBusyTool(tool)
    setError('')
    setNote('')
    try {
      if (tool === 'dedupe') {
        const res = await window.tek.actions.dedupe(folder)
        setPlan({
          title: 'Remove duplicate files',
          subtitle:
            res.operations.length > 0
              ? `${res.groups.length} duplicate group${res.groups.length === 1 ? '' : 's'} — ${formatBytes(res.wastedBytes)} reclaimable. Oldest copy of each is kept.`
              : undefined,
          operations: res.operations
        })
      } else if (tool === 'organize-type' || tool === 'organize-date') {
        const strategy = tool === 'organize-type' ? 'by-type' : 'by-date'
        const res = await window.tek.actions.organize(folder, strategy)
        if (res.error) throw new Error(res.error)
        setPlan({
          title: strategy === 'by-type' ? 'Organize by file type' : 'Organize by date',
          operations: res.operations
        })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusyTool(null)
    }
  }

  async function runRename(): Promise<void> {
    if (!folder || busyTool) return
    setBusyTool('rename')
    setError('')
    setNote('AI rename reads each file and asks the local model for a better name — this can take a minute…')
    try {
      // Rename operates on the loose files at the folder root (depth 1).
      const res = await window.tek.actions.organize(folder, 'by-type')
      const rootFiles = res.operations.map((op) => op.src)
      if (rootFiles.length === 0) {
        setNote('No loose files at the root of that folder to rename.')
        return
      }
      const planRes = await window.tek.actions.rename(rootFiles.slice(0, 25))
      setNote(planRes.errors?.length ? `${planRes.errors.length} file(s) skipped (no readable text).` : '')
      setPlan({ title: 'AI-suggested renames', operations: planRes.operations })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setNote('')
    } finally {
      setBusyTool(null)
    }
  }

  return (
    <div className="mx-auto h-full max-w-2xl space-y-5 overflow-y-auto px-6 py-6">
      <section>
        <h2 className="text-[15px] font-semibold text-zinc-100">File actions</h2>
        <p className="mt-0.5 text-xs text-zinc-500">
          Every action shows a full preview first — nothing is moved, renamed, or recycled until
          you approve it. Deletions only ever go to the recycle bin.
        </p>
      </section>

      <section className="flex items-center gap-3 rounded-2xl border border-ink-700/70 bg-ink-900/40 px-4 py-3">
        <span className="text-xs text-zinc-500">Target folder</span>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-zinc-300" title={folder}>
          {folder || '— none chosen —'}
        </span>
        <button
          onClick={() => void pick()}
          disabled={!online}
          className="h-8 shrink-0 rounded-lg border border-ink-700 px-3 text-xs text-zinc-300 transition-colors hover:bg-ink-800 disabled:opacity-40"
        >
          Choose…
        </button>
      </section>

      <section className="grid grid-cols-2 gap-3">
        {TOOLS.map((tool) => {
          const llmBlocked = tool.needsLlm && !ollama?.available
          const disabled = !online || !folder || busyTool !== null || llmBlocked
          return (
            <button
              key={tool.id}
              disabled={disabled}
              onClick={() => void (tool.id === 'rename' ? runRename() : run(tool.id))}
              className="group rounded-2xl border border-ink-700 bg-ink-900/50 px-4 py-4 text-left transition-all hover:border-teal-500/30 hover:bg-ink-900 disabled:cursor-not-allowed disabled:opacity-45"
            >
              <div className="flex items-center justify-between">
                <span className="text-[13px] font-medium text-zinc-200">
                  {busyTool === tool.id ? 'Working…' : tool.title}
                </span>
                {llmBlocked ? (
                  <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-medium text-amber-300">
                    needs Ollama
                  </span>
                ) : null}
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">{tool.desc}</p>
            </button>
          )
        })}
      </section>

      {note ? <p className="text-xs text-zinc-500">{note}</p> : null}
      {error ? (
        <div className="rounded-xl border border-rose-500/25 bg-rose-500/5 px-4 py-3 font-mono text-xs text-rose-300">
          {error}
        </div>
      ) : null}

      {plan ? (
        <PlanModal
          title={plan.title}
          subtitle={plan.subtitle}
          operations={plan.operations}
          onClose={() => setPlan(null)}
          onApplied={() => void window.tek.index.start().catch(() => undefined)}
        />
      ) : null}
    </div>
  )
}
