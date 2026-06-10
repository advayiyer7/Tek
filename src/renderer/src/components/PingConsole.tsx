import { useEffect, useRef, useState } from 'react'
import type { SidecarStatus } from '../../../shared/types'

interface ConsoleEntry {
  id: number
  role: 'user' | 'sidecar' | 'error'
  text: string
  /** Full renderer -> main -> sidecar -> back time, ms. */
  totalMs?: number
  /** Main -> sidecar HTTP leg, ms. */
  mainMs?: number
}

let nextId = 1

export function PingConsole({
  status,
  onRoundTrip
}: {
  status: SidecarStatus
  onRoundTrip: () => void
}): React.JSX.Element {
  const [entries, setEntries] = useState<ConsoleEntry[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const online = status.state === 'online'

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [entries])

  useEffect(() => {
    if (online) inputRef.current?.focus()
  }, [online])

  async function send(): Promise<void> {
    const message = input.trim()
    if (!message || busy || !online) return
    setInput('')
    setBusy(true)
    setEntries((prev) => [...prev, { id: nextId++, role: 'user', text: message }])

    const startedAt = performance.now()
    try {
      const result = await window.tek.ping(message)
      const totalMs = Math.round((performance.now() - startedAt) * 10) / 10
      setEntries((prev) => [
        ...prev,
        { id: nextId++, role: 'sidecar', text: result.reply, totalMs, mainMs: result.mainLatencyMs }
      ])
      onRoundTrip()
    } catch (err) {
      setEntries((prev) => [
        ...prev,
        { id: nextId++, role: 'error', text: err instanceof Error ? err.message : String(err) }
      ])
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-ink-700/70 bg-ink-900/40">
      <div className="flex items-center justify-between border-b border-ink-700/50 px-5 py-3">
        <h2 className="text-[13px] font-medium text-zinc-300">Round-trip console</h2>
        <span className="font-mono text-[10px] uppercase tracking-wider text-zinc-600">
          renderer → main → sidecar → back
        </span>
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto px-5 py-4">
        {entries.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-1.5 text-center">
            <p className="text-sm text-zinc-500">
              {online
                ? 'Send a message to test the full round-trip.'
                : status.state === 'error'
                  ? 'The sidecar failed to start — fix the error above and relaunch.'
                  : 'Waiting for the Python sidecar to come online…'}
            </p>
            {online ? (
              <p className="text-xs text-zinc-700">
                It travels through Electron IPC to the main process, over local HTTP to Python, and
                back.
              </p>
            ) : null}
          </div>
        ) : (
          entries.map((entry) => <Entry key={entry.id} entry={entry} />)
        )}
      </div>

      <form
        className="flex items-center gap-2 border-t border-ink-700/50 p-3"
        onSubmit={(e) => {
          e.preventDefault()
          void send()
        }}
      >
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={!online}
          placeholder={online ? 'Type a message…' : 'Sidecar offline'}
          maxLength={2000}
          className="h-10 min-w-0 flex-1 select-text rounded-xl border border-ink-700 bg-ink-950/80 px-4 text-sm text-zinc-200 placeholder-zinc-600 outline-none transition-colors focus:border-teal-500/50 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!online || busy || input.trim().length === 0}
          className="h-10 rounded-xl bg-teal-500 px-5 text-sm font-medium text-ink-950 transition-all hover:bg-teal-400 active:scale-95 disabled:cursor-not-allowed disabled:bg-ink-700 disabled:text-zinc-500"
        >
          {busy ? 'Sending…' : 'Send'}
        </button>
      </form>
    </section>
  )
}

function Entry({ entry }: { entry: ConsoleEntry }): React.JSX.Element {
  if (entry.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] select-text rounded-2xl rounded-br-md bg-teal-500/15 px-4 py-2 text-sm text-teal-100">
          {entry.text}
        </div>
      </div>
    )
  }
  if (entry.role === 'error') {
    return (
      <div className="flex">
        <div className="max-w-[80%] select-text rounded-2xl rounded-bl-md border border-rose-500/25 bg-rose-500/10 px-4 py-2 font-mono text-xs text-rose-300">
          {entry.text}
        </div>
      </div>
    )
  }
  return (
    <div className="flex">
      <div className="max-w-[80%] rounded-2xl rounded-bl-md border border-ink-700 bg-ink-800/80 px-4 py-2">
        <p className="select-text font-mono text-xs leading-relaxed text-zinc-300">{entry.text}</p>
        {entry.totalMs !== undefined ? (
          <p className="mt-1.5 font-mono text-[10px] text-zinc-600">
            round-trip {entry.totalMs}ms
            {entry.mainMs !== undefined ? ` · main→python ${entry.mainMs}ms` : ''}
          </p>
        ) : null}
      </div>
    </div>
  )
}
