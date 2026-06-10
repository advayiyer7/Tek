import { useEffect, useRef, useState } from 'react'
import type { Citation, OllamaStatus } from '../../../shared/types'

interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  text: string
  citations: Citation[]
  fallback?: 'ollama-offline' | 'model-missing' | 'no-results'
  error?: string
  streaming?: boolean
}

let nextId = 1

export function ChatPage({
  online,
  hasIndex,
  ollama,
  onGoToLibrary
}: {
  online: boolean
  hasIndex: boolean
  ollama: OllamaStatus | null
  onGoToLibrary: () => void
}): React.JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const activeRef = useRef<string | null>(null)
  activeRef.current = activeChatId

  useEffect(() => {
    const unsubscribe = window.tek.chat.onEvent(({ id, event }) => {
      if (id !== activeRef.current) return
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (!last || last.role !== 'assistant') return prev
        const updated = { ...last }
        switch (event.type) {
          case 'citations':
            updated.citations = event.citations
            break
          case 'token':
            updated.text += event.text
            break
          case 'fallback':
            updated.fallback = event.reason
            updated.text = event.text
            break
          case 'error':
            updated.error = event.error
            updated.streaming = false
            break
          case 'done':
          case 'closed':
            updated.streaming = false
            break
        }
        return [...prev.slice(0, -1), updated]
      })
      if (event.type === 'done' || event.type === 'closed' || event.type === 'error') {
        setActiveChatId(null)
      }
    })
    return unsubscribe
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages])

  async function ask(): Promise<void> {
    const question = input.trim()
    if (!question || activeChatId || !online) return
    setInput('')
    setMessages((prev) => [
      ...prev,
      { id: nextId++, role: 'user', text: question, citations: [] },
      { id: nextId++, role: 'assistant', text: '', citations: [], streaming: true }
    ])
    try {
      const id = await window.tek.chat.start(question)
      setActiveChatId(id)
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            streaming: false,
            error: err instanceof Error ? err.message : String(err)
          }
        }
        return updated
      })
    }
  }

  function stop(): void {
    if (activeChatId) void window.tek.chat.cancel(activeChatId)
  }

  const llmReady =
    ollama?.available && ollama.models.some((m) => m === ollama.configuredModel)

  return (
    <div className="flex h-full flex-col">
      {ollama && !llmReady ? (
        <div className="mx-6 mt-4 flex items-center gap-3 rounded-xl border border-amber-500/25 bg-amber-500/5 px-4 py-2.5 text-xs text-amber-200/90">
          <span className="size-1.5 shrink-0 rounded-full bg-amber-400" />
          {!ollama.available ? (
            <span>
              Ollama isn&apos;t running — answers will show the best matching passages from your
              files instead of generated responses. Search works fully.
            </span>
          ) : (
            <span>
              Model <code className="font-mono">{ollama.configuredModel}</code> isn&apos;t pulled
              in Ollama — pick an installed model in Settings.
            </span>
          )}
        </div>
      ) : null}

      <div ref={scrollRef} className="min-h-0 flex-1 space-y-5 overflow-y-auto px-6 py-5">
        {messages.length === 0 ? (
          <EmptyState hasIndex={hasIndex} onGoToLibrary={onGoToLibrary} />
        ) : (
          messages.map((m) => <Message key={m.id} message={m} />)
        )}
      </div>

      <form
        className="flex items-center gap-2 border-t border-ink-700/50 p-4"
        onSubmit={(e) => {
          e.preventDefault()
          void ask()
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={!online}
          placeholder={
            hasIndex ? 'Ask anything about your files…' : 'Index a folder first (Library) — then ask away'
          }
          maxLength={4000}
          className="h-11 min-w-0 flex-1 select-text rounded-xl border border-ink-700 bg-ink-950/80 px-4 text-sm text-zinc-200 placeholder-zinc-600 outline-none transition-colors focus:border-teal-500/50 disabled:opacity-50"
        />
        {activeChatId ? (
          <button
            type="button"
            onClick={stop}
            className="h-11 rounded-xl border border-ink-700 px-5 text-sm text-zinc-300 transition-colors hover:bg-ink-800"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!online || input.trim().length === 0}
            className="h-11 rounded-xl bg-teal-500 px-6 text-sm font-medium text-ink-950 transition-all hover:bg-teal-400 active:scale-95 disabled:cursor-not-allowed disabled:bg-ink-700 disabled:text-zinc-500"
          >
            Ask
          </button>
        )}
      </form>
    </div>
  )
}

function EmptyState({
  hasIndex,
  onGoToLibrary
}: {
  hasIndex: boolean
  onGoToLibrary: () => void
}): React.JSX.Element {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
      <div className="flex size-14 items-center justify-center rounded-2xl border border-ink-700 bg-ink-900">
        <svg viewBox="0 0 24 24" className="size-6 text-teal-400/70" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
        </svg>
      </div>
      <p className="text-sm text-zinc-400">Chat with your files — answers cite their sources.</p>
      {!hasIndex ? (
        <button
          onClick={onGoToLibrary}
          className="text-xs text-teal-400 underline-offset-4 hover:underline"
        >
          Start by indexing a folder in the Library →
        </button>
      ) : (
        <p className="text-xs text-zinc-600">
          Everything runs on this machine. Nothing leaves it.
        </p>
      )}
    </div>
  )
}

function Message({ message }: { message: ChatMessage }): React.JSX.Element {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] select-text rounded-2xl rounded-br-md bg-teal-500/15 px-4 py-2.5 text-sm leading-relaxed text-teal-100">
          {message.text}
        </div>
      </div>
    )
  }

  return (
    <div className="flex">
      <div className="max-w-[85%] space-y-2.5">
        {message.citations.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {message.citations.map((c) => (
              <button
                key={c.ref}
                onClick={() => void window.tek.file.open(c.path)}
                title={`${c.path}\n\n${c.preview}`}
                className="flex items-center gap-1.5 rounded-full border border-ink-700 bg-ink-900 px-2.5 py-1 font-mono text-[10px] text-zinc-400 transition-colors hover:border-teal-500/40 hover:text-teal-300"
              >
                <span className="text-teal-500">[{c.ref}]</span>
                {c.name}
              </button>
            ))}
          </div>
        ) : null}

        {message.fallback && message.fallback !== 'no-results' ? (
          <FallbackAnswer citations={message.citations} />
        ) : message.error ? (
          <div className="rounded-2xl rounded-bl-md border border-rose-500/25 bg-rose-500/10 px-4 py-2.5 font-mono text-xs text-rose-300">
            {message.error}
          </div>
        ) : (
          <div className="select-text whitespace-pre-wrap rounded-2xl rounded-bl-md border border-ink-700 bg-ink-900/70 px-4 py-2.5 text-sm leading-relaxed text-zinc-200">
            {message.text || (message.streaming ? <Cursor /> : '')}
            {message.text && message.streaming ? <Cursor /> : null}
          </div>
        )}
      </div>
    </div>
  )
}

function FallbackAnswer({ citations }: { citations: Citation[] }): React.JSX.Element {
  return (
    <div className="space-y-2">
      <p className="text-xs text-zinc-500">
        Best matching passages (connect Ollama for synthesized answers):
      </p>
      {citations.slice(0, 4).map((c) => (
        <button
          key={c.ref}
          onClick={() => void window.tek.file.open(c.path)}
          className="block w-full rounded-xl border border-ink-700 bg-ink-900/70 px-4 py-3 text-left transition-colors hover:border-teal-500/30"
        >
          <div className="mb-1 flex items-center justify-between gap-2">
            <span className="truncate font-mono text-[11px] text-teal-400">{c.name}</span>
            <span className="font-mono text-[10px] text-zinc-600">
              {(c.score * 100).toFixed(0)}%
            </span>
          </div>
          <p className="select-text text-xs leading-relaxed text-zinc-400">{c.preview}…</p>
        </button>
      ))}
    </div>
  )
}

function Cursor(): React.JSX.Element {
  return <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse rounded-sm bg-teal-400 align-middle" />
}
