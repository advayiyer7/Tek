import { useCallback, useEffect, useState } from 'react'
import type { SidecarStatus } from '../../shared/types'
import { Header } from './components/Header'
import { Pipeline } from './components/Pipeline'
import { PingConsole } from './components/PingConsole'

export default function App(): React.JSX.Element {
  const [status, setStatus] = useState<SidecarStatus>({ state: 'starting' })

  useEffect(() => {
    let mounted = true
    void window.tek.getStatus().then((s) => {
      if (mounted) setStatus(s)
    })
    const unsubscribe = window.tek.onStatus(setStatus)
    return () => {
      mounted = false
      unsubscribe()
    }
  }, [])

  const [pulseKey, setPulseKey] = useState(0)
  const onRoundTrip = useCallback(() => setPulseKey((k) => k + 1), [])

  return (
    <div className="relative flex h-full flex-col overflow-hidden">
      {/* Ambient backdrop */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(1100px 500px at 50% -10%, rgba(20,184,166,0.07), transparent 60%), radial-gradient(800px 400px at 85% 110%, rgba(99,102,241,0.05), transparent 60%)'
        }}
      />

      <Header status={status} />

      <main className="relative z-10 mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 overflow-hidden px-6 pb-6 pt-8">
        <Pipeline status={status} pulseKey={pulseKey} />
        <PingConsole status={status} onRoundTrip={onRoundTrip} />
      </main>

      <footer className="relative z-10 flex items-center justify-between border-t border-ink-700/50 px-6 py-2.5 font-mono text-[11px] text-zinc-600">
        <span>
          electron {window.tek.versions.electron} · chromium {window.tek.versions.chrome} · node{' '}
          {window.tek.versions.node}
          {status.python ? ` · python ${status.python}` : ''}
        </span>
        <span>phase 1 — sidecar wiring</span>
      </footer>
    </div>
  )
}
