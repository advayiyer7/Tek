import { useEffect, useState } from 'react'
import type { SidecarStatus } from '../../../shared/types'

function Node({
  title,
  subtitle,
  active,
  pulsing
}: {
  title: string
  subtitle: string
  active: boolean
  pulsing: boolean
}): React.JSX.Element {
  return (
    <div
      className={`flex min-w-28 flex-col items-center gap-0.5 rounded-xl border px-4 py-3 transition-all duration-500 ${
        active
          ? 'border-teal-500/40 bg-ink-800/80 shadow-lg shadow-teal-500/10'
          : 'border-ink-700 bg-ink-900/60'
      } ${pulsing ? 'scale-[1.04] border-teal-400/70' : ''}`}
    >
      <span className={`text-[13px] font-medium ${active ? 'text-zinc-100' : 'text-zinc-500'}`}>
        {title}
      </span>
      <span className="font-mono text-[10px] text-zinc-600">{subtitle}</span>
    </div>
  )
}

function Link({ active }: { active: boolean }): React.JSX.Element {
  return (
    <div className="relative mx-1 h-px flex-1 self-center overflow-hidden">
      <div
        className={`h-full transition-colors duration-500 ${
          active ? 'animate-flow' : 'bg-ink-700'
        }`}
        style={
          active
            ? {
                backgroundImage:
                  'repeating-linear-gradient(90deg, rgba(45,212,191,0.9) 0 6px, rgba(45,212,191,0.15) 6px 16px)'
              }
            : undefined
        }
      />
    </div>
  )
}

/** Visualizes the renderer -> main -> sidecar link; nodes pulse on each round-trip. */
export function Pipeline({
  status,
  pulseKey
}: {
  status: SidecarStatus
  pulseKey: number
}): React.JSX.Element {
  const online = status.state === 'online'
  const [pulsing, setPulsing] = useState(false)

  useEffect(() => {
    if (pulseKey === 0) return
    setPulsing(true)
    const timer = setTimeout(() => setPulsing(false), 450)
    return () => clearTimeout(timer)
  }, [pulseKey])

  return (
    <section className="rounded-2xl border border-ink-700/70 bg-ink-900/40 px-5 py-4">
      <div className="flex items-stretch">
        <Node title="Renderer" subtitle="react" active={online} pulsing={pulsing} />
        <Link active={online} />
        <Node title="Main" subtitle="electron" active={online} pulsing={pulsing} />
        <Link active={online} />
        <Node
          title="Sidecar"
          subtitle={status.python ? `python ${status.python}` : 'python'}
          active={online}
          pulsing={pulsing}
        />
      </div>
      {status.state === 'error' && status.error ? (
        <pre className="mt-3 max-h-28 select-text overflow-auto whitespace-pre-wrap rounded-lg border border-rose-500/20 bg-rose-500/5 p-3 font-mono text-[11px] leading-relaxed text-rose-300">
          {status.error}
        </pre>
      ) : null}
    </section>
  )
}
