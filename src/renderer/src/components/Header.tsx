import type { SidecarStatus } from '../../../shared/types'

const STATE_LABEL: Record<SidecarStatus['state'], string> = {
  stopped: 'Sidecar stopped',
  starting: 'Starting sidecar…',
  online: 'Sidecar online',
  error: 'Sidecar error'
}

const STATE_DOT: Record<SidecarStatus['state'], string> = {
  stopped: 'bg-zinc-500',
  starting: 'bg-amber-400 animate-pulse',
  online: 'bg-emerald-400',
  error: 'bg-rose-500'
}

export function Header({ status }: { status: SidecarStatus }): React.JSX.Element {
  return (
    <header className="relative z-10 flex items-center justify-between border-b border-ink-700/50 px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="flex size-8 items-center justify-center rounded-lg bg-gradient-to-br from-teal-400 to-emerald-600 shadow-lg shadow-teal-500/20">
          <svg viewBox="0 0 24 24" className="size-4.5 text-ink-950" fill="currentColor">
            {/* Abstract "T" / file-stack mark */}
            <path d="M4 5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1h-5v11a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1V8H5a1 1 0 0 1-1-1V5z" />
          </svg>
        </div>
        <div className="leading-tight">
          <h1 className="text-[15px] font-semibold tracking-wide text-zinc-100">Tek</h1>
          <p className="text-[11px] text-zinc-500">local-first file agent</p>
        </div>
      </div>

      <div
        className="flex items-center gap-2 rounded-full border border-ink-700 bg-ink-900/80 px-3 py-1.5"
        title={status.error}
      >
        <span className={`size-2 rounded-full ${STATE_DOT[status.state]}`} />
        <span className="text-xs text-zinc-400">
          {STATE_LABEL[status.state]}
          {status.state === 'online' && status.port ? (
            <span className="text-zinc-600"> · 127.0.0.1:{status.port}</span>
          ) : null}
        </span>
      </div>
    </header>
  )
}
