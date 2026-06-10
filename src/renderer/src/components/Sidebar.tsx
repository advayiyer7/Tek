import type { SidecarStatus } from '../../../shared/types'

export type Page = 'chat' | 'search' | 'library' | 'actions' | 'settings'

const NAV: { page: Page; label: string; icon: React.JSX.Element }[] = [
  {
    page: 'chat',
    label: 'Chat',
    icon: (
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
    )
  },
  {
    page: 'search',
    label: 'Search',
    icon: (
      <>
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.3-4.3" />
      </>
    )
  },
  {
    page: 'library',
    label: 'Library',
    icon: (
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    )
  },
  {
    page: 'actions',
    label: 'Actions',
    icon: <path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z" />
  },
  {
    page: 'settings',
    label: 'Settings',
    icon: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </>
    )
  }
]

const STATE_DOT: Record<SidecarStatus['state'], string> = {
  stopped: 'bg-zinc-500',
  starting: 'bg-amber-400 animate-pulse',
  online: 'bg-emerald-400',
  error: 'bg-rose-500'
}

const STATE_LABEL: Record<SidecarStatus['state'], string> = {
  stopped: 'Engine stopped',
  starting: 'Engine starting…',
  online: 'Engine online',
  error: 'Engine error'
}

export function Sidebar({
  page,
  onNavigate,
  status
}: {
  page: Page
  onNavigate: (page: Page) => void
  status: SidecarStatus
}): React.JSX.Element {
  return (
    <aside className="flex w-52 shrink-0 flex-col border-r border-ink-700/50 bg-ink-900/40">
      <div className="flex items-center gap-2.5 px-4 py-4">
        <div className="flex size-8 items-center justify-center rounded-lg bg-gradient-to-br from-teal-400 to-emerald-600 shadow-lg shadow-teal-500/20">
          <svg viewBox="0 0 24 24" className="size-4.5 text-ink-950" fill="currentColor">
            <path d="M4 5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1h-5v11a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1V8H5a1 1 0 0 1-1-1V5z" />
          </svg>
        </div>
        <div className="leading-tight">
          <h1 className="text-[15px] font-semibold tracking-wide text-zinc-100">Tek</h1>
          <p className="text-[10px] text-zinc-500">local-first file agent</p>
        </div>
      </div>

      <nav className="mt-2 flex flex-col gap-0.5 px-2">
        {NAV.map((item) => (
          <button
            key={item.page}
            onClick={() => onNavigate(item.page)}
            className={`flex items-center gap-3 rounded-lg px-3 py-2 text-left text-[13px] transition-colors ${
              page === item.page
                ? 'bg-teal-500/10 text-teal-300'
                : 'text-zinc-400 hover:bg-ink-800/70 hover:text-zinc-200'
            }`}
          >
            <svg
              viewBox="0 0 24 24"
              className="size-4 shrink-0"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {item.icon}
            </svg>
            {item.label}
          </button>
        ))}
      </nav>

      <div className="mt-auto px-4 py-3" title={status.error}>
        <div className="flex items-center gap-2 rounded-lg border border-ink-700 bg-ink-900/80 px-3 py-2">
          <span className={`size-2 shrink-0 rounded-full ${STATE_DOT[status.state]}`} />
          <span className="truncate text-[11px] text-zinc-500">{STATE_LABEL[status.state]}</span>
        </div>
      </div>
    </aside>
  )
}
