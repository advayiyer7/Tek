import { useEffect, useState } from 'react'
import { Sidebar, type Page } from './components/Sidebar'
import { useIndexStatus, useOllamaStatus, useSettings, useSidecarStatus } from './lib/hooks'
import { ActionsPage } from './pages/ActionsPage'
import { ChatPage } from './pages/ChatPage'
import { LibraryPage } from './pages/LibraryPage'
import { SearchPage } from './pages/SearchPage'
import { SettingsPage } from './pages/SettingsPage'

const PAGE_TITLE: Record<Page, { title: string; hint: string }> = {
  chat: { title: 'Chat', hint: 'grounded answers with citations' },
  search: { title: 'Search', hint: 'semantic search, fully offline' },
  library: { title: 'Library', hint: 'what Tek has indexed' },
  actions: { title: 'Actions', hint: 'preview first, you approve everything' },
  settings: { title: 'Settings', hint: 'engine, models, privacy' }
}

export default function App(): React.JSX.Element {
  const [page, setPage] = useState<Page>('chat')
  const sidecarStatus = useSidecarStatus()
  const online = sidecarStatus.state === 'online'
  const { settings, update, reload } = useSettings()
  const indexStatus = useIndexStatus(online)
  const { status: ollama, refresh: refreshOllama } = useOllamaStatus(online)

  useEffect(() => {
    if (online) void reload()
  }, [online, reload])

  const hasIndex = (indexStatus?.stats.files ?? 0) > 0

  return (
    <div className="relative flex h-full overflow-hidden">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(1100px 500px at 60% -10%, rgba(20,184,166,0.06), transparent 60%)'
        }}
      />

      <Sidebar page={page} onNavigate={setPage} status={sidecarStatus} />

      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        <header className="flex items-baseline gap-3 border-b border-ink-700/50 px-6 py-3.5">
          <h1 className="text-[15px] font-semibold text-zinc-100">{PAGE_TITLE[page].title}</h1>
          <span className="text-[11px] text-zinc-600">{PAGE_TITLE[page].hint}</span>
          {sidecarStatus.state === 'error' ? (
            <span className="ml-auto truncate font-mono text-[11px] text-rose-400" title={sidecarStatus.error}>
              engine error — see Settings
            </span>
          ) : null}
        </header>

        <main className="min-h-0 flex-1">
          {page === 'chat' ? (
            <ChatPage
              online={online}
              hasIndex={hasIndex}
              ollama={ollama}
              onGoToLibrary={() => setPage('library')}
            />
          ) : page === 'search' ? (
            <SearchPage online={online} />
          ) : page === 'library' ? (
            <LibraryPage
              online={online}
              settings={settings}
              updateSettings={update}
              indexStatus={indexStatus}
            />
          ) : page === 'actions' ? (
            <ActionsPage online={online} ollama={ollama} />
          ) : (
            <SettingsPage
              online={online}
              settings={settings}
              updateSettings={update}
              ollama={ollama}
              refreshOllama={refreshOllama}
            />
          )}
        </main>
      </div>
    </div>
  )
}
