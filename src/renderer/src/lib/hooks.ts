import { useCallback, useEffect, useRef, useState } from 'react'
import type { IndexStatus, OllamaStatus, SidecarStatus, TekSettings } from '../../../shared/types'

export function useSidecarStatus(): SidecarStatus {
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
  return status
}

export function useSettings(): {
  settings: TekSettings | null
  update: (changes: Partial<TekSettings>) => Promise<void>
  reload: () => Promise<void>
} {
  const [settings, setSettings] = useState<TekSettings | null>(null)
  const reload = useCallback(async () => {
    try {
      setSettings(await window.tek.settings.get())
    } catch {
      // sidecar not up yet — caller retries via key or status change
    }
  }, [])
  const update = useCallback(async (changes: Partial<TekSettings>) => {
    setSettings(await window.tek.settings.set(changes))
  }, [])
  useEffect(() => {
    void reload()
  }, [reload])
  return { settings, update, reload }
}

/** Polls /index/status — fast while an index runs, slow otherwise. */
export function useIndexStatus(online: boolean): IndexStatus | null {
  const [status, setStatus] = useState<IndexStatus | null>(null)
  const runningRef = useRef(false)
  useEffect(() => {
    if (!online) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const tick = async (): Promise<void> => {
      try {
        const s = await window.tek.index.status()
        if (cancelled) return
        setStatus(s)
        runningRef.current = s.running
      } catch {
        // transient — sidecar restarting
      }
      if (!cancelled) timer = setTimeout(tick, runningRef.current ? 600 : 4000)
    }
    void tick()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [online])
  return status
}

export function useOllamaStatus(online: boolean): { status: OllamaStatus | null; refresh: () => void } {
  const [status, setStatus] = useState<OllamaStatus | null>(null)
  const refresh = useCallback(() => {
    void window.tek
      .ollamaStatus()
      .then(setStatus)
      .catch(() => setStatus(null))
  }, [])
  useEffect(() => {
    if (!online) return
    refresh()
    const interval = setInterval(refresh, 15_000)
    return () => clearInterval(interval)
  }, [online, refresh])
  return { status, refresh }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unit = ''
  for (const u of units) {
    value /= 1024
    unit = u
    if (value < 1024) break
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${unit}`
}
