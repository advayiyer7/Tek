import type { TekApi } from './index'

declare global {
  interface Window {
    tek: TekApi
  }
}

export {}
