// Browser side of the import server. When the server isn't running (for example the
// single-file build), link and photo import fall back to what the browser can do alone.

import type { ProductInfo } from './productPage'

const BASE = (import.meta.env.VITE_IMPORT_API as string | undefined) ?? ''

export interface ImportStatus {
  /** The import server answered. */
  server: boolean
  /** The server can call Claude (photo recognition and the link fallback). */
  llm: boolean
}

let statusPromise: Promise<ImportStatus> | undefined

export function importStatus(): Promise<ImportStatus> {
  statusPromise ??= fetch(`${BASE}/api/status`, { signal: AbortSignal.timeout(3000) })
    .then(async (r) => {
      if (!r.ok) return { server: false, llm: false }
      const j = await r.json()
      return { server: !!j.ok, llm: !!j.llm }
    })
    .catch(() => ({ server: false, llm: false }))
  return statusPromise
}

async function post(path: string, body: unknown): Promise<ProductInfo> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(120000) })
  } catch (e) {
    throw new Error((e as Error).name === 'TimeoutError' ? 'That took too long. Try again, or paste the size instead.' : "Couldn't reach the import server.")
  }
  const j = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(j.error ?? `The import server answered with HTTP ${res.status}.`)
  return j as ProductInfo
}

export const importLink = (url: string) => post('/api/import/link', { url })
export const importPhoto = (image: string, note?: string) => post('/api/import/photo', { image, note })

/** Load a store's 3D model through the import server, since stores rarely allow cross-site loads. */
export const proxiedModelUrl = (url: string) => (/^https?:\/\//i.test(url) ? `${BASE}/api/model?url=${encodeURIComponent(url)}` : url)
