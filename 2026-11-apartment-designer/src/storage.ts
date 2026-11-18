import type { Doc } from './model/types'

// Milestone 1 keeps everything in the browser. The rest of the app only talks to
// these functions, so a server-backed store (e.g. Supabase) can replace them later.

const KEY = 'apartment-designer:doc:v1'

export function loadDoc(): Doc | null {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return null
    return parseDoc(raw)
  } catch {
    return null
  }
}

export function saveDoc(doc: Doc) {
  try {
    localStorage.setItem(KEY, JSON.stringify(doc))
  } catch {
    // Storage can be unavailable (private mode); the app still works for the session.
  }
}

export function parseDoc(raw: string): Doc | null {
  const d = JSON.parse(raw)
  if (!d || d.version !== 1 || !Array.isArray(d.rooms) || !Array.isArray(d.items) || !Array.isArray(d.layouts) || !d.layouts.length) return null
  if (!d.layouts.some((l: { id: string }) => l.id === d.activeLayoutId)) d.activeLayoutId = d.layouts[0].id
  return d as Doc
}

export function downloadJSON(doc: Doc) {
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' })
  downloadBlob(blob, `${slug(doc.apartment.name)}.json`)
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function slug(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'apartment'
}
