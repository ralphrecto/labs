import type { Doc } from './model/types'

// Everything lives in the browser. The rest of the app only talks to these functions, so a
// server-backed store (e.g. Supabase) can replace them later.
//
// Several homes, each in its own slot: an index (`apartment-designer:homes`) lists them and
// remembers which one is open, and each home's document is kept under `apartment-designer:home:<id>`,
// where the id is the document's `apartment.id`. Saving a document with a new apartment id makes a
// new home, so nothing ever overwrites another home. Before homes existed there was one slot
// (`apartment-designer:doc:v1`); it is copied into the first home and left in place untouched.

const LEGACY_KEY = 'apartment-designer:doc:v1'
const INDEX_KEY = 'apartment-designer:homes'
const homeKey = (id: string) => `apartment-designer:home:${id}`

export interface HomeEntry {
  id: string
  name: string
  updatedAt: number
  /** Scans the home uses, so stored scan files of homes that aren't open are kept. */
  scanIds?: string[]
}

interface HomeIndex {
  version: 1
  current: string | null
  homes: HomeEntry[]
}

function store(): Storage | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null
  }
}

function readIndex(): HomeIndex {
  const ls = store()
  if (!ls) return { version: 1, current: null, homes: [] }
  try {
    const raw = ls.getItem(INDEX_KEY)
    if (raw) {
      const ix = JSON.parse(raw) as HomeIndex
      if (ix && Array.isArray(ix.homes)) return { version: 1, current: ix.current ?? null, homes: ix.homes.filter((h) => h && typeof h.id === 'string') }
    }
  } catch {
    // A broken index is rebuilt from the legacy slot below (or starts empty).
  }
  return migrate(ls)
}

/** One-time move from the single legacy slot into the first home. The legacy slot is kept. */
function migrate(ls: Storage): HomeIndex {
  const ix: HomeIndex = { version: 1, current: null, homes: [] }
  try {
    const raw = ls.getItem(LEGACY_KEY)
    const doc = raw ? parseDoc(raw) : null
    if (doc) {
      ls.setItem(homeKey(doc.apartment.id), JSON.stringify(doc))
      ix.homes.push(entryFor(doc))
      ix.current = doc.apartment.id
    }
    ls.setItem(INDEX_KEY, JSON.stringify(ix))
  } catch {
    // Storage full or unavailable: keep going with what we have in memory.
  }
  return ix
}

function writeIndex(ix: HomeIndex) {
  try {
    store()?.setItem(INDEX_KEY, JSON.stringify(ix))
  } catch {
    // Storage can be unavailable (private mode); the app still works for the session.
  }
}

const entryFor = (doc: Doc): HomeEntry => ({
  id: doc.apartment.id,
  name: doc.apartment.name,
  updatedAt: Date.now(),
  scanIds: (doc.scans ?? []).map((s) => s.id),
})

/** All saved homes, most recently changed first. */
export function listHomes(): HomeEntry[] {
  return [...readIndex().homes].sort((a, b) => b.updatedAt - a.updatedAt)
}

export function currentHomeId(): string | null {
  return readIndex().current
}

export function loadHome(id: string): Doc | null {
  try {
    const raw = store()?.getItem(homeKey(id))
    return raw ? parseDoc(raw) : null
  } catch {
    return null
  }
}

/** The home that was open last (after migrating the legacy slot if needed). */
export function loadDoc(): Doc | null {
  const ix = readIndex()
  const ids = [ix.current, ...ix.homes.map((h) => h.id)].filter((x): x is string => !!x)
  for (const id of ids) {
    const d = loadHome(id)
    if (d) return d
  }
  return null
}

/** Save a document into its home's slot (a new home if its apartment id is new). */
export function saveDoc(doc: Doc) {
  const ls = store()
  if (!ls) return
  try {
    ls.setItem(homeKey(doc.apartment.id), JSON.stringify(doc))
  } catch {
    return
  }
  const ix = readIndex()
  const entry = entryFor(doc)
  const i = ix.homes.findIndex((h) => h.id === entry.id)
  if (i >= 0) ix.homes[i] = entry
  else ix.homes.push(entry)
  if (!ix.current) ix.current = entry.id
  writeIndex(ix)
}

/** Remember which home is open, so a reload comes back to it. */
export function setCurrentHome(id: string) {
  const ix = readIndex()
  if (ix.current === id) return
  ix.current = id
  writeIndex(ix)
}

/** Remove a home for good. The last home can't be deleted. */
export function deleteHome(id: string): boolean {
  const ix = readIndex()
  if (!ix.homes.some((h) => h.id === id) || ix.homes.length <= 1) return false
  ix.homes = ix.homes.filter((h) => h.id !== id)
  if (ix.current === id) ix.current = ix.homes[0]?.id ?? null
  writeIndex(ix)
  try {
    store()?.removeItem(homeKey(id))
  } catch {
    // Nothing more to do.
  }
  return true
}

/** Scan ids used by any saved home (plus `extra`), for cleaning up stored scan files. */
export function allHomeScanIds(extra: string[] = []): string[] {
  const ids = new Set(extra)
  for (const h of readIndex().homes) {
    const scans = h.scanIds ?? loadHome(h.id)?.scans?.map((s) => s.id) ?? []
    for (const s of scans) ids.add(s)
  }
  return [...ids]
}

export function parseDoc(raw: string): Doc | null {
  const d = JSON.parse(raw)
  if (!d || d.version !== 1 || !Array.isArray(d.rooms) || !Array.isArray(d.items) || !Array.isArray(d.layouts) || !d.layouts.length) return null
  if (!d.apartment || typeof d.apartment.id !== 'string') return null
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
