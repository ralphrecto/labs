import { hostAt, HOSTED } from './model/placement'
import type { Doc, Item } from './model/types'

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
const TRASH_PREFIX = 'apartment-designer:trash:'
const trashKey = (id: string) => `${TRASH_PREFIX}${id}`
/** Best level per design, kept by the game (gameStore.ts), keyed `<home id>:<layout id>` or a bare home id. */
const BEST_LEVEL_KEY = 'apartment-designer:best-level'
/** A deleted home stays in the trash this long, so "Undo" can bring it back. */
export const TRASH_DAYS = 7
const DAY_MS = 24 * 60 * 60 * 1000

interface TrashEntry {
  deletedAt: number
  doc: Doc
  entry: HomeEntry
}

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

/** False when the index couldn't be written (storage full or unavailable). */
function writeIndex(ix: HomeIndex): boolean {
  const ls = store()
  if (!ls) return false
  try {
    ls.setItem(INDEX_KEY, JSON.stringify(ix))
    return true
  } catch {
    // Storage full or unavailable (private mode); the caller decides whether to retry.
    return false
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
  purgeTrash()
  const ix = readIndex()
  const ids = [ix.current, ...ix.homes.map((h) => h.id)].filter((x): x is string => !!x)
  for (const id of ids) {
    const d = loadHome(id)
    if (d) return d
  }
  return null
}

/**
 * Save a document into its home's slot (a new home if its apartment id is new). Returns false when
 * it couldn't be written (storage full, or no storage at all), so the caller can keep it and retry.
 */
export function saveDoc(doc: Doc): boolean {
  const ls = store()
  if (!ls) return false
  try {
    ls.setItem(homeKey(doc.apartment.id), JSON.stringify(doc))
  } catch {
    return false
  }
  const ix = readIndex()
  const entry = entryFor(doc)
  const i = ix.homes.findIndex((h) => h.id === entry.id)
  if (i >= 0) ix.homes[i] = entry
  else ix.homes.push(entry)
  if (!ix.current) ix.current = entry.id
  return writeIndex(ix)
}

/** Remember which home is open, so a reload comes back to it. */
export function setCurrentHome(id: string) {
  const ix = readIndex()
  if (ix.current === id) return
  ix.current = id
  writeIndex(ix)
}

/**
 * Delete a home: it moves to the trash (`apartment-designer:trash:<id>`) for TRASH_DAYS, where
 * restoreHome can bring it back. The last home can't be deleted.
 */
export function deleteHome(id: string, now = Date.now()): boolean {
  const ls = store()
  const ix = readIndex()
  const entry = ix.homes.find((h) => h.id === id)
  if (!ls || !entry || ix.homes.length <= 1) return false
  const doc = loadHome(id)
  if (doc) {
    const trash = JSON.stringify({ deletedAt: now, doc, entry } satisfies TrashEntry)
    try {
      ls.setItem(trashKey(id), trash)
    } catch {
      // Storage full: free the home's own slot first, then the same bytes fit in the trash.
      try {
        ls.removeItem(homeKey(id))
        ls.setItem(trashKey(id), trash)
      } catch {
        try {
          ls.setItem(homeKey(id), JSON.stringify(doc))
        } catch {
          // Nothing more to do.
        }
        return false
      }
    }
  }
  ix.homes = ix.homes.filter((h) => h.id !== id)
  if (ix.current === id) ix.current = ix.homes[0]?.id ?? null
  if (!writeIndex(ix)) {
    try {
      if (doc && !ls.getItem(homeKey(id))) ls.setItem(homeKey(id), JSON.stringify(doc))
      ls.removeItem(trashKey(id))
    } catch {
      // ignore
    }
    return false
  }
  try {
    ls.removeItem(homeKey(id))
  } catch {
    // Nothing more to do.
  }
  if (!doc) forgetBestLevels(id)
  return true
}

function readTrash(id: string): TrashEntry | null {
  try {
    const raw = store()?.getItem(trashKey(id))
    if (!raw) return null
    const t = JSON.parse(raw) as TrashEntry
    if (!t || typeof t.deletedAt !== 'number' || !t.doc || !t.entry) return null
    const doc = parseDoc(JSON.stringify(t.doc))
    return doc ? { ...t, doc } : null
  } catch {
    return null
  }
}

/** Ids of the homes in the trash. */
export function trashedHomeIds(): string[] {
  const ls = store()
  if (!ls) return []
  const ids: string[] = []
  try {
    for (let i = 0; i < ls.length; i++) {
      const k = ls.key(i)
      if (k?.startsWith(TRASH_PREFIX)) ids.push(k.slice(TRASH_PREFIX.length))
    }
  } catch {
    // ignore
  }
  return ids
}

/** Bring a deleted home back from the trash. False when it's gone (or can't be written). */
export function restoreHome(id: string): boolean {
  const ls = store()
  const t = readTrash(id)
  if (!ls || !t) return false
  try {
    ls.setItem(homeKey(id), JSON.stringify(t.doc))
  } catch {
    return false
  }
  const ix = readIndex()
  if (!ix.homes.some((h) => h.id === id)) ix.homes.push({ ...t.entry, updatedAt: Date.now() })
  if (!writeIndex(ix)) return false
  try {
    ls.removeItem(trashKey(id))
  } catch {
    // ignore
  }
  return true
}

/** Empty the trash of homes deleted more than TRASH_DAYS ago, with their best levels. */
export function purgeTrash(now = Date.now()) {
  const ls = store()
  if (!ls) return
  for (const id of trashedHomeIds()) {
    let deletedAt = 0
    try {
      deletedAt = Number(JSON.parse(ls.getItem(trashKey(id)) ?? '{}')?.deletedAt) || 0
    } catch {
      // Unreadable: purge it.
    }
    if (now - deletedAt < TRASH_DAYS * DAY_MS) continue
    try {
      ls.removeItem(trashKey(id))
    } catch {
      // ignore
    }
    forgetBestLevels(id)
  }
}

/** Drop a home's best-level records (per design `<id>:<layout>`, or the old bare `<id>`). */
function forgetBestLevels(id: string) {
  const ls = store()
  if (!ls) return
  try {
    const all = JSON.parse(ls.getItem(BEST_LEVEL_KEY) ?? '{}') as Record<string, number>
    if (!all || typeof all !== 'object') return
    const keep = Object.fromEntries(Object.entries(all).filter(([k]) => k !== id && !k.startsWith(`${id}:`)))
    if (Object.keys(keep).length !== Object.keys(all).length) ls.setItem(BEST_LEVEL_KEY, JSON.stringify(keep))
  } catch {
    // ignore
  }
}

/** Scan ids used by any saved home (plus `extra`), for cleaning up stored scan files. */
export function allHomeScanIds(extra: string[] = []): string[] {
  const ids = new Set(extra)
  for (const h of readIndex().homes) {
    const scans = h.scanIds ?? loadHome(h.id)?.scans?.map((s) => s.id) ?? []
    for (const s of scans) ids.add(s)
  }
  // Homes in the trash keep their scans, so Undo brings them back whole.
  for (const id of trashedHomeIds()) for (const s of readTrash(id)?.doc.scans ?? []) ids.add(s.id)
  return [...ids]
}

export function parseDoc(raw: string): Doc | null {
  const d = JSON.parse(raw)
  if (!d || d.version !== 1 || !Array.isArray(d.rooms) || !Array.isArray(d.items) || !Array.isArray(d.layouts) || !d.layouts.length) return null
  if (!d.apartment || typeof d.apartment.id !== 'string') return null
  if (!d.layouts.some((l: { id: string }) => l.id === d.activeLayoutId)) d.activeLayoutId = d.layouts[0].id
  // The sample's first design used to be called "Current".
  if (typeof d.apartment.name === 'string' && d.apartment.name.startsWith('Sample apartment'))
    for (const l of d.layouts) if (l.name === 'Current') l.name = 'Move-in day'
  migrateTableLamps(d as Doc)
  return d as Doc
}

/**
 * Table lamps used to be short floor lamps standing in the same spot as the nightstand (and
 * flagged as overlapping it). Give them their own category and stand them on the piece under them.
 */
function migrateTableLamps(d: Doc) {
  for (const it of d.items as Item[])
    if (it.category === 'floor-lamp' && it.height <= 0.9 && /table|desk|bedside/i.test(it.name ?? '')) it.category = 'table-lamp'
  const byId = new Map((d.items as Item[]).map((i) => [i.id, i]))
  for (const l of d.layouts) {
    if (!Array.isArray(l.placements)) continue
    for (const p of l.placements) {
      const it = byId.get(p.itemId)
      if (!it || !HOSTED.has(it.category) || p.on) continue
      const host = hostAt(d, l, { x: p.x, y: p.y }, it)
      if (host) {
        p.on = host.id
        p.rotation = host.p.rotation
      }
    }
  }
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
