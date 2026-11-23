import { create } from 'zustand'
import { catalogEntry, itemFromCatalog, type CatalogEntry } from './model/catalog'
import { add, dist, norm, pointInPolygon, scale, sub } from './model/geometry'
import { findDrop, HOSTED, reconcileHosts, type DropResult } from './model/placement'
import { reconcileWallDecor, wallDecorMessage, type WallDecorReport } from './model/wallDecor'
import { emptyDoc, rectRoom, sampleDoc, uid } from './model/sample'
import type { SnapSettings } from './model/snapping'
import type { Doc, Item, Layout, Opening, Placement, Room, ScanRef, ScanReference, Units, Vec2 } from './model/types'
import { calibrate, rescaleAbout } from './scan/align'
import type { TwistId } from './model/types'
import { remixedLayout, TWISTS } from './game/goals'
import { useGame } from './game/gameStore'
import { deleteHome as dropHome, listHomes, loadDoc, loadHome, restoreHome as undropHome, saveDoc, setCurrentHome } from './storage'

export type Mode = 'select' | 'draw-room' | 'door' | 'window' | 'scan-measure' | 'scan-straighten' | 'scan-move'
export type View = 'split' | 'plan' | '3d' | 'compare'

export type Selection =
  | { kind: 'placement'; id: string }
  | { kind: 'room'; id: string }
  | { kind: 'opening'; roomId: string; id: string }
  | { kind: 'scan'; id: string }
  | null

interface UIState {
  mode: Mode
  view: View
  selection: Selection
  snap: SnapSettings
  showClearance: boolean
  showPath: boolean
  compare: [string, string] | null
  toast: string | null
  /** The scan that measure / straighten / move modes act on. */
  activeScanId: string | null
  /** A reference length being drawn on a scan (plan coordinates), waiting for its true length. */
  scanDraft: { scanId: string; a: Vec2; b?: Vec2 } | null
  /** 3D view: how to show scans. */
  scan3d: 'off' | 'with-model' | 'scan-only'
  /** The last save didn't reach storage (full, most likely). It's retried on every change. */
  saveFailed: boolean
}

interface State extends UIState {
  doc: Doc
  past: Doc[]
  future: Doc[]
  /** True while a drag is in progress: changes don't add history entries. */
  gesture: boolean

  // History
  update: (fn: (d: Doc) => Doc) => void
  beginGesture: () => void
  endGesture: () => void
  undo: () => void
  redo: () => void

  // UI
  setMode: (m: Mode) => void
  setView: (v: View) => void
  select: (s: Selection) => void
  setSnap: (s: Partial<SnapSettings>) => void
  setUI: (s: Partial<UIState>) => void
  notify: (msg: string) => void

  // Document
  replaceDoc: (d: Doc) => void
  setUnits: (u: Units) => void
  renameApartment: (name: string) => void
  addRect: (name: string, w: number, d: number) => void
  addRoom: (name: string, polygon: Vec2[]) => string
  updateRoom: (id: string, patch: Partial<Room>) => void
  deleteRoom: (id: string) => void
  moveRoom: (id: string, delta: Vec2) => void
  setVertex: (roomId: string, index: number, p: Vec2) => void
  insertVertex: (roomId: string, wall: number, p: Vec2) => void
  deleteVertex: (roomId: string, index: number) => void
  setWallLength: (roomId: string, wall: number, length: number) => void
  addOpening: (roomId: string, o: Omit<Opening, 'id'>) => void
  updateOpening: (roomId: string, id: string, patch: Partial<Opening>) => void
  deleteOpening: (roomId: string, id: string) => void

  /** Without `at`, the piece goes to a free spot that suits it; when none exists nothing is added and a toast says why. */
  addItemFromCatalog: (entry: CatalogEntry, at?: PlaceAt) => PlaceResult
  /** Add a new item to the library and place it in the active layout. */
  addItem: (item: Item, at?: PlaceAt) => PlaceResult
  placeItem: (itemId: string, at?: PlaceAt) => PlaceResult
  updateItem: (id: string, patch: Partial<Item>) => void
  deleteItem: (id: string) => void
  updatePlacement: (id: string, patch: Partial<Placement>) => void
  deletePlacement: (id: string) => void
  duplicatePlacement: (id: string) => void
  /**
   * Unpack a moving box (an item with `unpack`): its contents become owned pieces placed near it
   * (or waiting in Mine when nothing fits) and the box goes, all in one undoable step. Returns the
   * new placement ids.
   */
  unpackBox: (placementId: string) => string[]
  rotatePlacement: (id: string, delta: number) => void

  addLayout: () => void
  duplicateLayout: (id: string) => void
  /** A remix of a design: only its big pieces (ANCHORS), plus a twist to meet. Becomes the active design. */
  remixLayout: (sourceId: string, twist: TwistId) => void
  /** A remix's first Dream home: noted on the design, with no undo entry. */
  markRemixDream: (layoutId: string) => void
  renameLayout: (id: string, name: string) => void
  deleteLayout: (id: string) => void
  setActiveLayout: (id: string) => void

  addScan: (ref: ScanRef) => void
  updateScan: (id: string, patch: Partial<ScanRef>) => void
  deleteScan: (id: string) => void
  /** Add or remove a reference length and recalibrate the scale, keeping `anchor` fixed on the plan. */
  addScanReference: (id: string, r: Omit<ScanReference, 'id'>, anchor: Vec2) => void
  updateScanReference: (id: string, refId: string, patch: Partial<ScanReference>, anchor: Vec2) => void
  deleteScanReference: (id: string, refId: string, anchor: Vec2) => void
}

const HISTORY_LIMIT = 200
let toastTimer: ReturnType<typeof setTimeout> | undefined

export const activeLayout = (d: Doc): Layout => d.layouts.find((l) => l.id === d.activeLayoutId) ?? d.layouts[0]

const mapRoom = (d: Doc, id: string, fn: (r: Room) => Room): Doc => ({ ...d, rooms: d.rooms.map((r) => (r.id === id ? fn(r) : r)) })
const mapLayout = (d: Doc, fn: (l: Layout) => Layout): Doc => ({ ...d, layouts: d.layouts.map((l) => (l.id === d.activeLayoutId ? fn(l) : l)) })

/**
 * Where a new piece lands when it isn't dropped somewhere specific: a legal spot (fully inside a
 * room, clear of other pieces and door swings) in the room that suits it, backed onto a wall or
 * next to its partner piece. See findDrop in model/placement.ts. Null with a reason when nothing fits.
 */
export function defaultDrop(d: Doc, item: Item, near?: Vec2): DropResult {
  return findDrop(d, activeLayout(d), item, { near })
}

/** Where a piece goes when the caller picks the spot: a point, optionally turned, optionally on a host (table lamps). */
export type PlaceAt = Vec2 & { rotation?: number; on?: string; mountY?: number }

/** What happened when a piece was added: where it went, or why it couldn't be placed. */
export type PlaceResult = { ok: true; placementId: string; roomName?: string } | { ok: false; reason: string }

/** Table lamps in the active layout standing on a placement that `gone` matches, with their host's name. */
function hostedOn(d: Doc, gone: (p: Placement) => boolean) {
  const l = activeLayout(d)
  const names = new Map(d.items.map((i) => [i.id, i.name]))
  const hosts = new Map(l.placements.filter(gone).map((p) => [p.id, p]))
  return l.placements.filter((p) => p.on && hosts.has(p.on) && !gone(p)).map((p) => ({ id: p.id, lamp: names.get(p.itemId) ?? 'lamp', host: names.get(hosts.get(p.on!)!.itemId) ?? 'table' }))
}

/** Say what happened to the lamps that stood on a deleted piece. */
function tellHosted(d: Doc, watch: ReturnType<typeof hostedOn>, notify: (m: string) => void) {
  if (!watch.length) return
  const l = activeLayout(d)
  const names = new Map(d.items.map((i) => [i.id, i.name]))
  const w = watch[0]
  const now = l.placements.find((p) => p.id === w.id)
  const host = now?.on && l.placements.find((p) => p.id === now.on)
  if (host) notify(`The ${w.lamp.toLowerCase()} moved to the ${(names.get(host.itemId) ?? 'other table').toLowerCase()}.`)
  else notify(`The ${w.lamp.toLowerCase()} came off with the ${w.host.toLowerCase()}. It’s waiting in Mine.`)
}

const withArticle = (name: string) => `${/^[aeiou]/.test(name) ? 'an' : 'a'} ${name}`
const listOf = (xs: string[]) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`)

/** What unpackDoc did: the new doc, the new placement ids, and the names placed and left waiting in Mine. */
export interface UnpackResult {
  doc: Doc
  ids: string[]
  placed: string[]
  waiting: string[]
}

/**
 * Unpack the moving box at `placementId` in the active layout (pure; see State.unpackBox). The box's
 * placement goes and its item is marked `unpacked`; each category inside becomes an owned item
 * placed at the nearest free spot to the box. Null when it isn't a packed box.
 */
export function unpackDoc(d0: Doc, placementId: string): UnpackResult | null {
  const box = activeLayout(d0).placements.find((p) => p.id === placementId)
  const item = box && d0.items.find((i) => i.id === box.itemId)
  if (!box || !item?.unpack?.length || item.unpacked) return null
  const at = { x: box.x, y: box.y }
  let d = mapLayout({ ...d0, items: d0.items.map((i) => (i.id === item.id ? { ...i, unpacked: true } : i)) }, (l) => ({ ...l, placements: l.placements.filter((p) => p.id !== placementId) }))
  const ids: string[] = []
  const placed: string[] = []
  const waiting: string[] = []
  for (const c of item.unpack) {
    const it = itemFromCatalog(catalogEntry(c), uid(), { owned: true })
    d = { ...d, items: [...d.items, it] }
    // Nearest first: a spot in the box's own room if there is one, then anywhere it suits.
    const home = d.rooms.find((r) => pointInPolygon(at, r.polygon))
    const local = home ? defaultDrop({ ...d, rooms: [home] }, it, at) : null
    const drop = local?.ok ? local : defaultDrop(d, it, at)
    const name = it.name.toLowerCase()
    if (!drop.ok) {
      waiting.push(name)
      continue
    }
    const pid = uid()
    const s = drop.spot
    d = mapLayout(d, (l) => ({ ...l, placements: [...l.placements, { id: pid, itemId: it.id, x: s.x, y: s.y, rotation: s.rotation, ...(s.on ? { on: s.on } : {}) }] }))
    ids.push(pid)
    placed.push(name)
  }
  return { doc: d, ids, placed, waiting }
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

/**
 * The toast after unpacking, short enough for two lines on a phone. Up to two things read in full
 * ("Unpacked a plant! A table lamp is waiting in Mine."); more are counted ("Unpacked 4 things! 3
 * are waiting in Mine.").
 */
export function unpackMessage(r: Pick<UnpackResult, 'placed' | 'waiting'>): string {
  const n = r.placed.length + r.waiting.length
  if (n > 2) {
    const head = `Unpacked ${n} things!`
    if (!r.waiting.length) return head
    return `${head} ${r.waiting.length === 1 ? '1 is' : `${r.waiting.length} are`} waiting in Mine.`
  }
  // Round 15: the piece comes first ("Table lamp → waiting in Mine"), so a narrow toast never cuts it off.
  if (!r.placed.length) return r.waiting.length ? `${cap(listOf(r.waiting))} → waiting in Mine. Unpacked!` : 'Unpacked the box!'
  const head = `Unpacked ${listOf(r.placed.map(withArticle))}!`
  if (!r.waiting.length) return head
  return `${cap(r.waiting[0])} → waiting in Mine. ${head}`
}

export const useStore = create<State>((set, get) => ({
  doc: loadDoc() ?? sampleDoc(),
  past: [],
  future: [],
  gesture: false,

  mode: 'select',
  view: 'split',
  selection: null,
  snap: { grid: 0.05, walls: true, items: true },
  showClearance: true,
  showPath: true,
  compare: null,
  toast: null,
  activeScanId: null,
  scanDraft: null,
  scan3d: 'with-model',
  saveFailed: false,

  update: (fn) =>
    set((s) => {
      const changed = fn(s.doc)
      if (changed === s.doc) return s
      // Table lamps follow their host, or move to another top (or come off) when it goes.
      let next = reconcileHosts(s.doc, changed)
      // --- wall decor (round 14): re-snap to the walls when rooms/openings change, size curtains to
      // their window, and say so when a piece comes down to Mine.
      const wallReport: WallDecorReport = { toMine: [] }
      next = reconcileWallDecor(s.doc, next, wallReport)
      const wallMsg = wallDecorMessage(wallReport)
      if (wallMsg) queueMicrotask(() => get().notify(wallMsg))
      // --- end wall decor
      if (s.gesture) return { doc: next }
      return { doc: next, past: [...s.past, s.doc].slice(-HISTORY_LIMIT), future: [] }
    }),
  beginGesture: () => set((s) => (s.gesture ? s : { gesture: true, past: [...s.past, s.doc].slice(-HISTORY_LIMIT), future: [] })),
  endGesture: () =>
    set((s) => {
      if (!s.gesture) return s
      // Drop the history entry if nothing actually changed.
      const last = s.past[s.past.length - 1]
      if (last === s.doc) return { gesture: false, past: s.past.slice(0, -1) }
      return { gesture: false }
    }),
  undo: () =>
    set((s) => {
      if (!s.past.length) return s
      const prev = s.past[s.past.length - 1]
      return { doc: prev, past: s.past.slice(0, -1), future: [s.doc, ...s.future], selection: validSelection(prev, s.selection) }
    }),
  redo: () =>
    set((s) => {
      if (!s.future.length) return s
      const next = s.future[0]
      return { doc: next, past: [...s.past, s.doc], future: s.future.slice(1), selection: validSelection(next, s.selection) }
    }),

  setMode: (mode) => set({ mode, scanDraft: null }),
  setView: (view) => set({ view }),
  select: (selection) => set({ selection }),
  setSnap: (p) => set((s) => ({ snap: { ...s.snap, ...p } })),
  setUI: (p) => set(p),
  notify: (msg) => {
    set({ toast: msg })
    clearTimeout(toastTimer)
    toastTimer = setTimeout(() => set({ toast: null }), 4000)
  },

  replaceDoc: (d) => set((s) => ({ doc: d, past: [...s.past, s.doc], future: [], selection: null })),
  setUnits: (units) => get().update((d) => ({ ...d, apartment: { ...d.apartment, units } })),
  renameApartment: (name) => get().update((d) => ({ ...d, apartment: { ...d.apartment, name } })),

  addRect: (name, w, dd) => {
    // Put the new room to the right of the existing ones.
    const maxX = get().doc.rooms.flatMap((r) => r.polygon).reduce((m, p) => Math.max(m, p.x), -Infinity)
    const room = rectRoom(name, Number.isFinite(maxX) ? maxX + 0.12 : 0, 0, w, dd)
    get().update((d) => ({ ...d, rooms: [...d.rooms, room] }))
    set({ selection: { kind: 'room', id: room.id } })
  },
  addRoom: (name, polygon) => {
    const room: Room = { id: uid(), name, polygon, ceilingHeight: 2.6, wallThickness: 0.12, openings: [] }
    get().update((d) => ({ ...d, rooms: [...d.rooms, room] }))
    set({ selection: { kind: 'room', id: room.id } })
    return room.id
  },
  updateRoom: (id, patch) => get().update((d) => mapRoom(d, id, (r) => ({ ...r, ...patch }))),
  deleteRoom: (id) => {
    get().update((d) => ({ ...d, rooms: d.rooms.filter((r) => r.id !== id) }))
    set({ selection: null })
  },
  moveRoom: (id, delta) => get().update((d) => mapRoom(d, id, (r) => ({ ...r, polygon: r.polygon.map((p) => add(p, delta)) }))),
  setVertex: (roomId, index, p) =>
    get().update((d) => mapRoom(d, roomId, (r) => ({ ...r, polygon: r.polygon.map((q, i) => (i === index ? p : q)) }))),
  insertVertex: (roomId, wall, p) =>
    get().update((d) =>
      mapRoom(d, roomId, (r) => {
        const polygon = [...r.polygon.slice(0, wall + 1), p, ...r.polygon.slice(wall + 1)]
        const a = r.polygon[wall]
        const splitAt = dist(a, p)
        const openings = r.openings.map((o) => {
          if (o.wall < wall) return o
          if (o.wall > wall) return { ...o, wall: o.wall + 1 }
          return o.offset >= splitAt ? { ...o, wall: wall + 1, offset: o.offset - splitAt } : o
        })
        return { ...r, polygon, openings }
      }),
    ),
  deleteVertex: (roomId, index) =>
    get().update((d) =>
      mapRoom(d, roomId, (r) => {
        if (r.polygon.length <= 3) return r
        const n = r.polygon.length
        const polygon = r.polygon.filter((_, i) => i !== index)
        // Openings on the two walls touching this vertex are dropped; later walls shift down.
        const before = (index - 1 + n) % n
        const openings = r.openings
          .filter((o) => o.wall !== index && o.wall !== before)
          .map((o) => (o.wall > index ? { ...o, wall: o.wall - 1 } : o))
        return { ...r, polygon, openings }
      }),
    ),
  setWallLength: (roomId, wall, length) =>
    get().update((d) =>
      mapRoom(d, roomId, (r) => {
        // Lengthen the wall along its own direction and shift the next vertex too,
        // so the following wall keeps its length and direction (a rectangle stays a rectangle).
        const n = r.polygon.length
        const a = r.polygon[wall]
        const b = r.polygon[(wall + 1) % n]
        const dir = norm(sub(b, a))
        const delta = scale(dir, length - dist(a, b))
        const i1 = (wall + 1) % n
        const i2 = (wall + 2) % n
        const polygon = r.polygon.map((p, i) => (i === i1 || (i === i2 && i2 !== wall) ? add(p, delta) : p))
        return { ...r, polygon }
      }),
    ),
  addOpening: (roomId, o) => {
    const id = uid()
    get().update((d) => mapRoom(d, roomId, (r) => ({ ...r, openings: [...r.openings, { ...o, id }] })))
    set({ selection: { kind: 'opening', roomId, id } })
  },
  updateOpening: (roomId, id, patch) =>
    get().update((d) => {
      let next = mapRoom(d, roomId, (r) => ({ ...r, openings: r.openings.map((o) => (o.id === id ? { ...o, ...patch } : o)) }))
      // Only one entrance door.
      if (patch.entrance)
        next = {
          ...next,
          rooms: next.rooms.map((r) => ({ ...r, openings: r.openings.map((o) => (o.id === id ? o : { ...o, entrance: false })) })),
        }
      return next
    }),
  deleteOpening: (roomId, id) => {
    get().update((d) => mapRoom(d, roomId, (r) => ({ ...r, openings: r.openings.filter((o) => o.id !== id) })))
    set({ selection: null })
  },

  addItemFromCatalog: (entry, at) => get().addItem(itemFromCatalog(entry, uid()), at),
  addItem: (item, at) => {
    const d0 = get().doc
    const drop = at ? null : defaultDrop(d0, item)
    if (drop && !drop.ok) {
      get().notify(drop.reason)
      return drop
    }
    const spot: { x: number; y: number; rotation: number; on?: string; mountY?: number } = drop?.ok ? drop.spot : { x: at!.x, y: at!.y, rotation: at!.rotation ?? 0, on: at!.on, mountY: at!.mountY }
    const pid = uid()
    get().update((d) => mapLayout({ ...d, items: [...d.items, item] }, (l) => ({ ...l, placements: [...l.placements, { id: pid, itemId: item.id, x: spot.x, y: spot.y, rotation: spot.rotation, ...(spot.on ? { on: spot.on } : {}), ...(spot.mountY != null ? { mountY: spot.mountY } : {}) }] })))
    set({ selection: { kind: 'placement', id: pid }, mode: 'select' })
    return { ok: true, placementId: pid, roomName: drop?.ok ? drop.spot.roomName : undefined }
  },
  placeItem: (itemId, at) => {
    const d0 = get().doc
    const item = d0.items.find((i) => i.id === itemId)
    if (!item) return { ok: false, reason: 'That piece is gone.' }
    const drop = at ? null : defaultDrop(d0, item)
    if (drop && !drop.ok) {
      get().notify(drop.reason)
      return drop
    }
    const spot: { x: number; y: number; rotation: number; on?: string; mountY?: number } = drop?.ok ? drop.spot : { x: at!.x, y: at!.y, rotation: at!.rotation ?? 0, on: at!.on, mountY: at!.mountY }
    const pid = uid()
    get().update((d) => mapLayout(d, (l) => ({ ...l, placements: [...l.placements, { id: pid, itemId, x: spot.x, y: spot.y, rotation: spot.rotation, ...(spot.on ? { on: spot.on } : {}), ...(spot.mountY != null ? { mountY: spot.mountY } : {}) }] })))
    set({ selection: { kind: 'placement', id: pid }, mode: 'select' })
    return { ok: true, placementId: pid, roomName: drop?.ok ? drop.spot.roomName : undefined }
  },
  updateItem: (id, patch) => get().update((d) => ({ ...d, items: d.items.map((i) => (i.id === id ? { ...i, ...patch } : i)) })),
  deleteItem: (id) => {
    const watch = hostedOn(get().doc, (p) => p.itemId === id)
    get().update((d) => ({
      ...d,
      items: d.items.filter((i) => i.id !== id),
      layouts: d.layouts.map((l) => ({ ...l, placements: l.placements.filter((p) => p.itemId !== id) })),
    }))
    tellHosted(get().doc, watch, get().notify)
    set({ selection: null })
  },
  updatePlacement: (id, patch) => get().update((d) => mapLayout(d, (l) => ({ ...l, placements: l.placements.map((p) => (p.id === id ? { ...p, ...patch } : p)) }))),
  deletePlacement: (id) => {
    const d0 = get().doc
    const watch = hostedOn(d0, (p) => p.id === id)
    const item = d0.items.find((i) => i.id === activeLayout(d0).placements.find((p) => p.id === id)?.itemId)
    // The last copy of a piece you bought goes to Mine (marked owned) rather than vanishing, so it
    // can be placed again. Moving boxes and pieces already yours just come off the floor.
    const keep = !!item && !item.owned && !item.unpack?.length && !d0.layouts.some((l) => l.placements.some((p) => p.id !== id && p.itemId === item.id))
    get().update((d) => {
      const off = mapLayout(d, (l) => ({ ...l, placements: l.placements.filter((p) => p.id !== id) }))
      return keep ? { ...off, items: off.items.map((i) => (i.id === item!.id ? { ...i, owned: true } : i)) } : off
    })
    tellHosted(get().doc, watch, get().notify)
    if (keep) get().notify(`Moved the ${item!.name.toLowerCase()} to Mine. Place it again any time.`)
    set({ selection: null })
  },
  unpackBox: (placementId) => {
    const r = unpackDoc(get().doc, placementId)
    if (!r) return []
    get().update(() => r.doc)
    set({ selection: null, mode: 'select' })
    get().notify(unpackMessage(r))
    return r.ids
  },
  duplicatePlacement: (id) => {
    const d0 = get().doc
    const p = activeLayout(d0).placements.find((x) => x.id === id)
    const item = d0.items.find((i) => i.id === p?.itemId)
    if (!p || !item) return
    // A duplicate is a new item (its own size and name), placed a little offset.
    const copy: Item = { ...item, id: uid(), clearance: { ...item.clearance } }
    const pid = uid()
    // Prefer a free spot near the original (the other side of the bed for a nightstand); else a small offset.
    const drop = findDrop(d0, activeLayout(d0), copy, { near: { x: p.x, y: p.y } })
    // A table lamp never floats: with no free top, nothing is copied.
    if (!drop.ok && HOSTED.has(item.category)) {
      get().notify(drop.reason)
      return
    }
    const spot = drop.ok ? drop.spot : { x: p.x + 0.3, y: p.y + 0.3, rotation: p.rotation, on: undefined }
    get().update((d) =>
      mapLayout({ ...d, items: [...d.items, copy] }, (l) => ({
        ...l,
        placements: [...l.placements, { ...p, id: pid, itemId: copy.id, x: spot.x, y: spot.y, rotation: spot.rotation, on: spot.on, mountY: undefined }],
      })),
    )
    set({ selection: { kind: 'placement', id: pid } })
  },
  rotatePlacement: (id, delta) =>
    get().update((d) =>
      mapLayout(d, (l) => ({
        ...l,
        placements: l.placements.map((p) => {
          if (p.id !== id) return p
          const step = Math.abs(delta)
          const r = Math.round((p.rotation + delta) / step) * step
          return { ...p, rotation: ((r % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) }
        }),
      })),
    ),

  // Switching designs (including into a new one or a copy) starts that design's undo history
  // afresh, so Undo never jumps from one design to another.
  addLayout: () =>
    switchDesign((d) => {
      const id = uid()
      return { ...d, layouts: [...d.layouts, { id, name: designName(d.layouts.map((l) => l.name)), placements: [] }], activeLayoutId: id }
    }),
  duplicateLayout: (id) =>
    switchDesign((d) => {
      const src = d.layouts.find((l) => l.id === id)
      if (!src) return d
      const nid = uid()
      // Placements reference the shared item library, so a copy just needs new placement ids
      // (and table lamps pointed at the copies of their hosts).
      const ids = new Map(src.placements.map((p) => [p.id, uid()]))
      const placements = src.placements.map((p) => ({ ...p, id: ids.get(p.id)!, ...(p.on ? { on: ids.get(p.on) ?? p.on } : {}) }))
      const copy: Layout = { id: nid, name: copyName(src.name, d.layouts.map((l) => l.name)), placements }
      inheritBest(d, src.id, nid)
      return { ...d, layouts: [...d.layouts, copy], activeLayoutId: nid }
    }),
  remixLayout: (sourceId, twist) => {
    const copy = remixedLayout(get().doc, sourceId, twist, uid)
    if (!copy) return
    switchDesign((d) => ({ ...d, layouts: [...d.layouts, copy], activeLayoutId: copy.id }))
    const n = copy.placements.length
    get().notify(`${TWISTS[twist].name}! Kept ${n === 1 ? '1 big piece' : `${n} big pieces`}; the rest wait in Mine.`)
  },
  markRemixDream: (layoutId) =>
    set((s) => {
      const l = s.doc.layouts.find((x) => x.id === layoutId)
      if (!l?.remix || l.remix.dreamAt) return s
      const remix = { ...l.remix, dreamAt: Date.now() }
      return { doc: { ...s.doc, layouts: s.doc.layouts.map((x) => (x.id === layoutId ? { ...x, remix } : x)) } }
    }),
  renameLayout: (id, name) => get().update((d) => ({ ...d, layouts: d.layouts.map((l) => (l.id === id ? { ...l, name } : l)) })),
  deleteLayout: (id) =>
    get().update((d) => {
      if (d.layouts.length <= 1) return d
      const layouts = d.layouts.filter((l) => l.id !== id)
      return { ...d, layouts, activeLayoutId: d.activeLayoutId === id ? layouts[0].id : d.activeLayoutId }
    }),
  setActiveLayout: (id) => switchDesign((d) => (d.activeLayoutId === id || !d.layouts.some((l) => l.id === id) ? d : { ...d, activeLayoutId: id })),

  addScan: (ref) => {
    get().update((d) => ({ ...d, scans: [...(d.scans ?? []), ref] }))
    set({ selection: { kind: 'scan', id: ref.id }, activeScanId: ref.id })
  },
  updateScan: (id, patch) => get().update((d) => mapScan(d, id, (r) => ({ ...r, ...patch }))),
  deleteScan: (id) => {
    get().update((d) => ({ ...d, scans: (d.scans ?? []).filter((r) => r.id !== id) }))
    set((s) => ({ selection: null, activeScanId: s.activeScanId === id ? null : s.activeScanId, mode: s.mode.startsWith('scan-') ? 'select' : s.mode }))
  },
  addScanReference: (id, r, anchor) => get().update((d) => mapScan(d, id, (s) => recalibrated(s, [...s.references, { ...r, id: uid() }], anchor))),
  updateScanReference: (id, refId, patch, anchor) =>
    get().update((d) => mapScan(d, id, (s) => recalibrated(s, s.references.map((r) => (r.id === refId ? { ...r, ...patch } : r)), anchor))),
  deleteScanReference: (id, refId, anchor) => get().update((d) => mapScan(d, id, (s) => recalibrated(s, s.references.filter((r) => r.id !== refId), anchor))),
}))

/** A change that moves to another design: applied without an undo entry, and history starts over. */
function switchDesign(fn: (d: Doc) => Doc) {
  const s = useStore.getState()
  const next = fn(s.doc)
  if (next === s.doc) return
  useStore.setState({ doc: next, past: [], future: [], selection: null, gesture: false })
}

/** An exact copy starts at its source's best level, said outright rather than left to the HUD. */
function inheritBest(d: Doc, sourceId: string, copyId: string) {
  const g = useGame.getState()
  const legacy = d.layouts[0]?.id === sourceId ? g.bestLevel[d.apartment.id] : undefined
  const best = g.bestLevel[`${d.apartment.id}:${sourceId}`] ?? legacy ?? 0
  if (best > 0) g.noteLevel(`${d.apartment.id}:${copyId}`, best)
}

/** A copy is named after its source: "Move-in day" → "Move-in day 2", then "Move-in day 3". */
export function copyName(source: string, taken: string[]) {
  const t = new Set(taken)
  const base = source.replace(/ \d+$/, '') || source
  for (let n = 2; ; n++) if (!t.has(`${base} ${n}`)) return `${base} ${n}`
}

const mapScan = (d: Doc, id: string, fn: (r: ScanRef) => ScanRef): Doc => ({ ...d, scans: (d.scans ?? []).map((r) => (r.id === id ? fn(r) : r)) })

/** New references → new scale (the median of what they imply). With none left, the scale stays. */
function recalibrated(s: ScanRef, references: ScanReference[], anchor: Vec2): ScanRef {
  const c = calibrate(references)
  if (!c) return { ...s, references }
  return { ...s, references, ...rescaleAbout(s, c.scale, anchor) }
}

function validSelection(d: Doc, s: Selection): Selection {
  if (!s) return s
  if (s.kind === 'placement') return activeLayout(d).placements.some((p) => p.id === s.id) ? s : null
  if (s.kind === 'room') return d.rooms.some((r) => r.id === s.id) ? s : null
  if (s.kind === 'scan') return (d.scans ?? []).some((r) => r.id === s.id) ? s : null
  return d.rooms.some((r) => r.id === s.roomId && r.openings.some((o) => o.id === s.id)) ? s : null
}

/** Is this home still a blank slate (nothing drawn, placed or scanned)? */
const isBlank = (d: Doc) => !d.rooms.length && !d.items.length && !(d.scans ?? []).length

/** A name no other home has: "Sample apartment", then "Sample apartment 2"... */
function freshHomeName(base: string, except: string) {
  const taken = new Set(listHomes().filter((h) => h.id !== except).map((h) => h.name))
  if (!taken.has(base)) return base
  for (let n = 2; ; n++) if (!taken.has(`${base} ${n}`)) return `${base} ${n}`
}

/**
 * Open a brand new home in its own save slot. The home that was open stays saved and can be
 * picked again from Your designs. A blank home being left (a scan that was abandoned) is dropped
 * instead of piling up.
 */
function moveInto(d: Doc) {
  const prev = useStore.getState().doc
  const doc = { ...d, apartment: { ...d.apartment, name: freshHomeName(d.apartment.name, d.apartment.id) } }
  useStore.setState({ doc, past: [], future: [], selection: null, activeScanId: null, scanDraft: null, mode: 'select' })
  flushSave()
  if (isBlank(prev)) dropHome(prev.apartment.id)
}

export function newEmptyDoc() {
  const d = emptyDoc()
  moveInto({ ...d, layouts: d.layouts.map((l) => ({ ...l, name: DESIGN_NAMES[0] })) })
}
export function loadSample() {
  moveInto(sampleDoc())
}

/** Open a document from a file as a home of its own (a copy if that home is already saved). */
export function openDocAsHome(d: Doc) {
  const taken = listHomes().some((h) => h.id === d.apartment.id) || useStore.getState().doc.apartment.id === d.apartment.id
  moveInto(taken ? { ...d, apartment: { ...d.apartment, id: uid() } } : d)
}

/** Open another saved home. Undo history belongs to the home it was made in, so it starts fresh. */
export function switchHome(id: string): boolean {
  if (useStore.getState().doc.apartment.id === id) return true
  flushSave()
  const d = loadHome(id)
  if (!d) return false
  useStore.setState({ doc: d, past: [], future: [], selection: null, activeScanId: null, scanDraft: null, mode: 'select' })
  return true
}

/** Delete a saved home (never the last one). Deleting the open home switches to another first. */
export function deleteHome(id: string): boolean {
  const homes = listHomes()
  if (homes.length <= 1 || !homes.some((h) => h.id === id)) return false
  if (useStore.getState().doc.apartment.id === id) {
    const other = homes.find((h) => h.id !== id)
    if (!other || !switchHome(other.id)) return false
  }
  flushSave()
  // A save of the deleted home still waiting for space must not bring it back.
  pending.delete(id)
  return dropHome(id)
}

/** Bring back a home deleted with deleteHome (it stays in the trash for a week). */
export function restoreHome(id: string): boolean {
  flushSave()
  return undropHome(id)
}

/** Playful names for new designs and copies; numbered once they run out. */
export const DESIGN_NAMES = ['First idea', 'Cozy idea', 'Plan B', 'Fresh start', 'Sunday shuffle', 'Big move', 'Wild card', 'Snug version', 'Open plan', 'Plan C', 'Dream version', 'One more idea']
export function designName(taken: string[]) {
  const t = new Set(taken)
  const free = DESIGN_NAMES.find((n) => !t.has(n))
  if (free) return free
  for (let n = 2; ; n++) if (!t.has(`Idea ${n}`)) return `Idea ${n}`
}
export { catalogEntry }

// Persist the document locally (debounced), into the open home's own slot. A save that fails
// (storage full) stays pending, per home, and is retried on the next change and on pagehide; until
// then a toast that stays up offers a backup download.
let saveTimer: ReturnType<typeof setTimeout> | undefined
const pending = new Map<string, Doc>()
export function flushSave(): boolean {
  clearTimeout(saveTimer)
  for (const [id, d] of [...pending]) if (saveDoc(d)) pending.delete(id)
  const failed = pending.size > 0
  if (useStore.getState().saveFailed !== failed) useStore.setState({ saveFailed: failed })
  return !failed
}
useStore.subscribe((s, prev) => {
  if (s.doc === prev.doc) return
  // A different home: finish saving the one being left right away, then remember the new one.
  if (s.doc.apartment.id !== prev.doc.apartment.id) {
    flushSave()
    setCurrentHome(s.doc.apartment.id)
  }
  clearTimeout(saveTimer)
  pending.set(s.doc.apartment.id, s.doc)
  saveTimer = setTimeout(flushSave, 300)
})
if (typeof window !== 'undefined') window.addEventListener('pagehide', flushSave)
