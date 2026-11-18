import { create } from 'zustand'
import { catalogEntry, itemFromCatalog, type CatalogEntry } from './model/catalog'
import { add, dist, norm, scale, sub, v } from './model/geometry'
import { emptyDoc, rectRoom, sampleDoc, uid } from './model/sample'
import type { SnapSettings } from './model/snapping'
import type { Doc, Item, Layout, Opening, Placement, Room, Units, Vec2 } from './model/types'
import { loadDoc, saveDoc } from './storage'

export type Mode = 'select' | 'draw-room' | 'door' | 'window'
export type View = 'split' | 'plan' | '3d' | 'compare'

export type Selection =
  | { kind: 'placement'; id: string }
  | { kind: 'room'; id: string }
  | { kind: 'opening'; roomId: string; id: string }
  | null

interface UIState {
  mode: Mode
  view: View
  selection: Selection
  snap: SnapSettings
  showClearance: boolean
  showPath: boolean
  compare: [string, string] | null
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

  addItemFromCatalog: (entry: CatalogEntry, at?: Vec2) => void
  placeItem: (itemId: string, at?: Vec2) => void
  updateItem: (id: string, patch: Partial<Item>) => void
  deleteItem: (id: string) => void
  updatePlacement: (id: string, patch: Partial<Placement>) => void
  deletePlacement: (id: string) => void
  duplicatePlacement: (id: string) => void
  rotatePlacement: (id: string, delta: number) => void

  addLayout: () => void
  duplicateLayout: (id: string) => void
  renameLayout: (id: string, name: string) => void
  deleteLayout: (id: string) => void
  setActiveLayout: (id: string) => void
}

const HISTORY_LIMIT = 200

export const activeLayout = (d: Doc): Layout => d.layouts.find((l) => l.id === d.activeLayoutId) ?? d.layouts[0]

const mapRoom = (d: Doc, id: string, fn: (r: Room) => Room): Doc => ({ ...d, rooms: d.rooms.map((r) => (r.id === id ? fn(r) : r)) })
const mapLayout = (d: Doc, fn: (l: Layout) => Layout): Doc => ({ ...d, layouts: d.layouts.map((l) => (l.id === d.activeLayoutId ? fn(l) : l)) })

/** Center of the rooms' bounding box, as a default drop point. */
function defaultDrop(d: Doc): Vec2 {
  if (!d.rooms.length) return v(1, 1)
  const pts = d.rooms[0].polygon
  const sx = pts.reduce((a, p) => a + p.x, 0) / pts.length
  const sy = pts.reduce((a, p) => a + p.y, 0) / pts.length
  return v(sx, sy)
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

  update: (fn) =>
    set((s) => {
      const next = fn(s.doc)
      if (next === s.doc) return s
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

  setMode: (mode) => set({ mode }),
  setView: (view) => set({ view }),
  select: (selection) => set({ selection }),
  setSnap: (p) => set((s) => ({ snap: { ...s.snap, ...p } })),
  setUI: (p) => set(p),

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

  addItemFromCatalog: (entry, at) => {
    const item = itemFromCatalog(entry, uid())
    const pid = uid()
    get().update((d) => {
      const p = at ?? defaultDrop(d)
      return mapLayout({ ...d, items: [...d.items, item] }, (l) => ({ ...l, placements: [...l.placements, { id: pid, itemId: item.id, x: p.x, y: p.y, rotation: 0 }] }))
    })
    set({ selection: { kind: 'placement', id: pid }, mode: 'select' })
  },
  placeItem: (itemId, at) => {
    const pid = uid()
    get().update((d) => {
      const p = at ?? defaultDrop(d)
      return mapLayout(d, (l) => ({ ...l, placements: [...l.placements, { id: pid, itemId, x: p.x, y: p.y, rotation: 0 }] }))
    })
    set({ selection: { kind: 'placement', id: pid }, mode: 'select' })
  },
  updateItem: (id, patch) => get().update((d) => ({ ...d, items: d.items.map((i) => (i.id === id ? { ...i, ...patch } : i)) })),
  deleteItem: (id) => {
    get().update((d) => ({
      ...d,
      items: d.items.filter((i) => i.id !== id),
      layouts: d.layouts.map((l) => ({ ...l, placements: l.placements.filter((p) => p.itemId !== id) })),
    }))
    set({ selection: null })
  },
  updatePlacement: (id, patch) => get().update((d) => mapLayout(d, (l) => ({ ...l, placements: l.placements.map((p) => (p.id === id ? { ...p, ...patch } : p)) }))),
  deletePlacement: (id) => {
    get().update((d) => mapLayout(d, (l) => ({ ...l, placements: l.placements.filter((p) => p.id !== id) })))
    set({ selection: null })
  },
  duplicatePlacement: (id) => {
    const d0 = get().doc
    const p = activeLayout(d0).placements.find((x) => x.id === id)
    const item = d0.items.find((i) => i.id === p?.itemId)
    if (!p || !item) return
    // A duplicate is a new item (its own size and name), placed a little offset.
    const copy: Item = { ...item, id: uid(), clearance: { ...item.clearance } }
    const pid = uid()
    get().update((d) =>
      mapLayout({ ...d, items: [...d.items, copy] }, (l) => ({ ...l, placements: [...l.placements, { ...p, id: pid, itemId: copy.id, x: p.x + 0.3, y: p.y + 0.3 }] })),
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

  addLayout: () =>
    get().update((d) => {
      const id = uid()
      return { ...d, layouts: [...d.layouts, { id, name: `Layout ${d.layouts.length + 1}`, placements: [] }], activeLayoutId: id }
    }),
  duplicateLayout: (id) =>
    get().update((d) => {
      const src = d.layouts.find((l) => l.id === id)
      if (!src) return d
      const nid = uid()
      // Placements reference the shared item library, so a copy just needs new placement ids.
      const copy: Layout = { id: nid, name: `${src.name} (copy)`, placements: src.placements.map((p) => ({ ...p, id: uid() })) }
      return { ...d, layouts: [...d.layouts, copy], activeLayoutId: nid }
    }),
  renameLayout: (id, name) => get().update((d) => ({ ...d, layouts: d.layouts.map((l) => (l.id === id ? { ...l, name } : l)) })),
  deleteLayout: (id) =>
    get().update((d) => {
      if (d.layouts.length <= 1) return d
      const layouts = d.layouts.filter((l) => l.id !== id)
      return { ...d, layouts, activeLayoutId: d.activeLayoutId === id ? layouts[0].id : d.activeLayoutId }
    }),
  setActiveLayout: (id) => {
    get().update((d) => ({ ...d, activeLayoutId: id }))
    set({ selection: null })
  },
}))

function validSelection(d: Doc, s: Selection): Selection {
  if (!s) return s
  if (s.kind === 'placement') return activeLayout(d).placements.some((p) => p.id === s.id) ? s : null
  if (s.kind === 'room') return d.rooms.some((r) => r.id === s.id) ? s : null
  return d.rooms.some((r) => r.id === s.roomId && r.openings.some((o) => o.id === s.id)) ? s : null
}

export function newEmptyDoc() {
  useStore.getState().replaceDoc(emptyDoc())
}
export function loadSample() {
  useStore.getState().replaceDoc(sampleDoc())
}
export { catalogEntry }

// Persist the document locally (debounced).
let saveTimer: ReturnType<typeof setTimeout> | undefined
useStore.subscribe((s, prev) => {
  if (s.doc === prev.doc) return
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => saveDoc(s.doc), 300)
})
