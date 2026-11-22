// Keyboard and selection-bar edits on the selected piece: rotating (with a re-snap so it stays
// flush and clear of its neighbours) and nudging (along the screen in 3D). A run of nudges or
// turns is one undo step.
import { add, orientedRect, pointInPolygon, polygonInsideRoom, polygonsOverlap, rot, v } from '../model/geometry'
import { clampOnHost, HOSTED, hostAt } from '../model/placement'
import { snapPlacement } from '../model/snapping'
import type { Item, Placement, Room, Vec2 } from '../model/types'
import { activeLayout, useStore, type PlaceAt } from '../store'
import { viewFrame } from '../three/viewFrame'

const IDLE_MS = 600
let runTimer: ReturnType<typeof setTimeout> | undefined
let inRun = false

/** Close the current run of nudges and turns, so the next edit is its own undo step. */
export function flushEditRun() {
  clearTimeout(runTimer)
  runTimer = undefined
  if (inRun) {
    inRun = false
    useStore.getState().endGesture()
  }
}

function inEditRun(fn: () => void) {
  const s = useStore.getState()
  // A drag (or anything else) already holds the history open: don't nest.
  if (!inRun && s.gesture) return fn()
  if (!inRun) {
    s.beginGesture()
    inRun = true
  }
  fn()
  clearTimeout(runTimer)
  runTimer = setTimeout(flushEditRun, IDLE_MS)
}

const body = (item: Item, p: Pick<Placement, 'x' | 'y' | 'rotation'>) => orientedRect(v(p.x, p.y), item.width / 2, item.depth / 2, p.rotation)

export type Others = { id: string; item: Item; p: Placement }[]
export type Conflicts = { outside: boolean; hits: Set<string> }

/** What a pose runs into: out of every room, and which solid pieces it overlaps (rugs never block). */
function conflicts(item: Item, p: Placement, others: Others, rooms: Room[] = useStore.getState().doc.rooms): Conflicts {
  if (p.on) return { outside: false, hits: new Set() }
  others = others.filter((o) => !o.p.on)
  const b = body(item, p)
  const outside = rooms.length > 0 && !rooms.some((r) => polygonInsideRoom(b, r.polygon))
  const hits = new Set(item.floorLevel ? [] : others.filter((o) => !o.item.floorLevel && polygonsOverlap(b, body(o.item, o.p))).map((o) => o.id))
  return { outside, hits }
}

/** A pose is fine if it's no worse than where the piece was: no new wall crossing, no new overlap. */
const fineAt = (item: Item, p: Placement, others: Others, base: Conflicts, rooms?: Room[]) => {
  const c = conflicts(item, p, others, rooms)
  return (!c.outside || base.outside) && [...c.hits].every((h) => base.hits.has(h))
}

/**
 * The closest fine spot to `p` (same rotation) within 2 m, nearest rings first. With `walls`, only
 * staying inside the room counts (the fallback when every nearby spot touches a neighbour).
 */
function nearestFine(item: Item, p: Placement, others: Others, base: Conflicts, walls = false, rooms?: Room[], maxRings = 40): Placement | null {
  const step = 0.05
  const ok = walls ? (q: Placement) => !conflicts(item, q, others, rooms).outside : (q: Placement) => fineAt(item, q, others, base, rooms)
  for (let ring = 1; ring <= maxRings; ring++) {
    let best: Placement | null = null
    let bestD = Infinity
    for (let i = -ring; i <= ring; i++)
      for (let j = -ring; j <= ring; j++) {
        if (Math.max(Math.abs(i), Math.abs(j)) !== ring) continue
        const q = { ...p, x: p.x + i * step, y: p.y + j * step }
        const d = i * i + j * j
        if (d < bestD && ok(q)) {
          best = q
          bestD = d
        }
      }
    if (best) return best
  }
  return null
}

/**
 * Dragging a piece: `start` is where it was picked up. A pose is clean when it's no worse than the
 * start (no new wall crossing, no new overlap), which is what the floor footprint shows green.
 */
export function dragConflictBase(item: Item, start: Placement, others: Others, rooms: Room[]): Conflicts {
  return conflicts(item, start, others, rooms)
}
export function isCleanPose(item: Item, p: Placement, others: Others, rooms: Room[], base: Conflicts): boolean {
  return fineAt(item, p, others, base, rooms)
}

/** How far (m) a dropped piece may slide to the nearest clean spot before it goes back instead. */
export const DROP_SLIDE = 0.35

/**
 * Where a dragged piece settles when it's let go at `drop`: right there when that's clean; else
 * the nearest clean spot within DROP_SLIDE (like Rotate's slide); else back to `lastClean`, the
 * last clean pose the drag passed through. `bounced` says the piece didn't stay where it was dropped.
 */
export function settleDrop(
  item: Item,
  drop: Placement,
  lastClean: Placement,
  others: Others,
  rooms: Room[],
  base: Conflicts,
): { placement: Placement; bounced: boolean } {
  if (fineAt(item, drop, others, base, rooms)) return { placement: drop, bounced: false }
  const near = nearestFine(item, drop, others, base, false, rooms, Math.round(DROP_SLIDE / 0.05))
  return { placement: near ?? lastClean, bounced: true }
}

const TAU = Math.PI * 2
const normAngle = (a: number) => ((a % TAU) + TAU) % TAU

/**
 * Turn the selected piece by `delta` (snapped to multiples of it). Quarter turns re-snap to walls
 * and edges; if the turned piece now runs into a wall or a neighbour, it slides to the nearest
 * free spot.
 */
export function rotatePlacementSmart(id: string, delta: number) {
  const s = useStore.getState()
  const layout = activeLayout(s.doc)
  const p = layout.placements.find((x) => x.id === id)
  const item = p && s.doc.items.find((i) => i.id === p.itemId)
  if (!p || !item) return
  const byId = new Map(s.doc.items.map((i) => [i.id, i]))
  const others: Others = layout.placements.flatMap((o) => {
    const it = o.id === id ? undefined : byId.get(o.itemId)
    return it ? [{ id: o.id, item: it, p: o }] : []
  })
  const step = Math.abs(delta)
  let next: Placement = { ...p, rotation: normAngle(Math.round((p.rotation + delta) / step) * step) }
  const host = p.on ? layout.placements.find((x) => x.id === p.on) : undefined
  const hostItem = host && byId.get(host.itemId)
  if (host && hostItem) {
    // A table lamp turns in place on its host and stays on the top.
    const c = clampOnHost({ item: hostItem, p: host }, item, v(next.x, next.y))
    inEditRun(() => useStore.getState().updatePlacement(id, { x: c.x, y: c.y, rotation: next.rotation }))
    return
  }
  const base = conflicts(item, p, others)
  const quarter = Math.abs(Math.round(next.rotation / (Math.PI / 2)) * (Math.PI / 2) - next.rotation) < 1e-6
  if (!fineAt(item, next, others, base))
    next = nearestFine(item, next, others, base) ?? (base.outside ? null : nearestFine(item, next, others, base, true)) ?? next
  if (quarter) {
    // Wall snapping would pull a fine angle back square, so only quarter turns re-snap.
    const snapped = snapPlacement(item, next, s.doc.rooms, others, s.snap).placement
    const keep = { ...snapped, rotation: next.rotation }
    if (fineAt(item, keep, others, base)) next = keep
  }
  inEditRun(() => useStore.getState().updatePlacement(id, { x: next.x, y: next.y, rotation: next.rotation }))
}

/**
 * Move the selected piece by a screen-direction step: (1, 0) is right, (0, -1) is up. In 3D the
 * direction turns with the camera (snapped to the nearest plan axis) so Right always moves right.
 */
export function nudgePlacement(id: string, screen: Vec2, step: number) {
  const s = useStore.getState()
  const p = activeLayout(s.doc).placements.find((x) => x.id === id)
  if (!p) return
  let d = screen
  const az = viewFrame.azimuth
  if (az !== null) d = rot(screen, Math.round(az / (Math.PI / 2)) * (Math.PI / 2))
  const to = add(v(p.x, p.y), v(Math.round(d.x) * step, Math.round(d.y) * step))
  if (p.on) {
    // A table lamp only slides around its host's top; a step off the edge is ignored.
    const host = activeLayout(s.doc).placements.find((x) => x.id === p.on)
    const hostItem = host && s.doc.items.find((i) => i.id === host.itemId)
    const item = s.doc.items.find((i) => i.id === p.itemId)
    if (host && hostItem && item) {
      const c = clampOnHost({ item: hostItem, p: host }, item, to)
      if (Math.hypot(c.x - to.x, c.y - to.y) > 1e-6) return
    }
  }
  inEditRun(() => useStore.getState().updatePlacement(id, { x: to.x, y: to.y }))
}

/**
 * Where a piece dragged in from the drawer lands when it's let go at `pt` (plan meters): snapped
 * like a moved piece, and nudged (up to a metre) to the nearest spot clear of walls and
 * neighbours. A table lamp goes on the top under the pointer. Null when the point is outside every
 * room (or nothing near it fits): the caller falls back to the usual smart placement.
 */
export function dropSpotAt(item: Item, pt: Vec2): PlaceAt | null {
  const s = useStore.getState()
  const doc = s.doc
  const layout = activeLayout(doc)
  if (HOSTED.has(item.category)) {
    const host = hostAt(doc, layout, pt, item)
    if (!host) return null
    const c = clampOnHost(host, item, pt)
    return { x: c.x, y: c.y, rotation: host.p.rotation, on: host.id }
  }
  const room = doc.rooms.find((r) => pointInPolygon(pt, r.polygon))
  if (!room) return null
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  const others: Others = layout.placements.flatMap((o) => {
    const it = byId.get(o.itemId)
    return it && !o.on ? [{ id: o.id, item: it, p: o }] : []
  })
  const proposed: Placement = { id: '', itemId: item.id, x: pt.x, y: pt.y, rotation: 0 }
  const snapped = { ...proposed, ...snapPlacement(item, proposed, doc.rooms, others, s.snap).placement }
  const clean: Conflicts = { outside: false, hits: new Set() }
  const spot = fineAt(item, snapped, others, clean, doc.rooms) ? snapped : nearestFine(item, snapped, others, clean, false, doc.rooms, 20)
  if (!spot || !pointInPolygon(v(spot.x, spot.y), room.polygon)) return null
  return { x: spot.x, y: spot.y, rotation: spot.rotation }
}
