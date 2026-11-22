// Wall decor: framed art, a round mirror, a floating shelf, a clock and curtains. These pieces hang
// flush on a wall instead of standing on the floor, so they take no floor area: no clearance, no
// share of a room's density, and they never block a walkway or a floor piece (art can hang over
// the sofa, a shelf over the desk). Their placement is a spot along a wall face: the centre sits
// just inside the wall, the front faces into the room, and the height comes from the category.
//
// Kept apart from placement.ts on purpose: findDrop only hands wall pieces over to findWallDrop.
import { add, dist, dot, norm, openingEndpoints, pointInPolygon, polygonArea, scale, sub, v, wallInwardNormal, wallSegments } from './geometry'
import type { Doc, Item, Layout, Placement, Room, Vec2 } from './types'

/** Catalog categories that hang on a wall. */
export const WALL_DECOR = new Set(['wall-art', 'mirror', 'wall-shelf', 'wall-clock', 'curtains'])
export const isWallDecor = (category: string | undefined) => !!category && WALL_DECOR.has(category)

/** Pieces that hang beside or over the art on a wall (all but curtains, which go on windows). */
const FRAMES = new Set(['wall-art', 'mirror', 'wall-shelf', 'wall-clock'])

/** Height (m) of the centre of each piece above the floor: eye level for art and mirrors. */
const CENTRE: Record<string, number> = { 'wall-art': 1.5, mirror: 1.5, 'wall-shelf': 1.6, 'wall-clock': 1.85 }
/** Curtains hang from a rod above the window down to just above the floor. */
const CURTAIN_GAP = 0.03
/** How far a piece keeps from a door, a window or another frame on the same wall. */
const GAP = 0.05
/** Furniture a piece looks good hung over, with a little bonus when centred on it. */
const OVER = new Set(['sofa', 'bed', 'desk', 'sideboard', 'dresser', 'console-table', 'tv-stand'])

/** Height of the bottom edge of a wall piece above the floor. */
export function mountBottom(item: Pick<Item, 'category'> & { height?: number }): number {
  if (item.category === 'curtains') return CURTAIN_GAP
  return Math.max(0.3, (CENTRE[item.category] ?? 1.5) - (item.height ?? 0.6) / 2)
}

type WallItem = Pick<Item, 'category' | 'width' | 'depth'> & { height?: number }

export interface WallSpot {
  x: number
  y: number
  rotation: number
  roomId: string
  roomName: string
  wall: number
}

/** Rotation that turns a piece's front (local +y) toward `d` (same as placement.facing). */
const facing = (d: Vec2) => Math.atan2(-d.x, d.y)

type Interval = [number, number]

/** Spans along a wall face (meters from its start) that a piece may not cover. */
function blockedSpans(doc: Doc, layout: Layout | undefined, room: Room, wall: number, item: WallItem, exclude?: string): { doors: Interval[]; windows: Interval[]; frames: Interval[] } {
  const [a, b] = wallSegments(room)[wall]
  const d = norm(sub(b, a))
  const n = wallInwardNormal(room, wall)
  const reach = room.wallThickness + 0.3
  const doors: Interval[] = []
  const windows: Interval[] = []
  // Openings on this wall, or on the other face of it (a door stored on the neighbouring room).
  for (const r of doc.rooms)
    for (const o of r.openings) {
      const [p, q] = openingEndpoints(r, o)
      const off = (pt: Vec2) => dot(sub(pt, a), n)
      if (Math.abs(off(p)) > reach || Math.abs(off(q)) > reach) continue
      const t0 = dot(sub(p, a), d)
      const t1 = dot(sub(q, a), d)
      ;(o.kind === 'door' ? doors : windows).push([Math.min(t0, t1) - GAP, Math.max(t0, t1) + GAP])
    }
  const frames: Interval[] = []
  if (layout) {
    const byId = new Map(doc.items.map((i) => [i.id, i]))
    for (const p of layout.placements) {
      if (p.id === exclude) continue
      const it = byId.get(p.itemId)
      if (!it || !isWallDecor(it.category)) continue
      // Curtains and frames share a wall without minding each other (curtains sit on windows, frames never do).
      if ((it.category === 'curtains') !== (item.category === 'curtains')) continue
      const c = v(p.x, p.y)
      if (Math.abs(dot(sub(c, a), n) - it.depth / 2) > 0.06) continue
      const t = dot(sub(c, a), d)
      if (t < -0.01 || t > dist(a, b) + 0.01) continue
      frames.push([t - it.width / 2 - GAP, t + it.width / 2 + GAP])
    }
  }
  return { doors, windows, frames }
}

const overlaps = (lo: number, hi: number, s: Interval[]) => s.some(([x, y]) => lo < y && hi > x)

function spotAt(room: Room, wall: number, t: number, item: WallItem): WallSpot {
  const [a, b] = wallSegments(room)[wall]
  const d = norm(sub(b, a))
  const n = wallInwardNormal(room, wall)
  const c = add(add(a, scale(d, t)), scale(n, item.depth / 2 + 0.004))
  return { x: c.x, y: c.y, rotation: (facing(n) + 2 * Math.PI) % (2 * Math.PI), roomId: room.id, roomName: room.name, wall }
}

/**
 * The legal spot on wall `wall` of `room` closest to `t` (meters along the wall), or null when the
 * wall has none. Curtains centre on the window nearest `t`; everything else keeps off doors,
 * windows and other frames.
 */
function bestOnWall(doc: Doc, layout: Layout | undefined, room: Room, wall: number, t: number, item: WallItem, exclude?: string): { spot: WallSpot; t: number } | null {
  const [a, b] = wallSegments(room)[wall]
  const L = dist(a, b)
  const hw = item.width / 2
  if (L < item.width + 0.04) return null
  const { doors, windows, frames } = blockedSpans(doc, layout, room, wall, item, exclude)
  const lo = hw + 0.02
  const hi = L - hw - 0.02
  if (item.category === 'curtains') {
    // Only windows on this very face (not the neighbour's), and not one that already has curtains.
    const own = room.openings.filter((o) => o.kind === 'window' && o.wall === wall)
    let best: { spot: WallSpot; t: number } | null = null
    for (const o of own) {
      const tc = Math.max(lo, Math.min(hi, o.offset + o.width / 2))
      if (overlaps(tc - hw + GAP, tc + hw - GAP, doors) || overlaps(tc - 0.01, tc + 0.01, frames)) continue
      if (!best || Math.abs(tc - t) < Math.abs(best.t - t)) best = { spot: spotAt(room, wall, tc, item), t: tc }
    }
    return best
  }
  const blocked = [...doors, ...windows, ...frames]
  const ok = (x: number) => x >= lo - 1e-9 && x <= hi + 1e-9 && !overlaps(x - hw, x + hw, blocked)
  const cands = [Math.max(lo, Math.min(hi, t)), ...blocked.flatMap(([x, y]) => [x - hw - 1e-6, y + hw + 1e-6]), lo, hi]
  let best: number | undefined
  for (const c of cands) if (ok(c) && (best === undefined || Math.abs(c - t) < Math.abs(best - t))) best = c
  return best === undefined ? null : { spot: spotAt(room, wall, best, item), t: best }
}

/**
 * Snap a wall piece dragged to `pt` onto the nearest wall face it fits on: it slides along that
 * wall, faces into the room and keeps off doors (and, unless it's curtains, windows). Curtains snap
 * to the nearest window. Null when no wall nearby takes it.
 */
export function snapToWall(doc: Doc, layout: Layout | undefined, item: WallItem, pt: Vec2, exclude?: string): WallSpot | null {
  const inside = doc.rooms.filter((r) => pointInPolygon(pt, r.polygon))
  const rooms = inside.length ? inside : doc.rooms
  let best: { spot: WallSpot; d: number } | null = null
  for (const room of rooms)
    wallSegments(room).forEach(([a, b], wall) => {
      const d = norm(sub(b, a))
      const t = dot(sub(pt, a), d)
      const r = bestOnWall(doc, layout, room, wall, t, item, exclude)
      if (!r) return
      // Distance from the pointer to the spot on the wall line: the nearest wall wins, then the nearest spot.
      const onWall = add(a, scale(d, r.t))
      const gap = dist(pt, onWall)
      if (!best || gap < best.d) best = { spot: r.spot, d: gap }
    })
  return best ? (best as { spot: WallSpot }).spot : null
}

/** Whether a placed wall piece hangs where snapToWall would leave it (on a wall, off doors). */
export function onWallOk(doc: Doc, layout: Layout, item: WallItem, p: Pick<Placement, 'x' | 'y' | 'rotation' | 'id'>): boolean {
  const s = snapToWall(doc, layout, item, v(p.x, p.y), p.id)
  return !!s && Math.hypot(s.x - p.x, s.y - p.y) < 0.02
}

type DropOk = { ok: true; spot: { x: number; y: number; rotation: number; roomId?: string; roomName?: string } } | { ok: false; reason: string }

const LIVING = /living|lounge|family|sitting|den|studio|great/i
const BEDROOM = /bed|sleep|guest|master|nursery|kid/i
const UTILITY = /bath|wc|toilet|closet|laundry|storage|utility|pantry|hall|entry|corridor/i

/**
 * Where a wall piece goes when bought: art and mirrors over the sofa or bed in the living room or a
 * bedroom, curtains on a bedroom window first. Never behind a wardrobe or tall bookcase.
 */
export function findWallDrop(doc: Doc, layout: Layout, item: WallItem & { name?: string }, opts: { near?: Vec2; nextTo?: string } = {}): DropOk {
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  const solids = layout.placements.flatMap((p) => {
    const it = byId.get(p.itemId)
    return it && !p.on && !it.floorLevel && !isWallDecor(it.category) ? [{ p, it }] : []
  })
  const target = opts.nextTo ? layout.placements.find((p) => p.id === opts.nextTo) : undefined
  const bottom = mountBottom(item)
  const curtains = item.category === 'curtains'
  let best: { score: number; spot: WallSpot } | undefined
  for (const room of doc.rooms) {
    if (polygonArea(room.polygon) < 2) continue
    let roomScore = LIVING.test(room.name) ? 20 : BEDROOM.test(room.name) ? (curtains ? 30 : 16) : 0
    if (UTILITY.test(room.name)) roomScore -= item.category === 'mirror' ? 0 : 25
    wallSegments(room).forEach(([a, b], wall) => {
      const L = dist(a, b)
      const d = norm(sub(b, a))
      const n = wallInwardNormal(room, wall)
      const ts = curtains ? room.openings.filter((o) => o.kind === 'window' && o.wall === wall).map((o) => o.offset + o.width / 2) : Array.from({ length: Math.max(1, Math.floor(L / 0.2)) }, (_, i) => (i + 0.5) * (L / Math.max(1, Math.floor(L / 0.2))))
      for (const t0 of ts) {
        const r = bestOnWall(doc, layout, room, wall, t0, item)
        if (!r || Math.abs(r.t - t0) > 0.15) continue
        const at = add(a, scale(d, r.t))
        let score = roomScore
        for (const { p, it } of solids) {
          const c = v(p.x, p.y)
          const out = dot(sub(c, at), n)
          const along = dot(sub(c, at), d)
          const halfSpan = Math.max(it.width, it.depth) / 2
          if (out < 0 || out > Math.max(it.width, it.depth) / 2 + 0.25 || Math.abs(along) > halfSpan + item.width / 2) continue
          // Hidden behind something tall: no.
          if (it.height > bottom + 0.1 && !curtains) score -= 80
          else if (it.height > 1.0 && curtains) score -= 30
          else if (OVER.has(it.category) && FRAMES.has(item.category)) score += 30 - 20 * Math.min(1, Math.abs(along) / Math.max(0.1, halfSpan))
        }
        if (target) score -= 3 * dist(v(target.x, target.y), r.spot)
        if (opts.near) score -= 0.5 * dist(opts.near, r.spot)
        // A little preference for the middle of a wall over a corner.
        score -= 2 * Math.abs(r.t - L / 2) / Math.max(L, 1)
        if (!best || score > best.score) best = { score, spot: r.spot }
      }
    })
  }
  if (!best)
    return {
      ok: false,
      reason: curtains ? 'Curtains hang on a window. Every window already has a pair, or there are none.' : 'No free wall for that. Clear some wall space first.',
    }
  const s = best.spot
  return { ok: true, spot: { x: s.x, y: s.y, rotation: s.rotation, roomId: s.roomId, roomName: s.roomName } }
}
