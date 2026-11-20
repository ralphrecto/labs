// Where a new piece of furniture lands when it isn't dropped somewhere specific.
//
// Candidate search: every spot is a real rotated footprint that must sit fully inside a room,
// clear of the other furniture, door swings and the landing in front of each door. Rooms are
// ranked by what the piece is (a bed goes to the bedroom, a sofa to the living room, nothing big
// goes in the bathroom), wall pieces back onto a wall facing into the room, and companions go to
// their partner (a nightstand to the head of the bed, a dining chair to the table, a coffee table
// in front of the sofa). When nothing fits, nothing is placed and the caller gets a reason.
import {
  add,
  bounds,
  centroid,
  clearanceZones,
  convexOverlap,
  dist,
  doorSwing,
  dot,
  norm,
  openingEndpoints,
  orientedRect,
  pointInPolygon,
  polygonArea,
  polygonInsideRoom,
  rot,
  scale,
  sub,
  v,
  wallInwardNormal,
  wallSegments,
} from './geometry'
import type { Doc, Item, Layout, Placement, Room, Vec2 } from './types'
import { sideCompanion, walkingPaths } from './checks'

/** Rooms that aren't for living in: furniture other than small decor stays out. */
export const UTILITY_ROOM = /bath|wc|toilet|closet|laundry|storage|utility|pantry|hall|entry|corridor/i
export const BEDROOM = /bed|sleep|guest|master|nursery|kid/i
export const LIVING = /living|lounge|family|sitting|den|studio|great/i
export const DINING = /dining|kitchen|eat/i
export const OFFICE = /office|study|work|library/i
/** Rooms smaller than this (m²) are closets for placement purposes. */
const MIN_ROOM_AREA = 3

/** Pieces that belong with their back against a wall. */
export const WALL_BACKED = new Set(['bed', 'wardrobe', 'dresser', 'desk', 'sofa', 'tv-stand', 'bookshelf', 'sideboard', 'console-table'])
/** Pieces that sit out in the room (no wall spots tried). */
const CENTER_PIECES = new Set(['dining-table', 'coffee-table', 'rug', 'ottoman'])
/** Pieces that belong between a sofa and the TV. */
const VIEW_OK = new Set(['coffee-table', 'ottoman', 'rug', 'tv-stand', 'side-table', 'floor-lamp', 'plant', 'sofa'])
/** Pieces that only go in a bedroom when the home has one (never next to the sofa). */
export const BEDROOM_ONLY = new Set(['bed', 'nightstand', 'wardrobe'])
/** Pieces whose front must stay open: someone gets in, or doors and drawers swing out. */
const FRONT_OPEN = new Set(['bed', 'wardrobe', 'dresser'])
/** Small things that may go in a bathroom or hall. */
const UTILITY_OK = new Set(['plant', 'box', 'rug'])

/** How much each kind of room suits a category. Unlisted rooms score 0. */
const AFFINITY: Record<string, [RegExp, number][]> = {
  bed: [[BEDROOM, 30]],
  nightstand: [[BEDROOM, 30]],
  wardrobe: [[BEDROOM, 30]],
  dresser: [[BEDROOM, 30]],
  sofa: [[LIVING, 30]],
  'tv-stand': [[LIVING, 30]],
  'coffee-table': [[LIVING, 30]],
  ottoman: [[LIVING, 30]],
  'console-table': [[LIVING, 20]],
  armchair: [[LIVING, 30], [BEDROOM, 10]],
  'side-table': [[LIVING, 25], [BEDROOM, 10]],
  'floor-lamp': [[LIVING, 25], [BEDROOM, 10]],
  'dining-table': [[DINING, 30], [LIVING, 15]],
  'dining-chair': [[DINING, 30], [LIVING, 15]],
  bench: [[DINING, 30], [LIVING, 15]],
  sideboard: [[DINING, 30], [LIVING, 15]],
  desk: [[OFFICE, 30], [LIVING, 10], [BEDROOM, 10]],
  'office-chair': [[OFFICE, 30], [LIVING, 10], [BEDROOM, 10]],
  bookshelf: [[OFFICE, 30], [LIVING, 20], [BEDROOM, 10]],
  plant: [[LIVING, 10]],
  rug: [[LIVING, 10]],
}

export interface DropSpot {
  x: number
  y: number
  rotation: number
  roomId?: string
  roomName?: string
}

export type DropResult = { ok: true; spot: DropSpot } | { ok: false; reason: string }

type DropItem = Pick<Item, 'width' | 'depth' | 'category' | 'name' | 'clearance' | 'floorLevel'>

interface Other {
  id: string
  item: Item
  p: Placement
  body: Vec2[]
  room?: Room
}

interface Cand {
  c: Vec2
  rotation: number
  room: Room
  score: number
  body: Vec2[]
}

/** Rotation (clockwise, plan) that makes an item's front face direction `d`. */
export const facing = (d: Vec2) => Math.atan2(-d.x, d.y)
const normAngle = (a: number) => ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)

const isBedroom = (r: Room) => BEDROOM.test(r.name) && !UTILITY_ROOM.test(r.name) && polygonArea(r.polygon) >= MIN_ROOM_AREA

/** Room suitability for a category: >0 good, 0 neutral, -Infinity not allowed. */
export function roomAffinity(room: Room, category: string, allRooms: Room[]): number {
  // A bed, nightstand or wardrobe never lands in the living room when there's a bedroom for it.
  if (BEDROOM_ONLY.has(category) && allRooms.some(isBedroom) && !isBedroom(room)) return -Infinity
  const small = polygonArea(room.polygon) < MIN_ROOM_AREA
  if ((small || UTILITY_ROOM.test(room.name)) && !UTILITY_OK.has(category)) {
    // Only when there is nowhere else (a plan made of one "Hall").
    const livable = allRooms.some((r) => polygonArea(r.polygon) >= MIN_ROOM_AREA && !UTILITY_ROOM.test(r.name))
    return livable ? -Infinity : -20
  }
  let s = 0
  for (const [re, w] of AFFINITY[category] ?? []) if (re.test(room.name)) s = Math.max(s, w)
  if (UTILITY_ROOM.test(room.name) || small) s -= 10
  return s
}

const fmtSize = (it: DropItem, units: Doc['apartment']['units']) =>
  units === 'metric'
    ? `${Math.round(it.width * 100)} × ${Math.round(it.depth * 100)} cm`
    : `${Math.round(it.width / 0.0254)} × ${Math.round(it.depth / 0.0254)} in`

/**
 * Find a legal spot for `item` in the active layout. `near`, when given, gives spots close to it a
 * bonus (the camera target, or the piece being copied).
 */
export function findDrop(doc: Doc, layout: Layout, item: DropItem, opts: { near?: Vec2; walkCheck?: boolean } = {}): DropResult {
  const rooms = doc.rooms
  if (!rooms.length) return { ok: true, spot: { x: 1, y: 1, rotation: 0 } }
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  const others: Other[] = layout.placements.flatMap((p) => {
    const it = byId.get(p.itemId)
    if (!it) return []
    const body = orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation)
    return [{ id: p.id, item: it, p, body, room: rooms.find((r) => pointInPolygon(v(p.x, p.y), r.polygon)) }]
  })
  const solids = others.filter((o) => !o.item.floorLevel)
  const rugs = others.filter((o) => o.item.floorLevel)
  const zonesOfOthers = solids.flatMap((o) => clearanceZones(o.item, o.p).map((z) => ({ poly: z.poly, side: z.side, owner: o.item.category })))

  // The space between a sofa and what it faces (the TV, or 2.5 m of room) stays open.
  const viewZones = solids
    .filter((o) => o.item.category === 'sofa')
    .map((o) => {
      const tv = solids.filter((t) => t.item.category === 'tv-stand').map((t) => dist(v(t.p.x, t.p.y), v(o.p.x, o.p.y)))
      const reach = Math.min(3, ...tv) - o.item.depth / 2
      return orientedRect(v(o.p.x, o.p.y), o.item.width / 2, reach / 2, o.p.rotation, v(0, o.item.depth / 2 + reach / 2))
    })

  // Door swings and a landing either side of every door stay clear.
  const keepOut: Vec2[][] = []
  for (const room of rooms)
    for (const o of room.openings) {
      if (o.kind !== 'door') continue
      const swing = doorSwing(room, o)
      if (swing) keepOut.push(swing)
      const [a, b] = openingEndpoints(room, o)
      const n = wallInwardNormal(room, o.wall)
      const inD = 0.5
      const outD = room.wallThickness + 0.5
      keepOut.push([add(a, scale(n, inD)), add(b, scale(n, inD)), add(b, scale(n, -outD)), add(a, scale(n, -outD))])
    }
  const windows = rooms.flatMap((r) =>
    r.openings.filter((o) => o.kind === 'window').map((o) => ({ room: r, wall: o.wall, ends: openingEndpoints(r, o) })),
  )

  const cat = item.category
  const hw = item.width / 2
  const hd = item.depth / 2
  const solid = !item.floorLevel
  const frontZone = FRONT_OPEN.has(cat) ? Math.min(item.clearance?.front ?? 0, 0.6) : 0
  const affinity = new Map(rooms.map((r) => [r.id, roomAffinity(r, cat, rooms)]))
  const cands: Cand[] = []

  const roomAt = (c: Vec2) => rooms.find((r) => pointInPolygon(c, r.polygon))

  /** Hard rules: inside one room, no overlaps, no door swings. Returns the room or null. */
  function legal(c: Vec2, rotation: number): { room: Room; body: Vec2[] } | null {
    const room = roomAt(c)
    if (!room || affinity.get(room.id) === -Infinity) return null
    const body = orientedRect(c, hw, hd, rotation)
    if (!polygonInsideRoom(body, room.polygon)) return null
    if (solid) {
      for (const o of solids) if (convexOverlap(body, o.body, 0.002)) return null
      for (const k of keepOut) if (convexOverlap(body, k, 0.005)) return null
      // A bed or wardrobe with furniture right up against its front reads as jammed in (and
      // looks overlapping in 3D): its front clearance must be clear of other pieces.
      if (frontZone) {
        const fz = orientedRect(c, hw, frontZone / 2, rotation, v(0, hd + frontZone / 2))
        for (const o of solids) if (convexOverlap(fz, o.body, 0.005) && !sideCompanion(cat, 'front', o.item.category)) return null
      }
    } else {
      for (const o of rugs) if (convexOverlap(body, o.body, 0.01)) return null
    }
    return { room, body }
  }

  /** Soft rules: blocked clearances, crowding, distance from the room's middle. */
  function softScore(c: Vec2, rotation: number, body: Vec2[], room: Room): number {
    let s = affinity.get(room.id) ?? 0
    if (!solid) return s
    const pl = { id: '', itemId: '', x: c.x, y: c.y, rotation }
    // A dining table needs room to pull chairs out, though it has no clearance of its own.
    const asItem = cat === 'dining-table' ? { ...(item as Item), clearance: { front: 0.75, back: 0.75, left: 0.5, right: 0.5 } } : (item as Item)
    for (const z of clearanceZones(asItem, pl)) {
      if (!polygonInsideRoom(z.poly, room.polygon, 0.01)) s -= 6
      else if (solids.some((o) => convexOverlap(z.poly, o.body, 0.005) && !sideCompanion(item.category, z.side, o.item.category))) s -= 6
    }
    if (!VIEW_OK.has(cat)) for (const z of viewZones) if (convexOverlap(body, z, 0.005)) s -= 10
    for (const z of zonesOfOthers) if (convexOverlap(body, z.poly, 0.005) && !sideCompanion(z.owner, z.side, item.category)) s -= 6
    // Some breathing room from other pieces in this room (centers vs. rough radii).
    const r0 = Math.hypot(hw, hd)
    const near = solids.filter((o) => o.room?.id === room.id)
    if (near.length) {
      const gap = Math.min(...near.map((o) => dist(v(o.p.x, o.p.y), c) - Math.hypot(o.item.width, o.item.depth) / 2 - r0))
      s += Math.max(-0.5, Math.min(gap, 0.8)) * 4
    } else s += 3
    if (opts.near) s -= dist(opts.near, c) * 1.5
    return s
  }

  function consider(c: Vec2, rotation: number, bonus: number) {
    const ok = legal(c, rotation)
    if (!ok) return
    cands.push({ c, rotation: normAngle(rotation), room: ok.room, body: ok.body, score: bonus + softScore(c, rotation, ok.body, ok.room) })
  }

  // ---- Companions: next to the piece they belong with --------------------
  const local = (o: Other, l: Vec2) => add(v(o.p.x, o.p.y), rot(l, o.p.rotation))
  const ofCat = (...cs: string[]) => others.filter((o) => cs.includes(o.item.category))
  if (cat === 'nightstand' || (cat === 'side-table' && !ofCat('sofa', 'armchair').length)) {
    for (const b of ofCat('bed'))
      for (const s of [-1, 1])
        for (const gap of [0.03, 0.1])
          consider(local(b, v(s * (b.item.width / 2 + gap + hw), -b.item.depth / 2 + hd)), b.p.rotation, 25)
  }
  if (cat === 'dining-chair' || cat === 'bench') {
    for (const t of ofCat('dining-table')) {
      const thw = t.item.width / 2
      const thd = t.item.depth / 2
      for (const [n, along, off, long] of [
        [v(0, 1), thw, thd, thw >= thd],
        [v(0, -1), thw, thd, thw >= thd],
        [v(1, 0), thd, thw, thd > thw],
        [v(-1, 0), thd, thw, thd > thw],
      ] as [Vec2, number, number, boolean][]) {
        const perp = v(-n.y, n.x)
        for (const u of [0, -along / 2, along / 2]) {
          if (Math.abs(u) + hw > along + 0.1) continue
          const l = add(scale(n, off + 0.05 + hd), scale(perp, u))
          const nw = rot(n, t.p.rotation)
          consider(local(t, l), facing(scale(nw, -1)), 25 + (long ? 2 : 0) - Math.abs(u))
        }
      }
    }
  }
  if (cat === 'office-chair')
    for (const d of ofCat('desk')) consider(local(d, v(0, d.item.depth / 2 + 0.03 + hd)), d.p.rotation + Math.PI, 25)
  if (cat === 'coffee-table' || cat === 'ottoman')
    for (const s of ofCat('sofa')) for (const gap of [0.42, 0.38, 0.48]) consider(local(s, v(0, s.item.depth / 2 + gap + hd)), s.p.rotation, cat === 'coffee-table' ? 25 : 12)
  if (cat === 'rug')
    for (const s of ofCat('sofa')) consider(local(s, v(0, s.item.depth / 2 - 0.25 + hd)), s.p.rotation, 20)
  if (cat === 'floor-lamp' || cat === 'side-table')
    for (const s of ofCat('sofa', 'armchair'))
      for (const sd of [-1, 1]) consider(local(s, v(sd * (s.item.width / 2 + 0.05 + hw), -s.item.depth / 2 + hd + 0.05)), s.p.rotation, 15)

  // ---- Along the walls -----------------------------------------------------
  const sofas = ofCat('sofa')
  const tvs = ofCat('tv-stand')
  if (!CENTER_PIECES.has(cat)) {
    const wallBacked = WALL_BACKED.has(cat)
    for (const room of rooms) {
      if (affinity.get(room.id) === -Infinity) continue
      const walls = wallSegments(room)
      walls.forEach(([a, b], wi) => {
        const L = dist(a, b)
        if (L < item.width) return
        const dir = norm(sub(b, a))
        const n = wallInwardNormal(room, wi)
        const rotation = facing(n)
        const step = 0.1
        for (let t = hw; t <= L - hw + 1e-6; t += step) {
          const c = add(add(a, scale(dir, t)), scale(n, hd + 0.005))
          let bonus = wallBacked ? 6 : 1
          bonus -= Math.abs(t - L / 2) * 0.3
          if (cat === 'bed') {
            // Headboard against solid wall, not under a window.
            for (const w of windows)
              if (w.room.id === room.id && w.wall === wi) {
                const t0 = dot(sub(w.ends[0], a), dir)
                const t1 = dot(sub(w.ends[1], a), dir)
                if (Math.min(t0, t1) < t + hw && Math.max(t0, t1) > t - hw) bonus -= 6
              }
          }
          if (cat === 'desk' && windows.some((w) => dist(c, scale(add(w.ends[0], w.ends[1]), 0.5)) - hd < 1.5)) bonus += 6
          if (cat === 'tv-stand')
            for (const s of sofas) {
              const sf = rot(v(0, 1), s.p.rotation)
              const to = sub(c, v(s.p.x, s.p.y))
              const d = Math.hypot(to.x, to.y)
              if (s.room?.id === room.id && d > 1.4 && d < 4.5 && dot(sf, norm(to)) > Math.cos((25 * Math.PI) / 180)) bonus += 15
            }
          if (cat === 'sofa')
            for (const tv of tvs) {
              const to = sub(v(tv.p.x, tv.p.y), c)
              const d = Math.hypot(to.x, to.y)
              if (tv.room?.id === room.id && d > 1.4 && d < 4.5 && dot(n, norm(to)) > Math.cos((25 * Math.PI) / 180)) bonus += 12
            }
          consider(c, rotation, bonus)
        }
      })
    }
  }

  // ---- Out in the room (a grid, square to the room's longest wall) ------------
  if (!WALL_BACKED.has(cat) || !cands.length) {
    for (const room of rooms) {
      if (affinity.get(room.id) === -Infinity) continue
      const walls = wallSegments(room)
      const longest = walls.reduce((m, w) => (dist(w[0], w[1]) > dist(m[0], m[1]) ? w : m), walls[0])
      const base = Math.atan2(longest[1].y - longest[0].y, longest[1].x - longest[0].x)
      const mid = centroid(room.polygon)
      const bb = bounds(room.polygon)
      const span = Math.max(bb.maxX - bb.minX, bb.maxY - bb.minY)
      const step = span > 8 ? 0.2 : 0.1
      for (let x = bb.minX + step / 2; x < bb.maxX; x += step)
        for (let y = bb.minY + step / 2; y < bb.maxY; y += step) {
          const c = v(x, y)
          if (!pointInPolygon(c, room.polygon)) continue
          const centerPull = CENTER_PIECES.has(cat) ? 0.8 : 0.3
          for (let k = 0; k < 4; k++) {
            const rotation = base + (k * Math.PI) / 2
            // Seats face the middle of the room; symmetric pieces only need two turns.
            const front = rot(v(0, 1), rotation)
            const toMid = norm(sub(mid, c))
            const faceBonus = cat === 'armchair' || cat === 'dining-chair' || cat === 'office-chair' ? dot(front, toMid) * 2 : 0
            if (!faceBonus && k >= 2) continue
            consider(c, rotation, faceBonus - dist(c, mid) * centerPull)
          }
        }
    }
  }

  if (!cands.length && BEDROOM_ONLY.has(cat) && rooms.some(isBedroom)) {
    const many = rooms.filter(isBedroom).length > 1
    const where = many ? "The bedrooms are full" : "The bedroom's full"
    return {
      ok: false,
      reason: cat === 'bed' ? `${where}. Make some room there, or remove the old bed to swap it.` : `${where}. Make some room there for the ${item.name.toLowerCase()}.`,
    }
  }
  if (!cands.length)
    return {
      ok: false,
      reason: `No free spot for ${/^[aeiou]/i.test(item.name) ? 'an' : 'a'} ${fmtSize(item, doc.apartment.units)} ${item.name.toLowerCase()}. Make some room or drag it in yourself.`,
    }

  cands.sort((a, b) => b.score - a.score)
  let pick = cands[0]

  // Don't cut off a walkway when a nearly-as-good spot keeps it open.
  if (solid && opts.walkCheck !== false) {
    const before = walkingPaths(doc, layout)
    if (before.entrance) {
      const ok0 = new Map(before.results.map((r) => [r.roomId, r]))
      const tmpItem = { ...(item as Item), id: '__drop__' }
      const tmpDoc = { ...doc, items: [...doc.items, tmpItem] }
      const tried: Cand[] = []
      for (const c of cands) {
        if (tried.length >= 6) break
        if (tried.some((t) => dist(t.c, c.c) < 0.3 && t.room.id === c.room.id)) continue
        if (c.score < cands[0].score - 12) break
        tried.push(c)
        const l2 = { ...layout, placements: [...layout.placements, { id: '__drop__', itemId: '__drop__', x: c.c.x, y: c.c.y, rotation: c.rotation }] }
        const after = walkingPaths(tmpDoc, l2)
        // Main routes (to living spaces) keep 90 cm; bathrooms and halls keep 60 cm.
        const worse = after.results.some((r) => {
          const b = ok0.get(r.roomId)
          if (!b?.reachable) return false
          const need = UTILITY_ROOM.test(r.roomName) ? 0.6 : 0.9
          return !r.reachable || (r.narrowest < need && b.narrowest >= need)
        })
        if (!worse) {
          pick = c
          break
        }
      }
    }
  }
  return { ok: true, spot: { x: pick.c.x, y: pick.c.y, rotation: pick.rotation, roomId: pick.room.id, roomName: pick.room.name } }
}
