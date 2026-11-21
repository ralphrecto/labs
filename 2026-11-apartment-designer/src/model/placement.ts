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
import { chairBackNeed, CHAIR_REACH, DRAWER_PIECES, MIN_DRAWER_FRONT, polygonGap, sideCompanion, TABLE_PULL_OUT, walkingPaths } from './checks'

export { chairBackNeed, MIN_DRAWER_FRONT, polygonGap, TABLE_PULL_OUT }

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
export const VIEW_OK = new Set(['coffee-table', 'ottoman', 'rug', 'tv-stand', 'side-table', 'floor-lamp', 'plant', 'sofa'])
/** Pieces that only go in a bedroom when the home has one (never next to the sofa). */
export const BEDROOM_ONLY = new Set(['bed', 'nightstand', 'wardrobe', 'dresser'])
/** Living-room pieces that never belong in a kitchen when the home has a living room. */
const LIVING_ONLY = new Set(['sofa', 'tv-stand', 'coffee-table', 'rug', 'ottoman'])
/** Pieces that only make sense pulled up to a partner: they never land far from it. */
const PARTNERS: Record<string, string[]> = { 'dining-chair': ['dining-table'], bench: ['dining-table'], 'office-chair': ['desk'], nightstand: ['bed'] }
/** A piece counts as "with" its partner when this close (edge to edge, m): a chair pulled up, a nightstand against the bed. */
export const PARTNER_REACH: Record<string, number> = { 'dining-chair': CHAIR_REACH, bench: CHAIR_REACH, 'office-chair': 0.35, nightstand: 0.15 }
/** Pieces whose front must stay open: someone gets in or sits down, or doors and drawers swing out. */
const FRONT_OPEN = new Set(['bed', 'wardrobe', 'dresser', 'desk'])
/** Wall pieces that never stand out in the open room: with no free wall, they get a friendly refusal. */
const WALL_ONLY = new Set(['desk', 'bookshelf', 'sideboard', 'wardrobe', 'dresser', 'console-table'])
/** A dining table this long (m) seats four or more. */
const FOUR_SEATER = 1.2
/** Pieces that may stand in a dining table's pull-out space. */
const AT_TABLE = new Set(['dining-chair', 'bench'])
/** Score cost of a table for four with a long side to the wall: any spot off the wall wins. */
const AGAINST_WALL = 1000
/** Share of a room's floor that furniture may cover before it feels crammed (as the density goal). */
const FILL_MAX = 0.35
const FILL_MAX_BEDROOM = 0.38
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

const isLiving = (r: Room) => LIVING.test(r.name) && !UTILITY_ROOM.test(r.name) && polygonArea(r.polygon) >= MIN_ROOM_AREA
const isBedroom = (r: Room) => BEDROOM.test(r.name) && !UTILITY_ROOM.test(r.name) && polygonArea(r.polygon) >= MIN_ROOM_AREA

/** Room suitability for a category: >0 good, 0 neutral, -Infinity not allowed. */
export function roomAffinity(room: Room, category: string, allRooms: Room[]): number {
  // A bed, nightstand, wardrobe or dresser never lands in the living room when there's a bedroom for it.
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
  // The sofa, TV and rug don't spill into the kitchen when there's a living room for them.
  if (LIVING_ONLY.has(category) && DINING.test(room.name) && !LIVING.test(room.name) && allRooms.some(isLiving)) s -= 40
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
export function findDrop(doc: Doc, layout: Layout, item: DropItem, opts: { near?: Vec2; walkCheck?: boolean; nextTo?: string } = {}): DropResult {
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
  // The space in front of a bed, wardrobe, dresser or desk is where you get in, sit or open the
  // doors, and a double bed needs both sides to get in: nothing stands there.
  const openFronts = solids.flatMap((o) =>
    clearanceZones(o.item, o.p)
      .filter((z) => (z.side === 'front' && FRONT_OPEN.has(o.item.category)) || (o.item.category === 'bed' && o.item.width >= 1.2 && (z.side === 'left' || z.side === 'right')))
      .map((z) => ({ poly: z.poly, side: z.side, owner: o.item.category })),
  )

  // The space between a sofa and what it faces (the TV, or 2.5 m of room) stays open: nothing
  // solid but a coffee table or the like stands between a sofa and its TV (soft with no TV).
  const viewZones = solids
    .filter((o) => o.item.category === 'sofa')
    .map((o) => {
      const sf = rot(v(0, 1), o.p.rotation)
      const tv = solids
        .filter((t) => t.item.category === 'tv-stand' && dot(sf, norm(sub(v(t.p.x, t.p.y), v(o.p.x, o.p.y)))) > 0.7)
        .map((t) => dot(sf, sub(v(t.p.x, t.p.y), v(o.p.x, o.p.y))) - t.item.depth / 2)
      const reach = Math.max(0.1, (tv.length ? Math.min(...tv) : 3) - o.item.depth / 2)
      return { poly: orientedRect(v(o.p.x, o.p.y), o.item.width / 2, reach / 2, o.p.rotation, v(0, o.item.depth / 2 + reach / 2)), tv: tv.length > 0 }
    })
  // A dining table for four or more keeps its long sides free to pull the chairs out.
  const pullOuts = solids
    .filter((o) => o.item.category === 'dining-table' && Math.max(o.item.width, o.item.depth) >= FOUR_SEATER)
    .flatMap((o) => tablePullOuts(o.item, o.p))

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
  const frontIdeal = FRONT_OPEN.has(cat) ? item.clearance?.front ?? 0 : 0
  // Dressers and wardrobes must have 75 cm to open; their full ideal (90 cm) only scores better.
  const frontZone = DRAWER_PIECES.has(cat) ? Math.min(frontIdeal, MIN_DRAWER_FRONT) : frontIdeal
  const ownPullOuts = cat === 'dining-table' && Math.max(item.width, item.depth) >= FOUR_SEATER
  const affinity = new Map(rooms.map((r) => [r.id, roomAffinity(r, cat, rooms)]))
  const cands: Cand[] = []
  const used = new Set(others.map((o) => o.room?.id))
  const emptyRooms = new Set(used.size ? rooms.filter((r) => polygonArea(r.polygon) >= MIN_ROOM_AREA && !used.has(r.id)).map((r) => r.id) : [])

  const roomAt = (c: Vec2) => rooms.find((r) => pointInPolygon(c, r.polygon))

  /** Hard rules: inside one room, no overlaps, no door swings. Returns the room or null. */
  function legal(c: Vec2, rotation: number): { room: Room; body: Vec2[]; penalty: number } | null {
    let penalty = 0
    const room = roomAt(c)
    if (!room || affinity.get(room.id) === -Infinity) return null
    const body = orientedRect(c, hw, hd, rotation)
    if (!polygonInsideRoom(body, room.polygon)) return null
    if (solid) {
      for (const o of solids) if (convexOverlap(body, o.body, 0.002)) return null
      for (const k of keepOut) if (convexOverlap(body, k, 0.005)) return null
      // A bed, wardrobe or dresser needs its whole front clearance (to get in, or to open the doors
      // and drawers): inside the room, and clear of other pieces.
      if (frontZone) {
        const fz = orientedRect(c, hw, frontZone / 2, rotation, v(0, hd + frontZone / 2))
        if (!polygonInsideRoom(fz, room.polygon, 0.01)) return null
        for (const o of solids) if (convexOverlap(fz, o.body, 0.005) && !sideCompanion(cat, 'front', o.item.category)) return null
        // A desk's chair sits in front of it (centered or a little to one side): that spot can't
        // be between a sofa and its TV, or the chair has nowhere to go.
        if (cat === 'desk' && viewZones.some((z) => z.tv)) {
          const seat = (u: number) => orientedRect(c, 0.33, 0.35, rotation, v(u, hd + 0.35))
          if (![0, -0.25, 0.25].some((u) => !viewZones.some((z) => z.tv && convexOverlap(seat(u), z.poly, 0.005)))) return null
        }
      }
      for (const z of openFronts) if (convexOverlap(body, z.poly, 0.005) && !sideCompanion(z.owner, z.side, cat)) return null
      // Nothing solid between a sofa and its TV but a coffee table, ottoman, lamp or the like.
      if (!VIEW_OK.has(cat)) for (const z of viewZones) if (z.tv && convexOverlap(body, z.poly, 0.005)) return null
      if (!AT_TABLE.has(cat)) for (const z of pullOuts) if (convexOverlap(body, z, 0.005)) return null
      // A table for four comes off the wall: both long sides keep room to pull a chair out. Only
      // when no spot anywhere allows that may one long side go to the wall (a far worse score), so
      // a crowded living room still takes a table. Other pieces there only cost score.
      if (ownPullOuts) {
        const open = tablePullOuts(item as Item, { id: '', itemId: '', x: c.x, y: c.y, rotation }).filter((z) => polygonInsideRoom(z, room.polygon, 0.01)).length
        if (!open) return null
        if (open < 2) penalty = AGAINST_WALL
      }
    } else {
      for (const o of rugs) if (convexOverlap(body, o.body, 0.01)) return null
    }
    return { room, body, penalty }
  }

  /** Soft rules: blocked clearances, crowding, distance from the room's middle. */
  function softScore(c: Vec2, rotation: number, body: Vec2[], room: Room): number {
    let s = affinity.get(room.id) ?? 0
    // A plant is the easy way to give an empty room something.
    if (cat === 'plant' && emptyRooms.has(room.id)) s += 25
    if (!solid) return s
    const pl = { id: '', itemId: '', x: c.x, y: c.y, rotation }
    // A dining table needs room to pull chairs out, though it has no clearance of its own.
    // A dining chair at a table needs its pull-out measured from the table edge (chairBackNeed), as the checks do.
    const asItem =
      cat === 'dining-table'
        ? { ...(item as Item), clearance: { front: TABLE_PULL_OUT, back: TABLE_PULL_OUT, left: 0.5, right: 0.5 } }
        : cat === 'dining-chair' && item.clearance
          ? { ...(item as Item), clearance: { ...item.clearance, back: chairBackNeed({ item: item as Item, body }, diningTables) } }
          : (item as Item)
    // A dining chair with no room behind it to pull it out doesn't count as a seat: it goes elsewhere at the table if it can.
    const blockedCost = cat === 'dining-chair' ? 16 : 6
    for (const z of clearanceZones(asItem, pl)) {
      if (!polygonInsideRoom(z.poly, room.polygon, 0.01)) s -= blockedCost
      else if (solids.some((o) => convexOverlap(z.poly, o.body, 0.005) && !sideCompanion(item.category, z.side, o.item.category))) s -= blockedCost
    }
    if (!VIEW_OK.has(cat)) for (const z of viewZones) if (convexOverlap(body, z.poly, 0.005)) s -= 10
    // A room past the designers' third of the floor feels crammed (the density goal): a roomier room wins.
    const fill = (solids.filter((o) => o.room?.id === room.id).reduce((a, o) => a + polygonArea(o.body), 0) + 4 * hw * hd) / polygonArea(room.polygon)
    if (fill > (BEDROOM.test(room.name) ? FILL_MAX_BEDROOM : FILL_MAX)) s -= 20
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

  const diningTables = solids.filter((o) => o.item.category === 'dining-table')
  /** True when a dining chair here has no room behind it to pull out (a wall or a piece in the way). */
  function chairBackBlocked(c: Vec2, rotation: number, body: Vec2[], room: Room): boolean {
    const need = chairBackNeed({ item: item as Item, body }, diningTables)
    const z = orientedRect(c, hw, need / 2, rotation, v(0, -hd - need / 2))
    return !polygonInsideRoom(z, room.polygon, 0.01) || solids.some((o) => convexOverlap(z, o.body, 0.005))
  }

  function consider(c: Vec2, rotation: number, bonus: number, reject?: (c: Vec2, rotation: number, body: Vec2[], room: Room) => boolean) {
    const ok = legal(c, rotation)
    if (!ok) return
    if (reject?.(c, rotation, ok.body, ok.room)) return
    cands.push({ c, rotation: normAngle(rotation), room: ok.room, body: ok.body, score: bonus - ok.penalty + softScore(c, rotation, ok.body, ok.room) })
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
      const sides = (
        [
          [v(0, 1), thw, thd, thw >= thd],
          [v(0, -1), thw, thd, thw >= thd],
          [v(1, 0), thd, thw, thd > thw],
          [v(-1, 0), thd, thw, thd > thw],
        ] as [Vec2, number, number, boolean][]
      ).map(([n, along, off, long]) => {
        // Free floor off this side of the table (m, up to the full pull-out), ignoring the chairs already at it.
        let free = 0
        for (let d = 0.05; d <= TABLE_PULL_OUT + 1e-6; d += 0.05) {
          const strip = orientedRect(v(t.p.x, t.p.y), Math.abs(n.x) ? d / 2 : along, Math.abs(n.x) ? along : d / 2, t.p.rotation, scale(n, off + d / 2))
          const room = t.room
          if (!room || !polygonInsideRoom(strip, room.polygon, 0.01)) break
          if (solids.some((o) => o !== t && !AT_TABLE.has(o.item.category) && convexOverlap(strip, o.body, 0.005))) break
          free = d
        }
        return { n, along, off, long, free }
      })
      const anyFree = sides.some((sd) => sd.free >= TABLE_PULL_OUT - 1e-6)
      for (const { n, along, off, long, free } of sides) {
        const perp = v(-n.y, n.x)
        for (const u of [0, -along / 2, along / 2, -along / 3, along / 3, -(along - hw), along - hw]) {
          if (Math.abs(u) + hw > along + 0.1) continue
          const l = add(scale(n, off + 0.05 + hd), scale(perp, u))
          const nw = rot(n, t.p.rotation)
          // The side with the most free floor wins; a chair with no room to pull out is refused
          // outright while another side of the table is open.
          consider(local(t, l), facing(scale(nw, -1)), 25 + (long ? 2 : 0) - Math.abs(u) + free * 8, cat === 'dining-chair' && anyFree ? chairBackBlocked : undefined)
        }
      }
    }
  }
  if (cat === 'office-chair')
    for (const d of ofCat('desk'))
      for (const gap of [0.03, 0.12, 0.25])
        for (const u of [0, -0.25, 0.25]) {
          if (Math.abs(u) + hw > d.item.width / 2 + 0.1) continue
          consider(local(d, v(u, d.item.depth / 2 + gap + hd)), d.p.rotation + Math.PI, 25 - gap * 4 - Math.abs(u) * 2)
        }
  if (cat === 'coffee-table' || cat === 'ottoman')
    for (const s of ofCat('sofa')) for (const gap of [0.46, 0.48, 0.5]) consider(local(s, v(0, s.item.depth / 2 + gap + hd)), s.p.rotation, cat === 'coffee-table' ? 25 : 12)
  if (cat === 'rug')
    for (const s of ofCat('sofa')) consider(local(s, v(0, s.item.depth / 2 - 0.25 + hd)), s.p.rotation, 20)
  if (cat === 'floor-lamp' || cat === 'side-table') {
    // Seats without a lamp or side table yet come first (the armchair's reading corner before a second lamp by the sofa).
    const lit = (s: Other) => others.some((o) => o.id !== s.id && ['floor-lamp', 'side-table', 'bookshelf'].includes(o.item.category) && polygonGap(s.body, o.body) <= 0.5)
    for (const s of ofCat('sofa', 'armchair')) {
      const bonus = lit(s) ? 6 : s.item.category === 'armchair' ? 22 : 18
      for (const sd of [-1, 1])
        for (const [gap, back] of [
          [0.05, 0.05],
          [0.2, 0.05],
          [0.05, s.item.depth / 2 - hd],
        ])
          consider(local(s, v(sd * (s.item.width / 2 + gap + hw), -s.item.depth / 2 + hd + back)), s.p.rotation, bonus - gap * 5)
      // Behind a corner of an armchair out in the room.
      if (s.item.category === 'armchair')
        for (const sd of [-1, 1]) consider(local(s, v(sd * (s.item.width / 2 - hw + 0.05), -s.item.depth / 2 - 0.05 - hd)), s.p.rotation, bonus - 2)
    }
  }
  if (cat === 'armchair')
    // At an end of the coffee table, turned toward the sofa, clear of the sofa's view of the TV.
    for (const t of ofCat('coffee-table')) {
      const s = ofCat('sofa').find((x) => polygonGap(x.body, t.body) < 0.8)
      if (!s) continue
      const lt = rot(sub(v(t.p.x, t.p.y), v(s.p.x, s.p.y)), -s.p.rotation)
      for (const sd of [-1, 1])
        for (const turn of [Math.PI / 2, (2 * Math.PI) / 3])
          for (const gap of [0.1, 0.25]) {
            const ext = Math.abs(Math.sin(turn)) * hd + Math.abs(Math.cos(turn)) * hw
            const x = sd * (Math.max(s.item.width / 2, Math.abs(lt.x) + t.item.width / 2) + gap + ext)
            // Facing in toward the coffee table (a quarter turn from the TV), or a little further, toward the sofa.
            const rotation = s.p.rotation + sd * turn
            consider(local(s, v(x, lt.y)), rotation, 24 - gap * 4)
          }
    }

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
  if (!WALL_BACKED.has(cat) || (!cands.length && !WALL_ONLY.has(cat))) {
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

  // A chair goes at its table or desk, or not at all: one stranded across the room is no use.
  const partners = ofCat(...(PARTNERS[cat] ?? []))
  if (partners.length) {
    const near = cands.filter((c) =>
      partners.some((o) => {
        if (polygonGap(c.body, o.body) > PARTNER_REACH[cat]) return false
        const l = rot(sub(c.c, v(o.p.x, o.p.y)), -o.p.rotation)
        const ohw = o.item.width / 2
        const ohd = o.item.depth / 2
        // A nightstand stands beside the bed, by the pillows.
        if (cat === 'nightstand') return Math.abs(l.x) >= ohw && l.y <= -ohd + hd * 2 + 0.2
        // A chair faces its table or desk from one of its sides (not off a corner)...
        const front = rot(v(0, 1), c.rotation)
        if (dot(front, norm(sub(v(o.p.x, o.p.y), c.c))) < 0.5) return false
        if (Math.abs(l.x) > ohw + 0.05 && Math.abs(l.y) > ohd + 0.05) return false
        // ...and a desk chair sits at the front of the desk.
        return cat !== 'office-chair' || l.y > ohd
      }),
    )
    if (!near.length) {
      const what = partners[0].item.category
      const name = item.name.toLowerCase()
      return {
        ok: false,
        reason:
          what === 'desk'
            ? `No room at the desk for the ${name}. Clear the space in front of the desk, or pull it off the wall.`
            : what === 'bed'
              ? `No room beside the bed for ${ofCat(cat).length ? 'another' : 'a'} ${name}. Make some room by the pillows, or move the bed out of the corner.`
              : `No room at the table for another ${name}. Pull the table away from the wall${cat === 'bench' ? '' : ', or try a bench'}.`,
      }
    }
    cands.length = 0
    cands.push(...near)
  }

  // Asked to go beside one piece (a lamp for the armchair's reading corner): keep only spots within reach of it, when there are any.
  const target = opts.nextTo ? others.find((o) => o.id === opts.nextTo) : undefined
  if (target) {
    const close = cands.filter((c) => polygonGap(c.body, target.body) <= NEXT_TO_REACH)
    if (close.length) cands.splice(0, cands.length, ...close)
  }

  if (!cands.length && BEDROOM_ONLY.has(cat) && rooms.some(isBedroom)) {
    const many = rooms.filter(isBedroom).length > 1
    const where = many ? "The bedrooms are full" : "The bedroom's full"
    return {
      ok: false,
      reason: cat === 'bed' ? `${where}. Make some room there, or remove the old bed to swap it.` : `${where}. Make some room there for the ${item.name.toLowerCase()}.`,
    }
  }
  if (!cands.length && WALL_ONLY.has(cat))
    return {
      ok: false,
      reason: `No free wall for ${/^[aeiou]/i.test(item.name) ? 'an' : 'a'} ${item.name.toLowerCase()} (${fmtSize(item, doc.apartment.units)}): it goes with its back to a wall and space in front. Clear a stretch of wall, or drag it in yourself.`,
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
      let blocked: string | undefined
      const keepsPaths = (c: Cand) => {
        const l2 = { ...layout, placements: [...layout.placements, { id: '__drop__', itemId: '__drop__', x: c.c.x, y: c.c.y, rotation: c.rotation }] }
        const after = walkingPaths(tmpDoc, l2)
        // Main routes (to living spaces) keep 90 cm; bathrooms and halls keep 60 cm.
        const worse = after.results.find((r) => {
          const b = ok0.get(r.roomId)
          if (!b?.reachable) return false
          const need = UTILITY_ROOM.test(r.roomName) ? 0.6 : 0.9
          // Worse: newly below the norm, or an already tight route squeezed further.
          return !r.reachable || (r.narrowest < need && (b.narrowest >= need || r.narrowest < b.narrowest - 0.05))
        })
        blocked ??= worse?.roomName
        return !worse
      }
      // First the best few spots; then, if each of them closes a walkway, spots further afield.
      let found: Cand | undefined
      for (const [limit, spacing, window] of [
        [10, 0.3, 12],
        [20, 0.6, Infinity],
      ]) {
        for (const c of cands) {
          if (found || tried.length >= limit) break
          if (tried.some((t) => dist(t.c, c.c) < spacing && t.room.id === c.room.id)) continue
          if (c.score < cands[0].score - window) break
          tried.push(c)
          if (keepsPaths(c)) found = c
        }
        if (found) break
      }
      if (found) {
        pick = found
        blocked = undefined
      }
      // A chair or nightstand only goes where it belongs; if every spot there closes a walkway, say so.
      if (blocked && partners.length)
        return { ok: false, reason: `${/^[aeiou]/i.test(item.name) ? 'An' : 'A'} ${item.name.toLowerCase()} by the ${partners[0].item.name.toLowerCase()} would block the way to the ${blocked}. Make some room around it first.` }
    }
  }
  return { ok: true, spot: { x: pick.c.x, y: pick.c.y, rotation: pick.rotation, roomId: pick.room.id, roomName: pick.room.name } }
}

/** "Next to" a piece, edge to edge (m): a lamp this close makes a reading corner. */
export const NEXT_TO_REACH = 0.5

/** The pull-out zones along a dining table's long sides. */
export function tablePullOuts(item: Pick<Item, 'width' | 'depth'>, p: Placement): Vec2[][] {
  const hw = item.width / 2
  const hd = item.depth / 2
  const c = v(p.x, p.y)
  const d = TABLE_PULL_OUT
  return hw >= hd
    ? [orientedRect(c, hw, d / 2, p.rotation, v(0, hd + d / 2)), orientedRect(c, hw, d / 2, p.rotation, v(0, -hd - d / 2))]
    : [orientedRect(c, d / 2, hd, p.rotation, v(hw + d / 2, 0)), orientedRect(c, d / 2, hd, p.rotation, v(-hw - d / 2, 0))]
}

/**
 * The armchair that still wants a reading corner (no lamp within reach),
 * or undefined. The nook goal's hint points here, and the lamp it suggests lands beside it.
 */
export function nookTarget(doc: Doc, layout: Layout): string | undefined {
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  const placed = layout.placements.flatMap((p) => {
    const it = byId.get(p.itemId)
    return it ? [{ p, it, body: orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation) }] : []
  })
  const lights = placed.filter((x) => x.it.category === 'floor-lamp')
  return placed.find((a) => a.it.category === 'armchair' && !lights.some((l) => polygonGap(a.body, l.body) <= NEXT_TO_REACH))?.p.id
}
