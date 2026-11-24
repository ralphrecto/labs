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
  closestPointOnSegment,
} from './geometry'
import type { Doc, Item, Layout, Placement, Room, Vec2 } from './types'
import { findWallDrop, isWallDecor } from './wallDecor'
import { catalogEntry, itemFromCatalog } from './catalog'
import { chairBackNeed, CHAIR_REACH, doorKeepOut, DRAWER_PIECES, MIN_DRAWER_FRONT, polygonGap, sideCompanion, TABLE_PULL_OUT, walkingPaths } from './checks'

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
export const VIEW_OK = new Set(['coffee-table', 'ottoman', 'rug', 'tv-stand', 'side-table', 'floor-lamp', 'table-lamp', 'plant', 'sofa'])
/** Lights: a floor lamp or a table lamp lights a seat, a bed side or a desk. */
export const LAMPS = new Set(['floor-lamp', 'table-lamp', 'pendant'])
/** Pieces that stand on another piece of furniture (Placement.on) instead of the floor. */
export const HOSTED = new Set(['table-lamp', 'pendant'])
/** Furniture with a top a table lamp can stand on. */
export const SURFACES = new Set(['nightstand', 'side-table', 'desk', 'dresser', 'console-table', 'sideboard', 'coffee-table'])
/** How many table lamps each surface holds. */
export const SURFACE_SLOTS: Record<string, number> = { nightstand: 1, 'side-table': 1, desk: 1, dresser: 2, 'console-table': 2, sideboard: 2, 'dining-table': 1, 'coffee-table': 1 }
/** What each hosted piece goes on: a table lamp stands on a surface, a pendant hangs over a dining table. */
export const HOST_FOR: Record<string, Set<string>> = { 'table-lamp': SURFACES, pendant: new Set(['dining-table']) }
const hostCats = (category: string | undefined) => HOST_FOR[category ?? 'table-lamp'] ?? SURFACES
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
/** Room behind a desk chair to roll it back and stand up (m). */
const OFFICE_CHAIR_ROLL = 0.6
/** Small decor that tucks against a wall or into a corner, out of everyone's way. */
const DECOR = new Set(['plant', 'box'])
/** Pieces that may stand in a dining table's pull-out space. */
const AT_TABLE = new Set(['dining-chair', 'bench'])
/** Score cost of a table for four with a long side to the wall: any spot off the wall wins. */
const AGAINST_WALL = 1000
/** Share of a room's floor that furniture may cover before it feels crammed (as the density goal). */
const FILL_MAX = 0.35
const FILL_MAX_BEDROOM = 0.42
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
  'table-lamp': [[BEDROOM, 20], [OFFICE, 20], [LIVING, 15]],
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
  /** Placement id of the host (table lamps). */
  on?: string
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
  /** The score's parts: placement bonus less penalties, and the soft score before any `near` pull. */
  pre: number
  soft: number
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

function floorDrops(doc: Doc, layout: Layout, item: DropItem, opts: { walkCheck?: boolean; nextTo?: string }, nears: (Vec2 | undefined)[]): DropResult[] {
  const one = (r: DropResult) => nears.map(() => r)
  const rooms = doc.rooms
  if (!rooms.length) return one({ ok: true, spot: { x: 1, y: 1, rotation: 0 } })
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  const others: Other[] = layout.placements.flatMap((p) => {
    const it = byId.get(p.itemId)
    if (!it) return []
    const body = orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation)
    return [{ id: p.id, item: it, p, body, room: rooms.find((r) => pointInPolygon(v(p.x, p.y), r.polygon)) }]
  })
  const solids = others.filter((o) => !o.item.floorLevel && !o.p.on && !isWallDecor(o.item.category))
  const rugs = others.filter((o) => o.item.floorLevel)
  const zonesOfOthers = solids.flatMap((o) => clearanceZones(o.item, o.p).map((z) => ({ poly: z.poly, side: z.side, owner: o.item.category })))
  // The space in front of a bed, wardrobe, dresser or desk is where you get in, sit or open the
  // doors, and a double bed needs both sides to get in: nothing stands there.
  const openFronts = solids.flatMap((o) =>
    clearanceZones(o.item, o.p)
      .filter((z) => (z.side === 'front' && FRONT_OPEN.has(o.item.category)) || (o.item.category === 'bed' && o.item.width >= 1.2 && (z.side === 'left' || z.side === 'right')))
      .map((z) => ({ poly: z.poly, side: z.side, owner: o.item.category, room: o.room?.id })),
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
      return { poly: orientedRect(v(o.p.x, o.p.y), o.item.width / 2, reach / 2, o.p.rotation, v(0, o.item.depth / 2 + reach / 2)), tv: tv.length > 0, room: o.room?.id }
    })
  // A dining table for four or more keeps its long sides free to pull the chairs out.
  // (Round 17: these zones keep to their own room; a table by a wall doesn't block the room behind it.)
  const pullOuts = solids
    .filter((o) => o.item.category === 'dining-table' && Math.max(o.item.width, o.item.depth) >= FOUR_SEATER)
    .flatMap((o) => tablePullOuts(o.item, o.p).map((poly) => ({ poly, room: o.room?.id })))

  // Door swings and a landing either side of every door stay clear.
  const keepOut: Vec2[][] = doorKeepOut(rooms)
  const doorMids = rooms.flatMap((r) => r.openings.filter((o) => o.kind === 'door').map((o) => scale(add(...openingEndpoints(r, o)), 0.5)))
  const windows = rooms.flatMap((r) =>
    r.openings.filter((o) => o.kind === 'window').map((o) => ({ room: r, wall: o.wall, ends: openingEndpoints(r, o) })),
  )

  const cat = item.category
  const hw = item.width / 2
  const hd = item.depth / 2
  const solid = !item.floorLevel
  // A remix rebuilds around the same big pieces: where the source design had this kind of piece,
  // a spot scores a little lower, so the rebuild visibly differs (the other side of the bed, the
  // other end of the sofa) whenever a nearly-as-good spot exists.
  const remixSrc = layout.remix ? doc.layouts.find((l) => l.id === layout.remix!.from) : undefined
  // (Not pieces that need a wall: walls are scarce, and one taken early leaves the desk nowhere.)
  const srcSpots = remixSrc && !WALL_BACKED.has(cat)
    ? remixSrc.placements.flatMap((p) => (byId.get(p.itemId)?.category === cat && !layout.placements.some((q) => q.x === p.x && q.y === p.y) ? [v(p.x, p.y)] : []))
    : []
  /** Rooms named for eating (a kitchen or dining room), for a dining table to lean toward. */
  const eatRooms = rooms.filter((r) => DINING.test(r.name))

  // Floor kept for pieces still to come: other pieces may stand there only when nothing else fits.
  // A bedroom with no bed keeps a bed-sized spot (with room to get in) free, so a desk or dresser
  // placed first never leaves the bed nowhere to go; the floor in front of a sofa with no coffee
  // table yet keeps room for one (the knee room plus the table).
  const reserved: Vec2[][] = []
  /** Bedrooms with no bed yet: a piece placed there must still leave a spot for one. */
  const bedless = new Set<string>()
  let bedProbe: Item | undefined
  if (solid && cat !== 'bed') {
    for (const room of rooms.filter(isBedroom)) {
      if (others.some((o) => o.item.category === 'bed' && o.room?.id === room.id)) continue
      try {
        const bed = (bedProbe ??= itemFromCatalog(catalogEntry('bed'), '__bed_probe'))
        // (All the rooms stay in: a door hung in the next room still swings into this one.)
        const r = findDrop(doc, layout, bed, { walkCheck: false })
        if (!r.ok || r.spot.roomId !== room.id) continue
        bedless.add(room.id)
        const pl = { id: '', itemId: '', x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation }
        reserved.push(orientedRect(v(pl.x, pl.y), bed.width / 2, bed.depth / 2, pl.rotation), ...clearanceZones(bed, pl).filter((z) => z.side !== 'back').map((z) => z.poly))
      } catch {
        // No bed in the catalog: nothing to keep free.
      }
    }
  }
  if (solid && cat !== 'coffee-table' && cat !== 'ottoman')
    for (const s of solids) {
      if (s.item.category !== 'sofa') continue
      const sdp = s.item.depth / 2
      const served = solids.some((t) => {
        if (t.item.category !== 'coffee-table' && t.item.category !== 'ottoman') return false
        const l = rot(sub(v(t.p.x, t.p.y), v(s.p.x, s.p.y)), -s.p.rotation)
        return l.y > sdp && Math.abs(l.x) < s.item.width / 2 && polygonGap(s.body, t.body) < 1
      })
      if (!served) reserved.push(orientedRect(v(s.p.x, s.p.y), s.item.width / 2, COFFEE_ZONE / 2, s.p.rotation, v(0, sdp + COFFEE_ZONE / 2)))
    }
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
    // A dining chair's own pull-out room never runs through a wall (at a table end by the wall).
    if (cat === 'dining-chair' && chairBackThroughWall(c, rotation, body, room)) return null
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
      for (const z of openFronts) if (z.room === room.id && convexOverlap(body, z.poly, 0.005) && !sideCompanion(z.owner, z.side, cat)) return null
      // A floor lamp lights a seat from beside it, not from the knee room in front of it.
      if (cat === 'floor-lamp')
        for (const o of others)
          if ((o.item.category === 'sofa' || o.item.category === 'armchair') && convexOverlap(body, orientedRect(v(o.p.x, o.p.y), o.item.width / 2, 0.3, o.p.rotation, v(0, o.item.depth / 2 + 0.3)), 0.005)) penalty += 15
      // Nothing solid between a sofa and its TV but a coffee table, ottoman, lamp or the like.
      if (!VIEW_OK.has(cat)) for (const z of viewZones) if (z.tv && z.room === room.id && convexOverlap(body, z.poly, 0.005)) return null
      if (!AT_TABLE.has(cat)) for (const z of pullOuts) if (z.room === room.id && convexOverlap(body, z.poly, 0.005)) return null
      if (reserved.length) {
        const fz = frontZone ? orientedRect(c, hw, frontZone / 2, rotation, v(0, hd + frontZone / 2)) : null
        // A dining table's chairs pull out into its pull-out zones: those count too.
        const own = ownPullOuts ? tablePullOuts(item as Item, { id: '', itemId: '', x: c.x, y: c.y, rotation }) : []
        if (reserved.some((z) => convexOverlap(body, z, 0.005) || (fz && convexOverlap(fz, z, 0.005)) || own.some((q) => convexOverlap(q, z, 0.005)))) penalty += RESERVED_COST
      }
      // A table for four comes off the wall: both long sides keep room to pull a chair out. Only
      // when no spot anywhere allows that may one long side go to the wall (a far worse score), so
      // a crowded living room still takes a table. Other pieces there only cost score.
      if (ownPullOuts) {
        const inRoom = tablePullOuts(item as Item, { id: '', itemId: '', x: c.x, y: c.y, rotation }).filter((z) => polygonInsideRoom(z, room.polygon, 0.01))
        if (!inRoom.length) return null
        // (A side whose chairs would sit in the sofa's view of the TV is no more open than a wall.)
        const open = inRoom.filter((z) => !viewZones.some((w) => w.tv && convexOverlap(z, w.poly, 0.005))).length
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
          : cat === 'office-chair'
            ? // A desk chair rolls back to stand up: that space counts like a dining chair's pull-out.
              { ...(item as Item), clearance: { front: 0, back: OFFICE_CHAIR_ROLL, left: 0, right: 0 } }
            : (item as Item)
    // A chair with no room behind it to pull it out doesn't count as a seat: it goes elsewhere at the table or desk if it can.
    const blockedCost = cat === 'dining-chair' || cat === 'office-chair' ? 16 : 6
    for (const z of clearanceZones(asItem, pl)) {
      if (!polygonInsideRoom(z.poly, room.polygon, 0.01)) s -= blockedCost
      else if (solids.some((o) => convexOverlap(z.poly, o.body, 0.005) && !sideCompanion(item.category, z.side, o.item.category))) s -= blockedCost
    }
    if (!VIEW_OK.has(cat)) for (const z of viewZones) if (convexOverlap(body, z.poly, 0.005)) s -= 10
    // A desk chair keeps its distance from the other pieces in the room (the foot of the bed).
    if (cat === 'office-chair')
      for (const o of solids) {
        if (o.item.category === 'desk' || o.room?.id !== room.id) continue
        const g = polygonGap(body, o.body)
        if (g < OFFICE_CHAIR_ROLL) s -= (OFFICE_CHAIR_ROLL - g) * 20
      }
    // ...and rolls on bare floor, off the living room's rug (round 17).
    if (cat === 'office-chair') for (const o of rugs) if (convexOverlap(body, o.body, 0.01)) s -= 20
    // Small decor stays out of the way: off any table's chair pull-out floor and other pieces'
    // clearances, against a wall or in a corner rather than out in the middle of the floor.
    if (DECOR.has(cat)) {
      for (const t of diningTables) for (const z of tablePullOuts(t.item, t.p)) if (convexOverlap(body, z, 0.005)) s -= 15
      for (const z of zonesOfOthers) if (convexOverlap(body, z.poly, 0.005)) s -= 6
      // (A plant alone in an empty room just goes where it looks best.)
      const wallGaps = emptyRooms.has(room.id) ? [] : wallSegments(room)
        .map(([a, b]) => closestPointOnSegment(c, a, b).distance - Math.min(hw, hd))
        .sort((a, b) => a - b)
      s -= Math.min(wallGaps[0] ?? 0, 2) * 4
      // ...and not by a door, where it narrows the way in.
      if (wallGaps.length) for (const dm of doorMids) s -= Math.max(0, 1.2 - dist(c, dm)) * 10
    }
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
    // A dining table keeps out of the living corner: off the floor in front of a sofa (and its
    // view of the TV, above), toward the kitchen or dining side of the room.
    if (cat === 'dining-table') {
      for (const o of solids) {
        if (o.item.category !== 'sofa' || o.room?.id !== room.id) continue
        const sc = v(o.p.x, o.p.y)
        const away = dist(sc, c)
        s += Math.min(away, 4) * 1.5
        if (away < 3.5 && dot(rot(v(0, 1), o.p.rotation), norm(sub(c, sc))) > 0.5) s -= 8
      }
      const eat = eatRooms.filter((r) => r.id !== room.id)
      if (eat.length && !DINING.test(room.name)) s -= Math.min(...eat.map((r) => Math.min(dist(c, centroid(r.polygon)), 6))) * 0.8
    }
    if (srcSpots.some((q) => dist(q, c) < 0.6)) s -= 6
    return s
  }

  const diningTables = solids.filter((o) => o.item.category === 'dining-table')
  /** True when a dining chair here has no room behind it to pull out (a wall or a piece in the way). */
  function chairBackBlocked(c: Vec2, rotation: number, body: Vec2[], room: Room): boolean {
    const need = chairBackNeed({ item: item as Item, body }, diningTables)
    const z = orientedRect(c, hw, need / 2, rotation, v(0, -hd - need / 2))
    return !polygonInsideRoom(z, room.polygon, 0.01) || solids.some((o) => convexOverlap(z, o.body, 0.005))
  }
  /** A dining chair's own pull-out room runs through a wall: never a spot, even when every side is tight. */
  function chairBackThroughWall(c: Vec2, rotation: number, body: Vec2[], room: Room): boolean {
    const need = chairBackNeed({ item: item as Item, body }, diningTables)
    const z = orientedRect(c, hw, need / 2, rotation, v(0, -hd - need / 2))
    return !polygonInsideRoom(z, room.polygon, 0.01)
  }

  function consider(c: Vec2, rotation: number, bonus: number, reject?: (c: Vec2, rotation: number, body: Vec2[], room: Room) => boolean) {
    const ok = legal(c, rotation)
    if (!ok) return
    if (reject?.(c, rotation, ok.body, ok.room)) return
    const pre = bonus - ok.penalty
    const soft = softScore(c, rotation, ok.body, ok.room)
    cands.push({ c, rotation: normAngle(rotation), room: ok.room, body: ok.body, pre, soft, score: pre + soft })
  }

  // ---- Companions: next to the piece they belong with --------------------
  const local = (o: Other, l: Vec2) => add(v(o.p.x, o.p.y), rot(l, o.p.rotation))
  const ofCat = (...cs: string[]) => others.filter((o) => cs.includes(o.item.category))
  const nextToCat = opts.nextTo ? others.find((o) => o.id === opts.nextTo)?.item.category : undefined
  if (cat === 'nightstand' || (cat === 'side-table' && (!ofCat('sofa', 'armchair').length || nextToCat === 'bed'))) {
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
        // The seats already pulled up to this side (table-local positions).
        const seats = others.flatMap((o) => {
          if (!AT_TABLE.has(o.item.category) || polygonGap(o.body, t.body) > CHAIR_REACH + 0.05) return []
          const l = rot(sub(v(o.p.x, o.p.y), v(t.p.x, t.p.y)), -t.p.rotation)
          return dot(l, n) > off - 0.05 && Math.abs(dot(l, v(-n.y, n.x))) < along + 0.1 ? [l] : []
        })
        return { n, along, off, long, free, seats }
      })
      const anyFree = sides.some((sd) => sd.free >= TABLE_PULL_OUT - 1e-6)
      // A second chair goes across from the first (two people face each other over the table), not
      // beside it on the same side: an empty side scores, straight across from a chair scores most.
      const anySeated = sides.some((sd) => sd.seats.length > 0)
      for (const { n, along, off, long, free, seats } of sides) {
        const perpN = v(-n.y, n.x)
        const across = sides.filter((sd) => sd.seats.length && dot(sd.n, n) < -0.5).flatMap((sd) => sd.seats.map((l) => dot(l, perpN)))
        const company = (u: number) =>
          !anySeated ? 0 : seats.length ? -4 * seats.length : across.length ? 10 - 4 * Math.min(1, Math.min(...across.map((a) => Math.abs(a - u)))) : 6
        const perp = v(-n.y, n.x)
        for (const u of [0, -along / 2, along / 2, -along / 3, along / 3, -(along - hw), along - hw, ...across]) {
          if (Math.abs(u) + hw > along + 0.1) continue
          const l = add(scale(n, off + 0.05 + hd), scale(perp, u))
          const nw = rot(n, t.p.rotation)
          // The side with the most free floor wins; a chair with no room to pull out is refused
          // outright while another side of the table is open.
          consider(local(t, l), facing(scale(nw, -1)), 25 + (long ? 2 : 0) - Math.abs(u) + free * 8 + company(u), cat === 'dining-chair' ? (anyFree ? chairBackBlocked : chairBackThroughWall) : undefined)
        }
      }
    }
  }
  if (cat === 'office-chair')
    for (const d of ofCat('desk'))
      for (const gap of [0.03, 0.12, 0.25])
        for (const u of [0, -0.12, 0.12, -0.25, 0.25]) {
          if (Math.abs(u) + hw > d.item.width / 2 + 0.1) continue
          // Centred on the desk's front edge; off to one side only when that keeps it clear.
          consider(local(d, v(u, d.item.depth / 2 + gap + hd)), d.p.rotation + Math.PI, 25 - gap * 4 - Math.abs(u) * 12)
        }
  if (cat === 'coffee-table' || cat === 'ottoman')
    for (const s of ofCat('sofa')) for (const gap of [0.46, 0.48, 0.5]) consider(local(s, v(0, s.item.depth / 2 + gap + hd)), s.p.rotation, cat === 'coffee-table' ? 25 : 12)
  // A sofa after its coffee table: behind the table, the same ~46 cm off, along either side.
  const loneTables = cat === 'sofa' ? ofCat('coffee-table').filter((t) => !ofCat('sofa').some((x) => polygonGap(x.body, t.body) < 0.8)) : []
  for (const t of loneTables)
    for (const k of [0, 1, 2, 3]) {
      const rotation = t.p.rotation + (k * Math.PI) / 2
      const ext = k % 2 ? t.item.width / 2 : t.item.depth / 2
      const fwd = rot(v(0, 1), rotation)
      consider(sub(v(t.p.x, t.p.y), scale(fwd, ext + SOFA_TABLE_GAP + hd)), rotation, (k % 2 ? 16 : 20))
    }
  if (cat === 'rug')
    // Under the front of the sofa and its coffee table (round 17: slid along the sofa, or turned,
    // when the room's wall is close at one end, so the seating stays grouped on it).
    for (const s of ofCat('sofa')) {
      const table = ofCat('coffee-table', 'ottoman').find((t) => {
        const l = rot(sub(v(t.p.x, t.p.y), v(s.p.x, s.p.y)), -s.p.rotation)
        return l.y > s.item.depth / 2 && Math.abs(l.x) < s.item.width / 2 && polygonGap(s.body, t.body) < 1
      })
      const tl = table ? rot(sub(v(table.p.x, table.p.y), v(s.p.x, s.p.y)), -s.p.rotation) : undefined
      for (const [ext, turn, bonus] of [
        [hd, 0, 20],
        [hw, Math.PI / 2, 14],
      ] as const) {
        const ys = [s.item.depth / 2 - 0.25 + ext]
        // Centred between the sofa's front and the table's far edge, when that still reaches under the sofa.
        if (tl && table) {
          const mid = (s.item.depth / 2 + tl.y + table.item.depth / 2) / 2
          if (mid - ext < s.item.depth / 2 - 0.1) ys.push(mid)
        }
        for (const y of ys)
          for (const u of [0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0, 1.2, -1.2])
            consider(local(s, v(u, y)), s.p.rotation + turn, bonus - Math.abs(u) * 3 + (y === ys[0] ? 0 : 1))
      }
    }
  if (cat === 'floor-lamp' || cat === 'side-table') {
    // Seats without a lamp or side table yet come first (the armchair's reading corner before a second lamp by the sofa).
    const lit = (s: Other) => others.some((o) => o.id !== s.id && (LAMPS.has(o.item.category) || ['side-table', 'bookshelf'].includes(o.item.category)) && polygonGap(s.body, o.body) <= 0.5)
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
        const along: number[] = []
        for (let t = hw; t <= L - hw + 1e-6; t += step) along.push(t)
        // r13-b sofa/coffee pairing: also try the spot square behind each lone coffee table.
        if (cat === 'sofa') for (const t of loneTables) along.push(...pairedAlong(a, dir, L, hw, v(t.p.x, t.p.y)))
        for (const t of along) {
          const c = add(add(a, scale(dir, t)), scale(n, hd + 0.005))
          let bonus = wallBacked ? 6 : 1
          bonus -= Math.abs(t - L / 2) * 0.3
          if (cat === 'bed' || cat === 'tv-stand') {
            // Headboard against solid wall, not under a window. A TV under a window fights the
            // glare (checkLayout flags it): a solid wall the sofa faces beats it by far.
            for (const w of windows)
              if (w.room.id === room.id && w.wall === wi) {
                const t0 = dot(sub(w.ends[0], a), dir)
                const t1 = dot(sub(w.ends[1], a), dir)
                if (Math.min(t0, t1) < t + hw - 0.05 && Math.max(t0, t1) > t - hw + 0.05) bonus -= cat === 'bed' ? 6 : 20
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
            for (const t of loneTables) {
              // Backed onto this wall with the coffee table just in front: the pair, kept together.
              const g = polygonGap(orientedRect(c, hw, hd, rotation), t.body)
              const across = Math.abs(dot(sub(v(t.p.x, t.p.y), c), dir))
              if (dot(sub(v(t.p.x, t.p.y), c), n) > 0) bonus += pairBonus(g, across)
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
      return one({
        ok: false,
        reason:
          what === 'desk'
            ? `No room at the desk for the ${name}. Clear the space in front of the desk, or pull it off the wall.`
            : what === 'bed'
              ? `No room beside the bed for ${ofCat(cat).length ? 'another' : 'a'} ${name}. Make some room by the pillows, or move the bed out of the corner.`
              : `No room at the table for another ${name}. Pull the table away from the wall${cat === 'bench' ? '' : ', or try a bench'}.`,
      })
    }
    cands.length = 0
    cands.push(...near)
  }

  // Asked to go beside one piece (a lamp for the armchair's reading corner): keep only spots within reach of it, when there are any.
  const target = opts.nextTo ? others.find((o) => o.id === opts.nextTo) : undefined
  if (target) {
    // A coffee table (or ottoman) for a sofa goes in front of it, past the knee room, or not at all.
    const inFront = (cat === 'coffee-table' || cat === 'ottoman') && target.item.category === 'sofa'
    const ahead = (c: Cand) => {
      const l = rot(sub(c.c, v(target.p.x, target.p.y)), -target.p.rotation)
      const gap = polygonGap(c.body, target.body)
      return l.y > target.item.depth / 2 && Math.abs(l.x) < target.item.width / 2 && gap >= COFFEE_GAP[0] - 0.01 && gap <= COFFEE_GAP[1] + 0.01
    }
    const close = cands.filter((c) => (inFront ? ahead(c) : polygonGap(c.body, target.body) <= NEXT_TO_REACH))
    if (close.length) cands.splice(0, cands.length, ...close)
    else if (inFront)
      return one({ ok: false, reason: `No room in front of the sofa for ${/^[aeiou]/i.test(item.name) ? 'an' : 'a'} ${item.name.toLowerCase()}. Clear the floor in front of it first.` })
  }

  if (!cands.length && BEDROOM_ONLY.has(cat) && rooms.some(isBedroom)) {
    const many = rooms.filter(isBedroom).length > 1
    const where = many ? "The bedrooms are full" : "The bedroom's full"
    return one({
      ok: false,
      reason:
        cat === 'bed'
          ? ofCat('bed').length
            ? `${where}. Make some room there, or remove the old bed to swap it.`
            : `${where}. Move a piece out of there to make room for the bed.`
          : `${where}. Make some room there for the ${item.name.toLowerCase()}.`,
    })
  }
  if (!cands.length && WALL_ONLY.has(cat))
    return one({
      ok: false,
      reason: `No free wall for ${/^[aeiou]/i.test(item.name) ? 'an' : 'a'} ${item.name.toLowerCase()}. Clear some wall, or drag it in.`,
    })
  if (!cands.length)
    return one({
      ok: false,
      reason: `No free spot for ${/^[aeiou]/i.test(item.name) ? 'an' : 'a'} ${fmtSize(item, doc.apartment.units)} ${item.name.toLowerCase()}. Make some room or drag it in yourself.`,
    })


  // Round 17: the search above is the same for every `near`; only the ranking (and the dry runs
  // that follow it, kept per spot) differ, so several targets cost little more than one.
  const all = cands
  const candKey = (c: Cand) => `${c.room.id}:${c.c.x}:${c.c.y}:${c.rotation}`
  const bedFits = new Map<string, boolean>()
  const pathChecks = new Map<string, { ok: boolean; worse?: string }>()
  let pathsBefore: ReturnType<typeof walkingPaths> | undefined
  return nears.map((near) => rank(near))

  function rank(near: Vec2 | undefined): DropResult {
    const cands = all.map((c) => ({ ...c, score: near ? c.pre + (c.soft - dist(near, c.c) * 1.5) : c.pre + c.soft }))
    cands.sort((a, b) => b.score - a.score)

    // In a bedroom with no bed yet, a spot only counts if a bed still fits after it (with a desk's
    // chair in front of the desk): the best few spots are dry-run, and ones that crowd the bed out go.
    if (bedless.size && bedProbe) {
      const bed = bedProbe
      const failed = new Set<Cand>()
      const tried: Cand[] = []
      for (const c of cands) {
        if (!bedless.has(c.room.id)) break
        const twin = tried.find((t) => t.room.id === c.room.id && dist(t.c, c.c) < 0.25)
        if (twin) {
          if (failed.has(twin)) failed.add(c)
          continue
        }
        if (tried.length >= 12) break
        tried.push(c)
        const seatAt = cat === 'desk' ? add(c.c, rot(v(0, hd + 0.3), c.rotation)) : undefined
        const extra = [{ ...(item as Item), id: '__drop__' }, ...(seatAt ? [{ ...(item as Item), id: '__seat__', category: 'office-chair', width: 0.55, depth: 0.55 }] : [])]
        const l2 = {
          ...layout,
          placements: [
            ...layout.placements,
            { id: '__drop__', itemId: '__drop__', x: c.c.x, y: c.c.y, rotation: c.rotation },
            ...(seatAt ? [{ id: '__seat__', itemId: '__seat__', x: seatAt.x, y: seatAt.y, rotation: c.rotation }] : []),
          ],
        }
        const bk = candKey(c)
        let fits = bedFits.get(bk)
        if (fits === undefined) bedFits.set(bk, (fits = findDrop({ ...doc, items: [...doc.items, ...extra] }, l2, bed, { walkCheck: false }).ok))
        if (fits) break
        failed.add(c)
      }
      const kept = cands.filter((c) => !failed.has(c))
      if (kept.length) cands.splice(0, cands.length, ...kept)
    }
    let pick = cands[0]

    // Don't cut off a walkway when a nearly-as-good spot keeps it open.
    if (solid && opts.walkCheck !== false) {
      const before = (pathsBefore ??= walkingPaths(doc, layout))
      if (before.entrance) {
        const ok0 = new Map(before.results.map((r) => [r.roomId, r]))
        const tmpItem = { ...(item as Item), id: '__drop__' }
        // A desk's chair will stand in front of it: the walkway has to pass the chair too.
        const seat = cat === 'desk' ? { ...(item as Item), id: '__seat__', category: 'office-chair', width: 0.55, depth: 0.55, clearance: { front: 0, back: 0, left: 0, right: 0 } } : undefined
        const tmpDoc = { ...doc, items: [...doc.items, tmpItem, ...(seat ? [seat] : [])] }
        const tried: Cand[] = []
        let blocked: string | undefined
        const keepsPaths = (c: Cand) => {
          const pk = candKey(c)
          const known = pathChecks.get(pk)
          if (known) {
            blocked ??= known.worse
            return known.ok
          }
          const seatAt = seat ? add(c.c, rot(v(0, hd + 0.3), c.rotation)) : undefined
          const l2 = {
            ...layout,
            placements: [
              ...layout.placements,
              { id: '__drop__', itemId: '__drop__', x: c.c.x, y: c.c.y, rotation: c.rotation },
              ...(seatAt ? [{ id: '__seat__', itemId: '__seat__', x: seatAt.x, y: seatAt.y, rotation: c.rotation }] : []),
            ],
          }
          const after = walkingPaths(tmpDoc, l2)
          // Main routes (to living spaces) keep 90 cm; bathrooms and halls keep 60 cm.
          const worse = after.results.find((r) => {
            const b = ok0.get(r.roomId)
            if (!b?.reachable) return false
            const need = UTILITY_ROOM.test(r.roomName) ? 0.6 : 0.9
            // Worse: newly below the norm, or an already tight route squeezed further.
            return !r.reachable || (r.narrowest < need && (b.narrowest >= need || r.narrowest < b.narrowest - 0.05))
          })
          pathChecks.set(pk, { ok: !worse, worse: worse?.roomName })
          blocked ??= worse?.roomName
          return !worse
        }
        // First the best few spots; then, if each of them closes a walkway, spots further afield.
        let found: Cand | undefined
        for (const [limit, spacing, window] of [
          [10, 0.3, 12],
          // A desk (with its chair) tries further: a spot by the door that boxes the chair in is a dead end.
          [seat ? 60 : 20, seat ? 0.3 : 0.6, Infinity],
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
        // A coffee table for a sofa (round 15): one that can only stand in front of it by narrowing a
        // walkway, or only far from it, isn't offered; Next moves on to another goal instead.
        const forSofa = (cat === 'coffee-table' || cat === 'ottoman') && !!opts.nextTo
        if (forSofa && (!found || found.score < cands[0].score - 12)) {
          const where = blocked ?? 'next room'
          return { ok: false, reason: `${/^[aeiou]/i.test(item.name) ? 'An' : 'A'} ${item.name.toLowerCase()} there would block the way to the ${where}. Make some room by the sofa first.` }
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
}

/**
 * Find a legal spot for `item` in the active layout. `near`, when given, gives spots close to it a
 * bonus (the camera target, or the piece being copied).
 */
export function findDrop(doc: Doc, layout: Layout, item: DropItem, opts: { near?: Vec2; walkCheck?: boolean; nextTo?: string } = {}): DropResult {
  if (HOSTED.has(item.category)) return findHostDrop(doc, layout, item, opts)
  // Wall decor hangs on a wall (wallDecor.ts) and plays no part in the floor search below.
  if (isWallDecor(item.category)) return findWallDrop(doc, layout, item, opts)
  return floorDrops(doc, layout, item, opts, [opts.near])[0]
}

/** findDrop for each of several `near` targets, from one search (round 17: the Next chip's move candidates). */
export function findDrops(doc: Doc, layout: Layout, item: DropItem, nears: Vec2[], opts: { walkCheck?: boolean; nextTo?: string } = {}): DropResult[] {
  if (HOSTED.has(item.category) || isWallDecor(item.category)) return nears.map((near) => findDrop(doc, layout, item, { ...opts, near }))
  return floorDrops(doc, layout, item, opts, nears)
}

/** Knee room plus a coffee table, in front of a sofa (m): kept free of other pieces until it has one. */
const COFFEE_ZONE = 1.1
/** A coffee table placed for a sofa sits this far (m, edge to edge) in front of it. */
const COFFEE_GAP = [0.45, 0.5] as const
/** Where a sofa placed after its coffee table goes: this far (m, edge to edge) behind the table. */
const SOFA_TABLE_GAP = 0.47
// ---- r13-b: sofa <-> coffee table pairing (centred within PAIR_ACROSS, PAIR_GAP edge to edge) ----
/** How far (m) a coffee table may sit off the sofa's centre line and still count as its pair. */
const PAIR_ACROSS = 0.15
/** The edge-to-edge gap (m) between a paired sofa and coffee table. */
const PAIR_GAP = [0.4, 0.55] as const
/** Bonus for a wall-backed sofa spot with a lone coffee table in front: only when truly paired. */
function pairBonus(gap: number, across: number): number {
  if (gap < PAIR_GAP[0] || gap > PAIR_GAP[1] || across > PAIR_ACROSS) return 0
  return 18 - Math.abs(gap - SOFA_TABLE_GAP) * 10 - across * 20
}
/** Positions along a wall (from `a`, direction `dir`, length `L`) square behind `pt`, when a piece `hw` wide fits there. */
function pairedAlong(a: Vec2, dir: Vec2, L: number, hw: number, pt: Vec2): number[] {
  const t = dot(sub(pt, a), dir)
  return t >= hw - 1e-6 && t <= L - hw + 1e-6 ? [t] : []
}
// ---- end r13-b pairing ----

/** Score cost of standing on floor kept for a bed or a coffee table: only when nothing else fits. */
const RESERVED_COST = 40

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
  const lights = placed.filter((x) => LAMPS.has(x.it.category))
  return placed.find((a) => a.it.category === 'armchair' && !lights.some((l) => polygonGap(a.body, l.body) <= NEXT_TO_REACH))?.p.id
}

// ---- Table lamps: pieces that stand on another piece's top ----

type LampLike = Pick<Item, 'width' | 'depth'> & { category?: string }
type HostLike = { item: Pick<Item, 'width' | 'depth' | 'category' | 'height'>; p: Pick<Placement, 'x' | 'y' | 'rotation'> }

/** Height (m) of a host's top, where a hosted piece stands. */
export const hostTop = (host: Pick<Item, 'height'>) => host.height

/** Whether `lamp` fits on top of `host` with a little to spare. */
export const fitsOn = (lamp: LampLike, host: Pick<Item, 'width' | 'depth'>) =>
  lamp.category === 'pendant' || Math.min(host.width, host.depth) >= Math.max(lamp.width, lamp.depth) + 0.04

/** Where lamp number `slot` stands on `host` (plan coordinates). Rotation is the host's. */
export function spotOn(host: HostLike, lamp: LampLike, slot: 0 | 1 = 0): Vec2 {
  const hw = host.item.width / 2
  const hd = host.item.depth / 2
  const lw = lamp.width / 2
  const ld = lamp.depth / 2
  const cat = host.item.category
  let local: Vec2
  if (cat === 'dining-table') local = v(0, 0)
  else if (cat === 'nightstand' || cat === 'side-table') local = v(0, -hd + ld + 0.03)
  // r13-b: a coffee table's lamp stands toward one end, centred front to back.
  else if (cat === 'coffee-table') local = v(-Math.max(0, hw - lw - 0.1), 0)
  else if (cat === 'desk') local = v((slot === 0 ? -1 : 1) * Math.max(0, hw - lw - 0.05), -hd + ld + 0.05)
  else local = v((slot === 0 ? -1 : 1) * Math.max(0, hw - lw - 0.08), -hd + ld + 0.05)
  return add(v(host.p.x, host.p.y), rot(local, host.p.rotation))
}

/** Clamp a hosted piece's centre so its footprint stays on the host top (inset 2 cm). */
export function clampOnHost(host: HostLike, lamp: LampLike, pt: Vec2): Vec2 {
  const c = v(host.p.x, host.p.y)
  const local = rot(sub(pt, c), -host.p.rotation)
  const mx = Math.max(0, host.item.width / 2 - lamp.width / 2 - 0.02)
  const my = Math.max(0, host.item.depth / 2 - lamp.depth / 2 - 0.02)
  const cl = v(Math.max(-mx, Math.min(mx, local.x)), Math.max(-my, Math.min(my, local.y)))
  return add(c, rot(cl, host.p.rotation))
}

function hostsOf(doc: Doc, layout: Layout, category?: string) {
  const cats = hostCats(category)
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  return layout.placements.flatMap((p) => {
    const it = byId.get(p.itemId)
    if (!it || p.on || !cats.has(it.category)) return []
    return [{ id: p.id, item: it, p, body: orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation) }]
  })
}

/** The surface whose top contains `pt` and fits `lamp` (preferring `preferId`), or undefined. */
export function hostAt(doc: Doc, layout: Layout, pt: Vec2, lamp: LampLike, preferId?: string): { id: string; item: Item; p: Placement } | undefined {
  const hits = hostsOf(doc, layout, lamp.category).filter((h) => fitsOn(lamp, h.item) && pointInPolygon(pt, h.body))
  return hits.find((h) => h.id === preferId) ?? hits[0]
}

/**
 * Where a table lamp goes: on a free nightstand, desk, side table or other top. `nextTo` is the
 * host itself or a piece the host stands beside; `near` favours close hosts.
 */
export function findHostDrop(doc: Doc, layout: Layout, item: DropItem, opts: { near?: Vec2; nextTo?: string; exclude?: string; preferRoom?: string; onlyCategory?: string; onlyRoom?: string } = {}): DropResult {
  const pendant = item.category === 'pendant'
  const all = hostsOf(doc, layout, item.category)
  if (!all.length)
    return { ok: false, reason: pendant ? 'A pendant hangs over a dining table. Add one first.' : 'A table lamp needs something to stand on. Add a nightstand, side table or desk first.' }
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  const placed = layout.placements.flatMap((p) => {
    const it = byId.get(p.itemId)
    return it && p.id !== opts.exclude ? [{ p, it, body: orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation) }] : []
  })
  const load = (id: string) => placed.filter((x) => x.p.on === id).length
  const lights = placed.filter((x) => LAMPS.has(x.it.category))
  const target = opts.nextTo ? placed.find((x) => x.p.id === opts.nextTo) : undefined
  let best: { score: number; spot: DropSpot } | undefined
  for (const h of all) {
    if (!fitsOn(item, h.item)) continue
    const used = load(h.id)
    const slots = SURFACE_SLOTS[h.item.category] ?? 1
    if (used >= slots) continue
    if (opts.onlyCategory && h.item.category !== opts.onlyCategory) continue
    const room = doc.rooms.find((r) => pointInPolygon(v(h.p.x, h.p.y), r.polygon))
    if (opts.onlyRoom !== undefined && room?.id !== opts.onlyRoom) continue
    const unlit = !lights.some((l) => l.p.on === h.id || (!l.p.on && polygonGap(h.body, l.body) <= NEXT_TO_REACH))
    let score = 0
    if (opts.nextTo === h.id) score += 100
    else if (target && polygonGap(h.body, target.body) <= NEXT_TO_REACH) score += 50
    const cat = h.item.category
    if (cat === 'nightstand') score += unlit ? 30 : 10
    else if (cat === 'desk') score += used === 0 ? 25 : 10
    else if (cat === 'side-table') {
      const bySeat = placed.some((x) => ['sofa', 'armchair'].includes(x.it.category) && polygonGap(h.body, x.body) <= NEXT_TO_REACH)
      score += bySeat && unlit ? 22 : 10
    } else if (cat === 'coffee-table') score += 6 // r13-b: a lamp's last-choice top, after side tables
    else score += 10
    if (room) {
      const a = roomAffinity(room, pendant ? 'dining-table' : 'table-lamp', doc.rooms)
      if (a !== -Infinity) score += a
      if (opts.preferRoom && room.id === opts.preferRoom) score += 40
    }
    if (opts.near) score -= dist(opts.near, v(h.p.x, h.p.y))
    const c = spotOn(h, item, used === 0 ? 0 : 1)
    if (!best || score > best.score) best = { score, spot: { x: c.x, y: c.y, rotation: h.p.rotation, on: h.id, roomId: room?.id, roomName: room?.name } }
  }
  if (!best) return { ok: false, reason: pendant ? 'Every dining table already has a pendant over it.' : 'Every table top already has a lamp. Add a side table or nightstand for this one.' }
  return { ok: true, spot: best.spot }
}

const samePose = (a: Pick<Placement, 'x' | 'y' | 'rotation'>, b: Pick<Placement, 'x' | 'y' | 'rotation'>) =>
  Math.abs(a.x - b.x) < 1e-9 && Math.abs(a.y - b.y) < 1e-9 && Math.abs(a.rotation - b.rotation) < 1e-9

export interface ReconcileReport {
  /** Hosted pieces taken off because nothing could hold them (placement ids). */
  removed: string[]
  /** Hosted pieces moved onto another host (placement ids). */
  rehosted: string[]
}

/**
 * Keep hosted pieces (table lamps) on their hosts after an edit: a lamp follows its host's move or
 * turn, slides back onto a shrunk top, and moves to another free top (or comes off) when its host
 * goes. Returns `next` itself when nothing needed fixing.
 */
export function reconcileHosts(prev: Doc, next: Doc, report?: ReconcileReport): Doc {
  if (!next.layouts.some((l) => l.placements.some((p) => p.on))) return next
  const items = new Map(next.items.map((i) => [i.id, i]))
  const prevItems = new Map([...prev.items, ...next.items].map((i) => [i.id, i]))
  let changed = false
  const layouts = next.layouts.map((layout) => {
    if (!layout.placements.some((p) => p.on)) return layout
    const before = prev.layouts.find((l) => l.id === layout.id)
    const prevById = new Map((before?.placements ?? []).map((p) => [p.id, p]))
    const prevHostId = (id: string) => prevById.get(id)?.itemId
    const byId = new Map(layout.placements.map((p) => [p.id, p]))
    let placements = layout.placements
    let lc = false
    for (const c0 of layout.placements) {
      if (!c0.on) continue
      const lamp = items.get(c0.itemId)
      let c = byId.get(c0.id)!
      const host = byId.get(c0.on)
      const hostItem = host && items.get(host.itemId)
      const rehost = () => {
        // Only onto a free top of the same kind in the same room (a bedside lamp never wanders off to
        // the desk across the home); otherwise it comes off and waits in Mine.
        const cur: Layout = { ...layout, placements }
        const room = next.rooms.find((r) => pointInPolygon(v(c.x, c.y), r.polygon))
        const wasOn = hostItem ?? (prevHostId(c0.on!) && prevItems.get(prevHostId(c0.on!)!))
        const drop =
          lamp && wasOn && room
            ? findHostDrop(next, cur, lamp, { near: v(c.x, c.y), exclude: c.id, onlyCategory: wasOn.category, onlyRoom: room.id })
            : undefined
        if (drop?.ok) {
          const n = { ...c, x: drop.spot.x, y: drop.spot.y, rotation: drop.spot.rotation, on: drop.spot.on }
          placements = placements.map((p) => (p.id === c.id ? n : p))
          byId.set(c.id, n)
          report?.rehosted.push(c.id)
        } else {
          placements = placements.filter((p) => p.id !== c.id)
          byId.delete(c.id)
          report?.removed.push(c.id)
        }
        lc = true
      }
      if (!host || !hostItem || host.on || !lamp || !fitsOn(lamp, hostItem)) {
        rehost()
        continue
      }
      const hPrev = prevById.get(host.id)
      const cPrev = prevById.get(c.id)
      if (hPrev && cPrev && !samePose(hPrev, host) && samePose(cPrev, c)) {
        const local = rot(sub(v(c.x, c.y), v(hPrev.x, hPrev.y)), -hPrev.rotation)
        const w = add(v(host.x, host.y), rot(local, host.rotation))
        c = { ...c, x: w.x, y: w.y, rotation: c.rotation + host.rotation - hPrev.rotation }
        placements = placements.map((p) => (p.id === c.id ? c : p))
        byId.set(c.id, c)
        lc = true
      }
      const cl = clampOnHost({ item: hostItem, p: host }, lamp, v(c.x, c.y))
      if (Math.abs(cl.x - c.x) > 1e-6 || Math.abs(cl.y - c.y) > 1e-6) {
        c = { ...c, x: cl.x, y: cl.y }
        placements = placements.map((p) => (p.id === c.id ? c : p))
        byId.set(c.id, c)
        lc = true
      }
    }
    if (!lc) return layout
    changed = true
    return { ...layout, placements }
  })
  return changed ? { ...next, layouts } : next
}
