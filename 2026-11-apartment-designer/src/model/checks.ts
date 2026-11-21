import {
  add,
  bounds,
  centroid,
  clearanceZones,
  closestPointOnSegment,
  convexOverlap,
  doorSwing,
  footprint,
  openingEndpoints,
  pointInPolygon,
  polygonInsideRoom,
  scale,
  v,
  wallInwardNormal,
} from './geometry'
import type { Doc, Item, Layout, Placement, Room, Vec2 } from './types'
import { formatLength } from './units'

export type Severity = 'error' | 'warning' | 'info'

export interface Issue {
  id: string
  severity: Severity
  kind: 'overlap' | 'wall' | 'clearance' | 'door' | 'path'
  message: string
  placementIds: string[]
  /** Area to highlight on the plan. */
  area?: Vec2[]
}

interface Placed {
  p: Placement
  item: Item
  body: Vec2[]
}

export function placedItems(doc: Doc, layout: Layout): Placed[] {
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  const out: Placed[] = []
  for (const p of layout.placements) {
    const item = byId.get(p.itemId)
    if (item) out.push({ p, item, body: footprint(item, p) })
  }
  return out
}

/** Doors' swing polygons across all rooms, a convex quarter disc each. */
export function doorSwings(rooms: Room[]) {
  const out: { room: Room; openingId: string; poly: Vec2[] }[] = []
  for (const room of rooms)
    for (const o of room.openings) {
      const poly = doorSwing(room, o)
      if (poly) out.push({ room, openingId: o.id, poly })
    }
  return out
}

/** Pull-out space (m) kept free along the long sides of a dining table that seats four or more, measured from the table edge. */
export const TABLE_PULL_OUT = 0.75
/** A dining chair counts as at its table when this close (edge to edge, m). */
export const CHAIR_REACH = 0.3
/** Storage whose doors or drawers need this much free floor in front, at the very least (m). Up to the catalog ideal is a soft preference. */
export const MIN_DRAWER_FRONT = 0.75
export const DRAWER_PIECES = new Set(['dresser', 'wardrobe'])
/** Less than this behind a chair pulled up to a table always gets flagged (m). */
const MIN_CHAIR_BACK = 0.15

/** Smallest distance between two convex polygons (0 when they touch or overlap). */
export function polygonGap(a: Vec2[], b: Vec2[]): number {
  if (convexOverlap(a, b, 0)) return 0
  let best = Infinity
  for (const [pts, poly] of [
    [a, b],
    [b, a],
  ] as const)
    for (const pt of pts)
      for (let i = 0; i < poly.length; i++) best = Math.min(best, closestPointOnSegment(pt, poly[i], poly[(i + 1) % poly.length]).distance)
  return best
}

/**
 * Free floor a dining chair needs behind it (m). Pulled up to a table, the pull-out is measured
 * from the table edge: TABLE_PULL_OUT less the gap to the table and the chair's own depth (never
 * under 15 cm). Away from any table it needs its own back clearance.
 */
export function chairBackNeed(chair: { item: Pick<Item, 'depth' | 'clearance'>; body: Vec2[] }, tables: { body: Vec2[] }[]): number {
  const gaps = tables.map((t) => polygonGap(chair.body, t.body)).filter((g) => g <= CHAIR_REACH)
  if (!gaps.length) return chair.item.clearance.back
  return Math.max(MIN_CHAIR_BACK, TABLE_PULL_OUT - Math.min(...gaps) - chair.item.depth)
}

/**
 * The clearance a piece is checked against: a dresser or wardrobe needs only MIN_DRAWER_FRONT in
 * front (its catalog ideal is a placement preference), and a dining chair at a table needs only
 * what chairBackNeed says behind it.
 */
export function checkedClearance(item: Item, body: Vec2[], tables: { body: Vec2[] }[]): Item['clearance'] {
  const c = item.clearance
  if (DRAWER_PIECES.has(item.category) && c.front > MIN_DRAWER_FRONT) return { ...c, front: MIN_DRAWER_FRONT }
  if (item.category === 'dining-chair' && c.back > 0) return { ...c, back: chairBackNeed({ item, body }, tables) }
  return c
}

export function checkLayout(doc: Doc, layout: Layout): Issue[] {
  const issues: Issue[] = []
  const units = doc.apartment.units
  const placed = placedItems(doc, layout)
  const solid = placed.filter((x) => !x.item.floorLevel && !x.p.on)

  // Items overlapping each other.
  for (let i = 0; i < solid.length; i++)
    for (let j = i + 1; j < solid.length; j++) {
      const a = solid[i]
      const b = solid[j]
      if (convexOverlap(a.body, b.body, 0.002))
        issues.push({
          id: `overlap:${a.p.id}:${b.p.id}`,
          severity: 'error',
          kind: 'overlap',
          message: `${a.item.name} overlaps ${b.item.name}`,
          placementIds: [a.p.id, b.p.id],
        })
    }

  // Items crossing walls or outside every room.
  for (const x of placed) {
    // A table lamp stands on its host, which already answers for the walls.
    if (x.p.on) continue
    const inside = doc.rooms.some((r) => polygonInsideRoom(x.body, r.polygon))
    if (!inside)
      issues.push({
        id: `wall:${x.p.id}`,
        severity: 'error',
        kind: 'wall',
        message: `${x.item.name} crosses a wall or sits outside the rooms`,
        placementIds: [x.p.id],
      })
  }

  // Clearance zones blocked by other items or walls.
  const tables = solid.filter((o) => o.item.category === 'dining-table')
  for (const x of placed) {
    if (x.p.on) continue
    const checked = { ...x.item, clearance: checkedClearance(x.item, x.body, tables) }
    for (const zone of clearanceZones(checked, x.p)) {
      const need = checked.clearance[zone.side]
      const blockers = solid.filter((o) => o.p.id !== x.p.id && !sideCompanion(x.item.category, zone.side, o.item.category) && convexOverlap(zone.poly, o.body, 0.005))
      for (const b of blockers)
        issues.push({
          id: `clear:${x.p.id}:${zone.side}:${b.p.id}`,
          severity: 'warning',
          kind: 'clearance',
          message: `${x.item.name} needs ${formatLength(need, units)} ${sideLabel(zone.side)}; ${b.item.name} is in the way`,
          placementIds: [x.p.id, b.p.id],
          area: zone.poly,
        })
      const inRoom = doc.rooms.some((r) => polygonInsideRoom(zone.poly, r.polygon, 0.01))
      if (!inRoom && doc.rooms.some((r) => pointInPolygon(centerOf(x.body), r.polygon)))
        issues.push({
          id: `clearwall:${x.p.id}:${zone.side}`,
          severity: 'warning',
          kind: 'clearance',
          message: `${x.item.name} needs ${formatLength(need, units)} ${sideLabel(zone.side)}; a wall is in the way`,
          placementIds: [x.p.id],
          area: zone.poly,
        })
    }
  }

  // Door swings blocked.
  for (const s of doorSwings(doc.rooms)) {
    for (const x of solid) {
      if (convexOverlap(s.poly, x.body, 0.005))
        issues.push({
          id: `door:${s.openingId}:${x.p.id}`,
          severity: 'error',
          kind: 'door',
          message: `${x.item.name} blocks the ${s.room.name} door swing`,
          placementIds: [x.p.id],
          area: s.poly,
        })
    }
  }

  return issues
}

/**
 * Pieces that belong in another's clearance and don't block it: a nightstand or side table beside
 * a bed or seat, the chair at a desk.
 */
export function sideCompanion(ownerCategory: string, side: string, blockerCategory: string) {
  if (side === 'front' && ownerCategory === 'desk' && blockerCategory === 'office-chair') return true
  return (side === 'left' || side === 'right') && ['bed', 'sofa', 'armchair'].includes(ownerCategory) && ['nightstand', 'side-table', 'floor-lamp', 'table-lamp'].includes(blockerCategory)
}

function centerOf(poly: Vec2[]) {
  return centroid(poly)
}

function sideLabel(side: string) {
  return side === 'front' ? 'in front' : side === 'back' ? 'behind' : `on the ${side}`
}

// ---------------------------------------------------------------------------
// Walking path: occupancy grid + widest path from the entrance to each room.
// ---------------------------------------------------------------------------

export interface PathResult {
  roomId: string
  roomName: string
  reachable: boolean
  /** Narrowest free width along the widest available path, in meters. */
  narrowest: number
  /** Where the narrowest point is. */
  pinch?: Vec2
  path: Vec2[]
}

export interface PathReport {
  entrance: boolean
  results: PathResult[]
}

/** Width reported where only walls (not furniture) limit the path. */
export const ARCH = 10
export const MIN_WALKWAY = 0.6
export const COMFORTABLE_WALKWAY = 0.9

export function walkingPaths(doc: Doc, layout: Layout, cell = 0.05): PathReport {
  const rooms = doc.rooms
  let entrance: { room: Room; at: Vec2; inward: Vec2 } | null = null
  for (const room of rooms)
    for (const o of room.openings)
      if (o.kind === 'door' && o.entrance) {
        const [a, b] = openingEndpoints(room, o)
        entrance = { room, at: scale(add(a, b), 0.5), inward: wallInwardNormal(room, o.wall) }
      }
  if (!entrance || rooms.length === 0) return { entrance: false, results: [] }

  const allPts = rooms.flatMap((r) => r.polygon)
  const bb = bounds(allPts)
  const pad = 0.5
  const x0 = bb.minX - pad
  const y0 = bb.minY - pad
  const W = Math.ceil((bb.maxX - bb.minX + 2 * pad) / cell)
  const H = Math.ceil((bb.maxY - bb.minY + 2 * pad) / cell)
  const free = new Uint8Array(W * H)
  const roomOf = new Int16Array(W * H).fill(-1)
  const cellCenter = (i: number, j: number) => v(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell)

  // Rasterize rooms.
  rooms.forEach((room, ri) => {
    const b = bounds(room.polygon)
    const i0 = Math.max(0, Math.floor((b.minX - x0) / cell))
    const i1 = Math.min(W - 1, Math.ceil((b.maxX - x0) / cell))
    const j0 = Math.max(0, Math.floor((b.minY - y0) / cell))
    const j1 = Math.min(H - 1, Math.ceil((b.maxY - y0) / cell))
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++)
        if (pointInPolygon(cellCenter(i, j), room.polygon)) {
          free[j * W + i] = 1
          roomOf[j * W + i] = ri
        }
  })
  // Carve door passages through the walls.
  for (const room of rooms)
    for (const o of room.openings) {
      if (o.kind !== 'door') continue
      const [a, b] = openingEndpoints(room, o)
      const n = wallInwardNormal(room, o.wall)
      const depthOut = room.wallThickness + 0.1
      const poly = [
        add(a, scale(n, 0.05)),
        add(b, scale(n, 0.05)),
        add(b, scale(n, -depthOut)),
        add(a, scale(n, -depthOut)),
      ]
      fillConvex(poly)
    }
  const emptyFree = free.slice()
  // Block furniture (rugs don't block).
  for (const x of placedItems(doc, layout)) {
    if (x.item.floorLevel || x.p.on) continue
    fillConvex(x.body, 0)
  }

  function fillConvex(poly: Vec2[], value = 1) {
    const b = bounds(poly)
    const i0 = Math.max(0, Math.floor((b.minX - x0) / cell))
    const i1 = Math.min(W - 1, Math.ceil((b.maxX - x0) / cell))
    const j0 = Math.max(0, Math.floor((b.minY - y0) / cell))
    const j1 = Math.min(H - 1, Math.ceil((b.maxY - y0) / cell))
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) if (pointInPolygon(cellCenter(i, j), poly)) free[j * W + i] = value
  }

  // Distance (in meters) from each free cell to the nearest blocked cell, with and without furniture.
  const distM = distanceTransform(free, W, H).map((d) => Math.sqrt(d) * cell)
  const distEmpty = distanceTransform(emptyFree, W, H).map((d) => Math.sqrt(d) * cell)
  // Free width through a cell ≈ twice the distance to the nearest obstacle, minus one cell.
  const rawWidth = (k: number) => Math.max(0, 2 * distM[k] - cell)
  // Only furniture-made pinches count: where the width is the same as in the empty apartment,
  // the walls (a doorway, a narrow hall) are the limit and there's nothing to rearrange.
  const width = (k: number) => (rawWidth(k) < Math.max(0, 2 * distEmpty[k] - cell) - 1.5 * cell ? rawWidth(k) : ARCH)

  // Start: nearest free cell just inside the entrance.
  const startPt = add(entrance.at, scale(entrance.inward, 0.15))
  const start = nearestFree(startPt)
  if (start < 0) return { entrance: true, results: rooms.map((r) => ({ roomId: r.id, roomName: r.name, reachable: false, narrowest: 0, path: [] })) }

  // Widest-path (maximin) search.
  const best = new Float64Array(W * H).fill(-1)
  const parent = new Int32Array(W * H).fill(-1)
  const heap = new MaxHeap()
  best[start] = width(start)
  heap.push(start, best[start])
  const dirs = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ]
  while (heap.size) {
    const [k, val] = heap.pop()
    if (val < best[k]) continue
    const i = k % W
    const j = (k - i) / W
    for (const [di, dj] of dirs) {
      const ni = i + di
      const nj = j + dj
      if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue
      const nk = nj * W + ni
      if (!free[nk]) continue
      if (di && dj && (!free[j * W + ni] || !free[nj * W + i])) continue
      const nv = Math.min(val, width(nk))
      if (nv > best[nk]) {
        best[nk] = nv
        parent[nk] = k
        heap.push(nk, nv)
      }
    }
  }

  const results: PathResult[] = rooms.map((room, ri) => {
    // Target: the most open reachable spot in the room.
    let target = -1
    for (let k = 0; k < W * H; k++) if (roomOf[k] === ri && free[k] && best[k] >= 0 && (target < 0 || distM[k] > distM[target])) target = k
    if (target < 0) return { roomId: room.id, roomName: room.name, reachable: false, narrowest: 0, path: [] }
    const bottleneck = best[target]
    // The widest path can wander; draw the shortest route that is still that wide, preferring open space.
    const route = shortestRoute(target, bottleneck)
    let narrowest = Infinity
    let pinch: Vec2 | undefined
    for (const k of route) {
      if (width(k) < narrowest) {
        narrowest = width(k)
        const i = k % W
        pinch = cellCenter(i, (k - i) / W)
      }
    }
    if (narrowest >= ARCH) pinch = undefined
    const path = route.map((k) => {
      const i = k % W
      return cellCenter(i, (k - i) / W)
    })
    return { roomId: room.id, roomName: room.name, reachable: bottleneck >= 0.3, narrowest: Math.min(narrowest, bottleneck), pinch, path: simplifyPath(path) }
  })
  return { entrance: true, results }

  function shortestRoute(target: number, minWidth: number): number[] {
    const cost = new Float64Array(W * H).fill(Infinity)
    const prev = new Int32Array(W * H).fill(-1)
    const heap = new MaxHeap() // max-heap on negative cost
    cost[start] = 0
    heap.push(start, 0)
    while (heap.size) {
      const [k, negc] = heap.pop()
      const c = -negc
      if (c > cost[k]) continue
      if (k === target) break
      const i = k % W
      const j = (k - i) / W
      for (const [di, dj] of dirs) {
        const ni = i + di
        const nj = j + dj
        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue
        const nk = nj * W + ni
        if (!free[nk] || width(nk) < minWidth - 1e-9) continue
        if (di && dj && (!free[j * W + ni] || !free[nj * W + i])) continue
        const step = (di && dj ? Math.SQRT2 : 1) * (1 + 0.25 / Math.max(0.05, rawWidth(nk)))
        const nc = c + step
        if (nc < cost[nk]) {
          cost[nk] = nc
          prev[nk] = k
          heap.push(nk, -nc)
        }
      }
    }
    const out: number[] = []
    for (let k = target; k >= 0; k = prev[k]) {
      out.push(k)
      if (k === start) break
    }
    return out.reverse()
  }

  function nearestFree(p: Vec2) {
    const ci = Math.floor((p.x - x0) / cell)
    const cj = Math.floor((p.y - y0) / cell)
    for (let r = 0; r < 20; r++)
      for (let dj = -r; dj <= r; dj++)
        for (let di = -r; di <= r; di++) {
          const i = ci + di
          const j = cj + dj
          if (i >= 0 && j >= 0 && i < W && j < H && free[j * W + i]) return j * W + i
        }
    return -1
  }
}

export function pathIssues(report: PathReport, units: 'metric' | 'imperial'): Issue[] {
  if (!report.entrance) return []
  const out: Issue[] = []
  for (const r of report.results) {
    if (!r.reachable)
      out.push({ id: `path:${r.roomId}`, severity: 'error', kind: 'path', message: `No way to walk from the entrance to ${r.roomName}`, placementIds: [] })
    else if (r.narrowest < MIN_WALKWAY)
      out.push({
        id: `path:${r.roomId}`,
        severity: 'warning',
        kind: 'path',
        message: `Path to ${r.roomName} narrows to ${formatLength(r.narrowest, units)} (aim for ${formatLength(COMFORTABLE_WALKWAY, units)})`,
        placementIds: [],
      })
  }
  return out
}

function simplifyPath(path: Vec2[]): Vec2[] {
  if (path.length < 3) return path
  const out = [path[0]]
  for (let i = 1; i < path.length - 1; i++) {
    const a = out[out.length - 1]
    const b = path[i]
    const c = path[i + 1]
    const cr = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    if (Math.abs(cr) > 1e-9) out.push(b)
  }
  out.push(path[path.length - 1])
  return out
}

/** Squared Euclidean distance transform (Felzenszwalb & Huttenlocher) in cell units. Zero on blocked cells. */
export function distanceTransform(free: Uint8Array, W: number, H: number): Float64Array {
  const INF = 1e20
  const f = new Float64Array(W * H)
  for (let k = 0; k < W * H; k++) f[k] = free[k] ? INF : 0
  const n = Math.max(W, H)
  const d = new Float64Array(n)
  const zv = new Int32Array(n)
  const z = new Float64Array(n + 1)
  const buf = new Float64Array(n)
  const edt1 = (len: number) => {
    let k = 0
    zv[0] = 0
    z[0] = -INF
    z[1] = INF
    for (let q = 1; q < len; q++) {
      let s = (buf[q] + q * q - (buf[zv[k]] + zv[k] * zv[k])) / (2 * q - 2 * zv[k])
      while (s <= z[k]) {
        k--
        s = (buf[q] + q * q - (buf[zv[k]] + zv[k] * zv[k])) / (2 * q - 2 * zv[k])
      }
      k++
      zv[k] = q
      z[k] = s
      z[k + 1] = INF
    }
    k = 0
    for (let q = 0; q < len; q++) {
      while (z[k + 1] < q) k++
      d[q] = (q - zv[k]) * (q - zv[k]) + buf[zv[k]]
    }
  }
  for (let i = 0; i < W; i++) {
    for (let j = 0; j < H; j++) buf[j] = f[j * W + i]
    edt1(H)
    for (let j = 0; j < H; j++) f[j * W + i] = d[j]
  }
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) buf[i] = f[j * W + i]
    edt1(W)
    for (let i = 0; i < W; i++) f[j * W + i] = d[i]
  }
  return f
}

class MaxHeap {
  private keys: number[] = []
  private vals: number[] = []
  get size() {
    return this.keys.length
  }
  push(k: number, val: number) {
    this.keys.push(k)
    this.vals.push(val)
    let i = this.keys.length - 1
    while (i > 0) {
      const p = (i - 1) >> 1
      if (this.vals[p] >= this.vals[i]) break
      this.swap(i, p)
      i = p
    }
  }
  pop(): [number, number] {
    const top: [number, number] = [this.keys[0], this.vals[0]]
    const lk = this.keys.pop()!
    const lv = this.vals.pop()!
    if (this.keys.length) {
      this.keys[0] = lk
      this.vals[0] = lv
      let i = 0
      for (;;) {
        const l = 2 * i + 1
        const r = l + 1
        let m = i
        if (l < this.keys.length && this.vals[l] > this.vals[m]) m = l
        if (r < this.keys.length && this.vals[r] > this.vals[m]) m = r
        if (m === i) break
        this.swap(i, m)
        i = m
      }
    }
    return top
  }
  private swap(a: number, b: number) {
    ;[this.keys[a], this.keys[b]] = [this.keys[b], this.keys[a]]
    ;[this.vals[a], this.vals[b]] = [this.vals[b], this.vals[a]]
  }
}
