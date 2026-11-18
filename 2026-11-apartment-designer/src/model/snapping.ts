import { add, closestPointOnSegment, DEG, dot, norm, rot, scale, sub, v, wallInwardNormal, wallSegments } from './geometry'
import type { Item, Placement, Room, Vec2 } from './types'

export interface SnapSettings {
  grid: number // 0 = off
  walls: boolean
  items: boolean
}

export interface SnapGuide {
  a: Vec2
  b: Vec2
}

export interface SnapResult {
  placement: Placement
  guides: SnapGuide[]
}

const WALL_GAP = 0.2
const WALL_ANGLE = 25 * DEG
const EDGE_THRESHOLD = 0.06

const SIDES = [
  { local: v(0, 1), along: 'depth' as const },
  { local: v(0, -1), along: 'depth' as const },
  { local: v(1, 0), along: 'width' as const },
  { local: v(-1, 0), along: 'width' as const },
]

function angleDiff(a: number, b: number) {
  let d = (a - b) % (2 * Math.PI)
  if (d > Math.PI) d -= 2 * Math.PI
  if (d < -Math.PI) d += 2 * Math.PI
  return d
}

export function snapPlacement(
  item: Item,
  proposed: Placement,
  rooms: Room[],
  others: { item: Item; p: Placement }[],
  settings: SnapSettings,
): SnapResult {
  let p = { ...proposed }
  const guides: SnapGuide[] = []

  if (settings.grid > 0) {
    p.x = Math.round(p.x / settings.grid) * settings.grid
    p.y = Math.round(p.y / settings.grid) * settings.grid
  }

  if (settings.items) {
    const aligned = alignToItems(item, p, others)
    p = aligned.placement
    guides.push(...aligned.guides)
  }

  if (settings.walls) {
    const snapped = snapToWalls(item, p, rooms)
    if (snapped) {
      p = snapped.placement
      guides.push(...snapped.guides)
    }
  }
  return { placement: p, guides }
}

function snapToWalls(item: Item, p: Placement, rooms: Room[]): SnapResult | null {
  type Best = { gap: number; placement: Placement; guide: SnapGuide }
  let best = null as Best | null
  const center = v(p.x, p.y)
  for (const room of rooms) {
    const walls = wallSegments(room)
    for (let wi = 0; wi < walls.length; wi++) {
      const [a, b] = walls[wi]
      const n = wallInwardNormal(room, wi)
      const wallDir = norm(sub(b, a))
      for (const side of SIDES) {
        const sideNormal = rot(side.local, p.rotation)
        // The side must face the wall: its outward normal points against the wall's inward normal.
        const target = scale(n, -1)
        const diff = angleDiff(Math.atan2(target.y, target.x), Math.atan2(sideNormal.y, sideNormal.x))
        if (Math.abs(diff) > WALL_ANGLE) continue
        const rotation = p.rotation + diff
        const half = side.along === 'depth' ? item.depth / 2 : item.width / 2
        const otherHalf = side.along === 'depth' ? item.width / 2 : item.depth / 2
        // Signed distance from center to the wall line, measured along the inward normal.
        const dCenter = dot(sub(center, a), n)
        const gap = dCenter - half
        if (gap < -half || gap > WALL_GAP) continue
        // The item must sit along this wall segment, not past its ends.
        const t = dot(sub(center, a), wallDir)
        const wallLen = dot(sub(b, a), wallDir)
        if (t < -otherHalf * 0.5 || t > wallLen + otherHalf * 0.5) continue
        const absGap = Math.abs(gap)
        if (!best || absGap < best.gap) {
          const newCenter = add(center, scale(n, half - dCenter))
          const foot = closestPointOnSegment(newCenter, a, b).point
          best = {
            gap: absGap,
            placement: { ...p, x: newCenter.x, y: newCenter.y, rotation },
            guide: { a: add(foot, scale(wallDir, -otherHalf)), b: add(foot, scale(wallDir, otherHalf)) },
          }
        }
      }
    }
  }
  return best ? { placement: best.placement, guides: [best.guide] } : null
}

function alignToItems(item: Item, p: Placement, others: { item: Item; p: Placement }[]): SnapResult {
  type Best = { d: number; guide: SnapGuide; frame: number; other: Placement }
  let bestX = null as Best | null
  let bestY = null as Best | null
  for (const o of others) {
    const rd = angleDiff(p.rotation, o.p.rotation)
    const quarter = Math.round(rd / (Math.PI / 2))
    if (Math.abs(rd - quarter * (Math.PI / 2)) > 2 * DEG) continue
    const swapped = Math.abs(quarter) % 2 === 1
    // Work in the other item's frame.
    const local = rot(sub(v(p.x, p.y), v(o.p.x, o.p.y)), -o.p.rotation)
    const myHx = (swapped ? item.depth : item.width) / 2
    const myHy = (swapped ? item.width : item.depth) / 2
    const oHx = o.item.width / 2
    const oHy = o.item.depth / 2
    const candX = [
      [-oHx - myHx, -oHx],
      [oHx + myHx, oHx],
      [-oHx + myHx, -oHx],
      [oHx - myHx, oHx],
      [0, 0],
    ]
    for (const [target, edge] of candX) {
      const d = target - local.x
      if (Math.abs(d) < EDGE_THRESHOLD && (!bestX || Math.abs(d) < Math.abs(bestX.d))) {
        const ga = add(v(o.p.x, o.p.y), rot(v(edge, -oHy - 0.3), o.p.rotation))
        const gb = add(v(o.p.x, o.p.y), rot(v(edge, oHy + 0.3), o.p.rotation))
        bestX = { d, guide: { a: ga, b: gb }, frame: o.p.rotation, other: o.p }
      }
    }
    const candY = [
      [-oHy - myHy, -oHy],
      [oHy + myHy, oHy],
      [-oHy + myHy, -oHy],
      [oHy - myHy, oHy],
      [0, 0],
    ]
    for (const [target, edge] of candY) {
      const d = target - local.y
      if (Math.abs(d) < EDGE_THRESHOLD && (!bestY || Math.abs(d) < Math.abs(bestY.d))) {
        const ga = add(v(o.p.x, o.p.y), rot(v(-oHx - 0.3, edge), o.p.rotation))
        const gb = add(v(o.p.x, o.p.y), rot(v(oHx + 0.3, edge), o.p.rotation))
        bestY = { d, guide: { a: ga, b: gb }, frame: o.p.rotation, other: o.p }
      }
    }
  }
  let center = v(p.x, p.y)
  const guides: SnapGuide[] = []
  if (bestX) {
    center = add(center, rot(v(bestX.d, 0), bestX.frame))
    guides.push(bestX.guide)
  }
  if (bestY) {
    center = add(center, rot(v(0, bestY.d), bestY.frame))
    guides.push(bestY.guide)
  }
  return { placement: { ...p, x: center.x, y: center.y }, guides }
}

/** Snap a polyline point to 45° steps relative to the previous point, and to the grid. */
export function snapDrawPoint(prev: Vec2 | null, p: Vec2, grid: number, angleSnap: boolean): Vec2 {
  if (!prev) return grid > 0 ? v(Math.round(p.x / grid) * grid, Math.round(p.y / grid) * grid) : p
  let d = sub(p, prev)
  if (angleSnap) {
    const a = Math.round(Math.atan2(d.y, d.x) / (Math.PI / 4)) * (Math.PI / 4)
    const l = Math.hypot(d.x, d.y)
    const lr = grid > 0 ? Math.round(l / grid) * grid : l
    d = v(Math.cos(a) * lr, Math.sin(a) * lr)
    return add(prev, d)
  }
  return grid > 0 ? v(Math.round(p.x / grid) * grid, Math.round(p.y / grid) * grid) : p
}
