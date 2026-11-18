import type { Item, Placement, Room, Vec2, Opening } from './types'

export const v = (x: number, y: number): Vec2 => ({ x, y })
export const add = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y })
export const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y })
export const scale = (a: Vec2, s: number): Vec2 => ({ x: a.x * s, y: a.y * s })
export const dot = (a: Vec2, b: Vec2) => a.x * b.x + a.y * b.y
export const cross = (a: Vec2, b: Vec2) => a.x * b.y - a.y * b.x
export const len = (a: Vec2) => Math.hypot(a.x, a.y)
export const dist = (a: Vec2, b: Vec2) => len(sub(a, b))
export const norm = (a: Vec2): Vec2 => {
  const l = len(a)
  return l < 1e-12 ? { x: 0, y: 0 } : { x: a.x / l, y: a.y / l }
}
export const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })
/** Rotate clockwise on the plan (y down), matching SVG rotate(). */
export const rot = (a: Vec2, theta: number): Vec2 => {
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  return { x: a.x * c - a.y * s, y: a.x * s + a.y * c }
}

export const DEG = Math.PI / 180

export function normalizeAngle(a: number) {
  const t = a % (2 * Math.PI)
  return t < 0 ? t + 2 * Math.PI : t
}

export function signedArea(poly: Vec2[]) {
  let a = 0
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i]
    const q = poly[(i + 1) % poly.length]
    a += p.x * q.y - q.x * p.y
  }
  return a / 2
}

export function polygonArea(poly: Vec2[]) {
  return Math.abs(signedArea(poly))
}

export function centroid(poly: Vec2[]): Vec2 {
  const a = signedArea(poly)
  if (Math.abs(a) < 1e-9) {
    const s = poly.reduce((acc, p) => add(acc, p), v(0, 0))
    return scale(s, 1 / Math.max(1, poly.length))
  }
  let cx = 0
  let cy = 0
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i]
    const q = poly[(i + 1) % poly.length]
    const f = p.x * q.y - q.x * p.y
    cx += (p.x + q.x) * f
    cy += (p.y + q.y) * f
  }
  return { x: cx / (6 * a), y: cy / (6 * a) }
}

export function pointInPolygon(p: Vec2, poly: Vec2[]) {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i]
    const b = poly[j]
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside
  }
  return inside
}

/** Proper intersection of two segments (touching endpoints doesn't count). */
export function segmentsCross(a: Vec2, b: Vec2, c: Vec2, d: Vec2, eps = 1e-6) {
  const r = sub(b, a)
  const s = sub(d, c)
  const denom = cross(r, s)
  if (Math.abs(denom) < 1e-12) return false
  const t = cross(sub(c, a), s) / denom
  const u = cross(sub(c, a), r) / denom
  return t > eps && t < 1 - eps && u > eps && u < 1 - eps
}

export function closestPointOnSegment(p: Vec2, a: Vec2, b: Vec2): { point: Vec2; t: number; distance: number } {
  const ab = sub(b, a)
  const l2 = dot(ab, ab)
  const t = l2 < 1e-12 ? 0 : Math.max(0, Math.min(1, dot(sub(p, a), ab) / l2))
  const point = lerp(a, b, t)
  return { point, t, distance: dist(p, point) }
}

/** Separating axis test for two convex polygons. Returns true if interiors overlap by more than eps. */
export function convexOverlap(a: Vec2[], b: Vec2[], eps = 1e-4) {
  for (const poly of [a, b]) {
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i]
      const q = poly[(i + 1) % poly.length]
      const axis = norm({ x: -(q.y - p.y), y: q.x - p.x })
      let minA = Infinity
      let maxA = -Infinity
      let minB = Infinity
      let maxB = -Infinity
      for (const pt of a) {
        const d = dot(pt, axis)
        minA = Math.min(minA, d)
        maxA = Math.max(maxA, d)
      }
      for (const pt of b) {
        const d = dot(pt, axis)
        minB = Math.min(minB, d)
        maxB = Math.max(maxB, d)
      }
      if (maxA <= minB + eps || maxB <= minA + eps) return false
    }
  }
  return true
}

/** Oriented rectangle from a center, half extents and rotation. Corners in order. */
export function orientedRect(center: Vec2, halfW: number, halfD: number, rotation: number, offset: Vec2 = v(0, 0)): Vec2[] {
  const corners = [v(-halfW, -halfD), v(halfW, -halfD), v(halfW, halfD), v(-halfW, halfD)]
  return corners.map((c) => add(center, rot(add(c, offset), rotation)))
}

/** Footprint of a placed item. Local +y is the item's front. */
export function footprint(item: Item, p: Placement): Vec2[] {
  return orientedRect(v(p.x, p.y), item.width / 2, item.depth / 2, p.rotation)
}

export type ZoneSide = 'front' | 'back' | 'left' | 'right'

export function clearanceZones(item: Item, p: Placement): { side: ZoneSide; poly: Vec2[] }[] {
  const zones: { side: ZoneSide; poly: Vec2[] }[] = []
  const hw = item.width / 2
  const hd = item.depth / 2
  const c = item.clearance
  const center = v(p.x, p.y)
  if (c.front > 0) zones.push({ side: 'front', poly: orientedRect(center, hw, c.front / 2, p.rotation, v(0, hd + c.front / 2)) })
  if (c.back > 0) zones.push({ side: 'back', poly: orientedRect(center, hw, c.back / 2, p.rotation, v(0, -hd - c.back / 2)) })
  // Left/right are from the point of view of someone using the item (facing out of its front).
  if (c.left > 0) zones.push({ side: 'left', poly: orientedRect(center, c.left / 2, hd, p.rotation, v(hw + c.left / 2, 0)) })
  if (c.right > 0) zones.push({ side: 'right', poly: orientedRect(center, c.right / 2, hd, p.rotation, v(-hw - c.right / 2, 0)) })
  return zones
}

export function wallSegments(room: Room): [Vec2, Vec2][] {
  const n = room.polygon.length
  return room.polygon.map((p, i) => [p, room.polygon[(i + 1) % n]] as [Vec2, Vec2])
}

/** Unit normal of a wall pointing into the room. */
export function wallInwardNormal(room: Room, wall: number): Vec2 {
  const [a, b] = wallSegments(room)[wall]
  const d = norm(sub(b, a))
  // For a counterclockwise polygon in a y-down frame... use the sign of the area to pick the side.
  const left = v(d.y, -d.x)
  const ccw = signedArea(room.polygon) > 0
  return ccw ? scale(left, -1) : left
}

export function openingEndpoints(room: Room, o: Opening): [Vec2, Vec2] {
  const [a, b] = wallSegments(room)[o.wall]
  const d = norm(sub(b, a))
  return [add(a, scale(d, o.offset)), add(a, scale(d, o.offset + o.width))]
}

/** Polygon approximating a door's swing area (a quarter disc). */
export function doorSwing(room: Room, o: Opening, steps = 12): Vec2[] | null {
  if (o.kind !== 'door') return null
  const [s, e] = openingEndpoints(room, o)
  const hingeAtStart = (o.hinge ?? 'start') === 'start'
  const hinge = hingeAtStart ? s : e
  const free = hingeAtStart ? e : s
  let n = wallInwardNormal(room, o.wall)
  if ((o.swing ?? 'in') === 'out') n = scale(n, -1)
  const r = o.width
  const startDir = norm(sub(free, hinge))
  const pts: Vec2[] = [hinge]
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * (Math.PI / 2)
    const dir = add(scale(startDir, Math.cos(t)), scale(n, Math.sin(t)))
    pts.push(add(hinge, scale(dir, r)))
  }
  return pts
}

/** Split a polygon (possibly non-convex, like a door arc which is convex) — callers only pass convex shapes to SAT. */
export function polygonsOverlap(a: Vec2[], b: Vec2[]) {
  return convexOverlap(a, b)
}

/** True if the convex polygon lies inside the room polygon (corners inside and no edge crossings). */
export function polygonInsideRoom(poly: Vec2[], room: Vec2[], tolerance = 0.005) {
  // Shrink the test polygon a little toward its center so items flush against a wall pass.
  const c = centroid(poly)
  const shrunk = poly.map((p) => {
    const d = sub(p, c)
    const l = len(d)
    return l < tolerance ? p : add(c, scale(d, (l - tolerance) / l))
  })
  if (!shrunk.every((p) => pointInPolygon(p, room))) return false
  for (let i = 0; i < shrunk.length; i++) {
    const a = shrunk[i]
    const b = shrunk[(i + 1) % shrunk.length]
    for (let j = 0; j < room.length; j++) {
      if (segmentsCross(a, b, room[j], room[(j + 1) % room.length])) return false
    }
  }
  return true
}

export function bounds(points: Vec2[]) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const p of points) {
    minX = Math.min(minX, p.x)
    minY = Math.min(minY, p.y)
    maxX = Math.max(maxX, p.x)
    maxY = Math.max(maxY, p.y)
  }
  return { minX, minY, maxX, maxY }
}

/** Offset a simple polygon outward by d (miter joins). Used to draw wall thickness. */
export function offsetPolygon(poly: Vec2[], d: number): Vec2[] {
  const ccw = signedArea(poly) > 0
  const n = poly.length
  return poly.map((p, i) => {
    const prev = poly[(i - 1 + n) % n]
    const next = poly[(i + 1) % n]
    const d1 = norm(sub(p, prev))
    const d2 = norm(sub(next, p))
    // outward normals
    const n1 = ccw ? v(d1.y, -d1.x) : v(-d1.y, d1.x)
    const n2 = ccw ? v(d2.y, -d2.x) : v(-d2.y, d2.x)
    const m = norm(add(n1, n2))
    const cosHalf = dot(m, n1)
    const k = Math.abs(cosHalf) < 0.2 ? d : d / cosHalf
    return add(p, scale(m, k))
  })
}
