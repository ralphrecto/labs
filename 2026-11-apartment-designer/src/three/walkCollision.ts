import { add, dist, norm, openingEndpoints, pointInPolygon, scale, sub, v, wallInwardNormal, wallSegments } from '../model/geometry'
import type { Opening, Room, Vec2 } from '../model/types'

// Walking in: keep the eye out of the walls, and let it through doorways between rooms.

/** How far the eye keeps from a wall's inside face (m). */
export const WALL_GAP = 0.25

/** A solid stretch of a wall's inside face, in plan coordinates. */
export type WallSeg = [Vec2, Vec2]

/**
 * Doors cut through to the room on the other side: a door drawn on one room's wall opens the
 * neighbouring room's wall where it lies against it, so you can see and walk through.
 * Returns, per room id, the extra door openings its own walls need.
 */
export function sharedDoorways(rooms: Room[]): Map<string, Opening[]> {
  const out = new Map<string, Opening[]>()
  for (const src of rooms) {
    for (const o of src.openings) {
      if (o.kind !== 'door') continue
      const [p, q] = openingEndpoints(src, o)
      const dir = norm(sub(q, p))
      for (const dst of rooms) {
        if (dst === src) continue
        wallSegments(dst).forEach(([a, b], wi) => {
          const L = dist(a, b)
          if (L < 1e-6) return
          const d = norm(sub(b, a))
          // Parallel and close: the two walls are the same wall seen from both sides.
          if (Math.abs(d.x * dir.x + d.y * dir.y) < 0.99) return
          const n = v(-d.y, d.x)
          const gap = Math.abs((p.x - a.x) * n.x + (p.y - a.y) * n.y)
          if (gap > src.wallThickness + dst.wallThickness + 0.05) return
          const s0 = (p.x - a.x) * d.x + (p.y - a.y) * d.y
          const s1 = (q.x - a.x) * d.x + (q.y - a.y) * d.y
          const lo = Math.max(0, Math.min(s0, s1))
          const hi = Math.min(L, Math.max(s0, s1))
          if (hi - lo < 0.3) return
          // The neighbour already has its own opening here.
          if (dst.openings.some((x) => x.wall === wi && x.offset < hi && x.offset + x.width > lo)) return
          const list = out.get(dst.id) ?? []
          list.push({ id: `${o.id}~${dst.id}`, kind: 'door', wall: wi, offset: lo, width: hi - lo, height: o.height, sill: 0 })
          out.set(dst.id, list)
        })
      }
    }
  }
  return out
}

/**
 * The solid wall faces to collide with. A door counts as a gap when there's a room on its far side
 * (the front door leads nowhere you can walk, so it stays shut).
 */
export function wallSegmentsForWalking(rooms: Room[], shared = sharedDoorways(rooms)): WallSeg[] {
  const segs: WallSeg[] = []
  for (const room of rooms) {
    const openings = [...room.openings, ...(shared.get(room.id) ?? [])]
    wallSegments(room).forEach(([a, b], wi) => {
      const L = dist(a, b)
      if (L < 1e-6) return
      const d = norm(sub(b, a))
      const n = wallInwardNormal(room, wi)
      const gaps = openings
        .filter((o) => o.wall === wi && o.kind === 'door')
        .filter((o) => {
          const mid = add(a, scale(d, o.offset + o.width / 2))
          const beyond = add(mid, scale(n, -(room.wallThickness + 0.2)))
          return rooms.some((r) => r !== room && pointInPolygon(beyond, r.polygon))
        })
        .map((o) => [Math.max(0, o.offset), Math.min(L, o.offset + o.width)] as const)
        .sort((x, y) => x[0] - y[0])
      let cursor = 0
      for (const [g0, g1] of gaps) {
        if (g0 > cursor + 1e-4) segs.push([add(a, scale(d, cursor)), add(a, scale(d, g0))])
        cursor = Math.max(cursor, g1)
      }
      if (L > cursor + 1e-4) segs.push([add(a, scale(d, cursor)), b])
    })
  }
  return segs
}

function closestOnSeg(p: Vec2, [a, b]: WallSeg): Vec2 {
  const ab = sub(b, a)
  const l2 = ab.x * ab.x + ab.y * ab.y
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / l2)) : 0
  return add(a, scale(ab, t))
}

/** Push a point out of every wall it's closer than `r` to (a few passes settle corners). */
function pushOut(p: Vec2, from: Vec2, segs: WallSeg[], r: number): Vec2 {
  let cur = p
  for (let pass = 0; pass < 4; pass++) {
    let moved = false
    for (const s of segs) {
      const c = closestOnSeg(cur, s)
      const dd = dist(cur, c)
      if (dd >= r - 1e-6) continue
      let away: Vec2
      if (dd > 1e-6) away = scale(sub(cur, c), 1 / dd)
      else {
        // Exactly on the face: step back toward the side we came from.
        const d = norm(sub(s[1], s[0]))
        const n = v(-d.y, d.x)
        away = (from.x - c.x) * n.x + (from.y - c.y) * n.y >= 0 ? n : scale(n, -1)
      }
      cur = add(c, scale(away, r))
      moved = true
    }
    if (!moved) break
  }
  return cur
}

function segsCross(p: Vec2, q: Vec2, [a, b]: WallSeg): boolean {
  const cr = (o: Vec2, s: Vec2, t: Vec2) => (s.x - o.x) * (t.y - o.y) - (s.y - o.y) * (t.x - o.x)
  const d1 = cr(a, b, p)
  const d2 = cr(a, b, q)
  const d3 = cr(p, q, a)
  const d4 = cr(p, q, b)
  return d1 * d2 < 0 && d3 * d4 < 0
}

/**
 * Move from `from` by `delta`, sliding along walls instead of passing through them. Long moves are
 * taken in short steps so a fast glide can't jump a wall.
 */
export function walkStep(from: Vec2, delta: Vec2, segs: WallSeg[], r = WALL_GAP): Vec2 {
  const len = Math.hypot(delta.x, delta.y)
  if (len < 1e-9) return from
  const steps = Math.max(1, Math.ceil(len / (r * 0.4)))
  const step = scale(delta, 1 / steps)
  let cur = from
  for (let i = 0; i < steps; i++) {
    const want = add(cur, step)
    // A step that crosses a wall face outright stays on this side of it.
    if (segs.some((s) => segsCross(cur, want, s))) {
      const slid = pushOut(cur, cur, segs, r)
      cur = slid
      break
    }
    cur = pushOut(want, cur, segs, r)
  }
  return cur
}

/** Whether a straight walk from p to q stays clear of the walls (with a little room to spare). */
export function clearPath(p: Vec2, q: Vec2, segs: WallSeg[], r = WALL_GAP * 0.8): boolean {
  const len = dist(p, q)
  const n = Math.max(2, Math.ceil(len / 0.1))
  for (const s of segs) {
    if (segsCross(p, q, s)) return false
    for (let i = 1; i < n; i++) {
      const t = i / n
      const pt = v(p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t)
      if (dist(pt, closestOnSeg(pt, s)) < r) return false
    }
  }
  return true
}

/**
 * Where to walk to reach `to` from `from`: straight there when nothing's in the way, otherwise
 * through one or two doorways (the shortest route found). Returns the points to pass, ending at `to`.
 */
export function routeTo(from: Vec2, tapped: Vec2, rooms: Room[], segs: WallSeg[]): Vec2[] {
  // A spot right by a wall is reached as near as the wall allows.
  const to = pushOut(tapped, from, segs, WALL_GAP)
  if (clearPath(from, to, segs)) return [to]
  // Doorways as pairs of points just either side of the door.
  const shared = sharedDoorways(rooms)
  const doors: [Vec2, Vec2][] = []
  for (const room of rooms) {
    for (const o of [...room.openings, ...(shared.get(room.id) ?? [])]) {
      if (o.kind !== 'door') continue
      const [a, b] = openingEndpoints(room, o)
      const mid = scale(add(a, b), 0.5)
      const n = wallInwardNormal(room, o.wall)
      const depth = room.wallThickness + 0.45
      doors.push([add(mid, scale(n, 0.45)), add(mid, scale(n, -depth))])
    }
  }
  // Each door can be crossed either way.
  const sides: [Vec2, Vec2][] = doors.flatMap(([a, b]) => [[a, b], [b, a]] as [Vec2, Vec2][])
  let best: { len: number; path: Vec2[] } | null = null
  const consider = (path: Vec2[]) => {
    let len = 0
    let prev = from
    for (const p of path) {
      len += dist(prev, p)
      prev = p
    }
    if (!best || len < best.len) best = { len, path }
  }
  for (const [in1, out1] of sides) {
    if (!clearPath(from, in1, segs) || !clearPath(in1, out1, segs, 0.05)) continue
    if (clearPath(out1, to, segs)) consider([in1, out1, to])
    for (const [in2, out2] of sides) {
      if (in2 === out1 || out2 === in1) continue
      if (clearPath(out1, in2, segs) && clearPath(in2, out2, segs, 0.05) && clearPath(out2, to, segs)) consider([in1, out1, in2, out2, to])
    }
  }
  return (best as { len: number; path: Vec2[] } | null)?.path ?? [to]
}
