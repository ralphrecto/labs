import { add, closestPointOnSegment, dot, norm, offsetPolygon, pointInPolygon, polygonArea, rot, scale, signedArea, sub, v } from '../model/geometry'
import type { Vec2 } from '../model/types'
import { backSide, classifyBox } from './classify'
import { levelFloor, quantile, rng, rotateXZ, sample, toCloud, type Cloud } from './cloud'
import type { DetectedObject, DetectedOpening, DetectedRoom, SceneEstimate } from './estimate'
import { Grid, traceOutline } from './grid'
import type { ScanData } from './types'

// On-device scan analysis: level the scan, square it up with the walls, find the room's
// outline, doors and windows, and pull out furniture as sized boxes. It's geometric (no ML),
// so it needs a reasonably complete scan of one room. Results go to the review step, where the
// person fixes what it got wrong.

const CELL = 0.05

export interface AnalyzeOptions {
  maxPoints?: number
  /** Skip leveling when the importer already put the floor at y = 0. */
  level?: boolean
}

export interface AnalyzeResult {
  estimate: SceneEstimate
  /** The leveled cloud the estimate refers to (scan world frame, floor at 0). */
  cloud: Cloud
}

export function analyzeScan(scan: ScanData, opts: AnalyzeOptions = {}): AnalyzeResult {
  const notes: string[] = []
  let cloud = toCloud(scan, opts.maxPoints ?? 600_000)
  if (opts.level !== false) {
    const lv = levelFloor(cloud)
    cloud = lv.cloud
    if (lv.tilt > 1) notes.push(`Leveled the scan (it was tilted ${lv.tilt.toFixed(1)}°).`)
  }
  if (!scan.metric) notes.push("The scan's scale hasn't been set, so sizes may be off. Set it from a measured length first.")

  const ceiling = findCeiling(cloud)
  if (!ceiling.seen) notes.push(`No ceiling in the scan; using ${ceiling.height.toFixed(2)} m as the ceiling height.`)
  const heading = findHeading(cloud, ceiling.height)
  const { px, py } = rotateXZ(cloud, heading)

  const room = findRoom(cloud, px, py, ceiling.height)
  if (!room) notes.push("Couldn't find the room's walls. You can still add the furniture and draw the room by hand.")
  const objects = findObjects(cloud, px, py, room?.polygon ?? null, ceiling.height)

  return {
    cloud,
    estimate: {
      scanId: scan.id,
      detector: 'heuristic',
      transform: { x: 0, y: 0, rotation: heading },
      room,
      objects,
      notes,
    },
  }
}

// ---------------------------------------------------------------- ceiling and heading

function findCeiling(cloud: Cloud): { height: number; seen: boolean } {
  const rand = rng(3)
  const hs = sample(cloud.h, 60_000, rand)
  const top = quantile(hs, 0.995)
  // Histogram of the upper part in 2 cm bins; a ceiling is a sharp layer near the top.
  const bins = new Map<number, number>()
  let total = 0
  for (const h of Array.from(hs)) {
    if (h < 1.9) continue
    const b = Math.round(h / 0.02)
    bins.set(b, (bins.get(b) ?? 0) + 1)
    total++
  }
  let best = 0
  let bestCount = 0
  for (const [b, c] of bins) if (c > bestCount && b * 0.02 > top - 0.4) [best, bestCount] = [b, c]
  const mean = total / Math.max(1, bins.size)
  if (bestCount > 4 * mean && bestCount > 0.004 * hs.length) return { height: Math.round(best * 0.02 * 100) / 100, seen: true }
  return { height: Math.max(2.4, Math.round(Math.min(top + 0.05, 3.2) * 100) / 100), seen: false }
}

/** Rotation (clockwise on the plan) that lines the walls up with the plan axes. */
export function findHeading(cloud: Cloud, ceiling: number): number {
  const rand = rng(5)
  const lo = 1.2
  const hi = Math.max(lo + 0.3, ceiling - 0.15)
  const xs: number[] = []
  const zs: number[] = []
  const stride = Math.max(1, Math.floor(cloud.n / 150_000))
  for (let i = 0; i < cloud.n; i += stride) {
    if (cloud.h[i] > lo && cloud.h[i] < hi) {
      xs.push(cloud.x[i])
      zs.push(cloud.z[i])
    }
  }
  if (xs.length < 100) return 0
  const k = Math.min(xs.length, 40_000)
  const sx = new Float32Array(k)
  const sz = new Float32Array(k)
  for (let i = 0; i < k; i++) {
    const j = xs.length > k ? Math.floor(rand() * xs.length) : i
    sx[i] = xs[j]
    sz[i] = zs[j]
  }
  // Walls make sharp peaks in the x and y histograms when they're axis aligned.
  const score = (theta: number) => {
    const c = Math.cos(theta)
    const s = Math.sin(theta)
    const hx = new Map<number, number>()
    const hy = new Map<number, number>()
    for (let i = 0; i < k; i++) {
      const bx = Math.floor((sx[i] * c - sz[i] * s) / 0.04)
      const by = Math.floor((sx[i] * s + sz[i] * c) / 0.04)
      hx.set(bx, (hx.get(bx) ?? 0) + 1)
      hy.set(by, (hy.get(by) ?? 0) + 1)
    }
    let e = 0
    for (const c2 of hx.values()) e += c2 * c2
    for (const c2 of hy.values()) e += c2 * c2
    return e
  }
  const DEG = Math.PI / 180
  let best = 0
  let bestScore = -1
  for (let a = 0; a < 90; a += 1) {
    const sc = score(a * DEG)
    if (sc > bestScore) [best, bestScore] = [a * DEG, sc]
  }
  for (let a = -1; a <= 1; a += 0.1) {
    const sc = score(best + a * DEG)
    if (sc > bestScore) [best, bestScore] = [best + a * DEG, sc]
  }
  // Prefer the smallest rotation among the four equivalent ones.
  if (best > 45 * DEG) best -= 90 * DEG
  return best
}

// ---------------------------------------------------------------- room outline

interface Frame {
  cell: number
  ox: number
  oy: number
  w: number
  h: number
}

function frameFor(px: Float32Array, py: Float32Array, cell = CELL): Frame {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (let i = 0; i < px.length; i++) {
    if (px[i] < minX) minX = px[i]
    if (px[i] > maxX) maxX = px[i]
    if (py[i] < minY) minY = py[i]
    if (py[i] > maxY) maxY = py[i]
  }
  const m = 0.3
  const ox = Math.floor((minX - m) / cell) * cell
  const oy = Math.floor((minY - m) / cell) * cell
  return { cell, ox, oy, w: Math.ceil((maxX - ox + m) / cell), h: Math.ceil((maxY - oy + m) / cell) }
}

export function findRoom(cloud: Cloud, px: Float32Array, py: Float32Array, ceiling: number): DetectedRoom | null {
  if (cloud.n < 200) return null
  const f = frameFor(px, py)
  if (f.w * f.h > 4_000_000) return null // more than 100 m across: not one room
  const floorCnt = new Uint32Array(f.w * f.h)
  const lowCnt = new Uint32Array(f.w * f.h)
  const highCnt = new Uint32Array(f.w * f.h)
  // Wall evidence comes from above the furniture. If the scan doesn't reach that high, use mid height.
  let above = 0
  for (let i = 0; i < cloud.n; i++) if (cloud.h[i] > 2.05 && cloud.h[i] < ceiling - 0.08) above++
  const wallLo = above > 0.02 * cloud.n ? 2.05 : 1.3
  const wallHi = ceiling - 0.08
  for (let i = 0; i < cloud.n; i++) {
    const ci = Math.floor((px[i] - f.ox) / CELL)
    const cj = Math.floor((py[i] - f.oy) / CELL)
    if (ci < 0 || cj < 0 || ci >= f.w || cj >= f.h) continue
    const k = cj * f.w + ci
    const hh = cloud.h[i]
    if (hh > -0.04 && hh < 0.04) floorCnt[k]++
    // Anything below wall height is floor or furniture standing on it, so it's inside the room.
    else if (hh >= 0.04 && hh < wallLo) lowCnt[k]++
    if (hh > wallLo && hh < wallHi) highCnt[k]++
  }
  // A wall cell has a lot of points above furniture height, relative to typical wall density.
  const nz: number[] = []
  for (const c of highCnt) if (c > 0) nz.push(c)
  if (!nz.length) return null
  const wallThresh = Math.max(2, quantile(nz, 0.6) * 0.25)
  const floorNz: number[] = []
  for (const c of floorCnt) if (c > 0) floorNz.push(c)
  const floorThresh = Math.max(1, quantile(floorNz, 0.5) * 0.1)

  const walls = new Grid(f.w, f.h)
  const known = new Grid(f.w, f.h)
  for (let k = 0; k < walls.data.length; k++) {
    if (highCnt[k] >= wallThresh) walls.data[k] = 1
    if (floorCnt[k] >= floorThresh || lowCnt[k] >= floorThresh) known.data[k] = 1
  }
  // Floor, closed over small gaps (under furniture, sparse spots), minus the walls.
  let region = known.close(4).andNot(walls.dilate(0))
  // Cut thin connections (doorways into the next room, a hallway seen through a door): open with a
  // ~90 cm square, keep the biggest piece, and grow it back within the region.
  const core = region.erode(9).largest()
  if (!core.count()) return null
  region = core.dilate(9).and(region).largest().fillHoles()
  // Small closing smooths ragged edges left by furniture against the walls.
  region = region.close(2).andNot(walls).largest().fillHoles()

  const corners = traceOutline(region).map((p) => v(f.ox + p.x * CELL, f.oy + p.y * CELL))
  if (corners.length < 4) return null
  let poly = simplifyRectilinear(corners, 0.3)
  if (poly.length < 4 || polygonArea(poly) < 1.5) return null

  poly = snapToWalls(poly, cloud, px, py, wallLo, wallHi)
  poly = simplifyRectilinear(poly, 0.2)
  // Clockwise on screen, like rooms drawn by hand, and start at the top-left corner.
  if (signedArea(poly) < 0) poly = poly.reverse()
  const startAt = poly.reduce((bi, p, i) => (p.x + p.y < poly[bi].x + poly[bi].y ? i : bi), 0)
  poly = [...poly.slice(startAt), ...poly.slice(0, startAt)].map((p) => v(round(p.x), round(p.y)))

  const openings = findOpenings(cloud, px, py, poly, ceiling)
  // Confidence: how much of the outline is backed by wall points.
  return { polygon: poly, ceilingHeight: ceiling, openings, confidence: wallCoverage(poly, walls, f) }
}

const round = (x: number) => Math.round(x * 1000) / 1000

interface Edge {
  horiz: boolean
  c: number
}

function toEdges(poly: Vec2[]): Edge[] {
  return poly.map((p, k) => {
    const q = poly[(k + 1) % poly.length]
    const horiz = Math.abs(q.y - p.y) < Math.abs(q.x - p.x)
    return { horiz, c: horiz ? (p.y + q.y) / 2 : (p.x + q.x) / 2 }
  })
}

function fromEdges(edges: Edge[]): Vec2[] {
  const n = edges.length
  return edges.map((e, k) => {
    const prev = edges[(k - 1 + n) % n]
    return e.horiz ? v(prev.c, e.c) : v(e.c, prev.c)
  })
}

/** Remove jogs shorter than minLen from an axis-aligned polygon, keeping it axis-aligned. */
export function simplifyRectilinear(poly: Vec2[], minLen: number): Vec2[] {
  let edges = toEdges(poly)
  // Drop collinear repeats first (two parallel edges in a row).
  edges = edges.filter((e, k) => e.horiz !== edges[(k + 1) % edges.length].horiz)
  for (let guard = 0; guard < 500 && edges.length > 4; guard++) {
    const pts = fromEdges(edges)
    const n = edges.length
    const lens = pts.map((p, k) => Math.hypot(pts[(k + 1) % n].x - p.x, pts[(k + 1) % n].y - p.y))
    let k = -1
    for (let i = 0; i < n; i++) if (lens[i] < minLen && (k < 0 || lens[i] < lens[k])) k = i
    if (k < 0) break
    const a = (k - 1 + n) % n
    const b = (k + 1) % n
    const merged = lens[a] >= lens[b] ? edges[a].c : edges[b].c
    const next: Edge[] = []
    for (let s = 1; s <= n - 2; s++) next.push({ ...edges[(b + s) % n] })
    next[next.length - 1].c = merged
    // Merging can leave two parallel edges next to each other; fold them.
    edges = next.filter((e, i) => e.horiz !== next[(i + 1) % next.length].horiz)
  }
  return fromEdges(edges)
}

/** Move each wall onto the wall surface the points show (the grid outline is only 5 cm accurate). */
function snapToWalls(poly: Vec2[], cloud: Cloud, px: Float32Array, py: Float32Array, lo: number, hi: number): Vec2[] {
  const edges = toEdges(poly)
  const n = poly.length
  const wx: number[] = []
  const wy: number[] = []
  const stride = Math.max(1, Math.floor(cloud.n / 300_000))
  for (let i = 0; i < cloud.n; i += stride) {
    if (cloud.h[i] > lo && cloud.h[i] < hi) {
      wx.push(px[i])
      wy.push(py[i])
    }
  }
  const R = 0.2
  const out = edges.map((e, k) => {
    const p = poly[k]
    const q = poly[(k + 1) % n]
    const a0 = e.horiz ? Math.min(p.x, q.x) : Math.min(p.y, q.y)
    const a1 = e.horiz ? Math.max(p.x, q.x) : Math.max(p.y, q.y)
    const trim = Math.min(0.15, (a1 - a0) / 4)
    const bins = new Float32Array(Math.round((2 * R) / 0.01) + 1)
    let total = 0
    for (let i = 0; i < wx.length; i++) {
      const along = e.horiz ? wx[i] : wy[i]
      const across = e.horiz ? wy[i] : wx[i]
      if (along < a0 + trim || along > a1 - trim) continue
      const d = across - e.c
      if (d < -R || d > R) continue
      bins[Math.round((d + R) / 0.01)]++
      total++
    }
    if (total < 30) return e
    let best = 1
    let bestS = -1
    for (let b = 1; b < bins.length - 1; b++) {
      const sm = bins[b - 1] + 2 * bins[b] + bins[b + 1]
      if (sm > bestS) [best, bestS] = [b, sm]
    }
    const mean = total / bins.length
    if (bestS / 4 < 3 * mean) return e
    return { ...e, c: e.c + (best * 0.01 - R) }
  })
  return fromEdges(out)
}

function wallCoverage(poly: Vec2[], walls: Grid, f: Frame) {
  let hit = 0
  let total = 0
  for (let k = 0; k < poly.length; k++) {
    const a = poly[k]
    const b = poly[(k + 1) % poly.length]
    const L = Math.hypot(b.x - a.x, b.y - a.y)
    const d = norm(sub(b, a))
    for (let t = 0.05; t < L - 0.05; t += CELL) {
      const p = add(a, scale(d, t))
      total++
      const ci = Math.floor((p.x - f.ox) / CELL)
      const cj = Math.floor((p.y - f.oy) / CELL)
      let found = false
      for (let dj = -2; dj <= 2 && !found; dj++) for (let di = -2; di <= 2 && !found; di++) if (walls.get(ci + di, cj + dj)) found = true
      if (found) hit++
    }
  }
  return total ? Math.round((hit / total) * 100) / 100 : 0
}

// ---------------------------------------------------------------- doors and windows

/** Inward normal for a clockwise-on-screen polygon edge (y down): the right-hand side. */
function inward(poly: Vec2[], k: number) {
  const a = poly[k]
  const b = poly[(k + 1) % poly.length]
  const d = norm(sub(b, a))
  const right = v(-d.y, d.x)
  return signedArea(poly) > 0 ? right : scale(right, -1)
}

export function findOpenings(cloud: Cloud, px: Float32Array, py: Float32Array, poly: Vec2[], ceiling: number): DetectedOpening[] {
  const out: DetectedOpening[] = []
  const BIN = 0.05
  for (let k = 0; k < poly.length; k++) {
    const a = poly[k]
    const b = poly[(k + 1) % poly.length]
    const L = Math.hypot(b.x - a.x, b.y - a.y)
    if (L < 0.6) continue
    const d = norm(sub(b, a))
    const nIn = inward(poly, k)
    const nb = Math.ceil(L / BIN)
    const low = new Float32Array(nb)
    const mid = new Float32Array(nb)
    const front = new Float32Array(nb)
    const lowMax: number[][] = Array.from({ length: nb }, () => [])
    const highMin: number[][] = Array.from({ length: nb }, () => [])
    for (let i = 0; i < cloud.n; i++) {
      const rx = px[i] - a.x
      const ry = py[i] - a.y
      const t = rx * d.x + ry * d.y
      if (t < 0 || t >= L) continue
      const inside = rx * nIn.x + ry * nIn.y
      const hh = cloud.h[i]
      const bi = Math.floor(t / BIN)
      if (inside > -0.2 && inside < 0.06) {
        // On (or just behind) the wall surface.
        if (hh > 0.15 && hh < 0.85) low[bi]++
        else if (hh > 0.95 && hh < 1.85) mid[bi]++
        if (hh > 0.15 && hh < 1.4) lowMax[bi].push(hh)
        if (hh > 1.2 && hh < ceiling - 0.05) highMin[bi].push(hh)
      } else if (inside >= 0.12 && inside < 0.8 && hh > 1.2 && hh < 1.9) {
        // Something tall just inside the wall (a wardrobe hides the wall the way a doorway would).
        front[bi]++
      }
    }
    const refLow = quantile(low, 0.75)
    const refMid = quantile(mid, 0.75)
    if (refMid < 2) continue // no wall evidence on this side
    const hasLow = Array.from(low, (c) => c >= Math.max(1, 0.15 * refLow))
    const hasMid = Array.from(mid, (c) => c >= Math.max(1, 0.15 * refMid))
    const clearFront = Array.from(front, (c) => c <= Math.max(2, 0.1 * refMid))
    const runs = (pred: (i: number) => boolean) => {
      const r: [number, number][] = []
      let s = -1
      for (let i = 0; i <= nb; i++) {
        const on = i < nb && pred(i)
        if (on && s < 0) s = i
        if (!on && s >= 0) {
          // Bridge one-bin blips.
          if (r.length && s - r[r.length - 1][1] <= 1) r[r.length - 1][1] = i
          else r.push([s, i])
          s = -1
        }
      }
      return r
    }
    for (const [s, e] of runs((i) => !hasLow[i] && !hasMid[i])) {
      const width = (e - s) * BIN
      if (width < 0.6 || width > 1.4 || s === 0 || e === nb) continue
      let clear = 0
      for (let i = s; i < e; i++) if (clearFront[i]) clear++
      if (clear / (e - s) < 0.6) continue // a wardrobe or shelf in front, not a door
      const tops = highMin.slice(s, e).flat()
      const head = tops.length > 10 ? quantile(tops, 0.05) : 2.03
      out.push({ kind: 'door', wall: k, offset: round(s * BIN), width: round(width), height: round(Math.min(Math.max(head, 1.9), ceiling - 0.02)), sill: 0, confidence: 0.6 })
    }
    for (const [s, e] of runs((i) => hasLow[i] && !hasMid[i])) {
      const width = (e - s) * BIN
      if (width < 0.4 || width > 3.0) continue
      const lows = lowMax.slice(s, e).flat()
      const highs = highMin.slice(s, e).flat().filter((x) => x > 1.5)
      const sill = lows.length > 10 ? Math.min(1.3, Math.max(0.3, quantile(lows, 0.95))) : 0.9
      const top = highs.length > 10 ? quantile(highs, 0.05) : sill + 1.2
      out.push({ kind: 'window', wall: k, offset: round(s * BIN), width: round(width), height: round(Math.max(0.3, top - sill)), sill: round(sill), confidence: 0.4 })
    }
  }
  return out
}

// ---------------------------------------------------------------- furniture

export function findObjects(cloud: Cloud, px: Float32Array, py: Float32Array, poly: Vec2[] | null, ceiling: number): DetectedObject[] {
  const top = Math.min(2.4, ceiling - 0.2)
  const inset = poly ? offsetPolygon(poly, -0.07) : null
  const sel: number[] = []
  for (let i = 0; i < cloud.n; i++) {
    const hh = cloud.h[i]
    if (hh < 0.05 || hh > top) continue
    if (inset && !pointInPolygon(v(px[i], py[i]), inset)) continue
    sel.push(i)
  }
  if (sel.length < 20) return []
  const sx = new Float32Array(sel.length)
  const sy = new Float32Array(sel.length)
  for (let k = 0; k < sel.length; k++) {
    sx[k] = px[sel[k]]
    sy[k] = py[sel[k]]
  }
  // 5 cm cells, or 10 cm for a sparse scan where 5 cm cells would leave holes in tabletops.
  const binPoints = (cell: number) => {
    const f = frameFor(sx, sy, cell)
    const cnt = new Uint32Array(f.w * f.h)
    const hmax = new Float32Array(f.w * f.h)
    const cellOf = new Int32Array(sel.length)
    for (let k = 0; k < sel.length; k++) {
      const c = Math.floor((sy[k] - f.oy) / cell) * f.w + Math.floor((sx[k] - f.ox) / cell)
      cellOf[k] = c
      cnt[c]++
      const hh = cloud.h[sel[k]]
      if (hh > hmax[c]) hmax[c] = hh
    }
    return { f, cnt, hmax, cellOf }
  }
  let binned = binPoints(CELL)
  if (quantile(binned.cnt.filter((c) => c > 0), 0.5) < 1.5) binned = binPoints(2 * CELL)
  const { f, cnt, hmax, cellOf } = binned
  const nz: number[] = []
  for (const c of cnt) if (c > 0) nz.push(c)
  const minCnt = Math.max(2, Math.round(quantile(nz, 0.5) * 0.1))
  const occ = new Grid(f.w, f.h)
  for (let c = 0; c < cnt.length; c++) if (cnt[c] >= minCnt) occ.data[c] = 1

  // Union-find over occupied cells. Neighbors join unless their heights differ a lot (a desk next
  // to a tall shelf). Cells with few points (slivers at an object's edge, stray points) can join a
  // piece but never link two pieces together.
  const perCell = quantile(nz, 0.5)
  const strong = (c: number) => cnt[c] >= 0.3 * perCell
  const parent = new Int32Array(f.w * f.h).map((_, i) => i)
  const findP = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]]
      i = parent[i]
    }
    return i
  }
  const unite = (a: number, b: number) => {
    const ra = findP(a)
    const rb = findP(b)
    if (ra !== rb) parent[ra] = rb
  }
  const NB = [[1, 0], [0, 1], [1, 1], [-1, 1], [-1, 0], [0, -1], [-1, -1], [1, -1]]
  for (let j = 0; j < f.h; j++)
    for (let i = 0; i < f.w; i++) {
      const c = j * f.w + i
      if (!occ.data[c] || !strong(c)) continue
      for (const [di, dj] of NB.slice(0, 4)) {
        const ii = i + di
        const jj = j + dj
        if (ii < 0 || jj < 0 || ii >= f.w || jj >= f.h) continue
        const c2 = jj * f.w + ii
        if (occ.data[c2] && strong(c2) && Math.abs(hmax[c] - hmax[c2]) < 0.6) unite(c, c2)
      }
    }
  for (let j = 0; j < f.h; j++)
    for (let i = 0; i < f.w; i++) {
      const c = j * f.w + i
      if (!occ.data[c] || strong(c)) continue
      let bestN = -1
      for (const [di, dj] of NB) {
        const ii = i + di
        const jj = j + dj
        if (ii < 0 || jj < 0 || ii >= f.w || jj >= f.h) continue
        const c2 = jj * f.w + ii
        if (occ.data[c2] && strong(c2) && (bestN < 0 || cnt[c2] > cnt[bestN])) bestN = c2
      }
      if (bestN >= 0) unite(c, bestN)
      else occ.data[c] = 0
    }
  // A tall piece standing next to a low one can still end up joined through its edge cells
  // (a shelf beside a sofa). Split pieces that have both a sizable tall part and a sizable low part.
  const label = new Int32Array(f.w * f.h)
  const cellsByRoot = new Map<number, number[]>()
  for (let c = 0; c < occ.data.length; c++) {
    if (!occ.data[c]) continue
    const r = findP(c)
    const l = cellsByRoot.get(r)
    if (l) l.push(c)
    else cellsByRoot.set(r, [c])
  }
  const cellArea = f.cell * f.cell
  let nextLabel = 1
  for (const cells of cellsByRoot.values()) {
    const tall = cells.filter((c) => hmax[c] >= 1.15)
    const low = cells.filter((c) => hmax[c] < 1.15)
    if (tall.length * cellArea < 0.06 || low.length * cellArea < 0.15) {
      const l = nextLabel++
      for (const c of cells) label[c] = l
      continue
    }
    for (const part of [tall, low]) {
      // Connected pieces of this part; slivers go to the other part's label later.
      const inPart = new Set(part)
      for (const c0 of part) {
        if (label[c0]) continue
        const l = nextLabel++
        const stack = [c0]
        label[c0] = l
        while (stack.length) {
          const c = stack.pop()!
          const i = c % f.w
          const j = (c - i) / f.w
          for (const [di, dj] of NB) {
            const ii = i + di
            const jj = j + dj
            if (ii < 0 || jj < 0 || ii >= f.w || jj >= f.h) continue
            const c2 = jj * f.w + ii
            if (inPart.has(c2) && !label[c2]) {
              label[c2] = l
              stack.push(c2)
            }
          }
        }
      }
    }
  }
  // Gather points per component.
  const comps = new Map<number, number[]>()
  for (let k = 0; k < sel.length; k++) {
    const c = cellOf[k]
    if (!occ.data[c]) continue
    const r = label[c]
    const l = comps.get(r)
    if (l) l.push(k)
    else comps.set(r, [k])
  }
  const rand = rng(11)
  const objects: DetectedObject[] = []
  let n = 0
  for (const members of comps.values()) {
    if (members.length < 30) continue
    // Sparse specks (stray points, slivers of a neighbor) have far fewer points per cell than real surfaces.
    const cells = new Set(members.map((k) => cellOf[k])).size
    if (members.length / cells < 0.35 * perCell) continue
    const pts = members.length > 4000 ? Array.from({ length: 4000 }, () => members[Math.floor(rand() * members.length)]) : members
    const box = orientedBox(pts.map((k) => v(sx[k], sy[k])))
    if (!box) continue
    const heights = pts.map((k) => cloud.h[sel[k]])
    const height = quantile(heights, 0.98)
    const [w0, d0] = [box.size.x, box.size.y]
    if (height < 0.12 || w0 * d0 < 0.04 || Math.min(w0, d0) < 0.12) continue
    let color: string | undefined
    if (cloud.rgb) {
      let r = 0
      let g = 0
      let b = 0
      for (const k of pts) {
        const i = sel[k]
        r += cloud.rgb[i * 3]
        g += cloud.rgb[i * 3 + 1]
        b += cloud.rgb[i * 3 + 2]
      }
      color = toHex(r / pts.length, g / pts.length, b / pts.length)
    }
    let cls = classifyBox(w0, d0, height)
    let placed = orient(box, cls.category, poly)
    // Pushing it flush to the wall changes its size a little; check the category again.
    const again = classifyBox(placed.width, placed.depth, height)
    if (again.category !== cls.category) {
      cls = again
      placed = orient(box, cls.category, poly)
    }
    objects.push({
      id: `det-${++n}`,
      category: cls.category,
      name: cls.name,
      confidence: cls.confidence,
      alternatives: cls.alternatives,
      x: round(placed.center.x),
      y: round(placed.center.y),
      rotation: placed.rotation,
      width: round(placed.width),
      depth: round(placed.depth),
      height: round(height),
      color,
      pointCount: members.length,
      againstWall: placed.againstWall,
    })
  }
  // Biggest first: that's roughly the order people check them in.
  return objects.sort((a, b) => b.width * b.depth - a.width * a.depth)
}

const toHex = (r: number, g: number, b: number) =>
  '#' + [r, g, b].map((c) => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, '0')).join('')

export interface Box2 {
  center: Vec2
  /** Angle of the box's first axis (clockwise on the plan). */
  angle: number
  /** Extent along the first and second axes. */
  size: Vec2
}

/** Minimum-area rectangle around points, preferring wall-aligned (0°) when it's nearly as tight. */
export function orientedBox(pts: Vec2[]): Box2 | null {
  if (pts.length < 3) return null
  const DEG = Math.PI / 180
  const extent = (theta: number) => {
    const us: number[] = []
    const ws: number[] = []
    const c = Math.cos(theta)
    const s = Math.sin(theta)
    for (const p of pts) {
      // Coordinates along the rotated axes.
      us.push(p.x * c + p.y * s)
      ws.push(-p.x * s + p.y * c)
    }
    us.sort((a, b) => a - b)
    ws.sort((a, b) => a - b)
    const q = (a: number[], t: number) => a[Math.min(a.length - 1, Math.max(0, Math.round(t * (a.length - 1))))]
    const u0 = q(us, 0.005)
    const u1 = q(us, 0.995)
    const w0 = q(ws, 0.005)
    const w1 = q(ws, 0.995)
    return { u0, u1, w0, w1, area: (u1 - u0) * (w1 - w0) }
  }
  let best = { theta: 0, e: extent(0) }
  const aligned = best
  for (let a = 2; a < 90; a += 2) {
    const e = extent(a * DEG)
    if (e.area < best.e.area) best = { theta: a * DEG, e }
  }
  if (aligned.e.area <= best.e.area * 1.12) best = aligned
  const { theta, e } = best
  const cu = (e.u0 + e.u1) / 2
  const cw = (e.w0 + e.w1) / 2
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  return { center: v(cu * c - cw * s, cu * s + cw * c), angle: theta, size: v(e.u1 - e.u0, e.w1 - e.w0) }
}

/**
 * Decide which way a box faces. Its back goes against the nearest wall (a long side for a sofa,
 * a short side for a bed's headboard); free-standing pieces face the room's middle.
 */
export function orient(box: Box2, category: string, poly: Vec2[] | null) {
  const e1 = rot(v(1, 0), box.angle)
  const e2 = rot(v(0, 1), box.angle)
  const sides = [
    { n: e1, along: box.size.y, across: box.size.x },
    { n: scale(e1, -1), along: box.size.y, across: box.size.x },
    { n: e2, along: box.size.x, across: box.size.y },
    { n: scale(e2, -1), along: box.size.x, across: box.size.y },
  ]
  const want = backSide(category)
  const longSide = Math.max(box.size.x, box.size.y)
  const candidates = sides.filter((sd) => {
    if (want === 'any' || Math.abs(box.size.x - box.size.y) < 0.05) return true
    const isLong = sd.along >= longSide - 1e-9
    return want === 'long' ? isLong : !isLong
  })
  let back = null as null | (typeof sides)[number]
  let bestGap = Infinity
  if (poly) {
    for (const sd of candidates) {
      const mid = add(box.center, scale(sd.n, sd.across / 2))
      for (let k = 0; k < poly.length; k++) {
        const a = poly[k]
        const b = poly[(k + 1) % poly.length]
        const wd = norm(sub(b, a))
        if (Math.abs(dot(wd, sd.n)) > 0.3) continue // wall not parallel to this side
        const { distance, t } = closestPointOnSegment(mid, a, b)
        if (t <= 0 || t >= 1) continue
        if (distance < bestGap) [bestGap, back] = [distance, sd]
      }
    }
  }
  const againstWall = !!back && bestGap < 0.3
  if (!againstWall) {
    // Face the room's middle (or the scan origin without a room).
    const mid = poly ? polyCenter(poly) : v(0, 0)
    const toMid = sub(mid, box.center)
    back = candidates.reduce((p, q) => (dot(q.n, toMid) < dot(p.n, toMid) ? q : p))
  }
  const front = scale(back!.n, -1)
  // Points right at the wall are dropped as wall, so a piece against a wall comes out a few cm
  // short of it. Close that gap: furniture against a wall is pushed flush.
  let center = box.center
  let across = back!.across
  if (againstWall && bestGap < 0.12) {
    across += bestGap
    center = add(center, scale(back!.n, bestGap / 2))
  }
  // Front is local +y: rotation theta has front (-sin θ, cos θ).
  let rotation = Math.atan2(-front.x, front.y)
  rotation = ((rotation % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
  rotation = Math.round(rotation * 1e4) / 1e4
  return { center, rotation, width: back!.along, depth: across, againstWall }
}

function polyCenter(poly: Vec2[]) {
  const s = poly.reduce((acc, p) => add(acc, p), v(0, 0))
  return scale(s, 1 / poly.length)
}
