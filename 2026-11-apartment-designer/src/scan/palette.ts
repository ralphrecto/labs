// Colors taken from the real apartment scan, for the styled 3D views: what the floor looks like
// from above (a small top-down color image) and the paint color of every wall. The Cartoon style
// posterizes these into flat, bright bands; the Lifelike style uses them as they are.
//
// Input is scan points in the app world frame (meters, Y up, floor at 0, plan (x, y) = world (x, z)),
// which is what `toScanData` returns.

import { closestPointOnSegment, offsetPolygon, pointInPolygon, wallInwardNormal, wallSegments } from '../model/geometry'
import type { Room } from '../model/types'
import type { ScanPoints } from './types'

/** A top-down color image of the floor. Pixel (i, j) covers plan x0 + i*cell, y0 + j*cell. */
export interface FloorImage {
  x0: number
  y0: number
  cell: number
  width: number
  height: number
  /** RGBA, row-major, alpha 0 where nothing was seen. */
  rgba: Uint8ClampedArray
}

export interface ScanPalette {
  floor: FloorImage | null
  /** Per room id: one color per wall (same order as wallSegments), undefined where the scan saw too little. */
  walls: Record<string, (string | undefined)[]>
  /** Per room id: the typical wall color, used for walls the scan missed. */
  wallColor: Record<string, string | undefined>
  /** Per room id: the typical floor color. */
  floorColor: Record<string, string | undefined>
}

const FLOOR_BAND = 0.06
const WALL_DEPTH = 0.22
const MIN_SAMPLES = 25

const hex = (r: number, g: number, b: number) =>
  '#' + [r, g, b].map((c) => Math.round(Math.max(0, Math.min(1, c)) * 255).toString(16).padStart(2, '0')).join('')

/** Per-channel median: robust to a picture frame or a lamp in front of a wall. */
function median3(samples: number[]): [number, number, number] | null {
  const n = samples.length / 3
  if (n < MIN_SAMPLES) return null
  const out: [number, number, number] = [0, 0, 0]
  for (let c = 0; c < 3; c++) {
    const ch = new Float32Array(n)
    for (let i = 0; i < n; i++) ch[i] = samples[i * 3 + c]
    ch.sort()
    out[c] = ch[Math.floor(n / 2)]
  }
  return out
}

export interface PaletteOptions {
  /** Plan cell size of the floor image (m). Left out, it follows the scan's density: 5 cm, finer where floor points are dense. */
  cell?: number
  /** Paint over rug-shaped patches with the boards around them (the rug is a piece of furniture now). */
  eraseRugs?: boolean
}

/** Floor points this close to a room's walls are left out: they're mostly the foot of the wall (m). */
const EDGE_INSET = 0.15

export function scanPalette(scans: ScanPoints[], rooms: Room[], opts: PaletteOptions | number = {}): ScanPalette {
  const o: PaletteOptions = typeof opts === 'number' ? { cell: opts } : opts
  const palette: ScanPalette = { floor: null, walls: {}, wallColor: {}, floorColor: {} }
  const colored = scans.filter((s) => s.colors && s.count > 0)
  if (!colored.length) return palette

  // ---- wall and floor sampling
  const segs = rooms.map((r) => wallSegments(r))
  const normals = rooms.map((r, ri) => segs[ri].map((_, i) => wallInwardNormal(r, i)))
  const wallSamples = segs.map((rs) => rs.map(() => [] as number[]))
  const floorSamples = rooms.map(() => [] as number[])
  const cap = 6000
  // The floor image only takes points well inside a room: along the walls a scan sees the wall's
  // foot (pale paint, skirting) at floor height, which made a ragged pale band. The edge is filled
  // from the floor further in instead.
  const insets = rooms.map((r) => offsetPolygon(r.polygon, -EDGE_INSET))
  const keep = (x: number, y: number) => !rooms.length || insets.some((q) => pointInPolygon({ x, y }, q))
  const fp: number[] = []

  for (const s of colored) {
    const p = s.positions
    const c = s.colors!
    for (let i = 0; i < s.count; i++) {
      const x = p[i * 3]
      const hgt = p[i * 3 + 1]
      const y = p[i * 3 + 2]
      const r = c[i * 3]
      const g = c[i * 3 + 1]
      const b = c[i * 3 + 2]
      if (hgt <= FLOOR_BAND && hgt >= -FLOOR_BAND) {
        if (keep(x, y)) fp.push(x, y, hgt, r, g, b)
        // Every 7th floor point is plenty for the room's typical color.
        if (i % 7 === 0) {
          for (let ri = 0; ri < rooms.length; ri++) {
            if (floorSamples[ri].length < cap * 3 && pointInPolygon({ x, y }, rooms[ri].polygon)) {
              floorSamples[ri].push(r, g, b)
              break
            }
          }
        }
        continue
      }
      for (let ri = 0; ri < rooms.length; ri++) {
        const room = rooms[ri]
        if (hgt < 0.35 || hgt > Math.min(2.1, room.ceilingHeight - 0.2)) continue
        const rs = segs[ri]
        for (let wi = 0; wi < rs.length; wi++) {
          const bucket = wallSamples[ri][wi]
          if (bucket.length >= cap * 3) continue
          const [a, bb] = rs[wi]
          const q = closestPointOnSegment({ x, y }, a, bb)
          if (q.t < 0.04 || q.t > 0.96) continue
          const n = normals[ri][wi]
          const inward = (x - q.point.x) * n.x + (y - q.point.y) * n.y
          if (inward > WALL_DEPTH || inward < -0.1) continue
          bucket.push(r, g, b)
        }
      }
    }
  }

  rooms.forEach((room, ri) => {
    const all: number[] = []
    palette.walls[room.id] = wallSamples[ri].map((smp) => {
      for (let i = 0; i < smp.length; i += 9) all.push(smp[i], smp[i + 1], smp[i + 2])
      const m = median3(smp)
      return m ? hex(...m) : undefined
    })
    const mw = median3(all)
    palette.wallColor[room.id] = mw ? hex(...mw) : undefined
    const mf = median3(floorSamples[ri])
    palette.floorColor[room.id] = mf ? hex(...mf) : undefined
  })

  const pts = Float32Array.from(fp)
  const cell = o.cell ?? floorCell(pts)
  // The image covers the rooms too, so their edges can be filled out to the walls.
  const extra = rooms.flatMap((r) => r.polygon)
  const top = floorTop(pts, cell, extra)
  if (top) {
    // Cells the fill may reach: inside a room, plus a cell's margin so the texture's edge pixels aren't blank.
    let inside: Uint8Array | undefined
    if (rooms.length) {
      const grown = rooms.map((r) => offsetPolygon(r.polygon, cell * 1.5))
      inside = new Uint8Array(top.w * top.h)
      for (let j = 0; j < top.h; j++)
        for (let i = 0; i < top.w; i++) {
          const q = { x: top.x0 + (i + 0.5) * cell, y: top.y0 + (j + 0.5) * cell }
          if (grown.some((g) => pointInPolygon(q, g))) inside[j * top.w + i] = 1
        }
    }
    if (o.eraseRugs) {
      for (const rug of findRugPatches(top)) for (const k of grow([...rug.cells, ...rug.rim], top.w, top.h, 2)) {
        top.count[k] = 0
        top.seen[k] = 0
      }
    }
    palette.floor = finishFloor({ ...top, inside }, top.x0, top.y0, cell)
  }
  return palette
}

/** A floor cell size that leaves about 8 points per cell: 5 cm for a typical phone scan, down to 2.5 cm for a dense one. */
function floorCell(pts: Float32Array) {
  const n = pts.length / 6
  if (n < 100) return 0.05
  const occupied = new Set<number>()
  for (let i = 0; i < n; i++) occupied.add(Math.floor(pts[i * 6] / 0.1) * 100003 + Math.floor(pts[i * 6 + 1] / 0.1))
  const density = n / (occupied.size * 0.01)
  const c = Math.sqrt(8 / density)
  return Math.round(Math.max(0.025, Math.min(0.05, c)) * 200) / 200
}

/** Cells within r (8-neighbourhood steps) of the given ones. */
function grow(cells: number[], w: number, h: number, r: number): Set<number> {
  const out = new Set(cells)
  for (const k of cells) {
    const x = k % w
    const y = (k - x) / w
    for (let dy = -r; dy <= r; dy++)
      for (let dx = -r; dx <= r; dx++) {
        const xx = x + dx
        const yy = y + dy
        if (xx >= 0 && yy >= 0 && xx < w && yy < h) out.add(yy * w + xx)
      }
  }
  return out
}

/** The floor's top-layer cells. `pts` is (x, y, height, r, g, b) per point; `extra` widens the bounds. */
export function floorTop(pts: Float32Array, cell: number, extra: { x: number; y: number }[] = []) {
  const n = pts.length / 6
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (let i = 0; i < n; i++) {
    const x = pts[i * 6]
    const y = pts[i * 6 + 1]
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  if (!(maxX > minX && maxY > minY)) return null
  for (const p of extra) {
    minX = Math.min(minX, p.x - cell)
    minY = Math.min(minY, p.y - cell)
    maxX = Math.max(maxX, p.x + cell)
    maxY = Math.max(maxY, p.y + cell)
  }
  const w = Math.min(1024, Math.ceil((maxX - minX) / cell) + 1)
  const h = Math.min(1024, Math.ceil((maxY - minY) / cell) + 1)
  // Floor points bucketed by cell (counted first, then filled), so each cell can keep its top layer.
  const count = new Uint32Array(w * h)
  const cellOf = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const px = Math.floor((pts[i * 6] - minX) / cell)
    const py = Math.floor((pts[i * 6 + 1] - minY) / cell)
    cellOf[i] = px >= 0 && py >= 0 && px < w && py < h ? py * w + px : -1
    if (cellOf[i] >= 0) count[cellOf[i]]++
  }
  const start = new Uint32Array(w * h + 1)
  for (let k = 0; k < w * h; k++) start[k + 1] = start[k] + count[k]
  const m = start[w * h]
  const hgt = new Float32Array(m)
  const rgb = new Float32Array(m * 3)
  const fill = new Uint32Array(w * h)
  for (let i = 0; i < n; i++) {
    const k = cellOf[i]
    if (k < 0) continue
    const j = start[k] + fill[k]++
    hgt[j] = pts[i * 6 + 2]
    rgb[j * 3] = pts[i * 6 + 3]
    rgb[j * 3 + 1] = pts[i * 6 + 4]
    rgb[j * 3 + 2] = pts[i * 6 + 5]
  }
  return { ...topLayer({ w, h, count, start, hgt, rgb }), x0: minX, y0: minY, cell }
}

/** Add the cells a patch encloses (a table's legs, a pattern's darker middle) to it. */
function fillHoles(cells: number[], w: number) {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const k of cells) {
    const x = k % w
    const y = (k - x) / w
    x0 = Math.min(x0, x)
    x1 = Math.max(x1, x)
    y0 = Math.min(y0, y)
    y1 = Math.max(y1, y)
  }
  // Flood the box around the patch (one cell wider) from its border; what the flood can't reach is a hole.
  x0--
  y0--
  x1++
  y1++
  const bw = x1 - x0 + 1
  const bh = y1 - y0 + 1
  const mark = new Uint8Array(bw * bh)
  for (const k of cells) mark[((k - (k % w)) / w - y0) * bw + (k % w) - x0] = 1
  const stack = [0]
  mark[0] = 2
  while (stack.length) {
    const b = stack.pop()!
    const bx = b % bw
    const by = (b - bx) / bw
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const xx = bx + dx
      const yy = by + dy
      if (xx < 0 || yy < 0 || xx >= bw || yy >= bh || mark[yy * bw + xx]) continue
      mark[yy * bw + xx] = 2
      stack.push(yy * bw + xx)
    }
  }
  for (let b = 0; b < bw * bh; b++) if (!mark[b]) cells.push((Math.floor(b / bw) + y0) * w + (b % bw) + x0)
}

export interface RugPatch {
  /** Cell indices (row-major) of the patch. */
  cells: number[]
  /** Median colour, 0..1. */
  color: [number, number, number]
  /** Cells around the patch that still look like rug (its blurred or broken edge). */
  rim: number[]
}

/**
 * Rugs on a scanned floor: compact, roughly rectangular patches whose colour stands clear of the
 * floor's own (the boards' plank-to-plank shading doesn't count). Works on floorTop's cells.
 */
export function findRugPatches(top: { sum: Float32Array; count: Uint32Array; seen: Uint32Array; w: number; h: number; cell: number }): RugPatch[] {
  const { count, seen, w, h, cell } = top
  // Each cell's top-layer colour. A sparse rug leaves cells whose few points are all boards seen
  // through it; those take the colour of the top-layer cells around them.
  const col = new Float32Array(w * h * 3)
  let have = new Uint8Array(w * h)
  for (let k = 0; k < w * h; k++) {
    if (!count[k]) continue
    for (let c = 0; c < 3; c++) col[k * 3 + c] = top.sum[k * 3 + c] / count[k]
    have[k] = 1
  }
  for (let pass = 0; pass < 2; pass++) {
    const next = have.slice()
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const k = y * w + x
        if (have[k] || !seen[k]) continue
        const acc = [0, 0, 0]
        let n = 0
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx
            const yy = y + dy
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
            const kk = yy * w + xx
            if (!have[kk]) continue
            for (let c = 0; c < 3; c++) acc[c] += col[kk * 3 + c]
            n++
          }
        if (!n) continue
        for (let c = 0; c < 3; c++) col[k * 3 + c] = acc[c] / n
        next[k] = 1
      }
    have = next
  }
  const sum = col
  const seenCells: number[] = []
  for (let k = 0; k < w * h; k++) if (have[k]) seenCells.push(k)
  if (seenCells.length < 50) return []
  const med = [0, 1, 2].map((c) => {
    const a = seenCells.map((k) => sum[k * 3 + c]).sort((x, y) => x - y)
    return a[a.length >> 1]
  })
  const dist = new Float32Array(w * h)
  for (const k of seenCells) {
    let d = 0
    for (let c = 0; c < 3; c++) d += (sum[k * 3 + c] - med[c]) ** 2
    dist[k] = Math.sqrt(d)
  }
  const ds = seenCells.map((k) => dist[k]).sort((x, y) => x - y)
  const cut = Math.max(0.12, 3 * ds[Math.floor(ds.length * 0.6)])
  let mask = new Uint8Array(w * h)
  for (const k of seenCells) if (dist[k] > cut) mask[k] = 1
  // Open the mask (erode, then grow back) so thin lines (a sofa's foot, a skirting) drop out.
  const step = (src: Uint8Array, erode: boolean) => {
    const out = new Uint8Array(w * h)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let all = true
        let any = false
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx
            const yy = y + dy
            const v = xx >= 0 && yy >= 0 && xx < w && yy < h ? src[yy * w + xx] : 0
            if (v) any = true
            else all = false
          }
        out[y * w + x] = erode ? (all ? 1 : 0) : any ? 1 : 0
      }
    return out
  }
  const opened = step(step(mask, true), false)
  for (let k = 0; k < w * h; k++) opened[k] &= mask[k]
  mask = opened
  const out: RugPatch[] = []
  const done = new Uint8Array(w * h)
  for (let k0 = 0; k0 < w * h; k0++) {
    if (!mask[k0] || done[k0]) continue
    const cells: number[] = []
    const stack = [k0]
    done[k0] = 1
    while (stack.length) {
      const k = stack.pop()!
      cells.push(k)
      const x = k % w
      const y = (k - x) / w
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const xx = x + dx
        const yy = y + dy
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
        const kk = yy * w + xx
        if (mask[kk] && !done[kk]) {
          done[kk] = 1
          stack.push(kk)
        }
      }
    }
    fillHoles(cells, w)
    const area = cells.length * cell * cell
    if (area < 0.5 || cells.length > 0.6 * seenCells.length) continue
    // Shape from second moments: a filled rectangle has sides sqrt(12 λ) along its axes.
    let mx = 0
    let my = 0
    for (const k of cells) {
      mx += k % w
      my += Math.floor(k / w)
    }
    mx /= cells.length
    my /= cells.length
    let sxx = 0
    let syy = 0
    let sxy = 0
    for (const k of cells) {
      const dx = (k % w) - mx
      const dy = Math.floor(k / w) - my
      sxx += dx * dx
      syy += dy * dy
      sxy += dx * dy
    }
    sxx = sxx / cells.length + 1 / 12
    syy = syy / cells.length + 1 / 12
    sxy /= cells.length
    const tr = sxx + syy
    const det = sxx * syy - sxy * sxy
    const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det))
    const l1 = tr / 2 + disc
    const l2 = Math.max(1e-6, tr / 2 - disc)
    const sideA = Math.sqrt(12 * l1)
    const sideB = Math.sqrt(12 * l2)
    if (sideB * cell < 0.5 || cells.length / (sideA * sideB) < 0.7) continue
    const color = [0, 1, 2].map((c) => {
      const a = cells.map((k) => sum[k * 3 + c]).sort((x, y) => x - y)
      return a[a.length >> 1]
    }) as [number, number, number]
    // The patch's ragged rim: nearby cells closer to the rug's colour than to the boards (or with no
    // top layer of their own), so painting the rug out leaves no pale scraps at its corners.
    const rim: number[] = []
    const inPatch = new Set(cells)
    for (const k of grow(cells, w, h, 6)) {
      if (inPatch.has(k)) continue
      if (!have[k]) {
        if (seen[k]) rim.push(k)
        continue
      }
      let dr = 0
      let df = 0
      for (let c = 0; c < 3; c++) {
        dr += (sum[k * 3 + c] - color[c]) ** 2
        df += (sum[k * 3 + c] - med[c]) ** 2
      }
      if (dr < df) rim.push(k)
    }
    out.push({ cells, color, rim })
  }
  return out
}

/** How far below the local top a floor point may sit and still count as the top layer (m). */
const LAYER_TOL = 0.006
const NEAR_CAP = 240
/** Points higher than a rug can be (a sofa's or a box's foot) don't set the top layer. */
const RUG_MAX = 0.03
/** The local top is read over the cell and this many cells around it. */
const LAYER_R = 2

/**
 * Each cell's colour from the top layer only: a rug lies a centimetre above the boards, and a scan
 * sees boards through and around it, so averaging every floor point washes the rug out. The local
 * top is a high height quantile over the cell and the cells around it (a stray point can't set it); points
 * within LAYER_TOL (or a share of the local spread, for noisy scans) of it count, and the cell takes
 * their per-channel median. A cell with points but none in the top layer is left for the hole fill,
 * keeping its all-points median as a fallback.
 */
function topLayer(img: { w: number; h: number; count: Uint32Array; start: Uint32Array; hgt: Float32Array; rgb: Float32Array }) {
  const { w, h, count, start, hgt, rgb } = img
  const sum = new Float32Array(w * h * 3)
  const cnt = new Uint32Array(w * h)
  const fallback = new Float32Array(w * h * 3)
  const near = new Float32Array(NEAR_CAP + 25)
  const ch: number[][] = [[], [], []]
  const median = (a: number[]) => {
    a.sort((x, y) => x - y)
    return a[a.length >> 1]
  }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const k = y * w + x
      if (!count[k]) continue
      for (let c = 0; c < 3; c++) ch[c].length = 0
      for (let j = start[k]; j < start[k] + count[k]; j++) for (let c = 0; c < 3; c++) ch[c].push(rgb[j * 3 + c])
      for (let c = 0; c < 3; c++) fallback[k * 3 + c] = median(ch[c])
      // The neighbourhood's heights (thinned to at most ~NEAR_CAP, so a dense scan stays quick).
      let total = 0
      for (let dy = -LAYER_R; dy <= LAYER_R; dy++)
        for (let dx = -LAYER_R; dx <= LAYER_R; dx++) {
          const xx = x + dx
          const yy = y + dy
          if (xx >= 0 && yy >= 0 && xx < w && yy < h) total += count[yy * w + xx]
        }
      const stride = Math.ceil(total / NEAR_CAP)
      let m = 0
      let t = 0
      for (let dy = -LAYER_R; dy <= LAYER_R; dy++)
        for (let dx = -LAYER_R; dx <= LAYER_R; dx++) {
          const xx = x + dx
          const yy = y + dy
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
          const kk = yy * w + xx
          for (let j = start[kk]; j < start[kk] + count[kk]; j++) if (t++ % stride === 0 && hgt[j] <= RUG_MAX) near[m++] = hgt[j]
        }
      if (!m) continue
      const hs = near.subarray(0, m).sort()
      const top = hs[Math.max(0, m - 1 - Math.floor(m * 0.05))]
      const mid = hs[m >> 1]
      const cut = top - Math.max(LAYER_TOL, 0.6 * (top - mid))
      for (let c = 0; c < 3; c++) ch[c].length = 0
      for (let j = start[k]; j < start[k] + count[k]; j++) if (hgt[j] >= cut && hgt[j] <= RUG_MAX) for (let c = 0; c < 3; c++) ch[c].push(rgb[j * 3 + c])
      if (!ch[0].length) continue
      for (let c = 0; c < 3; c++) sum[k * 3 + c] = median(ch[c])
      cnt[k] = 1
    }
  return { sum, count: cnt, fallback, seen: count, w, h }
}

/** Points in a cell at which it keeps its own colour through the softening blur. */
const DENSE = 6

/**
 * Average the cells, fill small holes (a seen cell the fill misses takes its fallback colour) from
 * their neighbors, fill the rest of `inside` (the rooms) out from the floor around it, then soften
 * with a blur that only sparse cells get in full: where the scan is dense, each cell keeps its own
 * colour, so the boards stay crisp.
 */
function finishFloor(img: { sum: Float32Array; count: Uint32Array; w: number; h: number; fallback?: Float32Array; seen?: Uint32Array; inside?: Uint8Array }, x0: number, y0: number, cell: number): FloorImage {
  const { w, h } = img
  let rgb = new Float32Array(w * h * 3)
  let have = new Uint8Array(w * h)
  for (let k = 0; k < w * h; k++) {
    const n = img.count[k]
    if (!n) continue
    rgb[k * 3] = img.sum[k * 3] / n
    rgb[k * 3 + 1] = img.sum[k * 3 + 1] / n
    rgb[k * 3 + 2] = img.sum[k * 3 + 2] / n
    have[k] = 1
  }
  // Grow into gaps a few cells (furniture shadows, sparse splats).
  for (let pass = 0; pass < 4; pass++) {
    const nrgb = rgb.slice()
    const nhave = have.slice()
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const k = y * w + x
        if (have[k]) continue
        let r = 0
        let g = 0
        let b = 0
        let n = 0
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx
            const yy = y + dy
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
            const kk = yy * w + xx
            if (!have[kk]) continue
            r += rgb[kk * 3]
            g += rgb[kk * 3 + 1]
            b += rgb[kk * 3 + 2]
            n++
          }
        if (n >= 2) {
          nrgb[k * 3] = r / n
          nrgb[k * 3 + 1] = g / n
          nrgb[k * 3 + 2] = b / n
          nhave[k] = 1
        }
      }
    rgb = nrgb
    have = nhave
  }
  if (img.fallback && img.seen)
    for (let k = 0; k < w * h; k++)
      if (!have[k] && img.seen[k]) {
        for (let c = 0; c < 3; c++) rgb[k * 3 + c] = img.fallback[k * 3 + c]
        have[k] = 1
      }
  // Rooms: keep growing until every cell inside is coloured (the edge along the walls, under a bookcase).
  const inside = img.inside
  if (inside) {
    for (let pass = 0; pass < 400; pass++) {
      const add: number[] = []
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          const k = y * w + x
          if (have[k] || !inside[k]) continue
          let r = 0
          let g = 0
          let b = 0
          let n = 0
          for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const xx = x + dx
            const yy = y + dy
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
            const kk = yy * w + xx
            if (!have[kk]) continue
            r += rgb[kk * 3]
            g += rgb[kk * 3 + 1]
            b += rgb[kk * 3 + 2]
            n++
          }
          if (n) add.push(k, r / n, g / n, b / n)
        }
      if (!add.length) break
      for (let i = 0; i < add.length; i += 4) {
        const k = add[i]
        rgb[k * 3] = add[i + 1]
        rgb[k * 3 + 1] = add[i + 2]
        rgb[k * 3 + 2] = add[i + 3]
        have[k] = 1
      }
    }
  }
  // One 3x3 box blur over the seen cells, mixed with the cell's own colour where it had plenty of points.
  const out = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const k = y * w + x
      if (!have[k]) continue
      let r = 0
      let g = 0
      let b = 0
      let n = 0
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx
          const yy = y + dy
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
          const kk = yy * w + xx
          if (!have[kk]) continue
          r += rgb[kk * 3]
          g += rgb[kk * 3 + 1]
          b += rgb[kk * 3 + 2]
          n++
        }
      const own = img.seen && img.count[k] ? 0.8 * Math.min(1, img.seen[k] / DENSE) : 0
      out[k * 4] = (own * rgb[k * 3] + (1 - own) * (r / n)) * 255
      out[k * 4 + 1] = (own * rgb[k * 3 + 1] + (1 - own) * (g / n)) * 255
      out[k * 4 + 2] = (own * rgb[k * 3 + 2] + (1 - own) * (b / n)) * 255
      out[k * 4 + 3] = 255
    }
  return { x0, y0, cell, width: w, height: h, rgba: out }
}

// ------------------------------------------------------------------ cartoon colors

/**
 * Turn a real, slightly muddy scanned color into a cartoon one: a little more saturated, a little
 * brighter, and with lightness snapped to bands so neighboring shades read as one flat color.
 */
export function toonRGB(r: number, g: number, b: number, bands = 6): [number, number, number] {
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  let l = (max + min) / 2
  let s = 0
  let hue = 0
  if (max !== min) {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    hue = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4
    hue /= 6
  }
  s = Math.min(1, s * 1.35 + 0.04)
  l = Math.round((0.12 + l * 0.82) * bands) / bands
  l = Math.max(0.16, Math.min(0.93, l))
  if (s === 0) return [l, l, l]
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  const f = (t: number) => {
    if (t < 0) t += 1
    if (t > 1) t -= 1
    if (t < 1 / 6) return p + (q - p) * 6 * t
    if (t < 1 / 2) return q
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
    return p
  }
  return [f(hue + 1 / 3), f(hue), f(hue - 1 / 3)]
}

export function toonHex(color: string, bands = 6): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(color.trim())
  if (!m) return color
  const n = parseInt(m[1], 16)
  return hex(...toonRGB(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, bands))
}

/** A box blur of radius `r` over the seen (opaque) pixels, run twice so it is close to a gaussian. */
function blurFloor(img: FloorImage, r: number): Float32Array {
  const { width: w, height: h, rgba } = img
  let src = new Float32Array(w * h * 3)
  for (let k = 0; k < w * h; k++) for (let c = 0; c < 3; c++) src[k * 3 + c] = rgba[k * 4 + c]
  for (let pass = 0; pass < 2; pass++) {
    const dst = new Float32Array(src.length)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const k = y * w + x
        if (!rgba[k * 4 + 3]) continue
        let rr = 0
        let gg = 0
        let bb = 0
        let n = 0
        for (let dy = -r; dy <= r; dy++)
          for (let dx = -r; dx <= r; dx++) {
            const xx = x + dx
            const yy = y + dy
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
            const kk = yy * w + xx
            if (!rgba[kk * 4 + 3]) continue
            rr += src[kk * 3]
            gg += src[kk * 3 + 1]
            bb += src[kk * 3 + 2]
            n++
          }
        dst[k * 3] = rr / n
        dst[k * 3 + 1] = gg / n
        dst[k * 3 + 2] = bb / n
      }
    src = dst
  }
  return src
}

/** Bilinear sample of a w x h RGB float image at (fx, fy) in pixel units, ignoring unseen pixels. */
function sampleRGB(rgb: Float32Array, alpha: Uint8ClampedArray, w: number, h: number, fx: number, fy: number, out: number[]): boolean {
  const x0 = Math.max(0, Math.min(w - 1, Math.floor(fx)))
  const y0 = Math.max(0, Math.min(h - 1, Math.floor(fy)))
  const x1 = Math.min(w - 1, x0 + 1)
  const y1 = Math.min(h - 1, y0 + 1)
  const tx = Math.max(0, Math.min(1, fx - x0))
  const ty = Math.max(0, Math.min(1, fy - y0))
  out[0] = out[1] = out[2] = 0
  let wsum = 0
  for (const [x, y, wt] of [
    [x0, y0, (1 - tx) * (1 - ty)],
    [x1, y0, tx * (1 - ty)],
    [x0, y1, (1 - tx) * ty],
    [x1, y1, tx * ty],
  ]) {
    const k = y * w + x
    if (!alpha[k * 4 + 3] || wt <= 0) continue
    out[0] += rgb[k * 3] * wt
    out[1] += rgb[k * 3 + 1] * wt
    out[2] += rgb[k * 3 + 2] * wt
    wsum += wt
  }
  if (!wsum) return false
  out[0] /= wsum
  out[1] /= wsum
  out[2] /= wsum
  return true
}

/**
 * A cartoon copy of the floor image: blurred first (so scan speckle doesn't become a pixel
 * mosaic), upsampled 3x so the colour bands get smooth outlines, then posterized into flat patches.
 */
export function toonFloor(img: FloorImage): FloorImage {
  const { width: w, height: h } = img
  const blurred = blurFloor(img, 1)
  const up = Math.max(1, Math.min(3, Math.floor(1536 / Math.max(w, h))))
  const W = w * up
  const H = h * up
  const out = new Uint8ClampedArray(W * H * 4)
  const c = [0, 0, 0]
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const fx = (x + 0.5) / up - 0.5
      const fy = (y + 0.5) / up - 0.5
      // Unseen pixels stay unseen (transparent).
      const sk = Math.min(h - 1, Math.floor(y / up)) * w + Math.min(w - 1, Math.floor(x / up))
      if (!img.rgba[sk * 4 + 3]) continue
      if (!sampleRGB(blurred, img.rgba, w, h, fx, fy, c)) continue
      const [r, g, b] = toonRGB(c[0] / 255, c[1] / 255, c[2] / 255, 5)
      const k = (y * W + x) * 4
      out[k] = r * 255
      out[k + 1] = g * 255
      out[k + 2] = b * 255
      out[k + 3] = 255
    }
  return { ...img, width: W, height: H, cell: img.cell / up, rgba: out }
}

/**
 * The floor image for Lifelike: a scan's colours come out washed out (a rug barely shows on the
 * boards), so stretch the contrast around the floor's average and add back a little saturation.
 */
export function realFloor(img: FloorImage): FloorImage {
  const { width: w, height: h, rgba } = img
  const soft = new Float32Array(w * h * 3)
  for (let k = 0; k < w * h; k++) for (let c = 0; c < 3; c++) soft[k * 3 + c] = rgba[k * 4 + c]
  let sum = 0
  let n = 0
  for (let k = 0; k < w * h; k++) {
    if (!rgba[k * 4 + 3]) continue
    sum += 0.299 * soft[k * 3] + 0.587 * soft[k * 3 + 1] + 0.114 * soft[k * 3 + 2]
    n++
  }
  const mean = n ? sum / n : 128
  const out = new Uint8ClampedArray(rgba.length)
  for (let k = 0; k < w * h; k++) {
    if (!rgba[k * 4 + 3]) continue
    const r = soft[k * 3]
    const g = soft[k * 3 + 1]
    const b = soft[k * 3 + 2]
    const l = 0.299 * r + 0.587 * g + 0.114 * b
    // Lightness away from the mean x1.5; colour away from grey x1.25.
    const l2 = mean + (l - mean) * 1.5
    for (let ch = 0; ch < 3; ch++) out[k * 4 + ch] = l2 + ([r, g, b][ch] - l) * 1.25
    out[k * 4 + 3] = 255
  }
  return { ...img, rgba: out }
}
