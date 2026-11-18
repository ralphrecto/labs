// Pure scan math: turning a scan upright, finding the floor, straightening it on the plan,
// calibrating its scale from tape-measured lengths, and drawing top-down underlays.
// Everything here works on flat Float32Arrays so it runs on a million points in well under a second.

import type { ScanRef, ScanReference, UpAxis, Vec2 } from '../model/types'
import type { ScanData, ScanPoints } from './types'

// ------------------------------------------------------------------ up axis

/** Rotate a file-space point so that `up` becomes +Y (a proper rotation, never a mirror). */
export function upright(up: UpAxis, x: number, y: number, z: number): [number, number, number] {
  switch (up) {
    case 'y':
      return [x, y, z]
    case '-y':
      return [x, -y, -z] // 180° about X
    case 'z':
      return [x, z, -y] // -90° about X
    case '-z':
      return [x, -z, y] // +90° about X
  }
}

/** Euler X rotation (radians) that applies `upright` in three.js. */
export function uprightRotationX(up: UpAxis): number {
  return up === 'y' ? 0 : up === '-y' ? Math.PI : up === 'z' ? -Math.PI / 2 : Math.PI / 2
}

/** A copy of the points turned upright. */
export function uprightPositions(pos: Float32Array, up: UpAxis): Float32Array {
  const out = new Float32Array(pos.length)
  for (let i = 0; i < pos.length; i += 3) {
    const [x, y, z] = upright(up, pos[i], pos[i + 1], pos[i + 2])
    out[i] = x
    out[i + 1] = y
    out[i + 2] = z
  }
  return out
}

/** Every `step`-th point, so analysis stays fast on big scans. */
export function subsample(pos: Float32Array, max: number): Float32Array {
  const n = pos.length / 3
  if (n <= max) return pos
  const step = n / max
  const out = new Float32Array(max * 3)
  for (let i = 0; i < max; i++) {
    const j = Math.floor(i * step) * 3
    out[i * 3] = pos[j]
    out[i * 3 + 1] = pos[j + 1]
    out[i * 3 + 2] = pos[j + 2]
  }
  return out
}

function percentile(sorted: Float32Array, q: number) {
  if (!sorted.length) return 0
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]
}

function axisValues(pos: Float32Array, axis: number) {
  const n = pos.length / 3
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = pos[i * 3 + axis]
  return out
}

interface Peak {
  /** Center of the peak, in file units. */
  at: number
  /** Share of all points within the peak. */
  share: number
}

/** Histogram the values and return the strongest peak inside [lo, hi] (as fractions of the robust range). */
function strongestPeak(values: Float32Array, lo: number, hi: number, binSize?: number): Peak | null {
  if (!values.length) return null
  const sorted = Float32Array.from(values).sort()
  const p1 = percentile(sorted, 0.005)
  const p99 = percentile(sorted, 0.995)
  const range = p99 - p1
  if (!(range > 0)) return null
  const bin = binSize ?? range / 300
  const nb = Math.ceil(range / bin) + 1
  const hist = new Uint32Array(nb)
  for (const v of values) {
    const b = Math.floor((v - p1) / bin)
    if (b >= 0 && b < nb) hist[b]++
  }
  const b0 = Math.floor(lo * (nb - 1))
  const b1 = Math.ceil(hi * (nb - 1))
  let best = -1
  let bestCount = 0
  for (let b = b0; b <= b1; b++) {
    // Sum of three bins, so a plane split across two bins still wins.
    const c = (hist[b - 1] ?? 0) + hist[b] + (hist[b + 1] ?? 0)
    if (c > bestCount) {
      bestCount = c
      best = b
    }
  }
  if (best < 0) return null
  // Refine with the mean of the values in the winning bins.
  const a = p1 + (best - 1) * bin
  const z = p1 + (best + 2) * bin
  let sum = 0
  let cnt = 0
  for (const v of values)
    if (v >= a && v < z) {
      sum += v
      cnt++
    }
  return { at: cnt ? sum / cnt : p1 + (best + 0.5) * bin, share: bestCount / values.length }
}

/**
 * Guess which file axis points up. The floor is usually the biggest flat surface a phone scan
 * sees, so the up axis is the one with the sharpest peak, and the floor sits at its low end.
 * Phone scans (ARKit) are almost always +Y up, so Y gets a small head start.
 */
export function detectUpAxis(pos: Float32Array): UpAxis {
  const pts = subsample(pos, 200_000)
  let best: { up: UpAxis; score: number } = { up: 'y', score: -1 }
  for (const axis of [1, 2] as const) {
    const vals = axisValues(pts, axis)
    const low = strongestPeak(vals, 0, 0.5)
    const high = strongestPeak(vals, 0.5, 1)
    if (!low && !high) continue
    const lowShare = low?.share ?? 0
    const highShare = high?.share ?? 0
    const score = Math.max(lowShare, highShare) * (axis === 1 ? 1.25 : 1)
    const floorLow = lowShare >= highShare
    const up: UpAxis = axis === 1 ? (floorLow ? 'y' : '-y') : floorLow ? 'z' : '-z'
    if (score > best.score) best = { up, score }
  }
  return best.up
}

/** Floor and ceiling heights (upright file units) from the height histogram. */
export function detectFloorCeiling(uprightPos: Float32Array): { floor: number; ceiling?: number } {
  const vals = axisValues(subsample(uprightPos, 300_000), 1)
  const floor = strongestPeak(vals, 0, 0.45)
  const ceil = strongestPeak(vals, 0.6, 1)
  const f = floor?.at ?? 0
  // A ceiling needs a real peak (phone scans often miss most of it) and a plausible height.
  const ok = ceil && ceil.share > 0.01 && ceil.at - f > 1.8 && ceil.at - f < 6
  return { floor: f, ceiling: ok ? ceil.at : undefined }
}

// ------------------------------------------------------------------ plan transform

export type ScanAlign = Pick<ScanRef, 'upAxis' | 'floor' | 'scale' | 'yaw' | 'x' | 'y'>

/** Upright top-down coordinates (ux, uz) to plan coordinates. */
export function scanToPlan(t: ScanAlign, p: Vec2): Vec2 {
  const c = Math.cos(t.yaw)
  const s = Math.sin(t.yaw)
  return { x: t.x + t.scale * (c * p.x - s * p.y), y: t.y + t.scale * (s * p.x + c * p.y) }
}

/** Plan coordinates back to upright top-down coordinates. */
export function planToScan(t: ScanAlign, p: Vec2): Vec2 {
  const c = Math.cos(t.yaw)
  const s = Math.sin(t.yaw)
  const dx = (p.x - t.x) / t.scale
  const dy = (p.y - t.y) / t.scale
  return { x: c * dx + s * dy, y: -s * dx + c * dy }
}

/** The SVG transform that draws upright top-down coordinates on the plan. */
export function svgTransform(t: ScanAlign) {
  return `translate(${t.x} ${t.y}) rotate(${(t.yaw * 180) / Math.PI}) scale(${t.scale})`
}

/**
 * The scan in the app's world frame: meters, Y up, floor at zero, plan (x, y) = world (x, z).
 * This is what furniture detection consumes.
 */
export function toScanData(ref: ScanRef, points: ScanPoints, maxPoints = 1_000_000): ScanData {
  const n = points.count
  const step = Math.max(1, n / maxPoints)
  const m = Math.min(n, Math.floor(n / step))
  const positions = new Float32Array(m * 3)
  const colors = points.colors ? new Float32Array(m * 3) : undefined
  const c = Math.cos(ref.yaw)
  const s = Math.sin(ref.yaw)
  for (let i = 0; i < m; i++) {
    const j = Math.floor(i * step) * 3
    const [ux, uy, uz] = upright(ref.upAxis, points.positions[j], points.positions[j + 1], points.positions[j + 2])
    positions[i * 3] = ref.x + ref.scale * (c * ux - s * uz)
    positions[i * 3 + 1] = ref.scale * (uy - ref.floor)
    positions[i * 3 + 2] = ref.y + ref.scale * (s * ux + c * uz)
    if (colors && points.colors) {
      colors[i * 3] = points.colors[j]
      colors[i * 3 + 1] = points.colors[j + 1]
      colors[i * 3 + 2] = points.colors[j + 2]
    }
  }
  const source = /scaniverse/i.test(ref.name + ref.fileName) || ref.fileName.toLowerCase().endsWith('.spz') ? 'scaniverse' : 'ply'
  return { id: ref.id, name: ref.name, source, points: { count: m, positions, colors }, upAxis: 'y', metric: ref.references.length > 0 }
}

// ------------------------------------------------------------------ straightening

/** Upright top-down points (ux, uz) whose height falls in the wall band. */
export function wallBandPoints(uprightPos: Float32Array, ref: Pick<ScanRef, 'floor' | 'scale' | 'sliceLow' | 'sliceHigh'>, max = Infinity): Float32Array {
  const lo = ref.floor + ref.sliceLow / ref.scale
  const hi = ref.floor + ref.sliceHigh / ref.scale
  const n = uprightPos.length / 3
  const step = Math.max(1, n / Math.min(n, max * 4))
  const tmp: number[] = []
  for (let f = 0; f < n; f += step) {
    const i = Math.floor(f) * 3
    const h = uprightPos[i + 1]
    if (h >= lo && h <= hi) tmp.push(uprightPos[i], uprightPos[i + 2])
  }
  const out = Float32Array.from(tmp)
  if (out.length / 2 <= max) return out
  const k = out.length / 2 / max
  const res = new Float32Array(max * 2)
  for (let i = 0; i < max; i++) {
    const j = Math.floor(i * k) * 2
    res[i * 2] = out[j]
    res[i * 2 + 1] = out[j + 1]
  }
  return res
}

function alignmentScore(pts: Float32Array, angle: number, bin: number) {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  const hx = new Map<number, number>()
  const hy = new Map<number, number>()
  for (let i = 0; i < pts.length; i += 2) {
    const x = c * pts[i] - s * pts[i + 1]
    const y = s * pts[i] + c * pts[i + 1]
    const bx = Math.floor(x / bin)
    const by = Math.floor(y / bin)
    hx.set(bx, (hx.get(bx) ?? 0) + 1)
    hy.set(by, (hy.get(by) ?? 0) + 1)
  }
  let score = 0
  for (const v of hx.values()) score += v * v
  for (const v of hy.values()) score += v * v
  return score
}

/**
 * The plan rotation that lines the walls up with the plan axes: walls are dense lines in the
 * wall band, so the best angle makes the x and y histograms as peaky as possible.
 * Returns the angle in (-45°, 45°], to add to the scan's rotation.
 */
export function straightenAngle(bandPts: Float32Array, scale: number): number {
  if (bandPts.length < 20) return 0
  const bin = 0.03 / scale
  const pts = bandPts.length > 60_000 ? subsample2(bandPts, 30_000) : bandPts
  let best = 0
  let bestScore = -1
  const deg = Math.PI / 180
  for (let a = -45; a < 45; a += 0.5) {
    const sc = alignmentScore(pts, a * deg, bin)
    if (sc > bestScore) {
      bestScore = sc
      best = a * deg
    }
  }
  const coarse = best
  for (let a = -0.5; a <= 0.5; a += 0.05) {
    const sc = alignmentScore(pts, coarse + a * deg, bin)
    if (sc > bestScore) {
      bestScore = sc
      best = coarse + a * deg
    }
  }
  return best
}

function subsample2(pts: Float32Array, max: number) {
  const n = pts.length / 2
  const k = n / max
  const out = new Float32Array(max * 2)
  for (let i = 0; i < max; i++) {
    const j = Math.floor(i * k) * 2
    out[i * 2] = pts[j]
    out[i * 2 + 1] = pts[j + 1]
  }
  return out
}

/** Rotation that makes the line a→b (in plan coordinates) horizontal or vertical, whichever is closer. */
export function straightenFromLine(a: Vec2, b: Vec2): number {
  const ang = Math.atan2(b.y - a.y, b.x - a.x)
  const q = Math.PI / 2
  return Math.round(ang / q) * q - ang
}

// ------------------------------------------------------------------ calibration

export interface Calibration {
  /** Meters per file unit: the median of what each reference implies. */
  scale: number
  /** For each reference, how far its own scale is from the median (0.01 = 1%). */
  deviations: number[]
  /** The largest deviation. */
  spread: number
}

export function refLength(r: ScanReference) {
  return Math.hypot(r.b.x - r.a.x, r.b.y - r.a.y)
}

export function calibrate(refs: ScanReference[]): Calibration | null {
  const implied = refs.map((r) => (refLength(r) > 1e-9 ? r.length / refLength(r) : NaN)).filter((x) => Number.isFinite(x) && x > 0)
  if (!implied.length) return null
  const sorted = [...implied].sort((a, b) => a - b)
  const mid = sorted.length / 2
  const scale = sorted.length % 2 ? sorted[Math.floor(mid)] : (sorted[mid - 1] + sorted[mid]) / 2
  const deviations = implied.map((s) => s / scale - 1)
  return { scale, deviations, spread: Math.max(...deviations.map(Math.abs)) }
}

/** Rescale about a plan point so it stays put on screen (used when calibration changes the scale). */
export function rescaleAbout(t: ScanAlign, newScale: number, anchor: Vec2): Pick<ScanRef, 'scale' | 'x' | 'y'> {
  const local = planToScan(t, anchor)
  const moved = scanToPlan({ ...t, scale: newScale }, local)
  return { scale: newScale, x: t.x + anchor.x - moved.x, y: t.y + anchor.y - moved.y }
}

/** Turn the scan by `delta` about a plan point, which stays where it is. */
export function rotateAbout(t: ScanAlign, delta: number, anchor: Vec2): Pick<ScanRef, 'yaw' | 'x' | 'y'> {
  const local = planToScan(t, anchor)
  const yaw = t.yaw + delta
  const moved = scanToPlan({ ...t, yaw }, local)
  return { yaw, x: t.x + anchor.x - moved.x, y: t.y + anchor.y - moved.y }
}

// ------------------------------------------------------------------ underlays

export interface Underlay {
  width: number
  height: number
  /** RGBA, row 0 at the smallest uz. */
  data: Uint8ClampedArray
  /** Top-left corner and pixel size, in upright top-down file units. */
  u0: number
  v0: number
  pixel: number
}

/** Robust top-down bounds of the scan (upright file units). */
export function topBounds(uprightPos: Float32Array) {
  const pts = subsample(uprightPos, 200_000)
  const xs = axisValues(pts, 0).sort()
  const zs = axisValues(pts, 2).sort()
  const pad = (percentile(xs, 0.998) - percentile(xs, 0.002)) * 0.03
  return {
    minX: percentile(xs, 0.002) - pad,
    maxX: percentile(xs, 0.998) + pad,
    minZ: percentile(zs, 0.002) - pad,
    maxZ: percentile(zs, 0.998) + pad,
  }
}

function grid(bounds: ReturnType<typeof topBounds>, scale: number, maxDim: number, base = 0.015 / scale) {
  let pixel = base
  const w0 = (bounds.maxX - bounds.minX) / pixel
  const h0 = (bounds.maxZ - bounds.minZ) / pixel
  if (Math.max(w0, h0) > maxDim) pixel *= Math.max(w0, h0) / maxDim
  const width = Math.max(1, Math.ceil((bounds.maxX - bounds.minX) / pixel))
  const height = Math.max(1, Math.ceil((bounds.maxZ - bounds.minZ) / pixel))
  return { pixel, width, height, u0: bounds.minX, v0: bounds.minZ }
}

/**
 * Wall outline: how many points in the wall band land in each pixel. Walls are seen edge-on
 * from above, so they pile up into dark lines while furniture below the band disappears.
 */
export function wallUnderlay(uprightPos: Float32Array, ref: Pick<ScanRef, 'floor' | 'scale' | 'sliceLow' | 'sliceHigh'>, bounds = topBounds(uprightPos), maxDim = 2048): Underlay {
  const g = grid(bounds, ref.scale, maxDim)
  const counts = new Float32Array(g.width * g.height)
  const lo = ref.floor + ref.sliceLow / ref.scale
  const hi = ref.floor + ref.sliceHigh / ref.scale
  for (let i = 0; i < uprightPos.length; i += 3) {
    const h = uprightPos[i + 1]
    if (h < lo || h > hi) continue
    const px = Math.floor((uprightPos[i] - g.u0) / g.pixel)
    const py = Math.floor((uprightPos[i + 2] - g.v0) / g.pixel)
    if (px < 0 || py < 0 || px >= g.width || py >= g.height) continue
    counts[py * g.width + px]++
  }
  // Normalize against a high percentile of the non-empty pixels, on a log scale.
  const nz = counts.filter((c) => c > 0).sort()
  const cap = Math.max(1, percentile(nz, 0.97))
  const lc = Math.log1p(cap)
  const data = new Uint8ClampedArray(g.width * g.height * 4)
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i]
    if (!c) continue
    const t = Math.min(1, Math.log1p(c) / lc)
    data[i * 4] = 40
    data[i * 4 + 1] = 52
    data[i * 4 + 2] = 92
    data[i * 4 + 3] = Math.round(40 + 215 * t)
  }
  return { ...g, data }
}

/** A photo-like view from above, with everything over the cut height removed (no ceiling). */
export function topUnderlay(
  uprightPos: Float32Array,
  colors: Float32Array | undefined,
  ref: Pick<ScanRef, 'floor' | 'scale' | 'cutHeight'>,
  bounds = topBounds(uprightPos),
  maxDim = 2048,
): Underlay {
  // Sparse scans get bigger pixels, so the view reads as surfaces rather than speckles.
  const area = (bounds.maxX - bounds.minX) * (bounds.maxZ - bounds.minZ)
  const perPixel = Math.sqrt(area / Math.max(1, uprightPos.length / 3 / 2.5))
  const g = grid(bounds, ref.scale, maxDim, Math.min(0.04 / ref.scale, Math.max(0.015 / ref.scale, perPixel)))
  const n = g.width * g.height
  const top = new Float32Array(n).fill(-Infinity)
  const data = new Uint8ClampedArray(n * 4)
  const cut = ref.floor + ref.cutHeight / ref.scale
  for (let i = 0; i < uprightPos.length; i += 3) {
    const h = uprightPos[i + 1]
    if (h > cut) continue
    const px = Math.floor((uprightPos[i] - g.u0) / g.pixel)
    const py = Math.floor((uprightPos[i + 2] - g.v0) / g.pixel)
    if (px < 0 || py < 0 || px >= g.width || py >= g.height) continue
    const k = py * g.width + px
    if (h <= top[k]) continue
    top[k] = h
    const r = colors ? colors[i] : 0.6
    const gg = colors ? colors[i + 1] : 0.6
    const b = colors ? colors[i + 2] : 0.6
    data[k * 4] = r * 255
    data[k * 4 + 1] = gg * 255
    data[k * 4 + 2] = b * 255
    data[k * 4 + 3] = 255
  }
  // Fill single-pixel holes from neighbours so sparse areas read as surfaces.
  const filled = data.slice()
  for (let y = 1; y < g.height - 1; y++)
    for (let x = 1; x < g.width - 1; x++) {
      const k = y * g.width + x
      if (data[k * 4 + 3]) continue
      let r = 0
      let gg = 0
      let b = 0
      let c = 0
      for (const d of [-1, 1, -g.width, g.width]) {
        const j = (k + d) * 4
        if (data[j + 3]) {
          r += data[j]
          gg += data[j + 1]
          b += data[j + 2]
          c++
        }
      }
      if (c >= 2) {
        filled[k * 4] = r / c
        filled[k * 4 + 1] = gg / c
        filled[k * 4 + 2] = b / c
        filled[k * 4 + 3] = 255
      }
    }
  return { ...g, data: filled }
}

// ------------------------------------------------------------------ tracing help

/**
 * Pull a point onto the nearest wall line in the underlay: search a small radius for the
 * pixel with the most wall points. Used to snap corners while tracing.
 */
export function snapToWall(u: Underlay, p: Vec2, radius: number): Vec2 | null {
  const cx = Math.floor((p.x - u.u0) / u.pixel)
  const cy = Math.floor((p.y - u.v0) / u.pixel)
  const r = Math.max(1, Math.round(radius / u.pixel))
  // The strongest wall pixel in reach sets the bar; then take the closest pixel that nearly meets it,
  // so the point moves straight onto the wall instead of sliding along it.
  let max = 0
  const inReach = (x: number, y: number) => x >= 0 && y >= 0 && x < u.width && y < u.height && (x - cx) ** 2 + (y - cy) ** 2 <= r * r
  for (let y = cy - r; y <= cy + r; y++) for (let x = cx - r; x <= cx + r; x++) if (inReach(x, y)) max = Math.max(max, u.data[(y * u.width + x) * 4 + 3])
  if (max < 120) return null
  let bestD = Infinity
  let bx = 0
  let by = 0
  for (let y = cy - r; y <= cy + r; y++)
    for (let x = cx - r; x <= cx + r; x++) {
      if (!inReach(x, y) || u.data[(y * u.width + x) * 4 + 3] < max * 0.85) continue
      const d = (x - cx) ** 2 + (y - cy) ** 2
      if (d < bestD) {
        bestD = d
        bx = x
        by = y
      }
    }
  return { x: u.u0 + (bx + 0.5) * u.pixel, y: u.v0 + (by + 0.5) * u.pixel }
}

/** Wall strength (0..255) of the underlay at an upright top-down point. */
export function underlayAt(u: Underlay, p: Vec2) {
  const x = Math.floor((p.x - u.u0) / u.pixel)
  const y = Math.floor((p.y - u.v0) / u.pixel)
  if (x < 0 || y < 0 || x >= u.width || y >= u.height) return 0
  return u.data[(y * u.width + x) * 4 + 3]
}

/**
 * Find walls running along the plan axes near a point: a vertical wall gives an x, a horizontal
 * wall gives a y, and a corner gives both. `strength` samples the wall outline at plan points.
 * For each candidate offset it sums the outline along a short strip through the point, and keeps
 * the strongest strip if it is clearly a wall.
 */
export function snapAxes(strength: (p: Vec2) => number, p: Vec2, radius: number, step: number): { x?: number; y?: number } {
  const n = Math.max(2, Math.round(radius / step))
  const profile = (vertical: boolean) => {
    let best = -1
    let bestAt = 0
    let bestD = Infinity
    const vals: number[] = []
    for (let i = -n; i <= n; i++) {
      let sum = 0
      for (let j = -n; j <= n; j++) {
        const q = vertical ? { x: p.x + i * step, y: p.y + j * step } : { x: p.x + j * step, y: p.y + i * step }
        sum += strength(q)
      }
      vals.push(sum)
    }
    const max = Math.max(...vals)
    // A wall crossing the strip lights up at least a third of it.
    if (max < 0.33 * (2 * n + 1) * 160) return null
    // Of the clear peaks (local maxima), take the one closest to the cursor, then refine it
    // to sub-pixel precision with the centroid of its neighbours.
    for (let k = 0; k < vals.length; k++) {
      const v = vals[k]
      if (v < max * 0.6 || v < (vals[k - 1] ?? 0) || v < (vals[k + 1] ?? 0)) continue
      const d = Math.abs(k - n)
      if (d < bestD) {
        bestD = d
        best = k
      }
    }
    if (best < 0) return null
    const l = vals[best - 1] ?? 0
    const c = vals[best]
    const r = vals[best + 1] ?? 0
    bestAt = best - n + (r - l) / Math.max(1e-9, l + c + r)
    return bestAt * step
  }
  const dx = profile(true)
  const dy = profile(false)
  return { x: dx == null ? undefined : p.x + dx, y: dy == null ? undefined : p.y + dy }
}
