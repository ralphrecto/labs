// Colors taken from the real apartment scan, for the styled 3D views: what the floor looks like
// from above (a small top-down color image) and the paint color of every wall. The Cartoon style
// posterizes these into flat, bright bands; the Lifelike style uses them as they are.
//
// Input is scan points in the app world frame (meters, Y up, floor at 0, plan (x, y) = world (x, z)),
// which is what `toScanData` returns.

import { closestPointOnSegment, pointInPolygon, wallInwardNormal, wallSegments } from '../model/geometry'
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

export function scanPalette(scans: ScanPoints[], rooms: Room[], cell = 0.05): ScanPalette {
  const palette: ScanPalette = { floor: null, walls: {}, wallColor: {}, floorColor: {} }
  const colored = scans.filter((s) => s.colors && s.count > 0)
  if (!colored.length) return palette

  // ---- floor image bounds
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const s of colored) {
    const p = s.positions
    for (let i = 0; i < s.count; i++) {
      const h = p[i * 3 + 1]
      if (h > FLOOR_BAND || h < -FLOOR_BAND) continue
      const x = p[i * 3]
      const y = p[i * 3 + 2]
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
  }

  // ---- wall and floor sampling
  const segs = rooms.map((r) => wallSegments(r))
  const normals = rooms.map((r, ri) => segs[ri].map((_, i) => wallInwardNormal(r, i)))
  const wallSamples = segs.map((rs) => rs.map(() => [] as number[]))
  const floorSamples = rooms.map(() => [] as number[])
  const cap = 6000

  let img: { w: number; h: number; count: Uint32Array; start: Uint32Array; hgt: Float32Array; rgb: Float32Array; fill: Uint32Array } | null = null
  if (Number.isFinite(minX) && maxX > minX && maxY > minY) {
    const w = Math.min(1024, Math.ceil((maxX - minX) / cell) + 1)
    const h = Math.min(1024, Math.ceil((maxY - minY) / cell) + 1)
    // Floor points bucketed by cell (counted first, then filled), so each cell can keep its top layer.
    const count = new Uint32Array(w * h)
    for (const s of colored) {
      const p = s.positions
      for (let i = 0; i < s.count; i++) {
        const hh = p[i * 3 + 1]
        if (hh > FLOOR_BAND || hh < -FLOOR_BAND) continue
        const px = Math.floor((p[i * 3] - minX) / cell)
        const py = Math.floor((p[i * 3 + 2] - minY) / cell)
        if (px >= 0 && py >= 0 && px < w && py < h) count[py * w + px]++
      }
    }
    const start = new Uint32Array(w * h + 1)
    for (let k = 0; k < w * h; k++) start[k + 1] = start[k] + count[k]
    const n = start[w * h]
    img = { w, h, count, start, hgt: new Float32Array(n), rgb: new Float32Array(n * 3), fill: new Uint32Array(w * h) }
  }

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
      if (img && hgt <= FLOOR_BAND && hgt >= -FLOOR_BAND) {
        const px = Math.floor((x - minX) / cell)
        const py = Math.floor((y - minY) / cell)
        if (px >= 0 && py >= 0 && px < img.w && py < img.h) {
          const k = py * img.w + px
          const j = img.start[k] + img.fill[k]++
          img.hgt[j] = hgt
          img.rgb[j * 3] = r
          img.rgb[j * 3 + 1] = g
          img.rgb[j * 3 + 2] = b
        }
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

  if (img) palette.floor = finishFloor(topLayer(img), minX, minY, cell)
  return palette
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

/** Average the cells, fill small holes (a seen cell the fill misses takes its fallback colour) from their neighbors, then soften with a blur. */
function finishFloor(img: { sum: Float32Array; count: Uint32Array; w: number; h: number; fallback?: Float32Array; seen?: Uint32Array }, x0: number, y0: number, cell: number): FloorImage {
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
  // One 3x3 box blur over the seen cells.
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
      out[k * 4] = (r / n) * 255
      out[k * 4 + 1] = (g / n) * 255
      out[k * 4 + 2] = (b / n) * 255
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
