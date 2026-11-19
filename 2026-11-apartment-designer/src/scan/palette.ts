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

  let img: { sum: Float32Array; count: Uint32Array; w: number; h: number } | null = null
  if (Number.isFinite(minX) && maxX > minX && maxY > minY) {
    const w = Math.min(1024, Math.ceil((maxX - minX) / cell) + 1)
    const h = Math.min(1024, Math.ceil((maxY - minY) / cell) + 1)
    img = { sum: new Float32Array(w * h * 3), count: new Uint32Array(w * h), w, h }
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
          img.sum[k * 3] += r
          img.sum[k * 3 + 1] += g
          img.sum[k * 3 + 2] += b
          img.count[k]++
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

  if (img) palette.floor = finishFloor(img, minX, minY, cell)
  return palette
}

/** Average the cells, fill small holes from their neighbors, then soften with a blur. */
function finishFloor(img: { sum: Float32Array; count: Uint32Array; w: number; h: number }, x0: number, y0: number, cell: number): FloorImage {
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

/** A cartoon copy of the floor image: posterized colors and slightly bigger flat patches. */
export function toonFloor(img: FloorImage): FloorImage {
  const out = new Uint8ClampedArray(img.rgba.length)
  for (let k = 0; k < img.width * img.height; k++) {
    if (!img.rgba[k * 4 + 3]) continue
    const [r, g, b] = toonRGB(img.rgba[k * 4] / 255, img.rgba[k * 4 + 1] / 255, img.rgba[k * 4 + 2] / 255, 5)
    out[k * 4] = r * 255
    out[k * 4 + 1] = g * 255
    out[k * 4 + 2] = b * 255
    out[k * 4 + 3] = 255
  }
  return { ...img, rgba: out }
}
