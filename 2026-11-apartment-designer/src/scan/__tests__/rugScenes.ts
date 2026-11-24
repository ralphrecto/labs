import { v } from '../../model/geometry'
import type { Vec2 } from '../../model/types'
import { rng, type Cloud } from '../cloud'

// Made-up floor scans for testing rug detection: a 5 x 4 m room of planks (running along x), with
// rugs, sunlight, shadows and furniture bases painted in, and noise from a dense LiDAR-like scan
// down to a sparse photo scan (an iPhone without LiDAR).

type RGB = [number, number, number]

export interface SceneRug {
  /** Rectangle: center, size and angle; or round: center and radius. */
  x: number
  y: number
  width?: number
  depth?: number
  angle?: number
  radius?: number
  /** Colour at a point in the rug's own frame (u along width, w along depth, both from the center). */
  color: RGB | ((u: number, w: number) => RGB)
  /** Height of the rug's top above the boards (m). */
  height?: number
}

export interface SceneLight {
  /** A rectangle of the floor lit brighter (gain > 1, a little warmer) or shaded (gain < 1). */
  x0: number
  y0: number
  x1: number
  y1: number
  gain: number
  /** Warm shift added to red and taken from blue (sunlight). */
  warm?: number
}

export interface SceneBox {
  /** A piece of furniture standing on the floor: its footprint is unseen, its base is seen around it. */
  x0: number
  y0: number
  x1: number
  y1: number
  color: RGB
  /** Darken the floor this far around it (contact shadow), m. */
  shadow?: number
}

export interface SceneOptions {
  floor?: RGB
  rugs?: SceneRug[]
  lights?: SceneLight[]
  boxes?: SceneBox[]
  /** Floor points per square meter. */
  density?: number
  /** Height noise (sigma, m) and colour noise (± per channel). */
  hNoise?: number
  cNoise?: number
  /** Share of points with a random wrong colour (photogrammetry speckle). */
  outliers?: number
  /** Whether the scan saw boards through the rug (the sample does; a real scan mostly doesn't). */
  boardsUnderRug?: boolean
  seed?: number
}

export const ROOM: Vec2[] = [v(0, 0), v(5, 0), v(5, 4), v(0, 4)]

export function rugScene(o: SceneOptions = {}): Cloud {
  const { floor = [0.62, 0.45, 0.3], rugs = [], lights = [], boxes = [], density = 3000, hNoise = 0.002, cNoise = 0.03, outliers = 0, boardsUnderRug = false, seed = 5 } = o
  const rand = rng(seed)
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(1e-9, rand()))) * Math.cos(2 * Math.PI * rand())
  const X: number[] = []
  const H: number[] = []
  const Z: number[] = []
  const C: number[] = []
  const push = (x: number, h: number, y: number, c: RGB) => {
    X.push(x)
    H.push(h + gauss() * hNoise)
    Z.push(y)
    if (rand() < outliers) C.push(rand(), rand(), rand())
    else {
      const j = (rand() - 0.5) * 2 * cNoise
      C.push(...(c.map((ch) => Math.max(0, Math.min(1, ch + j + (rand() - 0.5) * cNoise))) as RGB))
    }
  }
  // Planks 18 cm wide, each a slightly different shade, with a seam every 1.2-ish m along x.
  const plankShade = (x: number, y: number) => {
    const row = Math.floor(y / 0.18)
    const seg = Math.floor((x + ((row * 0.37) % 1.2)) / 1.2)
    const hsh = Math.sin(row * 12.9898 + seg * 78.233) * 43758.5453
    return 0.9 + 0.12 * (hsh - Math.floor(hsh)) - (row % 2 ? 0.03 : 0)
  }
  const lightAt = (x: number, y: number, c: RGB): RGB => {
    let out = c
    for (const l of lights)
      if (x >= l.x0 && x <= l.x1 && y >= l.y0 && y <= l.y1) {
        const w = l.warm ?? 0
        out = [out[0] * l.gain + w, out[1] * l.gain, out[2] * l.gain - w]
      }
    return out
  }
  const rugAt = (x: number, y: number): { c: RGB; h: number } | null => {
    for (const r of rugs) {
      const dx = x - r.x
      const dy = y - r.y
      const a = r.angle ?? 0
      const u = dx * Math.cos(a) + dy * Math.sin(a)
      const w = -dx * Math.sin(a) + dy * Math.cos(a)
      const inside = r.radius !== undefined ? u * u + w * w <= r.radius * r.radius : Math.abs(u) <= r.width! / 2 && Math.abs(w) <= r.depth! / 2
      if (!inside) continue
      return { c: typeof r.color === 'function' ? r.color(u, w) : r.color, h: r.height ?? 0.01 }
    }
    return null
  }
  const inBox = (x: number, y: number, pad = 0) => boxes.find((b) => x >= b.x0 - pad && x <= b.x1 + pad && y >= b.y0 - pad && y <= b.y1 + pad)
  const n = Math.round(20 * density)
  for (let i = 0; i < n; i++) {
    const x = rand() * 5
    const y = rand() * 4
    if (inBox(x, y)) continue
    const rug = rugAt(x, y)
    let shade = 1
    for (const b of boxes) if (b.shadow && inBox(x, y, b.shadow) === b) shade *= 0.62
    if (rug && (!boardsUnderRug || rand() < 0.5)) {
      push(x, rug.h, y, lightAt(x, y, rug.c.map((c) => c * shade) as RGB))
      continue
    }
    const s = plankShade(x, y) * shade
    push(x, 0, y, lightAt(x, y, [floor[0] * s, floor[1] * s, floor[2] * s]))
  }
  // Furniture bases: the bottom few centimetres of each side.
  for (const b of boxes) {
    const per = 2 * (b.x1 - b.x0 + b.y1 - b.y0)
    for (let i = 0; i < per * density * 0.08; i++) {
      let d = rand() * per
      const h = rand() * 0.04
      const wx = b.x1 - b.x0
      const wy = b.y1 - b.y0
      if (d < wx) push(b.x0 + d, h, b.y0, b.color)
      else if ((d -= wx) < wy) push(b.x1, h, b.y0 + d, b.color)
      else if ((d -= wy) < wx) push(b.x1 - d, h, b.y1, b.color)
      else push(b.x0, h, b.y1 - (d - wx), b.color)
    }
  }
  // Walls: pale paint down to the floor.
  for (let k = 0; k < 4; k++) {
    const a = ROOM[k]
    const b = ROOM[(k + 1) % 4]
    const L = Math.hypot(b.x - a.x, b.y - a.y)
    for (let i = 0; i < L * density * 0.1; i++) {
      const t = rand()
      push(a.x + (b.x - a.x) * t + gauss() * hNoise, rand() * 0.3, a.y + (b.y - a.y) * t + gauss() * hNoise, [0.9, 0.88, 0.84])
    }
  }
  return { n: X.length, x: Float32Array.from(X), h: Float32Array.from(H), z: Float32Array.from(Z), rgb: Float32Array.from(C) }
}
