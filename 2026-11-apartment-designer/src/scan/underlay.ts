import type { ScanData } from './types'

// A top-down picture of a scan for the plan view: what's on the floor (furniture, walls), seen
// from above, in the scan's colors. The ceiling is left out so you can see in.

export interface TopDown {
  url: string
  /** World (x, z) of the image's top-left corner, and its size in meters. */
  x: number
  z: number
  w: number
  h: number
}

const cache = new WeakMap<ScanData, TopDown | null>()

export function topDownImage(scan: ScanData, ceiling = 2.6, res = 0.015): TopDown | null {
  if (cache.has(scan)) return cache.get(scan)!
  const out = render(scan, ceiling, res)
  cache.set(scan, out)
  return out
}

function render(scan: ScanData, ceiling: number, res: number): TopDown | null {
  if (typeof document === 'undefined') return null
  const { count, positions, colors } = scan.points
  let minX = Infinity
  let minZ = Infinity
  let maxX = -Infinity
  let maxZ = -Infinity
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3]
    const z = positions[i * 3 + 2]
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (z < minZ) minZ = z
    if (z > maxZ) maxZ = z
  }
  if (!Number.isFinite(minX)) return null
  // Keep the canvas under ~2048 px a side.
  const step = Math.max(res, (maxX - minX) / 2048, (maxZ - minZ) / 2048)
  const W = Math.max(1, Math.ceil((maxX - minX) / step))
  const H = Math.max(1, Math.ceil((maxZ - minZ) / step))
  // Per pixel: keep the highest point below the ceiling (that's what you'd see from above).
  const top = new Float32Array(W * H).fill(-Infinity)
  const rgb = new Uint8ClampedArray(W * H * 4)
  const cut = ceiling - 0.12
  for (let i = 0; i < count; i++) {
    const hgt = positions[i * 3 + 1]
    if (hgt > cut || hgt < -0.1) continue
    const px = Math.min(W - 1, Math.floor((positions[i * 3] - minX) / step))
    const pz = Math.min(H - 1, Math.floor((positions[i * 3 + 2] - minZ) / step))
    const k = pz * W + px
    if (hgt <= top[k]) continue
    top[k] = hgt
    if (colors) {
      rgb[k * 4] = colors[i * 3] * 255
      rgb[k * 4 + 1] = colors[i * 3 + 1] * 255
      rgb[k * 4 + 2] = colors[i * 3 + 2] * 255
    } else {
      // No colors: shade by height, dark = tall.
      const g = 230 - Math.min(1, hgt / 2.2) * 170
      rgb[k * 4] = g
      rgb[k * 4 + 1] = g
      rgb[k * 4 + 2] = g
    }
    rgb[k * 4 + 3] = 255
  }
  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.putImageData(new ImageData(rgb, W, H), 0, 0)
  return { url: canvas.toDataURL('image/png'), x: minX, z: minZ, w: W * step, h: H * step }
}
