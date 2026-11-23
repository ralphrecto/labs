import { describe, expect, it } from 'vitest'
import { rugTone } from '../ItemModel'
import { realFloor, toonFloor, type FloorImage } from '../../scan/palette'

/** HSL lightness of an sRGB hex colour. */
const L = (hex: string) => {
  const n = parseInt(hex.slice(1), 16)
  const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => v / 255)
  return (Math.max(...c) + Math.min(...c)) / 2
}

describe('rugTone', () => {
  it('darkens a sand rug that would vanish on light oak', () => {
    const out = rugTone('#c7a98a', '#cfae86')
    expect(L('#cfae86') - L(out)).toBeGreaterThanOrEqual(0.19)
  })
  it('leaves a rug that already stands out alone', () => {
    expect(rugTone('#3b4a6b', '#cfae86')).toBe('#3b4a6b')
  })
  it('lightens a rug on a dark floor', () => {
    const out = rugTone('#5a4030', '#5c4433')
    expect(L(out) - L('#5c4433')).toBeGreaterThanOrEqual(0.19)
  })
})

function img(w: number, h: number, f: (x: number, y: number) => [number, number, number] | null): FloorImage {
  const rgba = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const c = f(x, y)
      if (!c) continue
      rgba.set([...c, 255], (y * w + x) * 4)
    }
  return { x0: 0, y0: 0, cell: 0.05, width: w, height: h, rgba }
}

describe('scan floor looks', () => {
  // A speckled floor: every other cell a little darker.
  const speckle = img(20, 16, (x, y) => ((x + y) % 2 ? [180, 120, 70] : [200, 140, 90]))

  it('Cartoon floor smooths speckle into one flat colour and keeps the same extent', () => {
    const t = toonFloor(speckle)
    expect(t.width * t.cell).toBeCloseTo(speckle.width * speckle.cell)
    const colors = new Set<string>()
    for (let k = 0; k < t.width * t.height; k++) colors.add(`${t.rgba[k * 4]},${t.rgba[k * 4 + 1]},${t.rgba[k * 4 + 2]}`)
    expect(colors.size).toBe(1)
  })

  it('Cartoon floor keeps unseen cells unseen', () => {
    const holes = img(10, 10, (x) => (x < 5 ? [200, 140, 90] : null))
    const t = toonFloor(holes)
    const up = t.width / holes.width
    expect(t.rgba[(t.width - 1) * 4 + 3]).toBe(0)
    expect(t.rgba[Math.floor(up * 2) * 4 + 3]).toBe(255)
  })

  it('Lifelike floor stretches a faint rug away from the boards', () => {
    const faint = img(20, 20, (x, y) => (x > 5 && x < 14 && y > 5 && y < 14 ? [190, 160, 120] : [170, 130, 90]))
    const r = realFloor(faint)
    const at = (i: FloorImage, x: number, y: number) => i.rgba[(y * i.width + x) * 4 + 1]
    expect(at(r, 10, 10) - at(r, 1, 1)).toBeGreaterThan(at(faint, 10, 10) - at(faint, 1, 1))
  })
})
