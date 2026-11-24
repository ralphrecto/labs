import { describe, expect, it } from 'vitest'
import type { Room } from '../../model/types'
import { scanPalette, toonHex, toonRGB } from '../palette'

const room = {
  id: 'r1', name: 'Room', ceilingHeight: 2.5, wallThickness: 0.1, openings: [],
  polygon: [ { x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 3 }, { x: 0, y: 3 } ],
} as unknown as Room

/** A box room: red floor, a green wall along y = 0, grey elsewhere. */
function boxScan() {
  const pos: number[] = []
  const col: number[] = []
  for (let x = 0.05; x < 4; x += 0.05)
    for (let y = 0.05; y < 3; y += 0.05) {
      pos.push(x, 0, y)
      col.push(0.8, 0.2, 0.2)
    }
  for (let x = 0.05; x < 4; x += 0.05)
    for (let h = 0.4; h < 2; h += 0.05) {
      pos.push(x, h, 0.02)
      col.push(0.2, 0.7, 0.3)
      pos.push(x, h, 2.98)
      col.push(0.6, 0.6, 0.6)
    }
  return { count: pos.length / 3, positions: Float32Array.from(pos), colors: Float32Array.from(col) }
}

describe('scan palette', () => {
  it('reads wall and floor colors from the scan', () => {
    const p = scanPalette([boxScan()], [room])
    expect(p.walls.r1[0]).toMatch(/^#33b[23]4d$/)
    expect(p.walls.r1[2]).toBe('#999999')
    expect(p.walls.r1[1]).toBeUndefined()
    expect(p.floorColor.r1).toBe('#cc3333')
    expect(p.floor!.width).toBeGreaterThan(70)
    const mid = (Math.floor(p.floor!.height / 2) * p.floor!.width + Math.floor(p.floor!.width / 2)) * 4
    expect(p.floor!.rgba[mid]).toBeGreaterThan(190)
    expect(p.floor!.rgba[mid + 3]).toBe(255)
  })

  it('lets a rug lying on the boards show through, not the boards seen under it', () => {
    // Dense red boards everywhere, and a sparse blue rug 1 cm up over x, y in 1..3 (a scan sees the boards through it).
    const pos: number[] = []
    const col: number[] = []
    let seed = 3
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    for (let i = 0; i < 40000; i++) {
      pos.push(rand() * 4, (rand() - 0.5) * 0.006, rand() * 3)
      col.push(0.8, 0.2, 0.2)
    }
    for (let i = 0; i < 1200; i++) {
      pos.push(1 + rand() * 2, 0.01 + (rand() - 0.5) * 0.006, 1 + rand())
      col.push(0.2, 0.3, 0.8)
    }
    // A sofa's foot standing on the boards: too tall to be a rug.
    for (let i = 0; i < 600; i++) {
      pos.push(0.2 + rand() * 0.4, 0.035 + rand() * 0.02, 0.2 + rand() * 0.4)
      col.push(0.1, 0.8, 0.1)
    }
    const p = scanPalette([{ count: pos.length / 3, positions: Float32Array.from(pos), colors: Float32Array.from(col) }], [room])
    const f = p.floor!
    const at = (x: number, y: number) => {
      const k = (Math.floor((y - f.y0) / f.cell) * f.width + Math.floor((x - f.x0) / f.cell)) * 4
      return [f.rgba[k], f.rgba[k + 1], f.rgba[k + 2]]
    }
    const [r, , b] = at(2, 1.5)
    expect(b).toBeGreaterThan(r) // the rug's blue, not the boards' red
    expect(b).toBeGreaterThan(170)
    const [r2, g2] = at(3.5, 2.5)
    expect(r2).toBeGreaterThan(190) // plain boards stay red
    expect(at(0.4, 0.4)[1]).toBeLessThan(80) // the sofa foot doesn't paint the floor green
    expect(r2 - g2).toBeGreaterThan(120)
  })

  it('returns nothing for scans without colors', () => {
    const s = boxScan()
    const p = scanPalette([{ ...s, colors: undefined }], [room])
    expect(p.floor).toBeNull()
    expect(p.walls).toEqual({})
  })

  it('makes cartoon colors brighter and more saturated, snapping lightness to bands', () => {
    const [r, g, b] = toonRGB(0.45, 0.38, 0.33)
    expect(Math.max(r, g, b) - Math.min(r, g, b)).toBeGreaterThan(0.45 - 0.33)
    const light = (c: number[]) => (Math.max(...c) + Math.min(...c)) / 2
    expect(light(toonRGB(0.46, 0.4, 0.33))).toBeCloseTo(light(toonRGB(0.47, 0.4, 0.34)), 6)
    expect(toonHex('not-a-color')).toBe('not-a-color')
  })
})
