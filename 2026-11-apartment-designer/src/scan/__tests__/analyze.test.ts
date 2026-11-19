import { describe, expect, it } from 'vitest'
import { polygonArea } from '../../model/geometry'
import { analyzeScan, simplifyRectilinear } from '../analyze'
import { classifyBox } from '../classify'
import { STUDIO, syntheticScan } from '../synthetic'
import { v } from '../../model/geometry'

describe('scan analysis on the synthetic studio', () => {
  const { estimate } = analyzeScan(syntheticScan())
  it('finds the room outline', () => {
    const room = estimate.room!
    expect(room).toBeTruthy()
    expect(room.polygon.length).toBe(6)
    expect(polygonArea(room.polygon)).toBeCloseTo(polygonArea(STUDIO.polygon), 0)
    expect(Math.abs(polygonArea(room.polygon) - polygonArea(STUDIO.polygon))).toBeLessThan(0.4)
    expect(room.ceilingHeight).toBeCloseTo(2.6, 1)
  })
  it('finds the door and window', () => {
    const ops = estimate.room!.openings
    expect(ops.filter((o) => o.kind === 'door')).toHaveLength(1)
    expect(ops.filter((o) => o.kind === 'window')).toHaveLength(1)
    const door = ops.find((o) => o.kind === 'door')!
    expect(door.width).toBeGreaterThan(0.75)
    expect(door.width).toBeLessThan(0.95)
  })
  it('finds the furniture with sizes', () => {
    expect(estimate.objects).toHaveLength(STUDIO.boxes.length)
    const cats = estimate.objects.map((o) => o.category).sort()
    const truth = STUDIO.boxes.map((b) => b.category).sort()
    let hits = 0
    for (const c of new Set(truth)) hits += Math.min(cats.filter((x) => x === c).length, truth.filter((x) => x === c).length)
    expect(hits).toBeGreaterThanOrEqual(8)
    const bed = estimate.objects.find((o) => o.category === 'bed')!
    expect(bed.width).toBeCloseTo(1.52, 1)
    expect(bed.depth).toBeCloseTo(2.03, 1)
    expect(bed.name).toBe('Queen bed')
  })
})

describe('scan analysis is robust to how the phone held the scan', () => {
  it.each([
    ['z up', { upAxis: 'z' as const, seed: 5 }],
    ['turned 70°', { yaw: 1.22, tilt: 3, seed: 9 }],
    ['sparse and noisy', { density: 800, noise: 0.012, seed: 13 }],
  ])('%s', (_, opts) => {
    const { estimate } = analyzeScan(syntheticScan(opts))
    expect(estimate.room).toBeTruthy()
    expect(Math.abs(polygonArea(estimate.room!.polygon) - polygonArea(STUDIO.polygon))).toBeLessThan(0.6)
    expect(estimate.objects.length).toBeGreaterThanOrEqual(STUDIO.boxes.length - 1)
    expect(estimate.objects.length).toBeLessThanOrEqual(STUDIO.boxes.length + 1)
    expect(estimate.objects.some((o) => o.category === 'bed')).toBe(true)
  })
})

describe('rectilinear simplification', () => {
  it('removes a small jog', () => {
    const poly = [v(0, 0), v(2, 0), v(2, 1), v(2.05, 1), v(2.05, 2), v(0, 2)]
    const s = simplifyRectilinear(poly, 0.2)
    expect(s).toHaveLength(4)
  })
})

describe('classifier', () => {
  it('knows the obvious ones', () => {
    expect(classifyBox(2.03, 1.52, 0.55).category).toBe('bed')
    expect(classifyBox(2.1, 0.92, 0.85).category).toBe('sofa')
    expect(classifyBox(1.2, 0.6, 2.0).category).toBe('wardrobe')
    expect(classifyBox(0.3, 0.3, 0.05).category).toBe('box')
  })
})
