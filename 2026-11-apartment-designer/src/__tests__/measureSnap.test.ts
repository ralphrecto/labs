import { describe, expect, it } from 'vitest'
import { measureSkewed, offAxisDeg, pickMeasureSnap } from '../components/PlanView'

// A 5 m wall along y = 0 with a corner at (0, 0); the corner magnet reaches 0.6 m.
const R = 0.6
const corner = { x: 0, y: 0 }

describe('scan measure snapping', () => {
  it('a tap near a corner still snaps to it', () => {
    expect(pickMeasureSnap({ tap: { x: 0.15, y: 0.04 }, corner, wall: { y: 0 }, cornerRadius: R })).toBe(corner)
    // At a corner both walls light up: the corner wins.
    expect(pickMeasureSnap({ tap: { x: 0.05, y: 0.05 }, corner, wall: { x: 0, y: 0 }, cornerRadius: R })).toBe(corner)
  })

  it('a tap well inside the wall span goes onto the wall, not the corner', () => {
    const p = pickMeasureSnap({ tap: { x: 0.45, y: 0.05 }, corner, wall: { y: 0 }, cornerRadius: R })
    expect(p).toEqual({ x: 0.45, y: 0 })
    const q = pickMeasureSnap({ tap: { x: 0.04, y: 2.5 }, corner: { x: 0, y: 2.1 }, wall: { x: 0 }, cornerRadius: R })
    expect(q).toEqual({ x: 0, y: 2.5 })
  })

  it('the second tap prefers the wall when the corner would make a slanted line', () => {
    // First end on the wall y = 0 at x = 3; a corner of another wall at (0.25, 0.3) is nearby.
    const anchor = { x: 3, y: 0 }
    const p = pickMeasureSnap({ tap: { x: 0.2, y: 0.04 }, corner: { x: 0.25, y: 0.3 }, wall: { y: 0 }, anchor, cornerRadius: R })
    expect(p).toEqual({ x: 0.2, y: 0 })
    // But a square corner is kept.
    const c = { x: 0, y: 0 }
    expect(pickMeasureSnap({ tap: { x: 0.2, y: 0.04 }, corner: c, wall: { y: 0 }, anchor, cornerRadius: R })).toBe(c)
  })

  it('falls back to the wall hit or the tap itself', () => {
    expect(pickMeasureSnap({ tap: { x: 1, y: 0.05 }, corner: null, wall: { y: 0 }, cornerRadius: R })).toEqual({ x: 1, y: 0 })
    expect(pickMeasureSnap({ tap: { x: 1, y: 1 }, corner: null, wall: {}, cornerRadius: R })).toEqual({ x: 1, y: 1 })
  })

  it('flags lines more than 5 degrees off the wall axes', () => {
    expect(offAxisDeg({ x: 0, y: 0 }, { x: 3, y: 0 })).toBeCloseTo(0)
    expect(offAxisDeg({ x: 0, y: 0 }, { x: 0, y: -2 })).toBeCloseTo(0)
    expect(offAxisDeg({ x: 0, y: 0 }, { x: 1, y: 1 })).toBeCloseTo(45)
    expect(measureSkewed({ x: 0, y: 0 }, { x: 3, y: 0.2 })).toBe(false) // ~3.8°
    expect(measureSkewed({ x: 0, y: 0 }, { x: 3, y: 0.4 })).toBe(true) // ~7.6°
    expect(measureSkewed({ x: 0, y: 0 }, { x: 0.1, y: 0.1 })).toBe(false) // too short to judge
    expect(measureSkewed({ x: 0, y: 0 }, null)).toBe(false)
  })
})
