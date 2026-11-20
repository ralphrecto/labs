import { describe, expect, it } from 'vitest'
import { snapCorner } from '../scan/align'
import type { Vec2 } from '../model/types'

// A 5 m × 4 m room outline, walls 4 cm thick, sampled like the scan underlay (0..255).
const W = 5
const H = 4
const wall = (p: Vec2) => {
  const inX = p.x > -0.02 && p.x < W + 0.02
  const inY = p.y > -0.02 && p.y < H + 0.02
  const onV = inY && (Math.abs(p.x) < 0.02 || Math.abs(p.x - W) < 0.02)
  const onH = inX && (Math.abs(p.y) < 0.02 || Math.abs(p.y - H) < 0.02)
  return onV || onH ? 200 : 0
}

describe('snapCorner', () => {
  it('pulls a tap near a corner onto it', () => {
    const c = snapCorner(wall, { x: 0.18, y: 0.12 }, 0.3, 0.01)
    expect(c).not.toBeNull()
    expect(c!.x).toBeCloseTo(0, 1)
    expect(c!.y).toBeCloseTo(0, 1)
    const d = snapCorner(wall, { x: W - 0.2, y: -0.1 }, 0.3, 0.01)
    expect(d!.x).toBeCloseTo(W, 1)
    expect(d!.y).toBeCloseTo(0, 1)
  })
  it('measures the whole wall from two loose taps', () => {
    const a = snapCorner(wall, { x: 0.2, y: 0.05 }, 0.3, 0.01)!
    const b = snapCorner(wall, { x: W - 0.22, y: 0.04 }, 0.3, 0.01)!
    expect(Math.hypot(b.x - a.x, b.y - a.y)).toBeCloseTo(W, 1)
  })
  it('finds the end of a lone wall', () => {
    const lone = (p: Vec2) => (Math.abs(p.y) < 0.02 && p.x > 0 && p.x < 3 ? 200 : 0)
    const e = snapCorner(lone, { x: 2.8, y: 0.03 }, 0.3, 0.01)!
    expect(e.x).toBeCloseTo(3, 1)
    expect(e.y).toBeCloseTo(0, 1)
  })
  it('leaves taps along the middle of a wall or in open floor alone', () => {
    expect(snapCorner(wall, { x: 2.5, y: 0.05 }, 0.3, 0.01)).toBeNull()
    expect(snapCorner(wall, { x: 2.5, y: 2 }, 0.3, 0.01)).toBeNull()
  })
})
