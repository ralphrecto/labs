import { describe, expect, it } from 'vitest'
import { findRugs } from '../analyze'
import { ROOM, rugScene, type SceneOptions } from './rugScenes'

// Rug detection on made-up floors: real rugs are found at about their size, and sunlight, shadows,
// furniture bases and scan noise don't become rugs. Missing a faint rug is fine; inventing one isn't.

const detect = (o: SceneOptions) => {
  const c = rugScene(o)
  return findRugs(c, c.x, c.z, ROOM)
}
const sparse: SceneOptions = { density: 260, hNoise: 0.007, cNoise: 0.08, outliers: 0.05 }
// Sparser still: under 2 points per 10 cm square, a centimetre of height noise, a tenth of the colours wrong.
const sparser: SceneOptions = { density: 120, hNoise: 0.012, cNoise: 0.1, outliers: 0.08 }
const sizeOf = (r: { width: number; depth: number }) => [r.width, r.depth]
const near = (got: number, want: number, tol = 0.25) => Math.abs(got - want) <= tol

describe('no rug where there is none', () => {
  const cases: [string, SceneOptions][] = [
    ['plain boards', {}],
    ['plain boards, sparse photo scan', sparse],
    ['plain boards, sparse photo scan, another seed', { ...sparse, seed: 9 }],
    ['a big sunlit patch', { lights: [{ x0: 1, y0: 0.5, x1: 2.8, y1: 1.9, gain: 1.4, warm: 0.04 }] }],
    ['a long window streak of sun', { lights: [{ x0: 0.4, y0: 2.6, x1: 4.2, y1: 3.4, gain: 1.55, warm: 0.05 }] }],
    ['a deep shadow', { lights: [{ x0: 2, y0: 1.5, x1: 4, y1: 3.2, gain: 0.55 }] }],
    ['sun on a sparse scan', { ...sparse, lights: [{ x0: 1, y0: 0.5, x1: 2.8, y1: 1.9, gain: 1.45, warm: 0.04 }] }],
    [
      'furniture bases and their shadows',
      {
        boxes: [
          { x0: 1, y0: 0.15, x1: 3.2, y1: 1.05, color: [0.2, 0.26, 0.45], shadow: 0.25 },
          { x0: 3, y0: 2, x1: 4.6, y1: 3.9, color: [0.92, 0.9, 0.86], shadow: 0.2 },
          { x0: 0.2, y0: 2.6, x1: 0.65, y1: 3.8, color: [0.3, 0.2, 0.14], shadow: 0.3 },
        ],
      },
    ],
    ['shade lit by a blue sky', { lights: [{ x0: 1, y0: 0.5, x1: 2.8, y1: 1.9, gain: 0.6, warm: -0.03 }] }],
    ['sun on a grey floor', { floor: [0.6, 0.6, 0.6], lights: [{ x0: 1, y0: 0.5, x1: 2.8, y1: 1.9, gain: 1.4, warm: 0.04 }] }],
    ['plain boards, very sparse scan', sparser],
    ['sun on a very sparse scan', { ...sparser, lights: [{ x0: 1, y0: 0.5, x1: 2.8, y1: 1.9, gain: 1.5, warm: 0.05 }] }],
    [
      'furniture on a sparse scan',
      {
        ...sparse,
        boxes: [
          { x0: 1, y0: 0.15, x1: 3.2, y1: 1.05, color: [0.2, 0.26, 0.45], shadow: 0.25 },
          { x0: 3, y0: 2, x1: 4.6, y1: 3.9, color: [0.92, 0.9, 0.86], shadow: 0.2 },
        ],
      },
    ],
    ['a dark floor in shadow', { floor: [0.25, 0.17, 0.12], lights: [{ x0: 1.5, y0: 1, x1: 3.5, y1: 2.6, gain: 0.5 }] }],
  ]
  for (const [name, o] of cases)
    it(name, () => {
      for (const seed of [1, 2, 3]) expect(detect({ ...o, seed })).toEqual([])
    })
})

describe('real rugs are found at their size', () => {
  it('a rug close to the floor colour', () => {
    const rugs = detect({ rugs: [{ x: 2.5, y: 2, width: 2, depth: 1.4, color: [0.74, 0.58, 0.44] }] })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 2)).toBe(true)
    expect(near(rugs[0].depth, 1.4)).toBe(true)
  })

  it('a rug almost the floor colour is missed rather than misread', () => {
    const rugs = detect({ rugs: [{ x: 2.5, y: 2, width: 2, depth: 1.4, color: [0.65, 0.47, 0.32] }] })
    expect(rugs.length).toBeLessThanOrEqual(1)
    for (const r of rugs) expect(r.width).toBeLessThan(2.4)
  })

  it('a striped rug is one rug, not a rug per stripe', () => {
    const stripe = (u: number): [number, number, number] => (Math.floor((u + 5) / 0.2) % 2 ? [0.88, 0.84, 0.74] : [0.2, 0.25, 0.42])
    const rugs = detect({ rugs: [{ x: 2.4, y: 2.1, width: 2.2, depth: 1.5, color: stripe }] })
    expect(rugs).toHaveLength(1)
    expect(sizeOf(rugs[0]).map((s, i) => near(s, [2.2, 1.5][i]))).toEqual([true, true])
  })

  it('a patterned rug (a border and a busy middle)', () => {
    const pattern = (u: number, w: number): [number, number, number] => {
      if (Math.abs(u) > 1.0 || Math.abs(w) > 0.6) return [0.55, 0.12, 0.12]
      return (Math.floor(u / 0.15) + Math.floor(w / 0.15)) % 2 ? [0.85, 0.75, 0.55] : [0.2, 0.3, 0.45]
    }
    const rugs = detect({ rugs: [{ x: 2.5, y: 2, width: 2.3, depth: 1.5, color: pattern }] })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 2.3)).toBe(true)
    expect(near(rugs[0].depth, 1.5)).toBe(true)
  })

  it('a round rug', () => {
    const rugs = detect({ rugs: [{ x: 2.6, y: 2.2, radius: 0.8, color: [0.5, 0.6, 0.48] }] })
    expect(rugs).toHaveLength(1)
    // A round rug becomes a square-ish rug about its diameter.
    expect(rugs[0].width).toBeGreaterThan(1.25)
    expect(rugs[0].width).toBeLessThan(1.75)
    expect(rugs[0].depth).toBeGreaterThan(1.25)
  })

  it('two rugs', () => {
    const rugs = detect({
      rugs: [
        { x: 1.4, y: 1.2, width: 2, depth: 1.4, color: [0.45, 0.5, 0.58] },
        { x: 3.8, y: 3.0, width: 1.3, depth: 0.85, color: [0.62, 0.18, 0.16] },
      ],
    })
    expect(rugs).toHaveLength(2)
    const big = rugs.find((r) => r.x < 2.5)!
    const small = rugs.find((r) => r.x > 2.5)!
    expect(near(big.width, 2) && near(big.depth, 1.4)).toBe(true)
    expect(near(small.width, 1.3) && near(small.depth, 0.85)).toBe(true)
  })

  it('a turned rug', () => {
    const rugs = detect({ rugs: [{ x: 2.5, y: 2, width: 2, depth: 1.3, angle: 0.45, color: [0.82, 0.8, 0.74] }] })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 2) && near(rugs[0].depth, 1.3)).toBe(true)
    expect(Math.abs(rugs[0].rotation - 0.45)).toBeLessThan(0.12)
  })

  it('a dark rug on a dark floor', () => {
    const rugs = detect({ floor: [0.25, 0.17, 0.12], rugs: [{ x: 2.5, y: 2, width: 2.4, depth: 1.6, color: [0.12, 0.12, 0.14] }] })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 2.4) && near(rugs[0].depth, 1.6)).toBe(true)
  })

  it('a rug the scan sees boards through (like the sample)', () => {
    const rugs = detect({ boardsUnderRug: true, rugs: [{ x: 2.5, y: 2, width: 2.2, depth: 1.5, color: [0.78, 0.7, 0.55] }] })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 2.2) && near(rugs[0].depth, 1.5)).toBe(true)
  })

  it('a rug in front of a sofa, with the sofa and its shadow around', () => {
    const rugs = detect({
      boxes: [{ x0: 1, y0: 0.15, x1: 3.4, y1: 1.05, color: [0.2, 0.26, 0.45], shadow: 0.25 }],
      rugs: [{ x: 2.2, y: 2.1, width: 2.2, depth: 1.4, color: [0.8, 0.76, 0.66] }],
    })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 2.2) && near(rugs[0].depth, 1.4)).toBe(true)
  })

  it('a rug on a sparse, noisy photo scan, with sun nearby', () => {
    const rugs = detect({
      ...sparse,
      lights: [{ x0: 3.6, y0: 0.3, x1: 4.8, y1: 1.6, gain: 1.4, warm: 0.04 }],
      rugs: [{ x: 2, y: 2.3, width: 2.2, depth: 1.5, color: [0.42, 0.48, 0.6] }],
    })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 2.2, 0.35) && near(rugs[0].depth, 1.5, 0.35)).toBe(true)
  })

  it('a thin flat-weave rug, barely above the boards', () => {
    const rugs = detect({ rugs: [{ x: 2.5, y: 2, width: 2, depth: 1.4, height: 0.005, color: [0.74, 0.58, 0.44] }] })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 2) && near(rugs[0].depth, 1.4)).toBe(true)
  })

  it('a small mat', () => {
    const rugs = detect({ rugs: [{ x: 2.5, y: 2, width: 0.9, depth: 0.6, color: [0.42, 0.48, 0.6] }] })
    expect(rugs).toHaveLength(1)
    expect(near(rugs[0].width, 0.9, 0.12) && near(rugs[0].depth, 0.6, 0.12)).toBe(true)
  })

  it('on a very sparse scan a rug is found smaller or not at all, never bigger', () => {
    let found = 0
    for (const seed of [1, 2, 3]) {
      const rugs = detect({ ...sparser, seed, rugs: [{ x: 2.5, y: 2, width: 2, depth: 1.4, color: [0.42, 0.48, 0.6] }] })
      expect(rugs.length).toBeLessThanOrEqual(1)
      for (const r of rugs) {
        found++
        expect(r.width).toBeLessThan(2.2)
        expect(r.depth).toBeLessThan(1.6)
        expect(Math.hypot(r.x - 2.5, r.y - 2)).toBeLessThan(0.4)
      }
    }
    expect(found).toBeGreaterThanOrEqual(2)
  })
})
