import { describe, expect, it } from 'vitest'
import { decorBadgeCenter, wallLabelSpot } from '../components/PlanView'

// A 4 m north wall on y = 0, the room below it (inward normal +y), 1 screen px = 1 cm.
const px = 0.01
const wallT = 0.1
const n = { x: 0, y: 1 }
const mid = { x: 2, y: 0 }
const badgeR = 5.5 * px
const fs = 12 * px
// A "3.00 m" label lying along the wall.
const hx = (6 * fs * 0.92 * 0.58) / 2 + 4 * px
const hy = fs * 0.6
const boxAt = (q: { x: number; y: number }) => ({ minX: q.x - hx, maxX: q.x + hx, minY: q.y - hy, maxY: q.y + hy })
const overlaps = (a: ReturnType<typeof boxAt>, b: ReturnType<typeof boxAt>) => a.maxX > b.minX && a.minX < b.maxX && a.maxY > b.minY && a.minY < b.maxY
const badgeBox = (c: { x: number; y: number }) => ({ minX: c.x - badgeR, maxX: c.x + badgeR, minY: c.y - badgeR, maxY: c.y + badgeR })
const base = { mid, n, wallT, decorWallT: wallT, px, badgeR, halfN: hy, boxAt }

describe('decorBadgeCenter', () => {
  it('sits outside the wall behind the piece, on any wall', () => {
    const c = decorBadgeCenter({ x: 2, y: 0.02, rotation: 0 }, 0.04, wallT, px, badgeR)
    expect(c.x).toBeCloseTo(2)
    expect(c.y).toBeCloseTo(0 - wallT - 3 * px - badgeR)
    // East wall at x = 4, back facing +x.
    const e = decorBadgeCenter({ x: 3.98, y: 1, rotation: Math.PI / 2 }, 0.04, wallT, px, badgeR)
    expect(e.x).toBeCloseTo(4 + wallT + 3 * px + badgeR)
    expect(e.y).toBeCloseTo(1)
  })
})

describe('wallLabelSpot', () => {
  it('keeps the length just outside a bare wall', () => {
    const m = wallLabelSpot({ ...base, badges: [] })!
    expect(m.y).toBeCloseTo(-(wallT + 10 * px))
  })
  it('steps the length out past a print hung at the middle of the wall, clear of its badge', () => {
    const badge = badgeBox(decorBadgeCenter({ x: 2, y: 0.02, rotation: 0 }, 0.04, wallT, px, badgeR))
    const m = wallLabelSpot({ ...base, badges: [badge] })!
    expect(m.y).toBeLessThan(-(wallT + 10 * px))
    expect(overlaps(boxAt(m), badge)).toBe(false)
  })
  it('stays put when the decor hangs far along the wall', () => {
    const badge = badgeBox(decorBadgeCenter({ x: 0.4, y: 0.02, rotation: 0 }, 0.04, wallT, px, badgeR))
    expect(wallLabelSpot({ ...base, badges: [badge] })!.y).toBeCloseTo(-(wallT + 10 * px))
  })
  it('goes inside when both outside spots are taken, and nowhere when nothing is free', () => {
    const badge = badgeBox(decorBadgeCenter({ x: 2, y: 0.02, rotation: 0 }, 0.04, wallT, px, badgeR))
    const m = wallLabelSpot({ ...base, badges: [badge], ok: (q) => q.y > 0 })!
    expect(m.y).toBeCloseTo(14 * px)
    expect(wallLabelSpot({ ...base, badges: [badge], ok: () => false })).toBeNull()
  })
})
