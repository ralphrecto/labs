import { describe, expect, it } from 'vitest'
import { backAgainstWindow, checkLayout } from '../checks'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { findDrop } from '../placement'
import { emptyDoc, rectRoom, sampleDoc } from '../sample'
import type { Doc, Item, Placement } from '../types'
import { findWallDrop } from '../wallDecor'

// Round 16: an interior designer's notes on the sample and on auto-placement.
const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0
function put(d: Doc, category: string, x: number, y: number, rotation = 0, over: Partial<Item> = {}): Placement {
  const item = itemFromCatalog(catalogEntry(category), `dp${++n}`, over)
  const p: Placement = { id: `dpp${n}`, itemId: item.id, x, y, rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}
function drop(d: Doc, category: string, over: Partial<Item> = {}) {
  const item = itemFromCatalog(catalogEntry(category), `dd${++n}`, over)
  const r = findDrop(d, layoutOf(d), item)
  if (!r.ok) return null
  const p: Placement = { id: `ddp${n}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}
const item = (d: Doc, p: Placement) => d.items.find((i) => i.id === p.itemId)!

/** A 5 × 4 m living room, a window in the middle of the top wall (y = 0). */
function living(): Doc {
  const d = emptyDoc()
  const room = rectRoom('Living room', 0, 0, 5, 4)
  room.openings.push({ id: 'win', kind: 'window', wall: 0, offset: 1.7, width: 1.6, height: 1.4, sill: 0.8 })
  d.rooms.push(room)
  return d
}

describe('TV against a window', () => {
  it('flags a TV backed onto a window, not one on the solid wall beside it', () => {
    const d = living()
    const tv = put(d, 'tv-stand', 2.5, 0.225)
    expect(checkLayout(d, layoutOf(d)).some((i) => i.kind === 'window' && i.severity === 'warning' && i.placementIds[0] === tv.id)).toBe(true)
    expect(backAgainstWindow(d.rooms, item(d, tv), tv)).not.toBeNull()
    tv.x = 0.85
    expect(checkLayout(d, layoutOf(d)).some((i) => i.kind === 'window')).toBe(false)
    // Facing the window from across the room isn't "under" it.
    tv.x = 2.5
    tv.y = 4 - 0.225
    tv.rotation = Math.PI
    expect(checkLayout(d, layoutOf(d)).some((i) => i.kind === 'window')).toBe(false)
  })

  it('auto-placement puts the TV on a solid wall the sofa faces, not under the window', () => {
    const d = living()
    // A window left of the middle: the sofa against the bottom wall looks up at it.
    Object.assign(d.rooms[0].openings[0], { offset: 1.0, width: 1.2 })
    const sofa = put(d, 'sofa', 2.3, 4 - 0.46, Math.PI)
    const tv = drop(d, 'tv-stand')!
    expect(backAgainstWindow(d.rooms, item(d, tv), tv)).toBeNull()
    expect(checkLayout(d, layoutOf(d)).some((i) => i.kind === 'window')).toBe(false)
    // Still on the wall the sofa looks at, within the 25° the sofa-faces-TV goal allows.
    expect(tv.y).toBeLessThan(0.3)
    expect(Math.abs(Math.atan2(tv.x - sofa.x, sofa.y - tv.y))).toBeLessThan((25 * Math.PI) / 180)
  })

  it('the sample home has no TV under a window, and its sofa still faces the TV', () => {
    for (const settled of [false, true]) {
      const d = sampleDoc({ settled })
      expect(checkLayout(d, layoutOf(d))).toEqual([])
      const l = layoutOf(d)
      const tv = l.placements.find((p) => item(d, p).category === 'tv-stand')!
      const sofa = l.placements.find((p) => item(d, p).category === 'sofa')!
      // Front of the sofa (local +y) points at the TV.
      const fx = -Math.sin(sofa.rotation)
      const fy = Math.cos(sofa.rotation)
      const dx = tv.x - sofa.x
      const dy = tv.y - sofa.y
      expect((fx * dx + fy * dy) / Math.hypot(dx, dy)).toBeGreaterThan(0.95)
    }
  })
})

describe('seats for two', () => {
  it('the second dining chair goes across the table from the first, not beside it', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Dining room', 0, 0, 4, 4))
    const table = put(d, 'dining-table', 2, 2)
    const a = drop(d, 'dining-chair')!
    const b = drop(d, 'dining-chair')!
    // Table-local: the two chairs sit on opposite sides, and straight across from each other.
    const ay = a.y - table.y
    const by = b.y - table.y
    const ax = a.x - table.x
    const bx = b.x - table.x
    const sameSide = Math.sign(Math.abs(ay) > Math.abs(ax) ? ay : ax) === Math.sign(Math.abs(by) > Math.abs(bx) ? by : bx) && Math.abs(ay) > Math.abs(ax) === Math.abs(by) > Math.abs(bx)
    expect(sameSide).toBe(false)
    expect(Math.min(Math.abs(ax - bx), Math.abs(ay - by))).toBeLessThan(0.05)
  })
})

describe('art over a sofa or bed', () => {
  it('hangs dead centre over the sofa', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    // Off the wall's middle, at an odd spot the 20 cm sampling misses.
    const sofa = put(d, 'sofa', 1.93, 4 - 0.46, Math.PI)
    const r = findWallDrop(d, layoutOf(d), { ...itemFromCatalog(catalogEntry('wall-art'), 'art'), width: 1.0, height: 0.7 })
    expect(r.ok).toBe(true)
    if (r.ok) {
      expect(r.spot.x).toBeCloseTo(sofa.x, 2)
      expect(r.spot.y).toBeGreaterThan(3.9)
    }
  })

  it('never hangs behind the TV', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    put(d, 'tv-stand', 2.5, 0.225)
    put(d, 'sofa', 2.5, 4 - 0.46, Math.PI)
    const r = findWallDrop(d, layoutOf(d), itemFromCatalog(catalogEntry('wall-art'), 'art'))
    expect(r.ok).toBe(true)
    if (r.ok) expect(Math.abs(r.spot.y - 0.02) < 0.1 && Math.abs(r.spot.x - 2.5) < 1.1).toBe(false)
  })
})
