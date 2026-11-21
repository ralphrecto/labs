import { describe, expect, it } from 'vitest'
import { v } from '../../model/geometry'
import type { Item, Placement, Room } from '../../model/types'
import { dragConflictBase, isCleanPose, settleDrop, type Others } from '../editActions'

const room = { id: 'r', name: 'Living', polygon: [v(0, 0), v(5, 0), v(5, 4), v(0, 4)], openings: [], wallThickness: 0.1, ceilingHeight: 2.5 } as unknown as Room
const sofa = { id: 'sofa', name: 'Sofa', width: 2, depth: 0.9, height: 0.8 } as Item
const stand = { id: 'stand', name: 'TV stand', width: 1.6, depth: 0.45, height: 0.5 } as Item
const rug = { id: 'rug', name: 'Rug', width: 2.4, depth: 1.6, height: 0.01, floorLevel: true } as Item
const at = (id: string, itemId: string, x: number, y: number): Placement => ({ id, itemId, x, y, rotation: 0 }) as Placement
const others: Others = [
  { id: 'p-stand', item: stand, p: at('p-stand', 'stand', 2.5, 0.3) },
  { id: 'p-rug', item: rug, p: at('p-rug', 'rug', 2.5, 2) },
]

describe('dropping a dragged piece', () => {
  const start = at('p-sofa', 'sofa', 2.5, 3.4)
  const base = dragConflictBase(sofa, start, others, [room])

  it('stays where it is dropped on a clear spot (rugs never block)', () => {
    const drop = { ...start, y: 2 }
    expect(isCleanPose(sofa, drop, others, [room], base)).toBe(true)
    expect(settleDrop(sofa, drop, start, others, [room], base)).toEqual({ placement: drop, bounced: false })
  })

  it('goes back to the last clean spot when dropped deep on another piece', () => {
    const drop = { ...start, y: 0.4 }
    const last = { ...start, y: 2.2 }
    expect(isCleanPose(sofa, drop, others, [room], base)).toBe(false)
    const r = settleDrop(sofa, drop, last, others, [room], base)
    expect(r.bounced).toBe(true)
    expect(r.placement).toEqual(last)
  })

  it('slides a little to the nearest clean spot when it only just overlaps', () => {
    const drop = { ...start, y: 1.1 } // sofa back edge at 0.65, stand front at 0.525... overlap of 0.1 m
    const tight = { ...start, y: 0.95 }
    expect(isCleanPose(sofa, tight, others, [room], base)).toBe(false)
    const r = settleDrop(sofa, tight, drop, others, [room], base)
    expect(r.bounced).toBe(true)
    expect(isCleanPose(sofa, r.placement, others, [room], base)).toBe(true)
    expect(Math.hypot(r.placement.x - tight.x, r.placement.y - tight.y)).toBeLessThan(0.2)
  })

  it('bounces back from a wall crossing', () => {
    const drop = { ...start, x: 6 }
    const r = settleDrop(sofa, drop, start, others, [room], base)
    expect(r).toEqual({ placement: start, bounced: true })
  })
})
