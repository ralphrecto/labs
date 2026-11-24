import { describe, expect, it } from 'vitest'
import { rot, v } from '../../model/geometry'
import { emptyDoc } from '../../model/sample'
import { VISIT_BY_ID, type VisitPiece } from '../../game/visits'
import { analyzeScan } from '../analyze'
import { applyEstimate } from '../apply'
import { sampleRoomPoints, SAMPLE_TRUTH } from '../sampleScan'
import { scanPalette } from '../palette'
import { syntheticScan } from '../synthetic'
import type { ScanData } from '../types'

const { positions, colors } = sampleRoomPoints()
const sample: ScanData = { id: 'try-ours', name: 'Sample living room', upAxis: 'y', metric: true, points: { count: positions.length / 3, positions, colors } } as ScanData

describe('the scanned rug becomes a rug', () => {
  const { estimate } = analyzeScan(sample)
  const rug = estimate.objects.find((o) => o.category === 'rug')

  it('finds the sample rug by its colour on the boards, at its size', () => {
    expect(rug).toBeTruthy()
    // The sample's rug is 2.6 x 1.5 m.
    expect(rug!.width).toBeGreaterThan(2.4)
    expect(rug!.width).toBeLessThan(2.85)
    expect(rug!.depth).toBeGreaterThan(1.35)
    expect(rug!.depth).toBeLessThan(1.75)
    expect(estimate.objects.filter((o) => o.category === 'rug')).toHaveLength(1)
  })

  it('finds no rug on a plain floor', () => {
    expect(analyzeScan(syntheticScan()).estimate.objects.some((o) => o.category === 'rug')).toBe(false)
  })

  it('imports it as a real rug, so the play-rug visit asks to slide it, not to buy one', () => {
    const doc = emptyDoc()
    const res = applyEstimate(doc, estimate, { includeRoom: true, objects: estimate.objects })
    const layout = res.doc.layouts.find((l) => l.id === res.doc.activeLayoutId)!
    const pieces: VisitPiece[] = layout.placements.map((p) => {
      const it = res.doc.items.find((i) => i.id === p.itemId)!
      const hw = it.width / 2
      const hd = it.depth / 2
      const body = [v(-hw, -hd), v(hw, -hd), v(hw, hd), v(-hw, hd)].map((q) => {
        const r = rot(q, p.rotation)
        return v(r.x + p.x, r.y + p.y)
      })
      return { id: p.id, category: it.category, center: v(p.x, p.y), rotation: p.rotation, hw, hd, body }
    })
    const rugItem = res.doc.items.find((i) => i.category === 'rug')!
    expect(rugItem.source).toBe('scan')
    expect(rugItem.owned).toBe(true)
    const check = VISIT_BY_ID.cousins.check({ pieces, all: pieces, goal: () => undefined })
    expect(check.item).toBeUndefined() // no "add a rug" buy hint
    expect(check.progress).toBeGreaterThanOrEqual(0.7)
  })
})

describe('scan floor picture', () => {
  const room = { id: 'r', name: 'r', ceilingHeight: 2.55, wallThickness: 0.12, openings: [], polygon: SAMPLE_TRUTH.outline } as never
  const floorAt = (erase: boolean) => {
    const f = scanPalette([sample.points], [room], { eraseRugs: erase }).floor!
    return (x: number, y: number) => {
      const k = (Math.floor((y - f.y0) / f.cell) * f.width + Math.floor((x - f.x0) / f.cell)) * 4
      return [f.rgba[k], f.rgba[k + 1], f.rgba[k + 2], f.rgba[k + 3]]
    }
  }

  it('fills the edge along the walls from the boards, not the pale foot of the wall', () => {
    const at = floorAt(false)
    const mid = at(4.5, 1.3)
    for (const [x, y] of [[4.5, 0.03], [5.17, 1.3], [0.03, 3.5], [2, 4.07]]) {
      const e = at(x, y)
      expect(e[3]).toBe(255)
      expect(Math.abs(e[0] - mid[0]) + Math.abs(e[1] - mid[1]) + Math.abs(e[2] - mid[2])).toBeLessThan(40)
    }
  })

  it('paints the rug out of the floor once it is a piece of its own', () => {
    const before = floorAt(false)(2, 1.8)
    const after = floorAt(true)(2, 1.8)
    const boards = floorAt(true)(4.5, 1.3)
    expect(before[1] - boards[1]).toBeGreaterThan(30)
    expect(Math.abs(after[1] - boards[1])).toBeLessThan(15)
  })
})
