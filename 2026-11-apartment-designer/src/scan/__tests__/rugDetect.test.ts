import { describe, expect, it } from 'vitest'
import { rot, v } from '../../model/geometry'
import { emptyDoc } from '../../model/sample'
import { VISIT_BY_ID, type VisitPiece } from '../../game/visits'
import { analyzeScan } from '../analyze'
import { applyEstimate } from '../apply'
import { sampleRoomPoints, samplePly, SAMPLE_TRUTH } from '../sampleScan'
import { scanPalette, toonFloor, type FloorImage } from '../palette'
import { toScanData } from '../align'
import { readPlyPoints } from '../readPly'
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

  it('continues the planks where the rug was, instead of a flat patch', () => {
    const at = floorAt(true)
    // Plank-to-plank contrast: the mean change between cells 9 cm apart across the planks (which run along x).
    const contrast = (x0: number, x1: number, y0: number, y1: number) => {
      let sum = 0
      let n = 0
      for (let x = x0; x < x1; x += 0.05)
        for (let y = y0; y < y1; y += 0.09) {
          const a = at(x, y)
          const b = at(x, y + 0.09)
          sum += Math.abs(a[0] + a[1] + a[2] - b[0] - b[1] - b[2])
          n++
        }
      return sum / n
    }
    const underRug = contrast(1.4, 3.2, 1.3, 2.4)
    const boards = contrast(3.6, 5.0, 0.4, 2.3)
    expect(boards).toBeGreaterThan(10)
    expect(underRug).toBeGreaterThan(0.6 * boards)
  })

  it('leaves no stray specks in the Cartoon floor, even when the outline sits a little off the walls', () => {
    // Islands of a different colour band smaller than ~0.04 m2 (a cell or two of the scan).
    const specks = (f: FloorImage) => {
      // Its lightness band (Cartoon snaps lightness to bands and keeps the hue smooth).
      const key = (k: number) => {
        const c = [f.rgba[k * 4], f.rgba[k * 4 + 1], f.rgba[k * 4 + 2]]
        return Math.round(((Math.max(...c) + Math.min(...c)) / 510) * 10)
      }
      const done = new Uint8Array(f.width * f.height)
      const out: number[][] = []
      for (let k0 = 0; k0 < done.length; k0++) {
        if (done[k0] || !f.rgba[k0 * 4 + 3]) continue
        const stack = [k0]
        const cells: number[] = []
        done[k0] = 1
        while (stack.length) {
          const k = stack.pop()!
          cells.push(k)
          const i = k % f.width
          const j = (k - i) / f.width
          for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const ii = i + dx
            const jj = j + dy
            const kk = jj * f.width + ii
            if (ii < 0 || jj < 0 || ii >= f.width || jj >= f.height || done[kk] || !f.rgba[kk * 4 + 3] || key(kk) !== key(k0)) continue
            done[kk] = 1
            stack.push(kk)
          }
        }
        if (cells.length * f.cell * f.cell >= 0.04) continue
        const cx = cells.reduce((a, k) => a + (k % f.width) + 0.5, 0) / cells.length
        const cy = cells.reduce((a, k) => a + Math.floor(k / f.width) + 0.5, 0) / cells.length
        out.push([f.x0 + cx * f.cell, f.y0 + cy * f.cell, cells.length])
      }
      return out
    }
    // The "Try ours" flow: the sample file, placed the way the app calibrates it (a hair off true).
    const ref = { id: 's', name: 's', fileName: 's.ply', format: 'splat', upAxis: 'y', floor: 1.30055, ceiling: 3.7476, scale: 1.0416667, yaw: -0.2958333, x: 1.7975076, y: -0.0956184, references: [], visible: true } as never
    const fromFile = toScanData(ref, readPlyPoints(samplePly())!, 400_000).points
    for (const [pts, dy] of [[fromFile, 0], [fromFile, 0.01], [sample.points, 0], [sample.points, 0.03]] as const) {
      const poly = SAMPLE_TRUTH.outline.map((p) => ({ x: p.x, y: p.y > 3 ? p.y + 0.01 + dy : p.y - dy }))
      const r = { ...(room as object), polygon: poly } as never
      const f = toonFloor(scanPalette([pts], [r], { eraseRugs: true }).floor!)
      // Inside the room only (the image's margin outside the walls is hidden under them).
      const inRoom = specks(f).filter(([x, y]) => x > 0.01 && y > 0.01 && x < 5.19 && y < 4.09 && !(x > 3.39 && y > 2.59))
      expect(inRoom).toEqual([])
    }
  })
})
