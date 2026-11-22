import { describe, expect, it } from 'vitest'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { chairBackNeed } from '../checks'
import { footprint, orientedRect, polygonInsideRoom, v } from '../geometry'
import { findDrop, polygonGap } from '../placement'
import { emptyDoc, rectRoom, sampleDoc } from '../sample'
import type { Doc, Item, Placement } from '../types'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0
function drop(d: Doc, category: string) {
  const item = itemFromCatalog(catalogEntry(category), `c${++n}`)
  const r = findDrop(d, layoutOf(d), item)
  if (!r.ok) return null
  const p: Placement = { id: `cp${n}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return { item, p }
}
const local = (q: Placement, o: Placement) => {
  const dx = q.x - o.x
  const dy = q.y - o.y
  const c = Math.cos(-o.rotation)
  const s = Math.sin(-o.rotation)
  return v(dx * c - dy * s, dx * s + dy * c)
}
function studio(w = 5, d = 4): Doc {
  const doc = emptyDoc()
  doc.rooms = [rectRoom('Living room', 0, 0, w, d)]
  return doc
}
function emptySample(): Doc {
  const d = sampleDoc()
  d.items = []
  layoutOf(d).placements = []
  return d
}

describe('companions (round 12)', () => {
  it('a coffee table goes in front of the sofa, about 45 cm off', () => {
    for (const d of [studio(), studio(4, 3.4), emptySample()]) {
      const s = drop(d, 'sofa')!
      const t = drop(d, 'coffee-table')!
      expect(t).not.toBeNull()
      const gap = polygonGap(footprint(s.item, s.p), footprint(t.item, t.p))
      expect(gap).toBeGreaterThan(0.35)
      expect(gap).toBeLessThan(0.6)
      expect(local(t.p, s.p).y).toBeGreaterThan(s.item.depth / 2)
    }
  })

  it('a sofa placed after its coffee table goes behind it', () => {
    for (const d of [studio(4, 3.4), emptySample()]) {
      const t = drop(d, 'coffee-table')!
      const s = drop(d, 'sofa')!
      const gap = polygonGap(footprint(s.item, s.p), footprint(t.item, t.p))
      expect(gap).toBeGreaterThan(0.3)
      expect(gap).toBeLessThan(0.8)
      expect(local(t.p, s.p).y).toBeGreaterThan(s.item.depth / 2)
      expect(Math.abs(local(t.p, s.p).x)).toBeLessThan(s.item.width / 2)
    }
  })

  it('a floor lamp stands beside a seat, not in front of it', () => {
    for (const d of [studio(), studio(4, 3.4), emptySample()]) {
      const s = drop(d, 'sofa')!
      drop(d, 'coffee-table')
      const l = drop(d, 'floor-lamp')!
      expect(l).not.toBeNull()
      const at = local(l.p, s.p)
      expect(at.y).toBeLessThan(s.item.depth / 2)
      expect(polygonGap(footprint(s.item, s.p), footprint(l.item, l.p))).toBeLessThan(0.5)
    }
  })

  it("never puts a dining chair where its pull-out room runs through a wall", () => {
    for (const [w, dd] of [[3, 2.6], [2.6, 2.4], [2.4, 2.2], [4, 3]] as const) {
      const d = studio(w, dd)
      drop(d, 'dining-table')
      const table = layoutOf(d).placements[0]
      const tItem = d.items[0]
      for (let k = 0; k < 6; k++) {
        const ch = drop(d, 'dining-chair')
        if (!ch) continue
        const body = footprint(ch.item, ch.p)
        const need = chairBackNeed({ item: ch.item as Item, body }, [{ body: footprint(tItem, table) }])
        const z = orientedRect(v(ch.p.x, ch.p.y), ch.item.width / 2, need / 2, ch.p.rotation, v(0, -ch.item.depth / 2 - need / 2))
        expect(polygonInsideRoom(z, d.rooms[0].polygon, 0.01)).toBe(true)
      }
    }
  })
})
