import { describe, expect, it } from 'vitest'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { findDrop, findHostDrop, hostAt, reconcileHosts, SURFACES } from '../placement'
import { emptyDoc, rectRoom } from '../sample'
import type { Doc, Placement } from '../types'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0
function drop(d: Doc, category: string) {
  const item = itemFromCatalog(catalogEntry(category), `ct${++n}`)
  const r = findDrop(d, layoutOf(d), item)
  if (!r.ok) throw new Error(`${category}: ${r.reason}`)
  const p: Placement = { id: `ctp${n}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, ...(r.spot.on ? { on: r.spot.on } : {}) }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return { item, p }
}
function living(): Doc {
  const d = emptyDoc()
  d.rooms = [rectRoom('Living room', 0, 0, 5, 4)]
  return d
}

describe('coffee tables hold a table lamp (r13)', () => {
  it('is a surface with one slot', () => {
    expect(SURFACES.has('coffee-table')).toBe(true)
  })

  it('a table lamp lands on a coffee table when it is the only top, and the top is then full', () => {
    const d = living()
    drop(d, 'sofa')
    const t = drop(d, 'coffee-table')
    const lamp = drop(d, 'table-lamp')
    expect(lamp.p.on).toBe(t.p.id)
    expect(hostAt(d, layoutOf(d), { x: lamp.p.x, y: lamp.p.y }, lamp.item)?.id).toBe(t.p.id)
    const second = itemFromCatalog(catalogEntry('table-lamp'), 'ct-second')
    expect(findHostDrop(d, layoutOf(d), second).ok).toBe(false)
  })

  it('a side table by the sofa still wins over the coffee table', () => {
    const d = living()
    drop(d, 'sofa')
    drop(d, 'coffee-table')
    const side = drop(d, 'side-table')
    const lamp = drop(d, 'table-lamp')
    expect(lamp.p.on).toBe(side.p.id)
  })

  it('the lamp follows its coffee table when the table moves', () => {
    const d = living()
    drop(d, 'sofa')
    const t = drop(d, 'coffee-table')
    const lamp = drop(d, 'table-lamp')
    const next: Doc = structuredClone(d)
    const lay = layoutOf(next)
    lay.placements = lay.placements.map((p) => (p.id === t.p.id ? { ...p, x: p.x + 0.3, y: p.y + 0.1 } : p))
    const fixed = reconcileHosts(d, next)
    const moved = layoutOf(fixed).placements.find((p) => p.id === lamp.p.id)!
    expect(moved.on).toBe(t.p.id)
    expect(moved.x).toBeCloseTo(lamp.p.x + 0.3, 6)
    expect(moved.y).toBeCloseTo(lamp.p.y + 0.1, 6)
  })
})
