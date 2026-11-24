import { beforeEach, describe, expect, it } from 'vitest'
import { checkLayout, walkingPaths } from '../checks'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { orientedRect, pointInPolygon, v } from '../geometry'
import { findDrop, findHostDrop, nookTarget, reconcileHosts } from '../placement'
import { emptyDoc, rectRoom } from '../sample'
import type { Doc, Item, Placement } from '../types'
import { parseDoc } from '../../storage'
import { useStore } from '../../store'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0
function put(d: Doc, category: string, x: number, y: number, rotation = 0, extra: Partial<Placement> = {}, over: Partial<Item> = {}): Placement {
  const item = itemFromCatalog(catalogEntry(category), `t${++n}`, over)
  const p: Placement = { id: `tp${n}`, itemId: item.id, x, y, rotation, ...extra }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}
const itemOf = (d: Doc, p: Placement) => d.items.find((i) => i.id === p.itemId)!
const lampItem = () => itemFromCatalog(catalogEntry('table-lamp'), 'probe')
const inside = (d: Doc, host: Placement, x: number, y: number) => {
  const it = itemOf(d, host)
  return pointInPolygon(v(x, y), orientedRect(v(host.x, host.y), it.width / 2, it.depth / 2, host.rotation))
}

/** A 4 × 4 m bedroom: a queen bed against the top wall and a nightstand either side. */
function bedroom(stands = 2) {
  const d = emptyDoc()
  d.rooms.push(rectRoom('Bedroom', 0, 0, 4, 4))
  const bedItem = itemFromCatalog(catalogEntry('bed'), 'probe-bed')
  const bed = put(d, 'bed', 2, bedItem.depth / 2 + 0.05)
  const off = bedItem.width / 2 + 0.27
  const left = put(d, 'nightstand', 2 - off, 0.27)
  const right = stands > 1 ? put(d, 'nightstand', 2 + off, 0.27) : undefined
  return { d, bed, left, right }
}

describe('table lamps on furniture', () => {
  it('refuses politely with nothing to stand on', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Bedroom', 0, 0, 4, 4))
    const r = findDrop(d, layoutOf(d), lampItem())
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.reason).toMatch(/nightstand/)
  })

  it('stands on a nightstand, then on the other one', () => {
    const { d, left, right } = bedroom()
    const r = findDrop(d, layoutOf(d), lampItem())
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect([left.id, right!.id]).toContain(r.spot.on)
    const host = r.spot.on === left.id ? left : right!
    expect(inside(d, host, r.spot.x, r.spot.y)).toBe(true)
    put(d, 'table-lamp', r.spot.x, r.spot.y, r.spot.rotation, { on: r.spot.on })
    const r2 = findDrop(d, layoutOf(d), lampItem())
    expect(r2.ok && r2.spot.on).toBe(host === left ? right!.id : left.id)
    if (r2.ok) put(d, 'table-lamp', r2.spot.x, r2.spot.y, r2.spot.rotation, { on: r2.spot.on })
    const r3 = findDrop(d, layoutOf(d), lampItem())
    expect(r3.ok).toBe(false)
    if (!r3.ok) expect(r3.reason).toMatch(/Every table top/)
  })

  it('goes on the desk it is asked to', () => {
    const { d } = bedroom()
    const desk = put(d, 'desk', 2, 3.6, Math.PI)
    const r = findHostDrop(d, layoutOf(d), lampItem(), { nextTo: desk.id })
    expect(r.ok && r.spot.on).toBe(desk.id)
  })

  it('a lamp on a nightstand never blocks the spot for a piece beside it', () => {
    const { d, left } = bedroom(1)
    const before = findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry('nightstand'), 'probe-n'))
    put(d, 'table-lamp', left.x, left.y - 0.02, 0, { on: left.id })
    const after = findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry('nightstand'), 'probe-n'))
    expect(after.ok).toBe(before.ok)
    if (before.ok && after.ok) expect([after.spot.x, after.spot.y]).toEqual([before.spot.x, before.spot.y])
  })

  it('a table lamp on the side table finishes the reading corner', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    const chair = put(d, 'armchair', 1, 1)
    const table = put(d, 'side-table', 1.75, 1)
    expect(nookTarget(d, layoutOf(d))).toBe(chair.id)
    put(d, 'table-lamp', table.x, table.y, 0, { on: table.id })
    expect(nookTarget(d, layoutOf(d))).toBeUndefined()
  })

  it('checks see no overlap, clearance, door or wall issue for a lamp on a nightstand', () => {
    const { d, left } = bedroom()
    const pathsBefore = walkingPaths(d, layoutOf(d))
    const lamp = put(d, 'table-lamp', left.x, left.y - 0.02, 0, { on: left.id })
    const issues = checkLayout(d, layoutOf(d)).filter((i) => i.severity !== 'info' && (i.placementIds.includes(lamp.id) || i.placementIds.includes(left.id)))
    expect(issues).toEqual([])
    expect(walkingPaths(d, layoutOf(d))).toEqual(pathsBefore)
  })

  it('a lamp follows its host when it moves or turns', () => {
    const { d, left } = bedroom()
    const lamp = put(d, 'table-lamp', left.x, left.y - 0.02, 0, { on: left.id })
    const move = (fn: (p: Placement) => Placement) => {
      const next: Doc = { ...d, layouts: d.layouts.map((l) => ({ ...l, placements: l.placements.map((p) => (p.id === left.id ? fn(p) : p)) })) }
      return layoutOf(reconcileHosts(d, next)).placements.find((p) => p.id === lamp.id)!
    }
    const moved = move((p) => ({ ...p, x: p.x + 1 }))
    expect(moved.x).toBeCloseTo(lamp.x + 1)
    expect(moved.y).toBeCloseTo(lamp.y)
    const turned = move((p) => ({ ...p, rotation: Math.PI / 2 }))
    // 2 cm behind the centre, turned a quarter about the host centre.
    expect(Math.hypot(turned.x - left.x, turned.y - left.y)).toBeCloseTo(0.02)
    expect(turned.rotation).toBeCloseTo(Math.PI / 2)
    expect(Math.abs(turned.x - left.x)).toBeCloseTo(0.02)
    // Nothing changed: the same doc comes back.
    expect(reconcileHosts(d, d)).toBe(d)
  })

  it('moves to the other nightstand when its host goes, or comes off when there is none', () => {
    const { d, left, right } = bedroom()
    const lamp = put(d, 'table-lamp', left.x, left.y, 0, { on: left.id })
    const without = (doc: Doc, id: string): Doc => ({ ...doc, layouts: doc.layouts.map((l) => ({ ...l, placements: l.placements.filter((p) => p.id !== id) })) })
    const once = reconcileHosts(d, without(d, left.id))
    const moved = layoutOf(once).placements.find((p) => p.id === lamp.id)!
    expect(moved.on).toBe(right!.id)
    const twice = reconcileHosts(once, without(once, right!.id))
    expect(layoutOf(twice).placements.some((p) => p.id === lamp.id)).toBe(false)
  })

  it('never wanders to a desk (same room or another): it comes off and waits in Mine', () => {
    const { d, left } = bedroom(1)
    put(d, 'desk', 2, 3.5)
    d.rooms.push(rectRoom('Office', 4, 0, 4, 4))
    put(d, 'desk', 6, 3.5)
    const lamp = put(d, 'table-lamp', left.x, left.y, 0, { on: left.id })
    const next: Doc = { ...d, layouts: d.layouts.map((l) => ({ ...l, placements: l.placements.filter((p) => p.id !== left.id) })) }
    const report = { removed: [] as string[], rehosted: [] as string[] }
    const out = reconcileHosts(d, next, report)
    expect(layoutOf(out).placements.some((p) => p.id === lamp.id)).toBe(false)
    expect(report.removed).toEqual([lamp.id])
    // The item itself stays owned.
    expect(out.items.some((i) => i.id === lamp.itemId)).toBe(true)
  })

  it('a desk holds one lamp', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Office', 0, 0, 4, 4))
    const desk = put(d, 'desk', 2, 3.5)
    const r = findDrop(d, layoutOf(d), lampItem())
    expect(r.ok && r.spot.on).toBe(desk.id)
    if (r.ok) put(d, 'table-lamp', r.spot.x, r.spot.y, r.spot.rotation, { on: r.spot.on })
    expect(findDrop(d, layoutOf(d), lampItem()).ok).toBe(false)
  })

  it('a pendant hangs over the centre of a dining table, and only there', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Dining', 0, 0, 4, 4))
    put(d, 'side-table', 0.4, 0.4)
    const pendant = () => itemFromCatalog(catalogEntry('pendant'), 'probe-pendant')
    const none = findDrop(d, layoutOf(d), pendant())
    expect(none.ok).toBe(false)
    if (!none.ok) expect(none.reason).toBe('A pendant hangs over a dining table. Add one first.')
    const table = put(d, 'dining-table', 2, 2)
    const r = findDrop(d, layoutOf(d), pendant())
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.spot.on).toBe(table.id)
    expect(r.spot.x).toBeCloseTo(2)
    expect(r.spot.y).toBeCloseTo(2)
    put(d, 'pendant', r.spot.x, r.spot.y, r.spot.rotation, { on: r.spot.on })
    expect(findDrop(d, layoutOf(d), pendant()).ok).toBe(false)
    // A table lamp never takes the dining table.
    expect(findDrop(d, layoutOf(d), lampItem()).ok && (findDrop(d, layoutOf(d), lampItem()) as { spot: { on?: string } }).spot.on).not.toBe(table.id)
  })

  it('old saves: a short "Table lamp" floor lamp on a nightstand becomes a table lamp standing on it', () => {
    const { d, left } = bedroom()
    const old = put(d, 'floor-lamp', left.x, left.y, 0, {}, { name: 'Table lamp', height: 0.55, width: 0.3, depth: 0.3 })
    const back = parseDoc(JSON.stringify(d))!
    expect(back.items.find((i) => i.id === old.itemId)!.category).toBe('table-lamp')
    expect(layoutOf(back).placements.find((p) => p.id === old.id)!.on).toBe(left.id)
  })

  describe('in the store', () => {
    let ids: { d: Doc; left: Placement; right?: Placement }
    beforeEach(() => {
      ids = bedroom()
      useStore.setState({ doc: ids.d, past: [], future: [], gesture: false })
    })

    it('copying a design keeps each lamp on the copy of its host', () => {
      const lamp = put(ids.d, 'table-lamp', ids.left.x, ids.left.y, 0, { on: ids.left.id })
      useStore.setState({ doc: { ...ids.d } })
      useStore.getState().duplicateLayout(ids.d.activeLayoutId)
      const doc = useStore.getState().doc
      const copy = layoutOf(doc)
      expect(copy.id).not.toBe(ids.d.layouts[0].id)
      const lampCopy = copy.placements.find((p) => p.itemId === lamp.itemId)!
      const hostCopy = copy.placements.find((p) => p.itemId === ids.left.itemId)!
      expect(lampCopy.on).toBe(hostCopy.id)
    })

    it('moving the host moves the lamp', () => {
      const lamp = put(ids.d, 'table-lamp', ids.left.x, ids.left.y, 0, { on: ids.left.id })
      useStore.setState({ doc: { ...ids.d } })
      useStore.getState().updatePlacement(ids.left.id, { x: ids.left.x + 0.5 })
      const now = layoutOf(useStore.getState().doc).placements.find((p) => p.id === lamp.id)!
      expect(now.x).toBeCloseTo(lamp.x + 0.5)
    })
  })
})
