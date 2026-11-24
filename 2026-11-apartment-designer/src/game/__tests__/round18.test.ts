import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { closestPointOnSegment, convexOverlap, dist, dot, norm, orientedRect, pointInPolygon, polygonInsideRoom, sub, v } from '../../model/geometry'
import { carryRugs, findDrop, polygonGap } from '../../model/placement'
import { emptyDoc, rectRoom, sampleDoc } from '../../model/sample'
import type { Doc, Placement } from '../../model/types'
import { activeLayout, useStore } from '../../store'
import { goodPicks, pickSpot, type PickTrial } from '../BuyDrawer'
import { nextGoal, pieceFits, scoreHome, starMove } from '../goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const cat = (d: Doc, p: Placement) => d.items.find((i) => i.id === p.itemId)!.category
const of = (d: Doc, c: string) => layoutOf(d).placements.find((p) => cat(d, p) === c)!
const bodyOf = (d: Doc, p: Placement) => {
  const it = d.items.find((i) => i.id === p.itemId)!
  return orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation)
}
let n = 0
function put(d: Doc, c: string, x: number, y: number, rotation = 0, o: object = {}): Placement {
  const it = itemFromCatalog(catalogEntry(c), `r18-${c}-${++n}`, o)
  d.items.push(it)
  const p = { id: `r18p-${n}`, itemId: it.id, x, y, rotation }
  layoutOf(d).placements.push(p)
  return p
}

describe('A desk or dresser goes to its own room, a smaller one if that fits (round 18)', () => {
  /** Round 17's crowded sample: the dresser took the last bedroom wall a catalog desk fits on. */
  function crowded() {
    const d = sampleDoc({ settled: true })
    const l = layoutOf(d)
    const keep = new Set(['sofa', 'tv-stand', 'bed', 'dining-table'])
    l.placements = l.placements.filter((p) => keep.has(cat(d, p)))
    Object.assign(of(d, 'sofa'), { x: 1.9, y: 3.7, rotation: Math.PI })
    Object.assign(of(d, 'tv-stand'), { x: 2.5, y: 0.23, rotation: 0 })
    Object.assign(of(d, 'bed'), { x: 7.9, y: 1.96, rotation: Math.PI / 2 })
    Object.assign(of(d, 'dining-table'), { x: 4.7, y: 0.8, rotation: Math.PI / 2 })
    put(d, 'dining-chair', 3.95, 0.25, (3 * Math.PI) / 2)
    put(d, 'dining-chair', 4.7, 1.9, Math.PI)
    put(d, 'dresser', 5.58, 0.6, (3 * Math.PI) / 2)
    put(d, 'floor-lamp', 3.2, 3.91, Math.PI)
    return d
  }

  it('offers a writing desk for the bedroom when the catalog desk fits nowhere, instead of a move', () => {
    const d = crowded()
    expect(findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry('desk'), 'probe')).ok).toBe(false)
    expect(pieceFits(d, layoutOf(d), 'desk')).toBe(true)
    const s = scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
    const work = s.goals.find((g) => g.id === 'work')!
    const next = nextGoal({ ...s, goals: s.goals.filter((g) => g.id === 'work' || g.done) }, (c, near) => pieceFits(d, layoutOf(d), c, near))!
    expect(next.id).toBe(work.id)
    expect(next.item).toBe('desk')
    // The pick (the writing desk, the first card) lands in the bedroom.
    const small = itemFromCatalog(catalogEntry('desk'), 'r18-ws', { name: 'Writing desk', width: 1.02, depth: 0.51 })
    const at = pickSpot(d, small, { category: 'desk', goalId: 'work' })
    expect(at.spot).toBeDefined()
    expect(pointInPolygon(v(at.spot!.x, at.spot!.y), d.rooms.find((r) => r.name === 'Bedroom')!.polygon)).toBe(true)
  }, 60000)

  it('orders the pick cards so a size that lands in its own room comes first', () => {
    const t = (home: boolean): PickTrial => ({ ok: true, stars: 10, warnings: 0, broken: 0, home })
    const trials: Record<string, PickTrial> = { Classic: t(false), Writing: t(true), Standing: t(false) }
    expect(goodPicks(['Classic', 'Writing', 'Standing'], (c) => trials[c])).toEqual(['Writing', 'Classic', 'Standing'])
  })

  it('a dresser is placed in the bedroom when it fits there', () => {
    const d = sampleDoc({ settled: true })
    const dresser = itemFromCatalog(catalogEntry('dresser'), 'r18-dr', { name: '3-drawer dresser', width: 0.81, depth: 0.46 })
    const at = pickSpot(d, dresser, { category: 'dresser', goalId: 'storage' })
    expect(at.spot).toBeDefined()
    expect(at.spot!.roomName).toBe('Bedroom')
  }, 60000)

  it('a desk that has to share the living room goes to a corner, out of the way between the TV and the dining table', () => {
    const d = sampleDoc({ settled: true })
    // The bedroom is full; the dining table stands in the far corner, its chairs gone.
    put(d, 'wardrobe', 8.2, 0.35, 0)
    put(d, 'dresser', 5.75, 2.4, Math.PI / 2)
    put(d, 'bookshelf', 8.7, 2.4, -Math.PI / 2)
    const l = layoutOf(d)
    l.placements = l.placements.filter((p) => cat(d, p) !== 'dining-chair')
    Object.assign(of(d, 'dining-table'), { x: 4.3, y: 3.2 })
    for (const w of [1.02, 1.22]) {
      const desk = itemFromCatalog(catalogEntry('desk'), `r18-desk-${w}`, { width: w, depth: w > 1.1 ? 0.61 : 0.51 })
      const r = findDrop(d, l, desk)
      expect(r.ok).toBe(true)
      if (!r.ok) continue
      const living = d.rooms.find((x) => x.name === 'Living room')!
      expect(r.spot.roomId).toBe(living.id)
      const c = v(r.spot.x, r.spot.y)
      const body = orientedRect(c, desk.width / 2, desk.depth / 2, r.spot.rotation)
      // Not in the floor between the TV and the dining table...
      const tv = v(of(d, 'tv-stand').x, of(d, 'tv-stand').y)
      const table = v(of(d, 'dining-table').x, of(d, 'dining-table').y)
      const u = dot(sub(c, tv), norm(sub(table, tv))) / dist(tv, table)
      const between = u > 0.1 && u < 0.9 && closestPointOnSegment(c, tv, table).distance < 1.0
      expect(between, `desk at ${c.x.toFixed(2)},${c.y.toFixed(2)}`).toBe(false)
      // ...clear of the seating and TV...
      for (const k of ['sofa', 'tv-stand', 'coffee-table']) expect(polygonGap(body, bodyOf(d, of(d, k)))).toBeGreaterThan(0.5)
      // ...with an end tucked into a corner.
      expect(Math.min(...body.flatMap((q) => living.polygon.map((k) => dist(q, k))))).toBeLessThan(0.3)
    }
  }, 60000)
})

describe('An office chair stays off the rug (round 18)', () => {
  function study(rugX: number, rugW = 2) {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Study', 0, 0, 4, 4))
    const desk = put(d, 'desk', 2, 0.35, 0, { width: 1.2, depth: 0.6 })
    const rug = put(d, 'rug', rugX, 2, 0, { width: rugW, depth: 3 })
    return { d, desk, rug }
  }
  const onRug = (d: Doc, rug: Placement, r: ReturnType<typeof findDrop>, w: number, dp: number) =>
    r.ok && convexOverlap(orientedRect(v(r.spot.x, r.spot.y), w / 2, dp / 2, r.spot.rotation), bodyOf(d, rug), 0.01)

  it('takes the bare floor at the desk even when that spot scores a little worse', () => {
    const { d, desk, rug } = study(3.1)
    // A box behind the bare spot cramps it (round 17 put the chair on the rug here).
    put(d, 'box', 1.3, 1.75, 0, { width: 0.5, depth: 0.4 })
    const chair = itemFromCatalog(catalogEntry('office-chair'), 'r18-oc')
    const r = findDrop(d, layoutOf(d), chair, { nextTo: desk.id })
    expect(r.ok).toBe(true)
    expect(onRug(d, rug, r, chair.width, chair.depth)).toBe(false)
  })

  it('still goes on the rug when every spot at the desk is on it', () => {
    const { d, desk, rug } = study(2, 4)
    const chair = itemFromCatalog(catalogEntry('office-chair'), 'r18-oc2')
    const r = findDrop(d, layoutOf(d), chair, { nextTo: desk.id })
    expect(r.ok).toBe(true)
    expect(onRug(d, rug, r, chair.width, chair.depth)).toBe(true)
  })
})

describe('A coffee table moved on its own takes its rug (round 18)', () => {
  it('dragged: the rug moves with it, in the same undo step', () => {
    const d = sampleDoc({ settled: true })
    useStore.setState({ doc: d, past: [], future: [], gesture: false })
    const st = useStore.getState()
    const table = of(d, 'coffee-table')
    const rug0 = { ...of(d, 'rug') }
    const before = useStore.getState().doc
    st.beginGesture()
    st.updatePlacement(table.id, { x: table.x + 0.4, y: table.y })
    st.updatePlacement(table.id, { x: table.x + 0.6, y: table.y })
    st.endGesture()
    const after = useStore.getState().doc
    const rug = activeLayout(after).placements.find((p) => p.id === rug0.id)!
    expect(rug.x).not.toBeCloseTo(rug0.x, 2)
    // Still under the table, inside the room.
    const t = activeLayout(after).placements.find((p) => p.id === table.id)!
    expect(convexOverlap(bodyOf(after, rug), bodyOf(after, t), 0.01)).toBe(true)
    expect(after.rooms.some((r) => polygonInsideRoom(bodyOf(after, rug), r.polygon))).toBe(true)
    // One undo puts both back.
    useStore.getState().undo()
    expect(useStore.getState().doc).toBe(before)
  })

  it('a rug that moved itself in the same gesture stays where it was put', () => {
    const d = sampleDoc({ settled: true })
    const l = layoutOf(d)
    const after: Doc = {
      ...d,
      layouts: d.layouts.map((x) =>
        x === l ? { ...x, placements: x.placements.map((p) => (cat(d, p) === 'coffee-table' ? { ...p, x: p.x + 0.5 } : cat(d, p) === 'rug' ? { ...p, x: p.x - 0.3 } : p)) } : x,
      ),
    }
    expect(carryRugs(d, after)).toBe(after)
  })

  it('moved by the chip back in front of the sofa: the rug is re-centred under the sofa and table', () => {
    const d = sampleDoc({ settled: true })
    const table = of(d, 'coffee-table')
    table.x += 0.9
    const m = starMove(d, layoutOf(d), table.id)
    expect(m).not.toBeNull()
    const rug = of(d, 'rug')
    const carried = m!.also?.find((q) => q.id === rug.id)
    expect(carried, 'the rug comes along').toBeDefined()
    for (const q of [m!, ...(m!.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
    expect(convexOverlap(bodyOf(d, of(d, 'rug')), bodyOf(d, of(d, 'coffee-table')), 0.01)).toBe(true)
    expect(convexOverlap(bodyOf(d, of(d, 'rug')), bodyOf(d, of(d, 'sofa')), 0.01)).toBe(true)
    // Centred on the sofa, not left off to one side.
    expect(Math.abs(of(d, 'rug').x - of(d, 'sofa').x)).toBeLessThan(0.05)
  }, 60000)
})

describe('Chip taps reuse what they already worked out (round 18)', () => {
  it('a drop search asked again after the home changed in place sees the change', () => {
    const d = sampleDoc({ settled: true })
    const lamp = itemFromCatalog(catalogEntry('plant'), 'r18-plant')
    const a = findDrop(d, layoutOf(d), lamp)
    expect(findDrop(d, layoutOf(d), lamp)).toEqual(a)
    expect(a.ok).toBe(true)
    if (!a.ok) return
    // Something now stands where the plant would have gone: the cached answer isn't reused.
    put(d, 'box', a.spot.x, a.spot.y, 0, { width: 0.6, depth: 0.6 })
    const b = findDrop(d, layoutOf(d), lamp)
    expect(b.ok && b.spot.x === a.spot.x && b.spot.y === a.spot.y).toBe(false)
  })

  it("an analysis handed out is the caller's own: changing it changes no one else's", () => {
    const d = sampleDoc({ settled: true })
    const a = analyze(d, layoutOf(d))
    a.paths.results = []
    a.issues.length = 0
    const b = analyze(d, layoutOf(d))
    expect(b.paths.results.length).toBeGreaterThan(0)
  })
})
