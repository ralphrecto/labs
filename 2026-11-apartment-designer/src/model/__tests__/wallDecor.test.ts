import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { scoreHome } from '../../game/goals'
import { checkLayout } from '../checks'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { dist, openingEndpoints, v, wallSegments } from '../geometry'
import { findDrop } from '../placement'
import { emptyDoc, rectRoom } from '../sample'
import type { Doc, Item, Placement } from '../types'
import { findWallDrop, mountBottom, onWallOk, snapToWall } from '../wallDecor'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0
function put(d: Doc, category: string, x: number, y: number, rotation = 0, over: Partial<Item> = {}): Placement {
  const item = itemFromCatalog(catalogEntry(category), `w${++n}`, over)
  const p: Placement = { id: `wp${n}`, itemId: item.id, x, y, rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}
const probe = (category: string) => itemFromCatalog(catalogEntry(category), `probe-${category}`)

/** A 4 × 4 m bedroom: a door on the left wall, a window on the bottom wall, the bed against the top wall. */
function bedroom() {
  const d = emptyDoc()
  const room = rectRoom('Bedroom', 0, 0, 4, 4)
  // rectRoom walls: 0 top (y = 0), 1 right, 2 bottom, 3 left.
  room.openings.push({ id: 'door', kind: 'door', wall: 3, offset: 1.5, width: 0.9, height: 2.1, sill: 0, hinge: 'start', swing: 'in' })
  room.openings.push({ id: 'win', kind: 'window', wall: 2, offset: 1.5, width: 1.2, height: 1.3, sill: 0.9 })
  d.rooms.push(room)
  const bed = put(d, 'bed', 2, 2.03 / 2 + 0.05)
  return { d, room, bed }
}

describe('wall decor', () => {
  it('hangs at a sensible height: art at eye level, curtains to the floor', () => {
    expect(mountBottom(probe('wall-art')) + probe('wall-art').height / 2).toBeCloseTo(1.5)
    expect(mountBottom(probe('mirror')) + probe('mirror').height / 2).toBeCloseTo(1.5)
    expect(mountBottom(probe('wall-shelf')) + probe('wall-shelf').height / 2).toBeCloseTo(1.6)
    expect(mountBottom(probe('curtains'))).toBeLessThan(0.05)
  })

  it('snaps to the nearest wall, flush and facing into the room', () => {
    const { d } = bedroom()
    const art = probe('wall-art')
    // Near the right wall (x = 4): flush against it, front facing -x.
    const s = snapToWall(d, layoutOf(d), art, v(3.6, 2.2))!
    expect(s).toBeTruthy()
    expect(s.x).toBeCloseTo(4 - art.depth / 2, 2)
    expect(s.y).toBeCloseTo(2.2, 5)
    // Front = rot((0, 1), rotation) points into the room (-x).
    expect(-Math.sin(s.rotation)).toBeCloseTo(-1, 5)
    expect(Math.cos(s.rotation)).toBeCloseTo(0, 5)
    // Slides along the top wall.
    const t = snapToWall(d, layoutOf(d), art, v(1.2, 0.4))!
    expect(t.y).toBeCloseTo(art.depth / 2, 2)
    expect(t.x).toBeCloseTo(1.2, 5)
  })

  it('never covers a door, and art keeps off the window', () => {
    const { d, room } = bedroom()
    const art = probe('wall-art')
    const [da, db] = openingEndpoints(room, room.openings[0])
    const doorMid = v((da.x + db.x) / 2, (da.y + db.y) / 2)
    const s = snapToWall(d, layoutOf(d), art, v(0.3, doorMid.y))!
    // Still on the left wall, but slid clear of the door's span.
    const [y0, y1] = [Math.min(da.y, db.y), Math.max(da.y, db.y)]
    expect(s.x).toBeCloseTo(art.depth / 2, 2)
    expect(s.y + art.width / 2 <= y0 || s.y - art.width / 2 >= y1).toBe(true)
    const w = snapToWall(d, layoutOf(d), art, v(2.1, 3.7))!
    expect(w.y).toBeCloseTo(4 - art.depth / 2, 2)
    const [wa, wb] = openingEndpoints(room, room.openings[1])
    expect(w.x + art.width / 2 <= Math.min(wa.x, wb.x) || w.x - art.width / 2 >= Math.max(wa.x, wb.x)).toBe(true)
  })

  it('curtains snap to windows only, centred on them', () => {
    const { d, room } = bedroom()
    const curtains = probe('curtains')
    // Dragged near the top wall, which has no window: they still go to the window.
    const s = snapToWall(d, layoutOf(d), curtains, v(1, 0.5))!
    const [a, b] = openingEndpoints(room, room.openings[1])
    expect(dist(v(s.x, s.y), v((a.x + b.x) / 2, (a.y + b.y) / 2))).toBeCloseTo(curtains.depth / 2, 2)
    // No windows: nowhere to hang them.
    const bare = emptyDoc()
    bare.rooms.push(rectRoom('Bedroom', 0, 0, 4, 4))
    expect(snapToWall(bare, layoutOf(bare), curtains, v(1, 1))).toBeNull()
    expect(findDrop(bare, layoutOf(bare), curtains).ok).toBe(false)
    // One pair per window.
    put(d, 'curtains', s.x, s.y, s.rotation)
    expect(findWallDrop(d, layoutOf(d), curtains).ok).toBe(false)
  })

  it('takes no floor: no overlap, clearance or walkway issues, even over the bed', () => {
    const { d, bed } = bedroom()
    const before = checkLayout(d, layoutOf(d)).length
    // Art right over the headboard: its box overlaps the bed's footprint on the plan.
    const art = put(d, 'wall-art', bed.x, 0.02, 0)
    const shelf = put(d, 'wall-shelf', 3.5, 0.11, 0)
    expect(onWallOk(d, layoutOf(d), itemFromCatalog(catalogEntry('wall-art'), 'x'), art)).toBe(true)
    const issues = checkLayout(d, layoutOf(d))
    expect(issues.length).toBe(before)
    expect(issues.some((i) => i.placementIds.includes(art.id) || i.placementIds.includes(shelf.id))).toBe(false)
    // A floor piece can still go right below it.
    const drop = findDrop(d, layoutOf(d), probe('nightstand'))
    expect(drop.ok).toBe(true)
  })

  it('a bought frame hangs over the bed, not behind a wardrobe', () => {
    const { d, bed } = bedroom()
    put(d, 'wardrobe', 3.4, 2.5, -Math.PI / 2)
    const r = findDrop(d, layoutOf(d), probe('wall-art'))
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.spot.y).toBeLessThan(0.1)
    expect(Math.abs(r.spot.x - bed.x)).toBeLessThan(0.3)
  })

  it('does not add to density or the solid piece count', () => {
    const { d } = bedroom()
    const l = layoutOf(d)
    const base = scoreHome(d, l, analyze(d, l))
    const density = base.goals.find((g) => g.id.startsWith('density:'))
    put(d, 'curtains', 2.1, 4 - 0.06, Math.PI)
    put(d, 'wall-art', 2, 0.02, 0)
    const after = scoreHome(d, l, analyze(d, l))
    expect(after.goals.find((g) => g.id.startsWith('density:'))?.progress).toBeCloseTo(density?.progress ?? 0, 6)
    expect(after.errors).toBe(base.errors)
  })

  it('wall goals are bonus-only and point to the Walls tab', () => {
    const { d, room } = bedroom()
    const l = layoutOf(d)
    const s0 = scoreHome(d, l, analyze(d, l))
    const art = s0.goals.find((g) => g.id === 'wall-art')!
    const cur = s0.goals.find((g) => g.id === 'curtains')!
    expect(art.bonus && cur.bonus).toBe(true)
    expect(art.done || cur.done).toBe(false)
    expect([art.tab, art.item, art.placeIn]).toEqual(['wall', 'wall-art', room.id])
    expect([cur.tab, cur.item]).toEqual(['wall', 'curtains'])
    // Bonus goals never raise the bar: the level thresholds come from core goals alone.
    expect(s0.coreMax).toBe(s0.goals.filter((g) => !g.bonus).reduce((a, g) => a + g.stars, 0))
    const a = findDrop(d, l, probe('wall-art'))
    const c = findDrop(d, l, probe('curtains'))
    if (!a.ok || !c.ok) throw new Error('no drop')
    put(d, 'wall-art', a.spot.x, a.spot.y, a.spot.rotation)
    put(d, 'curtains', c.spot.x, c.spot.y, c.spot.rotation)
    const s1 = scoreHome(d, l, analyze(d, l))
    expect(s1.goals.find((g) => g.id === 'wall-art')!.done).toBe(true)
    expect(s1.goals.find((g) => g.id === 'curtains')!.done).toBe(true)
    expect(s1.stars).toBe(s0.stars + 2)
    expect(s1.coreMax).toBe(s0.coreMax)
  })

  it('every wall in a room is a candidate, walls of the room only', () => {
    const { room } = bedroom()
    expect(wallSegments(room).length).toBe(4)
  })
})
