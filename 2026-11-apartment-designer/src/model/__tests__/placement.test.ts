import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { checkLayout } from '../checks'
import { clearanceZones, convexOverlap, doorSwing, footprint, pointInPolygon, polygonInsideRoom, v, wallInwardNormal, wallSegments, closestPointOnSegment } from '../geometry'
import { findDrop, MIN_DRAWER_FRONT, polygonGap } from '../placement'
import { emptyDoc, rectRoom, sampleDoc } from '../sample'
import type { Doc, Item, Opening, Placement } from '../types'
import { scoreHome } from '../../game/goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0

function emptySample(): Doc {
  const d = sampleDoc()
  d.items = []
  layoutOf(d).placements = []
  return d
}

/** Places a catalog piece where findDrop says; returns the placement (or null). */
function drop(d: Doc, category: string, over: Partial<Item> = {}) {
  const item = itemFromCatalog(catalogEntry(category), `t${++n}`, over)
  const r = findDrop(d, layoutOf(d), item)
  if (!r.ok) return { item, p: null as Placement | null, reason: r.reason, roomName: undefined as string | undefined }
  const p = { id: `tp${n}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return { item, p, reason: '', roomName: r.spot.roomName }
}

const roomOf = (d: Doc, p: Placement) => d.rooms.find((r) => pointInPolygon(v(p.x, p.y), r.polygon))!

describe('findDrop', () => {
  it('puts a bed in the bedroom, inside the walls, backed onto a wall, clear of doors', () => {
    const d = emptySample()
    const { item, p } = drop(d, 'bed')
    expect(p).not.toBeNull()
    const room = roomOf(d, p!)
    expect(room.name).toBe('Bedroom')
    const body = footprint(item, p!)
    expect(polygonInsideRoom(body, room.polygon)).toBe(true)
    // The bed's back edge sits on a wall: the head center is within 2 cm of some wall.
    const back = v(p!.x + Math.sin(p!.rotation) * item.depth / 2, p!.y - Math.cos(p!.rotation) * item.depth / 2)
    const wallGap = Math.min(...wallSegments(room).map(([a, b]) => closestPointOnSegment(back, a, b).distance))
    expect(wallGap).toBeLessThan(0.02)
    for (const r of d.rooms) for (const o of r.openings) {
      const s = doorSwing(r, o)
      if (s) expect(convexOverlap(body, s, 0.005)).toBe(false)
    }
    expect(checkLayout(d, layoutOf(d)).filter((i) => i.severity === 'error')).toEqual([])
  })

  it('puts a bed into the full sample without overlapping anything', () => {
    const d = sampleDoc()
    const { p, reason } = drop(d, 'bed', { width: 0.97, depth: 1.91 })
    if (p) {
      expect(checkLayout(d, layoutOf(d)).filter((i) => i.severity === 'error')).toEqual([])
      expect(roomOf(d, p).name).not.toBe('Bathroom')
    } else expect(reason).toMatch(/No free spot|full/)
  })

  it('never puts a bed in the living room of the sample home (round 2 regression)', () => {
    // The sample as shipped: a queen bed already fills the bedroom. A player adds a sofa, then
    // taps Bed in the Bedroom tab. It used to wedge the bed between the TV stand and the coffee
    // table in the living room, 4 cm from each, blocking the sofa.
    const d = sampleDoc()
    const sofa = drop(d, 'sofa')
    expect(sofa.p).not.toBeNull()
    const before = layoutOf(d).placements.length
    const bed = drop(d, 'bed')
    expect(bed.p).toBeNull()
    expect(bed.reason).toMatch(/bedroom's full/i)
    expect(layoutOf(d).placements.length).toBe(before)
    for (const c of ['nightstand', 'wardrobe']) {
      const r = drop(d, c)
      if (r.p) expect(r.roomName).toBe('Bedroom')
      else expect(r.reason).toMatch(/bedroom's full/i)
    }
  })

  it('puts a bed in the sample bedroom, clear of everything, once the old one is gone', () => {
    const d = sampleDoc()
    const l = layoutOf(d)
    const old = d.items.find((i) => i.category === 'bed')!
    l.placements = l.placements.filter((p) => p.itemId !== old.id)
    const { item, p, roomName } = drop(d, 'bed')
    expect(p).not.toBeNull()
    expect(roomName).toBe('Bedroom')
    const body = footprint(item, p!)
    for (const q of l.placements) {
      if (q === p) continue
      const other = d.items.find((i) => i.id === q.itemId)!
      if (other.floorLevel) continue
      expect(convexOverlap(body, footprint(other, q), 0.002), other.name).toBe(false)
    }
    expect(checkLayout(d, l).filter((i) => i.severity === 'error')).toEqual([])
    expect(checkLayout(d, l).filter((i) => /^Bed needs .* in front/.test(i.message))).toEqual([])
  })

  it('furnishes a one-bedroom with no errors and sensible rooms', () => {
    const d = emptySample()
    const rooms: Record<string, string> = {}
    const pieces = ['bed', 'nightstand', 'nightstand', 'wardrobe', 'sofa', 'coffee-table', 'rug', 'tv-stand', 'armchair', 'floor-lamp', 'dining-table', 'dining-chair', 'dining-chair', 'desk', 'office-chair', 'plant']
    for (const c of pieces) {
      const r = drop(d, c)
      expect(r.p, `${c}: ${r.reason}`).not.toBeNull()
      rooms[`${c}${Object.keys(rooms).filter((k) => k.startsWith(c)).length || ''}`] = roomOf(d, r.p!).name
    }
    const errors = analyze(d, layoutOf(d)).issues.filter((i) => i.severity === 'error')
    expect(errors.map((e) => e.message)).toEqual([])
    expect(rooms.bed).toBe('Bedroom')
    expect(rooms.wardrobe).toBe('Bedroom')
    expect(rooms.sofa).toBe('Living room')
    expect(rooms['tv-stand']).toBe('Living room')
    expect(rooms.armchair).not.toBe('Bathroom')
    expect(Object.values(rooms).filter((r) => r === 'Bathroom').length).toBeLessThanOrEqual(1) // the plant at most
  })

  it('puts a nightstand at the head of the bed and a chair at the table', () => {
    const d = emptySample()
    const bed = drop(d, 'bed')
    const ns = drop(d, 'nightstand')
    // In bed coordinates: beside it (|x| past the side) and at the head end.
    const l = (q: Placement, o: Placement) => {
      const dx = q.x - o.x
      const dy = q.y - o.y
      const c = Math.cos(-o.rotation)
      const s = Math.sin(-o.rotation)
      return v(dx * c - dy * s, dx * s + dy * c)
    }
    const nl = l(ns.p!, bed.p!)
    expect(Math.abs(nl.x)).toBeGreaterThan(bed.item.width / 2)
    expect(Math.abs(nl.x)).toBeLessThan(bed.item.width / 2 + 0.6)
    expect(nl.y).toBeLessThan(0)
    const table = drop(d, 'dining-table')
    const chair = drop(d, 'dining-chair')
    const cl = l(chair.p!, table.p!)
    const gapX = Math.abs(cl.x) - table.item.width / 2
    const gapY = Math.abs(cl.y) - table.item.depth / 2
    expect(Math.max(gapX, gapY)).toBeLessThan(0.6)
  })

  it('returns a reason instead of placing a piece that fits nowhere', () => {
    const d = emptySample()
    const r = findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry('box'), 'huge', { name: 'Huge box', width: 9, depth: 9 }))
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.reason).toMatch(/No free spot for a 900 × 900 cm huge box/)
  })

  it('uses a studio even though its name is not a bedroom', () => {
    const d = emptySample()
    d.rooms = [rectRoom('Studio', 0, 0, 5, 4)]
    const { p } = drop(d, 'bed')
    expect(p).not.toBeNull()
    void wallInwardNormal
  })

  // ---- Round 3 (interior designer review) ----------------------------------

  /** The reviewer's tight one-bedroom: living 4.0×3.6, kitchen 2.6×3.6, bedroom 3.0×3.2, bathroom. */
  function tightOneBedroom(): Doc {
    const d = emptyDoc()
    const lv = rectRoom('Living room', 0, 0, 4.0, 3.6)
    const kt = rectRoom('Kitchen', 4.12, 0, 2.6, 3.6)
    const br = rectRoom('Bedroom', 0, 3.72, 3.0, 3.2)
    const ba = rectRoom('Bathroom', 3.12, 3.72, 2.0, 2.4)
    const door = (id: string, wall: number, offset: number, width: number, swing: 'in' | 'out', extra: Partial<Opening> = {}): Opening =>
      ({ id, kind: 'door', wall, offset, width, height: 2.1, sill: 0, hinge: 'end', swing, ...extra }) as Opening
    lv.openings.push(door('d1', 3, 1.0, 0.9, 'in', { entrance: true }), { id: 'w1', kind: 'window', wall: 0, offset: 1.2, width: 1.6, height: 1.4, sill: 0.8 } as Opening, door('d2', 1, 1.2, 0.9, 'out'), door('d3', 2, 2.0, 0.8, 'out'))
    br.openings.push(door('d4', 1, 0.6, 0.75, 'in', { hinge: 'start' }), { id: 'w2', kind: 'window', wall: 2, offset: 0.8, width: 1.2, height: 1.4, sill: 0.8 } as Opening)
    d.rooms = [lv, kt, br, ba]
    return d
  }
  const DESIGNER_LIST = ['bed', 'nightstand', 'nightstand', 'dresser', 'wardrobe', 'sofa', 'tv-stand', 'coffee-table', 'rug', 'armchair', 'floor-lamp', 'side-table', 'dining-table', 'dining-chair', 'dining-chair', 'dining-chair', 'dining-chair', 'bookshelf', 'desk', 'office-chair', 'plant', 'plant', 'console-table']
  const furnish = (d: Doc) => DESIGNER_LIST.map((c) => ({ c, ...drop(d, c, c === 'bed' ? { name: 'Queen bed' } : {}) }))
  const local = (q: Placement, o: Placement) => {
    const dx = q.x - o.x
    const dy = q.y - o.y
    const c = Math.cos(-o.rotation)
    const s = Math.sin(-o.rotation)
    return v(dx * c - dy * s, dx * s + dy * c)
  }

  it('keeps chairs at their table or desk, or says why not (round 3)', () => {
    for (const d of [tightOneBedroom(), emptySample()]) {
      const steps = furnish(d)
      const l = layoutOf(d)
      const ofCat = (c: string) => l.placements.filter((p) => d.items.find((i) => i.id === p.itemId)!.category === c)
      const body = (p: Placement) => footprint(d.items.find((i) => i.id === p.itemId)!, p)
      const tables = ofCat('dining-table')
      expect(tables.length).toBe(1)
      for (const ch of ofCat('dining-chair')) {
        expect(polygonGap(body(ch), body(tables[0]))).toBeLessThanOrEqual(0.3)
        expect(roomOf(d, ch).id).toBe(roomOf(d, tables[0]).id)
      }
      const desks = ofCat('desk')
      // With no desk in the home, an office chair is just a chair.
      for (const ch of desks.length ? ofCat('office-chair') : []) {
        expect(polygonGap(body(ch), body(desks[0]))).toBeLessThanOrEqual(0.35)
        const desk = d.items.find((i) => i.id === desks[0].itemId)!
        expect(local(ch, desks[0]).y).toBeGreaterThan(desk.depth / 2)
      }
      for (const s of steps.filter((x) => !x.p && /chair/.test(x.c))) expect(s.reason).toMatch(/No room at the (table|desk)|would block the way/)
      // Nightstands sit by the pillows of the bed.
      const bed = ofCat('bed')[0]
      const bedItem = d.items.find((i) => i.id === bed.itemId)!
      for (const ns of ofCat('nightstand')) {
        expect(polygonGap(body(ns), body(bed))).toBeLessThanOrEqual(0.15)
        expect(local(ns, bed).y).toBeLessThan(0)
        expect(Math.abs(local(ns, bed).x)).toBeGreaterThanOrEqual(bedItem.width / 2)
      }
    }
  }, 30000)

  it('keeps bedroom pieces in the bedroom and the sofa set out of the kitchen (round 3)', () => {
    const d = tightOneBedroom()
    const steps = furnish(d)
    for (const s of steps) {
      if (['bed', 'nightstand', 'dresser', 'wardrobe'].includes(s.c)) {
        if (s.p) expect(s.roomName, s.c).toBe('Bedroom')
        else expect(s.reason, s.c).toMatch(/bedroom's full/i)
      }
      if (['sofa', 'tv-stand', 'coffee-table', 'rug'].includes(s.c)) expect(s.roomName, s.c).toBe('Living room')
    }
    // The sofa faces the TV.
    const s = scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
    expect(s.goals.find((g) => g.id === 'sofa-tv')?.done).toBe(true)
  }, 30000)

  it('gives wardrobes, dressers and desks their whole front clearance, inside the room (round 3)', () => {
    for (const d of [sampleDoc(), tightOneBedroom(), emptySample()]) {
      if (d.items.length) for (const c of ['nightstand', 'nightstand', 'dresser', 'wardrobe', 'floor-lamp', 'plant', 'armchair', 'bookshelf', 'desk']) drop(d, c)
      else furnish(d)
      const l = layoutOf(d)
      for (const p of l.placements) {
        const item = d.items.find((i) => i.id === p.itemId)!
        if (!['wardrobe', 'dresser', 'desk'].includes(item.category) || !p.id.startsWith('tp')) continue
        // A desk keeps its whole front; a dresser or wardrobe at least 75 cm (90 cm is the ideal).
        const front = item.category === 'desk' ? item.clearance.front : MIN_DRAWER_FRONT
        const fz = clearanceZones({ ...item, clearance: { ...item.clearance, front } }, p).find((z) => z.side === 'front')!
        expect(polygonInsideRoom(fz.poly, roomOf(d, p).polygon, 0.01), item.name).toBe(true)
        for (const q of l.placements) {
          const o = d.items.find((i) => i.id === q.itemId)!
          if (q === p || o.floorLevel || (item.category === 'desk' && o.category === 'office-chair')) continue
          expect(convexOverlap(fz.poly, footprint(o, q), 0.005), `${o.name} in front of the ${item.name}`).toBe(false)
        }
      }
      const msgs = checkLayout(d, l).map((i) => i.message)
      expect(msgs.filter((m) => /^Desk needs .* in front/.test(m))).toEqual([])
    }
  }, 30000)

  it('fits a dresser beside the sample bed with 75 cm in front, preferring 90 cm (round 5)', () => {
    const d = sampleDoc()
    const { item, p } = drop(d, 'dresser')
    expect(p).not.toBeNull()
    expect(roomOf(d, p!).name).toBe('Bedroom')
    expect(item.clearance.front).toBe(0.9)
    // In an empty bedroom the full 90 cm wins.
    const e = emptySample()
    const r = drop(e, 'dresser')
    const fz = clearanceZones(r.item, r.p!).find((z) => z.side === 'front')!
    expect(polygonInsideRoom(fz.poly, roomOf(e, r.p!).polygon, 0.01)).toBe(true)
  })

  it('puts a lamp by the armchair when the sofa already has one (round 3)', () => {
    const d = emptySample()
    drop(d, 'sofa')
    drop(d, 'floor-lamp')
    const chair = drop(d, 'armchair')
    const lamp = drop(d, 'floor-lamp')
    expect(lamp.p).not.toBeNull()
    expect(polygonGap(footprint(lamp.item, lamp.p!), footprint(chair.item, chair.p!))).toBeLessThanOrEqual(0.5)
  }, 30000)

  it('puts a plant in an empty room (round 3)', () => {
    const d = emptySample()
    drop(d, 'sofa')
    drop(d, 'bed')
    const plant = drop(d, 'plant')
    expect(plant.roomName).toBe('Bathroom')
  }, 30000)

  it('never costs stars or a walkway for a chair pulled up to the table (round 3)', () => {
    for (const d of [tightOneBedroom(), emptySample()]) {
      let last = -1
      let walk: boolean | undefined
      for (const c of DESIGNER_LIST) {
        const r = drop(d, c)
        const s = scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
        const w = s.goals.find((g) => g.id === 'walkways')?.done
        if (c === 'dining-chair' && r.p) {
          expect(s.stars, 'a chair cost stars').toBeGreaterThanOrEqual(last)
          if (walk) expect(w, 'a chair closed a walkway').toBe(true)
        }
        last = s.stars
        walk = w
      }
    }
  }, 30000)
})

describe('findDrop round 5', () => {
  /** A living room with a sofa facing a TV across it. */
  function tvRoom(w = 5, h = 4.5): Doc {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, w, h))
    drop(d, 'sofa')
    drop(d, 'tv-stand')
    return d
  }
  const pieceAt = (d: Doc, c: string) => {
    const l = layoutOf(d)
    return l.placements.filter((p) => d.items.find((i) => i.id === p.itemId)!.category === c).map((p) => ({ p, item: d.items.find((i) => i.id === p.itemId)! }))
  }
  /** The rectangle between the sofa's front and the TV. */
  function viewRect(d: Doc) {
    const [s] = pieceAt(d, 'sofa')
    const [t] = pieceAt(d, 'tv-stand')
    const fr = v(-Math.sin(s.p.rotation), Math.cos(s.p.rotation))
    const reach = (t.p.x - s.p.x) * fr.x + (t.p.y - s.p.y) * fr.y - t.item.depth / 2 - s.item.depth / 2
    expect(reach).toBeGreaterThan(1)
    return footprint({ ...s.item, width: s.item.width - 0.1, depth: reach }, { ...s.p, x: s.p.x + fr.x * (s.item.depth / 2 + reach / 2), y: s.p.y + fr.y * (s.item.depth / 2 + reach / 2) })
  }

  it('never puts a desk, office chair, armchair or bookshelf between the sofa and the TV', () => {
    const d = tvRoom()
    const zone = viewRect(d)
    for (const c of ['desk', 'office-chair', 'armchair', 'bookshelf', 'dining-table', 'dining-chair', 'desk', 'wardrobe']) {
      const r = drop(d, c)
      if (r.p) expect(convexOverlap(footprint(r.item, r.p), zone, 0.005), c).toBe(false)
    }
    const s = scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
    expect(s.goals.find((g) => g.id === 'sofa-tv')?.done).toBe(true)
  })

  it('refuses a wall piece kindly when no wall is free, rather than standing it in the room', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Office', 0, 0, 2.2, 2.2))
    drop(d, 'desk')
    const second = drop(d, 'desk')
    expect(second.p).toBeNull()
    expect(second.reason).toMatch(/No free wall for a desk/)
    for (const c of ['bookshelf', 'sideboard', 'wardrobe']) {
      const r = drop(d, c)
      if (!r.p) expect(r.reason, c).toMatch(/No free wall|full/)
      else {
        // Backed onto a wall.
        const room = d.rooms[0]
        const back = footprint(r.item, r.p).map((q) => Math.min(...wallSegments(room).map(([a, b]) => closestPointOnSegment(q, a, b).distance)))
        expect(Math.min(...back), c).toBeLessThan(0.02)
      }
    }
  })

  it('keeps 75 cm free along both long sides of a table for four, off the wall', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Dining room', 0, 0, 4, 3.6))
    const { item, p } = drop(d, 'dining-table')
    expect(p).not.toBeNull()
    const room = d.rooms[0]
    const sides = [1, -1].map((sd) =>
      footprint({ ...item, depth: 0.75 }, { ...p!, x: p!.x - Math.sin(p!.rotation) * sd * (item.depth / 2 + 0.375), y: p!.y + Math.cos(p!.rotation) * sd * (item.depth / 2 + 0.375) }),
    )
    for (const z of sides) expect(polygonInsideRoom(z, room.polygon, 0.01)).toBe(true)
    // Chairs pulled up count, and nothing else lands in the pull-out space.
    for (const c of ['dining-chair', 'dining-chair', 'sideboard', 'bookshelf', 'armchair', 'plant']) {
      const r = drop(d, c)
      if (r.p && c !== 'dining-chair') for (const z of sides) expect(convexOverlap(footprint(r.item, r.p), z, 0.005), c).toBe(false)
    }
  })
})
