import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { checkLayout } from '../checks'
import { convexOverlap, doorSwing, footprint, pointInPolygon, polygonInsideRoom, v, wallInwardNormal, wallSegments, closestPointOnSegment } from '../geometry'
import { findDrop } from '../placement'
import { rectRoom, sampleDoc } from '../sample'
import type { Doc, Item, Placement } from '../types'

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
    } else expect(reason).toMatch(/No free spot/)
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
})
