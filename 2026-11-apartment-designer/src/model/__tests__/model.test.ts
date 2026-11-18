import { describe, expect, it } from 'vitest'
import { checkLayout, walkingPaths } from '../checks'
import { convexOverlap, footprint, orientedRect, polygonInsideRoom, v, wallInwardNormal } from '../geometry'
import { rectRoom, sampleDoc } from '../sample'
import { snapPlacement } from '../snapping'
import { formatLength, parseLength } from '../units'
import { catalogEntry, itemFromCatalog } from '../catalog'
import type { Doc } from '../types'

describe('units', () => {
  it('parses metric and imperial input', () => {
    expect(parseLength('350', 'metric')).toBeCloseTo(3.5)
    expect(parseLength('3.5m', 'metric')).toBeCloseTo(3.5)
    expect(parseLength('35mm', 'metric')).toBeCloseTo(0.035)
    expect(parseLength(`11'6"`, 'imperial')).toBeCloseTo(3.5052)
    expect(parseLength(`11' 6`, 'imperial')).toBeCloseTo(3.5052)
    expect(parseLength('138', 'imperial')).toBeCloseTo(3.5052)
    expect(parseLength('11ft 6in', 'metric')).toBeCloseTo(3.5052)
    expect(parseLength('abc', 'metric')).toBeNull()
  })
  it('formats', () => {
    expect(formatLength(3.5, 'metric')).toBe('3.50 m')
    expect(formatLength(0.9, 'metric')).toBe('90 cm')
    expect(formatLength(3.5052, 'imperial')).toBe('11′ 6″')
  })
})

describe('geometry', () => {
  it('inward normal points into the room', () => {
    const r = rectRoom('r', 0, 0, 4, 3)
    expect(wallInwardNormal(r, 0).x).toBeCloseTo(0)
    expect(wallInwardNormal(r, 0).y).toBeCloseTo(1)
    const reversed = { ...r, polygon: [...r.polygon].reverse() }
    const n = wallInwardNormal(reversed, 0) // from (0,3) to (4,3) → inward is -y
    expect(n.x).toBeCloseTo(0)
    expect(n.y).toBeCloseTo(-1)
  })
  it('detects overlap of rotated rectangles', () => {
    const a = orientedRect(v(0, 0), 1, 0.5, 0)
    const b = orientedRect(v(1.6, 0), 1, 0.5, Math.PI / 4)
    expect(convexOverlap(a, b)).toBe(true)
    const c = orientedRect(v(3, 0), 0.5, 0.5, 0)
    expect(convexOverlap(a, c)).toBe(false)
  })
  it('flush against a wall counts as inside', () => {
    const r = rectRoom('r', 0, 0, 4, 3)
    expect(polygonInsideRoom(orientedRect(v(1, 0.5), 1, 0.5, 0), r.polygon)).toBe(true)
    expect(polygonInsideRoom(orientedRect(v(1, 0.4), 1, 0.5, 0), r.polygon)).toBe(false)
  })
})

describe('snapping', () => {
  it('pushes an item back against a nearby wall and squares it up', () => {
    const room = rectRoom('r', 0, 0, 4, 3)
    const sofa = itemFromCatalog(catalogEntry('sofa'), 's')
    // Back of the sofa (rotation 0 → back faces -y) is ~10 cm from the top wall, slightly rotated.
    const res = snapPlacement(sofa, { id: 'p', itemId: 's', x: 2, y: sofa.depth / 2 + 0.1, rotation: 0.1 }, [room], [], { grid: 0, walls: true, items: false })
    expect(res.placement.y).toBeCloseTo(sofa.depth / 2)
    expect(res.placement.rotation).toBeCloseTo(0)
  })
})

describe('checks on the sample apartment', () => {
  const doc: Doc = sampleDoc()
  const layout = doc.layouts[0]
  it('has no overlaps or wall crossings', () => {
    const issues = checkLayout(doc, layout).filter((i) => i.kind === 'overlap' || i.kind === 'wall')
    expect(issues).toEqual([])
  })
  it('flags an item dropped into a door swing', () => {
    const box = itemFromCatalog(catalogEntry('box'), 'bx', { width: 0.4, depth: 0.4 })
    const d2 = { ...doc, items: [...doc.items, box] }
    const l2 = { ...layout, placements: [...layout.placements, { id: 'pb', itemId: 'bx', x: 4.4, y: 3.9, rotation: 0 }] }
    const issues = checkLayout(d2, l2)
    expect(issues.some((i) => i.kind === 'door' && i.placementIds.includes('pb'))).toBe(true)
  })
  it('finds a walking path from the entrance to every room', () => {
    const report = walkingPaths(doc, layout)
    expect(report.entrance).toBe(true)
    for (const r of report.results) {
      expect(r.reachable, r.roomName).toBe(true)
      expect(r.narrowest).toBeGreaterThan(0.5)
    }
  })
  it('reports a blocked room when a wardrobe fills the doorway', () => {
    const w = itemFromCatalog(catalogEntry('wardrobe'), 'w', { width: 1.2, depth: 0.6 })
    const d2 = { ...doc, items: [...doc.items, w] }
    // Living → bedroom door is on the living room's right wall (x = 5.2), y from 2.3 to 3.1.
    const l2 = { ...layout, placements: [...layout.placements, { id: 'pw', itemId: 'w', x: 5.2 - 0.3, y: 2.7, rotation: Math.PI / 2 }] }
    const report = walkingPaths(d2, l2)
    const bedroom = report.results.find((r) => r.roomName === 'Bedroom')!
    expect(bedroom.reachable).toBe(false)
  })
  it('footprint of a rotated item swaps width and depth', () => {
    const item = doc.items[0]
    const fp = footprint(item, { id: 'x', itemId: item.id, x: 0, y: 0, rotation: Math.PI / 2 })
    const xs = fp.map((p) => p.x)
    expect(Math.max(...xs) - Math.min(...xs)).toBeCloseTo(item.depth)
  })
})
