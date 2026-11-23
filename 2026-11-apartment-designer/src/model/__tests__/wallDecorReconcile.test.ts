import { describe, expect, it } from 'vitest'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { add, v, wallSegments } from '../geometry'
import { emptyDoc, rectRoom } from '../sample'
import type { Doc, Item, Opening, Placement, Room } from '../types'
import { curtainSpan, CURTAIN_OVERHANG, reconcileWallDecor, snapToWall, wallDecorMessage, type WallDecorReport } from '../wallDecor'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0
/** Hang a piece where snapToWall puts it near `pt`. */
function hang(d: Doc, category: string, pt: { x: number; y: number }, over: Partial<Item> = {}): Placement {
  const item = itemFromCatalog(catalogEntry(category), `r${++n}`, over)
  const s = snapToWall(d, layoutOf(d), item, v(pt.x, pt.y))!
  expect(s).toBeTruthy()
  const p: Placement = { id: `rp${n}`, itemId: item.id, x: s.x, y: s.y, rotation: s.rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}
const win = (id: string, wall: number, offset: number, width: number): Opening => ({ id, kind: 'window', wall, offset, width, height: 1.3, sill: 0.9 })
const door = (id: string, wall: number, offset: number): Opening => ({ id, kind: 'door', wall, offset, width: 0.9, height: 2.1, sill: 0, hinge: 'start', swing: 'in' })
const withRoom = (d: Doc, id: string, fn: (r: Room) => Room): Doc => ({ ...d, rooms: d.rooms.map((r) => (r.id === id ? fn(r) : r)) })
const placed = (d: Doc, id: string) => layoutOf(d).placements.find((p) => p.id === id)
const itemOf = (d: Doc, p: Placement) => d.items.find((i) => i.id === p.itemId)!

function room4() {
  const d = emptyDoc()
  const room = rectRoom('Living room', 0, 0, 4, 4)
  d.rooms.push(room)
  return { d, room }
}

describe('reconcileWallDecor', () => {
  it('leaves the doc alone when nothing about the walls changed', () => {
    const { d } = room4()
    hang(d, 'wall-art', { x: 2, y: 0.3 })
    const next = { ...d, layouts: [...d.layouts] }
    expect(reconcileWallDecor(d, next)).toBe(next)
  })

  it('follows a moved room, keeping its place along the wall', () => {
    const { d, room } = room4()
    const art = hang(d, 'wall-art', { x: 1.2, y: 0.3 })
    const moved = withRoom(d, room.id, (r) => ({ ...r, polygon: r.polygon.map((p) => add(p, v(3, 1))) }))
    const out = reconcileWallDecor(d, moved)
    const p = placed(out, art.id)!
    expect(p.x).toBeCloseTo(art.x + 3, 5)
    expect(p.y).toBeCloseTo(art.y + 1, 5)
  })

  it('keeps its relative position when the wall is stretched', () => {
    const { d, room } = room4()
    const art = hang(d, 'wall-art', { x: 1, y: 0.3 })
    // Stretch to 8 m wide: the piece at 1/4 of the top wall goes to 2 m.
    const wide = withRoom(d, room.id, (r) => ({ ...r, polygon: r.polygon.map((p) => (p.x > 2 ? v(8, p.y) : p)) }))
    const p = placed(reconcileWallDecor(d, wide), art.id)!
    expect(p.x).toBeCloseTo(2, 2)
    expect(p.y).toBeCloseTo(art.y, 5)
  })

  it('slides off a door added under it', () => {
    const { d, room } = room4()
    const art = hang(d, 'wall-art', { x: 2, y: 0.3 })
    const next = withRoom(d, room.id, (r) => ({ ...r, openings: [door('d', 0, 1.55)] }))
    const p = placed(reconcileWallDecor(d, next), art.id)!
    const w = itemOf(d, art).width
    expect(p.y).toBeCloseTo(art.y, 5)
    expect(p.x + w / 2 <= 1.55 || p.x - w / 2 >= 2.45).toBe(true)
  })

  it('goes to Mine with a message when no wall spot is left', () => {
    const d = emptyDoc()
    const room = rectRoom('Hall', 0, 0, 2, 2)
    d.rooms.push(room)
    const art = hang(d, 'wall-art', { x: 1, y: 0.2 })
    // Shrink the room to a closet too small for the frame on any wall.
    const tiny = withRoom(d, room.id, (r) => ({ ...r, polygon: [v(0, 0), v(0.3, 0), v(0.3, 0.3), v(0, 0.3)] }))
    const report: WallDecorReport = { toMine: [] }
    const out = reconcileWallDecor(d, tiny, report)
    expect(placed(out, art.id)).toBeUndefined()
    expect(itemOf(out, art).owned).toBe(true)
    expect(report.toMine).toHaveLength(1)
    expect(wallDecorMessage(report)).toMatch(/waiting in Mine/)
  })

  it('drops curtains to Mine when their window is removed', () => {
    const { d, room } = room4()
    room.openings.push(win('w', 2, 1.4, 1.2))
    const c = hang(d, 'curtains', { x: 2, y: 3.8 })
    const next = withRoom(d, room.id, (r) => ({ ...r, openings: [] }))
    const report: WallDecorReport = { toMine: [] }
    const out = reconcileWallDecor(d, next, report)
    expect(placed(out, c.id)).toBeUndefined()
    expect(report.toMine[0].why).toBe('window')
    expect(wallDecorMessage(report)).toBe('The curtains lost their window. They’re waiting in Mine.')
  })

  it('curtains follow their window when it moves along the wall', () => {
    const { d, room } = room4()
    room.openings.push(win('w', 2, 1.4, 1.2))
    const c = hang(d, 'curtains', { x: 2, y: 3.8 })
    const next = withRoom(d, room.id, (r) => ({ ...r, openings: [win('w', 2, 0.6, 1.2)] }))
    const out = reconcileWallDecor(d, next)
    const p = placed(out, c.id)!
    // Wall 2 runs from (4, 4) to (0, 4): offset 0.6..1.8 is x 3.4..2.2, centre 2.8.
    expect(p.x).toBeCloseTo(2.8, 2)
  })
})

describe('curtain widths', () => {
  it('a pair spans the window plus an overhang each side', () => {
    const { d, room } = room4()
    room.openings.push(win('w', 2, 1.4, 1.2))
    const c = hang(d, 'curtains', { x: 2, y: 3.8 })
    const out = reconcileWallDecor(d, { ...d, layouts: [...d.layouts] })
    expect(itemOf(out, c).width).toBeCloseTo(1.2 + 2 * CURTAIN_OVERHANG, 5)
    // Sized once, it stays put.
    expect(reconcileWallDecor(out, { ...out })).toEqual({ ...out })
  })

  it('a wide window gets wide curtains, a narrow one narrow curtains', () => {
    const { d, room } = room4()
    room.openings.push(win('big', 0, 0.8, 2.4), win('small', 2, 1.7, 0.6))
    expect(curtainSpan(d, room, 0, room.openings[0])).toEqual([0.5, 3.5].map((x) => expect.closeTo(x, 5)))
    const s = curtainSpan(d, room, 2, room.openings[1])!
    expect(s[1] - s[0]).toBeCloseTo(1.2, 5)
  })

  it('is clamped to the wall and kept off a neighbouring door', () => {
    const { d, room } = room4()
    // Window near the start of the top wall, a door just past it.
    room.openings.push(win('w', 0, 0.1, 1.0), door('d', 0, 1.25))
    const [lo, hi] = curtainSpan(d, room, 0, room.openings[0])!
    expect(lo).toBeCloseTo(0.02, 5)
    expect(hi).toBeLessThanOrEqual(1.25 - 0.05 + 1e-9)
    expect(hi).toBeGreaterThanOrEqual(1.1)
    // Snapped curtains take that span, centred on it (off the window's own centre).
    const c = hang(d, 'curtains', { x: 0.6, y: 0.3 })
    const out = reconcileWallDecor(d, { ...d, layouts: [...d.layouts] })
    const p = placed(out, c.id)!
    expect(itemOf(out, c).width).toBeCloseTo(hi - lo, 5)
    expect(p.x).toBeCloseTo((lo + hi) / 2, 3)
    expect(wallSegments(room)[0][0].y).toBe(0)
  })

  it('two windows close together share the gap between them', () => {
    const { d, room } = room4()
    room.openings.push(win('a', 0, 0.6, 1.0), win('b', 0, 1.8, 1.0))
    const a = curtainSpan(d, room, 0, room.openings[0])!
    const b = curtainSpan(d, room, 0, room.openings[1])!
    // The 0.2 m between them is split down the middle: no overlap.
    expect(a[1]).toBeCloseTo(1.7, 5)
    expect(b[0]).toBeCloseTo(1.7, 5)
    expect(a[0]).toBeCloseTo(0.3, 5)
  })
})

describe('store: wall decor after room edits', () => {
  it('removing a window takes its curtains down to Mine, with a toast', async () => {
    const { useStore } = await import('../../store')
    const { d, room } = room4()
    room.openings.push(win('w', 2, 1.4, 1.0))
    const c = hang(d, 'curtains', { x: 2, y: 3.8 })
    useStore.getState().replaceDoc(d)
    // Sized to the window on the first edit that touches it.
    useStore.getState().updateRoom(room.id, { name: 'Lounge' })
    expect(itemOf(useStore.getState().doc, c).width).toBeCloseTo(1.6, 5)
    useStore.getState().updateRoom(room.id, { openings: [] })
    const after = useStore.getState().doc
    expect(placed(after, c.id)).toBeUndefined()
    expect(itemOf(after, c).owned).toBe(true)
    await Promise.resolve()
    expect(useStore.getState().toast).toMatch(/waiting in Mine/)
  })
})
