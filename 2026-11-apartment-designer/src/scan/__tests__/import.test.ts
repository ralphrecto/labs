import { describe, expect, it } from 'vitest'
import { emptyDoc, sampleDoc } from '../../model/sample'
import { checkLayout } from '../../model/checks'
import { analyzeScan } from '../analyze'
import { applyEstimate } from '../apply'
import { parsePLY } from '../ply'
import { parseSpatialLM } from '../spatiallm'
import { syntheticScan } from '../synthetic'

describe('applying an estimate', () => {
  const { estimate } = analyzeScan(syntheticScan())
  it('adds a room with openings and owned, sized furniture in the active layout', () => {
    const doc = emptyDoc()
    const { doc: next, roomId, placementIds } = applyEstimate(doc, estimate, { includeRoom: true, objects: estimate.objects, roomName: 'Studio' })
    const room = next.rooms.find((r) => r.id === roomId)!
    expect(room.name).toBe('Studio')
    expect(room.scan?.scanId).toBe(estimate.scanId)
    expect(room.openings.map((o) => o.kind).sort()).toEqual(['door', 'window'])
    expect(placementIds).toHaveLength(estimate.objects.length)
    expect(next.items.every((i) => i.source === 'scan' && i.owned)).toBe(true)
    // The room starts at the origin and every detected piece sits inside it without crossing walls.
    const minX = Math.min(...room.polygon.map((p) => p.x))
    expect(minX).toBeCloseTo(0, 5)
    const layout = next.layouts.find((l) => l.id === next.activeLayoutId)!
    const wallIssues = checkLayout(next, layout).filter((i) => i.kind === 'wall')
    expect(wallIssues).toEqual([])
  })
  it('places a new scanned room beside existing rooms, or in place of one', () => {
    const doc = sampleDoc()
    const maxX = Math.max(...doc.rooms.flatMap((r) => r.polygon.map((p) => p.x)))
    const side = applyEstimate(doc, estimate, { includeRoom: true, objects: [] })
    const added = side.doc.rooms.find((r) => r.id === side.roomId)!
    expect(Math.min(...added.polygon.map((p) => p.x))).toBeGreaterThan(maxX)
    const target = doc.rooms[0]
    const swap = applyEstimate(doc, estimate, { includeRoom: true, replaceRoomId: target.id, objects: [] })
    expect(swap.doc.rooms).toHaveLength(doc.rooms.length)
    expect(swap.doc.rooms[0].id).toBe(target.id)
    expect(swap.doc.rooms[0].name).toBe(target.name)
    expect(swap.doc.rooms[0].scan).toBeTruthy()
  })
})

describe('SpatialLM layout text', () => {
  const text = `wall_0=Wall(0,0,0,4,0,0,2.7,0.1)
wall_1=Wall(4,0,0,4,-3,0,2.7,0.1)
wall_2=Wall(4,-3,0,0,-3,0,2.7,0.1)
wall_3=Wall(0,-3,0,0,0,0,2.7,0.1)
door_0=Door(wall_1,4,-1.5,1.0,0.9,2.0)
window_0=Window(wall_0,2,0,1.5,1.2,1.0)
bbox_0=Bbox(bed,1.0,-1.5,0.3,0,2.0,1.6,0.6)
bbox_1=Bbox(painting,2,0,1.6,0,1,0.05,0.7)
bbox_2=Bbox(sofa,3.5,-1.5,0.4,1.5708,2.0,0.9,0.8)`
  const est = parseSpatialLM(text)
  it('builds the room from walls', () => {
    expect(est.room?.polygon).toHaveLength(4)
    expect(est.room?.ceilingHeight).toBeCloseTo(2.7)
    expect(est.room?.openings).toHaveLength(2)
    const door = est.room!.openings.find((o) => o.kind === 'door')!
    expect(door.width).toBeCloseTo(0.9)
    expect(door.offset).toBeCloseTo(1.05, 1)
  })
  it('maps classes, skips wall art, and faces furniture away from walls', () => {
    expect(est.objects.map((o) => o.category)).toEqual(['bed', 'sofa'])
    const bed = est.objects[0]
    expect(bed.width).toBeCloseTo(1.6)
    expect(bed.depth).toBeCloseTo(2.0)
    expect(bed.name).toBe('Queen bed')
    // The bed's head is against the x = 0 wall, so it faces +x: rotation 270°.
    expect(bed.rotation).toBeCloseTo((3 * Math.PI) / 2, 2)
    const sofa = est.objects[1]
    expect(sofa.width).toBeCloseTo(2.0)
    // Back against the x = 4 wall, facing -x: rotation 90°.
    expect(sofa.rotation).toBeCloseTo(Math.PI / 2, 2)
  })
})

describe('PLY', () => {
  it('reads ASCII point clouds with colors', () => {
    const txt = `ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 0 0 255 0 0
1 0 0 0 255 0
0 1 2 0 0 255
`
    const s = parsePLY(new TextEncoder().encode(txt).buffer as ArrayBuffer, 'room.ply')
    expect(s.points.count).toBe(3)
    expect(Array.from(s.points.positions.slice(6, 9))).toEqual([0, 1, 2])
    expect(s.points.colors![3 * 1 + 1]).toBeCloseTo(1)
    expect(s.name).toBe('room')
  })
  it('reads binary splats, turning SH colors to rgb and dropping transparent ones', () => {
    const header = `ply
format binary_little_endian 1.0
element vertex 2
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
end_header
`
    const h = new TextEncoder().encode(header)
    const buf = new ArrayBuffer(h.length + 2 * 7 * 4)
    new Uint8Array(buf).set(h)
    const f = new Float32Array(buf, h.length)
    f.set([1, 2, 3, 0, 0, 0, 5, 4, 5, 6, 0, 0, 0, -5])
    const s = parsePLY(buf)
    expect(s.points.count).toBe(1)
    expect(Array.from(s.points.positions)).toEqual([1, 2, 3])
    expect(s.points.colors![0]).toBeCloseTo(0.5)
  })
})
