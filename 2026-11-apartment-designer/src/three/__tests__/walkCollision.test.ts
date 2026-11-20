import { describe, expect, it } from 'vitest'
import { sampleDoc } from '../../model/sample'
import { v } from '../../model/geometry'
import { ITEM_GAP, routeTo, sharedDoorways, walkObstacles, walkStep, walkTarget, wallSegmentsForWalking, WALL_GAP } from '../walkCollision'

const walk = (from: { x: number; y: number }, dx: number, dy: number, segs: ReturnType<typeof wallSegmentsForWalking>, steps = 200) => {
  let p = from
  for (let i = 0; i < steps; i++) p = walkStep(p, v(dx, dy), segs)
  return p
}

describe('walk-in collision', () => {
  const doc = sampleDoc()
  const segs = wallSegmentsForWalking(doc.rooms)

  it('stops short of a solid wall', () => {
    // Living room, heading right at y = 1 (no door there).
    const p = walk(v(2, 1), 0.05, 0, segs)
    expect(p.x).toBeLessThanOrEqual(5.2 - WALL_GAP + 1e-6)
    expect(p.x).toBeGreaterThan(5.2 - WALL_GAP - 0.05)
    // Straight up into the window wall.
    const q = walk(v(2, 2), 0, -0.05, segs)
    expect(q.y).toBeGreaterThanOrEqual(WALL_GAP - 1e-6)
  })

  it('does not jump a wall on one long step', () => {
    const p = walkStep(v(2, 1), v(6, 0), segs)
    expect(p.x).toBeLessThan(5.2)
  })

  it('walks through the door into the bedroom', () => {
    // The living room door on its right wall spans y 2.3 to 3.1.
    const p = walk(v(4.5, 2.7), 0.05, 0, segs, 60)
    expect(p.x).toBeGreaterThan(5.6)
  })

  it('keeps the front door shut (nothing outside)', () => {
    const p = walk(v(0.85, 3), 0, 0.05, segs)
    expect(p.y).toBeLessThan(4.2)
  })

  it('opens the neighbour wall behind a door', () => {
    const shared = sharedDoorways(doc.rooms)
    const bedroom = doc.rooms[1]
    expect(shared.get(bedroom.id)?.length).toBe(1)
  })

  it('routes a glide through a doorway', () => {
    const path = routeTo(v(2, 1), v(7, 1.5), doc.rooms, segs)
    expect(path.length).toBeGreaterThan(1)
    expect(path[path.length - 1]).toEqual(v(7, 1.5))
  })
})

describe('walking around furniture', () => {
  const sofa = { x: 2, y: 2, hw: 1, hd: 0.45, rotation: 0 }
  const box = (p: { x: number; y: number }) => Math.max(Math.abs(p.x - sofa.x) - sofa.hw, Math.abs(p.y - sofa.y) - sofa.hd)

  it('stops a tap on the sofa about 0.3 m short, on the side you come from', () => {
    const t = walkTarget(v(2.2, 2.1), v(2.2, 3.8), [sofa], [])
    expect(t.y).toBeCloseTo(2 + 0.45 + ITEM_GAP, 5)
    expect(t.x).toBeCloseTo(2.2, 5)
    expect(box(t)).toBeGreaterThanOrEqual(ITEM_GAP - 1e-6)
  })

  it('leaves a tap on open floor alone', () => {
    expect(walkTarget(v(4, 3.5), v(0.5, 3.5), [sofa], [])).toEqual(v(4, 3.5))
  })

  it('handles turned pieces', () => {
    const turned = { ...sofa, rotation: Math.PI / 2 }
    const t = walkTarget(v(2, 2.5), v(2, 3.9), [turned], [])
    // Turned a quarter, the sofa runs along y: its end is 1 m from the centre.
    expect(t.y).toBeCloseTo(3 + ITEM_GAP, 5)
  })

  it('keeps the eye out of a piece while walking, but lets it walk out of one it starts in', () => {
    let p = v(2, 3.5)
    for (let i = 0; i < 100; i++) p = walkStep(p, v(0, -0.05), walkObstacles(p, [], [sofa]))
    expect(box(p)).toBeGreaterThanOrEqual(ITEM_GAP - 0.01)
    let q = v(2, 2)
    for (let i = 0; i < 40; i++) q = walkStep(q, v(0, 0.05), walkObstacles(q, [], [sofa]))
    expect(q.y).toBeGreaterThan(3)
  })
})
