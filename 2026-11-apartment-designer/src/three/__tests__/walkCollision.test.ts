import { describe, expect, it } from 'vitest'
import { sampleDoc } from '../../model/sample'
import { v } from '../../model/geometry'
import { routeTo, sharedDoorways, walkStep, wallSegmentsForWalking, WALL_GAP } from '../walkCollision'

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
