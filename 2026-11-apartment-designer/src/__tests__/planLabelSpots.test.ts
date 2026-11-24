import { describe, expect, it } from 'vitest'
import { entranceLabelBox, entranceLabelSpot, roomLabelSpot, type Box } from '../components/PlanView'

// 1 screen px = 1 cm; a 4 x 3 m room with its front door on the south wall (y = 3), room above it.
const px = 0.01
const fs = 12 * px
const overlaps = (a: Box, b: Box) => a.maxX > b.minX && a.minX < b.maxX && a.maxY > b.minY && a.minY < b.maxY
const door = { a: { x: 2, y: 3 }, b: { x: 2.9, y: 3 }, n: { x: 0, y: -1 } }
// A rotated "2.00 m" length label of the next room's wall, standing just outside the doorway (the phone plan case).
const wallLabel: Box = { minX: 2.4, maxX: 2.54, minY: 3.05, maxY: 3.6 }

describe('entranceLabelSpot', () => {
  it('sits just outside the door when nothing is there', () => {
    const q = entranceLabelSpot({ ...door, fs, px, avoid: [] })
    expect(q.x).toBeCloseTo(2.45)
    expect(q.y).toBeCloseTo(3.45)
  })
  it('steps clear of a wall length label', () => {
    const q = entranceLabelSpot({ ...door, fs, px, avoid: [wallLabel] })
    expect(overlaps(entranceLabelBox(q, fs), wallLabel)).toBe(false)
    expect(Math.hypot(q.x - 2.45, q.y - 3.45)).toBeLessThan(1)
  })
  it('steps clear of a decor badge', () => {
    const badge: Box = { minX: 2.35, maxX: 2.5, minY: 3.38, maxY: 3.52 }
    const q = entranceLabelSpot({ ...door, fs, px, avoid: [badge] })
    expect(overlaps(entranceLabelBox(q, fs), badge)).toBe(false)
  })
})

describe('roomLabelSpot', () => {
  const room = [
    { x: 0, y: 0 },
    { x: 4, y: 0 },
    { x: 4, y: 3 },
    { x: 0, y: 3 },
  ]
  const w = 8 * fs * 0.58
  const nameBox = (q: { x: number; y: number }): Box => ({ minX: q.x - w / 2, maxX: q.x + w / 2, minY: q.y - fs * 1.1, maxY: q.y + fs * 1.5 })
  it('takes the centre when it is free', () => {
    expect(roomLabelSpot(room, { x: 2, y: 1.5 }, fs, w, [])).toEqual({ x: 2, y: 1.5 })
  })
  it('moves off a wall length label drawn inside the room', () => {
    const inside: Box = { minX: 1.8, maxX: 2.2, minY: 1.4, maxY: 1.6 }
    const q = roomLabelSpot(room, { x: 2, y: 1.5 }, fs, w, [inside])!
    expect(q).not.toBeNull()
    expect(overlaps(nameBox(q), inside)).toBe(false)
  })
})
