import { describe, expect, it } from 'vitest'
import { DECOR_TOUCH_PX, decorTouchTarget } from '../components/PlanView'

// A print 0.6 wide on the north wall (its back on y = 0, the room below it) above a sofa pushed to the wall.
const decor = { p: { id: 'print', x: 1, y: 0.02, rotation: 0 }, item: { width: 0.6, depth: 0.04 } }
const sofa = { p: { id: 'sofa', x: 1, y: 0.45, rotation: 0 }, item: { width: 2, depth: 0.9 } }
const px = 0.01 // 1 screen px = 1 cm

describe('decorTouchTarget', () => {
  it('gives a touch on the sofa (its name, its middle) to the sofa, not the print over it', () => {
    expect(decorTouchTarget({ at: { x: 1, y: 0.45 }, decor, floor: [sofa], px })).toBe('sofa')
    expect(decorTouchTarget({ at: { x: 1.1, y: 0.2 }, decor, floor: [sofa], px })).toBe('sofa')
  })
  it('keeps a touch within ~8px of the wall line, or on the wall side, for the print', () => {
    expect(decorTouchTarget({ at: { x: 1, y: (DECOR_TOUCH_PX - 1) * px }, decor, floor: [sofa], px })).toBe('print')
    expect(decorTouchTarget({ at: { x: 1, y: -0.15 }, decor, floor: [sofa], px })).toBe('print')
  })
  it('gives the print the touch when nothing stands under it', () => {
    expect(decorTouchTarget({ at: { x: 1, y: 0.3 }, decor, floor: [], px })).toBe('print')
  })
  it('works on a rotated wall (east wall, room to the west)', () => {
    // Rotated a quarter turn: the back faces +x, so the room is toward -x.
    const d = { p: { id: 'shelf', x: 3.98, y: 1, rotation: Math.PI / 2 }, item: { width: 0.8, depth: 0.04 } }
    const bed = { p: { id: 'bed', x: 3.2, y: 1, rotation: Math.PI / 2 }, item: { width: 1.6, depth: 1.6 } }
    expect(decorTouchTarget({ at: { x: 3.7, y: 1 }, decor: d, floor: [bed], px })).toBe('bed')
    expect(decorTouchTarget({ at: { x: 3.99, y: 1 }, decor: d, floor: [bed], px })).toBe('shelf')
  })
})
