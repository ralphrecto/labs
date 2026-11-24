import { describe, expect, it } from 'vitest'
import { heldScore, settleStars } from '../gameStore'

describe('steady stars while dragging (r13)', () => {
  it('holds the score from before the gesture while it is open, and shows the live one after', () => {
    const before = { stars: 10 }
    const mid = { stars: 7 }
    expect(heldScore(mid, before, true)).toBe(before)
    expect(heldScore(mid, before, false)).toBe(mid)
    expect(heldScore(mid, null, true)).toBe(mid)
  })

  it('settles once on the net change of the whole gesture', () => {
    // 10 -> 7 -> 12 mid-drag reads as one +2 at the end, never a loss.
    expect(settleStars(10, 12)).toEqual({ kind: 'gain', delta: 2 })
    // 10 -> 13 -> 9 mid-drag reads as one -1 (with Undo) at the end, never a cheer.
    expect(settleStars(10, 9)).toEqual({ kind: 'loss', delta: 1 })
    // Out and back: nothing to show.
    expect(settleStars(10, 10)).toEqual({ kind: 'none', delta: 0 })
  })
})
