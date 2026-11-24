import { describe, expect, it } from 'vitest'
import { guardPageZoom } from '../gestureGuard'

describe('guardPageZoom', () => {
  it("cancels iOS Safari's page-zoom gesture events, leaves pointer events alone, and cleans up", () => {
    const el = new EventTarget()
    const off = guardPageZoom(el)
    for (const t of ['gesturestart', 'gesturechange', 'gestureend']) {
      const e = new Event(t, { cancelable: true })
      el.dispatchEvent(e)
      expect(e.defaultPrevented).toBe(true)
    }
    const move = new Event('pointermove', { cancelable: true })
    el.dispatchEvent(move)
    expect(move.defaultPrevented).toBe(false)
    off()
    const after = new Event('gesturestart', { cancelable: true })
    el.dispatchEvent(after)
    expect(after.defaultPrevented).toBe(false)
  })
})
