import { afterEach, describe, expect, it, vi } from 'vitest'
import { ANNOUNCE_BATCH_MS, announce, onAnnounce, spellStars } from '../game/announce'

describe('spellStars', () => {
  it('reads ★ as stars', () => {
    expect(spellStars('+2★ Cozy reading nook')).toBe('+2 stars Cozy reading nook')
    expect(spellStars('+1 ★')).toBe('+1 star')
    expect(spellStars('−3★ Blocked door')).toBe('-3 stars Blocked door')
    expect(spellStars('½ ★')).toBe('half a star')
    expect(spellStars('Next goal ★')).toBe('Next goal stars')
  })
})

describe('announce', () => {
  afterEach(() => vi.useRealTimers())
  it('batches messages per politeness and drops repeats within a batch', () => {
    vi.useFakeTimers()
    const got: [string, string][] = []
    const off = onAnnounce((p, t) => got.push([p, t]))
    announce('Placed a sofa.')
    announce('+2★, 16★, Cozy')
    announce('Placed a sofa.')
    announce("Couldn't save.", 'assertive')
    announce('')
    expect(got).toEqual([])
    vi.advanceTimersByTime(ANNOUNCE_BATCH_MS)
    expect(got).toEqual([
      ['assertive', "Couldn't save."],
      ['polite', 'Placed a sofa. +2 stars, 16 stars, Cozy'],
    ])
    announce('Placed a sofa.')
    vi.advanceTimersByTime(ANNOUNCE_BATCH_MS)
    expect(got[2]).toEqual(['polite', 'Placed a sofa.'])
    off()
  })
})
