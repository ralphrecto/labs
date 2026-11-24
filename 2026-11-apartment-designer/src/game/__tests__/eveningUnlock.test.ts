import { describe, expect, it } from 'vitest'
import { useGame } from '../gameStore'
import '../PhotoShutter' // registers the Evening toggle as Cozy's unlock

describe('Evening unlock', () => {
  it('reaching Cozy makes the Evening toggle fresh ("New") until it is used', () => {
    useGame.getState().celebrateLevel(3, 'Cozy')
    expect(useGame.getState().freshUnlocks).toContain('evening')
    useGame.getState().spendUnlock('evening')
    expect(useGame.getState().freshUnlocks).not.toContain('evening')
  })
})
