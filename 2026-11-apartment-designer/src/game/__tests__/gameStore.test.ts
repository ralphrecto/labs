import { describe, expect, it } from 'vitest'
import { useGame } from '../gameStore'

describe('best level', () => {
  it('only goes up, per home', () => {
    const { noteLevel } = useGame.getState()
    noteLevel('home-a', 2)
    noteLevel('home-a', 1)
    noteLevel('home-b', 1)
    expect(useGame.getState().bestLevel['home-a']).toBe(2)
    expect(useGame.getState().bestLevel['home-b']).toBe(1)
    noteLevel('home-a', 3)
    expect(useGame.getState().bestLevel['home-a']).toBe(3)
  })
})
