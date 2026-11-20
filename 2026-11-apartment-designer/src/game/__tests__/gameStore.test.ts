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

  it('"Show me" opens the drawer tab for a furniture goal, else the goals sheet at the goal', () => {
    const g = useGame.getState()
    g.showGoal({ id: 'plant', tab: 'decor', item: 'plant' })
    expect(useGame.getState()).toMatchObject({ mode: 'buy', buyTab: 'decor', buyHint: { category: 'plant' }, sheet: null })
    g.showGoal({ id: 'walkways' })
    expect(useGame.getState()).toMatchObject({ sheet: 'goals', goalFocus: 'walkways' })
    g.openSheet(null)
    expect(useGame.getState().goalFocus).toBeNull()
  })
})
