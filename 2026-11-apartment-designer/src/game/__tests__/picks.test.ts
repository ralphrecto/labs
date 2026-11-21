import { describe, expect, it } from 'vitest'
import { catalogEntry } from '../../model/catalog'
import { sampleDoc } from '../../model/sample'
import { STARTER_CATALOG } from '../../model/starterCatalog'
import { useStore } from '../../store'
import { pickChip, presetItem } from '../BuyDrawer'
import { useGame } from '../gameStore'
import { familyOf, PICKS, picksFor } from '../swatches'

const STARTER = new Map(STARTER_CATALOG.map((p) => [p.id, p]))

describe('Pick one', () => {
  it('every pick is a starter piece of the category it is offered for', () => {
    for (const [category, picks] of Object.entries(PICKS)) {
      expect(picks.length, category).toBeGreaterThanOrEqual(2)
      expect(picks.length, category).toBeLessThanOrEqual(3)
      for (const pk of picks) {
        const p = STARTER.get(pk.id)
        expect(p, `${category}: ${pk.id}`).toBeDefined()
        expect(p!.category, `${category}: ${pk.id}`).toBe(category)
        expect(pk.color).toMatch(/^#[0-9a-f]{6}$/)
      }
      // Two cards with the same label would read as the same choice.
      expect(new Set(picks.map((p) => p.label)).size, category).toBe(picks.length)
    }
  })

  it('a placed pick keeps its colour, name and size', () => {
    useStore.setState({ doc: sampleDoc(), past: [], future: [] })
    const pick = PICKS['coffee-table'][1]
    const chip = pickChip(pick)!
    const item = presetItem(catalogEntry('coffee-table'), 'pick-test', chip)
    const res = useStore.getState().addItem(item)
    expect(res.ok).toBe(true)
    const placed = useStore.getState().doc.items.find((i) => i.id === 'pick-test')!
    expect(placed.color).toBe(pick.color)
    expect(placed.name).toBe('Round coffee table')
    expect(placed.width).toBeCloseTo(0.91)
    // Without a preset the piece keeps the category's own colour.
    expect(presetItem(catalogEntry('rug'), 'plain').color).toBe(catalogEntry('rug').color)
  })

  it('remembers taste per material family and offers that colour first', () => {
    expect(familyOf('desk')).toBe('wood')
    expect(familyOf('rug')).toBe('fabric')
    expect(familyOf('pendant')).toBe('light')
    expect(picksFor('desk', {})[0].id).toBe(PICKS.desk[0].id)
    useGame.getState().setTaste('wood', '#3d3d3d')
    const desks = picksFor('desk', useGame.getState().taste)
    expect(desks[0]).toMatchObject({ id: 'desk-standing', taste: true })
    expect(desks).toHaveLength(3)
    // Another family is untouched.
    expect(picksFor('rug', useGame.getState().taste)[0].taste).toBeUndefined()
    useGame.getState().setTaste('light', '#e0b04f')
    expect(picksFor('pendant', useGame.getState().taste)[0].label).toBe('Brass')
  })
})
