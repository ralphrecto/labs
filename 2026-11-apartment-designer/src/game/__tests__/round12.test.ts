import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { emptyDoc, rectRoom } from '../../model/sample'
import type { Doc, Item, Placement } from '../../model/types'
import { hueFamily, levelBlocker, nextGoal, paletteOf, RULES, scoreHome, VISITS, type Goal, type HomeScore } from '../goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))

let n = 0
function place(d: Doc, category: string, x: number, y: number, rotation = 0, over: Partial<Item> = {}): Placement {
  const item = itemFromCatalog(catalogEntry(category), `r12i${++n}`, over)
  const p = { id: `r12p${n}`, itemId: item.id, x, y, rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}
function livingRoom() {
  const d = emptyDoc()
  d.rooms.push(rectRoom('Living room', 0, 0, 6, 5))
  return d
}
const SAGE = '#5f8f5a'
const TERRACOTTA = '#b5534a'
const OAK = '#c7a98a'

describe('Your palette (round 12)', () => {
  it('names a palette from swatch names, most-used first', () => {
    expect(paletteOf([SAGE, SAGE, TERRACOTTA]).name).toBe('Sage & Terracotta')
    expect(paletteOf([SAGE, TERRACOTTA, OAK]).ok).toBe(true)
    expect(hueFamily('#c9c3b6')).toBe('neutral')
    expect(hueFamily(SAGE)).not.toBe(hueFamily(TERRACOTTA))
  })

  it('allows three hue families, or neighbouring hues plus one accent, but not a rainbow', () => {
    // Terracotta (red), sage (green), cornflower (blue), plum (violet): four far-apart families.
    const rainbow = paletteOf([TERRACOTTA, SAGE, '#5b7bb5', '#6f5a8f'])
    expect(rainbow.ok).toBe(false)
    expect(rainbow.odd).toBeTruthy()
    // Red, orange and yellow side by side, plus a blue accent.
    expect(paletteOf([TERRACOTTA, '#e0b04f', '#c8b040', '#5b7bb5']).ok).toBe(true)
    // Neutrals are free.
    expect(paletteOf([TERRACOTTA, SAGE, '#5b7bb5', '#3d3d3d', '#f2efe8']).ok).toBe(true)
  })

  it('is a bonus +1★ for three pieces painted away from their catalog colour, and never caps the level', () => {
    const d = livingRoom()
    place(d, 'sofa', 3, 1, 0, { color: SAGE })
    place(d, 'armchair', 1, 3, 0, { color: TERRACOTTA })
    place(d, 'bookshelf', 5.5, 4.6, Math.PI)
    place(d, 'plant', 0.4, 0.4)
    let g = score(d).goals.find((x) => x.id === 'palette')!
    expect(g.bonus).toBe(true)
    expect(g.stars).toBe(1)
    expect(g.done).toBe(false)
    expect(g.title).toBe('Your palette: Sage & Terracotta')
    expect(g.hint).toMatch(/2 of 3 pieces painted/)
    // Points at an unpainted piece to paint next.
    expect(g.near).toBeTruthy()
    const before = score(d)
    layoutOf(d).placements.slice(-2, -1).forEach((p) => (d.items.find((i) => i.id === p.itemId)!.color = SAGE))
    const after = score(d)
    g = after.goals.find((x) => x.id === 'palette')!
    expect(g.done).toBe(true)
    expect(after.stars).toBe(before.stars + 1)
    // A bonus: the stars a level asks for don't move.
    expect(after.coreMax).toBe(before.coreMax)
  })

  it('stays out of an empty home until something is painted', () => {
    const d = livingRoom()
    expect(score(d).goals.find((x) => x.id === 'palette')).toBeUndefined()
    place(d, 'plant', 1, 1, 0, { color: TERRACOTTA })
    expect(score(d).goals.find((x) => x.id === 'palette')).toBeDefined()
  })
})

describe('roomier rules (round 12)', () => {
  it('counts a coffee table 40 cm or 58 cm from the sofa', () => {
    expect(RULES.coffeeMin).toBeLessThanOrEqual(0.35)
    expect(RULES.coffeeMax).toBeGreaterThanOrEqual(0.6)
    for (const gap of [0.4, 0.58]) {
      const d = livingRoom()
      const sofa = place(d, 'sofa', 3, 1)
      const sofaD = catalogEntry('sofa').depth
      const tD = catalogEntry('coffee-table').depth
      place(d, 'coffee-table', 3, 1 + sofaD / 2 + gap + tD / 2)
      expect(score(d).goals.find((g) => g.id === `coffee:${sofa.id}`)!.done, `gap ${gap}`).toBe(true)
    }
  })
})

describe('level gates (round 12)', () => {
  const fake = (goals: object[], toNext: number, cap?: number) =>
    ({ errors: 0, level: { index: 3, name: 'Cozy', next: 'Designer home', toNext }, goals: goals as Goal[], cap }) as unknown as HomeScore

  it('mentions the gate when stars alone won’t reach the next level', () => {
    const core = [{ id: 'eat', title: 'A table to eat at', done: false, stars: 2, category: 'basics' }]
    expect(levelBlocker(fake(core, 2, 3))).toBe('2 more ★ and finish “A table to eat at” to reach Designer home')
    expect(levelBlocker(fake(core, 2, 5))).toBe('2 more ★ to Designer home')
    expect(levelBlocker(fake([{ id: 'walkways', title: 'Clear walkways', done: false, snag: true, stars: 2, category: 'flow' }], 1, 3))).toBe('1 more ★ and fix the snag to reach Designer home')
  })

  it('Next leads with the core goal that holds the level, not a bonus goal', () => {
    const goals = [
      { id: 'rug', title: 'A rug', done: false, bonus: true, stars: 1, category: 'style', tab: 'decor', item: 'rug', progress: 0.5 },
      { id: 'headboard', title: 'A restful bed', done: false, stars: 1, category: 'comfort', near: 'bed1', progress: 0.3 },
    ]
    expect(nextGoal(fake(goals, 0, 3), () => true)?.id).toBe('headboard')
    // With stars still to earn, the easy bonus piece is fine.
    expect(nextGoal(fake(goals, 2, 5), () => true)?.id).toBe('rug')
  })
})

describe('visitor copy (round 12)', () => {
  it('counts the plants still wanted', () => {
    const aunt = VISITS.find((v) => v.id === 'aunt')!
    const piece = (i: number) => ({ id: `pl${i}`, category: 'plant', center: { x: i, y: 0 }, rotation: 0, hw: 0.2, hd: 0.2, body: [] })
    const hint = (k: number) => {
      const pieces = Array.from({ length: k }, (_, i) => piece(i))
      return aunt.check({ pieces, all: pieces, goal: () => undefined }).hint
    }
    expect(hint(0)).toBe('0 of 3 plants. Add 3 more.')
    expect(hint(1)).toBe('1 of 3 plants. Add 2 more.')
    expect(hint(2)).toBe('2 of 3 plants. Add one more.')
  })
})
