import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { emptyDoc, rectRoom, sampleDoc } from '../../model/sample'
import type { Doc, Item, Placement } from '../../model/types'
import { scoreHome, type Goal } from '../goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
const goal = (d: Doc, pred: (g: Goal) => boolean) => score(d).goals.find(pred)

let n = 0
function place(d: Doc, category: string, x: number, y: number, rotation = 0, over: Partial<Item> = {}): Placement {
  const item = itemFromCatalog(catalogEntry(category), `i${++n}`, over)
  const p = { id: `p${n}`, itemId: item.id, x, y, rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}

function bedroomDoc() {
  const d = emptyDoc()
  d.rooms.push(rectRoom('Bedroom', 0, 0, 4, 4))
  return d
}

describe('scoreHome', () => {
  it('scores an empty doc at zero', () => {
    const s = score(emptyDoc())
    expect(s.stars).toBe(0)
    expect(s.vibe).toBe(0)
    expect(s.level.name).toBe('Empty shell')
    expect(s.goals.every((g) => !g.done)).toBe(true)
  })

  it('scores the sample apartment', () => {
    const s = score(sampleDoc())
    expect(s.stars).toBeGreaterThan(3)
    expect(s.stars).toBeLessThan(s.maxStars)
    expect(s.level.index).toBeGreaterThan(0)
    const done = s.goals.filter((g) => g.done).map((g) => g.id)
    expect(done).toEqual(expect.arrayContaining(['sleep', 'sit', 'eat', 'rug', 'sofa-tv']))
    // Per-room goals carry the room's name.
    expect(s.goals.some((g) => g.roomId && /Living room|Bedroom/.test(g.title))).toBe(true)
    // Unfinished goals come first.
    const firstDone = s.goals.findIndex((g) => g.done)
    expect(s.goals.slice(firstDone).every((g) => g.done)).toBe(true)
    expect(s.stars).toBe(s.goals.filter((g) => g.done).reduce((a, g) => a + g.stars, 0))
  })

  it('completes the sleep goal when a bed is added', () => {
    const d = bedroomDoc()
    expect(goal(d, (g) => g.id === 'sleep')!.done).toBe(false)
    place(d, 'bed', 2, 1.1)
    expect(goal(d, (g) => g.id === 'sleep')!.done).toBe(true)
    expect(score(d).stars).toBeGreaterThanOrEqual(2)
  })

  it('does not count a bed poking through a wall', () => {
    const d = bedroomDoc()
    place(d, 'bed', 2, 0.2)
    expect(goal(d, (g) => g.id === 'sleep')!.done).toBe(false)
  })

  it('fails the overlap goal when furniture overlaps', () => {
    const d = bedroomDoc()
    place(d, 'armchair', 1, 1)
    expect(goal(d, (g) => g.id === 'no-overlap')!.done).toBe(true)
    place(d, 'armchair', 1.3, 1.2)
    const g = goal(d, (g) => g.id === 'no-overlap')!
    expect(g.done).toBe(false)
    expect(g.hint).toMatch(/overlaps/)
  })

  it('checks the nightstand is within 60 cm of the bed side', () => {
    const d = bedroomDoc()
    const bed = place(d, 'bed', 2, 1.1) // queen, 1.52 wide: sides at x = 1.24 and 2.76, head at y = 0.085
    const ns = place(d, 'nightstand', 2.76 + 0.25 + 0.1, 0.3) // 10 cm gap
    const g = () => goal(d, (g) => g.id === `nightstand:${bed.id}`)!
    expect(g().title).toMatch(/Bedroom/)
    expect(g().done).toBe(true)
    ns.x = 2.76 + 0.25 + 0.9 // 90 cm away
    expect(g().done).toBe(false)
    expect(g().hint).toMatch(/90 cm/)
    // Beside the foot of the bed does not count.
    ns.x = 2.76 + 0.25 + 0.05
    ns.y = 1.9
    expect(g().done).toBe(false)
  })

  it('checks coffee table spacing in front of the sofa', () => {
    const d = bedroomDoc()
    d.rooms[0].name = 'Living room'
    const sofa = place(d, 'sofa', 2, 3.5, Math.PI) // faces -y, front edge at y = 3.04
    const t = place(d, 'coffee-table', 2, 3.04 - 0.4 - 0.3)
    const g = () => goal(d, (g) => g.id === `coffee:${sofa.id}`)!
    expect(g().done).toBe(true)
    t.y = 3.04 - 0.7 - 0.3
    expect(g().done).toBe(false)
  })

  it('levels up as stars grow', () => {
    const empty = score(bedroomDoc())
    const s = score(sampleDoc())
    expect(s.level.index).toBeGreaterThan(empty.level.index)
    expect(s.level.toNext).toBeGreaterThan(0)
  })
})
