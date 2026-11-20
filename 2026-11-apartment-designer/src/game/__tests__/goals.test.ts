import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { emptyDoc, rectRoom, sampleDoc } from '../../model/sample'
import type { Doc, Item, Placement } from '../../model/types'
import { bankableLevel, levelThresholds, scoreHome, withBestLevel, type Goal } from '../goals'

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
    place(d, 'armchair', 0.6, 0.6)
    // One piece isn't a layout yet: flow goals need a few pieces.
    expect(goal(d, (g) => g.id === 'no-overlap')!.done).toBe(false)
    place(d, 'armchair', 3.4, 0.6)
    place(d, 'plant', 0.4, 3.6)
    place(d, 'plant', 3.6, 3.6)
    expect(goal(d, (g) => g.id === 'no-overlap')!.done).toBe(true)
    place(d, 'armchair', 0.9, 0.8)
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
    ns.x = 2.76 + 0.25 + 0.7 // 70 cm away
    expect(g().done).toBe(false)
    expect(g().hint).toMatch(/70 cm/)
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

  it('does not reward a lone bed with flow stars', () => {
    const d = bedroomDoc()
    place(d, 'bed', 2, 1.1)
    const s = score(d)
    for (const id of ['no-overlap', 'in-walls', 'doors-clear']) {
      const g = s.goals.find((g) => g.id === id)!
      expect(g.done).toBe(false)
      expect(g.progress).toBeCloseTo(0.25)
    }
  })

  it('only counts pieces inside a room and free of errors', () => {
    const d = sampleDoc()
    d.rooms[0].name = 'Living room'
    const rug = layoutOf(d).placements.find((p) => d.items.find((i) => i.id === p.itemId)!.category === 'rug')!
    expect(goal(d, (g) => g.id === 'rug')!.done).toBe(true)
    // Push the rug through the living room's left wall: it no longer anchors anything.
    rug.x = 0.3
    expect(goal(d, (g) => g.id === 'rug')!.done).toBe(false)
  })

  it('wants the bed in the bedroom when there is one', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4), rectRoom('Bedroom', 5.12, 0, 4, 4))
    const bed = place(d, 'bed', 2.5, 1.1)
    const g = () => goal(d, (g) => g.id === 'sleep')!
    expect(g().done).toBe(false)
    expect(g().hint).toMatch(/Bedroom/)
    bed.x = 7.1
    expect(g().done).toBe(true)
    // A studio takes the bed anywhere.
    const studio = bedroomDoc()
    studio.rooms[0].name = 'Studio'
    place(studio, 'bed', 2, 1.1)
    expect(goal(studio, (g) => g.id === 'sleep')!.done).toBe(true)
  })

  it('never asks for soft light in a bathroom', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Bathroom', 0, 0, 3, 3))
    place(d, 'armchair', 1.5, 1.5)
    expect(score(d).goals.some((g) => /Bathroom/.test(g.title))).toBe(false)
    expect(goal(d, (g) => g.id === 'light')!.done).toBe(false)
  })

  it('bases level thresholds on core goals, so buying furniture never lowers the level', () => {
    const d = sampleDoc()
    const before = score(d)
    // The sample starts part way, with room to grow.
    expect(before.level.index).toBeGreaterThanOrEqual(1)
    expect(before.level.index).toBeLessThanOrEqual(2)
    // A desk away from the window adds a bonus goal it doesn't meet.
    place(d, 'desk', 0.36, 1.0, -Math.PI / 2)
    place(d, 'side-table', 8.5, 2.6)
    const after = score(d)
    expect(after.maxStars).toBeGreaterThanOrEqual(before.maxStars)
    expect(after.coreMax).toBe(before.coreMax)
    expect(after.thresholds).toEqual(before.thresholds)
    expect(analyze(d, layoutOf(d)).issues.filter((i) => i.severity === 'error').map((i) => i.message)).toEqual([])
    expect(after.goals.some((g) => g.id === 'desk-window' && !g.done && g.bonus)).toBe(true)
    expect(after.level.index).toBeGreaterThanOrEqual(before.level.index)
    expect(levelThresholds(after.coreMax)).toEqual(after.thresholds)
  })

  it('shows the best level reached even if the score dips', () => {
    const d = sampleDoc()
    const s = score(d)
    const shown = withBestLevel(s, 4)
    expect(shown.level.index).toBe(4)
    expect(shown.level.name).toBe('Designer home')
    expect(shown.stars).toBe(s.stars)
    // A lower best leaves the score alone.
    expect(withBestLevel(s, 0)).toBe(s)
    // With errors, the level neither drops below the best nor climbs past it.
    expect(withBestLevel({ ...s, errors: 1 }, 1).level.index).toBe(1)
    expect(withBestLevel({ ...s, errors: 1, level: { ...s.level, index: 0 } }, 1).level.index).toBe(1)
    // Each threshold is above the last.
    const t = levelThresholds(s.coreMax)
    for (let i = 1; i < t.length; i++) expect(t[i]).toBeGreaterThan(t[i - 1])
  })

  it('asks for 90 cm on main walkways and 60 cm on secondary ones', () => {
    const d = sampleDoc()
    expect(goal(d, (g) => g.id === 'walkways')!.done).toBe(true)
    const withPaths = (widths: Record<string, number>) => {
      const a = analyze(d, layoutOf(d))
      a.paths.results = a.paths.results.map((r) => ({ ...r, narrowest: widths[r.roomName] ?? r.narrowest }))
      return scoreHome(d, layoutOf(d), a).goals.find((g) => g.id === 'walkways')!
    }
    expect(withPaths({ Bathroom: 0.75 }).done).toBe(true)
    const tight = withPaths({ Bedroom: 0.75 })
    expect(tight.done).toBe(false)
    expect(tight.hint).toMatch(/Bedroom narrows to 75 cm; keep 90 cm/)
    expect(withPaths({ Bathroom: 0.5 }).done).toBe(false)
  })

  it('keeps the shown level from dropping as furniture is bought, moved and broken', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4), rectRoom('Bedroom', 5.12, 0, 4, 4))
    let best = 0
    const shown: number[] = []
    const step = () => {
      const s = score(d)
      best = Math.max(best, bankableLevel(s))
      shown.push(withBestLevel(s, best).level.index)
    }
    step()
    place(d, 'bed', 7.12, 1.02)
    step()
    place(d, 'sofa', 2.5, 3.54, Math.PI)
    step()
    place(d, 'dining-table', 1.2, 1.2)
    step()
    place(d, 'dining-chair', 1.2, 1.95, Math.PI)
    step()
    place(d, 'tv-stand', 2.5, 0.23) // adds a bonus goal
    step()
    const desk = place(d, 'desk', 4.6, 1.0, -Math.PI / 2)
    step()
    desk.x = 4.0 // now it overlaps nothing but sits away from the wall; then shove it through the wall
    step()
    desk.x = 5.0
    step()
    for (let i = 1; i < shown.length; i++) expect(shown[i]).toBeGreaterThanOrEqual(shown[i - 1])
    expect(shown[shown.length - 1]).toBeGreaterThan(0)
  })
})
