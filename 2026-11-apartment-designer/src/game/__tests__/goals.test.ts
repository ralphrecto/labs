import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { emptyDoc, rectRoom, sampleDoc } from '../../model/sample'
import type { Doc, Item, Placement } from '../../model/types'
import { findDrop } from '../../model/placement'
import { bankableLevel, bestLevelKey, inferEntrance, levelThresholds, nextGoal, scoreHome, SNAG_LEVEL_CAP, withBestLevel, type Goal } from '../goals'

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
    expect(before.level.index).toBeLessThanOrEqual(3)
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

  it('only asks for what a one-room home can have', () => {
    const d = emptyDoc()
    const room = rectRoom('Living room', 0, 0, 4, 4.2)
    room.openings.push({ id: 'd1', kind: 'door', wall: 1, offset: 1, width: 0.85, height: 2.03, sill: 0 })
    d.rooms.push(room)
    const s = score(d)
    const ids = s.goals.map((g) => g.id)
    // No dining room and too small for a table on the side: no dining goal.
    expect(ids).not.toContain('eat')
    // A studio takes the bed anywhere, and nothing mentions a bedroom it doesn't have.
    expect(s.goals.find((g) => g.id === 'sleep')!.title).toBe('A cozy corner to sleep')
    expect(s.goals.every((g) => !/the bedroom(?! tab)/i.test(g.hint))).toBe(true)
    // The only door is the front door.
    expect(s.goals.find((g) => g.id === 'walkways')!.hint).not.toMatch(/front door/)
    // Flow goals waiting on furniture are marked locked; the next goal is never one of them.
    const locked = s.goals.filter((g) => g.locked)
    expect(locked.map((g) => g.id).sort()).toEqual(['doors-clear', 'in-walls', 'no-overlap', 'walkways'])
    const next = nextGoal(s)!
    expect(next.locked).toBeFalsy()
    expect(next.tab).toBeTruthy()
    // A bed anywhere counts.
    place(d, 'bed', 2, 1.1)
    expect(goal(d, (g) => g.id === 'sleep')!.done).toBe(true)
  })

  it('drops bed goals for a plan with rooms but no bedroom', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4), rectRoom('Kitchen', 5.12, 0, 3, 4))
    const ids = score(d).goals.map((g) => g.id)
    expect(ids).not.toContain('sleep')
    expect(ids).not.toContain('headboard')
    expect(ids).toContain('eat')
  })

  it('infers the front door when none is marked', () => {
    const d = sampleDoc()
    const front = d.rooms[0].openings.find((o) => o.entrance)!
    front.entrance = false
    expect(inferEntrance(d)).toEqual({ roomId: d.rooms[0].id, openingId: front.id })
    expect(goal(d, (g) => g.id === 'walkways')!.done).toBe(true)
    // With one marked, nothing to infer.
    front.entrance = true
    expect(inferEntrance(d)).toBeNull()
  })

  it('names the level the same way in the pill and the sheet', () => {
    const s = score(sampleDoc())
    expect(s.headline).toBe(s.level.name)
    const shown = withBestLevel(s, 4)
    expect(shown.headline).toBe('Designer home')
  })

  it('keys the best level per design', () => {
    const d = sampleDoc()
    const a = bestLevelKey(d)
    d.layouts.push({ id: 'other', name: 'Idea 2', placements: [] })
    d.activeLayoutId = 'other'
    expect(bestLevelKey(d)).not.toBe(a)
    expect(bestLevelKey(d)).toContain(d.apartment.id)
  })

  // ---- Round 3 (interior designer review) ----------------------------------

  /** The sample home emptied and furnished by findDrop with a designer's list: a high-level home. */
  function furnishedSample(skip: string[] = []) {
    const d = sampleDoc()
    d.items = []
    layoutOf(d).placements = []
    for (const c of ['bed', 'nightstand', 'nightstand', 'sofa', 'tv-stand', 'coffee-table', 'rug', 'armchair', 'floor-lamp', 'dining-table', 'dining-chair', 'dining-chair', 'bookshelf', 'desk', 'office-chair', 'plant', 'plant'].filter((c) => !skip.includes(c))) {
      const item = itemFromCatalog(catalogEntry(c), `f${++n}`)
      const r = findDrop(d, layoutOf(d), item)
      if (!r.ok) continue
      d.items.push(item)
      layoutOf(d).placements.push({ id: `fp${n}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation })
    }
    return d
  }
  const tightened = (d: Doc, room: string, width: number) => {
    const a = analyze(d, layoutOf(d))
    a.paths.results = a.paths.results.map((r) => (r.roomName === room ? { ...r, narrowest: width } : r))
    return scoreHome(d, layoutOf(d), a)
  }

  it('points Next at a tight walkway or an overlap before anything to buy (round 3)', () => {
    const d = furnishedSample(['plant'])
    const s = tightened(d, 'Living room', 0.45)
    // There are still things to buy...
    expect(s.goals.some((g) => !g.done && g.tab)).toBe(true)
    // ...but the walkway comes first, and it isn't a shopping trip.
    const next = nextGoal(s)!
    expect(next.id).toBe('walkways')
    expect(next.tab).toBeUndefined()
    // An overlap beats the walkway.
    const sofa = layoutOf(d).placements.find((p) => d.items.find((i) => i.id === p.itemId)!.category === 'sofa')!
    const chair = layoutOf(d).placements.find((p) => d.items.find((i) => i.id === p.itemId)!.category === 'armchair')!
    chair.x = sofa.x
    chair.y = sofa.y
    expect(nextGoal(score(d))!.id).toBe('no-overlap')
  }, 30000)

  it('points Next at a TV the sofa cannot see, after errors and walkways (round 3)', () => {
    const s = score(sampleDoc())
    const g = (id: string, extra: Partial<Goal> = {}): Goal => ({ id, title: id, hint: '', done: false, stars: 1, category: 'style', ...extra })
    const buy = g('plant', { tab: 'decor', item: 'plant', progress: 0.9 })
    const tv = g('sofa-tv', { bonus: true, snag: true, progress: 0.5 })
    expect(nextGoal({ ...s, goals: [buy, tv] })!.id).toBe('sofa-tv')
    expect(nextGoal({ ...s, goals: [buy, tv, g('walkways', { category: 'flow', snag: true })] })!.id).toBe('walkways')
    // A walkway goal that only wants the front door marked doesn't jump the queue.
    expect(nextGoal({ ...s, goals: [buy, g('walkways', { category: 'flow' })] })!.id).toBe('plant')
  }, 30000)

  it('caps the level at Cozy while walkways are tight or the layout has errors (round 3)', () => {
    const d = furnishedSample()
    const s = score(d)
    expect(s.errors).toBe(0)
    expect(s.level.index).toBeGreaterThan(SNAG_LEVEL_CAP)
    const tight = tightened(d, 'Living room', 0.45)
    expect(tight.stars).toBeLessThanOrEqual(s.stars)
    expect(tight.level.index).toBe(SNAG_LEVEL_CAP)
    expect(tight.level.name).toBe('Cozy')
    expect(tight.level.toNext).toBe(0) // "Fix the snag to reach Designer home"
    // A tight secondary path (bathroom: 60 cm) caps it too, but 75 cm there is fine.
    expect(tightened(d, 'Bathroom', 0.5).level.index).toBe(SNAG_LEVEL_CAP)
    expect(tightened(d, 'Bathroom', 0.75).level.index).toBe(s.level.index)
    // A capped level is never banked above Cozy.
    expect(bankableLevel(tight)).toBeLessThanOrEqual(SNAG_LEVEL_CAP)
  }, 30000)

  it('aims for about a third of the floor and never sends you shopping for filler (round 3)', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 4, 4))
    const g0 = goal(d, (g) => g.id.startsWith('density:'))!
    expect(g0.done).toBe(false)
    expect(g0.tab).toBeUndefined()
    // 40% coverage used to be "just right"; now it's crowded.
    place(d, 'box', 2, 2, 0, { width: 2, depth: 3.2 })
    const g = goal(d, (g) => g.id.startsWith('density:'))!
    expect(g.done).toBe(false)
    expect(g.hint).toMatch(/crowded \(aim for 15–35%\)/)
  }, 30000)

  it('wants room to get into bed: both sides for a double, one for a twin (round 3)', () => {
    const access = (d: Doc) => goal(d, (g) => g.id.startsWith('bed-access:'))!
    // A queen in the middle of the head wall: free on both sides.
    let d = bedroomDoc()
    place(d, 'bed', 2, 1.02)
    expect(access(d).done).toBe(true)
    // Pushed into the corner: one side against the wall.
    d = bedroomDoc()
    place(d, 'bed', 0.77, 1.02)
    expect(access(d).done).toBe(false)
    expect(access(d).hint).toMatch(/both sides/)
    // A twin in the corner is fine.
    d = bedroomDoc()
    place(d, 'bed', 0.5, 0.96, 0, { width: 0.97, depth: 1.91 })
    expect(access(d).done).toBe(true)
    // A desk against the queen's side blocks it.
    d = bedroomDoc()
    place(d, 'bed', 2, 1.02)
    place(d, 'desk', 3.15, 1.6, Math.PI / 2)
    expect(access(d).done).toBe(false)
  }, 30000)

  it('checks rug size and TV distance (round 3)', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 5))
    place(d, 'sofa', 2.5, 0.46)
    place(d, 'tv-stand', 2.5, 3.5, Math.PI) // 3 m from the sofa: 1.6 m stand → 2.4–4.0 m
    place(d, 'rug', 2.5, 1.5, 0, { width: 1.6, depth: 1.2 }) // narrower than the sofa
    expect(goal(d, (g) => g.id === 'tv-distance')!.done).toBe(true)
    expect(goal(d, (g) => g.id === 'rug-size')!.done).toBe(false)
    const rug = layoutOf(d).placements[2]
    d.items.find((i) => i.id === rug.itemId)!.width = 2.4
    d.items.find((i) => i.id === rug.itemId)!.depth = 1.7
    rug.y = 1.6 // front legs on it
    expect(goal(d, (g) => g.id === 'rug-size')!.done).toBe(true)
    layoutOf(d).placements[1].y = 1.9 // 1.4 m: too close
    expect(goal(d, (g) => g.id === 'tv-distance')!.done).toBe(false)
  }, 30000)
})
