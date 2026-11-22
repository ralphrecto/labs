import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { emptyDoc, rectRoom, sampleDoc } from '../../model/sample'
import type { Doc, Item, Placement, TwistId } from '../../model/types'
import { ANCHORS, applicableTwists, finishedTwists, pickTwists, remixedLayout, sameArrangement, talkCornerSpot } from '../goals'
import { findDrop } from '../../model/placement'
import { unpackDoc } from '../../store'
import { bankableLevel, bestLevelKey, DREAM_BONUS_STARS, levelBlocker, inferEntrance, LEVEL_FRACTIONS, LEVELS, levelThresholds, nextGoal, scoreHome, SNAG_LEVEL_CAP, WARNING_LEVEL_CAP, withBestLevel, type Goal } from '../goals'

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

  it('starts the sample home on move-in day, with plenty to fill (round 4)', () => {
    const s = score(sampleDoc())
    expect(s.level.name).toBe('Lived-in')
    expect(s.errors).toBe(0)
    // Several core goals still open: storage, a table, light, a plant, the bathroom...
    expect(s.goals.filter((g) => !g.done && !g.bonus).length).toBeGreaterThanOrEqual(6)
    // The bed's head is against a solid wall from the start (round 6), so one star fewer to go.
    expect(s.level.toNext).toBeGreaterThanOrEqual(2)
  })

  it('scores the sample apartment', () => {
    const s = score(sampleDoc({ settled: true }))
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
    const t = place(d, 'coffee-table', 2, 3.04 - 0.47 - 0.3)
    const g = () => goal(d, (g) => g.id === `coffee:${sofa.id}`)!
    expect(g().done).toBe(true)
    // Inside the sofa's 45 cm knee room it blocks the sofa's clearance: it doesn't count (round 5).
    t.y = 3.04 - 0.4 - 0.3
    expect(g().done).toBe(false)
    expect(g().hint).toMatch(/knees/)
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
    const d = sampleDoc({ settled: true })
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
    const d = sampleDoc({ settled: true })
    const before = score(d)
    // The sample starts part way, with room to grow.
    expect(before.level.index).toBeGreaterThanOrEqual(1)
    expect(before.level.index).toBeLessThanOrEqual(3)
    // A desk away from the window adds a bonus goal it doesn't meet.
    place(d, 'desk', 0.36, 1.0, -Math.PI / 2, { clearance: { front: 0, back: 0, left: 0, right: 0 } })
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
    // Packed moving boxes don't count as furniture; as plain boxes the sample has enough pieces for the flow goals.
    for (const i of d.items) delete i.unpack
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
    // One room called "Living room" is a living room, not a studio: no bed goals, and Next
    // starts on the sofa (round 4).
    expect(ids).not.toContain('sleep')
    expect(s.livingFirst).toBe(true)
    expect(nextGoal(s)!.id).toBe('sit')
    room.name = 'Studio'
    const st = score(d)
    expect(st.livingFirst).toBe(false)
    // A studio takes the bed anywhere, and nothing mentions a bedroom it doesn't have.
    expect(st.goals.find((g) => g.id === 'sleep')!.title).toBe('A cozy corner to sleep')
    expect(st.goals.every((g) => !/the bedroom(?! tab)/i.test(g.hint))).toBe(true)
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
    // Packed moving boxes don't count as furniture; as plain boxes the sample has enough pieces for the flow goals.
    for (const i of d.items) delete i.unpack
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
    // Designer home also needs the core goals done (round 6), so the furnished sample may sit at Cozy.
    expect(s.level.index).toBeGreaterThanOrEqual(SNAG_LEVEL_CAP)
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

describe('nextGoal never dead-ends (round 4)', () => {
  it('suggests a stand-in when the hinted piece has no room, or moves on', () => {
    const s = score(sampleDoc())
    const storage: Goal = { id: 'storage', title: 'A place for your things', hint: '', done: false, stars: 1, category: 'basics', tab: 'bedroom', item: 'dresser' }
    const plant: Goal = { id: 'plant', title: 'Greenery', hint: '', done: false, stars: 1, category: 'style', tab: 'decor', item: 'plant' }
    const g = nextGoal({ ...s, goals: [storage, plant] }, (c) => c !== 'dresser' && c !== 'wardrobe')!
    expect(g.id).toBe('storage')
    expect(g.item).toBe('bookshelf')
    expect(g.tab).toBe('work')
    // Nothing for storage fits: the next goal instead.
    expect(nextGoal({ ...s, goals: [storage, plant] }, (c) => c === 'plant')!.id).toBe('plant')
    // Nothing fits at all: the goal, without a drawer trip (Show me opens the sheet).
    const last = nextGoal({ ...s, goals: [storage] }, () => false)!
    expect(last.tab).toBeUndefined()
  })

  it('suggests only pieces that fit, on the sample home', () => {
    const d = sampleDoc()
    const l = layoutOf(d)
    const fits = (c: string) => findDrop(d, l, itemFromCatalog(catalogEntry(c), 'probe')).ok
    for (let i = 0; i < 12; i++) {
      const g = nextGoal(score(d), fits)
      if (!g?.item || !g.tab) break
      const item = itemFromCatalog(catalogEntry(g.item), `n${i}`)
      const r = findDrop(d, l, item)
      expect(r.ok).toBe(true)
      if (!r.ok) break
      d.items.push(item)
      l.placements.push({ id: `np${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation })
    }
  }, 30000)

  it('never awards a bed that has no room to get in', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Studio', 0, 0, 3, 2.8))
    place(d, 'bed', 0.8, 1.05)
    const s = score(d)
    expect(s.goals.find((g) => g.id === 'sleep')!.done).toBe(false)
    expect(s.goals.find((g) => g.id === 'headboard')!.done).toBe(false)
  })
})

describe('round 5 (play-test review)', () => {
  let m = 0
  function put(d: Doc, category: string, x: number, y: number, rotation = 0, over: Partial<Item> = {}): Placement {
    const item = itemFromCatalog(catalogEntry(category), `r5i${++m}`, over)
    const p = { id: `r5p${m}`, itemId: item.id, x, y, rotation }
    d.items.push(item)
    layoutOf(d).placements.push(p)
    return p
  }
  function tvLiving() {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 5))
    put(d, 'sofa', 2.5, 0.47, 0) // faces +y
    put(d, 'tv-stand', 2.5, 4.0, Math.PI)
    return d
  }

  it('makes the sofa-TV goal a snag when a desk or office chair stands in the way', () => {
    const d = tvLiving()
    expect(goal(d, (g) => g.id === 'sofa-tv')!.done).toBe(true)
    const desk = put(d, 'desk', 2.5, 2.5)
    const g = goal(d, (g) => g.id === 'sofa-tv')!
    expect(g.done).toBe(false)
    expect(g.snag).toBe(true)
    expect(g.hint).toMatch(/desk is in the way/i)
    expect(nextGoal(score(d))!.id).toBe('sofa-tv')
    desk.x = 0.6 // aside, off the line of sight
    expect(goal(d, (g) => g.id === 'sofa-tv')!.done).toBe(true)
    put(d, 'office-chair', 2.0, 2.4)
    expect(goal(d, (g) => g.id === 'sofa-tv')!.snag).toBe(true)
  })

  it('a coffee table, rug, ottoman or side table between them is fine', () => {
    const d = tvLiving()
    put(d, 'coffee-table', 2.5, 0.93 + 0.47 + 0.3)
    put(d, 'rug', 2.5, 1.6)
    put(d, 'ottoman', 2.5, 2.8)
    expect(goal(d, (g) => g.id === 'sofa-tv')!.done).toBe(true)
  })

  it('holds the level at Designer home while a warning is on screen, and Dream home needs bonus stars', () => {
    const d = tvLiving()
    const a = analyze(d, layoutOf(d))
    const s = scoreHome(d, layoutOf(d), a)
    // Pretend every star is earned: the level is set by the caps alone.
    const all = { ...a, issues: [...a.issues, { id: 'clear:x:front:y', severity: 'warning' as const, kind: 'clearance' as const, message: 'Sofa needs 45 cm in front', placementIds: ['x'] }] }
    const warned = scoreHome(d, layoutOf(d), all)
    expect(warned.level.index).toBeLessThanOrEqual(WARNING_LEVEL_CAP)
    expect(s.level.index).toBeLessThan(LEVELS.length - 1)
    expect(LEVEL_FRACTIONS[4]).toBeGreaterThan(0.9)
  })

  it('never reaches Dream home with warnings or without bonus stars while furnishing by the Next chip', () => {
    const d = sampleDoc()
    const fits = (c: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe')).ok
    const levels: string[] = []
    for (let i = 0; i < 30; i++) {
      const a = analyze(d, layoutOf(d))
      const s = scoreHome(d, layoutOf(d), a)
      levels.push(s.level.name)
      if (s.level.index === LEVELS.length - 1) {
        expect(a.issues.filter((x) => x.severity !== 'info' && !(x.kind === 'clearance' && /^(Dresser|Wardrobe) needs/.test(x.message)))).toEqual([])
        expect(s.goals.filter((g) => g.bonus && g.done).reduce((t, g) => t + g.stars, 0)).toBeGreaterThanOrEqual(DREAM_BONUS_STARS)
      }
      const next = nextGoal(s, fits)
      if (next?.id === 'unpack' && next.near) {
        Object.assign(d, unpackDoc(d, next.near)!.doc)
        continue
      }
      if (!next?.item || !next.tab) break
      const item = itemFromCatalog(catalogEntry(next.item), `nx${i}`)
      const r = findDrop(d, layoutOf(d), item, { nextTo: next.item === 'floor-lamp' || next.item === 'table-lamp' ? next.near : undefined })
      if (!r.ok) break
      d.items.push(item)
      layoutOf(d).placements.push({ id: `nxp${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
    }
    // The sample takes a good few placements to go from Lived-in to Designer home.
    const designer = levels.indexOf('Designer home')
    // (Chairs now count at the table right away, so it's a little quicker than round 5's 8.)
    expect(designer === -1 || designer >= 6, levels.join(' > ')).toBe(true)
    expect(score(d).errors).toBe(0)
  }, 60000)

  it('counts a dining chair only when it can be pulled out', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Dining room', 0, 0, 4, 4))
    put(d, 'dining-table', 2, 2)
    // Jammed against the wall: its back 75 cm runs into the wall.
    const chair = put(d, 'dining-chair', 2, 0.25, 0)
    const t = () => goal(d, (g) => g.id === 'eat')!
    expect(t().done).toBe(false)
    // Pulled up to the table from the open side, with room behind.
    chair.y = 2 + 0.45 + 0.05 + 0.25
    chair.rotation = Math.PI
    expect(t().done).toBe(true)
  })

  it('gives the reading-corner goal the armchair as its target, and the lamp lands beside it', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    put(d, 'sofa', 2.5, 0.47)
    const chair = put(d, 'armchair', 4.2, 2.6, Math.PI / 2) // faces -x, into the room
    const g = goal(d, (g) => g.id === 'nook')!
    expect(g.near).toBe(chair.id)
    const lamp = itemFromCatalog(catalogEntry('floor-lamp'), 'lamp')
    const r = findDrop(d, layoutOf(d), lamp, { nextTo: g.near })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    d.items.push(lamp)
    layoutOf(d).placements.push({ id: 'lampP', itemId: lamp.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation })
    expect(goal(d, (g) => g.id === 'nook')!.done).toBe(true)
  })
})

describe('round 6 (play-test review)', () => {
  let k = 0
  const drop = (d: Doc, category: string, over: Partial<Item> = {}) => {
    const item = itemFromCatalog(catalogEntry(category), `r6i${++k}`, over)
    const r = findDrop(d, layoutOf(d), item)
    expect(r.ok, r.ok ? '' : r.reason).toBe(true)
    if (!r.ok) return undefined
    d.items.push(item)
    const p = { id: `r6p${k}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation }
    layoutOf(d).placements.push(p)
    return p
  }

  it('seats two chairs at a dining table for four with room to pull them out', () => {
    const d = sampleDoc()
    drop(d, 'dining-table', { width: 1.22, depth: 0.76 })
    const c1 = drop(d, 'dining-chair')!
    const c2 = drop(d, 'dining-chair')!
    const chairIssues = analyze(d, layoutOf(d)).issues.filter((i) => i.kind === 'clearance' && (i.placementIds.includes(c1.id) || i.placementIds.includes(c2.id)))
    expect(chairIssues.map((i) => i.message)).toEqual([])
    expect(goal(d, (g) => g.id === 'eat')!.done).toBe(true)
  })

  it('asks to move the table off the wall when the chairs at it have no room to pull out', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Dining room', 0, 0, 4, 4))
    const chairItem = itemFromCatalog(catalogEntry('dining-chair'), 'probe')
    const tableItem = itemFromCatalog(catalogEntry('dining-table'), 'probe2')
    const table = place(d, 'dining-table', 2, chairItem.depth + 0.05 + tableItem.depth / 2)
    place(d, 'dining-chair', 2, chairItem.depth / 2, 0) // against the wall, facing the table
    const g = goal(d, (g) => g.id === 'eat')!
    expect(g.done).toBe(false)
    expect(g.hint).toMatch(/off the wall/i)
    expect(g.item).toBeUndefined()
    expect(g.near).toBe(table.id)
  })

  it('needs a real lamp for the reading corner: a side table alone is a start', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    place(d, 'armchair', 2, 2)
    const a = itemFromCatalog(catalogEntry('armchair'), 'probe')
    const side = place(d, 'side-table', 2 + a.width / 2 + 0.3, 2)
    const g = goal(d, (g) => g.id === 'nook')!
    expect(g.done).toBe(false)
    expect(g.progress).toBeGreaterThan(0.5)
    expect(g.progress).toBeLessThan(1)
    side.y = 3.5
    place(d, 'floor-lamp', 2 + a.width / 2 + 0.3, 2)
    expect(goal(d, (g) => g.id === 'nook')!.done).toBe(true)
  })

  it('reaches Dream home by the Next chip in 12 to 25 placements, with no dead-end loops', () => {
    const d = sampleDoc()
    const fits = (c: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe')).ok
    let dream = -1
    let last = ''
    let repeats = 0
    let best = -1
    for (let i = 0; i < 30; i++) {
      const s = score(d)
      if (s.level.index === LEVELS.length - 1) {
        dream = i
        break
      }
      const next = nextGoal(s, fits)
      // "Unpack your boxes": the chip selects the box and its bar unpacks it; that's a step too.
      if (next?.id === 'unpack' && next.near) {
        const r = unpackDoc(d, next.near)
        expect(r, 'unpack chip points at a packed box').not.toBeNull()
        Object.assign(d, r!.doc)
        continue
      }
      if (!next?.item || !next.tab) break
      const key = `${next.id}:${next.item}`
      repeats = key === last && s.stars <= best ? repeats + 1 : 1
      expect(repeats, `${key} offered ${repeats} times without progress`).toBeLessThanOrEqual(3)
      last = key
      best = Math.max(best, s.stars)
      const item = itemFromCatalog(catalogEntry(next.item), `dn${i}`)
      const r = findDrop(d, layoutOf(d), item, { nextTo: next.near })
      if (!r.ok) break
      d.items.push(item)
      layoutOf(d).placements.push({ id: `dnp${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
    }
    expect(dream, `Dream home after ${dream} placements`).toBeGreaterThanOrEqual(12)
    expect(dream).toBeLessThanOrEqual(25)
  }, 60000)
})

describe('remix this home (round 8)', () => {
  let k = 0
  const ids = () => `rx${++k}`
  /** Remix the active design of d with a twist, and play the remix. */
  function remix(d: Doc, twist: TwistId) {
    const l = remixedLayout(d, d.activeLayoutId, twist, ids)!
    d.layouts.push(l)
    d.activeLayoutId = l.id
    return l
  }
  const twistOf = (d: Doc) => score(d).goals.find((g) => g.id.startsWith('twist:'))!
  const cat = (d: Doc, p: Placement) => d.items.find((i) => i.id === p.itemId)!.category

  it('keeps only the big pieces, names the remix after its twist and starts well below Dream home', () => {
    const d = sampleDoc()
    const src = layoutOf(d)
    const l = remix(d, 'window-seat')
    expect(l.name).toBe('Window seat')
    expect(l.remix).toMatchObject({ from: src.id, twist: 'window-seat', anchors: l.placements.length })
    expect(l.placements.length).toBeGreaterThan(0)
    expect(l.placements.every((p) => ANCHORS.has(cat(d, p)) && !p.on)).toBe(true)
    expect(l.placements.every((p) => !src.placements.some((q) => q.id === p.id))).toBe(true)
    expect(l.placements.length).toBeLessThan(src.placements.length)
    expect(score(d).level.index).toBeLessThanOrEqual(LEVELS.indexOf('Lived-in'))
    expect(remixedLayout(d, src.id, 'window-seat', ids)!.name).toBe('Window seat 2')
    // Pieces left out stay in the home's library, ready to place again.
    expect(d.items.length).toBe(sampleDoc().items.length)
  })

  it('offers only twists the plan can take', () => {
    expect(applicableTwists(sampleDoc())).toEqual(expect.arrayContaining(['bed-swap', 'talk-corner', 'less-is-more']))
    // The sample's sofa already looks out of the window: no challenge in that.
    expect(applicableTwists(sampleDoc())).not.toContain('window-seat')
    const bare = emptyDoc()
    bare.rooms.push(rectRoom('Office', 0, 0, 3, 3))
    const t = applicableTwists(bare, () => false)
    expect(t).not.toContain('window-seat')
    expect(t).not.toContain('bed-swap')
    expect(t).not.toContain('talk-corner')
  })

  it('window seat: the sofa looks at a window in its room', () => {
    const d = sampleDoc()
    remix(d, 'window-seat')
    expect(twistOf(d)).toMatchObject({ id: 'twist:window-seat', done: true, category: 'style', stars: 3 })
    const sofa = layoutOf(d).placements.find((p) => cat(d, p) === 'sofa')!
    sofa.rotation = 0
    expect(twistOf(d).done).toBe(false)
  })

  it('fresh spots: every big piece at least 80 cm from where it stood', () => {
    const d = bedroomDoc()
    place(d, 'bed', 2, 1.2)
    place(d, 'wardrobe', 3.5, 3.6, Math.PI)
    remix(d, 'fresh-spots')
    expect(twistOf(d)).toMatchObject({ done: false, progress: 0 })
    const [bed, wardrobe] = layoutOf(d).placements
    bed.y = 2.2
    expect(twistOf(d).progress).toBeCloseTo(0.5)
    wardrobe.x = 1.0
    expect(twistOf(d).done).toBe(true)
  })

  it('bed swap: the bed turned or on another wall', () => {
    const d = bedroomDoc()
    place(d, 'bed', 2, 2)
    remix(d, 'bed-swap')
    expect(twistOf(d).done).toBe(false)
    layoutOf(d).placements[0].rotation = Math.PI / 2
    expect(twistOf(d).done).toBe(true)
  })

  it('talk corner: two seats facing each other, close enough to talk', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 6, 5))
    remix(d, 'talk-corner')
    place(d, 'sofa', 3, 0.6, 0)
    const chair = place(d, 'armchair', 3, 2.6, Math.PI)
    expect(twistOf(d).done).toBe(true)
    chair.rotation = 0
    expect(twistOf(d).done).toBe(false)
    chair.rotation = Math.PI
    chair.y = 4.4
    expect(twistOf(d).done).toBe(false)
  })

  it('open floor: the middle of the living room stays clear, with 4 pieces in the room', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    remix(d, 'open-floor')
    place(d, 'sofa', 2.5, 3.5, Math.PI)
    place(d, 'armchair', 0.6, 0.6)
    place(d, 'plant', 4.6, 0.4)
    expect(twistOf(d).done).toBe(false)
    place(d, 'floor-lamp', 4.6, 3.6)
    expect(twistOf(d).done).toBe(true)
    place(d, 'coffee-table', 2.5, 2.2)
    expect(twistOf(d).done).toBe(false)
  })

  it('less is more: reads as a budget of 10 pieces', () => {
    const d = sampleDoc()
    remix(d, 'less-is-more')
    expect(twistOf(d).done).toBe(true)
    for (let i = 0; i < 9; i++) place(d, 'plant', 0.3 + i * 0.4, 0.3)
    const g = twistOf(d)
    expect(g.done).toBe(false)
    expect(g.hint).toMatch(/^1\d of 10 pieces/)
    expect(g.progress).toBeLessThan(1)
  })

  it('tells an untouched copy from its source', () => {
    const d = sampleDoc()
    const a = layoutOf(d)
    const b = { ...a, id: 'copy', placements: a.placements.map((p) => ({ ...p, id: `${p.id}c` })) }
    expect(sameArrangement(a, b)).toBe(true)
    b.placements[0] = { ...b.placements[0], x: b.placements[0].x + 0.3 }
    expect(sameArrangement(a, b)).toBe(false)
  })

  /** Furnishes d by the Next chip until Dream home: the number of placements it took, or -1. */
  function playToDream(d: Doc, tag: string) {
    const fits = (c: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe')).ok
    let last = ''
    let repeats = 0
    let best = -1
    for (let i = 0; i < 30; i++) {
      const s = score(d)
      if (s.level.index === LEVELS.length - 1) return i
      const next = nextGoal(s, fits)
      if (next?.id === 'unpack' && next.near) {
        Object.assign(d, unpackDoc(d, next.near)!.doc)
        continue
      }
      if (!next?.item || !next.tab) return -1
      const key = `${next.id}:${next.item}`
      repeats = key === last && s.stars <= best ? repeats + 1 : 1
      expect(repeats, `${key} offered ${repeats} times without progress`).toBeLessThanOrEqual(3)
      last = key
      best = Math.max(best, s.stars)
      const item = itemFromCatalog(catalogEntry(next.item), `${tag}n${i}`)
      const r = findDrop(d, layoutOf(d), item, { nextTo: next.near })
      if (!r.ok) return -1
      d.items.push(item)
      layoutOf(d).placements.push({ id: `${tag}p${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
    }
    return -1
  }

  it('a remix of the sample reaches Dream home by the Next chip, only with its twist done', () => {
    const d = sampleDoc()
    expect(playToDream(d, 'orig')).toBeGreaterThan(0)
    const l = remix(d, 'window-seat')
    // Both moving boxes are unpacked on the way (round 9), and that bonus belongs to the home, so
    // a remix can start one level up, at Cozy; it is still well short of Dream home.
    expect(score(d).level.index).toBeLessThanOrEqual(LEVELS.indexOf('Cozy'))
    const dream = playToDream(d, 'rx')
    console.log(`remix (window-seat) of the furnished sample: ${l.remix!.anchors} anchors, Dream home after ${dream} Next-chip placements`)
    expect(dream, `Dream home after ${dream} placements`).toBeGreaterThanOrEqual(6)
    expect(dream).toBeLessThanOrEqual(25)
    expect(twistOf(d).done).toBe(true)
    // The rebuild isn't a carbon copy: some of the smaller pieces land somewhere new.
    const src = d.layouts.find((x) => x.id === l.remix!.from)!
    const spot = (p: { itemId: string; x: number; y: number }) => `${cat(d, p as never)}:${p.x.toFixed(1)}:${p.y.toFixed(1)}`
    const before = new Set(src.placements.map(spot))
    const fresh = layoutOf(d).placements.filter((p) => !ANCHORS.has(cat(d, p)))
    expect(fresh.filter((p) => !before.has(spot(p))).length).toBeGreaterThanOrEqual(3)
    // Turn the sofa away from the window: the twist opens and Dream home is out of reach.
    const sofa = layoutOf(d).placements.find((p) => cat(d, p) === 'sofa')!
    sofa.rotation = 0
    expect(twistOf(d).done).toBe(false)
    expect(score(d).level.index).toBeLessThanOrEqual(WARNING_LEVEL_CAP)
  }, 60000)
})

describe('remix twists lead the day (round 10)', () => {
  let k = 0
  const ids = () => `tw${++k}`
  const cat = (d: Doc, p: Placement) => d.items.find((i) => i.id === p.itemId)!.category

  /** Follows the Next chip; a twist goal that is a move is done as the player would (the armchair to the talk spot). */
  function play(d: Doc, tag: string, log: string[]) {
    const fits = (c: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe')).ok
    for (let i = 0; i < 40; i++) {
      const s = score(d)
      if (s.level.index === LEVELS.length - 1) return i
      const next = nextGoal(s, fits)
      log.push(`${i} ★${s.stars} ${next?.id}:${next?.item ?? '(move)'}`)
      if (next?.id === 'unpack' && next.near) {
        Object.assign(d, unpackDoc(d, next.near)!.doc)
        continue
      }
      if (next?.id === 'twist:talk-corner' && !next.item) {
        const spot = talkCornerSpot(d, layoutOf(d))!
        const chair = layoutOf(d).placements.find((p) => p.id === next.near)!
        expect(cat(d, chair)).toBe('armchair')
        expect(next.hint).toMatch(/^(Turn|Move) the armchair/)
        Object.assign(chair, spot)
        continue
      }
      if (!next?.item || !next.tab) return -1
      const item = itemFromCatalog(catalogEntry(next.item), `${tag}n${i}`)
      const r = findDrop(d, layoutOf(d), item, { nextTo: next.near })
      if (!r.ok) return -1
      d.items.push(item)
      layoutOf(d).placements.push({ id: `${tag}p${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
    }
    return -1
  }

  it('a talk-corner remix of the sample leads with its twist and still reaches Dream home', () => {
    const d = sampleDoc()
    const log: string[] = []
    expect(play(d, 'o', log), log.join('\n')).toBeGreaterThan(0)
    expect(applicableTwists(d)).toContain('talk-corner')
    const l = remixedLayout(d, d.activeLayoutId, 'talk-corner', ids)!
    d.layouts.push(l)
    d.activeLayoutId = l.id
    const rx: string[] = []
    const dream = play(d, 'r', rx)
    expect(dream, rx.join('\n')).toBeGreaterThan(0)
    // The twist comes first after unpacking, not at the end of a replay.
    const first = rx.findIndex((x) => x.includes('twist:'))
    expect(first, rx.join('\n')).toBeGreaterThanOrEqual(0)
    expect(first, rx.join('\n')).toBeLessThanOrEqual(2)
    expect(score(d).goals.find((g) => g.id === 'twist:talk-corner')!.done).toBe(true)
    expect(score(d).goals.find((g) => g.id === 'nook')!.done).toBe(true)
  }, 60000)

  it('Dream home seats two at the dining table', () => {
    const d = sampleDoc()
    const log: string[] = []
    play(d, 'e', log)
    const g = score(d).goals.find((x) => x.id === 'eat-two')!
    expect(g.done, log.join('\n')).toBe(true)
    expect(log.some((x) => x.includes('eat-two:dining-chair')), log.join('\n')).toBe(true)
  }, 60000)

  it('offers twists not yet finished first, turned by the day, never one being played', () => {
    const d = sampleDoc()
    const all: TwistId[] = ['talk-corner', 'open-floor', 'window-seat', 'fresh-spots', 'bed-swap', 'less-is-more']
    expect(pickTwists(d, all, 0)).toEqual(['talk-corner', 'open-floor', 'window-seat'])
    expect(pickTwists(d, all, 1)).toEqual(['open-floor', 'window-seat', 'fresh-spots'])
    const done = { ...remixedLayout(d, d.activeLayoutId, 'open-floor', ids)!, remix: { from: 'x', twist: 'open-floor' as TwistId, anchors: 1, dreamAt: 1 } }
    const playing = remixedLayout(d, d.activeLayoutId, 'window-seat', ids)!
    d.layouts.push(done, playing)
    expect(finishedTwists(d)).toEqual(new Set(['open-floor']))
    expect(pickTwists(d, all, 0)).toEqual(['talk-corner', 'fresh-spots', 'bed-swap'])
    expect(pickTwists(d, all, 0, 6)).toEqual(['talk-corner', 'fresh-spots', 'bed-swap', 'less-is-more', 'open-floor'])
  })
})

describe('round 11: dead ends and blockers', () => {
  it('"Make room for a desk" never asks to move the bed out of the bedroom', () => {
    // A queen bed alone in the bedroom, a crowded living room with a dining table and two armchairs.
    const d = emptyDoc()
    d.rooms.push(rectRoom('Bedroom', 0, 0, 3.4, 3.4))
    d.rooms.push(rectRoom('Living room', 3.4, 0, 4, 3.4))
    place(d, 'bed', 1.7, 1.1, 0, { width: 1.6, depth: 2.0 })
    place(d, 'sofa', 5.4, 0.5)
    place(d, 'dining-table', 5.4, 2.4, 0, { width: 1.2, depth: 0.8 })
    place(d, 'armchair', 4.0, 1.6, Math.PI / 2)
    place(d, 'armchair', 6.9, 1.6, -Math.PI / 2)
    const work = score(d).goals.find((g) => g.id === 'work')!
    expect(work.item).toBe('desk')
    const s = score(d)
    const bed = d.items.find((i) => i.category === 'bed')!
    const bedP = layoutOf(d).placements.find((p) => p.itemId === bed.id)!
    expect(work.makeRoom).toBeDefined()
    expect(work.makeRoom!.id).not.toBe(bedP.id)
    // A spare armchair goes first: there are two of them.
    const moved = layoutOf(d).placements.find((p) => p.id === work.makeRoom!.id)!
    expect(d.items.find((i) => i.id === moved.itemId)!.category).toBe('armchair')
    // Next's "Make room" card names that piece, not the bed.
    const g = nextGoal(s, (c) => c !== 'desk')
    if (g?.title.startsWith('Make room')) expect(g.hint).not.toMatch(/bed/)
  })

  it('names the open core goal (not "the snag") when the stars are there but a goal is open', () => {
    const fake = (goals: Partial<Goal>[], errors = 0) =>
      ({ errors, level: { index: 4, name: 'Designer home', next: 'Dream home', toNext: 0 }, goals: goals as Goal[] }) as unknown as Parameters<typeof levelBlocker>[0]
    expect(levelBlocker(fake([{ id: 'sit', title: 'Seats for four', done: false, bonus: false, stars: 2 } as Goal]))).toBe('Finish \u201cSeats for four\u201d to reach Dream home')
    expect(levelBlocker(fake([{ id: 'overlap', title: 'Nothing overlaps', done: false, snag: true, stars: 2 } as Goal]))).toBe('Fix the snag to reach Dream home')
    expect(levelBlocker(fake([], 1))).toBe('Fix the snag to reach Dream home')
  })
})
