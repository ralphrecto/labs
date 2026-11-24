import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { sampleDoc } from '../../model/sample'
import type { Doc } from '../../model/types'
import { LEVELS, goalMove, nextGoal, scoreHome } from '../goals'
import { pickSpot } from '../BuyDrawer'
import { defaultDrop, unpackDoc } from '../../store'
import { findDrop } from '../../model/placement'
import { dueMoment, latestMood, momentKey, moodGoalId, moodGoals, moodOf, MOODS, MOOD_IDS, pickMood, piecesOf, withMood } from '../moods'
import { MOOD_TONES, PICKS, picksFor } from '../swatches'
import { STARTER_CATALOG } from '../../model/starterCatalog'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
const room = (d: Doc, name: string) => d.rooms.find((r) => r.name === name)!
const setMood = (d: Doc, roomId: string, m: 'airy' | 'snug' | 'bold' | undefined) => {
  d.layouts = d.layouts.map((l) => (l.id === d.activeLayoutId ? withMood(l, roomId, m) : l))
}
let k = 0
function place(d: Doc, category: string, x: number, y: number) {
  const item = itemFromCatalog(catalogEntry(category), `m${++k}`)
  d.items.push(item)
  layoutOf(d).placements.push({ id: `mp${k}`, itemId: item.id, x, y, rotation: 0 })
}

describe('Mood moments', () => {
  it('each mood asks for pieces the starter catalog has', () => {
    const cats = new Set(STARTER_CATALOG.map((p) => p.category))
    for (const id of MOOD_IDS) for (const n of MOODS[id].goal.needs) for (const c of n.cats) expect(cats.has(c) || c === 'rug', `${id}: ${c}`).toBe(true)
  })

  it('a moment is due from Lived-in, one room per level, living room first', () => {
    const d = sampleDoc({ settled: true })
    const l = layoutOf(d)
    // Move-in day (a few pieces) asks nothing yet, whatever the level.
    expect(dueMoment(d, l, 2, piecesOf(d, l).slice(0, 4), new Set())).toBeNull()
    const pieces = [...piecesOf(d, l), ...Array.from({ length: 8 }, () => ({ category: 'plant', room: room(d, 'Living room') }))]
    expect(dueMoment(d, l, 1, pieces, new Set())).toBeNull()
    expect(dueMoment(d, l, 2, pieces, new Set())?.name).toBe('Living room')
    setMood(d, room(d, 'Living room').id, 'snug')
    expect(dueMoment(d, layoutOf(d), 2, pieces, new Set())).toBeNull()
    // Cozy asks about the next room with furniture (the bathroom never).
    expect(dueMoment(d, layoutOf(d), 3, pieces, new Set())?.name).toBe('Bedroom')
    // "Not now" on the bedroom: not asked again on this design, nor is the bathroom.
    const skip = new Set([momentKey(d, layoutOf(d), room(d, 'Bedroom').id)])
    expect(dueMoment(d, layoutOf(d), 5, pieces, skip)).toBeNull()
  })

  it('a room mood adds a one-star bonus goal that never holds a level', () => {
    const d = sampleDoc({ settled: true })
    const before = score(d)
    const living = room(d, 'Living room')
    setMood(d, living.id, 'snug')
    const after = score(d)
    const g = after.goals.find((x) => x.id === moodGoalId(living.id))!
    // The settled sample's living room has a rug already: half way there, a lamp to go.
    expect(g).toMatchObject({ stars: 1, bonus: true, perk: 'mood', done: false, placeIn: living.id, item: 'floor-lamp', tab: 'decor', progress: 0.5 })
    expect(after.level.index).toBe(before.level.index)
    expect(after.cap).toBe(before.cap)
    expect(after.coreMax).toBe(before.coreMax)
    expect(after.thresholds).toEqual(before.thresholds)
    // A lamp in the living room finishes it: one more star.
    place(d, 'floor-lamp', 0.4, 3.8)
    const done = score(d)
    expect(done.goals.find((x) => x.id === g.id)?.done).toBe(true)
    expect(done.stars).toBeGreaterThan(after.stars)
    // Bold asks for a rug in a bold colour: the oak one doesn't count until it's painted.
    setMood(d, living.id, 'bold')
    place(d, 'wall-art', 2.0, 4.18)
    const bold = () => score(d).goals.find((x) => x.id === g.id)!
    expect(bold()).toMatchObject({ done: false, item: 'rug' })
    const rug = d.items.find((i) => i.category === 'rug')!
    rug.color = MOOD_TONES.bold.palette[2]
    expect(bold().done).toBe(true)
    // A piece in another room doesn't count for this one.
    const e = sampleDoc({ settled: true })
    setMood(e, room(e, 'Living room').id, 'airy')
    place(e, 'plant', 7.5, 2.8)
    expect(moodGoals(e, layoutOf(e), piecesOf(e, layoutOf(e))).find((x) => x.roomId === room(e, 'Living room').id)?.progress).toBe(0)
  })

  it('remembers moods per room on the design; the latest colours picks with no room', () => {
    const d = sampleDoc({ settled: true })
    const living = room(d, 'Living room').id
    const bed = room(d, 'Bedroom').id
    setMood(d, living, 'snug')
    setMood(d, bed, 'airy')
    const l = layoutOf(d)
    expect(moodOf(l, living)).toBe('snug')
    expect(latestMood(l)).toBe('airy')
    expect(pickMood(l, living)).toBe('snug')
    expect(pickMood(l)).toBe('airy')
    setMood(d, living, 'bold')
    expect(latestMood(layoutOf(d))).toBe('bold')
    setMood(d, living, undefined)
    setMood(d, bed, undefined)
    expect(layoutOf(d).moods).toBeUndefined()
  })

  it('a mood draws the picks in its colours and leads with one that suits it', () => {
    const plain = picksFor('coffee-table', {}, undefined)
    expect(plain.map((p) => p.color)).toEqual(PICKS['coffee-table'].map((p) => p.color))
    for (const id of MOOD_IDS) {
      for (const [category, list] of Object.entries(PICKS)) {
        const picks = picksFor(category, {}, id)
        // Same pieces, same labels, just recoloured and reordered.
        expect(picks.map((p) => `${p.id}/${p.label}`).sort(), `${id} ${category}`).toEqual(list.map((p) => `${p.id}/${p.label}`).sort())
        // A mood never changes the lead piece's shape or size (what fits where), only its colour.
        if (!list.every((q) => q.id === list[0].id)) expect(picks.map((p) => p.label), `${id} ${category}`).toEqual(list.map((p) => p.label))
        // The lead card wears the mood's colours (colour-only variants: when one of them does).
        if (!list.every((q) => q.id === list[0].id) || list.some((q) => MOOD_TONES[id].palette.includes(q.color))) expect(MOOD_TONES[id].palette, `${id} ${category}`).toContain(picks[0].color)
        // Colour-only variants (Linen, Honey lamps) keep their colours, so their names stay true.
        for (const p of picks) if (list.filter((q) => q.id === p.id).length > 1) expect(p.color).toBe(list.find((q) => q.label === p.label)!.color)
        expect(new Set(picks.filter((p) => list.filter((q) => q.id === p.id).length > 1).map((p) => p.color)).size).toBe(picks.filter((p) => list.filter((q) => q.id === p.id).length > 1).length)
      }
    }
    const snug = picksFor('coffee-table', {}, 'snug')
    expect(MOOD_TONES.snug.palette).toContain(snug[0].color)
    expect(snug[0]).toMatchObject({ mood: true, taste: true })
    const airy = picksFor('sofa', {}, 'airy')
    expect(MOOD_TONES.airy.palette).toContain(airy[0].color)
    // Bold pendants lead with Brass or Black, never Linen.
    expect(picksFor('pendant', {}, 'bold')[0].label).not.toBe('Linen')
  })
})

/**
 * Follows the Next chip as nextLoop.test does, answering each mood moment as it comes due (the
 * first mood for the first room, and so on). Returns the steps to Dream home and the log.
 */
function followWithMoods(d: Doc, tag: string, max = 45) {
  const log: string[] = []
  const skip = new Set<string>()
  let best = 0
  let stalls = 0
  let last = ''
  for (let i = 0; i < max; i++) {
    const s = score(d)
    best = Math.max(best, s.errors ? 0 : s.level.index)
    const due = dueMoment(d, layoutOf(d), best, piecesOf(d, layoutOf(d)), new Set())
    if (due) {
      const m = MOOD_IDS[Object.keys(layoutOf(d).moods ?? {}).length % MOOD_IDS.length]
      setMood(d, due.id, m)
      log.push(`   mood ${due.name}: ${m}`)
      continue
    }
    if (s.level.index === LEVELS.length - 1) return { dream: i, log, stalls, s }
    const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
    const next = nextGoal(s, fits, skip)
    if (!next || skip.has(next.id)) break
    const before = next.progress ?? 0
    const line = `${i} ★${s.stars} ${s.level.name} ${next.id}:${next.item ?? '(move)'}`
    const moved = () => {
      const g = score(d).goals.find((x) => x.id === next.id)
      return !g || g.done || (g.progress ?? 0) > before + 1e-6
    }
    if (last === next.id) stalls++
    last = next.id
    if (next.id === 'unpack' && next.near) {
      Object.assign(d, unpackDoc(d, next.near)!.doc)
      log.push(line)
      continue
    }
    if (!next.item || !next.tab) {
      const m = goalMove(d, layoutOf(d), next.id)
      if (m) Object.assign(layoutOf(d).placements.find((p) => p.id === m.id)!, { x: m.x, y: m.y, rotation: m.rotation })
      if (!m || !moved()) skip.add(next.id)
      log.push(`${line} move`)
      continue
    }
    const item = itemFromCatalog(catalogEntry(next.item), `${tag}n${i}`)
    const at = pickSpot(d, item, { category: next.item, roomId: next.placeIn, near: next.near, goalId: next.id })
    const r = at.fail ? null : at.spot ? { ok: true as const, spot: at.spot } : defaultDrop(d, item)
    if (!r?.ok) {
      skip.add(next.id)
      log.push(`${line} (refused)`)
      continue
    }
    d.items.push(item)
    layoutOf(d).placements.push({ id: `${tag}p${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
    if (!moved()) skip.add(next.id)
    log.push(`${line} -> ★${score(d).stars}`)
  }
  return { dream: -1, log, stalls, s: score(d) }
}

describe('Mood moments in play', () => {
  it('the sample home with every moment answered: Dream home in about the usual steps, mood goals on the way', () => {
    const d = sampleDoc()
    const r = followWithMoods(d, 'mm')
    const text = r.log.join('\n')
    expect(r.dream, text).toBeGreaterThan(0)
    expect(r.dream, text).toBeLessThanOrEqual(30)
    expect(Object.keys(layoutOf(d).moods ?? {}).length, text).toBeGreaterThanOrEqual(2)
    // The chip leads to at least one mood goal before Dream home, and finishes it.
    expect(r.log.some((x) => / mood:/.test(x)), text).toBe(true)
    expect(r.s.goals.filter((g) => g.perk === 'mood' && g.done).length, text).toBeGreaterThanOrEqual(1)
  }, 120000)
})
