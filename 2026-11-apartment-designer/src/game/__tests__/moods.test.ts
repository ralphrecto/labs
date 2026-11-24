import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { sampleDoc } from '../../model/sample'
import type { Doc } from '../../model/types'
import { LEVELS, goalMove, nextGoal, scoreHome } from '../goals'
import { pickSpot } from '../BuyDrawer'
import { defaultDrop, unpackDoc, useStore } from '../../store'
import { findDrop } from '../../model/placement'
import { pointInPolygon } from '../../model/geometry'
import { dueMoment, inTone, latestMood, momentKey, moodDoneTitle, moodGoalId, moodGoals, moodOf, moodRepaint, MOODS, MOOD_IDS, pickMood, piecesOf, repaintDoc, toneNeeded, withMood, moodLightSpot, changeMood, innerPoint } from '../moods'
import { MOOD_TONES, PICKS, picksFor, WOODEN } from '../swatches'
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
    // Bold asks for a rug in a bold colour: the oak one doesn't count until it's painted, and the
    // goal asks for that paint (the rug, selected and painted by the chip), not a second rug.
    setMood(d, living.id, 'bold')
    place(d, 'wall-art', 2.0, 4.18)
    const bold = () => score(d).goals.find((x) => x.id === g.id)!
    const oakRug = layoutOf(d).placements.find((p) => d.items.find((i) => i.id === p.itemId)?.category === 'rug')!
    expect(bold()).toMatchObject({ done: false, near: oakRug.id })
    expect(bold().item).toBeUndefined()
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
        // Wooden furniture (round 17 review) leads in the mood's wood, never its accent colours.
        if (WOODEN.has(category)) {
          expect(picks[0].color, `${id} ${category}`).toBe(MOOD_TONES[id].woods[0])
          for (const p of picks) expect(['#b5534a', '#e0b04f', '#5b7bb5', '#e8a27c'], `${id} ${category}`).not.toContain(p.color)
        } else if (!list.every((q) => q.id === list[0].id) || list.some((q) => MOOD_TONES[id].palette.includes(q.color))) expect(MOOD_TONES[id].palette, `${id} ${category}`).toContain(picks[0].color)
        // Colour-only variants (Linen, Honey lamps) keep their colours, so their names stay true.
        for (const p of picks) if (list.filter((q) => q.id === p.id).length > 1) expect(p.color).toBe(list.find((q) => q.label === p.label)!.color)
        expect(new Set(picks.filter((p) => list.filter((q) => q.id === p.id).length > 1).map((p) => p.color)).size).toBe(picks.filter((p) => list.filter((q) => q.id === p.id).length > 1).length)
      }
    }
    const snug = picksFor('coffee-table', {}, 'snug')
    expect(MOOD_TONES.snug.woods).toContain(snug[0].color)
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

/**
 * Follows the Next chip from the moment a room is given a mood, as a tap would (round 17 review):
 * the drawer's first pick card (in the room's mood colours) goes where the drawer puts it, and a
 * goal a pick didn't bring closer steps aside (Goals.tsx's skip). Returns the step the room's
 * mood goal was done at, or -1, and the log.
 */
function chipToMood(d: Doc, roomId: string, tag: string, max = 45) {
  const log: string[] = []
  const skip = new Set<string>()
  const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
  for (let i = 0; i < max; i++) {
    const s = score(d)
    if (s.goals.find((g) => g.id === moodGoalId(roomId))?.done) return { at: i, log }
    const next = nextGoal(s, fits, skip)
    if (!next) break
    const before = next.progress ?? 0
    log.push(`${i} ★${s.stars} ${s.level.name} ${next.id}:${next.item ?? '(move)'}`)
    const moved = () => {
      const g = score(d).goals.find((x) => x.id === next.id)
      return !g || g.done || (g.progress ?? 0) > before + 1e-6
    }
    if (next.id === 'unpack' && next.near) {
      Object.assign(d, unpackDoc(d, next.near)!.doc)
      continue
    }
    // A mood goal met by painting a piece the room has (Goals.showGoal paints it).
    const paint = moodRepaint(d, layoutOf(d), next.id)
    if (paint) {
      Object.assign(d, repaintDoc(d, paint.placementId, paint.color, `${tag}c${i}`))
      log.push('   (painted)')
      if (!moved()) skip.add(next.id)
      continue
    }
    if (!next.item || !next.tab) {
      const m = goalMove(d, layoutOf(d), next.id)
      if (m) for (const q of [m, ...(m.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
      if (!m || !moved()) skip.add(next.id)
      continue
    }
    // The first pick card, as the drawer draws it for this goal's room.
    const item = itemFromCatalog(catalogEntry(next.item), `${tag}n${i}`)
    const tone = toneNeeded(d, layoutOf(d), next.id, next.item)
    const pick = picksFor(next.item, {}, pickMood(layoutOf(d), next.placeIn)).find((p) => !tone || inTone(tone, p.color))
    if (pick) item.color = pick.color
    const at = pickSpot(d, item, { category: next.item, roomId: next.placeIn, near: next.near, goalId: next.id })
    const r = at.fail ? null : at.spot ? { ok: true as const, spot: at.spot } : defaultDrop(d, item)
    if (!r?.ok) {
      skip.add(next.id)
      log.push('   (refused)')
      continue
    }
    d.items.push(item)
    layoutOf(d).placements.push({ id: `${tag}p${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
    if (!moved()) {
      skip.add(next.id)
      log.push('   (no closer: skipped)')
    }
  }
  return { at: -1, log }
}

describe('Mood goals by the Next chip (round 17 review)', () => {
  for (const name of ['Living room', 'Bedroom'])
    for (const id of MOOD_IDS)
      it(`${MOODS[id].short} ${name.toLowerCase()}: the chip leads to its mood goal and finishes it`, () => {
        for (const settled of [false, true]) {
          const d = sampleDoc(settled ? { settled: true } : undefined)
          const r = room(d, name)
          setMood(d, r.id, id)
          const res = chipToMood(d, r.id, `${id}${settled ? 's' : 'e'}`)
          const text = res.log.join('\n')
          expect(res.at, `${settled ? 'settled' : 'empty'}\n${text}`).toBeGreaterThan(0)
          // The chip offered the mood goal itself on the way (not only a goal that happened to finish it).
          if (MOODS[id].goal.needs.some((n) => n.tone)) expect(res.log.some((x) => x.includes(` ${moodGoalId(r.id)}:`)), text).toBe(true)
        }
      }, 120000)

  it('a Bold rug must wear a bold colour: the drawer leads with one, and an oak rug of yours is passed over', () => {
    // A bedroom with no rug yet: the goal asks for one, in a bold colour.
    const d = sampleDoc({ settled: true })
    const living = room(d, 'Bedroom')
    setMood(d, living.id, 'bold')
    const c = moodLightSpot(living).at
    place(d, 'wall-art', c.x, c.y)
    const g = score(d).goals.find((x) => x.id === moodGoalId(living.id))!
    expect(g, JSON.stringify(g)).toMatchObject({ item: 'rug', placeIn: living.id, done: false })
    expect(toneNeeded(d, layoutOf(d), g.id, 'rug')).toBe('bold')
    expect(toneNeeded(d, layoutOf(d), g.id, 'plant')).toBeUndefined()
    expect(toneNeeded(d, layoutOf(d), 'plant', 'rug')).toBeUndefined()
    const oak = d.items.find((i) => i.category === 'rug')!.color
    expect(inTone('bold', oak)).toBe(false)
    // Every rug card in a Bold room counts.
    for (const p of picksFor('rug', {}, 'bold')) expect(inTone('bold', p.color), p.label).toBe(true)
    // A furnished room's oak rug: painted (a copy of the item when another design shares it), done.
    const e = sampleDoc({ settled: true })
    const lr = room(e, 'Living room')
    setMood(e, lr.id, 'bold')
    place(e, 'wall-art', 2.0, 4.18)
    e.layouts.push({ ...layoutOf(e), id: 'other', placements: layoutOf(e).placements.map((p) => ({ ...p })) })
    const paint = moodRepaint(e, layoutOf(e), moodGoalId(lr.id))!
    expect(paint.color).toBe('#5b7bb5')
    const e2 = repaintDoc(e, paint.placementId, paint.color, 'painted')
    expect(score(e2).goals.find((x) => x.id === moodGoalId(lr.id))!.done).toBe(true)
    const otherRug = e2.layouts.find((l) => l.id === 'other')!.placements.find((p) => p.id === paint.placementId)!
    expect(e2.items.find((i) => i.id === otherRug.itemId)!.color).not.toBe('#5b7bb5')
    expect(moodRepaint(e2, layoutOf(e2), moodGoalId(lr.id))).toBeNull()
    // Snug asks for any rug: no tone needed.
    setMood(d, living.id, 'snug')
    expect(toneNeeded(d, layoutOf(d), g.id, 'rug')).toBeUndefined()
  })

  it('a finished mood goal has its own headline', () => {
    const d = sampleDoc({ settled: true })
    const living = room(d, 'Living room')
    setMood(d, living.id, 'snug')
    place(d, 'floor-lamp', 0.4, 3.8)
    const g = score(d).goals.find((x) => x.id === moodGoalId(living.id))!
    expect(g.done).toBe(true)
    expect(moodDoneTitle(d, layoutOf(d), g)).toBe('🕯️ Your living room feels snug')
    expect(moodDoneTitle(d, layoutOf(d), score(d).goals.find((x) => !x.perk)!)).toBeUndefined()
  })

  // Round 18: changing a room's mood cost the finished mood's star.
  const change = (d: Doc, roomId: string, m: 'airy' | 'snug' | 'bold') => {
    const scored = !!score(d).goals.find((x) => x.id === moodGoalId(roomId))?.done
    d.layouts = d.layouts.map((l) => (l.id === d.activeLayoutId ? changeMood(d, l, roomId, m, scored) : l))
  }
  it('a finished mood keeps its star when the room changes mood', () => {
    const d = sampleDoc({ settled: true })
    const living = room(d, 'Living room')
    change(d, living.id, 'snug')
    place(d, 'floor-lamp', 0.4, 3.8)
    expect(score(d).goals.find((x) => x.id === moodGoalId(living.id))?.done).toBe(true)
    const finished = score(d).stars
    // Airy asks for a mirror and a plant the room may not have: the star stays all the same.
    change(d, living.id, 'airy')
    expect(layoutOf(d).moods?.[living.id]).toBe('airy')
    expect(layoutOf(d).moodsDone?.[living.id]).toEqual(['snug'])
    const g = score(d).goals.find((x) => x.id === moodGoalId(living.id))!
    expect(g).toMatchObject({ done: true, stars: 1 })
    expect(g.hint).toMatch(/keeps its ★/)
    expect(score(d).stars).toBe(finished)
    // Taking the snug lamp away later doesn't cost it either: the finished mood is remembered.
    layoutOf(d).placements = layoutOf(d).placements.filter((p) => d.items.find((i) => i.id === p.itemId)?.category !== 'floor-lamp')
    expect(score(d).goals.find((x) => x.id === moodGoalId(living.id))?.done).toBe(true)
  })

  it("switching moods back and forth can't add stars (one per room)", () => {
    const d = sampleDoc({ settled: true })
    const living = room(d, 'Living room')
    change(d, living.id, 'snug')
    place(d, 'floor-lamp', 0.4, 3.8)
    const finished = score(d).stars
    for (const m of ['airy', 'snug', 'bold', 'snug', 'airy', 'bold'] as const) {
      change(d, living.id, m)
      expect(score(d).stars, m).toBe(finished)
      expect(score(d).goals.filter((x) => x.roomId === living.id && x.perk === 'mood')).toHaveLength(1)
    }
    // Finishing a second mood too (Bold: art and a blue rug) still counts the room once.
    place(d, 'wall-art', 2.0, 4.18)
    d.items.find((i) => i.category === 'rug')!.color = MOOD_TONES.bold.palette[2]
    expect(score(d).goals.find((x) => x.id === moodGoalId(living.id))?.done).toBe(true)
    const both = score(d).stars
    change(d, living.id, 'airy')
    expect(score(d).stars).toBe(both)
    change(d, living.id, 'bold')
    expect(score(d).stars).toBe(both)
    const roomMood = score(d).goals.filter((x) => x.roomId === living.id && x.perk === 'mood')
    expect(roomMood.reduce((a, g) => a + (g.done ? g.stars : 0), 0)).toBe(1)
    expect(new Set(layoutOf(d).moodsDone?.[living.id])).toEqual(new Set(['snug', 'bold']))
    // An unfinished mood isn't remembered as finished.
    const e = sampleDoc({ settled: true })
    const bed = room(e, 'Bedroom')
    change(e, bed.id, 'airy')
    change(e, bed.id, 'snug')
    expect(layoutOf(e).moodsDone).toBeUndefined()
  })

  it("a room's mood light stays inside the room: its pool never reaches a wall", () => {
    const d = sampleDoc({ settled: true })
    for (const r of d.rooms) {
      const { at, inner, height } = moodLightSpot(r)
      expect(inner).toBeGreaterThan(0)
      // The cone's footprint (radius at most `inner`) fits inside the room.
      for (let k = 0; k < 16; k++) {
        const a = (k / 16) * Math.PI * 2
        const p = { x: at.x + Math.cos(a) * inner * 0.99, y: at.y + Math.sin(a) * inner * 0.99 }
        expect(pointInPolygon(p, r.polygon), r.name).toBe(true)
      }
      expect(height).toBeLessThan(r.ceilingHeight)
    }
    // An L-shaped room: the light hangs in its widest part, not at the centroid (which falls outside).
    const L = [{ x: 0, y: 0 }, { x: 6, y: 0 }, { x: 6, y: 1.2 }, { x: 1.2, y: 1.2 }, { x: 1.2, y: 6 }, { x: 0, y: 6 }]
    const ip = innerPoint(L)
    expect(pointInPolygon(ip.at, L)).toBe(true)
    expect(ip.inner).toBeGreaterThan(0.5)
  })

  it('a copy of a design keeps its room moods', () => {
    const d = sampleDoc({ settled: true })
    setMood(d, room(d, 'Living room').id, 'airy')
    useStore.getState().replaceDoc(d)
    useStore.getState().duplicateLayout(d.activeLayoutId)
    const doc = useStore.getState().doc
    expect(doc.activeLayoutId).not.toBe(d.activeLayoutId)
    expect(layoutOf(doc).moods).toEqual({ [room(d, 'Living room').id]: 'airy' })
    // A remix keeps them too, so its mood moments only ask about rooms still without one.
    useStore.getState().remixLayout(d.activeLayoutId, 'talk-corner')
    const rx = useStore.getState().doc
    expect(layoutOf(rx).remix).toBeTruthy()
    expect(layoutOf(rx).moods).toEqual({ [room(d, 'Living room').id]: 'airy' })
  })

  it('at Dream home the chip offers an open mood goal ahead of other bonus goals (and Remix)', () => {
    const d = sampleDoc()
    const r = followWithMoods(d, 'top')
    expect(r.dream, r.log.join('\n')).toBeGreaterThan(0)
    const living = room(d, 'Living room')
    setMood(d, living.id, 'bold')
    const s = score(d)
    expect(s.level.next).toBeUndefined()
    const g = s.goals.find((x) => x.id === moodGoalId(living.id))!
    if (!g.done) {
      const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
      expect(nextGoal(s, fits)?.id).toBe(g.id)
    }
  }, 120000)
})
