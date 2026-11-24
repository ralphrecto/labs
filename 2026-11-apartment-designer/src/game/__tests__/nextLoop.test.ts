import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { sampleDoc } from '../../model/sample'
import type { Doc, TwistId } from '../../model/types'
import { goalMove, LEVELS, nextGoal, remixedLayout, scoreHome, twistMove } from '../goals'
import { pickSpot } from '../BuyDrawer'
import { defaultDrop, unpackDoc } from '../../store'
import { findDrop } from '../../model/placement'

/** Set to log each step's goal, hint and the goals it finished or undid. */
const DEBUG = false
const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
let k = 0
const ids = () => `lp${++k}`

/**
 * Follows the Next chip as the HUD does (round 16): a twist move is the chip's own move, a piece
 * goes where the drawer puts it (pickSpot), and a goal a pick (or a move the player skips) didn't
 * bring closer steps aside, as VibeMeter's `skipped` set does. Returns each step's goal and whether
 * that step moved it on.
 */
function follow(d: Doc, tag: string, max = 40) {
  const log: string[] = []
  const steps: { id: string; moved: boolean }[] = []
  const skip = new Set<string>()
  for (let i = 0; i < max; i++) {
    const s = score(d)
    if (s.level.index === LEVELS.length - 1) return { dream: i, log, steps }
    const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
    const next = nextGoal(s, fits, skip)
    // Only goals already passed over are left: the chip offers them again (the player's call now).
    if (!next || skip.has(next.id)) break
    const before = next.progress ?? 0
    const line = `${i} ★${s.stars} ${s.level.name} ${next.id}:${next.item ?? '(move)'}${DEBUG ? ` near=${next.near} hint=${next.hint}` : ''}`
    const moved = () => {
      const g = score(d).goals.find((x) => x.id === next.id)
      return !g || g.done || (g.progress ?? 0) > before + 1e-6
    }
    if (next.id === 'unpack' && next.near) {
      Object.assign(d, unpackDoc(d, next.near)!.doc)
      steps.push({ id: next.id, moved: true })
      log.push(line)
      continue
    }
    if (!next.item || !next.tab) {
      const m = next.id.startsWith('twist:') ? twistMove(d, layoutOf(d)) : goalMove(d, layoutOf(d), next.id)
      // (A move that makes room for a piece, round 17, moves its goal on too: the piece fits now.)
      if (m) for (const q of [m, ...(m.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
      const ok = !!m && (moved() || m.done)
      if (!ok) skip.add(next.id)
      steps.push({ id: next.id, moved: ok })
      log.push(`${line}${m ? ` moved ${m.id}` : ' (skipped)'} -> ★${score(d).stars}`)
      continue
    }
    const item = itemFromCatalog(catalogEntry(next.item), `${tag}n${i}`)
    const at = pickSpot(d, item, { category: next.item, roomId: next.placeIn, near: next.near, goalId: next.id })
    const r = at.fail ? null : at.spot ? { ok: true as const, spot: at.spot } : defaultDrop(d, item)
    if (!r?.ok) {
      // The store refuses: the HUD stops suggesting that piece until the home changes.
      skip.add(next.id)
      steps.push({ id: next.id, moved: false })
      log.push(`${line} (refused)`)
      continue
    }
    d.items.push(item)
    layoutOf(d).placements.push({ id: `${tag}p${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
    const ok = moved()
    if (!ok) skip.add(next.id)
    steps.push({ id: next.id, moved: ok })
    const after = score(d)
    const diff = DEBUG ? ` [${after.goals.filter((g) => g.done !== !!s.goals.find((x) => x.id === g.id)?.done).map((g) => `${g.done ? '+' : '-'}${g.id}`).join(' ')}]` : ''
    log.push(`${line} -> ★${after.stars}${diff}`)
  }
  return { dream: -1, log, steps }
}

/** The longest run of one goal offered again after a step that didn't move it on. */
function longestStall(steps: { id: string; moved: boolean }[]) {
  let run = 1
  let most = steps.length ? 1 : 0
  for (let i = 1; i < steps.length; i++) {
    run = steps[i].id === steps[i - 1].id && !steps[i - 1].moved ? run + 1 : 1
    most = Math.max(most, run)
  }
  return most
}

describe('Next never loops on one goal (round 16)', () => {
  it('the sample home: no goal offered more than twice in a row without progress', () => {
    const d = sampleDoc()
    const r = follow(d, 'nl')
    expect(r.dream, r.log.join('\n')).toBeGreaterThan(0)
    expect(longestStall(r.steps), r.log.join('\n')).toBeLessThanOrEqual(2)
  }, 120000)

  for (const tw of ['fresh-spots', 'open-floor'] as TwistId[])
    it(`a ${tw} remix of the sample: no goal offered more than twice in a row without progress`, () => {
      const d = sampleDoc()
      expect(follow(d, `o${tw}`).dream).toBeGreaterThan(0)
      const l = remixedLayout(d, d.activeLayoutId, tw, ids)!
      d.layouts.push(l)
      d.activeLayoutId = l.id
      const r = follow(d, `r${tw}`)
      console.log(`${tw} remix: Dream home after ${r.dream} Next steps\n${r.log.join('\n')}`)
      expect(longestStall(r.steps), r.log.join('\n')).toBeLessThanOrEqual(2)
      // Every pick the chip makes keeps the stars it had.
      const lost = r.log.filter((x) => {
        const m = /★(\d+) .* -> ★(\d+)/.exec(x)
        return m && Number(m[2]) < Number(m[1])
      })
      expect(lost, r.log.join('\n')).toEqual([])
    }, 180000)
})

describe('Clear walkways is a move the chip can make (round 16)', () => {
  it('points at the piece by the pinch, and moves it so the way is clear at no cost in stars', () => {
    const d = sampleDoc({ settled: true })
    const shelf = itemFromCatalog(catalogEntry('bookshelf'), 'wk-shelf')
    d.items.push(shelf)
    layoutOf(d).placements.push({ id: 'wk-shelf-p', itemId: shelf.id, x: 3.5, y: 3.0, rotation: 0 })
    const s = score(d)
    const next = nextGoal(s, () => true)!
    expect(next.id).toBe('walkways')
    expect(next.tab).toBeUndefined()
    expect(layoutOf(d).placements.some((p) => p.id === next.near), next.hint).toBe(true)
    expect(next.hint).toMatch(/ by the /)
    const m = goalMove(d, layoutOf(d), 'walkways')!
    expect(m).not.toBeNull()
    expect(m.id).toBe(next.near)
    Object.assign(layoutOf(d).placements.find((p) => p.id === m.id)!, { x: m.x, y: m.y, rotation: m.rotation })
    const after = score(d)
    expect(after.goals.find((g) => g.id === 'walkways')!.done).toBe(m.done)
    expect(after.stars).toBeGreaterThanOrEqual(s.stars)
  }, 60000)
})

describe('Every room has something (round 16)', () => {
  it('leaves the bathroom out: its fixtures furnish it', () => {
    const d = sampleDoc()
    const g = score(d).goals.find((x) => x.id === 'rooms-used')!
    expect(g.hint).not.toMatch(/Bathroom/)
    expect(g.placeIn).not.toBe(d.rooms.find((r) => r.name === 'Bathroom')!.id)
  })
})

describe('Make room for a desk is a move the chip can make (round 17)', () => {
  /** The sample after a fresh start of a fresh start: the dresser took the last wall a desk fits on. */
  function crowded() {
    const d = sampleDoc({ settled: true })
    const l = layoutOf(d)
    const cat = (id: string) => d.items.find((i) => i.id === l.placements.find((p) => p.id === id)!.itemId)!.category
    const at = (c: string, x: number, y: number, deg: number) => Object.assign(l.placements.find((p) => cat(p.id) === c)!, { x, y, rotation: (deg * Math.PI) / 180 })
    l.placements = l.placements.filter((p) => ['sofa', 'tv-stand', 'bed', 'dining-table'].includes(cat(p.id)))
    at('sofa', 1.9, 3.7, 180)
    at('tv-stand', 2.5, 0.23, 0)
    at('bed', 7.9, 1.96, 90)
    at('dining-table', 4.7, 0.8, 90)
    const add = (c: string, x: number, y: number, deg: number) => {
      const item = itemFromCatalog(catalogEntry(c), `mr-${c}-${l.placements.length}`)
      d.items.push(item)
      l.placements.push({ id: `mr-p${l.placements.length}`, itemId: item.id, x, y, rotation: (deg * Math.PI) / 180 })
    }
    add('dining-chair', 3.95, 0.25, 270)
    add('dining-chair', 4.7, 1.9, 180)
    add('dresser', 5.58, 0.6, 270)
    add('floor-lamp', 3.2, 3.91, 180)
    return d
  }

  it('moves a piece so a desk fits, at no cost in stars, and Next then offers the desk', () => {
    const d = crowded()
    const desk = () => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry('desk'), 'mr-probe')).ok
    const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
    expect(desk()).toBe(false)
    const s = score(d)
    expect(s.errors).toBe(0)
    expect(s.goals.find((g) => g.id === 'work')!.makeRoom).toBeDefined()
    const m = goalMove(d, layoutOf(d), 'work')
    expect(m, 'a move makes room').not.toBeNull()
    for (const q of [m!, ...(m!.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
    expect(desk()).toBe(true)
    expect(score(d).stars).toBeGreaterThanOrEqual(s.stars)
    expect(score(d).errors).toBe(0)
    const next = nextGoal(score(d), fits)!
    expect(next.id === 'work' ? next.item : 'desk').toBe('desk')
  }, 60000)

  it('a fresh start of a fresh start of the sample still reaches Dream home by Next alone', () => {
    const d = sampleDoc()
    expect(follow(d, 'ff0').dream).toBeGreaterThan(0)
    for (const tag of ['ff1', 'ff2']) {
      const l = remixedLayout(d, d.activeLayoutId, 'fresh-spots', ids)!
      d.layouts.push(l)
      d.activeLayoutId = l.id
      const r = follow(d, tag)
      expect(r.dream, r.log.join('\n')).toBeGreaterThan(0)
    }
  }, 180000)
})

describe('Remixes lead with what is new (round 17)', () => {
  it('marks the source design’s finished pick goals as replays, never the twist or one the remix broke', () => {
    const d = sampleDoc()
    expect(follow(d, 'rp0').dream).toBeGreaterThan(0)
    const src = score(d)
    const l = remixedLayout(d, d.activeLayoutId, 'open-floor', ids)!
    d.layouts.push(l)
    d.activeLayoutId = l.id
    const s = score(d)
    const replays = s.goals.filter((g) => g.replay)
    expect(replays.map((g) => g.id)).toEqual(expect.arrayContaining(['plant', 'curtains']))
    for (const g of replays) {
      expect(g.done || g.id.startsWith('twist:') || !!(g.progress ?? 0)).toBe(false)
      expect(g.item && g.tab).toBeTruthy()
    }
    // Goals the source left open are new here.
    const open = new Set(src.goals.filter((g) => !g.done).map((g) => g.id))
    for (const g of s.goals) if (open.has(g.id) && g.item && g.tab) expect(g.replay, g.id).toBeFalsy()
  }, 120000)

  it('Next offers a new goal before a replay', () => {
    const goal = (id: string, item: string, replay?: boolean) => ({ id, title: id, hint: id, done: false, progress: 0, stars: 1, category: 'style' as const, bonus: true, tab: 'decor', item, replay })
    const score = {
      stars: 30,
      maxStars: 40,
      coreMax: 20,
      thresholds: [0, 1, 7, 15, 18, 20],
      level: { name: 'Designer home', index: 4, next: 'Dream home', toNext: 3 },
      goals: [goal('plant', 'plant', true), goal('rug', 'rug', true), goal('books', 'wall-shelf')],
      vibe: 70,
      errors: 0,
      headline: '',
    }
    expect(nextGoal(score, () => true)!.id).toBe('books')
    expect(nextGoal({ ...score, goals: score.goals.slice(0, 2) }, () => true)!.id).toBe('plant')
  })
})

describe('Chip moves are quick (round 17)', () => {
  /**
   * Times each chip move and pick of a fresh-start remix played by Next. Round 16 took 0.3-2 s a
   * tap in node; round 17 ranks spots on a quick score, searches several drop targets at once and
   * keeps the rooms' walking grid, for roughly 60-150 ms. The bounds here are loose for slow CI.
   */
  it('a fresh-start remix: chip moves and picks take well under a second', () => {
    const d = sampleDoc()
    expect(follow(d, 'qk0').dream).toBeGreaterThan(0)
    const l = remixedLayout(d, d.activeLayoutId, 'fresh-spots', ids)!
    d.layouts.push(l)
    d.activeLayoutId = l.id
    const times: number[] = []
    const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
    for (let i = 0; i < 12; i++) {
      const next = nextGoal(score(d), fits)
      if (!next) break
      if (next.id === 'unpack' && next.near) {
        Object.assign(d, unpackDoc(d, next.near)!.doc)
        continue
      }
      const t0 = performance.now()
      if (!next.item || !next.tab) {
        const m = next.id.startsWith('twist:') ? twistMove(d, layoutOf(d)) : goalMove(d, layoutOf(d), next.id)
        times.push(performance.now() - t0)
        if (!m) break
        for (const q of [m, ...(m.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
        continue
      }
      const item = itemFromCatalog(catalogEntry(next.item), `qk${i}`)
      const at = pickSpot(d, item, { category: next.item, roomId: next.placeIn, near: next.near, goalId: next.id })
      times.push(performance.now() - t0)
      const r = at.fail ? null : at.spot ? { ok: true as const, spot: at.spot } : defaultDrop(d, item)
      if (!r?.ok) break
      d.items.push(item)
      layoutOf(d).placements.push({ id: `qkp${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
    }
    const sorted = times.slice().sort((a, b) => a - b)
    const median = sorted[Math.floor(sorted.length / 2)]
    console.log(`chip taps (ms): ${times.map((t) => t.toFixed(0)).join(' ')}; median ${median.toFixed(0)}`)
    expect(times.length).toBeGreaterThanOrEqual(8)
    expect(median).toBeLessThan(400)
    expect(sorted[sorted.length - 1]).toBeLessThan(2500)
  }, 120000)
})
