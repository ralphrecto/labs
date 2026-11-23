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
      if (m) Object.assign(layoutOf(d).placements.find((p) => p.id === m.id)!, { x: m.x, y: m.y, rotation: m.rotation })
      const ok = !!m && moved()
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
