import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { sampleDoc } from '../../model/sample'
import type { Doc, TwistId } from '../../model/types'
import { goalMove, LEVELS, nextGoal, remixedLayout, scoreHome, TWIST_IDS, twistMove } from '../goals'
import { pickSpot } from '../BuyDrawer'
import { defaultDrop, unpackDoc } from '../../store'
import { findDrop } from '../../model/placement'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
let k = 0
const ids = () => `tc${++k}`

/**
 * Follows the Next chip exactly as a tap does: a twist that is a move is the chip's own move
 * (twistMove), any other move goal goalMove, a piece goes where the drawer puts it. Returns the step
 * the twist was done at and the step Dream home was reached at (-1: never).
 */
function chipOnly(d: Doc, tag: string, max = 45) {
  const log: string[] = []
  let twistAt = -1
  const skip = new Set<string>()
  for (let i = 0; i < max; i++) {
    const s = score(d)
    const tw = s.goals.find((g) => g.id.startsWith('twist:'))
    if (tw?.done && twistAt < 0) twistAt = i
    if (s.level.index === LEVELS.length - 1) return { twistAt, dream: i, log }
    const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
    const next = nextGoal(s, fits, skip)
    log.push(`${i} ★${s.stars} ${s.level.name} ${next?.id}:${next?.item ?? '(move)'}`)
    if (!next) break
    if (next.id === 'unpack' && next.near) {
      Object.assign(d, unpackDoc(d, next.near)!.doc)
      continue
    }
    if (!next.item || !next.tab) {
      const m = next.id.startsWith('twist:') ? twistMove(d, layoutOf(d)) : goalMove(d, layoutOf(d), next.id)
      if (!m) {
        log.push('  (no move)')
        if (skip.has(next.id)) break
        skip.add(next.id)
        continue
      }
      for (const q of [m, ...(m.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
      continue
    }
    const item = itemFromCatalog(catalogEntry(next.item), `${tag}n${i}`)
    const at = pickSpot(d, item, { category: next.item, roomId: next.placeIn, near: next.near, goalId: next.id })
    const r = at.fail ? null : at.spot ? { ok: true as const, spot: at.spot } : defaultDrop(d, item)
    if (!r?.ok) {
      skip.add(next.id)
      continue
    }
    d.items.push(item)
    layoutOf(d).placements.push({ id: `${tag}p${i}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
  }
  const tw = score(d).goals.find((g) => g.id.startsWith('twist:'))
  return { twistAt: tw?.done ? (twistAt < 0 ? max : twistAt) : twistAt, dream: -1, log }
}

describe('every remix twist can be finished by the Next chip alone (round 18)', () => {
  for (const tw of TWIST_IDS as TwistId[])
    it(`${tw}: the chip finishes the twist and reaches Dream home`, () => {
      const d = sampleDoc()
      const o = chipOnly(d, `o${tw}`)
      expect(o.dream, o.log.join('\n')).toBeGreaterThan(0)
      const l = remixedLayout(d, d.activeLayoutId, tw, ids)!
      d.layouts.push(l)
      d.activeLayoutId = l.id
      const r = chipOnly(d, `r${tw}`)
      expect(r.twistAt, r.log.join('\n')).toBeGreaterThanOrEqual(0)
      expect(score(d).goals.find((g) => g.id === `twist:${tw}`)!.done, r.log.join('\n')).toBe(true)
      expect(r.dream, r.log.join('\n')).toBeGreaterThan(0)
      expect(score(d).errors).toBe(0)
    }, 240000)
})

describe('the chip moves the bed and turns the sofa for their twists (round 18)', () => {
  const cat = (d: Doc, id: string) => d.items.find((i) => i.id === layoutOf(d).placements.find((p) => p.id === id)?.itemId)?.category
  const remix = (d: Doc, tw: TwistId) => {
    const l = remixedLayout(d, d.activeLayoutId, tw, ids)!
    d.layouts.push(l)
    d.activeLayoutId = l.id
    return l
  }
  it('bed swap: one tap puts the bed’s head to another wall, its nightstand along, doors clear and a bedside free', () => {
    const d = sampleDoc({ settled: true })
    remix(d, 'bed-swap')
    const bed = layoutOf(d).placements.find((p) => cat(d, p.id) === 'bed')!
    // A nightstand at the bed's head, as the player would have placed it.
    const r = pickSpot(d, itemFromCatalog(catalogEntry('nightstand'), 'tcns'), { category: 'nightstand', near: bed.id })
    const ns = itemFromCatalog(catalogEntry('nightstand'), 'tcns')
    const spot = r.spot ?? (findDrop(d, layoutOf(d), ns, { nextTo: bed.id }) as { spot: { x: number; y: number; rotation: number } }).spot
    d.items.push(ns)
    layoutOf(d).placements.push({ id: 'tcnsp', itemId: ns.id, x: spot.x, y: spot.y, rotation: spot.rotation })
    const before = score(d)
    expect(before.goals.find((g) => g.id === 'twist:bed-swap')!.done).toBe(false)
    const m = twistMove(d, layoutOf(d))!
    expect(m).not.toBeNull()
    expect(m.id).toBe(bed.id)
    expect(m.done).toBe(true)
    for (const q of [m, ...(m.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
    const after = score(d)
    expect(after.goals.find((g) => g.id === 'twist:bed-swap')!.done).toBe(true)
    expect(after.goals.find((g) => g.id === 'doors-clear')!.done).toBe(true)
    expect(after.errors).toBe(0)
    expect(after.stars).toBeGreaterThanOrEqual(before.stars)
  }, 60000)

  it('window seat: one tap turns the sofa to the window when it looks away', () => {
    const d = sampleDoc({ settled: true })
    remix(d, 'window-seat')
    const sofa = layoutOf(d).placements.find((p) => cat(d, p.id) === 'sofa')!
    sofa.rotation = sofa.rotation + Math.PI
    let g = score(d).goals.find((x) => x.id === 'twist:window-seat')!
    if (g.done) return
    const m = twistMove(d, layoutOf(d))
    expect(m).not.toBeNull()
    for (const q of [m!, ...(m!.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
    g = score(d).goals.find((x) => x.id === 'twist:window-seat')!
    expect(g.done).toBe(true)
  }, 60000)
})
