import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { emptyDoc, rectRoom, sampleDoc } from '../../model/sample'
import type { Doc, Item, Placement } from '../../model/types'
import { findDrop, polygonGap } from '../../model/placement'
import { orientedRect, v } from '../../model/geometry'
import { LEVELS, nextGoal, scoreHome } from '../goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))

let n = 0
function put(d: Doc, category: string, x: number, y: number, rotation = 0): Placement {
  const item = itemFromCatalog(catalogEntry(category), `nc${++n}`)
  const p = { id: `ncp${n}`, itemId: item.id, x, y, rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}
function drop(d: Doc, category: string, nextTo?: string): Placement | undefined {
  const item = itemFromCatalog(catalogEntry(category), `nd${++n}`)
  const r = findDrop(d, layoutOf(d), item, { nextTo })
  if (!r.ok) return undefined
  d.items.push(item)
  const p = { id: `ndp${n}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on }
  layoutOf(d).placements.push(p)
  return p
}
const body = (d: Doc, p: Placement) => {
  const it = d.items.find((i) => i.id === p.itemId)! as Item
  return orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation)
}

/** Follows the Next chip, placing each suggested piece as the drawer would (beside `near`). */
function followNext(d: Doc, max = 40) {
  const fits = (c: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe')).ok
  const log: string[] = []
  let last = ''
  let repeats = 0
  let best = -1
  for (let i = 0; i < max; i++) {
    const s = score(d)
    if (s.level.index === LEVELS.length - 1) return { dream: i, log }
    const next = nextGoal(s, fits)
    log.push(`${i} ★${s.stars} ${next?.id}:${next?.item ?? '(move)'}`)
    if (!next?.item || !next.tab) break
    const key = `${next.id}:${next.item}`
    repeats = key === last && s.stars <= best ? repeats + 1 : 1
    expect(repeats, `${key} offered ${repeats} times without progress\n${log.join('\n')}`).toBeLessThanOrEqual(3)
    last = key
    best = Math.max(best, s.stars)
    if (!drop(d, next.item, next.near)) break
  }
  return { dream: -1, log }
}

describe('Next chip dead ends (round 7)', () => {
  it('reaches Dream home from an empty layout of the sample home, with no loops', () => {
    const d = sampleDoc()
    layoutOf(d).placements = []
    d.items = []
    const r = followNext(d)
    expect(r.dream, r.log.join('\n')).toBeGreaterThan(0)
  }, 60000)

  it('turns the coffee goal into a move when a coffee table is already there, never another purchase', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    const sofa = put(d, 'sofa', 2.5, 0.5)
    const table = put(d, 'coffee-table', 4.3, 3.4)
    const g = score(d).goals.find((x) => x.id === `coffee:${sofa.id}`)!
    expect(g.done).toBe(false)
    expect(g.tab).toBeUndefined()
    expect(g.item).toBeUndefined()
    expect(g.near).toBe(table.id)
    expect(g.hint).toMatch(/^Move the coffee table/)
    // Whatever Next suggests, it never buys a second coffee table.
    for (let i = 0; i < 6; i++) {
      const next = nextGoal(score(d), () => true)
      expect(next?.item).not.toBe('coffee-table')
      if (!next?.item) break
      drop(d, next.item, next.near)
    }
  })

  it('places a coffee table from the hint in front of the sofa, 45-50 cm from the seat', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    const sofa = put(d, 'sofa', 2.5, 0.5)
    const g = score(d).goals.find((x) => x.id === `coffee:${sofa.id}`)!
    expect(g.item).toBe('coffee-table')
    expect(g.near).toBe(sofa.id)
    const t = drop(d, 'coffee-table', g.near)!
    const gap = polygonGap(body(d, sofa), body(d, t))
    expect(gap).toBeGreaterThanOrEqual(0.45 - 0.005)
    expect(gap).toBeLessThanOrEqual(0.5 + 0.005)
    expect(t.y).toBeGreaterThan(sofa.y)
    expect(score(d).goals.find((x) => x.id === `coffee:${sofa.id}`)!.done).toBe(true)
  })

  it('keeps a bed-sized spot free in an empty bedroom when the desk, chair and dresser go first', () => {
    const d = sampleDoc()
    layoutOf(d).placements = []
    d.items = []
    for (const c of ['desk', 'office-chair', 'dresser', 'bookshelf']) drop(d, c)
    const where = layoutOf(d).placements.map((p) => `${d.items.find((i) => i.id === p.itemId)!.category}@${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ')
    expect(findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry('bed'), 'probe')).ok, where).toBe(true)
  })

  it('says what to move when a core goal has no room, and only mentions an old bed when there is one', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Bedroom', 0, 0, 2.6, 2.6))
    const desk = put(d, 'desk', 1.3, 1.3)
    const bed = itemFromCatalog(catalogEntry('bed'), 'probe')
    const r = findDrop(d, layoutOf(d), bed)
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.reason).not.toMatch(/old bed/)
    // Other goals (a chair for the desk) may rank first; with only the sleep goal open, Next makes room.
    const s = score(d)
    const next = nextGoal({ ...s, goals: s.goals.filter((g) => g.id === 'sleep') }, (c) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe')).ok)!
    expect(next.id).toBe('sleep')
    expect(next.title).toMatch(/make room for a bed/i)
    expect(next.tab).toBeUndefined()
    expect(next.near).toBe(desk.id)
    expect(next.hint).toMatch(/desk/i)
  })
})
