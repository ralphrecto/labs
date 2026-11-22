import { beforeEach, describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { findDrop } from '../../model/placement'
import { emptyDoc, rectRoom, sampleDoc } from '../../model/sample'
import type { Doc, Placement } from '../../model/types'
import { unpackDoc } from '../../store'
import { useGame, VISIT_KEY } from '../gameStore'
import { applicableVisits, LEVELS, nextGoal, scoreHome, todayVisit, VISIT_LEVEL, VISITS, type VisitInput } from '../goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc, visit?: VisitInput) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)), visit)
const fitsIn = (d: Doc) => (c: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe')).ok

let n = 0
function put(d: Doc, category: string, x: number, y: number, rotation = 0): Placement {
  const item = itemFromCatalog(catalogEntry(category), `vt${++n}`)
  const p = { id: `vtp${n}`, itemId: item.id, x, y, rotation }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}
function drop(d: Doc, category: string, nextTo?: string): boolean {
  const item = itemFromCatalog(catalogEntry(category), `vd${++n}`)
  const r = findDrop(d, layoutOf(d), item, { nextTo })
  if (!r.ok) return false
  d.items.push(item)
  layoutOf(d).placements.push({ id: `vdp${n}`, itemId: item.id, x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation, on: r.spot.on })
  return true
}

describe('visitor of the day', () => {
  it('has 8-10 visits, each with a who, a line and a check', () => {
    expect(VISITS.length).toBeGreaterThanOrEqual(8)
    expect(VISITS.length).toBeLessThanOrEqual(10)
    expect(new Set(VISITS.map((x) => x.id)).size).toBe(VISITS.length)
    for (const x of VISITS) expect(x.emoji && x.who && x.line && x.title).toBeTruthy()
  })

  it('picks the same visitor for the same day, and varies across days', () => {
    const d = sampleDoc()
    expect(todayVisit(d, '2026-11-22').id).toBe(todayVisit(d, '2026-11-22').id)
    expect(todayVisit(d, new Date(2026, 10, 22, 9)).id).toBe(todayVisit(d, new Date(2026, 10, 22, 23)).id)
    const ids = new Set(Array.from({ length: 30 }, (_, i) => todayVisit(d, `2026-12-${String(i + 1).padStart(2, '0')}`).id))
    expect(ids.size).toBeGreaterThan(3)
  })

  it('only offers visits that suit the home, and prefers new stickers', () => {
    const tiny = emptyDoc()
    tiny.rooms.push(rectRoom('Bedroom', 0, 0, 3, 3))
    const ids = applicableVisits(tiny).map((x) => x.id)
    expect(ids).not.toContain('dinner')
    expect(ids).not.toContain('book-club')
    expect(applicableVisits(sampleDoc()).map((x) => x.id)).toContain('dinner')
    const d = sampleDoc()
    const all = applicableVisits(d).map((x) => x.id)
    const left = all[3]
    for (let i = 1; i <= 20; i++) expect(todayVisit(d, `2027-01-${String(i).padStart(2, '0')}`, all.filter((x) => x !== left)).id).toBe(left)
    // Every sticker collected: any suitable visit again.
    expect(all).toContain(todayVisit(d, '2027-02-02', all).id)
    // A wish the home already grants is skipped while another is on offer.
    expect(todayVisit(d, '2027-03-03', [], (id) => id !== 'aunt').id).toBe('aunt')
  })

  it('is a +2 bonus goal, done once the wish is granted (Grandma: a seat by a lamp)', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    expect(score(d).goals.some((g) => g.visit)).toBe(false)
    let g = score(d, { id: 'grandma' }).goals.find((x) => x.visit)!
    expect(g).toMatchObject({ id: 'visit:grandma', stars: 2, bonus: true, done: false, item: 'armchair', tab: 'living' })
    const chair = put(d, 'armchair', 2.5, 1)
    g = score(d, { id: 'grandma' }).goals.find((x) => x.visit)!
    expect(g).toMatchObject({ item: 'floor-lamp', near: chair.id, done: false })
    put(d, 'floor-lamp', 3.4, 1)
    g = score(d, { id: 'grandma' }).goals.find((x) => x.visit)!
    expect(g.done).toBe(true)
    // Done today stays done: the sticker and stars are kept even if the room changes.
    expect(score(emptyDoc(), { id: 'grandma', done: true }).goals.find((x) => x.visit)!.done).toBe(true)
  })

  it('never caps or gates the level: open it changes nothing, done it only adds stars', () => {
    const d = sampleDoc()
    for (const v of VISITS) {
      const base = score(d)
      const open = score(d, { id: v.id })
      const vg = open.goals.find((x) => x.visit)!
      expect(open.coreMax).toBe(base.coreMax)
      expect(open.thresholds).toEqual(base.thresholds)
      if (vg.done) expect(open.level.index).toBeGreaterThanOrEqual(base.level.index)
      else expect(open.level.index).toBe(base.level.index)
      expect(score(d, { id: v.id, done: true }).level.index).toBeGreaterThanOrEqual(base.level.index)
    }
   }, 30000)

  it('Next offers the visitor only from Cozy on, and still reaches Dream home with it in play', () => {
    const d = sampleDoc()
    layoutOf(d).placements = []
    d.items = []
    const visit = { id: 'aunt' }
    const fits = fitsIn(d)
    let dream = -1
    let offeredAt = -1
    for (let i = 0; i < 44; i++) {
      const s = score(d, visit)
      const plain = score(d)
      const vg = s.goals.find((g) => g.visit)!
      if (!vg.done) expect(s.level.index).toBe(plain.level.index)
      if (s.level.index === LEVELS.length - 1) {
        dream = i
        break
      }
      const next = nextGoal(s, fits)
      if (next?.visit && offeredAt < 0) offeredAt = s.level.index
      if (next?.visit) expect(s.level.index >= VISIT_LEVEL || !s.goals.some((g) => !g.bonus && !g.done)).toBe(true)
      if (next?.id === 'unpack' && next.near) {
        Object.assign(d, unpackDoc(d, next.near)!.doc)
        continue
      }
      if (!next?.item || !next.tab) break
      if (!drop(d, next.item, next.near)) break
    }
    expect(dream).toBeGreaterThan(0)
    expect(offeredAt).toBeGreaterThanOrEqual(VISIT_LEVEL)
  }, 60000)

  it('Next gives a tab-less hint when the piece has nowhere to go, never a dead end', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    const s = score(d, { id: 'aunt' })
    const only = { ...s, level: { ...s.level, index: VISIT_LEVEL }, goals: s.goals.filter((g) => g.visit) }
    const next = nextGoal(only, () => false)!
    expect(next.visit).toBe('aunt')
    expect(next.tab).toBeUndefined()
    expect(next.hint).toBeTruthy()
    // Below Cozy with core goals open, the visitor waits.
    const low = { ...s, level: { ...s.level, index: 1 } }
    expect(nextGoal(low, () => true)?.visit).toBeUndefined()
  })
})

describe('visit state', () => {
  beforeEach(() => {
    try {
      localStorage.removeItem(VISIT_KEY)
    } catch {
      // no storage in this environment
    }
    useGame.setState({ visit: { date: '', visitId: '', done: false, stickers: [] } })
  })

  it('rolls a visitor per day, completes it into a sticker, and a new day brings a new one', () => {
    const d = sampleDoc()
    const g = useGame.getState()
    g.rollVisit(d, '2026-11-22')
    const first = useGame.getState().visit
    expect(first).toMatchObject({ date: '2026-11-22', visitId: todayVisit(d, '2026-11-22').id, done: false, stickers: [] })
    g.rollVisit(d, '2026-11-22')
    expect(useGame.getState().visit).toBe(first)
    g.completeVisit('not-today')
    expect(useGame.getState().visit.done).toBe(false)
    g.completeVisit(first.visitId)
    g.completeVisit(first.visitId)
    expect(useGame.getState().visit).toMatchObject({ done: true, stickers: [first.visitId] })
    g.rollVisit(d, '2026-11-23')
    const second = useGame.getState().visit
    expect(second).toMatchObject({ date: '2026-11-23', done: false, stickers: [first.visitId] })
    expect(second.visitId).not.toBe(first.visitId)
    if (typeof localStorage !== 'undefined') expect(JSON.parse(localStorage.getItem(VISIT_KEY)!).date).toBe('2026-11-23')
  })
})
