import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { checkLayout, doorKeepOut } from '../../model/checks'
import { convexOverlap, orientedRect, pointInPolygon, polygonInsideRoom, v } from '../../model/geometry'
import { findDrop, polygonGap } from '../../model/placement'
import { sampleDoc } from '../../model/sample'
import type { Doc, Layout, Placement } from '../../model/types'
import { isWallDecor } from '../../model/wallDecor'
import { defaultDrop, unpackDoc } from '../../store'
import { pickSpot } from '../BuyDrawer'
import { goalMove, LEVELS, nextGoal, remixedLayout, scoreHome, twistMove } from '../goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
const cat = (d: Doc, p: Placement) => d.items.find((i) => i.id === p.itemId)!.category
const bodyOf = (d: Doc, p: Placement) => {
  const it = d.items.find((i) => i.id === p.itemId)!
  return orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation)
}
const solidsOf = (d: Doc, l: Layout) =>
  l.placements.filter((p) => {
    const it = d.items.find((i) => i.id === p.itemId)!
    return !p.on && !it.floorLevel && !isWallDecor(it.category)
  })

/** Follows the Next chip as the HUD does: chip moves (with the pieces they carry) and picks where the drawer puts them. */
function follow(d: Doc, tag: string, log: string[] = [], max = 40) {
  const skip = new Set<string>()
  for (let i = 0; i < max; i++) {
    const s = score(d)
    if (s.level.index === LEVELS.length - 1) return i
    const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
    const next = nextGoal(s, fits, skip)
    if (!next || skip.has(next.id)) break
    log.push(`${i} ★${s.stars} ${next.id}:${next.item ?? '(move)'}`)
    if (next.id === 'unpack' && next.near) {
      Object.assign(d, unpackDoc(d, next.near)!.doc)
      continue
    }
    if (!next.item || !next.tab) {
      const m = next.id.startsWith('twist:') ? twistMove(d, layoutOf(d)) : goalMove(d, layoutOf(d), next.id)
      if (!m) skip.add(next.id)
      else for (const q of [m, ...(m.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
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
  return -1
}

describe('Doorways stay clear (round 17)', () => {
  it('the checks flag a piece standing on the landing of a door, even one that swings the other way', () => {
    const d = sampleDoc()
    const ns = itemFromCatalog(catalogEntry('nightstand'), 'dw-ns')
    d.items.push(ns)
    // The bedroom door (to the bathroom) swings out, into the bathroom: its landing in the bedroom still counts.
    layoutOf(d).placements.push({ id: 'dw-p', itemId: ns.id, x: 8.1, y: 3.15, rotation: Math.PI })
    const issues = checkLayout(d, layoutOf(d)).filter((i) => i.placementIds.includes('dw-p'))
    expect(issues.map((i) => i.kind)).toContain('door')
  })

  it('a nightstand for a bed whose head is by the bathroom door never goes in a door’s swing or landing', () => {
    const d = sampleDoc()
    const bed = layoutOf(d).placements.find((p) => cat(d, p) === 'bed')!
    // The bed's head against the bedroom's south wall, beside the door to the bathroom.
    Object.assign(bed, { x: 6.96, y: 2.38, rotation: Math.PI })
    const keep = doorKeepOut(d.rooms)
    for (const side of [1, -1]) {
      const ns = itemFromCatalog(catalogEntry('nightstand'), `dk-ns${side}`)
      const at = pickSpot(d, ns, { category: 'nightstand', goalId: `nightstand:${bed.id}:${side}` })
      if (!at.spot) continue
      const body = orientedRect(v(at.spot.x, at.spot.y), ns.width / 2, ns.depth / 2, at.spot.rotation)
      expect(keep.some((k) => convexOverlap(body, k, 0.005)), `side ${side}`).toBe(false)
    }
  })

  it('two Fresh start remixes of the sample, played by Next: nothing solid stands in a door swing or doorway, and a bed keeps a bedside', () => {
    const d = sampleDoc()
    expect(follow(d, 'dk0')).toBeGreaterThan(0)
    let n = 0
    for (const tag of ['dk1', 'dk2']) {
      const l = remixedLayout(d, d.activeLayoutId, 'fresh-spots', () => `${tag}-${++n}`)!
      d.layouts.push(l)
      d.activeLayoutId = l.id
      const log: string[] = []
      expect(follow(d, tag, log), `${tag}\n${log.join('\n')}`).toBeGreaterThan(0)
      const keep = doorKeepOut(d.rooms)
      for (const p of solidsOf(d, l)) expect(keep.some((k) => convexOverlap(bodyOf(d, p), k, 0.005)), `${tag} ${cat(d, p)}`).toBe(false)
      const beds = l.placements.filter((p) => cat(d, p) === 'bed')
      for (const b of beds) expect(l.placements.some((p) => (cat(d, p) === 'nightstand' || cat(d, p) === 'side-table') && Math.hypot(p.x - b.x, p.y - b.y) < 1.6), tag).toBe(true)
    }
  }, 240000)
})

describe('Fresh start is fresh (round 17)', () => {
  /** The chip's twist moves on a Fresh start of the played sample, by remix id. */
  function freshMoves(src: Doc, seed: string) {
    const d = structuredClone(src)
    let n = 0
    const l = remixedLayout(d, d.activeLayoutId, 'fresh-spots', () => `${seed}-${++n}`)!
    d.layouts.push(l)
    d.activeLayoutId = l.id
    const moves: string[] = []
    for (let i = 0; i < 6; i++) {
      const m = twistMove(d, l)
      if (!m) break
      for (const q of [m, ...(m.also ?? [])]) {
        const p = l.placements.find((x) => x.id === q.id)!
        moves.push(`${cat(d, p)}@${q.x.toFixed(1)},${q.y.toFixed(1)},${q.rotation.toFixed(1)}`)
        Object.assign(p, { x: q.x, y: q.y, rotation: q.rotation })
      }
    }
    return { d, moves, s: score(d) }
  }

  it('different remixes of one design make different moves, and the sofa still faces the TV', () => {
    const src = sampleDoc()
    expect(follow(src, 'fv0')).toBeGreaterThan(0)
    const runs = ['s1', 's4'].map((seed) => freshMoves(src, seed))
    expect(runs[0].moves.join(' ')).not.toBe(runs[1].moves.join(' '))
    for (const r of runs) {
      expect(r.s.goals.find((g) => g.id === 'twist:fresh-spots')!.done, r.moves.join(' ')).toBe(true)
      expect(r.s.goals.find((g) => g.id === 'sofa-tv')!.done, r.moves.join(' ')).toBe(true)
      expect(r.s.errors).toBe(0)
    }
    // At least one of them puts the sofa on a new wall (with its TV across from it), not just a slide.
    const turned = runs.some((r) => {
      const sofa = layoutOf(r.d).placements.find((p) => cat(r.d, p) === 'sofa')!
      const was = src.layouts.find((l) => l.id === src.activeLayoutId)!.placements.find((p) => p.itemId === sofa.itemId)!
      return Math.abs(Math.cos(sofa.rotation - was.rotation) - 1) > 0.5
    })
    expect(turned, runs.map((r) => r.moves.join(' ')).join('\n')).toBe(true)
  }, 120000)
})

describe('The living room hangs together (round 17)', () => {
  it('a sofa the chip moves takes its coffee table and rug along', () => {
    const d = sampleDoc({ settled: true })
    const src = layoutOf(d)
    // A Fresh start that kept the settled home's seating as it stood.
    const l: Layout = { ...structuredClone(src), id: 'rg-remix', name: 'Fresh start', remix: { from: src.id, twist: 'fresh-spots', anchors: 4 } }
    l.placements = l.placements.map((p) => ({ ...p, id: `rg-${p.id}` }))
    d.layouts.push(l)
    d.activeLayoutId = l.id
    const of0 = (c: string) => l.placements.find((p) => cat(d, p) === c)!
    const gap0 = polygonGap(bodyOf(d, of0('rug')), bodyOf(d, of0('sofa')))
    let sofaMove = null
    for (let i = 0; i < 6 && !sofaMove; i++) {
      const m = twistMove(d, l)
      if (!m) break
      for (const q of [m, ...(m.also ?? [])]) Object.assign(l.placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
      if (cat(d, l.placements.find((p) => p.id === m.id)!) === 'sofa') sofaMove = m
    }
    expect(sofaMove, 'the chip moved the sofa').not.toBeNull()
    const carried = (sofaMove!.also ?? []).map((q) => cat(d, l.placements.find((p) => p.id === q.id)!))
    expect(carried).toEqual(expect.arrayContaining(['coffee-table', 'rug']))
    const of = (c: string) => l.placements.find((p) => cat(d, p) === c)!
    const rug = bodyOf(d, of('rug'))
    // Still in front of the sofa as before, and under the coffee table.
    expect(polygonGap(rug, bodyOf(d, of('sofa')))).toBeLessThanOrEqual(gap0 + 0.05)
    expect(convexOverlap(rug, bodyOf(d, of('coffee-table')), 0.01)).toBe(true)
    expect(d.rooms.some((r) => polygonInsideRoom(rug, r.polygon))).toBe(true)
  }, 60000)

  it('a rug for a sofa near a wall slides along it to stay under the sofa and its coffee table', () => {
    const d = sampleDoc({ settled: true })
    const l = layoutOf(d)
    l.placements = l.placements.filter((p) => cat(d, p) !== 'rug')
    Object.assign(l.placements.find((p) => cat(d, p) === 'sofa')!, { x: 1.05 })
    Object.assign(l.placements.find((p) => cat(d, p) === 'coffee-table')!, { x: 1.05, y: 2.6 })
    const rug = itemFromCatalog(catalogEntry('rug'), 'rg-new')
    const r = findDrop(d, l, rug)
    expect(r.ok).toBe(true)
    if (!r.ok) return
    const body = orientedRect(v(r.spot.x, r.spot.y), rug.width / 2, rug.depth / 2, r.spot.rotation)
    expect(convexOverlap(body, bodyOf(d, l.placements.find((p) => cat(d, p) === 'sofa')!), 0.01)).toBe(true)
    expect(convexOverlap(body, bodyOf(d, l.placements.find((p) => cat(d, p) === 'coffee-table')!), 0.01)).toBe(true)
  })

  it('a desk with no study goes to the bedroom before the living room', () => {
    const d = sampleDoc({ settled: true })
    const desk = itemFromCatalog(catalogEntry('desk'), 'dr-desk')
    const at = pickSpot(d, desk, { category: 'desk', goalId: 'work' })
    expect(at.spot).toBeDefined()
    const bedroom = d.rooms.find((r) => r.name === 'Bedroom')!
    expect(pointInPolygon(v(at.spot!.x, at.spot!.y), bedroom.polygon)).toBe(true)
  })
})
