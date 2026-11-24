import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { emptyDoc } from '../../model/sample'
import type { Doc } from '../../model/types'
import { goalMove, LEVELS, nextGoal, scoreHome } from '../goals'
import { pickSpot } from '../BuyDrawer'
import { defaultDrop } from '../../store'
import { findDrop } from '../../model/placement'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))

/** A scanned living room, applied as one room with the four pieces the scan found. */
function scannedLivingRoom(): Doc {
  const d = emptyDoc()
  d.rooms.push({
    id: 'scan-room',
    name: 'Living room',
    // (A roomy studio-sized living room, as a scan of an open-plan space comes in.)
    polygon: [
      { x: 0, y: 0 },
      { x: 6.4, y: 0 },
      { x: 6.4, y: 4.8 },
      { x: 0, y: 4.8 },
    ],
    ceilingHeight: 2.55,
    wallThickness: 0.12,
    openings: [
      { id: 'door', kind: 'door', wall: 1, offset: 0.9, width: 0.9, height: 2.1, sill: 0, hinge: 'end', swing: 'in', entrance: true },
      { id: 'win', kind: 'window', wall: 0, offset: 3.8, width: 1.6, height: 1.4, sill: 0.8 },
    ],
  } as Doc['rooms'][number])
  const put = (c: string, x: number, y: number, rotation: number) => {
    const it = { ...itemFromCatalog(catalogEntry(c), `scan-${c}`), source: 'scan' as const, scanId: 'scan1', owned: true }
    d.items.push(it)
    layoutOf(d).placements.push({ id: `scanp-${c}`, itemId: it.id, x, y, rotation })
  }
  put('sofa', 1.6, 3.2, Math.PI)
  put('coffee-table', 1.6, 2.1, 0)
  put('rug', 1.6, 2.2, 0)
  put('tv-stand', 1.6, 0.25, 0)
  return d
}

function follow(d: Doc, tag: string, max = 50) {
  const log: string[] = []
  const skip = new Set<string>()
  for (let i = 0; i < max; i++) {
    const s = score(d)
    if (s.level.index === LEVELS.length - 1) return { dream: i, log }
    const fits = (c: string, near?: string) => findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry(c), 'probe'), { nextTo: near }).ok
    const next = nextGoal(s, fits, skip)
    log.push(`${i} ★${s.stars}/${s.thresholds.join(',')} ${s.level.name} ${next?.id}:${next?.item ?? '(move)'}`)
    if (!next || skip.has(next.id)) break
    if (!next.item || !next.tab) {
      const m = goalMove(d, layoutOf(d), next.id)
      if (m) for (const q of [m, ...(m.also ?? [])]) Object.assign(layoutOf(d).placements.find((p) => p.id === q.id)!, { x: q.x, y: q.y, rotation: q.rotation })
      else skip.add(next.id)
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
  return { dream: -1, log }
}

describe('a one-room scanned home paces like a home (round 18)', () => {
  it('takes 14 or more Next steps to Dream home, and its scanned pieces earn no more than Lived-in', () => {
    const d = scannedLivingRoom()
    const s0 = score(d)
    console.log(`scanned start: ★${s0.stars}, ${s0.level.name}, thresholds ${s0.thresholds.join(',')}`)
    expect(s0.level.index).toBeLessThanOrEqual(LEVELS.indexOf('Lived-in'))
    const r = follow(d, 'sc')
    console.log(`one-room scan: Dream home after ${r.dream} Next steps\n${r.log.join('\n')}`)
    expect(r.dream, r.log.join('\n')).toBeGreaterThanOrEqual(14)
    // The same home had it all come with the scan: furnished, but no more than Lived-in.
    for (const it of d.items) it.scanId = 'scan1'
    expect(score(d).level.index).toBe(LEVELS.indexOf('Lived-in'))
  }, 120000)
})
