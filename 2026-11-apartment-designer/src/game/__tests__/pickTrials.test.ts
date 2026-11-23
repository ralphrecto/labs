import { describe, expect, it } from 'vitest'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { emptyDoc, rectRoom, sampleDoc } from '../../model/sample'
import type { Doc } from '../../model/types'
import { analyze } from '../../analysis'
import { remixedLayout, scoreHome } from '../goals'
import { goodPicks, pickChip, pickLabel, pickSpot, presetItem, trialPick, type PickTrial } from '../BuyDrawer'
import { picksFor } from '../swatches'

const t = (o: Partial<PickTrial>): PickTrial => ({ ok: true, stars: 10, warnings: 0, broken: 0, ...o })

describe('pick cards only offer variants that keep the home whole', () => {
  it('drops a variant that undoes a goal or adds a warning, keeping the order', () => {
    const trials: Record<string, PickTrial> = { Long: t({ broken: 1, stars: 9 }), Round: t({}), Small: t({ warnings: 1 }) }
    expect(goodPicks(['Long', 'Round', 'Small'], (c) => trials[c])).toEqual(['Round'])
    expect(goodPicks(['Round', 'Other'], () => t({}))).toEqual(['Round', 'Other'])
  })

  it('keeps only the least bad one when every variant costs something', () => {
    const trials: Record<string, PickTrial> = { A: t({ broken: 1 }), B: t({ warnings: 1 }), C: t({ ok: false }) }
    expect(goodPicks(['A', 'B', 'C'], (c) => trials[c])).toEqual(['B'])
  })

  it('dry-runs each armchair variant on the sample without touching the home', () => {
    const d = sampleDoc()
    const before = JSON.stringify(d)
    const entry = catalogEntry('armchair')
    const kept = goodPicks(picksFor('armchair'), (pk) => trialPick(d, presetItem(entry, '__t', pickChip(pk)!), { category: 'armchair' }))
    expect(kept.length).toBeGreaterThan(0)
    if (kept.length > 1) for (const pk of kept) expect(trialPick(d, presetItem(entry, '__t', pickChip(pk)!), { category: 'armchair' }).broken).toBe(0)
    expect(JSON.stringify(d)).toBe(before)
  })

  it('office chairs get pick cards like dining chairs', () => {
    expect(picksFor('office-chair').length).toBeGreaterThanOrEqual(2)
  })

  it('says "From Mine" for a piece no box held', () => {
    const d = sampleDoc()
    const lamp = { ...d.items[0], id: 'x', category: 'floor-lamp', owned: true }
    expect(pickLabel(lamp, d.items)).toBe('From Mine')
    const box = d.items.find((i) => i.unpack?.length)!
    const opened = { ...box, unpacked: true }
    const fromBox = { ...lamp, category: box.unpack![0] }
    expect(pickLabel(fromBox, [...d.items, opened])).toBe('From your boxes')
  })
})

describe('Next chip placement (round 15)', () => {
  const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
  const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
  it('puts the talk-corner armchair facing the sofa, so the twist is done in one tap', () => {
    const d = sampleDoc()
    const l = remixedLayout(d, d.activeLayoutId, 'talk-corner', () => 'rx-tc')!
    d.layouts.push(l)
    d.activeLayoutId = l.id
    const g = score(d).goals.find((x) => x.id === 'twist:talk-corner')!
    expect(g.done).toBe(false)
    expect(g.item).toBe('armchair')
    const item = itemFromCatalog(catalogEntry('armchair'), 'tc-chair')
    const { spot } = pickSpot(d, item, { category: 'armchair', near: g.near, goalId: g.id })
    expect(spot).toBeDefined()
    d.items.push(item)
    layoutOf(d).placements.push({ id: 'tc-p', itemId: item.id, x: spot!.x, y: spot!.y, rotation: spot!.rotation })
    expect(score(d).goals.find((x) => x.id === 'twist:talk-corner')!.done).toBe(true)
  })

  it('puts a dining table in the dining area when the goal names no room', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4), rectRoom('Dining area', 5, 0, 3.5, 4))
    const item = itemFromCatalog(catalogEntry('dining-table'), 'dt')
    const { spot } = pickSpot(d, item, { category: 'dining-table' })
    expect(spot?.x).toBeGreaterThan(5)
  })
})
