import { describe, expect, it } from 'vitest'
import { catalogEntry } from '../../model/catalog'
import { sampleDoc } from '../../model/sample'
import { goodPicks, pickChip, pickLabel, presetItem, trialPick, type PickTrial } from '../BuyDrawer'
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
