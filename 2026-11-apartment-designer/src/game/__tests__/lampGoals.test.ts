import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { findDrop } from '../../model/placement'
import { emptyDoc, rectRoom } from '../../model/sample'
import type { Doc, Item, Placement } from '../../model/types'
import { useStore } from '../../store'
import { rotatePlacementSmart, flushEditRun } from '../editActions'
import { nextGoal, scoreHome } from '../goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
let n = 0
function put(d: Doc, category: string, x: number, y: number, rotation = 0, extra: Partial<Placement> = {}, over: Partial<Item> = {}): Placement {
  const item = itemFromCatalog(catalogEntry(category), `g${++n}`, over)
  const p: Placement = { id: `gp${n}`, itemId: item.id, x, y, rotation, ...extra }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return p
}

function bedroom(bedWidth?: number) {
  const d = emptyDoc()
  d.rooms.push(rectRoom('Bedroom', 0, 0, 4, 4))
  const over = bedWidth ? { width: bedWidth } : {}
  const bedItem = itemFromCatalog(catalogEntry('bed'), 'probe', over)
  const bed = put(d, 'bed', 2, bedItem.depth / 2 + 0.05, 0, {}, over)
  const off = bedItem.width / 2 + 0.27
  return { d, bed, off }
}

describe('bedside and desk light goals', () => {
  it('a queen bed wants a light on each side; the lit side is done and the other points at its nightstand', () => {
    const { d, bed, off } = bedroom()
    const left = put(d, 'nightstand', 2 - off, 0.27)
    const right = put(d, 'nightstand', 2 + off, 0.27)
    put(d, 'table-lamp', left.x, left.y - 0.02, 0, { on: left.id })
    const gs = score(d).goals
    const l = gs.find((g) => g.id === `bedside-light:${bed.id}:-1`)!
    const r = gs.find((g) => g.id === `bedside-light:${bed.id}:1`)!
    expect(l.done).toBe(true)
    expect(r.done).toBe(false)
    expect(r.item).toBe('table-lamp')
    expect(r.near).toBe(right.id)
    expect(r.bonus).toBe(true)
    // Local +x is named "right" (seen from the foot of the bed, facing the headboard).
    expect(r.title).toMatch(/right side/)
    // With no table top free, Next offers a floor lamp instead.
    const s = score(d)
    const next = nextGoal({ ...s, goals: [r] }, (c) => c !== 'table-lamp')
    expect(next?.item).toBe('floor-lamp')
  })

  it('a single bed gets one light goal', () => {
    const { d, bed } = bedroom(0.9)
    expect(score(d).goals.filter((g) => g.id.startsWith(`bedside-light:${bed.id}`))).toHaveLength(1)
  })

  it('a lamp on the desk finishes the desk light', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Office', 0, 0, 4, 4))
    const desk = put(d, 'desk', 2, 0.4)
    const g = () => score(d).goals.find((x) => x.id === `desk-light:${desk.id}`)!
    expect(g().done).toBe(false)
    expect(g().item).toBe('table-lamp')
    put(d, 'table-lamp', 1.4, 0.25, 0, { on: desk.id })
    expect(g().done).toBe(true)
  })

  it('a table lamp on the side table by the sofa gives soft light', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 4))
    put(d, 'sofa', 2.5, 0.5)
    const sofaW = d.items[0].width
    const t = put(d, 'side-table', 2.5 + sofaW / 2 + 0.3, 0.3)
    expect(score(d).goals.find((g) => g.id === 'light')!.done).toBe(false)
    put(d, 'table-lamp', t.x, t.y, 0, { on: t.id })
    expect(score(d).goals.find((g) => g.id === 'light')!.done).toBe(true)
  })

  it('turning a lamp keeps it on its host', () => {
    const { d, off } = bedroom()
    const stand = put(d, 'nightstand', 2 - off, 0.27)
    const lamp = put(d, 'table-lamp', stand.x, stand.y - 0.03, 0, { on: stand.id })
    useStore.setState({ doc: d, past: [], future: [], gesture: false })
    rotatePlacementSmart(lamp.id, Math.PI / 4)
    flushEditRun()
    const now = layoutOf(useStore.getState().doc).placements.find((p) => p.id === lamp.id)!
    expect(now.on).toBe(stand.id)
    expect(now.rotation).toBeCloseTo(Math.PI / 4)
    expect(Math.abs(now.x - stand.x)).toBeLessThan(0.25)
    expect(Math.abs(now.y - stand.y)).toBeLessThan(0.2)
  })

  it('a nightstand the game would refuse is never suggested: no bedside goal there', () => {
    // A bed in the studio's living room while the home has a bedroom: findDrop keeps nightstands in the bedroom.
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 4, 4), rectRoom('Bedroom', 4, 0, 4, 4))
    const bedItem = itemFromCatalog(catalogEntry('bed'), 'probe')
    const bed = put(d, 'bed', 2, bedItem.depth / 2 + 0.05)
    expect(findDrop(d, layoutOf(d), itemFromCatalog(catalogEntry('nightstand'), 'probe-ns'), { nextTo: bed.id }).ok).toBe(false)
    expect(score(d).goals.filter((g) => g.id.startsWith(`bedside-light:${bed.id}`))).toEqual([])
    // In a bedroom, both sides are on offer.
    const { d: d2, bed: bed2 } = bedroom()
    expect(score(d2).goals.filter((g) => g.id.startsWith(`bedside-light:${bed2.id}`) && g.item === 'nightstand')).toHaveLength(2)
  })

  it('soft light done says so, not "bring it within"', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 5, 5))
    put(d, 'sofa', 2.5, 0.6)
    put(d, 'floor-lamp', 4, 0.4)
    const g = score(d).goals.find((x) => x.id === 'light')!
    expect(g.done).toBe(true)
    expect(g.hint).toBe('A warm pool of light by the sofa.')
  })

  it('a pendant over the dining table finishes the dining light', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Dining', 0, 0, 4, 4))
    const table = put(d, 'dining-table', 2, 2)
    let g = score(d).goals.find((x) => x.id === `dining-light:${table.id}`)!
    expect(g.done).toBe(false)
    expect(g.item).toBe('pendant')
    put(d, 'pendant', 2, 2, 0, { on: table.id })
    g = score(d).goals.find((x) => x.id === `dining-light:${table.id}`)!
    expect(g.done).toBe(true)
  })
})
