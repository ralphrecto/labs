import { describe, expect, it } from 'vitest'
import { analyze } from '../../analysis'
import { sampleDoc } from '../../model/sample'
import type { Doc } from '../../model/types'
import { parseDoc } from '../../storage'
import { unpackDoc, unpackMessage, useStore } from '../../store'
import { nextGoal, scoreHome } from '../goals'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
const score = (d: Doc) => scoreHome(d, layoutOf(d), analyze(d, layoutOf(d)))
const boxIds = (d: Doc) => layoutOf(d).placements.filter((p) => d.items.find((i) => i.id === p.itemId)?.unpack?.length).map((p) => p.id)

describe('moving boxes', () => {
  it('the sample has an open unpack goal worth 2 stars, ranked first by Next, and it selects a box', () => {
    const d = sampleDoc()
    const g = score(d).goals.find((x) => x.id === 'unpack')!
    expect(g.stars).toBe(2)
    expect(g.done).toBe(false)
    expect(g.bonus).toBe(true)
    expect(boxIds(d)).toContain(g.near)
    expect(nextGoal(score(d), () => true)!.id).toBe('unpack')
  })

  it('unpacking both boxes finishes the goal and the plant goal', () => {
    let d = sampleDoc()
    const before = score(d).stars
    for (const id of boxIds(d)) d = unpackDoc(d, id)!.doc
    const s = score(d)
    expect(s.goals.find((x) => x.id === 'unpack')!.done).toBe(true)
    expect(s.goals.find((x) => x.id === 'plant')!.done).toBe(true)
    expect(s.stars).toBeGreaterThan(before + 2)
    expect(boxIds(d)).toEqual([])
    expect(d.items.filter((i) => i.unpacked)).toHaveLength(2)
  })

  it('packed boxes do not count as furniture pieces', () => {
    const d = sampleDoc()
    const withBoxes = score(d).goals.find((x) => x.id === 'unpack')!
    // Three pieces besides the boxes: fewer than MIN_PIECES, so unpacking is urgent.
    expect(withBoxes.urgent).toBe(true)
  })

  it('unpackBox adds owned pieces in one step, and Undo puts the box back', () => {
    const d = sampleDoc()
    useStore.setState({ doc: d, past: [], future: [], gesture: false })
    const box = boxIds(d)[0]
    const items = d.items.length
    const ids = useStore.getState().unpackBox(box)
    const after = useStore.getState().doc
    const added = after.items.length - items
    expect(added).toBeGreaterThanOrEqual(2)
    expect(added).toBeLessThanOrEqual(3)
    expect(after.items.slice(items).every((i) => i.owned)).toBe(true)
    expect(ids.length).toBeGreaterThan(0)
    expect(layoutOf(after).placements.some((p) => p.id === box)).toBe(false)
    expect(useStore.getState().toast).toMatch(/^Unpacked a /)
    expect(useStore.getState().past).toHaveLength(1)
    useStore.getState().undo()
    const back = useStore.getState().doc
    expect(back.items).toHaveLength(items)
    expect(layoutOf(back).placements.some((p) => p.id === box)).toBe(true)
    expect(back.items.some((i) => i.unpacked)).toBe(false)
  })

  it('says what is waiting in Mine', () => {
    expect(unpackMessage({ placed: ['plant', 'table lamp'], waiting: [] })).toBe('Unpacked a plant and a table lamp')
    expect(unpackMessage({ placed: ['plant', 'floor lamp'], waiting: ['rug'] })).toBe('Unpacked a plant and a floor lamp · rug waiting in Mine')
  })

  it('gives old saves of the sample box contents', () => {
    const d = sampleDoc()
    for (const i of d.items) delete i.unpack
    const back = parseDoc(JSON.stringify(d))!
    expect(back.items.filter((i) => i.unpack?.length)).toHaveLength(2)
  })
})
