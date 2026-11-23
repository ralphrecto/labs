import { describe, expect, it } from 'vitest'
import { withArticle } from '../../game/BuyDrawer'
import { ART_MOTIFS, parts } from '../../three/ItemModel'
import { catalogEntry, itemFromCatalog } from '../catalog'
import { emptyDoc, rectRoom } from '../sample'
import { STARTER_CATALOG } from '../starterCatalog'
import type { Doc, Item, Placement } from '../types'
import { v } from '../geometry'
import { mountBottom, onWallOk, reconcileWallDecor, snapToWall } from '../wallDecor'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0
function put(d: Doc, category: string, x: number, y: number, over: Partial<Item> = {}, extra: Partial<Placement> = {}): { p: Placement; item: Item } {
  const item = itemFromCatalog(catalogEntry(category), `s${++n}`, over)
  const p: Placement = { id: `sp${n}`, itemId: item.id, x, y, rotation: 0, ...extra }
  d.items.push(item)
  layoutOf(d).placements.push(p)
  return { p, item }
}
const print = (over: Partial<Item> = {}) => itemFromCatalog(catalogEntry('wall-art'), `print${++n}`, { name: 'Small framed print', width: 0.4, height: 0.5, depth: 0.03, ...over })

/** A 4 × 4 m living room with framed art (0.6 × 0.8, 1.1 to 1.9 m up) on the top wall, centred at x = 2. */
function gallery(ceiling = 2.6) {
  const d = emptyDoc()
  d.rooms.push(rectRoom('Living room', 0, 0, 4, 4, ceiling))
  const art = put(d, 'wall-art', 2, 0.02 + 0.004)
  // Hang it where the snap puts it.
  const s = snapToWall(d, layoutOf(d), art.item, v(2, 0.3), art.p.id)!
  Object.assign(art.p, { x: s.x, y: s.y, rotation: s.rotation })
  return { d, art }
}
/** Hang `item` where a drop at `pt` puts it. */
function hang(d: Doc, item: Item, pt: { x: number; y: number }) {
  const s = snapToWall(d, layoutOf(d), item, v(pt.x, pt.y))!
  d.items.push(item)
  const p: Placement = { id: `hp${++n}`, itemId: item.id, x: s.x, y: s.y, rotation: s.rotation, ...(s.mountY != null ? { mountY: s.mountY } : {}) }
  layoutOf(d).placements.push(p)
  return { s, p }
}

describe('stacking frames on a gallery wall', () => {
  it('a placement can carry its own hanging height', () => {
    const art = itemFromCatalog(catalogEntry('wall-art'), 'h')
    expect(mountBottom(art) + art.height / 2).toBeCloseTo(1.5)
    expect(mountBottom(art, { mountY: 0.9 })).toBe(0.9)
    expect(mountBottom(itemFromCatalog(catalogEntry('curtains'), 'c'), { mountY: 1 })).toBeLessThan(0.05)
  })

  it('a print dropped onto framed art hangs straight above it instead of sliding aside', () => {
    const { d, art } = gallery()
    const small = print()
    const s = snapToWall(d, layoutOf(d), small, v(2.05, 0.2))!
    expect(s.x).toBeCloseTo(art.p.x, 5)
    expect(s.y).toBeCloseTo(small.depth / 2 + 0.004, 3)
    // Above the art's top (1.9 m) with a small gap, and under the ceiling.
    expect(s.mountY!).toBeGreaterThan(1.9)
    expect(s.mountY! + small.height).toBeLessThanOrEqual(2.6)
  })

  it('goes below when there is no room under the ceiling, and slides aside when furniture is in the way', () => {
    const low = gallery(2.4)
    const s = snapToWall(low.d, layoutOf(low.d), print(), v(2, 0.2))!
    expect(s.x).toBeCloseTo(2, 5)
    expect(s.mountY! + 0.5).toBeLessThan(1.1)
    // A sofa against the wall under the art: no room below either, so it slides beside the art.
    const sofa = gallery(2.4)
    put(sofa.d, 'sofa', 2, 0.5)
    const t = snapToWall(sofa.d, layoutOf(sofa.d), print(), v(2, 0.2))!
    expect(t.mountY).toBeUndefined()
    expect(Math.abs(t.x - 2)).toBeGreaterThanOrEqual(0.3 + 0.2 + 0.05 - 1e-6)
  })

  it('frames only block each other when their heights overlap', () => {
    const { d } = gallery()
    const top = hang(d, print(), { x: 2, y: 0.2 })
    expect(top.s.mountY).toBeGreaterThan(1.9)
    // A second print dropped on the art: the spot above is taken, so it goes below (no furniture there).
    const below = hang(d, print(), { x: 2, y: 0.2 })
    expect(below.s.mountY! + 0.5).toBeLessThan(1.1)
    // All three stay where they are: each fits its own row.
    for (const pl of layoutOf(d).placements) {
      const it = d.items.find((i) => i.id === pl.itemId)!
      expect(onWallOk(d, layoutOf(d), it, pl)).toBe(true)
    }
    // A plain print hung away from the stack, at its usual height, still keeps off the art.
    const side = snapToWall(d, layoutOf(d), print(), v(2.5, 0.6))!
    expect(side.mountY).toBeUndefined()
    expect(Math.abs(side.x - 2)).toBeGreaterThanOrEqual(0.55 - 1e-6)
  })

  it('a stacked frame keeps its height when the room is edited', () => {
    const { d } = gallery()
    const top = hang(d, print(), { x: 2, y: 0.2 })
    const next: Doc = { ...d, rooms: d.rooms.map((r) => ({ ...r, polygon: r.polygon.map((q) => ({ x: q.x * 1.25, y: q.y })) })) }
    const out = reconcileWallDecor(d, next)
    const p = layoutOf(out).placements.find((x) => x.id === top.p.id)!
    expect(p.mountY).toBe(top.s.mountY)
  })
})

describe('art variety', () => {
  const art = (id: string, over: Partial<Item> = {}) => itemFromCatalog(catalogEntry('wall-art'), id, over)
  it('prints pick different motifs, and every motif stays inside its frame', () => {
    const looks = new Set<string>()
    for (let i = 0; i < 40; i++) {
      const it = art(`a${i}`)
      const ps = parts(it)
      looks.add(JSON.stringify(ps.slice(2).map((p) => p.kind)))
      const frame = ps[0]
      for (const p of ps.slice(1)) {
        expect(Math.abs(p.pos[0]) + (p.rot ? Math.max(p.size[0], p.size[1]) : p.size[0]) / 2).toBeLessThanOrEqual(frame.size[0] / 2 + 1e-6)
        expect(Math.abs(p.pos[1] - frame.pos[1]) + (p.rot ? Math.max(p.size[0], p.size[1]) : p.size[1]) / 2).toBeLessThanOrEqual(frame.size[1] / 2 + 1e-6)
      }
    }
    expect(looks.size).toBeGreaterThanOrEqual(ART_MOTIFS.length - 1)
  })

  it('the print takes its colours from the item, the frame its finish', () => {
    const blue = parts(art('b', { color: '#7fa7c9' }))
    expect(blue.some((p) => p.art && p.color === '#7fa7c9')).toBe(true)
    expect(blue[0].color).toBe('#8a6b4f')
    const black = parts({ ...art('k'), frameColor: '#222222' } as Item)
    expect(black[0].color).toBe('#222222')
    // A stacked print is drawn at its own height.
    expect(parts(art('h'), { mountY: 2 })[0].pos[1]).toBeCloseTo(2 + 0.4)
  })

  it('the catalog has landscape and tall frames', () => {
    const arts = STARTER_CATALOG.filter((p) => p.category === 'wall-art')
    expect(arts.some((p) => p.width > p.height * 1.25 && p.width < 1)).toBe(true)
    expect(arts.some((p) => p.height > p.width * 1.6)).toBe(true)
  })
})

describe('placement copy', () => {
  it('curtains are a pair, not "a curtains"', () => {
    expect(withArticle('curtains')).toBe('a pair of curtains')
    expect(withArticle('Curtains (pair)')).toBe('a pair of Curtains')
    expect(withArticle('armchair')).toBe('an armchair')
    expect(withArticle('framed art')).toBe('a framed art')
  })
})
