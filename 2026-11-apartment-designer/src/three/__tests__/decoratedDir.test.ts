import { describe, expect, it } from 'vitest'
import { catalogEntry, itemFromCatalog } from '../../model/catalog'
import { v } from '../../model/geometry'
import { emptyDoc, rectRoom } from '../../model/sample'
import type { Doc } from '../../model/types'
import { snapToWall } from '../../model/wallDecor'
import { decoratedDir } from '../Scene3D'

const layoutOf = (d: Doc) => d.layouts.find((l) => l.id === d.activeLayoutId)!
let n = 0
function hang(d: Doc, pt: { x: number; y: number }) {
  const item = itemFromCatalog(catalogEntry('wall-art'), `t${++n}`)
  const s = snapToWall(d, layoutOf(d), item, v(pt.x, pt.y))!
  d.items.push(item)
  layoutOf(d).placements.push({ id: `tp${n}`, itemId: item.id, x: s.x, y: s.y, rotation: s.rotation })
  return s
}
/** Whether a camera far off along `dir` sees a piece from inside its room (as PlacedItem does). */
const seen = (dir: { x: number; z: number }, rot: number) => Math.sin(rot) * dir.x - Math.cos(rot) * dir.z < 0

describe('decoratedDir', () => {
  it('keeps the usual overview with no wall decor', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 4, 4))
    const dir = decoratedDir(d)
    expect(dir.z).toBeGreaterThan(0.5)
  })

  it('turns round to show whichever wall carries the art', () => {
    for (const pt of [v(2, 0.05), v(2, 3.95), v(0.05, 2), v(3.95, 2)]) {
      const d = emptyDoc()
      d.rooms.push(rectRoom('Living room', 0, 0, 4, 4))
      const s = hang(d, pt)
      hang(d, pt)
      expect(seen(decoratedDir(d), s.rotation)).toBe(true)
    }
  })
})

describe('decoratedDir counts pieces', () => {
  it('a wall of frames beats one big pair of curtains', () => {
    const d = emptyDoc()
    d.rooms.push(rectRoom('Living room', 0, 0, 4, 4))
    const big = itemFromCatalog(catalogEntry('wall-art'), 'big', { width: 3, height: 2 })
    d.items.push(big)
    const s0 = snapToWall(d, layoutOf(d), big, v(2, 0.05))!
    layoutOf(d).placements.push({ id: 'bigp', itemId: big.id, x: s0.x, y: s0.y, rotation: s0.rotation })
    const s = hang(d, v(1, 3.95))
    hang(d, v(2, 3.95))
    hang(d, v(3, 3.95))
    expect(seen(decoratedDir(d), s.rotation)).toBe(true)
  })
})
