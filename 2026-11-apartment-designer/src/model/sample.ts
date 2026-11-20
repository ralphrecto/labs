import { catalogEntry, itemFromCatalog } from './catalog'
import type { Doc, Room } from './types'
import { nanoid } from 'nanoid'

export const uid = () => nanoid(10)

export function rectRoom(name: string, x: number, y: number, w: number, d: number, ceiling = 2.6): Room {
  return {
    id: uid(),
    name,
    polygon: [
      { x, y },
      { x: x + w, y },
      { x: x + w, y: y + d },
      { x, y: y + d },
    ],
    ceilingHeight: ceiling,
    wallThickness: 0.12,
    openings: [],
  }
}

export function emptyDoc(): Doc {
  const layoutId = uid()
  return {
    version: 1,
    apartment: { id: uid(), name: 'My apartment', units: 'metric' },
    rooms: [],
    items: [],
    layouts: [{ id: layoutId, name: 'Layout 1', placements: [] }],
    activeLayoutId: layoutId,
  }
}

/**
 * A small one-bedroom so the app has something to play with on first load. Move-in day: the
 * movers left the sofa, the TV stand, the bed and a couple of boxes, so there's a whole home to
 * fill (storage, a table, a rug, lamps, plants...). `settled` is the home a few weekends later,
 * with the coffee table, rug and dining set in (tests use it as a half-furnished home).
 */
export function sampleDoc(opts: { settled?: boolean } = {}): Doc {
  const living = rectRoom('Living room', 0, 0, 5.2, 4.2)
  const bedroom = rectRoom('Bedroom', 5.32, 0, 3.6, 3.4)
  const bath = rectRoom('Bathroom', 5.32, 3.52, 3.6, 2.0)
  living.openings.push(
    { id: uid(), kind: 'door', wall: 2, offset: 0.4, width: 0.9, height: 2.1, sill: 0, hinge: 'end', swing: 'in', entrance: true },
    { id: uid(), kind: 'window', wall: 0, offset: 1.2, width: 1.6, height: 1.4, sill: 0.8 },
    { id: uid(), kind: 'door', wall: 1, offset: 2.3, width: 0.8, height: 2.1, sill: 0, hinge: 'end', swing: 'out' },
  )
  bedroom.openings.push(
    { id: uid(), kind: 'window', wall: 0, offset: 1.0, width: 1.4, height: 1.4, sill: 0.8 },
    { id: uid(), kind: 'door', wall: 2, offset: 0.4, width: 0.75, height: 2.1, sill: 0, hinge: 'start', swing: 'out' },
  )
  const sofa = itemFromCatalog(catalogEntry('sofa'), uid(), { owned: true })
  const tv = itemFromCatalog(catalogEntry('tv-stand'), uid(), { owned: true })
  const bed = itemFromCatalog(catalogEntry('bed'), uid(), { name: 'Queen bed', owned: true, clearance: { front: 0.6, back: 0, left: 0.6, right: 0 } })
  const hd = (i: typeof sofa) => i.depth / 2
  const items = [sofa, tv, bed]
  const at = (item: typeof sofa, x: number, y: number, rotation = 0) => ({ id: uid(), itemId: item.id, x, y, rotation })
  const placements = [at(sofa, 2.0, 4.2 - hd(sofa), Math.PI), at(tv, 2.0, hd(tv)), at(bed, 5.32 + 1.8, hd(bed))]
  if (opts.settled) {
    const coffee = itemFromCatalog(catalogEntry('coffee-table'), uid(), { owned: true })
    const rug = itemFromCatalog(catalogEntry('rug'), uid(), { owned: true })
    const table = itemFromCatalog(catalogEntry('dining-table'), uid(), { width: 1.2, depth: 0.8, owned: true })
    const chair1 = itemFromCatalog(catalogEntry('dining-chair'), uid(), { owned: true })
    const chair2 = itemFromCatalog(catalogEntry('dining-chair'), uid(), { owned: true })
    items.push(coffee, rug, table, chair1, chair2)
    placements.unshift(at(rug, 2.0, 2.0))
    placements.push(at(coffee, 2.0, 2.45), at(table, 4.3, 1.2, Math.PI / 2), at(chair1, 3.55, 1.2, -Math.PI / 2), at(chair2, 4.3, 2.35, Math.PI))
  } else {
    // Moving boxes, stacked in the corner away from the doors.
    const box1 = itemFromCatalog(catalogEntry('box'), uid(), { name: 'Moving box', width: 0.5, depth: 0.4, height: 0.4, owned: true })
    const box2 = itemFromCatalog(catalogEntry('box'), uid(), { name: 'Moving box', width: 0.5, depth: 0.4, height: 0.4, owned: true })
    items.push(box1, box2)
    placements.push(at(box1, 4.85, 0.3), at(box2, 4.3, 0.3, 0.2))
  }
  const layoutId = uid()
  return {
    version: 1,
    apartment: { id: uid(), name: 'Sample apartment', units: 'metric' },
    rooms: [living, bedroom, bath],
    items,
    layouts: [{ id: layoutId, name: 'Current', placements }],
    activeLayoutId: layoutId,
  }
}
