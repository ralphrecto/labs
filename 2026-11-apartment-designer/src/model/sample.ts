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

/** A small one-bedroom so the app has something to play with on first load. */
export function sampleDoc(): Doc {
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
  const items = [
    itemFromCatalog(catalogEntry('sofa'), uid(), { owned: true }),
    itemFromCatalog(catalogEntry('coffee-table'), uid(), { owned: true }),
    itemFromCatalog(catalogEntry('tv-stand'), uid(), { owned: true }),
    itemFromCatalog(catalogEntry('rug'), uid(), { owned: true }),
    itemFromCatalog(catalogEntry('dining-table'), uid(), { width: 1.2, depth: 0.8, owned: true }),
    itemFromCatalog(catalogEntry('dining-chair'), uid(), { owned: true }),
    itemFromCatalog(catalogEntry('dining-chair'), uid(), { owned: true }),
    itemFromCatalog(catalogEntry('bed'), uid(), { name: 'Queen bed', owned: true, clearance: { front: 0.6, back: 0, left: 0.6, right: 0 } }),
    itemFromCatalog(catalogEntry('nightstand'), uid(), { owned: true }),
    itemFromCatalog(catalogEntry('dresser'), uid(), { owned: true }),
  ]
  const [sofa, coffee, tv, rug, table, chair1, chair2, bed, night, dresser] = items
  const layoutId = uid()
  const hd = (i: typeof sofa) => i.depth / 2
  return {
    version: 1,
    apartment: { id: uid(), name: 'Sample apartment', units: 'metric' },
    rooms: [living, bedroom, bath],
    items,
    layouts: [
      {
        id: layoutId,
        name: 'Current',
        placements: [
          { id: uid(), itemId: rug.id, x: 2.0, y: 2.0, rotation: 0 },
          { id: uid(), itemId: sofa.id, x: 2.0, y: 4.2 - hd(sofa), rotation: Math.PI },
          { id: uid(), itemId: coffee.id, x: 2.0, y: 2.45, rotation: 0 },
          { id: uid(), itemId: tv.id, x: 2.0, y: hd(tv), rotation: 0 },
          { id: uid(), itemId: table.id, x: 4.3, y: 1.2, rotation: Math.PI / 2 },
          { id: uid(), itemId: chair1.id, x: 3.55, y: 1.2, rotation: -Math.PI / 2 },
          { id: uid(), itemId: chair2.id, x: 4.3, y: 2.35, rotation: Math.PI },
          { id: uid(), itemId: bed.id, x: 5.32 + 1.8, y: hd(bed), rotation: 0 },
          { id: uid(), itemId: night.id, x: 5.32 + 1.8 - 0.76 - 0.3, y: hd(night), rotation: 0 },
          { id: uid(), itemId: dresser.id, x: 5.32 + 3.6 - hd(dresser), y: 2.75, rotation: Math.PI / 2 },
        ],
      },
    ],
    activeLayoutId: layoutId,
  }
}
