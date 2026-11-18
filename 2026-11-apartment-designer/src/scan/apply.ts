import { catalogEntry, itemFromCatalog } from '../model/catalog'
import { bounds, centroid } from '../model/geometry'
import { uid } from '../model/sample'
import type { Doc, Item, Opening, Placement, Room, Vec2 } from '../model/types'
import type { DetectedObject, SceneEstimate } from './estimate'

// Turns a reviewed SceneEstimate into ordinary rooms, items and placements. After this, scanned
// furniture is regular furniture: it moves, rotates, resizes and gets checked like anything else.

export interface ApplyChoice {
  /** Add the detected room outline (with its doors and windows). */
  includeRoom: boolean
  /** Put the scanned room in place of an existing one (keeps its name and id). */
  replaceRoomId?: string
  roomName?: string
  /** The objects to add, as edited in review. */
  objects: DetectedObject[]
}

export interface ApplyResult {
  doc: Doc
  roomId?: string
  placementIds: string[]
}

const GAP = 0.24

/** How far the scan's coordinates move to land on the plan (also used to preview before applying). */
export function placementShift(doc: Doc, est: SceneEstimate, choice: ApplyChoice): Vec2 | null {
  const room = choice.includeRoom ? est.room : null
  const target = choice.replaceRoomId ? doc.rooms.find((r) => r.id === choice.replaceRoomId) : undefined
  const refPts: Vec2[] = room ? room.polygon : choice.objects.flatMap((o) => [{ x: o.x - o.width / 2, y: o.y - o.depth / 2 }, { x: o.x + o.width / 2, y: o.y + o.depth / 2 }])
  if (!refPts.length) return null
  const rb = bounds(refPts)
  const others = doc.rooms.filter((r) => r.id !== target?.id)
  if (target) {
    const tb = bounds(target.polygon)
    return { x: (tb.minX + tb.maxX) / 2 - (rb.minX + rb.maxX) / 2, y: (tb.minY + tb.maxY) / 2 - (rb.minY + rb.maxY) / 2 }
  }
  if (!room && others.length) {
    // Furniture only: drop it in the middle of the first room.
    const c = centroid(others[0].polygon)
    return { x: c.x - (rb.minX + rb.maxX) / 2, y: c.y - (rb.minY + rb.maxY) / 2 }
  }
  if (others.length) {
    // Next to the existing rooms, like a room added by size.
    const ob = bounds(others.flatMap((r) => r.polygon))
    return { x: ob.maxX + GAP - rb.minX, y: ob.minY - rb.minY }
  }
  return { x: -rb.minX, y: -rb.minY }
}

export function applyEstimate(doc: Doc, est: SceneEstimate, choice: ApplyChoice): ApplyResult {
  const room = choice.includeRoom ? est.room : null
  const target = choice.replaceRoomId ? doc.rooms.find((r) => r.id === choice.replaceRoomId) : undefined
  const delta = placementShift(doc, est, choice)
  if (!delta) return { doc, placementIds: [] }
  const { x: dx, y: dy } = delta
  const shift = (p: Vec2): Vec2 => ({ x: r3(p.x + dx), y: r3(p.y + dy) })

  let rooms = doc.rooms
  let roomId: string | undefined
  if (room) {
    roomId = target?.id ?? uid()
    const openings: Opening[] = room.openings.map((o) => ({
      id: uid(),
      kind: o.kind,
      wall: o.wall,
      offset: o.offset,
      width: o.width,
      height: o.height,
      sill: o.sill,
      ...(o.kind === 'door' ? { hinge: 'start' as const, swing: 'in' as const } : {}),
    }))
    const newRoom: Room = {
      id: roomId,
      name: target?.name ?? choice.roomName ?? 'Scanned room',
      polygon: room.polygon.map(shift),
      ceilingHeight: room.ceilingHeight,
      wallThickness: target?.wallThickness ?? 0.12,
      openings,
      scan: { scanId: est.scanId, x: r3(est.transform.x + dx), y: r3(est.transform.y + dy), rotation: est.transform.rotation },
    }
    rooms = target ? doc.rooms.map((r) => (r.id === target.id ? newRoom : r)) : [...doc.rooms, newRoom]
  }

  const items: Item[] = []
  const placements: Placement[] = []
  for (const o of choice.objects) {
    const entry = catalogEntry(o.category)
    const item = itemFromCatalog(entry, uid(), {
      name: o.name,
      width: o.width,
      depth: o.depth,
      height: o.height,
      color: o.color ?? entry.color,
      source: 'scan',
      owned: true,
      scanId: est.scanId,
      confidence: o.confidence,
    })
    items.push(item)
    const c = shift({ x: o.x, y: o.y })
    placements.push({ id: uid(), itemId: item.id, x: c.x, y: c.y, rotation: o.rotation })
  }
  const next: Doc = {
    ...doc,
    rooms,
    items: [...doc.items, ...items],
    layouts: doc.layouts.map((l) => (l.id === doc.activeLayoutId ? { ...l, placements: [...l.placements, ...placements] } : l)),
  }
  return { doc: next, roomId, placementIds: placements.map((p) => p.id) }
}

const r3 = (x: number) => Math.round(x * 1000) / 1000
