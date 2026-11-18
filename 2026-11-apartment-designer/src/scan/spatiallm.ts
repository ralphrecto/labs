import { closestPointOnSegment, dist, norm, polygonArea, signedArea, sub, v } from '../model/geometry'
import type { Vec2 } from '../model/types'
import { classifyBox, displayName } from './classify'
import { orient } from './analyze'
import type { DetectedObject, DetectedOpening, SceneEstimate } from './estimate'

// Reads the text layout that SpatialLM (a point-cloud-to-layout model meant to run on a GPU
// server) prints, and turns it into the same SceneEstimate the on-device detector makes:
//
//   wall_0=Wall(ax,ay,az,bx,by,bz,height,thickness)
//   door_0=Door(wall_0,x,y,z,width,height)
//   window_0=Window(wall_1,x,y,z,width,height)
//   bbox_0=Bbox(bed,x,y,z,angle_z,scale_x,scale_y,scale_z)
//
// SpatialLM works Z up, with box positions at the box center. World (x, y, z) maps to the plan
// as (x, -y), so its counterclockwise angle_z becomes a clockwise plan rotation of -angle_z.

/** SpatialLM class names mapped to catalog categories. Wall-mounted things (paintings, TVs) are left out. */
const CLASS_MAP: Record<string, string> = {
  bed: 'bed',
  sofa: 'sofa',
  couch: 'sofa',
  chair: 'dining-chair',
  dining_chair: 'dining-chair',
  armchair: 'armchair',
  office_chair: 'office-chair',
  stool: 'dining-chair',
  table: 'dining-table',
  dining_table: 'dining-table',
  coffee_table: 'coffee-table',
  side_table: 'nightstand',
  nightstand: 'nightstand',
  desk: 'desk',
  cabinet: 'dresser',
  dresser: 'dresser',
  chest_of_drawers: 'dresser',
  wardrobe: 'wardrobe',
  tv_cabinet: 'tv-stand',
  tv_stand: 'tv-stand',
  bookcase: 'bookshelf',
  bookshelf: 'bookshelf',
  shelf: 'bookshelf',
  shelving: 'bookshelf',
  plant: 'plant',
  lamp: 'floor-lamp',
  floor_lamp: 'floor-lamp',
  rug: 'rug',
  carpet: 'rug',
}
const SKIP = new Set(['painting', 'picture', 'tv', 'television', 'curtain', 'mirror', 'clock', 'light', 'chandelier', 'air_conditioner', 'window', 'door'])

const num = (s: string) => Number(s.trim())

export function parseSpatialLM(text: string, scanId = 'spatiallm'): SceneEstimate {
  const walls: { id: string; a: Vec2; b: Vec2; height: number }[] = []
  const doors: { kind: 'door' | 'window'; wall: string; c: Vec2; z: number; width: number; height: number }[] = []
  const boxes: { cls: string; c: Vec2; z: number; angle: number; sx: number; sy: number; sz: number }[] = []
  const notes: string[] = []
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim()
    const m = line.match(/^(\w+)\s*=\s*(Wall|Door|Window|Bbox)\((.*)\)\s*$/)
    if (!m) continue
    const [, id, kind, argStr] = m
    const args = argStr.split(',').map((s) => s.trim())
    if (kind === 'Wall' && args.length >= 7) {
      const [ax, ay, , bx, by, , h] = args.map(num)
      walls.push({ id, a: v(ax, -ay), b: v(bx, -by), height: h })
    } else if ((kind === 'Door' || kind === 'Window') && args.length >= 6) {
      const [x, y, z, w, h] = args.slice(1).map(num)
      doors.push({ kind: kind === 'Door' ? 'door' : 'window', wall: args[0], c: v(x, -y), z, width: w, height: h })
    } else if (kind === 'Bbox' && args.length >= 8) {
      const [x, y, z, angle, sx, sy, sz] = args.slice(1).map(num)
      boxes.push({ cls: args[0].toLowerCase(), c: v(x, -y), z, angle, sx, sy, sz })
    }
  }

  // Chain walls into a loop by matching endpoints.
  let polygon: Vec2[] | null = null
  if (walls.length >= 3) {
    const pts = chainWalls(walls.map((w) => [w.a, w.b] as [Vec2, Vec2]))
    if (pts && polygonArea(pts) > 1) polygon = signedArea(pts) < 0 ? pts.reverse() : pts
    else notes.push("The walls don't form a closed room; check the outline.")
  }
  const ceiling = walls.length ? walls.reduce((s, w) => s + w.height, 0) / walls.length : 2.6

  const openings: DetectedOpening[] = []
  if (polygon) {
    for (const d of doors) {
      // Which outline edge is nearest the opening's center.
      let best = -1
      let bestD = Infinity
      let along = 0
      for (let k = 0; k < polygon.length; k++) {
        const a = polygon[k]
        const b = polygon[(k + 1) % polygon.length]
        const r = closestPointOnSegment(d.c, a, b)
        if (r.distance < bestD) [best, bestD, along] = [k, r.distance, r.t * dist(a, b)]
      }
      if (best < 0 || bestD > 0.4) continue
      openings.push({
        kind: d.kind,
        wall: best,
        offset: Math.max(0, along - d.width / 2),
        width: d.width,
        height: d.height,
        sill: d.kind === 'window' ? Math.max(0, d.z - d.height / 2) : 0,
        confidence: 0.8,
      })
    }
  }

  const objects: DetectedObject[] = []
  let n = 0
  for (const b of boxes) {
    if (SKIP.has(b.cls)) continue
    const mapped = CLASS_MAP[b.cls]
    const guess = classifyBox(b.sx, b.sy, b.sz)
    const category = mapped ?? guess.category
    // Orientation: the model gives the box's axes but not which side is the front, so use the walls.
    const placed = orient({ center: b.c, angle: -b.angle, size: v(b.sx, b.sy) }, category, polygon)
    objects.push({
      id: `det-${++n}`,
      category,
      name: displayName(category, placed.width, placed.depth),
      confidence: mapped ? 0.8 : guess.confidence,
      alternatives: guess.alternatives,
      x: b.c.x,
      y: b.c.y,
      rotation: placed.rotation,
      width: placed.width,
      depth: placed.depth,
      height: b.sz,
      pointCount: 0,
      againstWall: placed.againstWall,
    })
  }
  if (!walls.length && !boxes.length) notes.push("Didn't find any Wall(...) or Bbox(...) lines in that text.")

  return {
    scanId,
    detector: 'spatiallm',
    // SpatialLM's frame is the scan's frame with y flipped; for a Y-up scan that's world (x, z) unchanged.
    transform: { x: 0, y: 0, rotation: 0 },
    room: polygon ? { polygon, ceilingHeight: Math.round(ceiling * 100) / 100, openings, confidence: 0.8 } : null,
    objects,
    notes,
  }
}

/** Order wall segments into a closed loop of corner points. */
function chainWalls(segs: [Vec2, Vec2][]): Vec2[] | null {
  const left = segs.slice(1)
  const loop: Vec2[] = [segs[0][0], segs[0][1]]
  while (left.length) {
    const end = loop[loop.length - 1]
    let bi = -1
    let flip = false
    let bd = Infinity
    left.forEach(([a, b], i) => {
      const da = dist(a, end)
      const db = dist(b, end)
      if (da < bd) [bi, bd, flip] = [i, da, false]
      if (db < bd) [bi, bd, flip] = [i, db, true]
    })
    if (bd > 0.3) return null
    const [a, b] = left.splice(bi, 1)[0]
    loop.push(flip ? a : b)
  }
  if (dist(loop[0], loop[loop.length - 1]) > 0.3) return null
  loop.pop()
  // Drop corners where consecutive walls are collinear.
  return loop.filter((p, i) => {
    const a = loop[(i - 1 + loop.length) % loop.length]
    const b = loop[(i + 1) % loop.length]
    const d1 = norm(sub(p, a))
    const d2 = norm(sub(b, p))
    return Math.abs(d1.x * d2.y - d1.y * d2.x) > 0.02
  })
}
