import { pointInPolygon, rot, v } from '../model/geometry'
import type { Vec2 } from '../model/types'
import { rng } from './cloud'
import type { ScanData } from './types'

// A made-up scan of an L-shaped studio, for trying detection without a phone scan and for tests.
// Points are sampled on the visible surfaces with a little noise, then the whole scan is turned,
// tilted and shifted the way a phone scan comes out, so detection has to undo that.

export interface TruthBox {
  category: string
  x: number
  y: number
  rotation: number
  width: number
  depth: number
  height: number
  color: [number, number, number]
}

export interface TruthOpening {
  kind: 'door' | 'window'
  wall: number
  offset: number
  width: number
  height: number
  sill: number
}

export const STUDIO = {
  polygon: [v(0, 0), v(5, 0), v(5, 4.2), v(2, 4.2), v(2, 3), v(0, 3)] as Vec2[],
  ceiling: 2.6,
  openings: [
    { kind: 'window', wall: 0, offset: 1.0, width: 1.6, height: 1.3, sill: 0.85 },
    { kind: 'door', wall: 1, offset: 3.0, width: 0.85, height: 2.03, sill: 0 },
  ] as TruthOpening[],
  boxes: [
    { category: 'bed', x: 1.015, y: 1.45, rotation: (3 * Math.PI) / 2, width: 1.52, depth: 2.03, height: 0.55, color: [0.85, 0.83, 0.78] },
    { category: 'nightstand', x: 0.2, y: 0.34, rotation: (3 * Math.PI) / 2, width: 0.45, depth: 0.4, height: 0.58, color: [0.55, 0.42, 0.3] },
    { category: 'nightstand', x: 0.2, y: 2.56, rotation: (3 * Math.PI) / 2, width: 0.45, depth: 0.4, height: 0.58, color: [0.55, 0.42, 0.3] },
    { category: 'wardrobe', x: 4.2, y: 0.3, rotation: 0, width: 1.2, depth: 0.6, height: 2.0, color: [0.75, 0.68, 0.58] },
    { category: 'desk', x: 4.7, y: 1.4, rotation: Math.PI / 2, width: 1.2, depth: 0.6, height: 0.75, color: [0.82, 0.77, 0.68] },
    { category: 'office-chair', x: 3.9, y: 1.4, rotation: Math.PI / 2, width: 0.62, depth: 0.62, height: 1.05, color: [0.2, 0.2, 0.22] },
    { category: 'sofa', x: 3.4, y: 3.75, rotation: Math.PI, width: 2.0, depth: 0.9, height: 0.85, color: [0.45, 0.52, 0.62] },
    { category: 'coffee-table', x: 3.4, y: 2.85, rotation: 0, width: 1.0, depth: 0.55, height: 0.42, color: [0.5, 0.38, 0.27] },
    { category: 'bookshelf', x: 2.15, y: 3.6, rotation: (3 * Math.PI) / 2, width: 0.8, depth: 0.3, height: 1.8, color: [0.68, 0.54, 0.4] },
    { category: 'floor-lamp', x: 4.75, y: 4.0, rotation: 0, width: 0.35, depth: 0.35, height: 1.6, color: [0.9, 0.86, 0.75] },
  ] as TruthBox[],
}

export interface SyntheticOptions {
  /** Points per square meter of surface. */
  density?: number
  noise?: number
  /** How the phone's frame differs from the room: yaw (radians), tilt (degrees), and offset. */
  yaw?: number
  tilt?: number
  offset?: [number, number, number]
  upAxis?: 'y' | 'z'
  seed?: number
}

export function syntheticScan(opts: SyntheticOptions = {}): ScanData {
  const { density = 1500, noise = 0.006, yaw = 0.4, tilt = 1.5, offset = [1.3, -1.45, -2.1], upAxis = 'y', seed = 42 } = opts
  const rand = rng(seed)
  const gauss = () => {
    const u = Math.max(1e-9, rand())
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
  }
  const P: number[] = []
  const C: number[] = []
  // Plan point (x, y) at height h -> world (x, h, y) before the phone transform.
  const put = (x: number, y: number, h: number, c: [number, number, number]) => {
    P.push(x + noise * gauss(), h + noise * gauss(), y + noise * gauss())
    const j = 0.04 * (rand() - 0.5)
    C.push(c[0] + j, c[1] + j, c[2] + j)
  }
  const area = (a: number) => Math.round(a * density)
  const { polygon, ceiling, openings, boxes } = STUDIO
  const footprints = boxes.map((b) => [v(-b.width / 2, -b.depth / 2), v(b.width / 2, -b.depth / 2), v(b.width / 2, b.depth / 2), v(-b.width / 2, b.depth / 2)].map((p) => {
    const r = rot(p, b.rotation)
    return v(r.x + b.x, r.y + b.y)
  }))
  const covered = (p: Vec2) => footprints.some((f) => pointInPolygon(p, f))

  // Floor (not under furniture) and a patchy ceiling.
  const [minX, maxX, minY, maxY] = [0, 5, 0, 4.2]
  const fa = (maxX - minX) * (maxY - minY)
  for (let i = 0; i < area(fa); i++) {
    const p = v(minX + rand() * (maxX - minX), minY + rand() * (maxY - minY))
    if (!pointInPolygon(p, polygon)) continue
    if (!covered(p)) put(p.x, p.y, 0, [0.62, 0.48, 0.34])
    if (rand() < 0.5) put(p.x, p.y, ceiling, [0.95, 0.95, 0.94])
  }
  // Walls with their openings.
  const n = polygon.length
  for (let k = 0; k < n; k++) {
    const a = polygon[k]
    const b = polygon[(k + 1) % n]
    const L = Math.hypot(b.x - a.x, b.y - a.y)
    const ops = openings.filter((o) => o.wall === k)
    for (let i = 0; i < area(L * ceiling); i++) {
      const t = rand() * L
      const h = rand() * ceiling
      if (ops.some((o) => t > o.offset && t < o.offset + o.width && h > o.sill && h < o.sill + o.height)) continue
      put(a.x + ((b.x - a.x) * t) / L, a.y + ((b.y - a.y) * t) / L, h, [0.9, 0.9, 0.88])
    }
  }
  // The hallway seen through the door: some floor and the far wall.
  for (let i = 0; i < area(1.3 * 1.6); i++) put(5.12 + rand() * 1.3, 2.6 + rand() * 1.6, 0, [0.6, 0.6, 0.6])
  for (let i = 0; i < area(1.6 * 2.4); i++) put(6.42, 2.6 + rand() * 1.6, rand() * 2.4, [0.85, 0.85, 0.85])
  // Furniture: top and four sides.
  boxes.forEach((b, bi) => {
    const f = footprints[bi]
    for (let i = 0; i < area(b.width * b.depth); i++) {
      const local = v((rand() - 0.5) * b.width, (rand() - 0.5) * b.depth)
      const r = rot(local, b.rotation)
      put(b.x + r.x, b.y + r.y, b.height, b.color)
    }
    for (let s = 0; s < 4; s++) {
      const a = f[s]
      const c = f[(s + 1) % 4]
      const L = Math.hypot(c.x - a.x, c.y - a.y)
      for (let i = 0; i < area(L * b.height); i++) {
        const t = rand()
        put(a.x + (c.x - a.x) * t, a.y + (c.y - a.y) * t, rand() * b.height, b.color)
      }
    }
  })

  // Phone frame: tilt about x, yaw about the vertical, then shift.
  const count = P.length / 3
  const positions = new Float32Array(count * 3)
  const ct = Math.cos((tilt * Math.PI) / 180)
  const st = Math.sin((tilt * Math.PI) / 180)
  const cy = Math.cos(yaw)
  const sy = Math.sin(yaw)
  for (let i = 0; i < count; i++) {
    let x = P[i * 3]
    let y = P[i * 3 + 1]
    let z = P[i * 3 + 2]
    ;[y, z] = [y * ct - z * st, y * st + z * ct]
    ;[x, z] = [x * cy - z * sy, x * sy + z * cy]
    x += offset[0]
    y += offset[1]
    z += offset[2]
    if (upAxis === 'z') {
      // Y up (x, y, z) -> Z up (x, -z, y)
      positions.set([x, -z, y], i * 3)
    } else positions.set([x, y, z], i * 3)
  }
  return {
    id: `synthetic-${seed}`,
    name: 'Sample studio scan',
    source: 'synthetic',
    points: { count, positions, colors: new Float32Array(C) },
    upAxis,
    metric: true,
  }
}
