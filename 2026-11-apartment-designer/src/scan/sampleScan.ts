// A synthetic Gaussian splat scan of an L-shaped living room, written as a standard 3DGS
// binary PLY (the format Scaniverse exports). It lets you try importing, calibrating and
// tracing without a real scan, and the tests use it as ground truth.
//
// The scan is deliberately "imperfect" the way a phone scan without LiDAR is: about 4% too
// small, rotated 17° on the floor, and floating 1.3 m above the origin.

export const SAMPLE_TRUTH = {
  /** File units are this many meters (the calibration should find it). */
  scale: 1 / 0.96,
  /** Degrees the scan is turned on the floor. */
  yawDeg: 17,
  floorY: 1.3,
  ceiling: 2.55,
  /** Interior outline in meters, before the distortion. */
  outline: [
    { x: 0, y: 0 },
    { x: 5.2, y: 0 },
    { x: 5.2, y: 2.6 },
    { x: 3.4, y: 2.6 },
    { x: 3.4, y: 4.1 },
    { x: 0, y: 4.1 },
  ],
  /** Tape-measured lengths you'd take in this room. */
  references: [
    { label: 'Long wall', a: { x: 0, y: 0 }, b: { x: 5.2, y: 0 }, length: 5.2 },
    { label: 'Left wall', a: { x: 0, y: 0 }, b: { x: 0, y: 4.1 }, length: 4.1 },
    { label: 'Short wall', a: { x: 3.4, y: 2.6 }, b: { x: 3.4, y: 4.1 }, length: 1.5 },
  ],
}

const PROPS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
const SH_C0 = 0.28209479177387814

/** Deterministic random numbers so the sample (and the tests) never change. */
function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

type RGB = [number, number, number]

/** Points of the sample room in true meters (Y up, floor at 0) with colors. */
export function sampleRoomPoints(density = 1): { positions: Float32Array; colors: Float32Array } {
  const rand = rng(7)
  const pos: number[] = []
  const col: number[] = []
  const T = SAMPLE_TRUTH
  const H = T.ceiling
  const push = (x: number, y: number, z: number, c: RGB, jitter = 0.06) => {
    pos.push(x + (rand() - 0.5) * 0.006, y + (rand() - 0.5) * 0.006, z + (rand() - 0.5) * 0.006)
    const k = 1 + (rand() - 0.5) * jitter
    col.push(Math.min(1, c[0] * k), Math.min(1, c[1] * k), Math.min(1, c[2] * k))
  }
  const area = (n: number) => Math.round(n * density)
  const outline = T.outline
  // Floor: wooden planks, with the L shape.
  const inside = (x: number, z: number) => (x >= 0 && x <= 5.2 && z >= 0 && z <= 2.6) || (x >= 0 && x <= 3.4 && z >= 0 && z <= 4.1)
  for (let i = 0; i < area(60_000); i++) {
    const x = rand() * 5.2
    const z = rand() * 4.1
    if (!inside(x, z)) continue
    const plank = Math.floor(z / 0.18) % 2 ? 0.92 : 1
    push(x, 0, z, [0.62 * plank, 0.45 * plank, 0.3 * plank])
  }
  // Walls, with a door gap on the left wall and a window on the long wall.
  for (let w = 0; w < outline.length; w++) {
    const a = outline[w]
    const b = outline[(w + 1) % outline.length]
    const L = Math.hypot(b.x - a.x, b.y - a.y)
    for (let i = 0; i < area(L * 4500); i++) {
      const t = rand()
      const y = rand() * H
      const x = a.x + (b.x - a.x) * t
      const z = a.y + (b.y - a.y) * t
      const along = t * L
      if (w === 5 && along > 0.6 && along < 1.5 && y < 2.05) continue // door
      const isWindow = w === 0 && along > 1.4 && along < 3.0 && y > 0.85 && y < 2.2
      push(x, y, z, isWindow ? [0.55, 0.68, 0.82] : [0.9, 0.88, 0.84])
    }
  }
  // Ceiling: a phone scan sees little of it.
  for (let i = 0; i < area(9000); i++) {
    const x = rand() * 5.2
    const z = rand() * 4.1
    if (inside(x, z)) push(x, H, z, [0.95, 0.95, 0.94])
  }
  // Furniture: a sofa against the long wall, a coffee table, a rug, a bookcase.
  const box = (x0: number, z0: number, x1: number, z1: number, h: number, c: RGB, n: number) => {
    const sides = [x1 - x0, z1 - z0, x1 - x0, z1 - z0]
    const per = sides.reduce((s, v) => s + v, 0)
    for (let i = 0; i < area(n); i++) {
      if (rand() < 0.5) push(x0 + rand() * (x1 - x0), h, z0 + rand() * (z1 - z0), c)
      else {
        let d = rand() * per
        const y = rand() * h
        if (d < sides[0]) push(x0 + d, y, z0, c)
        else if ((d -= sides[0]) < sides[1]) push(x1, y, z0 + d, c)
        else if ((d -= sides[1]) < sides[2]) push(x1 - d, y, z1, c)
        else push(x0, y, z1 - (d - sides[2]), c)
      }
    }
  }
  box(1.2, 0.05, 3.4, 0.95, 0.85, [0.25, 0.36, 0.55], 9000)
  box(1.7, 1.45, 2.9, 2.05, 0.42, [0.45, 0.3, 0.2], 3000)
  box(1.0, 1.1, 3.6, 2.6, 0.01, [0.78, 0.7, 0.55], 2500)
  box(0.05, 2.4, 0.4, 3.6, 1.9, [0.35, 0.25, 0.18], 4000)
  return { positions: Float32Array.from(pos), colors: Float32Array.from(col) }
}

/** The distortion applied to the sample: true meters → file coordinates. */
export function distortSample(x: number, y: number, z: number): [number, number, number] {
  const T = SAMPLE_TRUTH
  const s = 1 / T.scale
  const a = (T.yawDeg * Math.PI) / 180
  // Rotate on the floor (plan rotation, same sense as the app's yaw), shrink, lift.
  const c = Math.cos(a)
  const sn = Math.sin(a)
  const px = c * x - sn * z
  const pz = sn * x + c * z
  return [s * px - 1.7, s * y + T.floorY, s * pz - 0.4]
}

/** Inverse of the planar part: true plan point → file top-down point. */
export function sampleFilePoint(p: { x: number; y: number }) {
  const [x, , z] = distortSample(p.x, 0, p.y)
  return { x, y: z }
}

/** The sample as a binary little-endian 3DGS PLY file. */
export function samplePly(density = 1): Uint8Array {
  const { positions, colors } = sampleRoomPoints(density)
  const n = positions.length / 3
  const header =
    'ply\nformat binary_little_endian 1.0\ncomment synthetic room for apartment designer\n' +
    `element vertex ${n}\n` +
    PROPS.map((p) => `property float ${p}\n`).join('') +
    'end_header\n'
  const head = new TextEncoder().encode(header)
  const stride = PROPS.length * 4
  const buf = new Uint8Array(head.length + n * stride)
  buf.set(head, 0)
  const dv = new DataView(buf.buffer, head.length)
  const logScale = Math.log(0.018 / SAMPLE_TRUTH.scale)
  for (let i = 0; i < n; i++) {
    const [x, y, z] = distortSample(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
    const vals = [
      x,
      y,
      z,
      0,
      0,
      0,
      (colors[i * 3] - 0.5) / SH_C0,
      (colors[i * 3 + 1] - 0.5) / SH_C0,
      (colors[i * 3 + 2] - 0.5) / SH_C0,
      4, // opacity logit: ~0.98
      logScale,
      logScale,
      logScale,
      1,
      0,
      0,
      0,
    ]
    for (let k = 0; k < vals.length; k++) dv.setFloat32(i * stride + k * 4, vals[k], true)
  }
  return buf
}
