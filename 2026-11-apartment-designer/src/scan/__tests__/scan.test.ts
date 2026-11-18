import { describe, expect, it } from 'vitest'
import type { ScanRef, ScanReference } from '../../model/types'
import {
  calibrate,
  detectFloorCeiling,
  detectUpAxis,
  planToScan,
  rescaleAbout,
  scanToPlan,
  snapToWall,
  straightenAngle,
  straightenFromLine,
  toScanData,
  upright,
  uprightPositions,
  wallBandPoints,
  wallUnderlay,
} from '../align'
import { parsePlyHeader, plyKind, readPlyPoints } from '../readPly'
import { SAMPLE_TRUTH, sampleFilePoint, samplePly } from '../sampleScan'

const ply = samplePly(0.5)
const pts = readPlyPoints(ply)!

const baseRef = (): ScanRef => ({
  id: 's',
  name: 'Sample',
  fileName: 'room.ply',
  fileSize: ply.length,
  format: 'splat',
  upAxis: 'y',
  floor: 0,
  scale: 1,
  yaw: 0,
  x: 0,
  y: 0,
  references: [],
  visible: true,
  underlay: 'walls',
  opacity: 0.8,
  sliceLow: 1.0,
  sliceHigh: 2.0,
  cutHeight: 1.6,
})

describe('PLY reader', () => {
  it('reads a binary splat PLY', () => {
    const h = parsePlyHeader(ply)!
    expect(plyKind(h)).toBe('splat')
    expect(pts.count).toBeGreaterThan(30_000)
    expect(pts.colors).toBeDefined()
    // Colors come back from the SH DC terms into 0..1.
    expect(Math.max(...pts.colors!.slice(0, 300))).toBeLessThanOrEqual(1)
    expect(pts.opacity![0]).toBeCloseTo(0.982, 2)
  })
  it('reads ASCII point clouds with byte colors', () => {
    const text = 'ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n1 2 3 255 0 0\n4 5 6 0 255 0\n'
    const p = readPlyPoints(new TextEncoder().encode(text))!
    expect(Array.from(p.positions)).toEqual([1, 2, 3, 4, 5, 6])
    expect(p.colors![0]).toBe(1)
    expect(p.colors![4]).toBe(1)
    expect(plyKind(parsePlyHeader(new TextEncoder().encode(text))!)).toBe('points')
  })
  it('spots compressed splat PLYs', () => {
    const text = 'ply\nformat binary_little_endian 1.0\nelement chunk 1\nproperty float min_x\nelement vertex 1\nproperty uint packed_position\nend_header\n'
    expect(plyKind(parsePlyHeader(new TextEncoder().encode(text))!)).toBe('compressed-splat')
  })
})

describe('orientation', () => {
  it('upright rotations are proper rotations', () => {
    for (const up of ['y', '-y', 'z', '-z'] as const) {
      const [x, y, z] = upright(up, 1, 2, 3)
      expect(Math.hypot(x, y, z)).toBeCloseTo(Math.hypot(1, 2, 3))
    }
    expect(upright('z', 0, 0, 1)).toEqual([0, 1, -0])
    expect(upright('-y', 0, -1, 0)[1]).toBe(1)
  })
  it('finds the up axis, also when the file is Z up or upside down', () => {
    expect(detectUpAxis(pts.positions)).toBe('y')
    const zUp = new Float32Array(pts.positions.length)
    const down = new Float32Array(pts.positions.length)
    for (let i = 0; i < zUp.length; i += 3) {
      // Inverse of upright('z'): file (x, y, z) with y up → (x, -z, y).
      zUp[i] = pts.positions[i]
      zUp[i + 1] = -pts.positions[i + 2]
      zUp[i + 2] = pts.positions[i + 1]
      down[i] = pts.positions[i]
      down[i + 1] = -pts.positions[i + 1]
      down[i + 2] = -pts.positions[i + 2]
    }
    expect(detectUpAxis(zUp)).toBe('z')
    expect(detectUpAxis(down)).toBe('-y')
  })
  it('finds the floor and ceiling', () => {
    const { floor, ceiling } = detectFloorCeiling(uprightPositions(pts.positions, 'y'))
    expect(floor).toBeCloseTo(SAMPLE_TRUTH.floorY, 2)
    expect(ceiling! - floor).toBeCloseTo(SAMPLE_TRUTH.ceiling / SAMPLE_TRUTH.scale, 2)
  })
})

describe('plan transform', () => {
  it('round-trips and matches toScanData', () => {
    const t = { ...baseRef(), scale: 1.07, yaw: 0.4, x: 2, y: -1, floor: 1.3 }
    const p = { x: 0.3, y: -2.2 }
    const q = planToScan(t, scanToPlan(t, p))
    expect(q.x).toBeCloseTo(p.x)
    expect(q.y).toBeCloseTo(p.y)
    const data = toScanData(t, { count: 1, positions: Float32Array.of(p.x, 2.3, p.y) })
    const plan = scanToPlan(t, p)
    expect(data.points.positions[0]).toBeCloseTo(plan.x, 5)
    expect(data.points.positions[1]).toBeCloseTo(1.07, 5) // (2.3 - 1.3) * 1.07
    expect(data.points.positions[2]).toBeCloseTo(plan.y, 5)
    expect(data.upAxis).toBe('y')
    expect(data.metric).toBe(false)
  })
  it('rescales about an anchor without moving it', () => {
    const t = { ...baseRef(), scale: 1, yaw: 0.3, x: 1, y: 1 }
    const anchor = { x: 2.5, y: 0.5 }
    const next = { ...t, ...rescaleAbout(t, 1.2, anchor) }
    const before = planToScan(t, anchor)
    const after = scanToPlan(next, before)
    expect(after.x).toBeCloseTo(anchor.x)
    expect(after.y).toBeCloseTo(anchor.y)
  })
})

describe('straightening', () => {
  it('finds the rotation that squares the walls', () => {
    const ref = { ...baseRef(), floor: SAMPLE_TRUTH.floorY }
    const band = wallBandPoints(uprightPositions(pts.positions, 'y'), ref)
    const a = (straightenAngle(band, 1) * 180) / Math.PI
    // The sample is turned 17°, so undoing it is -17° (up to quarter turns).
    const err = (((a + SAMPLE_TRUTH.yawDeg) % 90) + 135) % 90 - 45
    expect(Math.abs(err)).toBeLessThan(0.3)
  })
  it('straightens from a line', () => {
    const r = straightenFromLine({ x: 0, y: 0 }, { x: 1, y: 0.2 })
    expect(r).toBeCloseTo(-Math.atan2(0.2, 1))
    expect(straightenFromLine({ x: 0, y: 0 }, { x: 0.1, y: 1 })).toBeCloseTo(Math.PI / 2 - Math.atan2(1, 0.1))
  })
})

describe('calibration', () => {
  const mk = (len: number, measured: number): ScanReference => ({ id: String(len), a: { x: 0, y: 0 }, b: { x: measured, y: 0 }, length: len })
  it('takes the median and reports the spread', () => {
    const c = calibrate([mk(5.2, 5.0), mk(4.1, 3.94), mk(1.5, 1.38)])!
    expect(c.scale).toBeCloseTo(4.1 / 3.94)
    expect(c.spread).toBeCloseTo(1.5 / 1.38 / (4.1 / 3.94) - 1, 5)
    expect(calibrate([])).toBeNull()
  })
  it('recovers the sample scale from the tape-measured references', () => {
    const refs = SAMPLE_TRUTH.references.map((r, i) => ({ id: String(i), a: sampleFilePoint(r.a), b: sampleFilePoint(r.b), length: r.length }))
    const c = calibrate(refs)!
    expect(c.scale).toBeCloseTo(SAMPLE_TRUTH.scale, 6)
    expect(c.spread).toBeLessThan(1e-6)
  })
})

describe('underlay', () => {
  it('draws walls dark and snaps to them', () => {
    const ref = { ...baseRef(), floor: SAMPLE_TRUTH.floorY }
    const up = uprightPositions(pts.positions, 'y')
    const u = wallUnderlay(up, ref)
    expect(u.width).toBeGreaterThan(100)
    // A point a few cm off the long wall snaps onto it.
    const a = sampleFilePoint({ x: 2.6, y: 0 })
    const near = sampleFilePoint({ x: 2.6, y: 0.06 })
    const s = snapToWall(u, near, 0.12)!
    expect(Math.hypot(s.x - a.x, s.y - a.y)).toBeLessThan(0.03)
    // The middle of the room is empty in the wall band.
    expect(snapToWall(u, sampleFilePoint({ x: 2.5, y: 2.0 }), 0.05)).toBeNull()
  })
})
