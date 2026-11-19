import type { ScanData } from './types'

// Point cloud helpers: convert a scan to the app frame (Y up), downsample, and level it so the
// floor is the plane y = 0. Scans from the import code are already leveled; this is a cheap
// safety net that changes nothing when the floor is already flat at zero.

/** A point cloud in the app world frame: x, height above floor, z. Plan (x, y) = world (x, z). */
export interface Cloud {
  n: number
  x: Float32Array
  h: Float32Array
  z: Float32Array
  /** rgb 0..1, interleaved, optional. */
  rgb?: Float32Array
}

/** Deterministic pseudo-random numbers, so detection is repeatable. */
export function rng(seed = 1) {
  let s = seed >>> 0 || 1
  return () => {
    s ^= s << 13
    s >>>= 0
    s ^= s >>> 17
    s ^= s << 5
    s >>>= 0
    return s / 4294967296
  }
}

/** Convert to Y up and keep at most `maxPoints` points (evenly spread through the array). */
export function toCloud(scan: ScanData, maxPoints = 600_000): Cloud {
  const { count, positions, colors } = scan.points
  const step = Math.max(1, count / maxPoints)
  const n = Math.min(count, Math.floor(count / step))
  const x = new Float32Array(n)
  const h = new Float32Array(n)
  const z = new Float32Array(n)
  const rgb = colors ? new Float32Array(n * 3) : undefined
  const zUp = scan.upAxis === 'z'
  for (let i = 0; i < n; i++) {
    const j = Math.floor(i * step)
    const px = positions[j * 3]
    const py = positions[j * 3 + 1]
    const pz = positions[j * 3 + 2]
    x[i] = px
    // Z up (x, y, z) -> Y up (x, z, -y), keeping the frame right-handed.
    h[i] = zUp ? pz : py
    z[i] = zUp ? -py : pz
    if (rgb && colors) {
      rgb[i * 3] = colors[j * 3]
      rgb[i * 3 + 1] = colors[j * 3 + 1]
      rgb[i * 3 + 2] = colors[j * 3 + 2]
    }
  }
  return { n, x, h, z, rgb }
}

export function quantile(values: ArrayLike<number>, q: number) {
  const a = Array.from(values).sort((p, r) => p - r)
  if (!a.length) return NaN
  const i = Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))))
  return a[i]
}

export interface LevelResult {
  cloud: Cloud
  /** Tilt that was removed, in degrees. */
  tilt: number
  /** Height of the original floor that was moved to zero. */
  floorShift: number
}

/**
 * Find the floor (the lowest large horizontal plane) with RANSAC and rotate and shift the cloud
 * so it lies at y = 0.
 */
export function levelFloor(cloud: Cloud, seed = 7): LevelResult {
  const rand = rng(seed)
  const { n, x, h, z } = cloud
  if (n < 50) return { cloud, tilt: 0, floorShift: 0 }
  // Candidates: the lowest 35% of points, subsampled.
  const cut = quantile(sample(h, 20_000, rand), 0.35)
  const idx: number[] = []
  for (let i = 0; i < n; i++) if (h[i] <= cut) idx.push(i)
  const cand = idx.length > 6000 ? Array.from({ length: 6000 }, () => idx[Math.floor(rand() * idx.length)]) : idx
  const maxTilt = Math.cos((20 * Math.PI) / 180)
  const planes: { nx: number; ny: number; nz: number; d: number; inliers: number; mean: number }[] = []
  for (let it = 0; it < 250; it++) {
    const a = cand[Math.floor(rand() * cand.length)]
    const b = cand[Math.floor(rand() * cand.length)]
    const c = cand[Math.floor(rand() * cand.length)]
    const ux = x[b] - x[a], uy = h[b] - h[a], uz = z[b] - z[a]
    const vx = x[c] - x[a], vy = h[c] - h[a], vz = z[c] - z[a]
    let nx = uy * vz - uz * vy
    let ny = uz * vx - ux * vz
    let nz = ux * vy - uy * vx
    const l = Math.hypot(nx, ny, nz)
    if (l < 1e-6) continue
    nx /= l
    ny /= l
    nz /= l
    if (ny < 0) {
      nx = -nx
      ny = -ny
      nz = -nz
    }
    if (ny < maxTilt) continue
    const d = -(nx * x[a] + ny * h[a] + nz * z[a])
    let inliers = 0
    let sum = 0
    for (const i of cand) {
      const e = nx * x[i] + ny * h[i] + nz * z[i] + d
      if (Math.abs(e) < 0.02) {
        inliers++
        sum += h[i]
      }
    }
    if (inliers > 0) planes.push({ nx, ny, nz, d, inliers, mean: sum / inliers })
  }
  if (!planes.length) return { cloud, tilt: 0, floorShift: 0 }
  const best = planes.reduce((p, q) => (q.inliers > p.inliers ? q : p))
  // A big furniture top (a bed) can beat a half-hidden floor; prefer the lowest strong plane.
  const rough = planes.filter((p) => p.inliers >= best.inliers * 0.5).reduce((p, q) => (q.mean < p.mean ? q : p))
  const floor = refinePlane(cloud, idx, rough)

  // Rotation taking the floor normal to +Y (Rodrigues).
  const tilt = (Math.acos(Math.min(1, floor.ny)) * 180) / Math.PI
  let out = cloud
  if (tilt > 0.3) {
    const kx = -floor.nz, kz = floor.nx // axis = n × Y, unnormalized
    const s = Math.hypot(kx, kz)
    const ax = kx / s, az = kz / s
    const cos = floor.ny, sin = s
    const nxArr = new Float32Array(n), nhArr = new Float32Array(n), nzArr = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      const px = x[i], py = h[i], pz = z[i]
      // v' = v cos + (k × v) sin + k (k·v)(1 - cos), with k = (ax, 0, az)
      const kv = ax * px + az * pz
      const cx = -az * py
      const cy = az * px - ax * pz
      const cz = ax * py
      nxArr[i] = px * cos + cx * sin + ax * kv * (1 - cos)
      nhArr[i] = py * cos + cy * sin
      nzArr[i] = pz * cos + cz * sin + az * kv * (1 - cos)
    }
    out = { ...cloud, x: nxArr, h: nhArr, z: nzArr }
  }
  // Shift: median height of floor inliers after rotation.
  const floorH: number[] = []
  const lim = quantile(sample(out.h, 20_000, rand), 0.35)
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 50_000))) if (out.h[i] <= lim) floorH.push(out.h[i])
  // The floor is the lowest dense layer: take the 2 cm bin with the most points among the low ones.
  const shift = densestLayer(floorH)
  const hh = out === cloud ? new Float32Array(n) : out.h
  for (let i = 0; i < n; i++) hh[i] = out.h[i] - shift
  out = { ...out, h: hh }
  return { cloud: out, tilt, floorShift: shift }
}

/** Least-squares refit (h = a x + b z + c) on the plane's inliers, tightening the band each round. */
function refinePlane(cloud: Cloud, idx: number[], p: { nx: number; ny: number; nz: number; d: number }) {
  const { x, h, z } = cloud
  let { nx, ny, nz, d } = p
  const stride = Math.max(1, Math.floor(idx.length / 100_000))
  for (const band of [0.03, 0.02, 0.012]) {
    // Normal equations for [a b c].
    let sxx = 0, sxz = 0, sx = 0, szz = 0, sz = 0, n = 0, sxh = 0, szh = 0, sh = 0
    for (let k = 0; k < idx.length; k += stride) {
      const i = idx[k]
      if (Math.abs(nx * x[i] + ny * h[i] + nz * z[i] + d) > band) continue
      sxx += x[i] * x[i]; sxz += x[i] * z[i]; sx += x[i]; szz += z[i] * z[i]; sz += z[i]; n++
      sxh += x[i] * h[i]; szh += z[i] * h[i]; sh += h[i]
    }
    if (n < 30) break
    const sol = solve3([sxx, sxz, sx, sxz, szz, sz, sx, sz, n], [sxh, szh, sh])
    if (!sol) break
    const [a, b, c] = sol
    const l = Math.hypot(a, 1, b)
    nx = -a / l
    ny = 1 / l
    nz = -b / l
    d = -c / l
  }
  return { nx, ny, nz, d }
}

function solve3(m: number[], r: number[]): [number, number, number] | null {
  const det = (q: number[]) => q[0] * (q[4] * q[8] - q[5] * q[7]) - q[1] * (q[3] * q[8] - q[5] * q[6]) + q[2] * (q[3] * q[7] - q[4] * q[6])
  const D = det(m)
  if (Math.abs(D) < 1e-12) return null
  const col = (k: number) => m.map((v, i) => (i % 3 === k ? r[(i - k) / 3] : v))
  return [det(col(0)) / D, det(col(1)) / D, det(col(2)) / D]
}

function densestLayer(values: number[]) {
  if (!values.length) return 0
  const min = Math.min(...values.slice(0, 100_000))
  const bins = new Map<number, number>()
  for (const v of values) {
    const b = Math.floor((v - min) / 0.02)
    bins.set(b, (bins.get(b) ?? 0) + 1)
  }
  let bestBin = 0
  let bestCount = -1
  for (const [b, c] of bins) if (c > bestCount) [bestBin, bestCount] = [b, c]
  // Mean of the points in that bin and its neighbors.
  let s = 0
  let k = 0
  for (const v of values) {
    const b = Math.floor((v - min) / 0.02)
    if (Math.abs(b - bestBin) <= 1) {
      s += v
      k++
    }
  }
  return s / k
}

export function sample(a: ArrayLike<number>, k: number, rand: () => number) {
  if (a.length <= k) return a
  const out = new Float32Array(k)
  for (let i = 0; i < k; i++) out[i] = a[Math.floor(rand() * a.length)]
  return out
}

/** Rotate plan coordinates (x, z) clockwise by theta, like geometry.rot. */
export function rotateXZ(cloud: Cloud, theta: number): { px: Float32Array; py: Float32Array } {
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const px = new Float32Array(cloud.n)
  const py = new Float32Array(cloud.n)
  for (let i = 0; i < cloud.n; i++) {
    px[i] = cloud.x[i] * c - cloud.z[i] * s
    py[i] = cloud.x[i] * s + cloud.z[i] * c
  }
  return { px, py }
}

/** Write a cloud back out as ScanData (Y up), e.g. to store the leveled version. */
export function cloudToScan(cloud: Cloud, base: ScanData): ScanData {
  const positions = new Float32Array(cloud.n * 3)
  for (let i = 0; i < cloud.n; i++) {
    positions[i * 3] = cloud.x[i]
    positions[i * 3 + 1] = cloud.h[i]
    positions[i * 3 + 2] = cloud.z[i]
  }
  return { ...base, upAxis: 'y', points: { count: cloud.n, positions, colors: cloud.rgb } }
}
