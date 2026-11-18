import type { ScanData } from './types'

// A small PLY reader for point clouds and Gaussian splat exports (ASCII or binary little endian).
// Splats store color as spherical-harmonic DC terms (f_dc_0..2) and opacity as a logit; nearly
// transparent splats are dropped since they're mostly floaters. The scan import code may bring a
// fuller reader; this one is enough for detection.

const SH_C0 = 0.28209479177387814
const SIZES: Record<string, number> = { char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2, int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 }

interface Prop {
  name: string
  type: string
  offset: number
}

export function parsePLY(buf: ArrayBuffer, name = 'scan.ply', maxPoints = 1_500_000): ScanData {
  const bytes = new Uint8Array(buf)
  const headerEnd = findHeaderEnd(bytes)
  if (headerEnd < 0) throw new Error('Not a PLY file')
  const header = new TextDecoder().decode(bytes.subarray(0, headerEnd))
  const lines = header.split(/\r?\n/)
  if (lines[0].trim() !== 'ply') throw new Error('Not a PLY file')
  const format = lines.find((l) => l.startsWith('format'))?.split(/\s+/)[1] ?? ''
  let count = 0
  const props: Prop[] = []
  let inVertex = false
  let stride = 0
  for (const l of lines) {
    const t = l.trim().split(/\s+/)
    if (t[0] === 'element') {
      inVertex = t[1] === 'vertex'
      if (inVertex) count = Number(t[2])
      else if (count) break // only the vertex element matters, and it comes first in point clouds
    } else if (t[0] === 'property') {
      if (t[1] === 'list') {
        if (inVertex) throw new Error('List properties on vertices are not supported')
        continue
      }
      const size = SIZES[t[1]]
      if (!size) throw new Error(`Unknown PLY type ${t[1]}`)
      if (inVertex) {
        props.push({ name: t[2], type: t[1], offset: stride })
        stride += size
      }
    }
  }
  const idx = (n: string) => props.findIndex((p) => p.name === n)
  const ix = idx('x'), iy = idx('y'), iz = idx('z')
  if (ix < 0 || iy < 0 || iz < 0 || !count) throw new Error('No vertex positions in this PLY')
  const rgbIdx = ['red', 'green', 'blue'].map(idx)
  const dcIdx = ['f_dc_0', 'f_dc_1', 'f_dc_2'].map(idx)
  const iop = idx('opacity')
  const hasRGB = rgbIdx.every((i) => i >= 0)
  const hasDC = dcIdx.every((i) => i >= 0)

  const every = Math.max(1, Math.ceil(count / maxPoints))
  const keep = Math.ceil(count / every)
  const positions = new Float32Array(keep * 3)
  const colors = hasRGB || hasDC ? new Float32Array(keep * 3) : undefined
  let n = 0

  const emit = (get: (p: number) => number, i: number) => {
    if (i % every || n >= keep) return
    if (iop >= 0 && 1 / (1 + Math.exp(-get(iop))) < 0.1) {
      // Skip the floater but keep sampling evenly.
      return
    }
    positions[n * 3] = get(ix)
    positions[n * 3 + 1] = get(iy)
    positions[n * 3 + 2] = get(iz)
    if (colors) {
      for (let c = 0; c < 3; c++) colors[n * 3 + c] = hasRGB ? get(rgbIdx[c]) / (props[rgbIdx[c]].type.includes('char') || props[rgbIdx[c]].type.includes('int8') ? 255 : 1) : Math.min(1, Math.max(0, 0.5 + SH_C0 * get(dcIdx[c])))
    }
    n++
  }

  if (format === 'ascii') {
    const body = new TextDecoder().decode(bytes.subarray(headerEnd)).split(/\r?\n/)
    let row = 0
    for (const l of body) {
      if (row >= count) break
      const t = l.trim()
      if (!t) continue
      const vals = t.split(/\s+/).map(Number)
      emit((p) => vals[p], row)
      row++
    }
  } else if (format === 'binary_little_endian') {
    const dv = new DataView(buf, headerEnd)
    const readers = props.map((p) => reader(p.type))
    for (let i = 0; i < count; i++) {
      if (i % every) continue
      const base = i * stride
      emit((p) => readers[p](dv, base + props[p].offset), i)
    }
  } else throw new Error(`PLY format ${format} isn't supported`)

  return {
    id: `ply-${Date.now().toString(36)}`,
    name: name.replace(/\.ply$/i, ''),
    source: 'ply',
    points: { count: n, positions: positions.subarray(0, n * 3), colors: colors?.subarray(0, n * 3) },
    upAxis: 'y',
    metric: true,
  }
}

function findHeaderEnd(b: Uint8Array) {
  const marker = 'end_header'
  const lim = Math.min(b.length, 64 * 1024)
  for (let i = 0; i < lim - marker.length; i++) {
    let ok = true
    for (let k = 0; k < marker.length && ok; k++) if (b[i + k] !== marker.charCodeAt(k)) ok = false
    if (ok) {
      let j = i + marker.length
      if (b[j] === 13) j++
      if (b[j] === 10) j++
      return j
    }
  }
  return -1
}

function reader(type: string): (dv: DataView, o: number) => number {
  switch (type) {
    case 'char':
    case 'int8':
      return (dv, o) => dv.getInt8(o)
    case 'uchar':
    case 'uint8':
      return (dv, o) => dv.getUint8(o)
    case 'short':
    case 'int16':
      return (dv, o) => dv.getInt16(o, true)
    case 'ushort':
    case 'uint16':
      return (dv, o) => dv.getUint16(o, true)
    case 'int':
    case 'int32':
      return (dv, o) => dv.getInt32(o, true)
    case 'uint':
    case 'uint32':
      return (dv, o) => dv.getUint32(o, true)
    case 'double':
    case 'float64':
      return (dv, o) => dv.getFloat64(o, true)
    default:
      return (dv, o) => dv.getFloat32(o, true)
  }
}
