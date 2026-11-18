// A small PLY reader for the points of a scan: positions and colors only. It reads binary
// (little or big endian) and ASCII files, Gaussian splat PLYs (colors from f_dc_*) and plain
// point clouds or meshes (colors from red/green/blue). Compressed splat PLYs (with a `chunk`
// element) are left to the splat renderer.

import type { ScanPoints } from './types'

const SIZES: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
}

interface Prop {
  name: string
  type: string
  /** For list properties (mesh faces). */
  countType?: string
}
interface Element {
  name: string
  count: number
  props: Prop[]
}

export interface PlyHeader {
  format: 'ascii' | 'binary_little_endian' | 'binary_big_endian'
  elements: Element[]
  /** Byte offset where the data starts. */
  dataOffset: number
}

export function parsePlyHeader(bytes: Uint8Array): PlyHeader | null {
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 64 * 1024)))
  if (!head.startsWith('ply')) return null
  const end = head.indexOf('end_header')
  if (end < 0) return null
  const nl = head.indexOf('\n', end)
  const lines = head.slice(0, end).split(/\r?\n/)
  let format: PlyHeader['format'] = 'binary_little_endian'
  const elements: Element[] = []
  for (const line of lines) {
    const t = line.trim().split(/\s+/)
    if (t[0] === 'format') format = t[1] as PlyHeader['format']
    else if (t[0] === 'element') elements.push({ name: t[1], count: Number(t[2]), props: [] })
    else if (t[0] === 'property' && elements.length) {
      const el = elements[elements.length - 1]
      if (t[1] === 'list') el.props.push({ name: t[4], type: t[3], countType: t[2] })
      else el.props.push({ name: t[2], type: t[1] })
    }
  }
  // The header is ASCII, so the character offset is the byte offset.
  return { format, elements, dataOffset: nl + 1 }
}

export type PlyKind = 'splat' | 'compressed-splat' | 'mesh' | 'points'

export function plyKind(h: PlyHeader): PlyKind {
  if (h.elements.some((e) => e.name === 'chunk')) return 'compressed-splat'
  const v = h.elements.find((e) => e.name === 'vertex')
  if (v?.props.some((p) => p.name === 'f_dc_0' || p.name === 'scale_0')) return 'splat'
  if (h.elements.some((e) => e.name === 'face' && e.count > 0)) return 'mesh'
  return 'points'
}

const SH_C0 = 0.28209479177387814

/** Read vertex positions and colors. Returns null for compressed splats. */
export function readPlyPoints(bytes: Uint8Array, h = parsePlyHeader(bytes)): (ScanPoints & { opacity?: Float32Array }) | null {
  if (!h) return null
  if (plyKind(h) === 'compressed-splat') return null
  const vi = h.elements.findIndex((e) => e.name === 'vertex')
  if (vi < 0) return null
  const v = h.elements[vi]
  const n = v.count
  const idx = (name: string) => v.props.findIndex((p) => p.name === name)
  const ix = idx('x')
  const iy = idx('y')
  const iz = idx('z')
  if (ix < 0 || iy < 0 || iz < 0) return null
  const dc = [idx('f_dc_0'), idx('f_dc_1'), idx('f_dc_2')]
  const rgb = [idx('red'), idx('green'), idx('blue')]
  const iop = idx('opacity')
  const hasDc = dc.every((i) => i >= 0)
  const hasRgb = rgb.every((i) => i >= 0)
  const positions = new Float32Array(n * 3)
  const colors = hasDc || hasRgb ? new Float32Array(n * 3) : undefined
  const opacity = iop >= 0 ? new Float32Array(n) : undefined
  const rgbScale = hasRgb ? (SIZES[v.props[rgb[0]].type] === 1 ? 1 / 255 : SIZES[v.props[rgb[0]].type] === 2 ? 1 / 65535 : 1) : 1

  const store = (i: number, vals: (k: number) => number) => {
    positions[i * 3] = vals(ix)
    positions[i * 3 + 1] = vals(iy)
    positions[i * 3 + 2] = vals(iz)
    if (colors) {
      for (let c = 0; c < 3; c++) {
        const val = hasDc ? 0.5 + SH_C0 * vals(dc[c]) : vals(rgb[c]) * rgbScale
        colors[i * 3 + c] = Math.min(1, Math.max(0, val))
      }
    }
    if (opacity) opacity[i] = 1 / (1 + Math.exp(-vals(iop)))
  }

  if (h.format === 'ascii') {
    const text = new TextDecoder().decode(bytes.subarray(h.dataOffset))
    const lines = text.split(/\r?\n/)
    // Skip elements before the vertices (rare, but allowed).
    let line = h.elements.slice(0, vi).reduce((s, e) => s + e.count, 0)
    for (let i = 0; i < n; i++, line++) {
      const t = lines[line].trim().split(/\s+/).map(Number)
      store(i, (k) => t[k])
    }
    return { count: n, positions, colors, opacity }
  }

  const le = h.format === 'binary_little_endian'
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (v.props.some((p) => p.countType)) return null // list properties on vertices: not a format scanners write
  if (h.elements.slice(0, vi).some((e) => e.props.some((p) => p.countType))) return null
  const offsets: number[] = []
  let stride = 0
  for (const p of v.props) {
    offsets.push(stride)
    stride += SIZES[p.type] ?? 4
  }
  let base = h.dataOffset
  for (const e of h.elements.slice(0, vi)) base += e.count * e.props.reduce((s, p) => s + (SIZES[p.type] ?? 4), 0)
  if (base + n * stride > bytes.length) return null
  const read = (off: number, type: string): number => {
    switch (type) {
      case 'float': case 'float32': return dv.getFloat32(off, le)
      case 'double': case 'float64': return dv.getFloat64(off, le)
      case 'uchar': case 'uint8': return dv.getUint8(off)
      case 'char': case 'int8': return dv.getInt8(off)
      case 'ushort': case 'uint16': return dv.getUint16(off, le)
      case 'short': case 'int16': return dv.getInt16(off, le)
      case 'uint': case 'uint32': return dv.getUint32(off, le)
      default: return dv.getInt32(off, le)
    }
  }
  // Fast path for the common all-float little-endian layout.
  const allFloat = le && v.props.every((p) => p.type === 'float' || p.type === 'float32')
  for (let i = 0; i < n; i++) {
    const row = base + i * stride
    if (allFloat) store(i, (k) => dv.getFloat32(row + k * 4, true))
    else store(i, (k) => read(row + offsets[k], v.props[k].type))
  }
  return { count: n, positions, colors, opacity }
}
