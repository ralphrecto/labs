// Scans loaded in this browser session: decoded points, 3D render data and plan underlays.
// The document only holds ScanRefs; this module turns them into something to look at.

import { create } from 'zustand'
import { uid } from '../model/sample'
import type { ScanRef, Vec2 } from '../model/types'
import {
  detectFloorCeiling,
  detectUpAxis,
  scanToPlan,
  straightenAngle,
  topBounds,
  topUnderlay,
  uprightPositions,
  wallBandPoints,
  wallUnderlay,
  type Underlay,
} from './align'
import { loadScanFile, saveScanFile } from './fileStore'
import { loadScanBytes, type LoadedScan } from './importers'

export type ScanStatus = 'loading' | 'ready' | 'missing' | 'error'

interface Entry {
  status: ScanStatus
  message?: string
  loaded?: LoadedScan
}

export const useScanRuntime = create<{ entries: Record<string, Entry> }>(() => ({ entries: {} }))

const setEntry = (id: string, e: Entry) => useScanRuntime.setState((s) => ({ entries: { ...s.entries, [id]: e } }))

export function scanEntry(id: string): Entry | undefined {
  return useScanRuntime.getState().entries[id]
}

// ------------------------------------------------------------------ derived data (cached)

const uprightCache = new Map<string, { key: string; pos: Float32Array; bounds: ReturnType<typeof topBounds> }>()

export function uprightOf(ref: ScanRef) {
  const loaded = scanEntry(ref.id)?.loaded
  if (!loaded) return null
  const c = uprightCache.get(ref.id)
  if (c && c.key === ref.upAxis) return c
  const pos = uprightPositions(loaded.points.positions, ref.upAxis)
  const next = { key: ref.upAxis, pos, bounds: topBounds(pos) }
  uprightCache.set(ref.id, next)
  return next
}

export interface UnderlayImage {
  underlay: Underlay
  href: string
}

const underlayCache = new Map<string, { key: string; img: UnderlayImage | null }>()

/** Thicken thin lines by a pixel for display (snapping uses the sharp original). */
function thicken(u: Underlay) {
  const out = new Uint8ClampedArray(u.data)
  const { width: w, height: h, data } = u
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const k = (y * w + x) * 4
      let a = data[k + 3]
      if (x > 0) a = Math.max(a, data[k - 4 + 3] * 0.8)
      if (x < w - 1) a = Math.max(a, data[k + 4 + 3] * 0.8)
      if (y > 0) a = Math.max(a, data[k - w * 4 + 3] * 0.8)
      if (y < h - 1) a = Math.max(a, data[k + w * 4 + 3] * 0.8)
      if (a > data[k + 3]) {
        out[k] = 40
        out[k + 1] = 52
        out[k + 2] = 92
        out[k + 3] = a
      }
    }
  return out
}

function toDataUrl(u: Underlay, thick: boolean) {
  const c = document.createElement('canvas')
  c.width = u.width
  c.height = u.height
  const ctx = c.getContext('2d')!
  ctx.putImageData(new ImageData(thick ? thicken(u) : new Uint8ClampedArray(u.data), u.width, u.height), 0, 0)
  return c.toDataURL('image/png')
}

/** The plan underlay for a scan, recomputed only when something that changes it changes. */
export function underlayOf(ref: ScanRef, kind: 'walls' | 'top' = ref.underlay === 'top' ? 'top' : 'walls'): UnderlayImage | null {
  const up = uprightOf(ref)
  const loaded = scanEntry(ref.id)?.loaded
  if (!up || !loaded) return null
  const key = kind === 'walls' ? `w:${ref.upAxis}:${ref.floor}:${ref.scale}:${ref.sliceLow}:${ref.sliceHigh}` : `t:${ref.upAxis}:${ref.floor}:${ref.scale}:${ref.cutHeight}`
  const cacheKey = `${ref.id}:${kind}`
  const c = underlayCache.get(cacheKey)
  if (c && c.key === key) return c.img
  const u = kind === 'walls' ? wallUnderlay(up.pos, ref, up.bounds) : topUnderlay(up.pos, loaded.points.colors, ref, up.bounds)
  const img = { underlay: u, href: toDataUrl(u, kind === 'walls') }
  underlayCache.set(cacheKey, { key, img })
  return img
}

/** Center of the scan on the plan. */
export function scanCenter(ref: ScanRef): Vec2 | null {
  const up = uprightOf(ref)
  if (!up) return null
  const b = up.bounds
  return scanToPlan(ref, { x: (b.minX + b.maxX) / 2, y: (b.minZ + b.maxZ) / 2 })
}

/** Plan-space corners of the scan's footprint. */
export function scanCorners(ref: ScanRef): Vec2[] | null {
  const up = uprightOf(ref)
  if (!up) return null
  const b = up.bounds
  return [
    { x: b.minX, y: b.minZ },
    { x: b.maxX, y: b.minZ },
    { x: b.maxX, y: b.maxZ },
    { x: b.minX, y: b.maxZ },
  ].map((p) => scanToPlan(ref, p))
}

/** Points of the wall band in upright top-down coordinates, for straightening. */
export function bandOf(ref: ScanRef) {
  const up = uprightOf(ref)
  return up ? wallBandPoints(up.pos, ref, 200_000) : null
}

// ------------------------------------------------------------------ loading

async function decode(id: string, bytes: Uint8Array, name: string) {
  setEntry(id, { status: 'loading', message: 'Reading the scan…' })
  try {
    const loaded = await loadScanBytes(bytes, name, (message) => setEntry(id, { status: 'loading', message }))
    if (loaded.points.count < 100) throw new Error('This scan has almost no points in it.')
    uprightCache.delete(id)
    for (const k of underlayCache.keys()) if (k.startsWith(`${id}:`)) underlayCache.delete(k)
    setEntry(id, { status: 'ready', loaded })
    return loaded
  } catch (err) {
    setEntry(id, { status: 'error', message: err instanceof Error ? err.message : String(err) })
    throw err
  }
}

const inflight = new Set<string>()

/** Load a scan's file from IndexedDB if it isn't in memory yet. */
export async function ensureScanLoaded(ref: ScanRef) {
  if (scanEntry(ref.id) || inflight.has(ref.id)) return
  inflight.add(ref.id)
  try {
    setEntry(ref.id, { status: 'loading', message: 'Opening the saved scan…' })
    const f = await loadScanFile(ref.id)
    if (!f) {
      setEntry(ref.id, { status: 'missing', message: 'The scan file isn’t saved in this browser. Open it again to show it.' })
      return
    }
    await decode(ref.id, new Uint8Array(await f.blob.arrayBuffer()), f.name).catch(() => undefined)
  } finally {
    inflight.delete(ref.id)
  }
}

/** Re-attach a file to an existing scan (e.g. after opening the project on another computer). */
export async function relinkScanFile(ref: ScanRef, file: File) {
  await saveScanFile(ref.id, { name: file.name, type: file.type, blob: file })
  await decode(ref.id, new Uint8Array(await file.arrayBuffer()), file.name)
}

/**
 * Import a scan file: store it, decode it, and work out a first alignment (up axis, floor,
 * ceiling, straightened walls). Returns the new ScanRef for the document.
 */
export async function importScanFile(file: { name: string; type?: string; blob: Blob }, place: (size: Vec2) => Vec2): Promise<ScanRef> {
  const id = uid()
  const bytes = new Uint8Array(await file.blob.arrayBuffer())
  const loaded = await decode(id, bytes, file.name)
  // Store the file after it decoded, so broken files don't pile up in the browser.
  await saveScanFile(id, { name: file.name, type: file.type ?? '', blob: file.blob }).catch(() => undefined)

  const isGltf = /\.(glb|gltf)$/i.test(file.name)
  const upAxis = isGltf ? 'y' : detectUpAxis(loaded.points.positions)
  const up = uprightPositions(loaded.points.positions, upAxis)
  const { floor, ceiling } = detectFloorCeiling(up)
  const ceilH = ceiling != null ? ceiling - floor : undefined
  const base: ScanRef = {
    id,
    name: file.name.replace(/\.[^.]+$/, ''),
    fileName: file.name,
    fileSize: file.blob.size,
    format: loaded.format,
    upAxis,
    floor,
    ceiling,
    scale: 1,
    yaw: 0,
    x: 0,
    y: 0,
    references: [],
    visible: true,
    underlay: 'walls',
    opacity: 0.85,
    sliceLow: 1.0,
    sliceHigh: ceilH ? Math.min(2.0, ceilH - 0.25) : 2.0,
    cutHeight: ceilH ? Math.min(2.0, ceilH - 0.3) : 1.8,
  }
  const yaw = straightenAngle(wallBandPoints(up, base, 200_000), 1)
  const b = topBounds(up)
  const ref = { ...base, yaw }
  // Put the scan's footprint where the caller wants it (next to the rooms already drawn).
  const corners = [
    { x: b.minX, y: b.minZ },
    { x: b.maxX, y: b.minZ },
    { x: b.maxX, y: b.maxZ },
    { x: b.minX, y: b.maxZ },
  ].map((p) => scanToPlan(ref, p))
  const minX = Math.min(...corners.map((p) => p.x))
  const minY = Math.min(...corners.map((p) => p.y))
  const size = { x: Math.max(...corners.map((p) => p.x)) - minX, y: Math.max(...corners.map((p) => p.y)) - minY }
  const at = place(size)
  return { ...ref, x: at.x - minX, y: at.y - minY }
}

/** Drop a scan from memory. Its file stays in IndexedDB until the next start, so undo can bring it back. */
export function forgetScan(id: string) {
  uprightCache.delete(id)
  for (const k of underlayCache.keys()) if (k.startsWith(`${id}:`)) underlayCache.delete(k)
  useScanRuntime.setState((s) => {
    const entries = { ...s.entries }
    delete entries[id]
    return { entries }
  })
}
