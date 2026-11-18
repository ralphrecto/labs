// Load scan files in the browser: Gaussian splats (Scaniverse PLY/SPZ, and .splat, .ksplat, .sog),
// meshes (GLB, glTF, OBJ, PLY with faces) and point clouds (PLY). Each loader returns the points
// (for the plan underlay, straightening and furniture detection) and something to draw in 3D.

import * as THREE from 'three'
import type { PackedSplats } from '@sparkjsdev/spark'
import type { ScanFormat } from '../model/types'
import { parsePlyHeader, plyKind, readPlyPoints } from './readPly'
import type { ScanPoints } from './types'

export type ScanRender =
  | { kind: 'splat'; packed: PackedSplats }
  | { kind: 'mesh'; object: THREE.Object3D }
  | { kind: 'points'; geometry: THREE.BufferGeometry }

export interface LoadedScan {
  format: ScanFormat
  /** Points in file coordinates. Splats with low opacity are left out. */
  points: ScanPoints
  render: ScanRender
}

export const SCAN_ACCEPT = '.ply,.spz,.splat,.ksplat,.sog,.glb,.gltf,.obj'

const SPLAT_EXT = ['spz', 'splat', 'ksplat', 'sog']
const ext = (name: string) => name.toLowerCase().split('.').pop() ?? ''

/** Points to keep for analysis: plenty for a sharp plan, few enough to stay quick. */
const MAX_POINTS = 1_500_000

export function isScanFileName(name: string) {
  return SCAN_ACCEPT.split(',').some((e) => name.toLowerCase().endsWith(e))
}

export async function loadScanBytes(bytes: Uint8Array, fileName: string, onProgress?: (msg: string) => void): Promise<LoadedScan> {
  const e = ext(fileName)
  if (e === 'ply') {
    const h = parsePlyHeader(bytes)
    if (!h) throw new Error("This PLY file's header couldn't be read.")
    const kind = plyKind(h)
    if (kind === 'splat' || kind === 'compressed-splat') return loadSplat(bytes, fileName, kind === 'splat' ? readPlyPoints(bytes, h) : null, onProgress)
    if (kind === 'mesh') return loadMeshPly(bytes)
    const pts = readPlyPoints(bytes, h)
    if (!pts) throw new Error("This PLY file's points couldn't be read.")
    return { format: 'points', points: pts, render: { kind: 'points', geometry: pointsGeometry(pts) } }
  }
  if (SPLAT_EXT.includes(e)) return loadSplat(bytes, fileName, null, onProgress)
  if (e === 'glb' || e === 'gltf') return loadGltf(bytes)
  if (e === 'obj') return loadObj(bytes)
  throw new Error(`.${e} files aren't supported. Export PLY, SPZ, GLB or OBJ from Scaniverse.`)
}

// ------------------------------------------------------------------ splats

async function loadSplat(bytes: Uint8Array, fileName: string, pre: (ScanPoints & { opacity?: Float32Array }) | null, onProgress?: (msg: string) => void): Promise<LoadedScan> {
  onProgress?.('Decoding splats…')
  // Spark (the splat renderer) is big, so it only loads once someone imports a splat.
  const { PackedSplats } = await import('@sparkjsdev/spark')
  // Spark keeps a reference to the bytes it's given, so pass a copy we don't touch again.
  const packed = new PackedSplats({ fileBytes: bytes.slice().buffer, fileName })
  await packed.initialized
  let points: ScanPoints
  if (pre) points = dropFaint(pre)
  else {
    const n = packed.numSplats
    const step = Math.max(1, n / MAX_POINTS)
    const pos: number[] = []
    const col: number[] = []
    let next = 0
    packed.forEachSplat((i, center, _scales, _q, opacity, color) => {
      if (i < next) return
      next += step
      if (opacity < 0.2) return
      pos.push(center.x, center.y, center.z)
      col.push(color.r, color.g, color.b)
    })
    points = { count: pos.length / 3, positions: Float32Array.from(pos), colors: Float32Array.from(col) }
  }
  return { format: 'splat', points, render: { kind: 'splat', packed } }
}

/** Drop near-transparent splats (floaters) and thin out very large scans. */
function dropFaint(p: ScanPoints & { opacity?: Float32Array }): ScanPoints {
  const step = Math.max(1, p.count / MAX_POINTS)
  const pos: number[] = []
  const col: number[] = []
  for (let f = 0; f < p.count; f += step) {
    const i = Math.floor(f)
    if (p.opacity && p.opacity[i] < 0.2) continue
    pos.push(p.positions[i * 3], p.positions[i * 3 + 1], p.positions[i * 3 + 2])
    if (p.colors) col.push(p.colors[i * 3], p.colors[i * 3 + 1], p.colors[i * 3 + 2])
  }
  return { count: pos.length / 3, positions: Float32Array.from(pos), colors: p.colors ? Float32Array.from(col) : undefined }
}

// ------------------------------------------------------------------ point clouds

export function pointsGeometry(p: ScanPoints) {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(p.positions, 3))
  if (p.colors) g.setAttribute('color', new THREE.BufferAttribute(p.colors, 3))
  return g
}

// ------------------------------------------------------------------ meshes

async function loadGltf(bytes: Uint8Array): Promise<LoadedScan> {
  const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js')
  const gltf = await new GLTFLoader().parseAsync(bytes.slice().buffer, '')
  return meshScan(gltf.scene)
}

async function loadObj(bytes: Uint8Array): Promise<LoadedScan> {
  const { OBJLoader } = await import('three/examples/jsm/loaders/OBJLoader.js')
  const obj = new OBJLoader().parse(new TextDecoder().decode(bytes))
  // OBJ textures live in separate files; without them, color by vertex colors or a neutral tone.
  obj.traverse((o) => {
    const m = o as THREE.Mesh
    if (m.isMesh) m.material = new THREE.MeshStandardMaterial({ color: '#d9d4c8', vertexColors: !!m.geometry.getAttribute('color'), roughness: 0.9, side: THREE.DoubleSide })
  })
  return meshScan(obj)
}

async function loadMeshPly(bytes: Uint8Array): Promise<LoadedScan> {
  const { PLYLoader } = await import('three/examples/jsm/loaders/PLYLoader.js')
  const g = new PLYLoader().parse(bytes.slice().buffer)
  g.computeVertexNormals()
  const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: g.getAttribute('color') ? '#ffffff' : '#d9d4c8', vertexColors: !!g.getAttribute('color'), roughness: 0.9, side: THREE.DoubleSide }))
  return meshScan(mesh)
}

type Pixels = { data: Uint8ClampedArray; width: number; height: number }
const pixelCache = new WeakMap<object, Pixels | null>()

/** Read a texture's pixels once, so surface samples can take its colors. */
function texturePixels(tex: THREE.Texture | null | undefined): Pixels | null {
  const img = tex?.image as (CanvasImageSource & { width: number; height: number }) | undefined
  if (!img || !img.width) return null
  if (pixelCache.has(img)) return pixelCache.get(img)!
  let px: Pixels | null = null
  try {
    const w = Math.min(1024, img.width)
    const h = Math.min(1024, img.height)
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    const ctx = c.getContext('2d')!
    ctx.drawImage(img, 0, 0, w, h)
    px = { data: ctx.getImageData(0, 0, w, h).data, width: w, height: h }
  } catch {
    px = null
  }
  pixelCache.set(img, px)
  return px
}

/** Sample points evenly over the mesh surface (by area), with colors from the texture or vertices. */
export function sampleMeshSurface(root: THREE.Object3D, target = 600_000): ScanPoints {
  root.updateMatrixWorld(true)
  type Tri = { a: THREE.Vector3; b: THREE.Vector3; c: THREE.Vector3; area: number; mesh: THREE.Mesh; i: [number, number, number] }
  const tris: Tri[] = []
  let total = 0
  root.traverse((o) => {
    const m = o as THREE.Mesh
    if (!m.isMesh) return
    const pos = m.geometry.getAttribute('position')
    if (!pos) return
    const index = m.geometry.getIndex()
    const n = index ? index.count : pos.count
    for (let k = 0; k + 2 < n; k += 3) {
      const i: [number, number, number] = index ? [index.getX(k), index.getX(k + 1), index.getX(k + 2)] : [k, k + 1, k + 2]
      const a = new THREE.Vector3().fromBufferAttribute(pos, i[0]).applyMatrix4(m.matrixWorld)
      const b = new THREE.Vector3().fromBufferAttribute(pos, i[1]).applyMatrix4(m.matrixWorld)
      const c = new THREE.Vector3().fromBufferAttribute(pos, i[2]).applyMatrix4(m.matrixWorld)
      const area = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a)).length() / 2
      if (area <= 0) continue
      tris.push({ a, b, c, area, mesh: m, i })
      total += area
    }
  })
  if (!tris.length) throw new Error('This file has no mesh surfaces in it.')
  const positions: number[] = []
  const colors: number[] = []
  // Deterministic sampling: a fixed number of points per triangle, carrying the remainder over.
  let carry = 0
  let seed = 1
  const rand = () => {
    seed = (seed * 16807) % 2147483647
    return seed / 2147483647
  }
  const p = new THREE.Vector3()
  const col = new THREE.Color()
  for (const t of tris) {
    carry += (t.area / total) * target
    const k = Math.floor(carry)
    carry -= k
    if (!k) continue
    const g = t.mesh.geometry
    const mat = (Array.isArray(t.mesh.material) ? t.mesh.material[0] : t.mesh.material) as THREE.MeshStandardMaterial
    const uv = g.getAttribute('uv')
    const vc = g.getAttribute('color')
    const px = uv ? texturePixels(mat?.map) : null
    for (let j = 0; j < k; j++) {
      let u = rand()
      let w = rand()
      if (u + w > 1) {
        u = 1 - u
        w = 1 - w
      }
      const v0 = 1 - u - w
      p.set(0, 0, 0).addScaledVector(t.a, v0).addScaledVector(t.b, u).addScaledVector(t.c, w)
      positions.push(p.x, p.y, p.z)
      if (px && uv) {
        const tu = uv.getX(t.i[0]) * v0 + uv.getX(t.i[1]) * u + uv.getX(t.i[2]) * w
        const tv = uv.getY(t.i[0]) * v0 + uv.getY(t.i[1]) * u + uv.getY(t.i[2]) * w
        const fu = tu - Math.floor(tu)
        const fv = tv - Math.floor(tv)
        // glTF textures are not flipped (flipY = false); others are.
        const yy = mat.map!.flipY ? 1 - fv : fv
        const x = Math.min(px.width - 1, Math.floor(fu * px.width))
        const y = Math.min(px.height - 1, Math.floor(yy * px.height))
        const o = (y * px.width + x) * 4
        colors.push(px.data[o] / 255, px.data[o + 1] / 255, px.data[o + 2] / 255)
      } else if (vc) {
        colors.push(
          vc.getX(t.i[0]) * v0 + vc.getX(t.i[1]) * u + vc.getX(t.i[2]) * w,
          vc.getY(t.i[0]) * v0 + vc.getY(t.i[1]) * u + vc.getY(t.i[2]) * w,
          vc.getZ(t.i[0]) * v0 + vc.getZ(t.i[1]) * u + vc.getZ(t.i[2]) * w,
        )
      } else {
        col.copy(mat?.color ?? new THREE.Color('#bbbbbb'))
        colors.push(col.r, col.g, col.b)
      }
    }
  }
  return { count: positions.length / 3, positions: Float32Array.from(positions), colors: Float32Array.from(colors) }
}

function meshScan(object: THREE.Object3D): LoadedScan {
  return { format: 'mesh', points: sampleMeshSurface(object), render: { kind: 'mesh', object } }
}
