import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { rot } from '../model/geometry'
import { isWallDecor } from '../model/wallDecor'
import { activeLayout, useStore } from '../store'
import { Ink, Surface } from './style'

// A window in a wall opening: a painted frame with a cross mullion, glass that shows a soft sky,
// a little sill on the inside, and a patch of daylight on the floor in front of it. Every part
// draws through <Surface> and <Ink>, so it follows the render style.

export interface WindowSpec {
  id: string
  /** Centre of the opening: on the wall's middle plane, halfway up the opening. */
  pos: [number, number, number]
  rotY: number
  width: number
  height: number
  sill: number
  /** Wall thickness. */
  depth: number
  /** +1 when the group's local +z points into the room, -1 when it points out. */
  inward: 1 | -1
  /** Outward normal and a point on the wall's outer face, for the cutaway (same test as the walls). */
  nx: number
  nz: number
  ox: number
  oz: number
}

const FRAME = '#f8f3ea'
const noRaycast = () => null

export function WindowModel({ w, mode }: { w: WindowSpec; mode: 'auto' | 'low' | 'full' }) {
  const ref = useRef<THREE.Group>(null)
  // Hidden with its wall when the wall is cut down to see into the room.
  useFrame(({ camera }) => {
    const g = ref.current
    if (!g) return
    const c = camera.position
    const outside = (c.x - w.ox) * w.nx + (c.z - w.oz) * w.nz > 0
    g.visible = !(mode === 'low' || (mode === 'auto' && outside))
  })
  const W = w.width
  const H = w.height
  const f = Math.min(0.06, W / 6, H / 6)
  const fd = w.depth + 0.02
  const inner = { w: W - 2 * f, h: H - 2 * f }
  const bar = 0.035
  // Casement look: a centre post on wide windows, a transom on tall ones.
  const mullion = W > 0.85
  const transomY = H > 1.1 ? H / 2 - f - inner.h * 0.3 : null
  const sillY = -H / 2 + 0.0025
  const sillZ = w.inward * (w.depth / 2 + 0.03)

  return (
    <>
      <group ref={ref} position={w.pos} rotation={[0, w.rotY, 0]}>
        {/* Glass: the sky beyond, seen from either side. */}
        <mesh>
          <planeGeometry args={[inner.w, inner.h]} />
          <Surface role="window" color="#bfe0f7" side={THREE.DoubleSide} />
        </mesh>
        {/* Frame: head, sill rail and two jambs, through the whole wall. */}
        <Part size={[W, f, fd]} pos={[0, H / 2 - f / 2, 0]} />
        <Part size={[W, f, fd]} pos={[0, -H / 2 + f / 2, 0]} />
        <Part size={[f, H - 2 * f, fd]} pos={[-W / 2 + f / 2, 0, 0]} />
        <Part size={[f, H - 2 * f, fd]} pos={[W / 2 - f / 2, 0, 0]} />
        {mullion && <Part size={[bar, inner.h, 0.05]} pos={[0, 0, 0]} />}
        {transomY != null && <Part size={[inner.w, bar, 0.05]} pos={[0, transomY, 0]} />}
        {/* Inside sill ledge. */}
        <Part size={[W + 0.12, 0.035, 0.09]} pos={[0, sillY, sillZ]} />
      </group>
      <Daylight w={w} />
    </>
  )
}

function Part({ size, pos }: { size: [number, number, number]; pos: [number, number, number] }) {
  return (
    <mesh position={pos} castShadow>
      <boxGeometry args={size} />
      <Surface role="item" color={FRAME} />
      <Ink />
    </mesh>
  )
}

/** A floor-standing piece's footprint: centre, half sizes and plan rotation. */
interface Block {
  x: number
  y: number
  hw: number
  hd: number
  r: number
  /** Height (m). */
  h: number
}

/**
 * Footprints of the pieces standing on the floor in the active design (no rugs, wall decor, or
 * lamps sitting on tables), as a string so the selector stays stable between unrelated changes.
 */
function blocksKey(st: ReturnType<typeof useStore.getState>): string {
  const doc = st.doc
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const out: number[] = []
  for (const p of activeLayout(doc).placements) {
    const it = items.get(p.itemId)
    if (!it || p.on || it.floorLevel || it.category === 'rug' || it.category === 'pendant' || isWallDecor(it.category)) continue
    out.push(+p.x.toFixed(3), +p.y.toFixed(3), +(it.width / 2).toFixed(3), +(it.depth / 2).toFixed(3), +p.rotation.toFixed(4), +it.height.toFixed(3))
  }
  return out.join(',')
}

function parseBlocks(key: string): Block[] {
  if (!key) return []
  const n = key.split(',').map(Number)
  const out: Block[] = []
  for (let i = 0; i + 5 < n.length; i += 6) out.push({ x: n[i], y: n[i + 1], hw: n[i + 2], hd: n[i + 3], r: n[i + 4], h: n[i + 5] })
  return out
}

/** Distance (m) from a plan point to the nearest footprint taller than `above`; 0 inside one. */
function blockDistance(x: number, y: number, blocks: Block[], above = 0): number {
  let best = Infinity
  for (const b of blocks) {
    if (b.h <= above) continue
    const l = rot({ x: x - b.x, y: y - b.y }, -b.r)
    const dx = Math.max(Math.abs(l.x) - b.hw, 0)
    const dy = Math.max(Math.abs(l.y) - b.hd, 0)
    best = Math.min(best, Math.hypot(dx, dy))
    if (best === 0) return 0
  }
  return best
}

/** How far past a piece's edge the sun patch takes to come back to full: a soft shadow edge. */
const SHADOW_SOFT = 0.1

/**
 * A soft patch of daylight on the floor in front of the window, slanting a little as if the sun
 * came in from the side. It fades away from the window (a flat band in Cartoon), and it is built as
 * a fine grid whose vertex alpha drops to nothing under furniture and in the shadow it throws
 * (traced back toward the window along the sun's slant), so a sideboard in front of the window
 * doesn't glow from beneath.
 */
function Daylight({ w }: { w: WindowSpec }) {
  const key = useStore(blocksKey)
  const geom = useMemo(() => {
    const W = w.width * 0.95
    const reach = Math.min(1.7, 0.55 * w.height + 0.5)
    const skew = reach * 0.35
    const z0 = w.depth / 2 + 0.1
    const z1 = z0 + reach
    const s = w.inward
    // Local frame on the floor: x along the wall, z into the room. Rows run from the window (v = 1) inward.
    const nx = Math.max(2, Math.ceil(W / 0.05))
    const nz = Math.max(2, Math.ceil(reach / 0.05))
    const pos = new Float32Array((nx + 1) * (nz + 1) * 3)
    const uv = new Float32Array((nx + 1) * (nz + 1) * 2)
    const col = new Float32Array((nx + 1) * (nz + 1) * 4)
    const blocks = parseBlocks(key)
    const c = Math.cos(w.rotY)
    const sn = Math.sin(w.rotY)
    // Only pieces anywhere near the patch matter.
    const near = blocks.filter((b) => Math.hypot(b.x - w.pos[0], b.y - w.pos[2]) < Math.hypot(W / 2 + skew, z1) + Math.hypot(b.hw, b.hd) + SHADOW_SOFT)
    // The sun: lands at z1 through the window head, slanting by `skew` along the wall.
    const head = w.pos[1] + w.height / 2
    const run = Math.hypot(reach + w.depth / 2, skew)
    const slope = head / run
    // Horizontal direction of the beam in plan (local (skew, reach) into the room, rotated).
    const lx = skew / run
    const lz = (reach / run) * s
    const sx = lx * c + lz * sn
    const sy = -lx * sn + lz * c
    const maxBack = Math.min(run, Math.max(0, ...near.map((b) => b.h)) / slope)
    for (let j = 0; j <= nz; j++) {
      const t = j / nz
      const z = z0 + reach * t
      for (let i = 0; i <= nx; i++) {
        const x = -W / 2 + W * (i / nx) + skew * t
        const k = j * (nx + 1) + i
        const lz = z * s
        pos.set([x, 0, lz], k * 3)
        uv.set([i / nx, 1 - t], k * 2)
        // The mesh's own transform: rotation about y by rotY, then the window's position.
        const px = w.pos[0] + x * c + lz * sn
        const py = w.pos[2] - x * sn + lz * c
        // Walk back toward the window along the sunbeam; the beam is `s * slope` above the floor
        // there, so any piece taller than that stands in its way.
        let d = Infinity
        for (let st = 0; near.length && st <= maxBack && d > 0; st += 0.05) {
          const bx = px - st * sx
          const by = py - st * sy
          d = Math.min(d, blockDistance(bx, by, near, st * slope))
        }
        const a = Math.min(1, d / SHADOW_SOFT)
        col.set([1, 1, 1, a * a * (3 - 2 * a)], k * 4)
      }
    }
    const index: number[] = []
    for (let j = 0; j < nz; j++)
      for (let i = 0; i < nx; i++) {
        const a = j * (nx + 1) + i
        const b = a + 1
        const cc = a + nx + 1
        const dd = cc + 1
        if (s > 0) index.push(a, cc, b, b, cc, dd)
        else index.push(a, b, cc, b, dd, cc)
      }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
    g.setAttribute('color', new THREE.BufferAttribute(col, 4))
    g.setIndex(index)
    g.computeVertexNormals()
    return g
  }, [w.width, w.height, w.depth, w.inward, w.rotY, w.pos, key])
  useEffect(() => () => geom.dispose(), [geom])
  return (
    <mesh geometry={geom} position={[w.pos[0], 0.005, w.pos[2]]} rotation={[0, w.rotY, 0]} raycast={noRaycast} renderOrder={1}>
      <Surface role="daylight" color="#fff3d6" side={THREE.DoubleSide} />
    </mesh>
  )
}
