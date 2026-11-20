import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
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

/**
 * A soft patch of daylight on the floor in front of the window, slanting a little as if the sun
 * came in from the side. It fades away from the window (a flat band in Cartoon).
 */
function Daylight({ w }: { w: WindowSpec }) {
  const geom = useMemo(() => {
    const W = w.width * 0.95
    const reach = Math.min(1.7, 0.55 * w.height + 0.5)
    const skew = reach * 0.35
    const z0 = w.depth / 2 + 0.1
    const z1 = z0 + reach
    const s = w.inward
    // Local frame on the floor: x along the wall, z into the room.
    const pts = [
      [-W / 2, z0],
      [W / 2, z0],
      [W / 2 + skew, z1],
      [-W / 2 + skew, z1],
    ]
    const pos = new Float32Array(pts.flatMap(([x, z]) => [x, 0, z * s]))
    const uv = new Float32Array([0, 1, 1, 1, 1, 0, 0, 0])
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
    g.setIndex(s > 0 ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3])
    g.computeVertexNormals()
    return g
  }, [w.width, w.height, w.depth, w.inward])
  useEffect(() => () => geom.dispose(), [geom])
  return (
    <mesh geometry={geom} position={[w.pos[0], 0.005, w.pos[2]]} rotation={[0, w.rotY, 0]} raycast={noRaycast} renderOrder={1}>
      <Surface role="daylight" color="#fff3d6" side={THREE.DoubleSide} />
    </mesh>
  )
}
