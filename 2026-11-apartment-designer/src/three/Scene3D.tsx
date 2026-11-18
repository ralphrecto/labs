import { OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useAnalysis } from '../analysis'
import { bounds, centroid, dist, norm, scale, sub, add, v, wallInwardNormal, wallSegments } from '../model/geometry'
import { snapPlacement } from '../model/snapping'
import type { Item, Placement, Room, Vec2 } from '../model/types'
import { activeLayout, useStore } from '../store'
import { ItemModel } from './ItemModel'

const FLOOR = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)

export type WallMode = 'auto' | 'low' | 'full'
const LOW_WALL = 0.3

interface WallPieceSpec {
  pos: [number, number, number]
  size: [number, number, number]
  rotY: number
  y0: number
  y1: number
  /** Outward normal and a point on the wall's outer face, for cutaway. */
  nx: number
  nz: number
  ox: number
  oz: number
}

function WallPiece({ w, mode, color }: { w: WallPieceSpec; mode: WallMode; color: string }) {
  const ref = useRef<THREE.Mesh>(null)
  useFrame(({ camera }) => {
    const m = ref.current
    if (!m) return
    const c = camera.position
    const outside = (c.x - w.ox) * w.nx + (c.z - w.oz) * w.nz > 0
    const cut = mode === 'low' || (mode === 'auto' && outside)
    if (!cut) {
      m.visible = true
      m.scale.y = 1
      m.position.y = w.pos[1]
      return
    }
    if (w.y0 >= LOW_WALL) {
      m.visible = false
      return
    }
    const top = Math.min(w.y1, LOW_WALL)
    m.visible = true
    m.scale.y = (top - w.y0) / (w.y1 - w.y0)
    m.position.y = (w.y0 + top) / 2
  })
  return (
    <mesh ref={ref} position={w.pos} rotation={[0, w.rotY, 0]} castShadow receiveShadow>
      <boxGeometry args={w.size} />
      <meshStandardMaterial color={color} roughness={0.9} />
    </mesh>
  )
}

function RoomMesh({ room, wallMode, selected }: { room: Room; wallMode: WallMode; selected: boolean }) {
  const floorGeom = useMemo(() => {
    // Shape in (x, -y) then rotate so plan y maps to world +z.
    const shape = new THREE.Shape(room.polygon.map((p) => new THREE.Vector2(p.x, -p.y)))
    const g = new THREE.ShapeGeometry(shape)
    g.rotateX(-Math.PI / 2)
    return g
  }, [room.polygon])

  const walls = useMemo(() => {
    const H = room.ceilingHeight
    const t = room.wallThickness
    const out: WallPieceSpec[] = []
    wallSegments(room).forEach(([a, b], wi) => {
      const L = dist(a, b)
      const dir = norm(sub(b, a))
      const n = wallInwardNormal(room, wi)
      const angle = Math.atan2(dir.y, dir.x)
      const jitter = wi * 0.0005 // avoid z-fighting where wall tops overlap at corners
      const piece = (s0: number, s1: number, y0: number, y1: number) => {
        if (s1 - s0 < 1e-4 || y1 - y0 < 1e-4) return
        const mid = add(add(a, scale(dir, (s0 + s1) / 2)), scale(n, -t / 2))
        const outer = add(a, scale(n, -t))
        out.push({ pos: [mid.x, (y0 + y1) / 2, mid.y], size: [s1 - s0, y1 - y0 + jitter, t], rotY: -angle, y0, y1, nx: -n.x, nz: -n.y, ox: outer.x, oz: outer.y })
      }
      // Extend past both ends by the wall thickness so corners are filled.
      const ops = room.openings.filter((o) => o.wall === wi).sort((x, y) => x.offset - y.offset)
      let cursor = -t
      for (const o of ops) {
        piece(cursor, o.offset, 0, H)
        const top = o.sill + o.height
        piece(o.offset, o.offset + o.width, 0, Math.min(H, o.sill))
        if (top < H) piece(o.offset, o.offset + o.width, top, H)
        cursor = o.offset + o.width
      }
      piece(cursor, L + t, 0, H)
    })
    return out
  }, [room])

  const glass = useMemo(
    () =>
      room.openings
        .filter((o) => o.kind === 'window')
        .map((o) => {
          const [a, b] = wallSegments(room)[o.wall]
          const dir = norm(sub(b, a))
          const n = wallInwardNormal(room, o.wall)
          const mid = add(add(a, scale(dir, o.offset + o.width / 2)), scale(n, -room.wallThickness / 2))
          return { id: o.id, pos: [mid.x, o.sill + o.height / 2, mid.y] as [number, number, number], size: [o.width, o.height, 0.01] as [number, number, number], rotY: -Math.atan2(dir.y, dir.x) }
        }),
    [room],
  )

  return (
    <group>
      <mesh geometry={floorGeom} receiveShadow position={[0, 0.001, 0]}>
        <meshStandardMaterial color={selected ? '#e9e1cf' : '#ddd3bf'} roughness={0.95} />
      </mesh>
      {walls.map((w, i) => (
        <WallPiece key={i} w={w} mode={wallMode} color={selected ? '#e5ecfa' : '#f2f0eb'} />
      ))}
      {glass.map((g) => (
        <mesh key={g.id} position={g.pos} rotation={[0, g.rotY, 0]}>
          <boxGeometry args={g.size} />
          <meshStandardMaterial color="#9cc3e6" transparent opacity={0.35} roughness={0.1} />
        </mesh>
      ))}
    </group>
  )
}

function PlacedItem({
  item,
  p,
  selected,
  flag,
  onGrab,
}: {
  item: Item
  p: Placement
  selected: boolean
  flag?: 'error' | 'warning'
  onGrab: (e: ThreeEvent<PointerEvent>, p: Placement) => void
}) {
  const highlight = selected ? '#2f6fde' : flag === 'error' ? '#d64545' : flag === 'warning' ? '#d68a1f' : undefined
  return (
    <group position={[p.x, 0, p.y]} rotation={[0, -p.rotation, 0]} onPointerDown={(e) => onGrab(e, p)}>
      <ItemModel item={item} highlight={highlight} />
      {selected && (
        <mesh position={[0, 0.003, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[item.width + 0.06, item.depth + 0.06]} />
          <meshBasicMaterial color="#2f6fde" transparent opacity={0.25} />
        </mesh>
      )}
    </group>
  )
}

function Scene({ wallMode, eye }: { wallMode: WallMode; eye: [number, number, number] | null }) {
  const doc = useStore((s) => s.doc)
  const selection = useStore((s) => s.selection)
  const snap = useStore((s) => s.snap)
  const layout = activeLayout(doc)
  const analysis = useAnalysis(doc, layout)
  const controls = useRef<OrbitControlsImpl>(null)
  const drag = useRef<{ id: string; grab: Vec2; moved: boolean } | null>(null)

  const byId = useMemo(() => new Map(doc.items.map((i) => [i.id, i])), [doc.items])
  const placed = layout.placements.flatMap((p) => {
    const item = byId.get(p.itemId)
    return item ? [{ p, item }] : []
  })

  const hit = (e: ThreeEvent<PointerEvent>) => {
    const pt = new THREE.Vector3()
    return e.ray.intersectPlane(FLOOR, pt) ? v(pt.x, pt.z) : null
  }

  const onGrab = (e: ThreeEvent<PointerEvent>, p: Placement) => {
    if (e.button !== 0) return
    e.stopPropagation()
    const s = useStore.getState()
    s.select({ kind: 'placement', id: p.id })
    const w = hit(e)
    if (!w) return
    drag.current = { id: p.id, grab: sub(w, v(p.x, p.y)), moved: false }
    if (controls.current) controls.current.enabled = false
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }

  const onMove = (e: ThreeEvent<PointerEvent>) => {
    const d = drag.current
    if (!d) return
    const w = hit(e)
    if (!w) return
    const s = useStore.getState()
    const cur = activeLayout(s.doc).placements.find((x) => x.id === d.id)
    const item = cur && byId.get(cur.itemId)
    if (!cur || !item) return
    if (!d.moved) {
      s.beginGesture()
      d.moved = true
    }
    const others = placed.filter((x) => x.p.id !== d.id)
    const proposed = { ...cur, x: w.x - d.grab.x, y: w.y - d.grab.y }
    const res = e.altKey ? { placement: proposed } : snapPlacement(item, proposed, s.doc.rooms, others, snap)
    s.updatePlacement(d.id, { x: res.placement.x, y: res.placement.y, rotation: res.placement.rotation })
  }

  const onUp = () => {
    if (drag.current?.moved) useStore.getState().endGesture()
    drag.current = null
    if (controls.current) controls.current.enabled = true
  }

  const target = useMemo(() => {
    const pts = doc.rooms.flatMap((r) => r.polygon)
    if (!pts.length) return new THREE.Vector3(0, 0, 0)
    const b = bounds(pts)
    return new THREE.Vector3((b.minX + b.maxX) / 2, 0, (b.minY + b.maxY) / 2)
  }, [doc.rooms])

  return (
    <>
      <ambientLight intensity={0.8} />
      <hemisphereLight args={['#ffffff', '#b9a98f', 0.6]} />
      <directionalLight
        position={[target.x + 4, 9, target.z + 6]}
        intensity={1.4}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-12}
        shadow-camera-right={12}
        shadow-camera-top={12}
        shadow-camera-bottom={-12}
      />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[target.x, -0.01, target.z]} receiveShadow onPointerDown={() => useStore.getState().select(null)}>
        <planeGeometry args={[200, 200]} />
        <meshStandardMaterial color="#e9e6df" />
      </mesh>
      {doc.rooms.map((r) => (
        <RoomMesh key={r.id} room={r} wallMode={eye ? 'full' : wallMode} selected={selection?.kind === 'room' && selection.id === r.id} />
      ))}
      <group onPointerMove={onMove} onPointerUp={onUp}>
        {placed.map(({ p, item }) => (
          <PlacedItem key={p.id} item={item} p={p} selected={selection?.kind === 'placement' && selection.id === p.id} flag={analysis.flagged.get(p.id)} onGrab={onGrab} />
        ))}
      </group>
      {eye ? (
        <OrbitControls ref={controls} makeDefault target={[eye[0], eye[1], eye[2] - 0.01]} enableZoom={false} enablePan={false} rotateSpeed={-0.4} />
      ) : (
        <OrbitControls ref={controls} target={target} makeDefault maxPolarAngle={Math.PI / 2 - 0.02} minDistance={0.3} maxDistance={60} />
      )}
    </>
  )
}

export function Scene3D() {
  const doc = useStore((s) => s.doc)
  const [wallMode, setWallMode] = useState<WallMode>('auto')
  const [camKey, setCamKey] = useState(0)
  const [eye, setEye] = useState<{ pos: [number, number, number] } | null>(null)

  const initial = useMemo(() => {
    const pts = doc.rooms.flatMap((r) => r.polygon)
    if (!pts.length) return [6, 8, 10] as [number, number, number]
    const b = bounds(pts)
    const size = Math.max(b.maxX - b.minX, b.maxY - b.minY)
    return [(b.minX + b.maxX) / 2 + size * 0.15, size * 1.05, b.maxY + size * 0.55] as [number, number, number]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camKey, doc.apartment.id])

  const eyeLevel = () => {
    const room = doc.rooms[0]
    if (!room) return
    const c = centroid(room.polygon)
    setEye({ pos: [c.x, 1.6, c.y] })
    setCamKey((k) => k + 1)
  }

  return (
    <div className="scene-wrap">
      <Canvas
        key={`${camKey}:${doc.apartment.id}`}
        shadows
        dpr={[1, 2]}
        camera={{ position: eye?.pos ?? initial, fov: eye ? 70 : 45, near: 0.05, far: 300 }}
        onPointerMissed={() => useStore.getState().select(null)}
        gl={{ preserveDrawingBuffer: true }}
      >
        <color attach="background" args={['#f6f4ef']} />
        <Scene wallMode={wallMode} eye={eye?.pos ?? null} />
      </Canvas>
      <div className="scene-tools">
        <button
          onClick={() => {
            setEye(null)
            setCamKey((k) => k + 1)
          }}
        >
          Overview
        </button>
        <button onClick={eyeLevel}>Eye level</button>
        <label title="Walls between you and the room are cut down so you can see in">
          Walls
          <select value={wallMode} onChange={(e) => setWallMode(e.target.value as WallMode)}>
            <option value="auto">Cut away</option>
            <option value="low">All low</option>
            <option value="full">Full height</option>
          </select>
        </label>
      </div>
    </div>
  )
}

