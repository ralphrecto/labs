import { OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { lazy, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useAnalysis } from '../analysis'
import { bounds, centroid, dist, norm, scale, sub, add, v, wallInwardNormal, wallSegments } from '../model/geometry'
import { snapPlacement } from '../model/snapping'
import type { Item, Placement, Room, Vec2 } from '../model/types'
import { activeLayout, useStore } from '../store'
import { ItemModel } from './ItemModel'
import { scanCenter, scanCorners, useScanRuntime } from '../scan/runtime'

// The scan backdrop pulls in the splat renderer, so it loads on first use.
const ScanBackdrop = lazy(() => import('./ScanBackdrop').then((m) => ({ default: m.ScanBackdrop })))
import { ScanPoints } from './ScanPoints'
import { floorTexture, Ink, StyleContext, StyleEnvironment, StyleLights, STYLE_STAGE, Surface, useStyle, woodTexture, type StyleId } from './style'
import { scanPalette, toonFloor, type ScanPalette } from '../scan/palette'
import { toScanData } from '../scan/align'

const FLOOR = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)

export type WallMode = 'auto' | 'low' | 'full'
const LOW_WALL = 0.3

interface WallPieceSpec {
  pos: [number, number, number]
  size: [number, number, number]
  rotY: number
  y0: number
  y1: number
  /** Which wall of the room this piece belongs to. */
  wi: number
  /** Outward normal and a point on the wall's outer face, for cutaway. */
  nx: number
  nz: number
  ox: number
  oz: number
}

function WallPiece({ w, mode, color, highlight }: { w: WallPieceSpec; mode: WallMode; color: string; highlight?: string }) {
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
      <Surface role="wall" color={color} highlight={highlight} />
      <Ink highlight={highlight} />
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
        out.push({ pos: [mid.x, (y0 + y1) / 2, mid.y], size: [s1 - s0, y1 - y0 + jitter, t], rotY: -angle, y0, y1, wi, nx: -n.x, nz: -n.y, ox: outer.x, oz: outer.y })
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

  const { style, palette } = useStyle()
  const floor = useFloorLook(room, floorGeom, style, palette)
  const wallColors = palette?.walls[room.id]
  const wallBase = palette?.wallColor[room.id] ?? (style === 'toon' ? '#f6ecdf' : '#f2f0eb')
  const pick = style === 'toon' || style === 'real'
  const hl = selected && style !== 'classic' ? '#2f6fde' : undefined

  return (
    <group>
      <mesh geometry={floor.geometry} receiveShadow position={[0, 0.001, 0]}>
        <Surface role="floor" color={floor.color} map={floor.map} highlight={hl} />
        <Ink />
      </mesh>
      {walls.map((w, i) => (
        <WallPiece
          key={i}
          w={w}
          mode={wallMode}
          highlight={hl}
          color={style === 'classic' ? (selected ? '#e5ecfa' : '#f2f0eb') : pick ? (wallColors?.[w.wi] ?? wallBase) : '#f2f0eb'}
        />
      ))}
      {glass.map((g) => (
        <mesh key={g.id} position={g.pos} rotation={[0, g.rotY, 0]}>
          <boxGeometry args={g.size} />
          <Surface role="glass" color="#9cc3e6" />
        </mesh>
      ))}
    </group>
  )
}

/**
 * What a room's floor looks like in the style: the scanned floor image (posterized for Cartoon),
 * oak planks for Lifelike without a scan, or a flat color.
 */
function useFloorLook(room: Room, base: THREE.BufferGeometry, style: StyleId, palette: ScanPalette | null) {
  const img = palette?.floor
  const useScan = !!img && (style === 'toon' || style === 'real')
  const scanned = useMemo(() => {
    if (!useScan || !img) return null
    const look = style === 'toon' ? toonFloor(img) : img
    const g = base.clone()
    const pos = g.getAttribute('position')
    const uv = new Float32Array(pos.count * 2)
    for (let i = 0; i < pos.count; i++) {
      uv[i * 2] = (pos.getX(i) - img.x0) / (img.width * img.cell)
      uv[i * 2 + 1] = (pos.getZ(i) - img.y0) / (img.height * img.cell)
    }
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
    return { geometry: g, map: floorTexture(look, style === 'toon') }
  }, [useScan, img, style, base])
  useEffect(() => () => {
    scanned?.geometry.dispose()
    scanned?.map.dispose()
  }, [scanned])
  if (scanned) return { geometry: scanned.geometry, map: scanned.map as THREE.Texture | null, color: '#ffffff' }
  const scannedColor = palette?.floorColor[room.id]
  if (style === 'real') return { geometry: base, map: scannedColor ? null : woodTexture(), color: scannedColor ?? '#c49a6c' }
  if (style === 'toon') return { geometry: base, map: null, color: scannedColor ?? '#d9a874' }
  return { geometry: base, map: null, color: '#ddd3bf' }
}

/** Colors from the visible scans, for the styles that paint with them. Recomputed when a scan moves. */
function useScanPalette(style: StyleId): ScanPalette | null {
  const scans = useStore((s) => s.doc.scans)
  const rooms = useStore((s) => s.doc.rooms)
  const entries = useScanRuntime((s) => s.entries)
  const wants = style === 'toon' || style === 'real'
  return useMemo(() => {
    if (!wants) return null
    const pts = (scans ?? []).flatMap((r) => {
      const loaded = r.visible ? entries[r.id]?.loaded : undefined
      return loaded?.points.colors ? [toScanData(r, loaded.points, 400_000).points] : []
    })
    if (!pts.length) return null
    return scanPalette(pts, rooms)
  }, [wants, scans, rooms, entries])
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

function Scene({ wallMode, eye, autoRotate, ground = '#e9e6df' }: { wallMode: WallMode; eye: [number, number, number] | null; autoRotate?: boolean; ground?: string }) {
  const { style } = useStyle()
  const stage = STYLE_STAGE[style]
  const doc = useStore((s) => s.doc)
  const selection = useStore((s) => s.selection)
  const snap = useStore((s) => s.snap)
  const layout = activeLayout(doc)
  const analysis = useAnalysis(doc, layout)
  const controls = useRef<OrbitControlsImpl>(null)
  const scan3d = useStore((s) => s.scan3d)
  const hasVisibleScan = (doc.scans ?? []).some((r) => r.visible)
  // The Scan style is the raw scan alone. The painted styles draw the rooms instead, and fall back
  // to the scan only while there are no rooms yet. The classic editor keeps its own setting.
  const scanOnly = style === 'classic' ? scan3d === 'scan-only' && hasVisibleScan : style === 'scan' && hasVisibleScan
  const showScan = style === 'classic' ? scan3d !== 'off' : style === 'scan' || !doc.rooms.length
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

  const entries = useScanRuntime((s) => s.entries)
  const target = useMemo(() => {
    const pts = sceneFootprint(doc)
    if (!pts.length) return new THREE.Vector3(0, 0, 0)
    const b = bounds(pts)
    return new THREE.Vector3((b.minX + b.maxX) / 2, 0, (b.minY + b.maxY) / 2)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc.rooms, doc.scans, entries])

  return (
    <>
      <StyleLights target={target} />
      <StyleEnvironment />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[target.x, -0.01, target.z]} receiveShadow onPointerDown={() => useStore.getState().select(null)} visible={!scanOnly && stage.showGround}>
        <planeGeometry args={[200, 200]} />
        {style === 'toon' ? <meshToonMaterial color={stage.ground ?? ground} /> : <meshStandardMaterial color={stage.ground ?? ground} roughness={1} />}
      </mesh>
      {style === 'blueprint' && <gridHelper args={[60, 120, '#7fb0ee', '#3a6fb5']} position={[target.x, -0.005, target.z]} />}
      {!scanOnly &&
        doc.rooms.map((r) => <RoomMesh key={r.id} room={r} wallMode={eye ? 'full' : wallMode} selected={selection?.kind === 'room' && selection.id === r.id} />)}
      {(doc.scans ?? []).length > 0 && showScan && (
        <Suspense fallback={null}>
          <ScanBackdrop eye={!!eye} force={style === 'scan'} />
        </Suspense>
      )}
      {(style === 'classic' || style === 'scan') && <ScanPoints />}
      <group onPointerMove={onMove} onPointerUp={onUp}>
        {placed.map(({ p, item }) => (
          <PlacedItem key={p.id} item={item} p={p} selected={selection?.kind === 'placement' && selection.id === p.id} flag={analysis.flagged.get(p.id)} onGrab={onGrab} />
        ))}
      </group>
      {!eye && <FitCamera key={(doc.scans ?? []).length} target={target} />}
      {eye ? (
        <OrbitControls ref={controls} makeDefault target={[eye[0], eye[1], eye[2] - 0.01]} enableZoom={false} enablePan={false} rotateSpeed={-0.4} />
      ) : (
        <OrbitControls ref={controls} target={target} makeDefault maxPolarAngle={Math.PI / 2 - 0.02} minDistance={0.3} maxDistance={60} autoRotate={autoRotate} autoRotateSpeed={0.6} />
      )}
    </>
  )
}

/** Place the overview camera so the whole apartment fits the pane, whatever its aspect ratio. */
function FitCamera({ target }: { target: THREE.Vector3 }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const size = useThree((s) => s.size)
  const doc = useStore((s) => s.doc)
  const done = useRef(false)
  useLayoutEffect(() => {
    if (done.current || size.width < 10) return
    done.current = true
    const pts = sceneFootprint(doc)
    if (!pts.length) return
    const b = bounds(pts)
    const radius = Math.hypot(b.maxX - b.minX, b.maxY - b.minY) / 2 + 0.3
    const vfov = (camera.fov * Math.PI) / 180
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * (size.width / size.height))
    const d = radius / Math.sin(Math.min(vfov, hfov) / 2)
    const dir = new THREE.Vector3(0.15, 1.0, 0.75).normalize()
    camera.position.copy(target).addScaledVector(dir, d)
    camera.lookAt(target)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, size, target])
  return null
}

/** Plan points the 3D view should frame: room corners plus the footprints of visible scans. */
function sceneFootprint(doc: ReturnType<typeof useStore.getState>['doc']): Vec2[] {
  const pts = doc.rooms.flatMap((r) => r.polygon)
  for (const r of doc.scans ?? []) if (r.visible) pts.push(...(scanCorners(r) ?? []))
  return pts
}

/** Camera and wall settings driven from outside (the game HUD), replacing the built-in tool strip. */
export interface SceneControl {
  /** Stand at eye level in the middle of the first room. */
  eye: boolean
  /** Bump to reset the camera to the overview. */
  resetKey: number
  autoRotate: boolean
  wallMode: WallMode
  /** Ground and sky colors. */
  ground?: string
  sky?: string
  /** Rendering style (map-layer style look). Leave out for the classic editor's plain look. */
  style?: StyleId
}

export function Scene3D({ control }: { control?: SceneControl } = {}) {
  const doc = useStore((s) => s.doc)
  const scan3d = useStore((s) => s.scan3d)
  const [ownWallMode, setWallMode] = useState<WallMode>('auto')
  const [ownCamKey, setCamKey] = useState(0)
  const [ownEye, setEye] = useState<{ pos: [number, number, number] } | null>(null)
  const wallMode = control?.wallMode ?? ownWallMode
  const style: StyleId = control?.style ?? 'classic'
  const palette = useScanPalette(style)
  const styleEnv = useMemo(() => ({ style, palette }), [style, palette])
  const stage = STYLE_STAGE[style]
  const eyePos = useMemo(() => {
    const room = doc.rooms[0]
    const scan = (doc.scans ?? []).find((r) => r.visible)
    const c = room ? centroid(room.polygon) : scan ? scanCenter(scan) : null
    return c ? { pos: [c.x, 1.6, c.y] as [number, number, number] } : null
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [control?.eye, control?.resetKey, doc.apartment.id])
  const eye = control ? (control.eye ? eyePos : null) : ownEye
  const camKey = control ? `${control.resetKey}:${control.eye}` : ownCamKey

  const initial = useMemo(() => {
    const pts = sceneFootprint(doc)
    if (!pts.length) return [6, 8, 10] as [number, number, number]
    const b = bounds(pts)
    const size = Math.max(b.maxX - b.minX, b.maxY - b.minY)
    return [(b.minX + b.maxX) / 2 + size * 0.15, size * 1.3, b.maxY + size * 0.8] as [number, number, number]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camKey, doc.apartment.id])

  const eyeLevel = () => {
    const room = doc.rooms[0]
    const scan = (doc.scans ?? []).find((r) => r.visible)
    const c = room ? centroid(room.polygon) : scan ? scanCenter(scan) : null
    if (!c) return
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
        <color attach="background" args={[stage.sky ?? control?.sky ?? '#f6f4ef']} />
        <StyleContext.Provider value={styleEnv}>
          <Scene wallMode={wallMode} eye={eye?.pos ?? null} autoRotate={control?.autoRotate && !eye} ground={control?.ground} />
        </StyleContext.Provider>
      </Canvas>
      {!control && (
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
        {(doc.scans ?? []).length > 0 && (
          <label title="How to show imported scans">
            Scan
            <select value={scan3d} onChange={(e) => useStore.getState().setUI({ scan3d: e.target.value as 'off' | 'with-model' | 'scan-only' })}>
              <option value="with-model">With model</option>
              <option value="scan-only">Scan only</option>
              <option value="off">Hidden</option>
            </select>
          </label>
        )}
        <label title="Walls between you and the room are cut down so you can see in">
          Walls
          <select value={wallMode} onChange={(e) => setWallMode(e.target.value as WallMode)}>
            <option value="auto">Cut away</option>
            <option value="low">All low</option>
            <option value="full">Full height</option>
          </select>
        </label>
      </div>
      )}
    </div>
  )
}

