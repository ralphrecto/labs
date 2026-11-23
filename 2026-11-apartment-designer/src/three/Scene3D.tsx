import { ContactShadows, Edges, Html, OrbitControls } from '@react-three/drei'
import { Canvas, invalidate, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { setCaptureRenderer } from './capture'
import { lazy, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useAnalysis } from '../analysis'
import { bounds, centroid, dist, norm, openingEndpoints, pointInPolygon, polygonArea, scale, sub, add, v, wallInwardNormal, wallSegments } from '../model/geometry'
import { clampOnHost, HOSTED, hostAt } from '../model/placement'
import { snapPlacement } from '../model/snapping'
import { isWallDecor, mountBottom, snapToWall } from '../model/wallDecor'
import type { Doc, Item, Opening, Placement, Room, Vec2 } from '../model/types'
import { activeLayout, useStore } from '../store'
import { ItemModel } from './ItemModel'
import { viewFrame } from './viewFrame'
import { scanCenter, scanCorners, useScanRuntime } from '../scan/runtime'
import { POP_MS, POP_STAGGER_MS, reduceMotion } from '../motion'
import { useGame } from '../game/gameStore'
import { isPackedBox, unpackAndCelebrate } from '../game/unpack'

// The scan backdrop pulls in the splat renderer, so it loads on first use.
const ScanBackdrop = lazy(() => import('./ScanBackdrop').then((m) => ({ default: m.ScanBackdrop })))
import { ScanPoints } from './ScanPoints'
import { floorTexture, Ink, StyleContext, StyleEnvironment, StyleLights, STYLE_STAGE, Surface, useStyle, woodTexture, type StyleId } from './style'
import { scanPalette, toonFloor, type ScanPalette } from '../scan/palette'
import { toScanData } from '../scan/align'
import { WindowModel, type WindowSpec } from './WindowModel'
import { clearOfItems, routeTo, sharedDoorways, walkObstacles, walkStep, walkTarget, wallSegmentsForWalking, type ItemBox, type WallSeg } from './walkCollision'
import { dragConflictBase, isCleanPose, settleDrop, type Conflicts } from '../game/editActions'

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
  /** Inward normal (toward the room), for the baseboard. */
  ix: number
  iz: number
  /** Outward normal and a point on the wall's outer face, for cutaway. */
  nx: number
  nz: number
  ox: number
  oz: number
}

const BASE_H = 0.08
const BASE_D = 0.015

function WallPiece({ w, mode, color, highlight }: { w: WallPieceSpec; mode: WallMode; color: string; highlight?: string }) {
  const ref = useRef<THREE.Mesh>(null)
  // The classic editor keeps its plain walls: no darker top edge, no baseboard.
  const trim = useStyle().style !== 'classic'
  const cap = useRef<THREE.Mesh>(null)
  useFrame(({ camera }) => {
    const m = ref.current
    if (!m) return
    // Keep the top edge 2 mm above the wall top whatever the cut-down squash.
    if (cap.current) cap.current.position.y = w.size[1] / 2 + 0.002 / Math.max(m.scale.y, 1e-3)
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
    // Cut-down walls in the dollhouse view would throw full-height shadows across the rooms, so
    // walls only cast shadows when they stand at full height (Full walls, or walking in).
    <>
      <mesh ref={ref} position={w.pos} rotation={[0, w.rotY, 0]} castShadow={mode === 'full'} receiveShadow>
        <boxGeometry args={w.size} />
        <Surface role="wall" color={color} highlight={highlight} />
        <Ink highlight={highlight} />
        {/* The top edge, in a darker tone so the room outlines read from above. It rides along when
            the wall is cut down (a flat plane doesn't mind the squash). */}
        {trim && <mesh ref={cap} position={[0, w.size[1] / 2 + 0.002, 0]} rotation={[-Math.PI / 2, 0, 0]} raycast={noRaycast}>
          <planeGeometry args={[w.size[0], w.size[2]]} />
          <Surface role="cap" color={color} highlight={highlight} />
        </mesh>}
      </mesh>
      {/* A baseboard along the foot of the wall's inner face. It is lower than a cut-down wall, so it
          always shows. */}
      {trim && w.y0 === 0 && w.y1 > BASE_H && (
        <mesh
          position={[w.pos[0] + w.ix * (w.size[2] / 2 + BASE_D / 2), BASE_H / 2, w.pos[2] + w.iz * (w.size[2] / 2 + BASE_D / 2)]}
          rotation={[0, w.rotY, 0]}
          receiveShadow
          raycast={noRaycast}
        >
          <boxGeometry args={[w.size[0], BASE_H, BASE_D]} />
          <Surface role="trim" color={color} />
          <Ink />
        </mesh>
      )}
    </>
  )
}

function RoomMesh({ room, wallMode, selected, doorways }: { room: Room; wallMode: WallMode; selected: boolean; doorways?: Opening[] }) {
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
        out.push({ pos: [mid.x, (y0 + y1) / 2, mid.y], size: [s1 - s0, y1 - y0 + jitter, t], rotY: -angle, y0, y1, wi, ix: n.x, iz: n.y, nx: -n.x, nz: -n.y, ox: outer.x, oz: outer.y })
      }
      // Extend past both ends by the wall thickness so corners are filled.
      // A neighbour's door through this wall opens it too (doorways), unless it overlaps one of ours.
      const ops = [...room.openings, ...(doorways ?? [])].filter((o) => o.wall === wi).sort((x, y) => x.offset - y.offset)
      let cursor = -t
      for (const o of ops) {
        piece(cursor, o.offset, 0, H)
        const top = o.sill + o.height
        piece(o.offset, o.offset + o.width, 0, Math.min(H, o.sill))
        if (top < H) piece(o.offset, o.offset + o.width, top, H)
        cursor = Math.max(cursor, o.offset + o.width)
      }
      piece(cursor, L + t, 0, H)
    })
    return out
  }, [room, doorways])

  const windows = useMemo(
    () =>
      room.openings
        .filter((o) => o.kind === 'window')
        .map((o): WindowSpec => {
          const [a, b] = wallSegments(room)[o.wall]
          const dir = norm(sub(b, a))
          const n = wallInwardNormal(room, o.wall)
          const t = room.wallThickness
          const mid = add(add(a, scale(dir, o.offset + o.width / 2)), scale(n, -t / 2))
          const outer = add(a, scale(n, -t))
          // The group's local +z is (-dir.y, dir.x) on the plan; which way is that relative to the room?
          const inward = -dir.y * n.x + dir.x * n.y >= 0 ? 1 : -1
          const height = Math.max(0.05, Math.min(o.height, room.ceilingHeight - o.sill))
          return {
            id: o.id,
            pos: [mid.x, o.sill + height / 2, mid.y],
            rotY: -Math.atan2(dir.y, dir.x),
            width: o.width,
            height,
            sill: o.sill,
            depth: t,
            inward,
            nx: -n.x,
            nz: -n.y,
            ox: outer.x,
            oz: outer.y,
          }
        }),
    [room],
  )

  const { style, palette } = useStyle()
  const floor = useFloorLook(room, floorGeom, style, palette)
  const wallColors = palette?.walls[room.id]
  const wallBase = palette?.wallColor[room.id] ?? (style === 'toon' ? '#f6ecdf' : style === 'real' ? '#ddcfbb' : '#f2f0eb')
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
      {windows.map((w) => (
        <WindowModel key={w.id} w={w} mode={wallMode} />
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

/** Height of the first visible surface along a pointer ray, other than the object handling it. */
function firstSolidHeight(e: ThreeEvent<MouseEvent>) {
  const rc = new THREE.Raycaster(e.ray.origin, e.ray.direction)
  let root: THREE.Object3D = e.object
  while (root.parent) root = root.parent
  for (const hit of rc.intersectObject(root, true)) {
    const o = hit.object as THREE.Mesh
    if (o === e.object || !o.isMesh) continue
    let shown = true
    for (let n: THREE.Object3D | null = o; n; n = n.parent) if (!n.visible) shown = false
    const mat = o.material as THREE.Material
    if (!shown || (mat && !Array.isArray(mat) && (!mat.visible || (mat.transparent && mat.opacity < 0.05)))) continue
    return hit.point.y
  }
  return 0
}

const FLAG_COLOR = { error: '#e0484e', warning: '#e0962b' } as const
const badges: Partial<Record<'error' | 'warning', THREE.CanvasTexture>> = {}
/** A round "!" badge in the flag's color, drawn once and shared. */
function flagBadge(flag: 'error' | 'warning') {
  if (badges[flag]) return badges[flag]
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const g = c.getContext('2d')!
  g.beginPath()
  g.arc(32, 32, 28, 0, Math.PI * 2)
  g.fillStyle = FLAG_COLOR[flag]
  g.fill()
  g.lineWidth = 5
  g.strokeStyle = '#ffffff'
  g.stroke()
  g.fillStyle = '#ffffff'
  g.font = '900 38px system-ui, sans-serif'
  g.textAlign = 'center'
  g.textBaseline = 'middle'
  g.fillText('!', 32, 34)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  return (badges[flag] = t)
}
const noRaycast = () => null

/** A selected piece: an outline in this color, a warm ring on the floor and a small hop. */
const SELECT_LINE = '#e0603a'
const SELECT_RING = '#e8643c'
const SELECT_GLOW = '#fff3d6'

/** A rounded-rectangle band around a w x d footprint, lying flat on the floor. */
function ringGeometry(w: number, d: number, gap: number, band: number) {
  const rr = (hw: number, hd: number, r: number, hole: boolean) => {
    const p = hole ? new THREE.Path() : new THREE.Shape()
    r = Math.min(r, hw, hd)
    p.moveTo(-hw + r, -hd)
    p.lineTo(hw - r, -hd)
    p.quadraticCurveTo(hw, -hd, hw, -hd + r)
    p.lineTo(hw, hd - r)
    p.quadraticCurveTo(hw, hd, hw - r, hd)
    p.lineTo(-hw + r, hd)
    p.quadraticCurveTo(-hw, hd, -hw, hd - r)
    p.lineTo(-hw, -hd + r)
    p.quadraticCurveTo(-hw, -hd, -hw + r, -hd)
    return p
  }
  const hw = w / 2 + gap
  const hd = d / 2 + gap
  const outer = rr(hw + band, hd + band, 0.1 + gap + band, false) as THREE.Shape
  outer.holes.push(rr(hw, hd, 0.1 + gap, true))
  const g = new THREE.ShapeGeometry(outer, 6)
  g.rotateX(-Math.PI / 2)
  return g
}

/** A soft cream band with a coral rim, so it reads on warm toon floors and pale wood alike. It pulses gently. */
function SelectRing({ w, d }: { w: number; d: number }) {
  const glow = useMemo(() => ringGeometry(w, d, 0.04, 0.07), [w, d])
  const rim = useMemo(() => ringGeometry(w, d, 0.11, 0.03), [w, d])
  useEffect(() => () => {
    glow.dispose()
    rim.dispose()
  }, [glow, rim])
  const a = useRef<THREE.MeshBasicMaterial>(null)
  const b = useRef<THREE.MeshBasicMaterial>(null)
  // The pulse keeps the canvas drawing while a piece is selected (frameloop is on demand).
  useFrame(({ clock, invalidate: more }) => {
    if (!a.current || !b.current || reduceMotion) return
    more()
    const k = 0.5 + 0.5 * Math.sin(clock.elapsedTime * 3.2)
    a.current.opacity = 0.45 + 0.3 * k
    b.current.opacity = 0.65 + 0.3 * k
  })
  const props = { transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, toneMapped: false } as const
  return (
    // Above a rug's height, so a piece standing on a rug still shows its ring.
    <group position={[0, 0.03, 0]}>
      <mesh geometry={glow} raycast={noRaycast} renderOrder={2}>
        <meshBasicMaterial ref={a} color={SELECT_GLOW} opacity={0.6} {...props} />
      </mesh>
      <mesh geometry={rim} raycast={noRaycast} renderOrder={2}>
        <meshBasicMaterial ref={b} color={SELECT_RING} opacity={0.8} {...props} />
      </mesh>
    </group>
  )
}

const HOP_MS = 260
/** Lifts its children 4 cm and sets them down again when `on` turns true (not with reduced motion). */
function Hop({ on, children }: { on: boolean; children: ReactNode }) {
  const ref = useRef<THREE.Group>(null)
  const start = useRef(0)
  useEffect(() => {
    if (on && !reduceMotion) {
      start.current = -1
      invalidate()
    }
  }, [on])
  useFrame(({ invalidate: more }) => {
    const g = ref.current
    if (!g || !start.current) return
    more()
    if (start.current < 0) start.current = performance.now()
    const t = (performance.now() - start.current) / HOP_MS
    if (t >= 1) {
      g.position.y = 0
      start.current = 0
      return
    }
    g.position.y = 0.04 * Math.sin(Math.PI * t)
  })
  return <group ref={ref}>{children}</group>
}

/**
 * A piece that just came out of a moving box pops in: scale 0 → 1.1 → 1 with a small hop, each
 * piece POP_STAGGER_MS after the one before (not with reduced motion).
 */
function Pop({ id, children }: { id: string; children: ReactNode }) {
  const unpacked = useGame((s) => s.unpacked)
  const order = unpacked ? unpacked.ids.indexOf(id) : -1
  const ref = useRef<THREE.Group>(null)
  const start = useRef(0)
  useLayoutEffect(() => {
    if (!unpacked || order < 0 || reduceMotion || Date.now() - unpacked.key > 2000) return
    start.current = performance.now() + order * POP_STAGGER_MS
    ref.current?.scale.setScalar(0.001)
    invalidate()
  }, [unpacked, order])
  useFrame(({ invalidate: more }) => {
    const g = ref.current
    if (!g || !start.current) return
    more()
    const t = (performance.now() - start.current) / POP_MS
    if (t < 0) return g.scale.setScalar(0.001)
    if (t >= 1) {
      g.scale.setScalar(1)
      g.position.y = 0
      start.current = 0
      return
    }
    const s = t < 0.6 ? 1.1 * (1 - (1 - t / 0.6) ** 3) : 1.1 - 0.1 * ((t - 0.6) / 0.4)
    g.scale.setScalar(Math.max(0.001, s))
    g.position.y = 0.12 * Math.sin(Math.PI * t)
  })
  return <group ref={ref}>{children}</group>
}

const BURST_COLORS = ['#e0603a', '#ffc24b', '#5fa67f', '#4f8be8', '#b5835a']
/** A small confetti burst where a moving box was just unpacked. */
function UnpackBurst() {
  const unpacked = useGame((s) => s.unpacked)
  if (!unpacked?.at || reduceMotion || Date.now() - unpacked.key > 2000) return null
  return (
    <Html key={unpacked.key} position={[unpacked.at.x, 0.4, unpacked.at.y]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
      <div className="goal-confetti unpack-burst" aria-hidden>
        {Array.from({ length: 18 }, (_, i) => {
          const a = (i / 18) * Math.PI * 2
          const r = 46 + ((i * 29) % 30)
          return (
            <i
              key={i}
              style={{
                background: BURST_COLORS[i % BURST_COLORS.length],
                ['--dx' as string]: `${Math.round(Math.cos(a) * r)}px`,
                ['--dy' as string]: `${Math.round(Math.sin(a) * r - 30)}px`,
                animationDelay: `${(i % 3) * 0.04}s`,
              }}
            />
          )
        })}
      </div>
    </Html>
  )
}

const WOBBLE_MS = 420
/** A gentle side-to-side shake ("no room there") each time `kick` goes up (not with reduced motion). */
function Wobble({ kick, children }: { kick: number; children: ReactNode }) {
  const ref = useRef<THREE.Group>(null)
  const start = useRef(0)
  useEffect(() => {
    if (kick && !reduceMotion) {
      start.current = -1
      invalidate()
    }
  }, [kick])
  useFrame(({ invalidate: more }) => {
    const g = ref.current
    if (!g || !start.current) return
    more()
    if (start.current < 0) start.current = performance.now()
    const t = (performance.now() - start.current) / WOBBLE_MS
    if (t >= 1) {
      g.rotation.y = 0
      start.current = 0
      return
    }
    g.rotation.y = 0.07 * Math.sin(t * Math.PI * 4) * (1 - t)
  })
  return <group ref={ref}>{children}</group>
}

const DRAG_OK = '#4cbf6b'
const DRAG_BAD = '#e0484e'
/** While dragging: the piece's footprint on the floor, green where it fits and red where it would collide. */
function DragFootprint({ w, d, ok }: { w: number; d: number; ok: boolean }) {
  return (
    <mesh position={[0, 0.035, 0]} rotation={[-Math.PI / 2, 0, 0]} raycast={noRaycast} renderOrder={3}>
      <planeGeometry args={[w + 0.04, d + 0.04]} />
      <meshBasicMaterial color={ok ? DRAG_OK : DRAG_BAD} transparent opacity={0.42} depthWrite={false} polygonOffset polygonOffsetFactor={-6} toneMapped={false} />
    </mesh>
  )
}

/**
 * Whether a camera at `cam` sees the wall piece at `p` from behind its wall, where the dollhouse
 * view cuts that wall away. The wall it hangs on: its back faces out of the room (front is
 * rot((0, 1), rotation)).
 */
function behindWall(p: Pick<Placement, 'x' | 'y' | 'rotation'>, depth: number, cam: { x: number; z: number }): boolean {
  const nx = Math.sin(p.rotation)
  const nz = -Math.cos(p.rotation)
  const ox = p.x + nx * (depth / 2)
  const oz = p.y + nz * (depth / 2)
  return (cam.x - ox) * nx + (cam.z - oz) * nz > 0
}

function PlacedItem({
  item,
  p,
  selected,
  flag,
  onGrab,
  onHover,
  dragOk,
  wobble = 0,
  lift = 0,
  wallMode,
}: {
  item: Item
  p: Placement
  /** Wall decor hides with its wall when the dollhouse view cuts that wall away. */
  wallMode?: WallMode
  /** Height of the top this piece stands on (a table lamp on its nightstand). */
  lift?: number
  selected: boolean
  flag?: 'error' | 'warning'
  onGrab: (e: ThreeEvent<PointerEvent>, p: Placement) => void
  /** The pointer came over the piece (true) or left it (false): drives the grab cursor. */
  onHover?: (on: boolean) => void
  /** Set while this piece is being dragged: whether it fits where it is. */
  dragOk?: boolean
  wobble?: number
}) {
  const { style } = useStyle()
  // The classic editor tints problem pieces. In the game a tint reads as red furniture, so the
  // piece keeps its own colors and gets an outline and a small "!" badge instead.
  const tint = style === 'classic'
  const highlight = selected ? SELECT_LINE : tint && flag === 'error' ? '#d64545' : tint && flag === 'warning' ? '#d68a1f' : undefined
  const h = Math.max(item.height, 0.02)
  const onWall = isWallDecor(item.category)
  const groupRef = useRef<THREE.Group>(null)
  const modelRef = useRef<THREE.Group>(null)
  const ghostRef = useRef<THREE.Mesh>(null)
  useFrame(({ camera }) => {
    const g = groupRef.current
    if (!g || !onWall) return
    // Seen from behind its wall it would show through (or float where the wall is cut): hide it.
    // A selected piece stays findable as a ghost outline where it hangs.
    const cut = wallMode === 'low' || (wallMode === 'auto' && behindWall(p, item.depth, camera.position))
    g.visible = !cut || selected
    if (modelRef.current) modelRef.current.visible = !cut
    if (ghostRef.current) ghostRef.current.visible = cut && selected
  })
  return (
    <group
      ref={groupRef}
      position={[p.x, lift, p.y]}
      rotation={[0, -p.rotation, 0]}
      onPointerDown={(e) => onGrab(e, p)}
      onPointerOver={onHover && ((e) => (e.stopPropagation(), onHover(true)))}
      onPointerOut={onHover && (() => onHover(false))}
    >
      <Wobble kick={wobble}>
      <Hop on={selected}>
        <group ref={modelRef}>
        <Pop id={p.id}>
        <ItemModel item={item} highlight={highlight} />
        </Pop>
        </group>
        {onWall && selected && (
          <mesh ref={ghostRef} visible={false} position={[0, mountBottom(item) + h / 2, 0]} renderOrder={5}>
            <boxGeometry args={[item.width, h, Math.max(item.depth, 0.02)]} />
            <meshBasicMaterial color={SELECT_LINE} transparent opacity={0.18} depthWrite={false} depthTest={false} />
            <Edges color={SELECT_LINE} lineWidth={2} raycast={noRaycast} renderOrder={6} />
          </mesh>
        )}
        {/* Models from a URL draw their own materials, so their outline is the bounding box. */}
        {selected && item.glbUrl && !tint && (
          <mesh position={[0, h / 2, 0]} raycast={noRaycast}>
            <boxGeometry args={[item.width + 0.02, h + 0.01, item.depth + 0.02]} />
            <meshBasicMaterial visible={false} />
            <Edges color={SELECT_LINE} lineWidth={1.8} raycast={noRaycast} />
          </mesh>
        )}
      </Hop>
      </Wobble>
      {dragOk !== undefined && !onWall && <DragFootprint w={item.width} d={item.depth} ok={dragOk} />}
      {selected && (tint ? (
        <mesh position={[0, 0.003, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[item.width + 0.06, item.depth + 0.06]} />
          <meshBasicMaterial color="#2f6fde" transparent opacity={0.25} />
        </mesh>
      ) : (
        <SelectRing w={item.width} d={item.depth} />
      ))}
      {flag && !selected && !tint && (
        <group>
          <mesh position={[0, h / 2, 0]} raycast={noRaycast}>
            <boxGeometry args={[item.width + 0.04, h + 0.02, item.depth + 0.04]} />
            <meshBasicMaterial visible={false} />
            <Edges color={FLAG_COLOR[flag]} lineWidth={2.6} raycast={noRaycast} />
          </mesh>
          <sprite position={[0, h + 0.22, 0]} scale={[0.032, 0.032, 1]} raycast={noRaycast} renderOrder={10}>
            <spriteMaterial map={flagBadge(flag)} sizeAttenuation={false} depthTest={false} transparent />
          </sprite>
        </group>
      )}
    </group>
  )
}

function Scene({
  resetKey = 0,
  eyeFov = 60,
  wallMode,
  eye,
  autoRotate,
  ground = '#e9e6df',
  inset,
  hideFlags,
  focusRoom,
  fullFrame,
}: {
  /** Bumped by the HUD's Home / recenter: the overview is framed afresh. */
  resetKey?: number
  eyeFov?: number
  wallMode: WallMode
  eye: EyeSpot | null
  autoRotate?: boolean
  ground?: string
  inset?: SceneInset
  hideFlags?: boolean
  focusRoom?: string | null
  fullFrame?: boolean
}) {
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
  const drag = useRef<{ id: string; grab: Vec2; moved: boolean; sx: number; sy: number; slop: number; start: Placement; lastClean: Placement; base: Conflicts | null; again?: boolean } | null>(null)
  /** The piece being dragged and whether it fits where it is (drives the floor footprint). */
  const [dragFx, setDragFx] = useState<{ id: string; ok: boolean } | null>(null)
  /** Pieces that just bounced back from a bad drop, and how many times (each bump wobbles once). */
  const [wobbles, setWobbles] = useState<Record<string, number>>({})

  const byId = useMemo(() => new Map(doc.items.map((i) => [i.id, i])), [doc.items])
  const placed = layout.placements.flatMap((p) => {
    const item = byId.get(p.itemId)
    return item ? [{ p, item }] : []
  })
  const pById = new Map(layout.placements.map((p) => [p.id, p]))
  /** How high a piece stands: a table lamp on its host's top, everything else on the floor. */
  // A table lamp stands on its host's top; a pendant hangs 76 cm above the dining table.
  const lift = (p: Placement) =>
    p.on ? (byId.get(pById.get(p.on)?.itemId ?? '')?.height ?? 0) + (byId.get(p.itemId)?.category === 'pendant' ? 0.76 : 0) : 0

  const hit = (e: ThreeEvent<PointerEvent>) => {
    const pt = new THREE.Vector3()
    return e.ray.intersectPlane(FLOOR, pt) ? v(pt.x, pt.z) : null
  }
  /** Where the pointer meets the level plane at height y (a host's top, for a table lamp). */
  const hitAt = (e: ThreeEvent<PointerEvent>, y: number) => {
    const pt = new THREE.Vector3()
    return e.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -y), pt) ? v(pt.x, pt.z) : null
  }

  // The canvas cursor: an open hand over a piece, a closed one while a piece is being dragged.
  const gl = useThree((s) => s.gl)
  const hovered = useRef(false)
  const setCursor = () => {
    gl.domElement.style.cursor = drag.current?.moved ? 'grabbing' : hovered.current && !eye ? 'grab' : ''
  }
  const onHover = (on: boolean) => {
    hovered.current = on
    setCursor()
  }
  useEffect(() => () => void (gl.domElement.style.cursor = ''), [gl])

  const onGrab = (e: ThreeEvent<PointerEvent>, p: Placement) => {
    // Walking around is for looking: furniture isn't picked up, and taps go to walking.
    if (e.button !== 0 || eye) return
    e.stopPropagation()
    floorDown.current = false
    const s = useStore.getState()
    // A second tap on a moving box that's already selected unpacks it (see onUp).
    const again = s.selection?.kind === 'placement' && s.selection.id === p.id
    s.select({ kind: 'placement', id: p.id })
    const grabbed = byId.get(p.itemId)
    // Wall decor is grabbed at its own height, so it tracks the pointer along the wall.
    const w = p.on ? hitAt(e, lift(p)) : grabbed && isWallDecor(grabbed.category) ? hitAt(e, mountBottom(grabbed) + grabbed.height / 2) : hit(e)
    if (!w) return
    // A finger jitters: it has to travel a few pixels before a tap turns into a drag.
    const ne = e.nativeEvent
    const slop = ne.pointerType === 'touch' ? 8 : 3
    drag.current = { id: p.id, grab: sub(w, v(p.x, p.y)), moved: false, sx: ne.clientX, sy: ne.clientY, slop, start: p, lastClean: p, base: null, again }
    if (controls.current) controls.current.enabled = false
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }

  const onMove = (e: ThreeEvent<PointerEvent>) => {
    const d = drag.current
    if (!d) return
    const s = useStore.getState()
    const cur = activeLayout(s.doc).placements.find((x) => x.id === d.id)
    const item = cur && byId.get(cur.itemId)
    if (!cur || !item) return
    if (HOSTED.has(item.category) && cur.on) {
      // A table lamp slides over table tops: on a top it's fine, off every top it shows red.
      const wl = hitAt(e, lift(cur))
      if (!wl) return
      if (!d.moved) {
        if (Math.hypot(e.nativeEvent.clientX - d.sx, e.nativeEvent.clientY - d.sy) < d.slop) return
        s.beginGesture()
        d.moved = true
        setCursor()
      }
      const pt = sub(wl, d.grab)
      const layoutNow = activeLayout(s.doc)
      const host = hostAt(s.doc, layoutNow, pt, item, cur.on)
      let next: Placement
      let ok = false
      if (host) {
        const c = clampOnHost(host, item, pt)
        next = { ...cur, x: c.x, y: c.y, rotation: host.p.rotation, on: host.id }
        ok = true
        d.lastClean = next
      } else next = { ...cur, x: pt.x, y: pt.y }
      setDragFx((was) => (was?.id === d.id && was.ok === ok ? was : { id: d.id, ok }))
      s.updatePlacement(d.id, { x: next.x, y: next.y, rotation: next.rotation, on: next.on })
      return
    }
    if (isWallDecor(item.category)) {
      // Wall decor slides along the walls: it snaps to the nearest wall face it fits on.
      const ww = hitAt(e, mountBottom(item) + item.height / 2)
      if (!ww) return
      if (!d.moved) {
        if (Math.hypot(e.nativeEvent.clientX - d.sx, e.nativeEvent.clientY - d.sy) < d.slop) return
        s.beginGesture()
        d.moved = true
        setCursor()
      }
      const at = sub(ww, d.grab)
      let spot = snapToWall(s.doc, activeLayout(s.doc), item, at, d.id)
      if (!spot) return
      // The wall nearest the pointer may be the one cut away from this camera, where the piece
      // would vanish mid-drag: nudge the pointer into the room for a wall that shows, if one is
      // near, or else keep the piece where it shows (orbiting round reveals that wall to hang on).
      const cam = e.camera.position
      if (wallMode === 'auto' && behindWall(spot, item.depth, cam)) {
        const n = v(Math.sin(spot.rotation), -Math.cos(spot.rotation))
        let alt: typeof spot | null = null
        for (const k of [0.35, 0.7, 1.1]) {
          const a = snapToWall(s.doc, activeLayout(s.doc), item, sub(at, scale(n, k)), d.id)
          if (a && !behindWall(a, item.depth, cam) && dist(a, at) < k + 0.9) {
            alt = a
            break
          }
        }
        if (alt) spot = alt
        else if (!behindWall(cur, item.depth, cam)) return
      }
      d.lastClean = { ...cur, x: spot.x, y: spot.y, rotation: spot.rotation }
      s.updatePlacement(d.id, { x: spot.x, y: spot.y, rotation: spot.rotation })
      return
    }
    const w = hit(e)
    if (!w) return
    const others = placed.filter((x) => x.p.id !== d.id && !x.p.on && !isWallDecor(x.item.category))
    const solid = others.map((x) => ({ id: x.p.id, item: x.item, p: x.p }))
    if (!d.moved) {
      if (Math.hypot(e.nativeEvent.clientX - d.sx, e.nativeEvent.clientY - d.sy) < d.slop) return
      s.beginGesture()
      d.moved = true
      setCursor()
      d.base = dragConflictBase(item, cur, solid, s.doc.rooms)
    }
    // The grab point stays under the pointer, so the piece tracks the finger 1:1 on the floor.
    const proposed = { ...cur, x: w.x - d.grab.x, y: w.y - d.grab.y }
    const res = e.altKey ? { placement: proposed } : snapPlacement(item, proposed, s.doc.rooms, others, snap)
    const next = { ...cur, x: res.placement.x, y: res.placement.y, rotation: res.placement.rotation }
    const ok = !!d.base && isCleanPose(item, next, solid, s.doc.rooms, d.base)
    if (ok) d.lastClean = next
    setDragFx((was) => (was?.id === d.id && was.ok === ok ? was : { id: d.id, ok }))
    s.updatePlacement(d.id, { x: next.x, y: next.y, rotation: next.rotation })
  }

  // A click on the floor (not a drag to orbit) deselects, unless a piece is within a few pixels
  // of it: small pieces are hard to hit, so a near miss picks the nearest one instead.
  const floorDown = useRef(false)
  const onFloorClick = (e: ThreeEvent<MouseEvent>) => {
    const down = floorDown.current
    floorDown.current = false
    if (!down || e.delta >= 5 || eye) return
    const near = nearestPlacementOnScreen(e, placed.map((x) => ({ ...x, lift: lift(x.p) })))
    useStore.getState().select(near ? { kind: 'placement', id: near } : null)
  }

  const onUp = () => {
    const d = drag.current
    if (d?.moved) {
      // Let go somewhere it doesn't fit: slide to the nearest clean spot close by, or go back to the
      // last clean one, with a little wobble. Fiddling never costs stars.
      const s = useStore.getState()
      const cur = activeLayout(s.doc).placements.find((x) => x.id === d.id)
      const item = cur && byId.get(cur.itemId)
      if (cur && item && HOSTED.has(item.category) && cur.on) {
        // Let go off every table top: back to the last top it stood on.
        if (!hostAt(s.doc, activeLayout(s.doc), v(cur.x, cur.y), item, cur.on)) {
          s.updatePlacement(d.id, { x: d.lastClean.x, y: d.lastClean.y, rotation: d.lastClean.rotation, on: d.lastClean.on })
          setWobbles((w) => ({ ...w, [d.id]: (w[d.id] ?? 0) + 1 }))
        }
      } else if (cur && item && d.base && !isWallDecor(item.category)) {
        const solid = placed.filter((x) => x.p.id !== d.id && !x.p.on && !isWallDecor(x.item.category)).map((x) => ({ id: x.p.id, item: x.item, p: x.p }))
        const r = settleDrop(item, cur, d.lastClean, solid, s.doc.rooms, d.base)
        if (r.bounced) {
          s.updatePlacement(d.id, { x: r.placement.x, y: r.placement.y, rotation: r.placement.rotation })
          setWobbles((w) => ({ ...w, [d.id]: (w[d.id] ?? 0) + 1 }))
        }
      }
      s.endGesture()
    }
    if (dragFx) setDragFx(null)
    drag.current = null
    setCursor()
    if (controls.current) controls.current.enabled = true
    if (d && !d.moved && d.again && isPackedBox(d.id)) unpackAndCelebrate(d.id)
  }

  // A drop that lands off the furniture (or a lost pointer capture) still ends the drag.
  const upRef = useRef(onUp)
  upRef.current = onUp
  useEffect(() => {
    const end = () => {
      if (drag.current) upRef.current()
    }
    window.addEventListener('pointerup', end)
    window.addEventListener('pointercancel', end)
    return () => {
      window.removeEventListener('pointerup', end)
      window.removeEventListener('pointercancel', end)
    }
  }, [])

  const entries = useScanRuntime((s) => s.entries)
  const { target, extent } = useMemo(() => {
    const pts = sceneFootprint(doc)
    if (!pts.length) return { target: new THREE.Vector3(0, 0, 0), extent: 6 }
    const b = bounds(pts)
    return {
      target: new THREE.Vector3((b.minX + b.maxX) / 2, 0, (b.minY + b.maxY) / 2),
      // Half the footprint's diagonal, plus a meter of margin, for the shadow camera and ground shadows.
      extent: Math.hypot(b.maxX - b.minX, b.maxY - b.minY) / 2 + 1,
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc.rooms, doc.scans, entries])
  // Walking in: doors open through to the room behind them, and the walls stop you.
  const doorways = useMemo(() => sharedDoorways(doc.rooms), [doc.rooms])
  const walls = useMemo(() => wallSegmentsForWalking(doc.rooms, doorways), [doc.rooms, doorways])
  // Furniture you walk around (rugs you walk over).
  const boxes = useMemo<ItemBox[]>(
    () => placed.filter(({ item }) => !item.floorLevel && !isWallDecor(item.category)).map(({ p, item }) => ({ x: p.x, y: p.y, hw: item.width / 2, hd: item.depth / 2, rotation: p.rotation })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [layout.placements, doc.items],
  )
  /** Where a tap-to-walk is heading: the points still to pass, through doorways when needed. */
  const glide = useRef<Vec2[] | null>(null)
  // Lifelike and Clay sit furniture on soft contact shadows. They're added a moment after the style
  // switches, so the switch itself shows at once instead of waiting on the extra shaders.
  const wantsContact = style === 'real' || style === 'clay'
  const [contact, setContact] = useState(false)
  useEffect(() => {
    setContact(false)
    if (!wantsContact) return
    const t = setTimeout(() => setContact(true), 700)
    return () => clearTimeout(t)
  }, [wantsContact, style])
  const shadowKey = useMemo(() => {
    let n = 0
    for (const { p, item } of placed) n = (n * 31 + Math.round(p.x * 100) * 7 + Math.round(p.y * 100) * 13 + Math.round(p.rotation * 100) + item.width * 10 + item.depth * 10) % 1e9
    return `${n}:${doc.rooms.length}`
  }, [placed, doc.rooms])

  return (
    <>
      <StyleLights target={target} extent={extent} />
      <StyleEnvironment />
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[target.x, -0.01, target.z]}
        receiveShadow
        onPointerDown={() => (floorDown.current = true)}
        onClick={onFloorClick}
        visible={!scanOnly && stage.showGround}
      >
        <planeGeometry args={[200, 200]} />
        {style === 'toon' ? <meshToonMaterial color={stage.ground ?? ground} /> : <meshStandardMaterial color={stage.ground ?? ground} roughness={1} envMapIntensity={0.25} />}
      </mesh>
      {contact && !scanOnly && (
        // Rendering contact shadows costs two extra passes, so they're drawn once and redrawn only
        // when furniture or rooms change (the key), not every frame.
        <ContactShadows
          key={shadowKey}
          frames={2}
          position={[target.x, 0.004, target.z]}
          scale={extent * 2}
          opacity={0.45}
          blur={2.4}
          far={1.2}
          resolution={512}
          color="#3a2a1c"
        />
      )}
      {eye && (
        // Tap the floor to walk there. Only a tap whose first hit is at floor level (the floor, or
        // a rug on it) counts, so taps on furniture or through walls don't move you.
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[target.x, 0.006, target.z]}
          onClick={(e) => {
            if (e.delta > 8 || firstSolidHeight(e) > 0.06) return
            const pt = v(e.point.x, e.point.z)
            if (doc.rooms.length && !doc.rooms.some((r) => pointInPolygon(pt, r.polygon))) return
            // Stop a little short of the spot, so you arrive looking at it rather than standing on it.
            const cam = e.camera.position
            const from = v(cam.x, cam.z)
            const route = routeTo(from, pt, doc.rooms, walls)
            const last = route.length > 1 ? route[route.length - 2] : from
            const len = dist(last, pt)
            if (len > 1) route[route.length - 1] = add(last, scale(norm(sub(pt, last)), len - 0.5))
            // ...and never inside a piece of furniture: stop about 0.3 m short of it.
            route[route.length - 1] = walkTarget(route[route.length - 1], last, boxes, walls)
            glide.current = route
            invalidate()
          }}
        >
          <planeGeometry args={[200, 200]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      )}
      {style === 'blueprint' && <gridHelper args={[60, 120, '#7fb0ee', '#3a6fb5']} position={[target.x, -0.005, target.z]} />}
      {!scanOnly &&
        doc.rooms.map((r) => <RoomMesh key={r.id} room={r} doorways={doorways.get(r.id)} wallMode={eye ? 'full' : wallMode} selected={selection?.kind === 'room' && selection.id === r.id} />)}
      {(doc.scans ?? []).length > 0 && showScan && (
        <Suspense fallback={null}>
          <ScanBackdrop eye={!!eye} force={style === 'scan'} />
        </Suspense>
      )}
      {(style === 'classic' || style === 'scan') && <ScanPoints />}
      <group onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        {placed.map(({ p, item }) => (
          <PlacedItem
            key={p.id}
            item={item}
            p={p}
            selected={selection?.kind === 'placement' && selection.id === p.id}
            flag={hideFlags ? undefined : analysis.flagged.get(p.id)}
            onGrab={onGrab}
            onHover={onHover}
            dragOk={dragFx?.id === p.id ? dragFx.ok : undefined}
            wobble={wobbles[p.id]}
            lift={lift(p)}
            wallMode={wallMode}
          />
        ))}
      </group>
      <UnpackBurst />
      <ScreenFloor />
      <CameraSpot eye={eye} eyeFov={eyeFov} />
      <DemandFrames spin={autoRotate} />
      {!eye && <KeyOrbit />}
      {!eye && <FaceWall wallMode={wallMode} />}
      {!eye && <FitCamera key={`${(doc.scans ?? []).length}:${resetKey}`} inset={inset} spin={autoRotate} focus={focusRoom} full={fullFrame} />}
      <ViewOffset inset={eye ? undefined : inset} full={fullFrame} />
      {eye ? (
        <>
          <OrbitControls ref={controls} makeDefault target={eyeTarget(eye)} enableZoom={false} enablePan={false} rotateSpeed={-0.4} />
          <EyeWalk controls={controls} glide={glide} rooms={doc.rooms} walls={walls} boxes={boxes} />
        </>
      ) : (
        // Left-drag orbits, right-drag pans, the wheel zooms towards the pointer.
        <OrbitControls ref={controls} target={target} makeDefault zoomToCursor maxPolarAngle={Math.PI / 2 - 0.02} minDistance={0.3} maxDistance={60} autoRotate={autoRotate} autoRotateSpeed={0.6} />
      )}
    </>
  )
}

/** Within this many CSS px of a missed click, a piece is picked anyway. */
const NEAR_PICK_PX = 12

/** The placement whose on-screen box (footprint and top) is nearest the click, if within NEAR_PICK_PX. */
function nearestPlacementOnScreen(e: ThreeEvent<MouseEvent>, placed: { p: Placement; item: Item; lift?: number }[]): string | null {
  const camera = e.camera
  const el = e.nativeEvent.target as HTMLElement | null
  const W = el?.clientWidth || 1
  const H = el?.clientHeight || 1
  const px = ((e.pointer.x + 1) / 2) * W
  const py = ((1 - e.pointer.y) / 2) * H
  const q = new THREE.Vector3()
  let best: string | null = null
  let bestD = NEAR_PICK_PX
  for (const { p, item, lift = 0 } of placed) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    const c = Math.cos(p.rotation), s = Math.sin(p.rotation)
    for (const [lx, ly] of [[-1, -1], [1, -1], [1, 1], [-1, 1]])
      for (const h of [lift, lift + Math.max(item.height, 0.02)]) {
        const ox = (lx * item.width) / 2, oy = (ly * item.depth) / 2
        q.set(p.x + ox * c - oy * s, h, p.y + ox * s + oy * c).project(camera)
        if (q.z > 1) continue
        const sx = ((q.x + 1) / 2) * W, sy = ((1 - q.y) / 2) * H
        x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy)
      }
    if (!Number.isFinite(x0)) continue
    const d = Math.hypot(Math.max(x0 - px, 0, px - x1), Math.max(y0 - py, 0, py - y1))
    if (d < bestD) {
      bestD = d
      best = p.id
    }
  }
  return best
}

/** Screen space (CSS px) covered by HUD at the top and bottom of the pane: the visible stage is in between. */
export interface SceneInset {
  top: number
  bottom: number
  /** A column of buttons down the right edge (the phone camera buttons): the home is framed beside it. */
  right?: number
  /** Room for the selection bar above the drawer when nothing is selected yet (the plan keeps it clear). */
  ctx?: number
}

/**
 * Whether the player has orbited or zoomed the game overview since it was last framed, so the
 * game's View button can offer to recenter first. Plain module state with listeners, like viewFrame.
 */
export const camMoved = {
  value: false,
  listeners: new Set<() => void>(),
  set(v: boolean) {
    if (camMoved.value === v) return
    camMoved.value = v
    camMoved.listeners.forEach((l) => l())
  },
}

/** Width of the pane the home is framed in: the whole width less the right-hand button column. */
const visibleWidth = (w: number, inset?: SceneInset) => Math.max(w * 0.5, w - (inset?.right ?? 0))

/** Height of the part of the pane the HUD leaves visible, never less than a quarter of it. */
const visibleHeight = (h: number, inset?: SceneInset) => Math.max(h * 0.25, h - (inset?.top ?? 0) - (inset?.bottom ?? 0))

/** How much of the visible stage the home spans across its tighter direction: about a 10% margin each side. */
const FILL = 0.8
/** Phones (portrait, narrow): the stage is small, so the home runs closer to its edges. */
const PHONE_FILL = 0.88
/** Look mode: nearly edge to edge. */
const FULL_FILL = 0.94
/** Laptops and desktops: the drawer is a side column, so the home can take more of what's left. */
const DESKTOP_FILL = 0.9
/** Look mode in the game: framed between the look bar and the mode bar, with a little margin. */
const LOOK_FILL = 0.9
/** Keep in step with DESKTOP_SIDE_DRAWER in GameApp.tsx and the side-drawer block in buy.css. */
const isDesktop = (w: number, h: number) => (w >= 1100 && h >= 700) || (w >= 1000 && h >= 640 && w > h)
/** Look mode frames between the look bar and the mode bar on wider screens; phones keep the whole screen. */
const lookInset = (inset: SceneInset | undefined, full: boolean | undefined, w: number, h: number) => (full && (w <= 700 || h <= 500) ? undefined : inset)
/** The overview looks down from the front, a little to the right. */
const OVERVIEW_DIR = new THREE.Vector3(0.15, 1.0, 0.75).normalize()
const UP = new THREE.Vector3(0, 1, 0)

/**
 * The 3D points the overview has to keep in view: every room corner (pushed out by the wall
 * thickness) at the floor and at the ceiling. With no rooms yet, the visible scans' footprints.
 */
function framePoints(doc: ReturnType<typeof useStore.getState>['doc'], focus?: string | null): THREE.Vector3[] {
  const out: THREE.Vector3[] = []
  if (doc.rooms.length) {
    const only = focus ? doc.rooms.filter((r) => r.id === focus) : []
    for (const r of only.length ? only : doc.rooms) {
      const t = r.wallThickness
      for (const p of r.polygon)
        for (const [dx, dz] of [[-t, -t], [t, -t], [t, t], [-t, t]])
          for (const y of [0, r.ceilingHeight]) out.push(new THREE.Vector3(p.x + dx, y, p.y + dz))
    }
    return out
  }
  for (const r of doc.scans ?? []) if (r.visible) for (const p of scanCorners(r) ?? []) for (const y of [0, 2.4]) out.push(new THREE.Vector3(p.x, y, p.y))
  return out
}

/**
 * Where to put the orbit target and how far back to stand so `pts` fill the visible part of the
 * pane (between the HUD at the top and bottom) to FILL, looking along `-dir`. The target is moved
 * so the home's picture is centred on it (ViewOffset centres the target in the visible band), and
 * stays on the floor. With `spin` (the Look mode turntable) the fit holds all the way round.
 */
function fitView(
  pts: THREE.Vector3[],
  dir: THREE.Vector3,
  fovDeg: number,
  width: number,
  height: number,
  inset?: SceneInset,
  spin = false,
  fill = FILL,
): { target: THREE.Vector3; dist: number } {
  const t = Math.tan((fovDeg * Math.PI) / 360)
  const ky = t * (visibleHeight(height, inset) / height) * fill
  const kx = t * (visibleWidth(width, inset) / height) * fill
  const box = new THREE.Box3().setFromPoints(pts)
  let target = new THREE.Vector3((box.min.x + box.max.x) / 2, 0, (box.min.z + box.max.z) / 2)
  const axes = (d: THREE.Vector3) => {
    const f = d.clone().negate()
    const r = new THREE.Vector3().crossVectors(f, UP).normalize()
    const u = new THREE.Vector3().crossVectors(r, f)
    return { f, r, u }
  }
  const q = new THREE.Vector3()
  const solve = (tg: THREE.Vector3, d: THREE.Vector3) => {
    const { f, r, u } = axes(d)
    let D = 0.5
    for (const p of pts) {
      q.subVectors(p, tg)
      const z = q.dot(f)
      D = Math.max(D, Math.abs(q.dot(r)) / kx - z, Math.abs(q.dot(u)) / ky - z)
    }
    return D
  }
  if (spin) {
    let D = 0
    for (let i = 0; i < 12; i++) D = Math.max(D, solve(target, dir.clone().applyAxisAngle(UP, (i / 12) * Math.PI * 2)))
    return { target, dist: D }
  }
  const { f, r, u } = axes(dir)
  let D = solve(target, dir)
  for (let it = 0; it < 6; it++) {
    // Picture extents, in tangent units around the view axis.
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    for (const p of pts) {
      q.subVectors(p, target)
      const z = D + q.dot(f)
      const sx = q.dot(r) / z
      const sy = q.dot(u) / z
      x0 = Math.min(x0, sx)
      x1 = Math.max(x1, sx)
      y0 = Math.min(y0, sy)
      y1 = Math.max(y1, sy)
    }
    const cx = (x0 + x1) / 2
    const cy = (y0 + y1) / 2
    if (Math.abs(cx) < 1e-4 && Math.abs(cy) < 1e-4) break
    // Slide the camera and target sideways to centre the picture, then run the target down the
    // view ray to the floor (which doesn't change the picture), and refit the distance.
    const shifted = target.clone().addScaledVector(r, cx * D).addScaledVector(u, cy * D)
    const s = f.y < -1e-3 ? -shifted.y / f.y : 0
    target = shifted.addScaledVector(f, s)
    D = solve(target, dir)
  }
  return { target, dist: D }
}

/**
 * Frame the whole home in the part of the pane the HUD leaves visible, and keep it framed as the
 * drawer and selection bar come and go (easing over about 300 ms), until the player moves the
 * camera themselves. Recentering (a new Canvas) starts it again.
 */
/** Lets the HUD map screen points to the floor and back through this camera (drawer drag and drop). */
function ScreenFloor() {
  const camera = useThree((s) => s.camera)
  const gl = useThree((s) => s.gl)
  useEffect(() => {
    const ray = new THREE.Raycaster()
    const ndc = (x: number, y: number) => {
      const r = gl.domElement.getBoundingClientRect()
      return new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1)
    }
    const toFloor = (x: number, y: number) => {
      ray.setFromCamera(ndc(x, y), camera)
      const pt = new THREE.Vector3()
      return ray.ray.intersectPlane(FLOOR, pt) ? v(pt.x, pt.z) : null
    }
    const toScreen = (p: Vec2, h = 0) => {
      const q = new THREE.Vector3(p.x, h, p.y).project(camera)
      if (q.z > 1) return null
      const r = gl.domElement.getBoundingClientRect()
      return { x: r.left + ((q.x + 1) / 2) * r.width, y: r.top + ((1 - q.y) / 2) * r.height }
    }
    viewFrame.toFloor = toFloor
    viewFrame.toScreen = toScreen
    // The 3D view unmounts on its own schedule: leave the plan's mapping alone if it's already in.
    return () => {
      if (viewFrame.toFloor === toFloor) viewFrame.toFloor = null
      if (viewFrame.toScreen === toScreen) viewFrame.toScreen = null
    }
  }, [camera, gl])
  return null
}

/**
 * Puts the camera where the view mode wants it when walking in or back out, on the canvas that
 * stays mounted: eye height with a wide lens inside, the overview lens outside (FitCamera, after
 * this in the tree, then frames the home).
 */
function CameraSpot({ eye, eyeFov }: { eye: EyeSpot | null; eyeFov: number }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const first = useRef(true)
  useLayoutEffect(() => {
    // The canvas was created with the right spot and lens.
    if (first.current) return void (first.current = false)
    if (eye) camera.position.set(...eye.pos)
    camera.fov = eye ? eyeFov : 45
    camera.updateProjectionMatrix()
    invalidate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eye])
  return null
}

/** How long after a change in the stores the scene keeps drawing (effects settle, HUD insets ease). */
const SETTLE_MS = 450
/** How long OrbitControls' damping takes to come to rest after a release (dampingFactor 0.05). */
const DAMP_MS = 1500

/**
 * The canvas draws on demand. Any change to the home or the game state (a drag, a placement, a
 * mode or style switch, evening light) and any re-render of the scene asks for frames for a short
 * while, so effects that set three.js state directly are drawn too. Also gives photo capture a
 * synchronous render (capture.ts).
 */
function DemandFrames({ spin }: { spin?: boolean }) {
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  const camera = useThree((s) => s.camera)
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null
  const until = useRef(0)
  const kick = (ms = SETTLE_MS) => {
    until.current = Math.max(until.current, performance.now() + ms)
    invalidate()
  }
  useEffect(() => kick())
  useEffect(() => {
    const k = () => kick()
    const offs = [useStore.subscribe(k), useGame.subscribe(k)]
    // A lost and restored WebGL context, or a page back from the bfcache, needs a fresh frame or the canvas stays blank.
    const canvas = gl.domElement
    canvas.addEventListener('webglcontextrestored', k)
    window.addEventListener('resize', k)
    window.addEventListener('pageshow', k)
    document.addEventListener('visibilitychange', k)
    return () => {
      offs.forEach((f) => f())
      canvas.removeEventListener('webglcontextrestored', k)
      window.removeEventListener('resize', k)
      window.removeEventListener('pageshow', k)
      document.removeEventListener('visibilitychange', k)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gl])
  // Letting go of an orbit (or stopping the spin) leaves the controls damping to a stop: draw the tail.
  useEffect(() => {
    if (!controls) return
    const end = () => kick(DAMP_MS)
    controls.addEventListener('end', end)
    return () => controls.removeEventListener('end', end)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controls])
  const spun = useRef(spin)
  useEffect(() => {
    if (spun.current && !spin) kick(DAMP_MS)
    spun.current = spin
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spin])
  useEffect(() => {
    setCaptureRenderer(() => gl.render(scene, camera))
    return () => setCaptureRenderer(null)
  }, [gl, scene, camera])
  useFrame(({ invalidate: more }) => {
    if (performance.now() < until.current) more()
  })
  return null
}

/** Smallest signed angle from `a` to `b`. */
const turnTo = (a: number, b: number) => Math.atan2(Math.sin(b - a), Math.cos(b - a))

/**
 * Selecting (or buying) a wall piece whose wall the dollhouse view has cut away eases the camera
 * round the home so that wall is seen from inside the room, rather than leaving the new piece hidden.
 */
function FaceWall({ wallMode }: { wallMode: WallMode }) {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null
  const camera = useThree((s) => s.camera)
  const goal = useRef<number | null>(null)
  const self = useRef(false)
  const emit = (type: 'start' | 'change' | 'end') => {
    if (!controls) return
    self.current = true
    controls.dispatchEvent({ type } as never)
    self.current = false
  }
  useEffect(() => {
    if (!controls || wallMode !== 'auto') return
    let last: string | null = null
    const check = (st: ReturnType<typeof useStore.getState>) => {
      const sel = st.selection
      const id = sel?.kind === 'placement' ? sel.id : null
      if (id === last) return
      last = id
      if (!id) return
      const p = activeLayout(st.doc)?.placements.find((x) => x.id === id)
      const item = p && st.doc.items.find((i) => i.id === p.itemId)
      if (!p || !item || !isWallDecor(item.category) || !behindWall(p, item.depth, camera.position)) return
      // Face the wall from inside the room (the camera's offset along the wall's inward normal),
      // turned a little towards where the camera already is, for a three-quarter view.
      const face = -p.rotation
      const off = camera.position.clone().sub(controls.target)
      const az = Math.atan2(off.x, off.z)
      const to = face + Math.sign(turnTo(face, az) || 1) * 0.45
      if (reduceMotion) {
        emit('start')
        off.applyAxisAngle(UP, turnTo(az, to))
        camera.position.copy(controls.target).add(off)
        controls.update()
        emit('change')
        emit('end')
        invalidate()
        return
      }
      goal.current = to
      emit('start')
      invalidate()
    }
    check(useStore.getState())
    const off = useStore.subscribe(check)
    // The player taking hold of the view lets go of the turn.
    const cancel = () => {
      if (!self.current) goal.current = null
    }
    controls.addEventListener('start', cancel)
    return () => {
      off()
      controls.removeEventListener('start', cancel)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controls, camera, wallMode])
  useFrame((_, dt) => {
    const g = goal.current
    if (g === null || !controls) return
    invalidate()
    const off = camera.position.clone().sub(controls.target)
    const az = Math.atan2(off.x, off.z)
    const left = turnTo(az, g)
    const k = 1 - Math.exp(-Math.min(dt, 0.05) * 6)
    const step = Math.abs(left) < 0.003 ? left : left * k
    off.applyAxisAngle(UP, step)
    camera.position.copy(controls.target).add(off)
    controls.update()
    emit('change')
    if (step === left) {
      goal.current = null
      emit('end')
    }
  })
  return null
}

/** Keyboard orbit for the overview: Q / E turn around the home, + / − zoom. */
const ORBIT_STEP = Math.PI / 12
const ZOOM_STEP = 0.85
function KeyOrbit() {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null
  const camera = useThree((s) => s.camera)
  useEffect(() => {
    if (!controls) return
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable) return
      if (e.metaKey || e.ctrlKey || e.altKey || document.querySelector('[aria-modal="true"]')) return
      const k = e.key.toLowerCase()
      const turn = k === 'q' ? -1 : k === 'e' ? 1 : 0
      const zoom = k === '+' || k === '=' ? ZOOM_STEP : k === '-' || k === '_' ? 1 / ZOOM_STEP : 0
      if (!turn && !zoom) return
      e.preventDefault()
      const off = camera.position.clone().sub(controls.target)
      if (turn) off.applyAxisAngle(new THREE.Vector3(0, 1, 0), turn * ORBIT_STEP)
      if (zoom) off.setLength(THREE.MathUtils.clamp(off.length() * zoom, controls.minDistance, controls.maxDistance))
      // Reads as a user move: the framing lets go and the recenter button lights up (FitCamera).
      controls.dispatchEvent({ type: 'start' } as never)
      camera.position.copy(controls.target).add(off)
      controls.update()
      controls.dispatchEvent({ type: 'change' } as never)
      controls.dispatchEvent({ type: 'end' } as never)
      invalidate()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [controls, camera])
  return null
}

/**
 * The overview direction, turned in quarter turns round the home to the side that shows the most
 * wall pieces from inside their room, so a fresh view or a photo doesn't miss the gallery wall
 * (counted by piece, so one big pair of curtains doesn't outweigh a wall of frames, and weighted
 * by how squarely each wall faces the camera; area breaks ties). With no decor, or a tie, the
 * usual overview.
 */
export function decoratedDir(doc: Doc): THREE.Vector3 {
  const layout = activeLayout(doc)
  const byId = new Map(doc.items.map((i) => [i.id, i]))
  const pieces = (layout?.placements ?? []).flatMap((p) => {
    const it = byId.get(p.itemId)
    // One per piece, plus a hair for its area (a big piece wins a tie).
    return it && isWallDecor(it.category) ? [{ p, a: 1 + Math.min(it.width * it.height, 10) * 1e-3 }] : []
  })
  let best = OVERVIEW_DIR.clone()
  let bestScore = -1
  for (let q = 0; q < 4; q++) {
    const dir = OVERVIEW_DIR.clone().applyAxisAngle(UP, (q * Math.PI) / 2)
    // Seen from far off along `dir`: a piece shows when its wall's outward normal faces away from
    // the camera, and reads better the more squarely its wall faces the camera.
    const h = Math.hypot(dir.x, dir.z)
    const score = pieces.reduce((sum, { p, a }) => sum + a * Math.max(0, (Math.cos(p.rotation) * dir.z - Math.sin(p.rotation) * dir.x) / h), 0)
    if (score > bestScore + 1e-6) {
      best = dir
      bestScore = score
    }
  }
  return best
}

function FitCamera({ inset, spin, focus, full }: { inset?: SceneInset; spin?: boolean; focus?: string | null; full?: boolean }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const size = useThree((s) => s.size)
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null
  const rooms = useStore((s) => s.doc.rooms)
  const scans = useStore((s) => s.doc.scans)
  const touched = useRef(false)
  const placed = useRef(false)
  const goal = useRef<{ target: THREE.Vector3; dist: number } | null>(null)
  // Focusing on a room (or back on the whole home) is a deliberate reframe, even after orbiting.
  const lastFocus = useRef(focus)
  if (lastFocus.current !== focus) {
    lastFocus.current = focus
    touched.current = false
  }
  useEffect(() => {
    if (!controls) return
    const stop = () => {
      touched.current = true
      goal.current = null
    }
    // A drag (not just a tap) moves the view off its framing: the View button then recenters.
    let dragging = false
    const down = () => (dragging = true)
    const up = () => (dragging = false)
    const change = () => dragging && camMoved.set(true)
    camMoved.set(false)
    controls.addEventListener('start', stop)
    controls.addEventListener('start', down)
    controls.addEventListener('end', up)
    controls.addEventListener('change', change)
    return () => {
      controls.removeEventListener('start', stop)
      controls.removeEventListener('start', down)
      controls.removeEventListener('end', up)
      controls.removeEventListener('change', change)
    }
  }, [controls])
  const placedFor = useRef(controls)
  useLayoutEffect(() => {
    if (touched.current || size.width < 10 || !controls) return
    // New controls (back out of walking): frame afresh from the overview direction.
    if (placedFor.current !== controls) {
      placedFor.current = controls
      placed.current = false
    }
    const pts = framePoints(useStore.getState().doc, focus)
    if (!pts.length) return
    const dir = placed.current ? camera.position.clone().sub(controls.target).normalize() : decoratedDir(useStore.getState().doc)
    // Look mode has no drawer: the home fills the screen between the look bar and the mode bar
    // (the game passes their inset), or the whole screen when there is no HUD.
    const frame = lookInset(inset, full, size.width, size.height)
    const fill = full ? (frame ? LOOK_FILL : FULL_FILL) : inset && size.width <= 700 && size.height > size.width ? PHONE_FILL : inset && isDesktop(size.width, size.height) ? DESKTOP_FILL : FILL
    const fit = fitView(pts, dir, camera.fov, size.width, size.height, frame, spin, fill)
    if (!placed.current) {
      placed.current = true
      controls.target.copy(fit.target)
      camera.position.copy(fit.target).addScaledVector(dir, fit.dist)
      camera.lookAt(fit.target)
      controls.update()
      goal.current = null
    } else goal.current = fit
    invalidate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, controls, size.width, size.height, inset?.top, inset?.bottom, inset?.right, rooms, scans, spin, focus, full])
  // The game's arrow keys follow this view (see viewFrame); the classic editor (no inset) keeps plan axes.
  useEffect(() => () => void (viewFrame.azimuth = null), [])
  useFrame((_, dt) => {
    if (controls && inset) viewFrame.azimuth = Math.atan2(controls.target.x - camera.position.x, -(controls.target.z - camera.position.z))
    const g = goal.current
    if (!g || !controls) return
    invalidate()
    const k = 1 - Math.exp(-Math.min(dt, 0.05) * 12)
    const off = camera.position.clone().sub(controls.target)
    const d = off.length()
    const nd = d + (g.dist - d) * k
    controls.target.lerp(g.target, k)
    camera.position.copy(controls.target).addScaledVector(off.normalize(), nd)
    if (Math.abs(nd - g.dist) < 1e-3 && controls.target.distanceTo(g.target) < 1e-3) goal.current = null
    controls.update()
  })
  return null
}

/**
 * Shift the picture so the orbit target sits in the middle of the visible stage (between the top
 * bar and the drawer or selection bar) rather than the middle of the pane. Eases in about 250 ms
 * whenever the HUD changes, without moving the camera itself.
 */
function ViewOffset({ inset: given, full }: { inset?: SceneInset; full?: boolean }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const size = useThree((s) => s.size)
  const inset = lookInset(given, full, size.width, size.height)
  const cur = useRef(0)
  const curX = useRef(0)
  useFrame((_, dt) => {
    const top = inset?.top ?? 0
    const vis = visibleHeight(size.height, inset)
    // Centre of the visible band, measured from the pane's centre.
    const goal = inset ? top + vis / 2 - size.height / 2 : 0
    const goalX = (inset?.right ?? 0) / 2
    let next = cur.current + (goal - cur.current) * (1 - Math.exp(-dt * 14))
    if (Math.abs(goal - next) < 0.5) next = goal
    // The side drawer (desktop) comes and goes with the mode: ease across too.
    let nextX = curX.current + (goalX - curX.current) * (1 - Math.exp(-dt * 14))
    if (Math.abs(goalX - nextX) < 0.5) nextX = goalX
    curX.current = nextX
    const view = camera.view
    // A right-hand column (phone camera buttons, desktop drawer) shifts the picture left by half its width.
    const same = view?.enabled
      ? view.fullWidth === size.width && view.fullHeight === size.height && view.offsetY === -next && view.offsetX === nextX
      : next === 0 && nextX === 0
    if (same) return
    invalidate()
    cur.current = next
    if (next === 0 && nextX === 0) camera.clearViewOffset()
    else camera.setViewOffset(size.width, size.height, nextX, -next, size.width, size.height)
  })
  useEffect(() => () => camera.clearViewOffset(), [camera])
  return null
}

/** Where walk-in starts: a spot in a room, and the point it looks at. */
export interface EyeSpot {
  pos: [number, number, number]
  look: [number, number, number]
}
const EYE_HEIGHT = 1.5
const LOOK_HEIGHT = 1.1

/** A target a hair in front of the eye, for OrbitControls to turn the head around. */
function eyeTarget(eye: EyeSpot): [number, number, number] {
  const d = new THREE.Vector3(eye.look[0] - eye.pos[0], eye.look[1] - eye.pos[1], eye.look[2] - eye.pos[2]).normalize().multiplyScalar(0.01)
  return [eye.pos[0] + d.x, eye.pos[1] + d.y, eye.pos[2] + d.z]
}

/**
 * Pick where walking in starts: just inside the door of the most furnished room (or, with no door,
 * in its corner farthest from the middle), looking across the room at its centre.
 */
export function eyeSpot(doc: ReturnType<typeof useStore.getState>['doc']): EyeSpot | null {
  if (!doc.rooms.length) {
    const scan = (doc.scans ?? []).find((r) => r.visible)
    const c = scan ? scanCenter(scan) : null
    return c ? { pos: [c.x, EYE_HEIGHT, c.y + 1], look: [c.x, LOOK_HEIGHT, c.y - 1] } : null
  }
  const layout = activeLayout(doc)
  const score = (r: Room) => layout.placements.filter((p) => pointInPolygon(v(p.x, p.y), r.polygon)).length * 100 + polygonArea(r.polygon)
  const room = [...doc.rooms].sort((a, b) => score(b) - score(a))[0]
  const c = centroid(room.polygon)
  const door = room.openings.find((o) => o.kind === 'door')
  let p: Vec2
  if (door) {
    const [a, b] = openingEndpoints(room, door)
    p = add(scale(add(a, b), 0.5), scale(wallInwardNormal(room, door.wall), 0.6))
  } else {
    const far = room.polygon.reduce((best, q) => (dist(q, c) > dist(best, c) ? q : best), room.polygon[0])
    p = add(far, scale(norm(sub(c, far)), Math.min(0.7, dist(far, c) * 0.5)))
  }
  // Too close to the middle to see much (a tiny room): step back from it along the same line.
  if (dist(p, c) < 0.8) p = add(c, scale(dist(p, c) > 1e-3 ? norm(sub(p, c)) : v(0, 1), 0.8))
  // Never start inside a piece of furniture.
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const boxes = layout.placements.flatMap((pl) => {
    const it = items.get(pl.itemId)
    return it && !it.floorLevel && !isWallDecor(it.category) ? [{ x: pl.x, y: pl.y, hw: it.width / 2, hd: it.depth / 2, rotation: pl.rotation }] : []
  })
  const clear = clearOfItems(p, boxes, c)
  if (pointInPolygon(clear, room.polygon)) p = clear
  return { pos: [p.x, EYE_HEIGHT, p.y], look: [c.x, LOOK_HEIGHT, c.y] }
}

const WALK_KEYS = new Set(['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'])

/**
 * Moving around at eye level: glide to a tapped spot on the floor, or walk with WASD / arrow keys
 * (left and right arrows turn). Moves the camera and the look-at target together.
 */
function EyeWalk({
  controls,
  glide,
  rooms,
  walls,
  boxes,
}: {
  controls: RefObject<OrbitControlsImpl | null>
  glide: RefObject<Vec2[] | null>
  rooms: Room[]
  walls: WallSeg[]
  boxes: ItemBox[]
}) {
  const stuck = useRef(0)
  const camera = useThree((s) => s.camera)
  const keys = useRef(new Set<string>())
  const box = useMemo(() => {
    const pts = rooms.flatMap((r) => r.polygon)
    return pts.length ? bounds(pts) : null
  }, [rooms])
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return
      const k = e.key.toLowerCase()
      if (!WALK_KEYS.has(k)) return
      e.preventDefault()
      keys.current.add(k)
      invalidate()
    }
    const up = (e: KeyboardEvent) => keys.current.delete(e.key.toLowerCase())
    const clear = () => keys.current.clear()
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', clear)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', clear)
    }
  }, [])
  useFrame((_, rawDt) => {
    const c = controls.current
    if (!c) return
    const dt = Math.min(rawDt, 0.05)
    const k = keys.current
    // Walking (keys held or a glide under way) keeps the canvas drawing.
    if (k.size || glide.current?.length) invalidate()
    const fwd = new THREE.Vector3()
    camera.getWorldDirection(fwd)
    fwd.y = 0
    if (fwd.lengthSq() < 1e-6) return
    fwd.normalize()
    const right = new THREE.Vector3(-fwd.z, 0, fwd.x)
    const turn = (k.has('arrowright') ? 1 : 0) - (k.has('arrowleft') ? 1 : 0)
    if (turn) {
      const off = c.target.clone().sub(camera.position).applyAxisAngle(new THREE.Vector3(0, 1, 0), -turn * 1.6 * dt)
      c.target.copy(camera.position).add(off)
      glide.current = null
    }
    const move = new THREE.Vector3()
    if (k.has('w') || k.has('arrowup')) move.add(fwd)
    if (k.has('s') || k.has('arrowdown')) move.sub(fwd)
    if (k.has('d')) move.add(right)
    if (k.has('a')) move.sub(right)
    if (move.lengthSq() > 0) {
      glide.current = null
      move.normalize().multiplyScalar(1.6 * dt)
    } else if (glide.current?.length) {
      const route = glide.current
      const goal = route[0]
      const d = new THREE.Vector3(goal.x - camera.position.x, 0, goal.y - camera.position.z)
      const len = d.length()
      // Waypoints on the way (doorways) are passed without stopping; the last one eases in.
      if (len < (route.length > 1 ? 0.15 : 0.01)) {
        route.shift()
        if (!route.length) glide.current = null
      } else if (route.length > 1) move.copy(d).multiplyScalar(Math.min(len, 2.2 * dt) / len)
      else move.copy(d).multiplyScalar(Math.min(1, Math.max(dt * 3.5, Math.min(len, 0.4 * dt) / len)))
    }
    if (move.lengthSq() > 0) {
      const from = v(camera.position.x, camera.position.z)
      let next = walkStep(from, v(move.x, move.z), walkObstacles(from, walls, boxes))
      if (box) next = v(THREE.MathUtils.clamp(next.x, box.minX + 0.2, box.maxX - 0.2), THREE.MathUtils.clamp(next.y, box.minY + 0.2, box.maxY - 0.2))
      // A glide that's pressed against a wall and getting nowhere gives up.
      const got = dist(from, next)
      if (glide.current && got < move.length() * 0.2) {
        stuck.current += dt
        if (stuck.current > 0.4) glide.current = null
      } else stuck.current = 0
      move.set(next.x - from.x, 0, next.y - from.y)
      camera.position.add(move)
      c.target.add(move)
    }
    if (turn || move.lengthSq() > 0) c.update()
  })
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
  /** Walk in: stand at eye level inside the most furnished room, looking across it. */
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
  /** Screen space the HUD covers, so the overview centres the home in what's left. */
  inset?: SceneInset
  /** Hide the problem outlines and badges (Look mode is for admiring). */
  hideFlags?: boolean
  /** Frame just this room (a phone furnishing a piece in it); null frames the whole home. */
  focusRoom?: string | null
  /** Look mode: frame the home between the look bar and the mode bar (phones: the whole screen). */
  fullFrame?: boolean
}

export function Scene3D({ control }: { control?: SceneControl } = {}) {
  const doc = useStore((s) => s.doc)
  const scan3d = useStore((s) => s.scan3d)
  const [ownWallMode, setWallMode] = useState<WallMode>('auto')
  const [ownCamKey, setCamKey] = useState(0)
  const [ownEye, setEye] = useState<EyeSpot | null>(null)
  const wallMode = control?.wallMode ?? ownWallMode
  const style: StyleId = control?.style ?? 'classic'
  const palette = useScanPalette(style)
  const styleEnv = useMemo(() => ({ style, palette }), [style, palette])
  const stage = STYLE_STAGE[style]
  const eyePos = useMemo(
    () => eyeSpot(doc),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [control?.eye, control?.resetKey, doc.apartment.id],
  )
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
    const spot = eyeSpot(doc)
    if (!spot) return
    setEye(spot)
    setCamKey((k) => k + 1)
  }
  // A wider lens at eye level, wider still on a portrait phone, so a room fits in view.
  const eyeFov = typeof window !== 'undefined' && window.innerHeight > window.innerWidth ? 72 : 60

  const placedCount = activeLayout(doc).placements.length
  const label = `${doc.apartment.name}${eye ? ', walking around inside' : ''}: ${doc.rooms.length} room${doc.rooms.length === 1 ? '' : 's'}, ${placedCount} item${placedCount === 1 ? '' : 's'}`
  return (
    <div className="scene-wrap" role="img" aria-label={label}>
      <Canvas
        // The game keeps one canvas for the home: recentering and walking in move the camera
        // (CameraSpot, FitCamera) rather than rebuilding the renderer. The classic editor still
        // starts afresh on Overview / Eye level.
        key={control ? doc.apartment.id : `${camKey}:${doc.apartment.id}`}
        shadows
        dpr={DPR}
        // Draws only when something changes: controls, drags, store changes and running
        // animations ask for frames (invalidate). Photos render and read in one go (capture.ts).
        frameloop="demand"
        camera={{ position: eye?.pos ?? initial, fov: eye ? eyeFov : 45, near: 0.05, far: 300 }}
        onPointerMissed={() => useStore.getState().select(null)}
      >
        <color attach="background" args={[stage.sky ?? control?.sky ?? '#f6f4ef']} />
        <StyleContext.Provider value={styleEnv}>
          <Scene resetKey={control?.resetKey} eyeFov={eyeFov} wallMode={wallMode} eye={eye} autoRotate={control?.autoRotate && !eye} ground={control?.ground} inset={control?.inset} hideFlags={control?.hideFlags} focusRoom={control?.focusRoom} fullFrame={control?.fullFrame} />
        </StyleContext.Provider>
      </Canvas>
      {control && eye && <EyeHint />}
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


/** Phones and tablets draw at most 1.5× their CSS pixels: sharper than that costs fill rate, not looks. */
const DPR: [number, number] = typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches ? [1, 1.5] : [1, 2]

const EYE_HINT_KEY = 'apartment-designer:eye-hint-seen'

/** A one-time coach mark the first time you walk in: how to look around and move. */
function EyeHint() {
  const [show, setShow] = useState(() => {
    try {
      return localStorage.getItem(EYE_HINT_KEY) !== '1'
    } catch {
      return true
    }
  })
  useEffect(() => {
    if (!show) return
    try {
      localStorage.setItem(EYE_HINT_KEY, '1')
    } catch {
      // Private mode: the hint shows again next visit.
    }
    const hide = () => setShow(false)
    const t = setTimeout(hide, 6000)
    window.addEventListener('pointerdown', hide)
    window.addEventListener('keydown', hide)
    return () => {
      clearTimeout(t)
      window.removeEventListener('pointerdown', hide)
      window.removeEventListener('keydown', hide)
    }
  }, [show])
  if (!show) return null
  const touch = typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches
  return (
    <div className="scene-eye-hint" role="status">
      <b>Drag to look around</b>
      <span>{touch ? 'Tap the floor to walk there' : 'Click the floor to walk there, or use WASD'}</span>
    </div>
  )
}
