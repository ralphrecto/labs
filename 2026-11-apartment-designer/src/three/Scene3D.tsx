import { ContactShadows, Edges, OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { lazy, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useAnalysis } from '../analysis'
import { bounds, centroid, dist, norm, openingEndpoints, pointInPolygon, polygonArea, scale, sub, add, v, wallInwardNormal, wallSegments } from '../model/geometry'
import { snapPlacement } from '../model/snapping'
import type { Item, Opening, Placement, Room, Vec2 } from '../model/types'
import { activeLayout, useStore } from '../store'
import { ItemModel } from './ItemModel'
import { viewFrame } from './viewFrame'
import { scanCenter, scanCorners, useScanRuntime } from '../scan/runtime'
import { reduceMotion } from '../motion'

// The scan backdrop pulls in the splat renderer, so it loads on first use.
const ScanBackdrop = lazy(() => import('./ScanBackdrop').then((m) => ({ default: m.ScanBackdrop })))
import { ScanPoints } from './ScanPoints'
import { floorTexture, Ink, StyleContext, StyleEnvironment, StyleLights, STYLE_STAGE, Surface, useStyle, woodTexture, type StyleId } from './style'
import { scanPalette, toonFloor, type ScanPalette } from '../scan/palette'
import { toScanData } from '../scan/align'
import { WindowModel, type WindowSpec } from './WindowModel'
import { routeTo, sharedDoorways, walkStep, wallSegmentsForWalking, type WallSeg } from './walkCollision'

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
  const wallBase = palette?.wallColor[room.id] ?? (style === 'toon' ? '#f6ecdf' : style === 'real' ? '#f1e8da' : '#f2f0eb')
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
const SELECT_LINE = '#4f8be8'
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
  useFrame(({ clock }) => {
    if (!a.current || !b.current || reduceMotion) return
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
    if (on && !reduceMotion) start.current = -1
  }, [on])
  useFrame(() => {
    const g = ref.current
    if (!g || !start.current) return
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
  const { style } = useStyle()
  // The classic editor tints problem pieces. In the game a tint reads as red furniture, so the
  // piece keeps its own colors and gets an outline and a small "!" badge instead.
  const tint = style === 'classic'
  const highlight = selected ? SELECT_LINE : tint && flag === 'error' ? '#d64545' : tint && flag === 'warning' ? '#d68a1f' : undefined
  const h = Math.max(item.height, 0.02)
  return (
    <group position={[p.x, 0, p.y]} rotation={[0, -p.rotation, 0]} onPointerDown={(e) => onGrab(e, p)}>
      <Hop on={selected}>
        <ItemModel item={item} highlight={highlight} />
        {/* Models from a URL draw their own materials, so their outline is the bounding box. */}
        {selected && item.glbUrl && !tint && (
          <mesh position={[0, h / 2, 0]} raycast={noRaycast}>
            <boxGeometry args={[item.width + 0.02, h + 0.01, item.depth + 0.02]} />
            <meshBasicMaterial visible={false} />
            <Edges color={SELECT_LINE} lineWidth={2.4} raycast={noRaycast} />
          </mesh>
        )}
      </Hop>
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
  wallMode,
  eye,
  autoRotate,
  ground = '#e9e6df',
  inset,
  hideFlags,
}: {
  wallMode: WallMode
  eye: EyeSpot | null
  autoRotate?: boolean
  ground?: string
  inset?: SceneInset
  hideFlags?: boolean
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
    // Walking around is for looking: furniture isn't picked up, and taps go to walking.
    if (e.button !== 0 || eye) return
    e.stopPropagation()
    floorDown.current = false
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

  // A click on the floor (not a drag to orbit) deselects, unless a piece is within a few pixels
  // of it: small pieces are hard to hit, so a near miss picks the nearest one instead.
  const floorDown = useRef(false)
  const onFloorClick = (e: ThreeEvent<MouseEvent>) => {
    const down = floorDown.current
    floorDown.current = false
    if (!down || e.delta >= 5 || eye) return
    const near = nearestPlacementOnScreen(e, placed)
    useStore.getState().select(near ? { kind: 'placement', id: near } : null)
  }

  const onUp = () => {
    if (drag.current?.moved) useStore.getState().endGesture()
    drag.current = null
    if (controls.current) controls.current.enabled = true
  }

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
            glide.current = route
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
      <group onPointerMove={onMove} onPointerUp={onUp}>
        {placed.map(({ p, item }) => (
          <PlacedItem
            key={p.id}
            item={item}
            p={p}
            selected={selection?.kind === 'placement' && selection.id === p.id}
            flag={hideFlags ? undefined : analysis.flagged.get(p.id)}
            onGrab={onGrab}
          />
        ))}
      </group>
      {!eye && <FitCamera key={(doc.scans ?? []).length} inset={inset} spin={autoRotate} />}
      <ViewOffset inset={eye ? undefined : inset} />
      {eye ? (
        <>
          <OrbitControls ref={controls} makeDefault target={eyeTarget(eye)} enableZoom={false} enablePan={false} rotateSpeed={-0.4} />
          <EyeWalk controls={controls} glide={glide} rooms={doc.rooms} walls={walls} />
        </>
      ) : (
        <OrbitControls ref={controls} target={target} makeDefault maxPolarAngle={Math.PI / 2 - 0.02} minDistance={0.3} maxDistance={60} autoRotate={autoRotate} autoRotateSpeed={0.6} />
      )}
    </>
  )
}

/** Within this many CSS px of a missed click, a piece is picked anyway. */
const NEAR_PICK_PX = 12

/** The placement whose on-screen box (footprint and top) is nearest the click, if within NEAR_PICK_PX. */
function nearestPlacementOnScreen(e: ThreeEvent<MouseEvent>, placed: { p: Placement; item: Item }[]): string | null {
  const camera = e.camera
  const el = e.nativeEvent.target as HTMLElement | null
  const W = el?.clientWidth || 1
  const H = el?.clientHeight || 1
  const px = ((e.pointer.x + 1) / 2) * W
  const py = ((1 - e.pointer.y) / 2) * H
  const q = new THREE.Vector3()
  let best: string | null = null
  let bestD = NEAR_PICK_PX
  for (const { p, item } of placed) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    const c = Math.cos(p.rotation), s = Math.sin(p.rotation)
    for (const [lx, ly] of [[-1, -1], [1, -1], [1, 1], [-1, 1]])
      for (const h of [0, Math.max(item.height, 0.02)]) {
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
}

/** Height of the part of the pane the HUD leaves visible, never less than a quarter of it. */
const visibleHeight = (h: number, inset?: SceneInset) => Math.max(h * 0.25, h - (inset?.top ?? 0) - (inset?.bottom ?? 0))

/** How much of the visible stage the home spans across its tighter direction: about a 10% margin each side. */
const FILL = 0.8
/** The overview looks down from the front, a little to the right. */
const OVERVIEW_DIR = new THREE.Vector3(0.15, 1.0, 0.75).normalize()
const UP = new THREE.Vector3(0, 1, 0)

/**
 * The 3D points the overview has to keep in view: every room corner (pushed out by the wall
 * thickness) at the floor and at the ceiling. With no rooms yet, the visible scans' footprints.
 */
function framePoints(doc: ReturnType<typeof useStore.getState>['doc']): THREE.Vector3[] {
  const out: THREE.Vector3[] = []
  if (doc.rooms.length) {
    for (const r of doc.rooms) {
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
): { target: THREE.Vector3; dist: number } {
  const t = Math.tan((fovDeg * Math.PI) / 360)
  const ky = t * (visibleHeight(height, inset) / height) * FILL
  const kx = t * (width / height) * FILL
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
function FitCamera({ inset, spin }: { inset?: SceneInset; spin?: boolean }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const size = useThree((s) => s.size)
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null
  const rooms = useStore((s) => s.doc.rooms)
  const scans = useStore((s) => s.doc.scans)
  const touched = useRef(false)
  const placed = useRef(false)
  const goal = useRef<{ target: THREE.Vector3; dist: number } | null>(null)
  useEffect(() => {
    if (!controls) return
    const stop = () => {
      touched.current = true
      goal.current = null
    }
    controls.addEventListener('start', stop)
    return () => controls.removeEventListener('start', stop)
  }, [controls])
  useLayoutEffect(() => {
    if (touched.current || size.width < 10 || !controls) return
    const pts = framePoints(useStore.getState().doc)
    if (!pts.length) return
    const dir = placed.current ? camera.position.clone().sub(controls.target).normalize() : OVERVIEW_DIR.clone()
    const fit = fitView(pts, dir, camera.fov, size.width, size.height, inset, spin)
    if (!placed.current) {
      placed.current = true
      controls.target.copy(fit.target)
      camera.position.copy(fit.target).addScaledVector(dir, fit.dist)
      camera.lookAt(fit.target)
      controls.update()
      goal.current = null
    } else goal.current = fit
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, controls, size.width, size.height, inset?.top, inset?.bottom, rooms, scans, spin])
  // The game's arrow keys follow this view (see viewFrame); the classic editor (no inset) keeps plan axes.
  useEffect(() => () => void (viewFrame.azimuth = null), [])
  useFrame((_, dt) => {
    if (controls && inset) viewFrame.azimuth = Math.atan2(controls.target.x - camera.position.x, -(controls.target.z - camera.position.z))
    const g = goal.current
    if (!g || !controls) return
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
function ViewOffset({ inset }: { inset?: SceneInset }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const size = useThree((s) => s.size)
  const cur = useRef(0)
  useFrame((_, dt) => {
    const top = inset?.top ?? 0
    const vis = visibleHeight(size.height, inset)
    // Centre of the visible band, measured from the pane's centre.
    const goal = inset ? top + vis / 2 - size.height / 2 : 0
    let next = cur.current + (goal - cur.current) * (1 - Math.exp(-dt * 14))
    if (Math.abs(goal - next) < 0.5) next = goal
    const view = camera.view
    const same = view?.enabled ? view.fullWidth === size.width && view.fullHeight === size.height && view.offsetY === -next : next === 0
    if (same) return
    cur.current = next
    if (next === 0) camera.clearViewOffset()
    else camera.setViewOffset(size.width, size.height, 0, -next, size.width, size.height)
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
}: {
  controls: RefObject<OrbitControlsImpl | null>
  glide: RefObject<Vec2[] | null>
  rooms: Room[]
  walls: WallSeg[]
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
      let next = walkStep(from, v(move.x, move.z), walls)
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

  return (
    <div className="scene-wrap">
      <Canvas
        key={`${camKey}:${doc.apartment.id}`}
        shadows
        dpr={[1, 2]}
        camera={{ position: eye?.pos ?? initial, fov: eye ? eyeFov : 45, near: 0.05, far: 300 }}
        onPointerMissed={() => useStore.getState().select(null)}
        gl={{ preserveDrawingBuffer: true }}
      >
        <color attach="background" args={[stage.sky ?? control?.sky ?? '#f6f4ef']} />
        <StyleContext.Provider value={styleEnv}>
          <Scene wallMode={wallMode} eye={eye} autoRotate={control?.autoRotate && !eye} ground={control?.ground} inset={control?.inset} hideFlags={control?.hideFlags} />
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
