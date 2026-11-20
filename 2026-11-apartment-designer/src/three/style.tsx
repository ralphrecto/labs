import { Edges } from '@react-three/drei'
import { useThree } from '@react-three/fiber'
import { createContext, useContext, useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { toonHex, type ScanPalette } from '../scan/palette'

// Rendering styles for the 3D view, switched like map layers. Every surface in the scene (walls,
// floors, furniture) draws through <Surface> and <Ink>, which pick materials for the active style.

export type RenderStyle = 'toon' | 'real' | 'scan' | 'clay' | 'blueprint'
/** `classic` is the plain look the classic editor keeps; it isn't offered in the game. */
export type StyleId = RenderStyle | 'classic'

export const RENDER_STYLES: { id: RenderStyle; label: string; blurb: string }[] = [
  { id: 'toon', label: 'Cartoon', blurb: 'Your real colors, drawn like a cozy cartoon' },
  { id: 'real', label: 'Lifelike', blurb: 'Soft daylight and real materials' },
  { id: 'scan', label: 'Scan', blurb: 'The raw scan of your place' },
  { id: 'clay', label: 'Clay', blurb: 'Plain white model, just the shapes' },
  { id: 'blueprint', label: 'Blueprint', blurb: 'Lines and sizes, like a plan' },
]

export interface StyleEnv {
  style: StyleId
  palette: ScanPalette | null
}

export const StyleContext = createContext<StyleEnv>({ style: 'classic', palette: null })
export const useStyle = () => useContext(StyleContext)

/** Background, ground and lighting for each style. */
export const STYLE_STAGE: Record<StyleId, { sky?: string; ground?: string; showGround: boolean; shadows: boolean }> = {
  classic: { showGround: true, shadows: true },
  toon: { sky: '#fde9cf', ground: '#f8d9b4', showGround: true, shadows: true },
  real: { sky: '#d9cbb6', ground: '#cfb08a', showGround: true, shadows: true },
  scan: { sky: '#1d2129', showGround: false, shadows: false },
  clay: { sky: '#f1e6d6', ground: '#efd6b2', showGround: true, shadows: true },
  blueprint: { sky: '#1c4a8a', showGround: false, shadows: false },
}

const BLUEPRINT_LINE = '#e8f2ff'
const BLUEPRINT_FILL = '#5d95e0'
const INK = '#3d2c25'
// Clay keeps warm putty tones with clear steps of value: floor darkest, walls lighter, pieces lightest,
// so the rooms separate from the stage and the shadows carry the form.
const CLAY = '#d8ccbc'
const CLAY_ITEM = '#fbf7f1'
const CLAY_FLOOR = '#cbbca8'

// ------------------------------------------------------------------ toon ramp

let ramp: THREE.DataTexture | null = null
/** Three flat light bands: shadow, mid, lit. The shadow band stays fairly light so shade reads warm, not muddy. */
function toonRamp() {
  if (ramp) return ramp
  const d = new Uint8Array([128, 128, 128, 255, 196, 196, 196, 255, 255, 255, 255, 255])
  ramp = new THREE.DataTexture(d, 3, 1, THREE.RGBAFormat)
  ramp.minFilter = THREE.NearestFilter
  ramp.magFilter = THREE.NearestFilter
  ramp.generateMipmaps = false
  ramp.needsUpdate = true
  return ramp
}

const mix = (a: string, b: string, t: number) => `#${new THREE.Color(a).lerp(new THREE.Color(b), t).getHexString()}`

/**
 * `cap` is the top edge of a wall (a darker tone, so room outlines read from above); `trim` is the
 * baseboard along the foot of a wall. Both are wall-like surfaces with their own tone per style.
 * `window` is window glass, painted with a soft sky so it reads as a window from inside and out;
 * `daylight` is the patch of light a window throws on the floor.
 */
export type SurfaceRole = 'wall' | 'floor' | 'item' | 'soft' | 'glass' | 'ground' | 'window' | 'daylight' | 'cap' | 'trim'

// ------------------------------------------------------------------ sky in the windows

const skies: Partial<Record<string, THREE.Texture>> = {}
/**
 * A vertical sky gradient for window glass: blue up top, bright and pale toward the horizon.
 * Cartoon gets three flat bands; Clay a nearly white, barely blue version.
 */
function skyTexture(kind: 'smooth' | 'bands' | 'clay') {
  const hit = skies[kind]
  if (hit || typeof document === 'undefined') return hit ?? null
  let t: THREE.Texture
  if (kind === 'bands') {
    // Bottom to top (DataTexture rows start at the bottom): horizon, mid, top.
    const rows = [
      [236, 247, 255],
      [196, 228, 250],
      [150, 203, 242],
    ]
    const d = new Uint8Array(rows.flatMap((c) => [...c, 255]))
    t = new THREE.DataTexture(d, 1, 3, THREE.RGBAFormat)
    t.magFilter = THREE.NearestFilter
    t.minFilter = THREE.NearestFilter
    t.generateMipmaps = false
  } else {
    const c = document.createElement('canvas')
    c.width = 4
    c.height = 128
    const g = c.getContext('2d')!
    const grad = g.createLinearGradient(0, 0, 0, 128)
    if (kind === 'clay') {
      grad.addColorStop(0, '#dfe6ec')
      grad.addColorStop(1, '#f7f8f8')
    } else {
      grad.addColorStop(0, '#8cc4f0')
      grad.addColorStop(0.55, '#cbe6fa')
      grad.addColorStop(1, '#f6fbff')
    }
    g.fillStyle = grad
    g.fillRect(0, 0, 4, 128)
    t = new THREE.CanvasTexture(c)
  }
  t.colorSpace = THREE.SRGBColorSpace
  t.needsUpdate = true
  return (skies[kind] = t)
}

let fade: THREE.Texture | null = null
/** Alpha that fades from full at the window (v = 1) to nothing across the room (v = 0). */
function fadeTexture() {
  if (fade) return fade
  const d = new Uint8Array(32 * 4)
  for (let i = 0; i < 32; i++) {
    const a = Math.round(255 * Math.pow(i / 31, 0.8))
    d.set([a, a, a, 255], i * 4)
  }
  fade = new THREE.DataTexture(d, 1, 32, THREE.RGBAFormat)
  fade.magFilter = THREE.LinearFilter
  fade.minFilter = THREE.LinearFilter
  fade.generateMipmaps = false
  fade.needsUpdate = true
  return fade
}

/**
 * The material for one surface in the active style. `color` is the object's own color (a scanned
 * wall, a sofa's fabric); `map` is an optional texture (the scanned floor) that some styles use.
 */
interface SurfaceProps {
  role: SurfaceRole
  color: string
  highlight?: string
  map?: THREE.Texture | null
  side?: THREE.Side
}

/**
 * Keyed on the style (and on whether there is a texture), so a mesh that stays mounted across a
 * style switch gets a fresh material: React would otherwise patch the old one, leaving props the new
 * style leaves out (opacity, visible, depthWrite) behind and a newly added map uncompiled.
 */
export function Surface(props: SurfaceProps) {
  const { style } = useStyle()
  return <SurfaceMaterial key={`${style}-${props.map ? 'map' : 'flat'}`} {...props} />
}

function SurfaceMaterial({ role, color, highlight, map, side }: SurfaceProps) {
  const { style } = useStyle()
  const edge = role === 'cap' || role === 'trim' ? role : null
  if (edge) {
    color = edgeTone(edge, color, style)
    role = 'wall'
  }
  const emissive = highlight ?? '#000000'
  const emissiveIntensity = highlight ? 0.35 : 0
  if (role === 'window') {
    // Unlit, so the sky stays bright whatever the room's lighting.
    if (style === 'blueprint') return <meshBasicMaterial color={BLUEPRINT_LINE} transparent opacity={0.16} depthWrite={false} side={side} />
    if (style === 'scan') return <meshBasicMaterial color="#cfe6ff" transparent opacity={0.18} depthWrite={false} side={side} />
    const sky = skyTexture(style === 'toon' ? 'bands' : style === 'clay' ? 'clay' : 'smooth')
    return <meshBasicMaterial color="#ffffff" map={sky} toneMapped={false} side={side} />
  }
  if (role === 'daylight') {
    if (style === 'blueprint' || style === 'scan') return <meshBasicMaterial visible={false} />
    if (style === 'toon') return <meshBasicMaterial color="#fff6dc" transparent opacity={0.4} depthWrite={false} side={side} polygonOffset polygonOffsetFactor={-2} />
    return (
      <meshBasicMaterial
        color={style === 'clay' ? '#ffffff' : color}
        alphaMap={fadeTexture()}
        transparent
        opacity={style === 'clay' ? 0.45 : 0.55}
        blending={style === 'clay' ? THREE.NormalBlending : THREE.AdditiveBlending}
        depthWrite={false}
        side={side}
        polygonOffset
        polygonOffsetFactor={-2}
      />
    )
  }
  if (role === 'glass') {
    if (style === 'blueprint') return <meshBasicMaterial color={BLUEPRINT_LINE} transparent opacity={0.08} depthWrite={false} />
    if (style === 'real') return <meshPhysicalMaterial color="#cfe3f2" transparent opacity={0.25} roughness={0.05} metalness={0} />
    return <meshStandardMaterial color="#9cc3e6" transparent opacity={0.35} roughness={0.1} />
  }
  switch (style) {
    case 'toon':
      return (
        <meshToonMaterial
          color={map ? '#ffffff' : toonHex(color)}
          map={map ?? null}
          gradientMap={toonRamp()}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          side={side}
        />
      )
    case 'real':
      return (
        <meshStandardMaterial
          color={map ? '#ffffff' : color}
          map={map ?? null}
          roughness={role === 'soft' ? 0.95 : role === 'floor' ? 0.55 : role === 'wall' ? 0.92 : 0.62}
          envMapIntensity={role === 'wall' ? 0.2 : 0.35}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          side={side}
        />
      )
    case 'clay':
      return (
        <meshStandardMaterial
          color={role === 'item' || role === 'soft' ? mix(CLAY_ITEM, color, 0.12) : role === 'floor' ? CLAY_FLOOR : edge ? color : CLAY}
          roughness={1}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          side={side}
        />
      )
    case 'blueprint':
      return (
        <meshBasicMaterial
          color={highlight ? '#ffd166' : BLUEPRINT_FILL}
          transparent
          opacity={role === 'floor' ? 0.22 : role === 'wall' ? 0.14 : highlight ? 0.5 : 0.2}
          depthWrite={false}
          side={side ?? THREE.DoubleSide}
        />
      )
    case 'scan':
      // Furniture floats over the scan as see-through ghosts.
      return <meshBasicMaterial color={highlight ?? '#ffffff'} transparent opacity={highlight ? 0.35 : 0.1} depthWrite={false} />
    default:
      return (
        <meshStandardMaterial
          color={color}
          roughness={role === 'wall' ? 0.9 : role === 'floor' ? 0.95 : 0.85}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          side={side}
        />
      )
  }
}

/** The tone of a wall's top edge or its baseboard, from the wall's own color. */
function edgeTone(role: 'cap' | 'trim', color: string, style: StyleId) {
  if (style === 'toon') return role === 'trim' ? '#fffaf2' : color
  if (style === 'clay') return role === 'cap' ? '#bfae98' : '#cdbfad'
  if (style === 'real') return mix(color, '#5e4a38', role === 'cap' ? 0.6 : 0.26)
  return color
}

/**
 * Outline lines on a mesh's creases: dark ink for Cartoon, white lines for Blueprint and Scan.
 * The <Edges> is keyed on the style: it sits at the same place in the tree for every style, and a
 * mesh that doesn't remount on a style switch (walls, windows) would otherwise keep the old lines.
 */
export function Ink({ highlight, soft }: { highlight?: string; soft?: boolean }) {
  const { style } = useStyle()
  const k = `${style}-${highlight ?? ''}`
  if (style === 'toon') return <Edges key={k} threshold={soft ? 40 : 20} color={highlight ?? INK} lineWidth={highlight ? 1.8 : 1.6} />
  if (style === 'blueprint') return <Edges key={k} threshold={20} color={highlight ? '#ffd166' : BLUEPRINT_LINE} lineWidth={highlight ? 1.8 : 1.2} />
  if (style === 'scan') return <Edges key={k} threshold={20} color={highlight ?? '#ffffff'} lineWidth={1.4} transparent opacity={0.85} />
  if (style === 'clay') return <Edges key={k} threshold={soft ? 40 : 20} color={highlight ?? '#000000'} lineWidth={highlight ? 1.8 : 1} transparent opacity={highlight ? 0.95 : 0.3} />
  // Lifelike draws no lines, except the outline of a selected piece.
  if (highlight && style === 'real') return <Edges key={k} threshold={soft ? 40 : 20} color={highlight} lineWidth={1.8} />
  return null
}

// ------------------------------------------------------------------ whole-object restyling (GLB models)

/** Swap a loaded model's materials for the style, keeping its textures where the style uses them. */
export function restyle(obj: THREE.Object3D, style: StyleId) {
  if (style === 'classic' || style === 'real') return obj
  obj.traverse((c) => {
    const m = c as THREE.Mesh
    if (!m.isMesh) return
    const swap = (mat: THREE.Material) => {
      const src = mat as THREE.MeshStandardMaterial
      const color = src.color ? `#${src.color.getHexString()}` : '#cccccc'
      if (style === 'toon') return new THREE.MeshToonMaterial({ color: src.map ? '#ffffff' : toonHex(color), map: src.map ?? null, gradientMap: toonRamp() })
      if (style === 'clay') return new THREE.MeshStandardMaterial({ color: CLAY_ITEM, roughness: 1 })
      if (style === 'blueprint') return new THREE.MeshBasicMaterial({ color: BLUEPRINT_FILL, transparent: true, opacity: 0.2, depthWrite: false })
      return new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.12, depthWrite: false })
    }
    m.material = Array.isArray(m.material) ? m.material.map(swap) : swap(m.material)
  })
  return obj
}

// ------------------------------------------------------------------ scene-wide setup

/** Soft studio light from a generated room (no download), for the Lifelike style's reflections. */
export function StyleEnvironment() {
  const { style } = useStyle()
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  useEffect(() => {
    if (style !== 'real') return
    const exposure = gl.toneMappingExposure
    gl.toneMappingExposure = 0.72
    const pm = new THREE.PMREMGenerator(gl)
    const env = pm.fromScene(new RoomEnvironment(), 0.04).texture
    scene.environment = env
    return () => {
      scene.environment = null
      gl.toneMappingExposure = exposure
      env.dispose()
      pm.dispose()
    }
  }, [style, gl, scene])
  return null
}

/**
 * Lights for the style. Cartoon wants one strong sun for clean bands; Lifelike leans on the environment.
 * `extent` is the radius (m) the shadows need to cover around `target`: the home's footprint plus a margin.
 */
export function StyleLights({ target, extent = 12 }: { target: THREE.Vector3; extent?: number }) {
  const { style } = useStyle()
  const shadows = STYLE_STAGE[style].shadows
  const sun = useMemo(() => {
    // Cartoon: high and nearly overhead, so furniture gets small crisp shadows and walls don't
    // throw long dark wedges across the rooms.
    if (style === 'toon') return { pos: [target.x + 2.5, 18, target.z + 2] as const, intensity: 2.2, color: '#fff3df' }
    if (style === 'real') return { pos: [target.x - 4, 8, target.z + 6] as const, intensity: 3.0, color: '#fff1dc' }
    if (style === 'clay') return { pos: [target.x - 3, 9, target.z + 5] as const, intensity: 1.8, color: '#fff3e2' }
    return { pos: [target.x + 4, 9, target.z + 6] as const, intensity: 1.4, color: '#ffffff' }
  }, [style, target])
  // The sun aims at the middle of the home (a directional light aims at its target's position).
  const aim = useMemo(() => new THREE.Object3D(), [])
  aim.position.copy(target)
  const r = Math.max(2, extent)
  if (style === 'blueprint' || style === 'scan') return <ambientLight intensity={1} />
  return (
    <>
      <ambientLight intensity={style === 'toon' ? 0.9 : style === 'real' ? 0 : style === 'clay' ? 1.15 : 0.8} color={style === 'toon' ? '#ffe9d6' : style === 'clay' ? '#fff4e6' : '#ffffff'} />
      <hemisphereLight args={[style === 'toon' ? '#fff8ec' : style === 'real' || style === 'clay' ? '#fff4e4' : '#ffffff', style === 'toon' ? '#e7b48c' : style === 'clay' ? '#d9c6aa' : '#b9a98f', style === 'real' ? 0.7 : style === 'clay' ? 0.9 : 0.6]} />
      <primitive object={aim} />
      <directionalLight
        key={`${r.toFixed(2)}`}
        position={sun.pos as unknown as [number, number, number]}
        target={aim}
        intensity={sun.intensity}
        color={sun.color}
        castShadow={shadows}
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-r}
        shadow-camera-right={r}
        shadow-camera-top={r}
        shadow-camera-bottom={-r}
        shadow-camera-near={0.5}
        shadow-camera-far={60}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
      />
    </>
  )
}

/** A floor image as a texture, mapped in plan meters through the floor's UVs. */
export function floorTexture(img: { width: number; height: number; rgba: Uint8ClampedArray }, crisp: boolean) {
  const t = new THREE.DataTexture(img.rgba, img.width, img.height, THREE.RGBAFormat)
  t.colorSpace = THREE.SRGBColorSpace
  t.magFilter = crisp ? THREE.NearestFilter : THREE.LinearFilter
  t.minFilter = THREE.LinearFilter
  t.generateMipmaps = false
  t.needsUpdate = true
  return t
}

let wood: THREE.CanvasTexture | null = null
/** Procedural oak planks for Lifelike floors when there's no scan to take the floor from. 1 repeat = 1 m. */
export function woodTexture() {
  if (wood || typeof document === 'undefined') return wood
  const c = document.createElement('canvas')
  c.width = 512
  c.height = 512
  const g = c.getContext('2d')!
  const rows = 8
  let seed = 7
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  for (let r = 0; r < rows; r++) {
    let x = -rnd() * 300
    while (x < 512) {
      const w = 220 + rnd() * 200
      const l = 62 + rnd() * 10
      g.fillStyle = `hsl(${30 + rnd() * 6}, ${38 + rnd() * 10}%, ${l}%)`
      g.fillRect(x, r * 64, w, 64)
      g.strokeStyle = 'rgba(80,50,25,0.12)'
      for (let k = 0; k < 6; k++) {
        g.beginPath()
        const y = r * 64 + 6 + rnd() * 52
        g.moveTo(x, y)
        g.bezierCurveTo(x + w / 3, y + rnd() * 6 - 3, x + (2 * w) / 3, y + rnd() * 6 - 3, x + w, y)
        g.stroke()
      }
      g.fillStyle = 'rgba(60,35,15,0.35)'
      g.fillRect(x, r * 64, 2, 64)
      x += w
    }
    g.fillStyle = 'rgba(60,35,15,0.3)'
    g.fillRect(0, r * 64, 512, 2)
  }
  wood = new THREE.CanvasTexture(c)
  wood.colorSpace = THREE.SRGBColorSpace
  wood.wrapS = wood.wrapT = THREE.RepeatWrapping
  wood.anisotropy = 4
  wood.repeat.set(0.5, 0.5)
  return wood
}
