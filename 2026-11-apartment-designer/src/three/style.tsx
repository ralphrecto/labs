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
  real: { sky: '#dfe4e8', ground: '#c9c4bb', showGround: true, shadows: true },
  scan: { sky: '#1d2129', showGround: false, shadows: false },
  clay: { sky: '#c9c2b8', ground: '#cdc5ba', showGround: true, shadows: true },
  blueprint: { sky: '#1c4a8a', showGround: false, shadows: false },
}

const BLUEPRINT_LINE = '#e8f2ff'
const BLUEPRINT_FILL = '#5d95e0'
const INK = '#3d2c25'
const CLAY = '#eee9e2'
const CLAY_ITEM = '#f4f0ea'
const CLAY_FLOOR = '#d9d2c7'

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

export type SurfaceRole = 'wall' | 'floor' | 'item' | 'soft' | 'glass' | 'ground'

/**
 * The material for one surface in the active style. `color` is the object's own color (a scanned
 * wall, a sofa's fabric); `map` is an optional texture (the scanned floor) that some styles use.
 */
export function Surface({
  role,
  color,
  highlight,
  map,
  side,
}: {
  role: SurfaceRole
  color: string
  highlight?: string
  map?: THREE.Texture | null
  side?: THREE.Side
}) {
  const { style } = useStyle()
  const emissive = highlight ?? '#000000'
  const emissiveIntensity = highlight ? 0.35 : 0
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
          color={role === 'item' || role === 'soft' ? mix(CLAY_ITEM, color, 0.08) : role === 'floor' ? CLAY_FLOOR : CLAY}
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

/** Outline lines on a mesh's creases: dark ink for Cartoon, white lines for Blueprint and Scan. */
export function Ink({ highlight, soft }: { highlight?: string; soft?: boolean }) {
  const { style } = useStyle()
  if (style === 'toon') return <Edges threshold={soft ? 40 : 20} color={highlight ?? INK} lineWidth={highlight ? 2.2 : 1.6} />
  if (style === 'blueprint') return <Edges threshold={20} color={highlight ? '#ffd166' : BLUEPRINT_LINE} lineWidth={highlight ? 2 : 1.2} />
  if (style === 'scan') return <Edges threshold={20} color={highlight ?? '#ffffff'} lineWidth={1.4} transparent opacity={0.85} />
  if (style === 'clay') return <Edges threshold={soft ? 40 : 20} color={highlight ?? '#000000'} lineWidth={highlight ? 2 : 1} transparent opacity={highlight ? 0.9 : 0.15} />
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
    gl.toneMappingExposure = 0.8
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
    if (style === 'clay') return { pos: [target.x - 3, 9, target.z + 5] as const, intensity: 2.6, color: '#ffffff' }
    return { pos: [target.x + 4, 9, target.z + 6] as const, intensity: 1.4, color: '#ffffff' }
  }, [style, target])
  // The sun aims at the middle of the home (a directional light aims at its target's position).
  const aim = useMemo(() => new THREE.Object3D(), [])
  aim.position.copy(target)
  const r = Math.max(2, extent)
  if (style === 'blueprint' || style === 'scan') return <ambientLight intensity={1} />
  return (
    <>
      <ambientLight intensity={style === 'toon' ? 0.9 : style === 'real' ? 0 : style === 'clay' ? 0.15 : 0.8} color={style === 'toon' ? '#ffe9d6' : '#ffffff'} />
      <hemisphereLight args={[style === 'toon' ? '#fff8ec' : '#ffffff', style === 'toon' ? '#e7b48c' : style === 'clay' ? '#8f8a84' : '#b9a98f', style === 'real' ? 0.15 : style === 'clay' ? 0.9 : 0.6]} />
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
