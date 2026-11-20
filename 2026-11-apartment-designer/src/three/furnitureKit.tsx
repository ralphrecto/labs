// Building blocks for the procedural furniture in ItemModel: a tiny "part" language (sharp boxes,
// soft rounded boxes, tapered cylinders, blobs), shared geometries, and outlines that work in
// every render style. Each part still draws through <Surface> and <Ink> from style.tsx.
//
// Outlines: Ink's crease lines (drei <Edges>) suit sharp boxes but turn noisy, or vanish, on
// rounded geometry. So in Cartoon, rounded parts get an inverted-hull outline instead (a slightly
// bigger back-face copy in ink), which traces silhouettes and the seams between cushions cleanly.
// In the line styles (Blueprint, Scan) a rounded box is drawn with the crisp lines of its plain box.
import { useMemo } from 'react'
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { Ink, Surface, useStyle } from './style'

export type V3 = [number, number, number]

export interface Part {
  /** box: sharp. soft: rounded box (radius r). cyl: cylinder, `top`/`bottom` scale its radius (seg 4 = square leg). ball: ellipsoid. */
  kind?: 'box' | 'soft' | 'cyl' | 'ball'
  /** Full size on x, y, z (for cylinders and balls, the bounding box). */
  size: V3
  /** Center. */
  pos: V3
  rot?: V3
  /** Explicit color; otherwise the item's color times `shade` (lightness factor). */
  color?: string
  shade?: number
  r?: number
  top?: number
  bottom?: number
  seg?: number
  /** Fabric (matte) instead of hard furniture surface. */
  fabric?: boolean
  /** Small props skip shadow casting (cheaper, and their shadows are just noise). */
  noShadow?: boolean
}

// Thumbnail palette (src/game/ItemThumb.tsx), so the 3D shares one look with the drawer.
export const PAL = {
  wood: '#8a6b4f',
  white: '#fbf6ee',
  linen: '#fffaf1',
  metal: '#9aa0a6',
  dark: '#2f2b28',
  screen: '#3b4452',
  pot: '#d98a5f',
  potRim: '#e8a27c',
  soil: '#6b4a35',
  shade: '#fff1c9',
  pillow: '#f6c445',
  ink: '#3d2c25',
  books: ['#ff7a59', '#5fa67f', '#4f8be8', '#ffc24b', '#8a5cc7', '#e8603f'],
}

export const shade = (hex: string, f: number) => {
  const c = new THREE.Color(hex)
  const hsl = { h: 0, s: 0, l: 0 }
  c.getHSL(hsl)
  c.setHSL(hsl.h, hsl.s, Math.max(0, Math.min(1, hsl.l * f)))
  return `#${c.getHexString()}`
}

// ------------------------------------------------------------------ shared geometries

const UNIT_BOX = new THREE.BoxGeometry(1, 1, 1)
const UNIT_BALL = new THREE.SphereGeometry(0.5, 12, 9)
const cyls = new Map<string, THREE.BufferGeometry>()
const rboxes = new Map<string, THREE.BufferGeometry>()

/** Unit-height cylinder fitting a unit box; 4 segments makes a (tapered) square leg. */
function cylGeometry(top = 1, bottom = 1, seg = 12) {
  const key = `${top}|${bottom}|${seg}`
  let g = cyls.get(key)
  if (!g) {
    const k = seg === 4 ? Math.SQRT1_2 : 0.5
    g = new THREE.CylinderGeometry(k * top, k * bottom, 1, seg, 1)
    if (seg === 4) g.rotateY(Math.PI / 4)
    cyls.set(key, g)
  }
  return g
}

const mm = (v: number) => Math.round(v * 1000)
function roundedGeometry([x, y, z]: V3, r: number) {
  const key = `${mm(x)}|${mm(y)}|${mm(z)}|${mm(r)}`
  let g = rboxes.get(key)
  if (!g) {
    g = new RoundedBoxGeometry(x, y, z, 2, Math.min(r, x / 2, y / 2, z / 2))
    rboxes.set(key, g)
  }
  return g
}

// The Cartoon outline: back faces pushed out along their normals by a constant number of screen
// pixels, so soft shapes get the same line weight as the <Edges> ink on sharp ones at any zoom.
const hullRes = { value: new THREE.Vector2(1, 1) }
const hullMats = new Map<string, THREE.ShaderMaterial>()
function hullMaterial(color: string, px: number) {
  const key = `${color}|${px}`
  let m = hullMats.get(key)
  if (!m) {
    m = new THREE.ShaderMaterial({
      uniforms: { color: { value: new THREE.Color(color) }, px: { value: px }, res: hullRes },
      vertexShader: /* glsl */ `
        uniform float px;
        uniform vec2 res;
        void main() {
          vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          vec3 n = normalize(normalMatrix * normal);
          vec2 dir = (projectionMatrix * vec4(n, 0.0)).xy;
          float len = length(dir);
          if (len > 1e-5) clip.xy += (dir / len) * px * 2.0 / res * clip.w;
          gl_Position = clip;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 color;
        void main() {
          gl_FragColor = vec4(color, 1.0);
          #include <colorspace_fragment>
        }`,
      side: THREE.BackSide,
    })
    hullMats.set(key, m)
  }
  return m
}
const _size = new THREE.Vector2()
/** Keeps the outline width in pixels: the hull shader needs the drawing buffer size and pixel ratio. */
function hullBeforeRender(this: THREE.Mesh, renderer: THREE.WebGLRenderer) {
  renderer.getDrawingBufferSize(_size)
  if (!_size.equals(hullRes.value)) hullRes.value.copy(_size)
  const m = this.material as THREE.ShaderMaterial
  const want = (m.userData.css as number) * renderer.getPixelRatio()
  if (m.uniforms.px.value !== want) m.uniforms.px.value = want
}
function outlineMaterial(color: string, cssPx: number) {
  const m = hullMaterial(color, cssPx)
  m.userData.css = cssPx
  return m
}

const noRaycast = () => null

// ------------------------------------------------------------------ one part

export function PartMesh({ part, base, highlight }: { part: Part; base: string; highlight?: string }) {
  const { style } = useStyle()
  const kind = part.kind ?? 'box'
  const color = part.color ?? (part.shade ? shade(base, part.shade) : base)
  const geo = useMemo(
    () => (kind === 'soft' ? roundedGeometry(part.size, part.r ?? 0.03) : kind === 'cyl' ? cylGeometry(part.top, part.bottom, part.seg) : kind === 'ball' ? UNIT_BALL : UNIT_BOX),
    [kind, part.size, part.r, part.top, part.bottom, part.seg],
  )
  // Soft boxes are built at their true size; the others are unit shapes scaled to size.
  const scale: V3 = kind === 'soft' ? [1, 1, 1] : part.size
  const round = kind !== 'box'
  const lines = style === 'blueprint' || style === 'scan'
  // A selected piece keeps its own material and gets an outline in the highlight color. Only the
  // line styles (their fill is the look) and the classic editor (a tint) color the surface itself.
  const tint = lines || style === 'classic' ? highlight : undefined
  const hull = round && (style === 'toon' || (!!highlight && !lines && style !== 'classic'))
  return (
    <group position={part.pos} rotation={part.rot}>
      <mesh geometry={geo} scale={scale} castShadow={!part.noShadow} receiveShadow>
        <Surface role={part.fabric ? 'soft' : 'item'} color={color} highlight={tint} />
        {(!round || kind === 'cyl' || (kind === 'ball' && lines)) && <Ink highlight={highlight} soft={round} />}
      </mesh>
      {hull && (
        <mesh geometry={geo} scale={scale} material={outlineMaterial(highlight ?? PAL.ink, highlight ? 2.4 : 1.5)} raycast={noRaycast} onBeforeRender={hullBeforeRender} />
      )}
      {kind === 'soft' && lines && (
        <mesh geometry={UNIT_BOX} scale={part.size} raycast={noRaycast}>
          <meshBasicMaterial visible={false} />
          <Ink highlight={highlight} />
        </mesh>
      )}
    </group>
  )
}

// ------------------------------------------------------------------ helpers for building parts

/** Four tapered legs of height `h` (from the floor), inset from the corners of a w x d footprint. */
export function legs(w: number, d: number, h: number, o: { t?: number; inset?: number; color?: string; shade?: number; taper?: number } = {}): Part[] {
  const t = o.t ?? 0.04
  const inset = o.inset ?? 0.03
  const out: Part[] = []
  for (const sx of [-1, 1])
    for (const sz of [-1, 1])
      out.push({
        kind: 'cyl',
        seg: 4,
        top: 1,
        bottom: o.taper ?? 0.62,
        size: [t, h, t],
        pos: [sx * (w / 2 - inset - t / 2), h / 2, sz * (d / 2 - inset - t / 2)],
        color: o.color,
        shade: o.color ? undefined : o.shade ?? 0.78,
      })
  return out
}

/** A small deterministic random sequence, so the same item always looks the same. */
export function rng(seed: string) {
  let s = 7
  for (let i = 0; i < seed.length; i++) s = (s * 31 + seed.charCodeAt(i)) % 2147483647
  if (s <= 0) s += 2147483646
  return () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646
}
