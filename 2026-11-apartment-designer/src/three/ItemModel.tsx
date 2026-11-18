import { useGLTF } from '@react-three/drei'
import { Suspense, useMemo } from 'react'
import * as THREE from 'three'
import type { Item } from '../model/types'

interface BoxSpec {
  size: [number, number, number]
  pos: [number, number, number]
  color?: string
  shade?: number
}

const shade = (hex: string, f: number) => {
  const c = new THREE.Color(hex)
  const hsl = { h: 0, s: 0, l: 0 }
  c.getHSL(hsl)
  c.setHSL(hsl.h, hsl.s, Math.max(0, Math.min(1, hsl.l * f)))
  return `#${c.getHexString()}`
}

/** Procedural low-poly looks for catalog categories. Local frame: width on x, depth on z, front is +z, floor at y=0. */
function parts(item: Item): BoxSpec[] {
  const { width: w, depth: d, height: h } = item
  const legs = (top: number, inset = 0.04, t = 0.04): BoxSpec[] =>
    [
      [-1, -1],
      [1, -1],
      [-1, 1],
      [1, 1],
    ].map(([sx, sz]) => ({ size: [t, top, t], pos: [sx * (w / 2 - inset - t / 2), top / 2, sz * (d / 2 - inset - t / 2)], shade: 0.7 }))
  switch (item.category) {
    case 'sofa':
    case 'armchair': {
      const seat = Math.min(0.45, h * 0.5)
      const arm = Math.min(0.18, w * 0.2)
      const back = Math.min(0.22, d * 0.3)
      return [
        { size: [w, seat, d], pos: [0, seat / 2, 0] },
        { size: [w, h, back], pos: [0, h / 2, -d / 2 + back / 2], shade: 0.9 },
        { size: [arm, Math.min(h, 0.65), d], pos: [-w / 2 + arm / 2, Math.min(h, 0.65) / 2, 0], shade: 0.92 },
        { size: [arm, Math.min(h, 0.65), d], pos: [w / 2 - arm / 2, Math.min(h, 0.65) / 2, 0], shade: 0.92 },
      ]
    }
    case 'bed': {
      const base = Math.min(0.32, h * 0.55)
      const mat = Math.max(0.05, h - base)
      return [
        { size: [w, base, d], pos: [0, base / 2, 0], shade: 0.6 },
        { size: [w - 0.04, mat, d - 0.06], pos: [0, base + mat / 2, 0.01], color: '#f1efe9' },
        { size: [w, 1.0, 0.08], pos: [0, 0.5, -d / 2 + 0.04], shade: 0.55 },
        { size: [w * 0.4, 0.12, 0.35], pos: [-w * 0.23, h + 0.06, -d / 2 + 0.3], color: '#ffffff' },
        { size: [w * 0.4, 0.12, 0.35], pos: [w * 0.23, h + 0.06, -d / 2 + 0.3], color: '#ffffff' },
        { size: [w - 0.02, 0.04, d * 0.55], pos: [0, h + 0.01, d * 0.2], shade: 1.05 },
      ]
    }
    case 'coffee-table':
    case 'dining-table':
    case 'desk':
      return [{ size: [w, 0.04, d], pos: [0, h - 0.02, 0] }, ...legs(h - 0.04)]
    case 'nightstand':
    case 'dresser':
    case 'tv-stand':
    case 'wardrobe':
      return [
        { size: [w, h - 0.08, d], pos: [0, (h - 0.08) / 2 + 0.08, 0] },
        { size: [w - 0.06, 0.08, d - 0.06], pos: [0, 0.04, 0], shade: 0.5 },
        { size: [w - 0.06, 0.005, 0.01], pos: [0, h * 0.55, d / 2 + 0.003], shade: 0.6 },
      ]
    case 'dining-chair':
    case 'office-chair': {
      const seatH = 0.46
      return [
        { size: [w, 0.05, d], pos: [0, seatH, 0] },
        { size: [w, h - seatH, 0.04], pos: [0, seatH + (h - seatH) / 2, -d / 2 + 0.02], shade: 0.9 },
        ...legs(seatH - 0.025, 0.02, 0.03),
      ]
    }
    case 'bookshelf': {
      const shelves = Math.max(2, Math.round(h / 0.35))
      const out: BoxSpec[] = [
        { size: [w, h, 0.02], pos: [0, h / 2, -d / 2 + 0.01], shade: 0.8 },
        { size: [0.02, h, d], pos: [-w / 2 + 0.01, h / 2, 0] },
        { size: [0.02, h, d], pos: [w / 2 - 0.01, h / 2, 0] },
      ]
      for (let i = 0; i <= shelves; i++) out.push({ size: [w, 0.02, d], pos: [0, Math.min(h - 0.01, (i * h) / shelves + 0.01), 0] })
      return out
    }
    case 'rug':
      return [{ size: [w, Math.max(0.005, h), d], pos: [0, Math.max(0.005, h) / 2, 0] }]
    case 'floor-lamp':
      return [
        { size: [Math.min(w, 0.3), 0.03, Math.min(d, 0.3)], pos: [0, 0.015, 0], shade: 0.4 },
        { size: [0.025, h - 0.3, 0.025], pos: [0, (h - 0.3) / 2, 0], shade: 0.4 },
        { size: [w, 0.3, d], pos: [0, h - 0.15, 0], color: '#f4ead2' },
      ]
    case 'plant':
      return [
        { size: [w * 0.6, 0.35, d * 0.6], pos: [0, 0.175, 0], color: '#b2765a' },
        { size: [w, h - 0.35, d], pos: [0, 0.35 + (h - 0.35) / 2, 0] },
      ]
    default:
      return [{ size: [w, h, d], pos: [0, h / 2, 0] }]
  }
}

export function ProceduralItem({ item, highlight }: { item: Item; highlight?: string }) {
  const specs = useMemo(() => parts(item), [item])
  return (
    <group>
      {specs.map((s, i) => (
        <mesh key={i} position={s.pos} castShadow receiveShadow>
          <boxGeometry args={s.size} />
          <meshStandardMaterial
            color={s.color ?? (s.shade ? shade(item.color, s.shade) : item.color)}
            roughness={0.85}
            emissive={highlight ?? '#000000'}
            emissiveIntensity={highlight ? 0.35 : 0}
          />
        </mesh>
      ))}
    </group>
  )
}

function GLBItem({ item }: { item: Item }) {
  const { scene } = useGLTF(item.glbUrl!)
  const obj = useMemo(() => {
    // Scale the model so its bounding box matches the item's true size; listed dimensions win.
    const clone = scene.clone(true)
    const box = new THREE.Box3().setFromObject(clone)
    const size = box.getSize(new THREE.Vector3())
    const center = box.getCenter(new THREE.Vector3())
    clone.position.set(-center.x, -box.min.y, -center.z)
    const g = new THREE.Group()
    g.add(clone)
    g.scale.set(item.width / (size.x || 1), item.height / (size.y || 1), item.depth / (size.z || 1))
    return g
  }, [scene, item.width, item.height, item.depth])
  return <primitive object={obj} />
}

export function ItemModel({ item, highlight }: { item: Item; highlight?: string }) {
  if (item.glbUrl)
    return (
      <Suspense fallback={<ProceduralItem item={item} highlight={highlight} />}>
        <GLBItem item={item} />
      </Suspense>
    )
  return <ProceduralItem item={item} highlight={highlight} />
}
