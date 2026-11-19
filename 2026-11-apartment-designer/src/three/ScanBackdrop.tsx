import { SparkRenderer, SplatEdit, SplatEditRgbaBlendMode, SplatEditSdf, SplatEditSdfType, SplatMesh } from '@sparkjsdev/spark'
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import type { ScanRef } from '../model/types'
import { uprightRotationX } from '../scan/align'
import { useScanRuntime } from '../scan/runtime'
import { useStore } from '../store'

/** Spark draws every splat in the scene through one renderer object per canvas. */
function SparkLayer() {
  const gl = useThree((s) => s.gl)
  // Not disposed on unmount: React's strict mode remounts effects with the same memoized object.
  const spark = useMemo(() => new SparkRenderer({ renderer: gl }), [gl])
  return <primitive object={spark} />
}

/** Hides splats above a height: a half-space edit that multiplies their opacity by zero. */
function SplatCut({ height }: { height: number }) {
  const edit = useMemo(() => {
    const e = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY, softEdge: 0.02 })
    const plane = new SplatEditSdf({ type: SplatEditSdfType.PLANE, opacity: 0 })
    // The plane's inside is its local -Z side; turn local +Z to point down, so "inside" is above.
    plane.rotation.x = Math.PI / 2
    e.addSdf(plane)
    e.add(plane)
    return { e, plane }
  }, [])
  useEffect(() => setHeight(edit.plane, height), [edit, height])
  return <primitive object={edit.e} />
}

function ScanObject({ scan, cut }: { scan: ScanRef; cut: number | null }) {
  const loaded = useScanRuntime((s) => s.entries[scan.id]?.loaded)
  const gl = useThree((s) => s.gl)
  const object = useMemo(() => {
    if (!loaded) return null
    const r = loaded.render
    if (r.kind === 'splat') return new SplatMesh({ packedSplats: r.packed })
    if (r.kind === 'mesh') {
      const o = r.object.clone(true)
      o.traverse((c) => {
        const m = c as THREE.Mesh
        if (m.isMesh) m.material = Array.isArray(m.material) ? m.material.map((x) => x.clone()) : m.material.clone()
      })
      return o
    }
    return new THREE.Points(r.geometry, new THREE.PointsMaterial({ size: 0.012, vertexColors: !!r.geometry.getAttribute('color'), sizeAttenuation: true }))
  }, [loaded])

  // Meshes and point clouds clip with a plane; splats use a Spark edit instead.
  useEffect(() => {
    if (object && loaded?.render.kind !== 'splat') setClip(gl, object, cut)
  }, [object, cut, gl, loaded])

  if (!object) return null
  return (
    <group position={[scan.x, -scan.scale * scan.floor, scan.y]} rotation={[0, -scan.yaw, 0]} scale={scan.scale}>
      <group rotation={[uprightRotationX(scan.upAxis), 0, 0]}>
        <primitive object={object} />
      </group>
    </group>
  )
}

function setHeight(o: THREE.Object3D, y: number) {
  o.position.y = y
}

/** Clip a mesh or point cloud above a height (world Y), or not at all. */
function setClip(gl: THREE.WebGLRenderer, object: THREE.Object3D, cut: number | null) {
  gl.localClippingEnabled = true
  const planes = cut == null ? [] : [new THREE.Plane(new THREE.Vector3(0, -1, 0), cut)]
  object.traverse((c) => {
    const m = c as THREE.Mesh
    if (!m.material) return
    for (const mat of Array.isArray(m.material) ? m.material : [m.material]) {
      mat.clippingPlanes = planes
      mat.side = THREE.DoubleSide
      mat.needsUpdate = true
    }
  })
}

/** Imported scans in the 3D view, positioned and scaled exactly like their plan underlay. */
export function ScanBackdrop({ eye, force }: { eye: boolean; force?: boolean }) {
  const scans = useStore((s) => s.doc.scans)
  const show = useStore((s) => s.scan3d)
  const entries = useScanRuntime((s) => s.entries)
  const visible = (scans ?? []).filter((r) => r.visible)
  if ((show === 'off' && !force) || !visible.length) return null
  const hasSplat = visible.some((r) => entries[r.id]?.loaded?.render.kind === 'splat')
  // From above, cut the ceiling away; at eye level, show the whole room.
  const cutOf = (r: ScanRef) => (eye ? null : r.cutHeight)
  const splatCut = !eye ? Math.max(...visible.map((r) => r.cutHeight)) : null
  return (
    <>
      {hasSplat && <SparkLayer />}
      {hasSplat && splatCut != null && <SplatCut height={splatCut} />}
      {visible.map((r) => (
        <ScanObject key={r.id} scan={r} cut={cutOf(r)} />
      ))}
    </>
  )
}
