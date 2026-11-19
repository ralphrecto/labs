import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { placementShift } from '../scan/apply'
import { reviewChoice, useScanUI } from '../scan/scanUi'
import type { ScanData } from '../scan/types'
import { useStore } from '../store'

// Scan point clouds in the 3D view, lined up with the rooms they were detected from.

const MAX_POINTS = 300_000

function geometryFor(scan: ScanData, ceiling: number) {
  const { count, positions, colors } = scan.points
  const every = Math.max(1, Math.ceil(count / MAX_POINTS))
  const cut = ceiling - 0.12
  const pos: number[] = []
  const col: number[] = []
  for (let i = 0; i < count; i += every) {
    const h = positions[i * 3 + 1]
    if (h > cut) continue // leave the ceiling out so you can see in
    pos.push(positions[i * 3], h, positions[i * 3 + 2])
    if (colors) col.push(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2])
    else col.push(0.6, 0.6, 0.6)
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
  return g
}

function Cloud({ scan, ceiling, x, y, rotation }: { scan: ScanData; ceiling: number; x: number; y: number; rotation: number }) {
  const geom = useMemo(() => geometryFor(scan, ceiling), [scan, ceiling])
  useEffect(() => () => geom.dispose(), [geom])
  // Plan rotation is clockwise looking down, which is a negative turn about +Y.
  return (
    <group position={[x, 0, y]} rotation={[0, -rotation, 0]}>
      <points geometry={geom} raycast={() => null}>
        <pointsMaterial size={0.012} vertexColors sizeAttenuation transparent opacity={0.9} />
      </points>
    </group>
  )
}

export function ScanPoints() {
  const doc = useStore((s) => s.doc)
  const show = useScanUI((s) => s.showUnderlay)
  const clouds = useScanUI((s) => s.clouds)
  const review = useScanUI((s) => s.review)
  const linked = doc.rooms.filter((r) => r.scan)
  useEffect(() => {
    for (const r of linked) useScanUI.getState().ensureCloud(r.scan!.scanId)
  }, [linked.map((r) => r.scan!.scanId).join()]) // eslint-disable-line react-hooks/exhaustive-deps
  const shift = useMemo(() => (review ? placementShift(doc, review.estimate, reviewChoice(review)) : null), [doc, review])
  const reviewCloud = review && clouds[review.estimate.scanId]
  return (
    <>
      {show &&
        linked.map((r) => {
          const c = clouds[r.scan!.scanId]
          return c ? <Cloud key={r.id} scan={c} ceiling={r.ceilingHeight} x={r.scan!.x} y={r.scan!.y} rotation={r.scan!.rotation} /> : null
        })}
      {review && reviewCloud && shift && (
        <Cloud
          scan={reviewCloud}
          ceiling={review.estimate.room?.ceilingHeight ?? 2.6}
          x={review.estimate.transform.x + shift.x}
          y={review.estimate.transform.y + shift.y}
          rotation={review.estimate.transform.rotation}
        />
      )}
    </>
  )
}
