// ---------------------------------------------------------------- detection output
// Output of furniture detection. types.ts (the scan input) is shared with the scan import code.
// Coordinates are plan coordinates (meters, x right, y down) unless noted.

import type { Vec2 } from '../model/types'

/** How scan world coordinates map to the plan: rotate world (x, z) by `rotation`, then add (x, y). */
export interface ScanTransform {
  x: number
  y: number
  rotation: number
}

export interface DetectedOpening {
  kind: 'door' | 'window'
  /** Wall index in the detected room's polygon. */
  wall: number
  offset: number
  width: number
  height: number
  sill: number
  confidence: number
}

export interface DetectedRoom {
  polygon: Vec2[]
  ceilingHeight: number
  openings: DetectedOpening[]
  confidence: number
}

export interface CategoryGuess {
  category: string
  /** Display name, e.g. "Queen bed". */
  name: string
  score: number
}

export interface DetectedObject {
  id: string
  category: string
  name: string
  /** 0..1, how sure the classifier is about the category (not about the box). */
  confidence: number
  /** Other likely categories, best first. */
  alternatives: CategoryGuess[]
  /** Footprint center on the plan, and rotation in the app's convention (front = local +y, clockwise). */
  x: number
  y: number
  rotation: number
  width: number
  depth: number
  height: number
  /** Mean point color, if the scan has colors. */
  color?: string
  pointCount: number
  /** True when the back of the box was found against a wall (so the front direction is reliable). */
  againstWall: boolean
}

/**
 * What a detector produces from a scan: a room shell and furniture boxes, in plan coordinates.
 * The on-device heuristic detector and server models (SpatialLM) both produce this, and the
 * review UI and `applyEstimate` only consume this.
 */
export interface SceneEstimate {
  scanId: string
  detector: 'heuristic' | 'spatiallm'
  transform: ScanTransform
  room: DetectedRoom | null
  objects: DetectedObject[]
  notes: string[]
}
