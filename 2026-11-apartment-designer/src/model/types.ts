// Core data model. Everything is stored in meters.
// Plan coordinates: x to the right, y "down" on the plan. In 3D, plan (x, y)
// maps to world (x, 0, y) with Y up and the floor at zero.

export type Vec2 = { x: number; y: number }

export type Units = 'metric' | 'imperial'

export type OpeningKind = 'door' | 'window'

export interface Opening {
  id: string
  kind: OpeningKind
  /** Index of the wall: the segment from polygon[wall] to polygon[wall + 1]. */
  wall: number
  /** Distance in meters from the wall's start vertex to the opening's near edge. */
  offset: number
  width: number
  height: number
  /** Height of the bottom edge above the floor (0 for doors). */
  sill: number
  /** Doors only: which end of the opening the hinge is on. */
  hinge?: 'start' | 'end'
  /** Doors only: whether the door swings into this room or away from it. */
  swing?: 'in' | 'out'
  /** Doors only: marks the apartment's front door, used by the walking path check. */
  entrance?: boolean
}

export interface Room {
  id: string
  name: string
  /** Interior outline of the room (inside faces of the walls), clockwise or counterclockwise. */
  polygon: Vec2[]
  ceilingHeight: number
  wallThickness: number
  openings: Opening[]
  /** Set for rooms detected from a scan: how scan world (x, z) maps onto the plan, for the point cloud underlay. */
  scan?: RoomScanLink
}

export interface RoomScanLink {
  scanId: string
  /** Plan position of the scan's world origin, and rotation (clockwise on the plan) applied to world (x, z). */
  x: number
  y: number
  rotation: number
}

export interface Clearance {
  front: number
  back: number
  left: number
  right: number
}

export type ItemSource = 'catalog' | 'custom' | 'scan' | 'link' | 'photo'

export interface Item {
  id: string
  name: string
  /** Catalog category key, used for the 3D look and default clearances. */
  category: string
  /** Size along the item's local x axis (side to side). */
  width: number
  /** Size along the item's local front/back axis. */
  depth: number
  height: number
  clearance: Clearance
  color: string
  /** Frame finish for framed wall pieces (wall art, mirrors); the model's wood frame when unset. */
  frameColor?: string
  /** Rugs and other floor coverings can sit under other items. */
  floorLevel?: boolean
  /** Optional model; the box is still used for collisions and snapping. */
  glbUrl?: string
  source: ItemSource
  price?: number
  link?: string
  /** True for items you already own (as opposed to items you're considering buying). */
  owned?: boolean
  /** Product photo: a small data URL from an uploaded photo, or the retailer's image URL. */
  imageUrl?: string
  /** The size was estimated (from a photo or typical sizes) rather than read from a listing. */
  sizeEstimated?: boolean
  /** Items detected in a scan: which scan, and how sure the detector was about the type (0..1). */
  scanId?: string
  confidence?: number
  /** A moving box: the catalog categories inside it, unpacked with the Unpack action. */
  unpack?: string[]
  /** The box has been unpacked (its placement is gone; the item stays so the goal can count it). */
  unpacked?: boolean
}

export interface Placement {
  id: string
  itemId: string
  /** Center of the footprint on the plan. */
  x: number
  y: number
  /** Rotation in radians, clockwise on the plan. 0 means the front faces +y (down on the plan). */
  rotation: number
  /** Placement id of the furniture this sits on (table lamps). Same layout. x/y stay absolute plan coordinates. */
  on?: string
  /** Wall decor only: height (m) of the piece's bottom edge when it was hung above or below another frame. Unset: the category's usual height (see wallDecor.mountBottom). */
  mountY?: number
}

export interface Layout {
  id: string
  name: string
  placements: Placement[]
  /** A remix: started from another design's big pieces, with a twist to meet (see TWISTS in game/goals). */
  remix?: Remix
  /** How the player wants each room to feel (round 17 mood moments), by room id. See game/moods. */
  moods?: Record<string, MoodId>
}

/** A room's mood, chosen on a mood moment card: Bright & airy, Warm & snug, Bold & playful. */
export type MoodId = 'airy' | 'snug' | 'bold'

/** The twist challenges a remix can take on. */
export type TwistId = 'window-seat' | 'fresh-spots' | 'bed-swap' | 'talk-corner' | 'open-floor' | 'less-is-more'

export interface Remix {
  /** The design it was remixed from. */
  from: string
  twist: TwistId
  /** How many anchor pieces it started with. */
  anchors: number
  /** When it first reached Dream home. */
  dreamAt?: number
}

export interface Apartment {
  id: string
  name: string
  units: Units
}

export interface Doc {
  version: 1
  apartment: Apartment
  rooms: Room[]
  items: Item[]
  layouts: Layout[]
  activeLayoutId: string
  /** Imported 3D scans (Gaussian splats, meshes, point clouds) used as a backdrop. The files live in IndexedDB. */
  scans?: ScanRef[]
}

/** Which axis of the scan file points up. Scaniverse and glTF use +Y; some PLY pipelines use -Y or Z. */
export type UpAxis = 'y' | '-y' | 'z' | '-z'

export type ScanFormat = 'splat' | 'mesh' | 'points'

/**
 * A tape-measured length on the scan. The two ends are in the scan's own top-down
 * coordinates (file units, after the up axis is applied), so the scale can change freely.
 */
export interface ScanReference {
  id: string
  a: Vec2
  b: Vec2
  /** The true length in meters. */
  length: number
  label?: string
}

/**
 * An imported scan and how it sits on the plan.
 *
 * A point p in the file is first turned upright (upAxis), giving (ux, uy, uz) with uy up.
 * Its plan position is (x, y) + scale * rotate(yaw) * (ux, uz), and its height above the
 * floor is scale * (uy - floor).
 */
export interface ScanRef {
  id: string
  name: string
  fileName: string
  fileSize: number
  format: ScanFormat
  upAxis: UpAxis
  /** Floor height in upright file units. */
  floor: number
  /** The floor height the importer found, when the user has nudged `floor` away from it. */
  floorDetected?: number
  /** Ceiling height in upright file units, when one was found. */
  ceiling?: number
  /** Meters per file unit. 1 until calibrated, since Scaniverse exports are roughly metric. */
  scale: number
  /** Rotation on the plan in radians (same sense as SVG rotate: clockwise on screen). */
  yaw: number
  x: number
  y: number
  references: ScanReference[]
  visible: boolean
  /** What the plan shows under the walls. */
  underlay: 'walls' | 'top' | 'off'
  opacity: number
  /** Height band, in meters above the floor, whose points draw the wall outline. */
  sliceLow: number
  sliceHigh: number
  /** Meters above the floor: the top view and the 3D backdrop hide everything above this. */
  cutHeight: number
}
