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
}

export interface Placement {
  id: string
  itemId: string
  /** Center of the footprint on the plan. */
  x: number
  y: number
  /** Rotation in radians, clockwise on the plan. 0 means the front faces +y (down on the plan). */
  rotation: number
}

export interface Layout {
  id: string
  name: string
  placements: Placement[]
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
}
