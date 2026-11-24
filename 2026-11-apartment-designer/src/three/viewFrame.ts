import type { Vec2 } from '../model/types'

/**
 * Which way the game's 3D overview camera looks, for keys that should follow the screen (arrow
 * nudges). `azimuth` turns the plan's "up" (-y) onto the camera's forward direction on the floor;
 * null when no game 3D view is showing (the plan, or the classic editor), so keys map to plan axes.
 *
 * `toFloor` / `toScreen` map between the screen (client px) and the floor (plan meters) for
 * whichever view is showing (the 3D camera or the plan), so a card dropped from the drawer lands
 * under the pointer. `toScreen` takes a height (m) for pieces that stand on a top.
 */
export const viewFrame: {
  azimuth: number | null
  toFloor: ((clientX: number, clientY: number) => Vec2 | null) | null
  toScreen: ((p: Vec2, height?: number) => { x: number; y: number } | null) | null
} = { azimuth: null, toFloor: null, toScreen: null }
