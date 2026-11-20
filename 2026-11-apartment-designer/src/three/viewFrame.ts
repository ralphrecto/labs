/**
 * Which way the game's 3D overview camera looks, for keys that should follow the screen (arrow
 * nudges). `azimuth` turns the plan's "up" (-y) onto the camera's forward direction on the floor;
 * null when no game 3D view is showing (the plan, or the classic editor), so keys map to plan axes.
 */
export const viewFrame: { azimuth: number | null } = { azimuth: null }
