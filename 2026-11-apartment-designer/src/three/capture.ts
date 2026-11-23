/**
 * The 3D canvas draws on demand and doesn't keep its drawing buffer, so a picture read back from it
 * later would be blank. The scene registers a synchronous render here; a reader calls
 * renderForCapture() and reads the canvas in the same task, before the frame is presented.
 *
 * A photo is framed on the home, not on the screen: the overview registers a framing that turns
 * the live camera into one fitted to the home's bounds in the middle of the canvas (where the
 * polaroid and album crops look), without the HUD's view offset. The live view never moves.
 */
let renderNow: (() => void) | null = null
let framing: (<C>(camera: C) => C) | null = null

export function setCaptureRenderer(fn: (() => void) | null) {
  renderNow = fn
}

/** The overview's capture framing (null when no overview is showing: walking, the plan). */
export function setCaptureFraming(fn: (<C>(camera: C) => C) | null) {
  framing = fn
}

/** The camera a capture draws through: the live one, fitted to the home when an overview is showing. */
export function captureCamera<C>(camera: C): C {
  try {
    return framing ? framing(camera) : camera
  } catch {
    return camera
  }
}

/** Draw the current view into the canvas right now. Read the canvas before yielding. */
export function renderForCapture() {
  try {
    renderNow?.()
  } catch {
    // A lost context: the reader gets whatever the canvas holds.
  }
}
