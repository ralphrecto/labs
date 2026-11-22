/**
 * The 3D canvas draws on demand and doesn't keep its drawing buffer, so a picture read back from it
 * later would be blank. The scene registers a synchronous render here; a reader calls
 * renderForCapture() and reads the canvas in the same task, before the frame is presented.
 */
let renderNow: (() => void) | null = null

export function setCaptureRenderer(fn: (() => void) | null) {
  renderNow = fn
}

/** Draw the current view into the canvas right now. Read the canvas before yielding. */
export function renderForCapture() {
  try {
    renderNow?.()
  } catch {
    // A lost context: the reader gets whatever the canvas holds.
  }
}
