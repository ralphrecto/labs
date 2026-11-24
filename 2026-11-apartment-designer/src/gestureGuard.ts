import { useEffect, type RefObject } from 'react'

/**
 * iOS Safari zooms the whole page on a two-finger pinch even where CSS `touch-action: none` says
 * not to, through its own (non-standard) gesturestart/gesturechange/gestureend events. Cancelling
 * those on `el` keeps the page at 1x. The app's own pinch (orbit controls, the plan's zoom) runs on
 * pointer/touch events, which still arrive, so it keeps working. Returns the cleanup.
 */
export function guardPageZoom(el: EventTarget): () => void {
  const stop = (e: Event) => e.preventDefault()
  const opts: AddEventListenerOptions = { passive: false }
  const types = ['gesturestart', 'gesturechange', 'gestureend'] as const
  for (const t of types) el.addEventListener(t, stop, opts)
  return () => {
    for (const t of types) el.removeEventListener(t, stop, opts)
  }
}

/** guardPageZoom on a ref's element while it is mounted. */
export function usePageZoomGuard(ref: RefObject<HTMLElement | SVGElement | null>) {
  useEffect(() => {
    const el = ref.current
    return el ? guardPageZoom(el) : undefined
  }, [ref])
}
