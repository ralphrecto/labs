import { activeLayout, useStore } from '../store'
import { useGame } from './gameStore'

/** The placement is a moving box that still has things in it. */
export function isPackedBox(placementId: string): boolean {
  const d = useStore.getState().doc
  const p = activeLayout(d).placements.find((x) => x.id === placementId)
  const item = p && d.items.find((i) => i.id === p.itemId)
  return !!item?.unpack?.length && !item.unpacked
}

/** Unpack a moving box (one undoable step) and play the pop and confetti for what came out. */
export function unpackAndCelebrate(placementId: string): string[] {
  const st = useStore.getState()
  const box = activeLayout(st.doc).placements.find((p) => p.id === placementId)
  if (!box || !isPackedBox(placementId)) return []
  const ids = st.unpackBox(placementId)
  // Keep the first piece selected: the selection bar stays up, so the camera doesn't refit to the
  // whole home (which would play the pop tiny) and the player's eye stays on the spot.
  if (ids.length) useStore.getState().select({ kind: 'placement', id: ids[0] })
  useGame.getState().setUnpacked({ key: Date.now(), ids, at: { x: box.x, y: box.y } })
  return ids
}
