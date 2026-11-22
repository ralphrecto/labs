// The photo album: every photo kept as a small keepsake (see storage.ts), plus a polaroid made by
// itself when a remix reaches its first Dream home, so finished twists become a visible collection.
import { create } from 'zustand'
import { analyze } from '../analysis'
import { useStore } from '../store'
import { addAlbumPhoto, deleteAlbumPhoto, listAlbum, type AlbumPhoto as StoredPhoto } from '../storage'
import { VISIT_BY_ID } from './visits'

/** A kept photo, plus the visitor who loved the room that day (their emoji, as a corner sticker). */
export type AlbumPhoto = StoredPhoto & { visitor?: string }
import { useGame } from './gameStore'
import { LEVELS, scoreHome, TWISTS } from './goals'
import { renderForCapture } from '../three/capture'

/** A small picture for each twist (the picker, the badge, the album's twist polaroids). */
export const TWIST_EMOJI: Record<string, string> = {
  'talk-corner': '💬',
  'open-floor': '🕺',
  'window-seat': '🪟',
  'fresh-spots': '🔀',
  'bed-swap': '🛏️',
  'less-is-more': '🍃',
}

/** Bumped whenever the album changes, so the Designs sheet re-reads it. */
export const useAlbum = create<{ rev: number }>(() => ({ rev: 0 }))
const bump = () => useAlbum.setState((s) => ({ rev: s.rev + 1 }))

/** The shot, cover-cropped to 4:3 and downscaled to `width`, as a JPEG data URL (null on failure). */
export function shotDataURL(shot: HTMLCanvasElement, width = 480): string | null {
  try {
    const W = width
    const H = Math.round(width * 0.75)
    const out = document.createElement('canvas')
    out.width = W
    out.height = H
    const g = out.getContext('2d')
    if (!g || !shot.width || !shot.height) return null
    const k = Math.max(W / shot.width, H / shot.height)
    const cw = W / k
    const ch = H / k
    g.fillStyle = '#efe6da'
    g.fillRect(0, 0, W, H)
    g.drawImage(shot, (shot.width - cw) / 2, (shot.height - ch) / 2, cw, ch, 0, 0, W, H)
    return out.toDataURL('image/jpeg', 0.6)
  } catch {
    return null
  }
}

const uid = () => `ph_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`

/** Keeps a photo of the home's current design. Never throws; false when nothing was kept. */
export function keepPhoto(src: string | null, extra: { level: string; stars: number; evening: boolean; twist?: string; layoutId?: string }): boolean {
  if (!src) return false
  try {
    const { doc } = useStore.getState()
    const layoutId = extra.layoutId ?? doc.activeLayoutId
    const layout = doc.layouts.find((l) => l.id === layoutId)
    // Today's visitor rides along on the photo when they were happy with the open design.
    const vs = useGame.getState().visit
    const visitor = vs.done && layoutId === doc.activeLayoutId ? VISIT_BY_ID[vs.visitId]?.emoji : undefined
    const photo: AlbumPhoto = {
      id: uid(),
      homeId: doc.apartment.id,
      layoutId,
      layoutName: layout?.name ?? '',
      date: Date.now(),
      level: extra.level,
      stars: extra.stars,
      evening: extra.evening,
      src,
      ...(visitor ? { visitor } : {}),
      ...(extra.twist ? { twist: extra.twist, emoji: TWIST_EMOJI[extra.twist], pinned: true } : {}),
    }
    const ok = addAlbumPhoto(photo)
    bump()
    return ok
  } catch {
    return false
  }
}

export const albumFor = (homeId: string): AlbumPhoto[] => listAlbum(homeId)
export function removePhoto(homeId: string, id: string) {
  const ok = deleteAlbumPhoto(homeId, id)
  bump()
  return ok
}
/** Puts a just-deleted photo back (the lightbox's Undo). */
export function restorePhoto(p: AlbumPhoto) {
  const ok = addAlbumPhoto(p)
  bump()
  return ok
}

/** No 3D view to snap (the plan is showing): a paper card with the twist's emoji. */
function paperCard(emoji: string): string | null {
  try {
    const c = document.createElement('canvas')
    c.width = 360
    c.height = 270
    const g = c.getContext('2d')
    if (!g) return null
    const grd = g.createLinearGradient(0, 0, 0, 270)
    grd.addColorStop(0, '#f6ead8')
    grd.addColorStop(1, '#e8d3b8')
    g.fillStyle = grd
    g.fillRect(0, 0, 360, 270)
    g.font = '120px system-ui, sans-serif'
    g.textAlign = 'center'
    g.textBaseline = 'middle'
    g.fillText(emoji, 180, 140)
    return c.toDataURL('image/jpeg', 0.6)
  } catch {
    return null
  }
}

/** A remix's first Dream home: snap the view (or a paper card) into the album, tagged with its twist. */
function dreamPolaroid(layoutId: string, twist: string) {
  try {
    const { doc } = useStore.getState()
    const layout = doc.layouts.find((l) => l.id === layoutId)
    if (!layout) return
    let stars = 0
    try {
      const vs = useGame.getState().visit
      const visit = vs.visitId && layoutId === doc.activeLayoutId ? { id: vs.visitId, done: vs.done } : null
      stars = scoreHome(doc, layout, analyze(doc, layout), visit).stars
    } catch {
      /* the caption just shows no stars */
    }
    const canvas = typeof document === 'undefined' ? null : document.querySelector<HTMLCanvasElement>('.hud-stage canvas')
    // The canvas keeps no drawing buffer: draw the view now and read it in the same task.
    if (canvas && layoutId === doc.activeLayoutId) renderForCapture()
    const src = (canvas && layoutId === doc.activeLayoutId ? shotDataURL(canvas) : null) ?? paperCard(TWIST_EMOJI[twist] ?? '↻')
    const kept = keepPhoto(src, {
      level: `Level ${LEVELS.length} · ${LEVELS[LEVELS.length - 1]}`,
      stars,
      evening: useGame.getState().evening,
      twist,
      layoutId,
    })
    if (kept) useStore.getState().notify(`${TWIST_EMOJI[twist] ?? '↻'} ${TWISTS[twist as keyof typeof TWISTS]?.name ?? 'Twist'} done! A polaroid went into your album.`)
  } catch {
    /* a keepsake is a bonus: never break the game over it */
  }
}

// Watch for a remix's first Dream home (markRemixDream sets `remix.dreamAt` once), in the same home.
let watching = false
export function watchRemixDreams() {
  if (watching) return
  watching = true
  useStore.subscribe((s, prev) => {
    if (s.doc === prev.doc || s.doc.apartment.id !== prev.doc.apartment.id) return
    for (const l of s.doc.layouts) {
      if (!l.remix?.dreamAt) continue
      const before = prev.doc.layouts.find((x) => x.id === l.id)
      if (before && !before.remix?.dreamAt) {
        const { id } = l
        const twist = l.remix.twist
        // After the frame that shows the finished room.
        const later = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (f: () => void) => setTimeout(f, 16)
        later(() => dreamPolaroid(id, twist))
      }
    }
  })
}
