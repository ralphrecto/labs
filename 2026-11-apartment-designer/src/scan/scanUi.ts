import { create } from 'zustand'
import { useStore } from '../store'
import { analyzeScan } from './analyze'
import { applyEstimate, type ApplyChoice } from './apply'
import { displayName } from './classify'
import { cloudToScan } from './cloud'
import type { DetectedObject, SceneEstimate } from './estimate'
import { cachedScan, loadScan, saveScan } from './scanStore'
import { parseSpatialLM } from './spatiallm'
import type { ScanData } from './types'

// UI state for turning a scan into rooms and furniture: the review step between detection and
// the plan. Kept apart from the document store; only `apply` touches the document (one undo step).

export interface ReviewObject extends DetectedObject {
  accepted: boolean
}

export interface ScanReview {
  estimate: SceneEstimate
  objects: ReviewObject[]
  includeRoom: boolean
  replaceRoomId?: string
  roomName: string
  focusId: string | null
}

interface ScanUIState {
  busy: string | null
  review: ScanReview | null
  showUnderlay: boolean
  /** Leveled point clouds by scan id, loaded on demand for the underlay. */
  clouds: Record<string, ScanData>

  analyze: (scan: ScanData) => Promise<void>
  importSpatialLM: (text: string) => void
  setObject: (id: string, patch: Partial<ReviewObject>) => void
  setCategory: (id: string, category: string) => void
  setAll: (accepted: boolean) => void
  setReview: (patch: Partial<ScanReview>) => void
  apply: () => void
  cancel: () => void
  ensureCloud: (id: string) => void
  setShowUnderlay: (v: boolean) => void
}

const nextFrame = () => new Promise((r) => setTimeout(r, 30))

export function reviewChoice(r: ScanReview): ApplyChoice {
  return {
    includeRoom: r.includeRoom && !!r.estimate.room,
    replaceRoomId: r.replaceRoomId,
    roomName: r.roomName,
    objects: r.objects.filter((o) => o.accepted),
  }
}

/** A file name that says nothing about the room: "Scan 2026-11-20 14.32", "3DGS_8f2a…", "IMG_0042", a date or an id. */
export function isGenericScanName(name: string) {
  const n = name.trim().replace(/\.[a-z0-9]+$/i, '')
  if (!n) return true
  if (/^(scan|3dgs|splat|capture|untitled|img|mesh|model|export|scaniverse|room ?plan|new scan)(?![a-z])/i.test(n)) return true
  if (/\d{4}[-_. ]\d{1,2}[-_. ]\d{1,2}|\d{1,2}[-_. ]\d{1,2}[-_. ]\d{2,4}|\d{1,2}[.:]\d{2}/.test(n)) return true
  if (!/[a-z]{3,}/i.test(n.replace(/[0-9a-f]{6,}/gi, ''))) return true
  return false
}

/** Room types a scan's file name may mention, and the name the room gets for each. */
const ROOM_WORDS: [RegExp, string][] = [
  [/living|lounge|family ?room|sitting/i, 'Living room'],
  [/bed ?room|master|guest ?room|nursery|kids? ?room/i, 'Bedroom'],
  [/kitchen/i, 'Kitchen'],
  [/dining/i, 'Dining room'],
  [/office|study|den\b/i, 'Office'],
  [/bath|shower|wc\b|toilet/i, 'Bathroom'],
  [/hall|entry|foyer/i, 'Hallway'],
  [/studio|loft/i, 'Studio'],
]

/**
 * Name for the found room. The room name shows on the plan, on the piece labels and in the goals,
 * so a file name ("my-living-room", "Scan 2026-11-20") never becomes it. A name the player typed
 * themselves ("Sam's bedroom", "Kitchen") is kept; a file slug that names a room type gives that
 * type ("Living room"); otherwise the furniture says what the room is, or it's just "Your room".
 */
export function friendlyRoomName(fileName: string, objects: { category: string }[]) {
  const name = fileName.trim().replace(/\.[a-z0-9]+$/i, '')
  const words = name.replace(/[-_.]+/g, ' ')
  const type = ROOM_WORDS.find(([re]) => re.test(words))?.[1]
  // A slug (dashes or underscores, no spaces) or an all-lowercase name reads as a file, not a name.
  const slug = /[-_]/.test(name) && !/\s/.test(name)
  const typed = !slug && !isGenericScanName(name) && /[A-Z]/.test(name) && !/^sample\b/i.test(name)
  if (type && typed) return name
  if (type) return type
  const has = (...c: string[]) => objects.some((o) => c.includes(o.category))
  if (has('bed')) return 'Bedroom'
  if (has('sofa', 'armchair', 'tv-stand', 'coffee-table')) return 'Living room'
  if (has('desk')) return 'Office'
  if (has('dining-table')) return 'Dining room'
  return 'Your room'
}

/** A picked file we don't read: say which formats do work, and how to get one out of Scaniverse. */
export function notScanFileMessage(name: string) {
  const e = name.toLowerCase().split('.').pop() ?? ''
  const unsupported = e === 'usdz' || e === 'fbx' || e === 'usd' || e === 'stl'
  return `${unsupported ? `That’s a .${e} file: .usdz and .fbx aren’t supported yet. ` : 'That isn’t a scan file. '}In Scaniverse, tap Share, then Export, choose PLY (or SPZ), and pick that file here. GLB and OBJ meshes work too.`
}

function startReview(estimate: SceneEstimate, name: string): ScanReview {
  return {
    estimate,
    // Unknown objects start unchecked: they're often clutter or bits of a bigger piece.
    objects: estimate.objects.map((o) => ({ ...o, accepted: o.category !== 'box' })),
    includeRoom: !!estimate.room,
    roomName: name,
    focusId: null,
  }
}

export const useScanUI = create<ScanUIState>((set, get) => ({
  busy: null,
  review: null,
  showUnderlay: true,
  clouds: {},

  analyze: async (scan) => {
    set({ busy: `Finding walls and furniture in ${scan.points.count.toLocaleString()} points…` })
    await nextFrame()
    try {
      const { estimate, cloud } = analyzeScan(scan)
      const leveled = cloudToScan(cloud, scan)
      set((s) => ({ clouds: { ...s.clouds, [scan.id]: leveled } }))
      void saveScan(leveled)
      set({ review: startReview(estimate, friendlyRoomName(scan.name, estimate.objects)), busy: null })
      const found = estimate.objects.length
      useStore.getState().notify(
        estimate.room
          ? `Found the room and ${found} piece${found === 1 ? '' : 's'} of furniture. Check them on the right, then add them to the plan.`
          : `Found ${found} piece${found === 1 ? '' : 's'} of furniture but no clear walls.`,
      )
    } catch (e) {
      set({ busy: null })
      useStore.getState().notify(`Couldn't read that scan: ${(e as Error).message}`)
    }
  },

  importSpatialLM: (text) => {
    // Attach to the scan being reviewed, if any, so the underlay still lines up.
    const cur = get().review
    const est = parseSpatialLM(text, cur?.estimate.scanId ?? 'spatiallm')
    if (cur && cur.estimate.detector === 'heuristic') est.transform = { x: 0, y: 0, rotation: 0 }
    if (!est.room && !est.objects.length) {
      useStore.getState().notify(est.notes[0] ?? 'Nothing to import in that text.')
      return
    }
    set({ review: startReview(est, cur?.roomName ?? 'Scanned room') })
  },

  setObject: (id, patch) => set((s) => (s.review ? { review: { ...s.review, objects: s.review.objects.map((o) => (o.id === id ? { ...o, ...patch } : o)) } } : s)),
  setCategory: (id, category) => {
    const o = get().review?.objects.find((x) => x.id === id)
    if (!o) return
    // A relabel keeps the measured size; only the name (and later the clearances) change.
    get().setObject(id, { category, name: displayName(category, o.width, o.depth), accepted: true, confidence: 1 })
  },
  setAll: (accepted) => set((s) => (s.review ? { review: { ...s.review, objects: s.review.objects.map((o) => ({ ...o, accepted })) } } : s)),
  setReview: (patch) => set((s) => (s.review ? { review: { ...s.review, ...patch } } : s)),

  apply: () => {
    const r = get().review
    if (!r) return
    const main = useStore.getState()
    let roomId: string | undefined
    let count = 0
    main.update((d) => {
      const res = applyEstimate(d, r.estimate, reviewChoice(r))
      roomId = res.roomId
      count = res.placementIds.length
      return res.doc
    })
    set({ review: null })
    if (roomId) main.select({ kind: 'room', id: roomId })
    const what = [roomId ? 'the room' : '', count ? `${count} piece${count === 1 ? '' : 's'}` : ''].filter(Boolean).join(' and ') || 'your scan'
    main.notify(`Moved in ${what}! Drag anything to fix it.`)
  },
  cancel: () => set({ review: null }),

  ensureCloud: (id) => {
    if (get().clouds[id]) return
    const hit = cachedScan(id)
    if (hit) {
      set((s) => ({ clouds: { ...s.clouds, [id]: hit } }))
      return
    }
    void loadScan(id).then((scan) => scan && set((s) => ({ clouds: { ...s.clouds, [id]: scan } })))
  },
  setShowUnderlay: (showUnderlay) => set({ showUnderlay }),
}))
