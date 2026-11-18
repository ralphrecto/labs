import { create } from 'zustand'
import { activeLayout, useStore } from '../store'
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
      set({ review: startReview(estimate, scan.name || 'Scanned room'), busy: null })
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
    main.notify(`Added ${roomId ? 'the room and ' : ''}${count} piece${count === 1 ? '' : 's'} to "${activeLayout(useStore.getState().doc).name}". Drag anything to fix it; Ctrl+Z undoes the whole import.`)
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
