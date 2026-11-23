import { useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import { useAnalysis } from '../analysis'
import { catalogEntry } from '../model/catalog'
import {
  add,
  bounds,
  centroid,
  clearanceZones,
  closestPointOnSegment,
  convexOverlap,
  dist,
  doorSwing,
  dot,
  footprint,
  norm,
  offsetPolygon,
  openingEndpoints,
  pointInPolygon,
  polygonArea,
  scale,
  sub,
  v,
  wallInwardNormal,
  wallSegments,
} from '../model/geometry'
import { clampOnHost, HOSTED, hostAt } from '../model/placement'
import { isWallDecor, snapToWall } from '../model/wallDecor'
import { snapDrawPoint, snapPlacement, type SnapGuide } from '../model/snapping'
import type { Doc, Layout, Room, Vec2 } from '../model/types'
import { formatLength, parseLength } from '../model/units'
import { activeLayout, useStore } from '../store'
import { planToScan, rotateAbout, scanToPlan, snapAxes, snapCorner, straightenFromLine, svgTransform, underlayAt } from '../scan/align'
import { scanCenter, scanCorners, underlayOf, useScanRuntime } from '../scan/runtime'
import type { ScanRef } from '../model/types'
import { PlanScanLayer } from './ScanPanels'
import { viewFrame } from '../three/viewFrame'
import { placementShift } from '../scan/apply'
import { reviewChoice, useScanUI } from '../scan/scanUi'

const COLORS = {
  paper: '#fbfaf7',
  grid: '#ece9e2',
  gridMajor: '#dcd8ce',
  floor: '#f3efe6',
  wall: '#3b3a36',
  text: '#3b3a36',
  muted: '#8b877d',
  accent: '#2f6fde',
  error: '#d64545',
  warning: '#d68a1f',
  path: '#2f9e6b',
  opening: '#fbfaf7',
}

type Drag =
  | { type: 'item'; id: string; grab: Vec2; moved: boolean }
  | { type: 'rotate'; id: string }
  | { type: 'vertex'; roomId: string; index: number }
  | { type: 'room'; roomId: string; last: Vec2 }
  | { type: 'opening'; roomId: string; id: string; grab: number }
  | { type: 'pan'; startScreen: Vec2; startCenter: Vec2 }
  | { type: 'scan'; id: string; last: Vec2 }

interface ViewState {
  cx: number
  cy: number
  /** Pixels per meter. */
  k: number
}


/** Tiny badge glyphs for wall decor in the plan. */
const WALL_GLYPH: Record<string, string> = { 'wall-art': '▣', mirror: '◯', 'wall-shelf': '☰', 'wall-clock': '◷', curtains: '⫴' }
export interface PlanViewProps {
  /** Show a specific layout read-only (used by compare). Defaults to the active layout, editable. */
  layoutId?: string
  readOnly?: boolean
  svgRef?: React.RefObject<SVGSVGElement | null>
  title?: string
  /**
   * Screen space (CSS px) the game HUD covers at the top, bottom and right: the plan is fitted into
   * what's left, with a tighter margin. Leave out for the classic editor (whole pane).
   */
  inset?: { top: number; bottom: number; right?: number; ctx?: number }
  /**
   * Scan measure step: told whether the measured line (or the live line to the cursor) runs more
   * than MEASURE_SKEW_DEG off the scan's wall axes, i.e. probably isn't along a wall. The plan's own
   * hint says so too; a step (Onboarding's MeasureStep) can use this to show its own warning.
   */
  onMeasureSkew?: (skewed: boolean) => void
}

/** A measured line further than this (degrees) off the scan's wall axes is probably not a wall. */
export const MEASURE_SKEW_DEG = 5

/** How far (degrees, 0..45) the line a→b is off the plan axes, which the scan's walls are squared to. */
export function offAxisDeg(a: Vec2, b: Vec2): number {
  const ang = (Math.abs(Math.atan2(b.y - a.y, b.x - a.x)) * 180) / Math.PI
  const m = ang % 90
  return Math.min(m, 90 - m)
}

/** Whether a measured line is long enough to judge and more than MEASURE_SKEW_DEG off the wall axes. */
export const measureSkewed = (a: Vec2, b: Vec2 | null | undefined) => !!b && dist(a, b) > 0.3 && offAxisDeg(a, b) > MEASURE_SKEW_DEG

/**
 * Pick where a scan-measure tap lands, given the corner magnet's hit (`corner`), the wall snap's
 * hit along the plan axes (`wall`: x for a vertical wall, y for a horizontal one, both at a corner)
 * and the line's first end (`anchor`, for the second tap). A tap well inside a wall's span (more
 * than half the corner radius from the corner, along that wall) goes onto the wall, not the
 * corner; so does a second tap whose corner would make the line skewed while the wall wouldn't.
 */
export function pickMeasureSnap(o: { tap: Vec2; corner: Vec2 | null; wall: { x?: number; y?: number }; anchor?: Vec2 | null; cornerRadius: number }): Vec2 {
  const { tap, corner, wall, anchor, cornerRadius } = o
  const one = (wall.x == null) !== (wall.y == null)
  const proj = wall.x != null || wall.y != null ? v(wall.x ?? tap.x, wall.y ?? tap.y) : null
  if (!corner) return proj ?? tap
  if (!proj || !one) return corner
  const along = wall.x != null ? Math.abs(tap.y - corner.y) : Math.abs(tap.x - corner.x)
  if (along > 0.5 * cornerRadius) return proj
  if (anchor && dist(anchor, corner) > 0.3 && offAxisDeg(anchor, corner) > MEASURE_SKEW_DEG && offAxisDeg(anchor, proj) <= MEASURE_SKEW_DEG) return proj
  return corner
}

export function PlanView({ layoutId, readOnly, svgRef: externalRef, title, inset, onMeasureSkew }: PlanViewProps) {
  const doc = useStore((s) => s.doc)
  const mode = useStore((s) => (readOnly ? 'select' : s.mode))
  const selection = useStore((s) => s.selection)
  const snap = useStore((s) => s.snap)
  const showClearance = useStore((s) => s.showClearance)
  const showPath = useStore((s) => s.showPath)
  const layout = layoutId ? (doc.layouts.find((l) => l.id === layoutId) ?? activeLayout(doc)) : activeLayout(doc)
  const analysis = useAnalysis(doc, layout)
  const units = doc.apartment.units

  const wrapRef = useRef<HTMLDivElement>(null)
  const innerRef = useRef<SVGSVGElement>(null)
  const svgRef = externalRef ?? innerRef
  const [size, setSize] = useState({ w: 0, h: 0 })
  const [view, setView] = useState<ViewState | null>(null)
  const [drag, setDrag] = useState<Drag | null>(null)
  const [guides, setGuides] = useState<SnapGuide[]>([])
  const [cursor, setCursor] = useState<Vec2 | null>(null)
  const cursorRef = useRef<Vec2 | null>(null)
  const [draft, setDraft] = useState<Vec2[]>([])
  const [lengthInput, setLengthInput] = useState('')
  const [straightenStart, setStraightenStart] = useState<Vec2 | null>(null)
  useScanRuntime((s) => s.entries) // re-render when scans finish loading
  const scanDraft = useStore((s) => s.scanDraft)
  const activeScanId = useStore((s) => s.activeScanId)
  const scans = doc.scans ?? []
  const visibleScans = scans.filter((r) => r.visible)
  const activeScan = scans.find((r) => r.id === activeScanId)

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth || 1, h: el.clientHeight || 1 }))
    ro.observe(el)
    setSize({ w: el.clientWidth || 1, h: el.clientHeight || 1 })
    return () => ro.disconnect()
  }, [])

  const review = useScanUI((s) => (readOnly ? null : s.review))
  const fit = (focus?: Vec2[]) => {
    const pts = focus ?? [...doc.rooms.flatMap((r) => r.polygon), ...(doc.scans ?? []).filter((r) => r.visible).flatMap((r) => scanCorners(r) ?? [])]
    // Include a scan waiting for review, which lands beside the rooms.
    const rv = useScanUI.getState().review
    const shift = !focus && rv && !readOnly ? placementShift(doc, rv.estimate, reviewChoice(rv)) : null
    if (rv && shift) for (const p of rv.estimate.room?.polygon ?? rv.estimate.objects) pts.push(v(p.x + shift.x, p.y + shift.y))
    if (!pts.length) {
      setView({ cx: 3, cy: 2, k: Math.min(size.w / 8, size.h / 6) })
      return
    }
    const b = bounds(pts)
    const pad = inset ? 0.8 : 1.6
    const w = b.maxX - b.minX + pad
    const h = b.maxY - b.minY + pad
    const top = inset?.top ?? 0
    const right = inset?.right ?? 0
    const availW = Math.max(size.w * 0.5, size.w - right)
    const availH = Math.max(size.h * 0.25, size.h - top - (inset?.bottom ?? 0) - (inset?.ctx ?? 0))
    const k = Math.min(availW / w, availH / h)
    // Centre the plan in the part of the pane the HUD leaves visible, not the middle of the pane.
    const dx = (availW - size.w) / 2
    const dy = top + availH / 2 - size.h / 2
    setView({ cx: (b.minX + b.maxX) / 2 - dx / k, cy: (b.minY + b.maxY) / 2 - dy / k, k })
  }
  // Fit on first layout and when the container size first becomes known.
  const fitted = useRef(false)
  useEffect(() => {
    if (!fitted.current && size.w > 10) {
      fitted.current = true
      fit()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size])
  // Re-fit when the HUD around the game's plan settles into a new shape.
  const insetKey = inset ? `${Math.round(inset.top)}:${Math.round(inset.bottom)}:${Math.round(inset.right ?? 0)}:${inset.ctx ?? 0}` : ''
  useEffect(() => {
    if (fitted.current) fit()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [insetKey])
  // Re-fit when the whole document is replaced (new apartment, import).
  const aptId = doc.apartment.id
  useEffect(() => {
    if (fitted.current) fit()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aptId])
  // Bring a newly imported scan into view.
  const scanIds = (doc.scans ?? []).map((r) => r.id).join(',')
  const lastScanIds = useRef(scanIds)
  const scansReady = useScanRuntime((s) => (doc.scans ?? []).filter((r) => s.entries[r.id]?.status === 'ready').length)
  useEffect(() => {
    if (scanIds === lastScanIds.current) return
    const prev = new Set(lastScanIds.current.split(','))
    const added = (doc.scans ?? []).find((r) => !prev.has(r.id))
    const corners = added && scanCorners(added)
    lastScanIds.current = scanIds
    if (corners && fitted.current) fit(corners)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanIds, scansReady])

  // Re-fit when a scan review starts or ends, so the scan (or the room it added) is in view.
  const reviewId = review?.estimate
  useEffect(() => {
    if (fitted.current) fit()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reviewId])

  useEffect(() => {
    if (mode !== 'draw-room') {
      setDraft([])
      setLengthInput('')
    }
    if (mode !== 'scan-straighten') setStraightenStart(null)
  }, [mode])

  const vs = view ?? { cx: 3, cy: 2, k: 80 }
  const px = 1 / vs.k // one screen pixel in meters
  const sw = Math.max(1, size.w)
  const sh = Math.max(1, size.h)
  const vb = { x: vs.cx - sw / 2 / vs.k, y: vs.cy - sh / 2 / vs.k, w: sw / vs.k, h: sh / vs.k }

  const toWorld = (e: { clientX: number; clientY: number }): Vec2 => {
    const r = svgRef.current!.getBoundingClientRect()
    return v(vb.x + (e.clientX - r.left) / vs.k, vb.y + (e.clientY - r.top) / vs.k)
  }

  // The HUD maps drawer drops through the plan when it's showing (see viewFrame).
  const mapRef = useRef({ toWorld, vb, k: vs.k })
  mapRef.current = { toWorld, vb, k: vs.k }
  useEffect(() => {
    const toFloor = (x: number, y: number) => (svgRef.current ? mapRef.current.toWorld({ clientX: x, clientY: y }) : null)
    const toScreen = (p: Vec2) => {
      const svg = svgRef.current
      if (!svg) return null
      const r = svg.getBoundingClientRect()
      const m = mapRef.current
      return { x: r.left + (p.x - m.vb.x) * m.k, y: r.top + (p.y - m.vb.y) * m.k }
    }
    viewFrame.toFloor = toFloor
    viewFrame.toScreen = toScreen
    return () => {
      if (viewFrame.toFloor === toFloor) viewFrame.toFloor = null
      if (viewFrame.toScreen === toScreen) viewFrame.toScreen = null
    }
  }, [])

  const store = useStore.getState

  /**
   * Pull a plan point onto walls seen by a visible scan (within ~12 px): onto both walls at a corner,
   * onto the one wall along a wall. Walls are found along the plan axes, so square the scan up first.
   */
  const scanSnap = (p: Vec2, only?: ScanRef): Vec2 | null => scanMagnet(p, only)?.p ?? null

  /**
   * scanSnap, plus a corner magnet: within ~25 px, a corner of the scan's wall outline (or the end
   * of a wall line) wins over the plain wall snap, so "wall to wall" taps reach the corners.
   */
  const scanMagnet = (p: Vec2, only?: ScanRef, anchor?: Vec2 | null): { p: Vec2; corner: boolean } | null => {
    const refs = (only ? [only] : visibleScans).flatMap((ref) => {
      const img = underlayOf(ref, 'walls')
      return img ? [{ ref, u: img.underlay }] : []
    })
    if (!refs.length) return null
    const strength = (q: Vec2) => {
      let m = 0
      for (const { ref, u } of refs) m = Math.max(m, underlayAt(u, planToScan(ref, q)))
      return m
    }
    const step = Math.max(0.005, Math.min(...refs.map(({ ref, u }) => u.pixel * ref.scale)))
    const cornerRadius = Math.min(0.6, Math.max(0.08, 25 * px))
    const corner = snapCorner(strength, p, cornerRadius, step)
    const hit = snapAxes(strength, p, Math.min(0.25, Math.max(0.06, 10 * px)), step)
    if (!corner && hit.x == null && hit.y == null) return null
    const q = pickMeasureSnap({ tap: p, corner, wall: hit, anchor, cornerRadius })
    return { p: q, corner: !!corner && q === corner }
  }
  // The scan measure line (second end live at the cursor) and whether it runs off the wall axes.
  const measureLive = mode === 'scan-measure' && scanDraft && !scanDraft.b && cursor && activeScan ? scanMagnet(cursor, activeScan, scanDraft.a) : null
  const measureEnd = scanDraft ? (scanDraft.b ?? (mode === 'scan-measure' && cursor ? (measureLive?.p ?? cursor) : null)) : null
  const measureSkew = mode === 'scan-measure' && !!scanDraft && measureSkewed(scanDraft.a, measureEnd)
  useEffect(() => {
    onMeasureSkew?.(measureSkew)
  }, [measureSkew, onMeasureSkew])

  /** A length label sits just above its line, or below it near the top edge (under the hint pill). */
  const labelY = (y: number) => (y - vb.y < 64 * px ? y + 18 * px : y - 7 * px)
  const cornerRing = (q: Vec2) => <circle cx={q.x} cy={q.y} r={9 * px} fill="none" stroke="#7a3fc2" strokeWidth={1.5 * px} opacity={0.85} pointerEvents="none" className="corner-ring" />

  /** Corner placement while tracing: walls from the scan first, then keep walls square to the previous corner. */
  const traceSnap = (last: Vec2 | null, w: Vec2, free: boolean): Vec2 => {
    const hit = !free && visibleScans.length ? scanSnap(w) : null
    if (!hit) return snapDrawPoint(last, w, snap.grid, !free)
    if (!last) return hit
    const d = sub(hit, last)
    // Nearly horizontal or vertical: line it up exactly with the previous corner.
    if (Math.abs(d.y) < Math.abs(d.x) * Math.tan((3 * Math.PI) / 180)) return v(hit.x, last.y)
    if (Math.abs(d.x) < Math.abs(d.y) * Math.tan((3 * Math.PI) / 180)) return v(last.x, hit.y)
    return hit
  }

  const placementsWithItems = useMemo(() => {
    const byId = new Map(doc.items.map((i) => [i.id, i]))
    return layout.placements.flatMap((p) => {
      const item = byId.get(p.itemId)
      return item ? [{ p, item }] : []
    })
  }, [doc.items, layout])

  // ---------------------------------------------------------------- pointer handling
  const onWheel = (e: React.WheelEvent) => {
    const w = toWorld(e)
    const factor = Math.exp(-e.deltaY * 0.0015)
    const k = Math.min(600, Math.max(10, vs.k * factor))
    // Keep the point under the cursor fixed.
    setView({ k, cx: w.x - (w.x - vs.cx) * (vs.k / k), cy: w.y - (w.y - vs.cy) * (vs.k / k) })
  }

  const startPan = (e: RPointerEvent) => {
    setDrag({ type: 'pan', startScreen: v(e.clientX, e.clientY), startCenter: v(vs.cx, vs.cy) })
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }

  const onBackgroundDown = (e: RPointerEvent) => {
    if (e.button === 1 || e.button === 2 || readOnly) return startPan(e)
    const w = toWorld(e)
    if (mode === 'draw-room') {
      if (draft.length >= 3 && dist(w, draft[0]) < 12 * px) return finishDraft()
      addDraftPoint(traceSnap(draft[draft.length - 1] ?? null, w, e.shiftKey || e.altKey))
      return
    }
    if (mode === 'scan-measure' && activeScan) {
      const second = !!scanDraft && !scanDraft.b && scanDraft.scanId === activeScan.id
      const p = (!e.altKey && scanMagnet(w, activeScan, second ? scanDraft!.a : null)?.p) || w
      const s = store()
      if (!scanDraft || scanDraft.b || scanDraft.scanId !== activeScan.id) s.setUI({ scanDraft: { scanId: activeScan.id, a: p } })
      else s.setUI({ scanDraft: { ...scanDraft, b: p } })
      return
    }
    if (mode === 'scan-straighten' && activeScan) {
      if (!straightenStart) return setStraightenStart(w)
      if (dist(straightenStart, w) < 0.1) return
      const mid = scale(add(straightenStart, w), 0.5)
      store().updateScan(activeScan.id, rotateAbout(activeScan, straightenFromLine(straightenStart, w), mid))
      setStraightenStart(null)
      store().setMode('select')
      store().notify('Scan straightened. Ctrl+Z to undo.')
      return
    }
    if (mode === 'scan-move' && activeScan) {
      begin({ type: 'scan', id: activeScan.id, last: w }, e)
      return
    }
    if (mode === 'door' || mode === 'window') {
      placeOpeningAt(w)
      return
    }
    store().select(null)
    startPan(e)
  }

  const onMove = (e: RPointerEvent) => {
    const w = toWorld(e)
    setCursor(w)
    cursorRef.current = w
    if (!drag) return
    const s = store()
    if (drag.type === 'pan') {
      setView({ ...vs, cx: drag.startCenter.x - (e.clientX - drag.startScreen.x) / vs.k, cy: drag.startCenter.y - (e.clientY - drag.startScreen.y) / vs.k })
    } else if (drag.type === 'item') {
      const pi = placementsWithItems.find((x) => x.p.id === drag.id)
      if (!pi) return
      const proposed = { ...pi.p, x: w.x - drag.grab.x, y: w.y - drag.grab.y }
      if (isWallDecor(pi.item.category)) {
        // Wall decor slides along the walls, snapping to the nearest wall face it fits on.
        if (!drag.moved) {
          s.beginGesture()
          setDrag({ ...drag, moved: true })
        }
        const spot = snapToWall(s.doc, activeLayout(s.doc), pi.item, proposed, drag.id)
        if (spot) s.updatePlacement(drag.id, { x: spot.x, y: spot.y, rotation: spot.rotation })
        return
      }
      if (pi.p.on && HOSTED.has(pi.item.category)) {
        // A table lamp slides over tops and onto another one under the pointer (the side table to
        // the desk); off every top it stays clamped to the one it's on.
        if (!drag.moved) {
          s.beginGesture()
          setDrag({ ...drag, moved: true })
        }
        const layoutNow = activeLayout(s.doc)
        const host =
          hostAt(s.doc, layoutNow, proposed, pi.item, pi.p.on) ??
          layoutNow.placements.flatMap((h) => {
            const it = h.id === pi.p.on ? doc.items.find((i) => i.id === h.itemId) : undefined
            return it ? [{ id: h.id, item: it, p: h }] : []
          })[0]
        if (!host) return
        const c = clampOnHost(host, pi.item, proposed)
        s.updatePlacement(drag.id, { x: c.x, y: c.y, rotation: host.p.rotation, on: host.id })
        return
      }
      const others = placementsWithItems.filter((x) => x.p.id !== drag.id && !isWallDecor(x.item.category))
      const res = e.altKey ? { placement: proposed, guides: [] } : snapPlacement(pi.item, proposed, doc.rooms, others, snap)
      if (!drag.moved) {
        s.beginGesture()
        setDrag({ ...drag, moved: true })
      }
      setGuides(res.guides)
      s.updatePlacement(drag.id, { x: res.placement.x, y: res.placement.y, rotation: res.placement.rotation })
    } else if (drag.type === 'rotate') {
      const pi = placementsWithItems.find((x) => x.p.id === drag.id)
      if (!pi) return
      const d = sub(w, v(pi.p.x, pi.p.y))
      // The handle sits in front of the item: front direction = rot((0,1), θ).
      let theta = Math.atan2(-d.x, d.y)
      if (!e.shiftKey) theta = Math.round(theta / (Math.PI / 12)) * (Math.PI / 12)
      s.updatePlacement(drag.id, { rotation: (theta + 2 * Math.PI) % (2 * Math.PI) })
    } else if (drag.type === 'vertex') {
      const room = doc.rooms.find((r) => r.id === drag.roomId)
      if (!room) return
      let p = snap.grid > 0 ? v(Math.round(w.x / snap.grid) * snap.grid, Math.round(w.y / snap.grid) * snap.grid) : w
      // Align with neighbouring vertices so walls stay straight.
      const n = room.polygon.length
      const nb = [room.polygon[(drag.index - 1 + n) % n], room.polygon[(drag.index + 1) % n]]
      const g: SnapGuide[] = []
      for (const q of nb) {
        if (Math.abs(p.x - q.x) < 8 * px) {
          p = { ...p, x: q.x }
          g.push({ a: q, b: p })
        }
        if (Math.abs(p.y - q.y) < 8 * px) {
          p = { ...p, y: q.y }
          g.push({ a: q, b: p })
        }
      }
      setGuides(g)
      s.setVertex(drag.roomId, drag.index, p)
    } else if (drag.type === 'room') {
      let delta = sub(w, drag.last)
      if (snap.grid > 0) {
        delta = v(Math.round(delta.x / snap.grid) * snap.grid, Math.round(delta.y / snap.grid) * snap.grid)
        if (delta.x === 0 && delta.y === 0) return
      }
      s.moveRoom(drag.roomId, delta)
      setDrag({ ...drag, last: add(drag.last, delta) })
    } else if (drag.type === 'scan') {
      const ref = (doc.scans ?? []).find((r) => r.id === drag.id)
      if (!ref) return
      const delta = sub(w, drag.last)
      s.updateScan(ref.id, { x: ref.x + delta.x, y: ref.y + delta.y })
      setDrag({ ...drag, last: w })
    } else if (drag.type === 'opening') {
      const room = doc.rooms.find((r) => r.id === drag.roomId)
      const o = room?.openings.find((x) => x.id === drag.id)
      if (!room || !o) return
      const [a, b] = wallSegments(room)[o.wall]
      const t = dot(sub(w, a), norm(sub(b, a)))
      const wallLen = dist(a, b)
      let offset = t - drag.grab
      if (snap.grid > 0) offset = Math.round(offset / snap.grid) * snap.grid
      offset = Math.max(0, Math.min(wallLen - o.width, offset))
      s.updateOpening(room.id, o.id, { offset })
    }
  }

  const onUp = () => {
    if (drag && drag.type !== 'pan') store().endGesture()
    setDrag(null)
    setGuides([])
  }

  const begin = (d: Drag, e: RPointerEvent) => {
    e.stopPropagation()
    if (e.button !== 0) return startPan(e)
    ;(svgRef.current as Element).setPointerCapture?.(e.pointerId)
    if (d.type !== 'item' && d.type !== 'pan') store().beginGesture()
    setDrag(d)
  }

  // ---------------------------------------------------------------- draw room
  const addDraftPoint = (p: Vec2) => {
    if (draft.length >= 3 && dist(p, draft[0]) < 10 * px) return finishDraft()
    setDraft([...draft, p])
    setLengthInput('')
  }
  const finishDraft = () => {
    if (draft.length >= 3) {
      const id = store().addRoom(`Room ${doc.rooms.length + 1}`, draft)
      // Traced over a scan that saw the ceiling: use its height.
      const c = centroid(draft)
      const under = visibleScans.find((r) => r.ceiling != null && insideScan(r, c))
      if (under?.ceiling != null) store().updateRoom(id, { ceilingHeight: Math.round(under.scale * (under.ceiling - under.floor) * 100) / 100 })
      store().setMode('select')
    }
    setDraft([])
  }
  const draftPreview = draft.length && cursor ? traceSnap(draft[draft.length - 1], cursor, false) : null
  const applyTypedLength = () => {
    const l = parseLength(lengthInput, units)
    if (l == null || l <= 0 || !draft.length) return
    const last = draft[draft.length - 1]
    const c = cursorRef.current
    const preview = c ? snapDrawPoint(last, c, 0, true) : null
    const dir = preview && dist(preview, last) > 1e-6 ? norm(sub(preview, last)) : v(1, 0)
    addDraftPoint(add(last, scale(dir, l)))
  }

  useEffect(() => {
    if (mode !== 'draw-room') return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setDraft([])
        store().setMode('select')
      } else if (e.key === 'Enter' && !lengthInput) finishDraft()
      else if (e.key === 'Backspace' && !lengthInput && draft.length) setDraft(draft.slice(0, -1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // ---------------------------------------------------------------- openings
  const nearestWall = (w: Vec2): { room: Room; wall: number; t: number; distance: number } | null => {
    let best: { room: Room; wall: number; t: number; distance: number } | null = null
    for (const room of doc.rooms)
      wallSegments(room).forEach(([a, b], i) => {
        const c = closestPointOnSegment(w, a, b)
        if (c.distance < 0.4 && (!best || c.distance < best.distance)) best = { room, wall: i, t: c.t * dist(a, b), distance: c.distance }
      })
    return best
  }
  const openingPreview = (mode === 'door' || mode === 'window') && cursor ? nearestWall(cursor) : null
  const placeOpeningAt = (w: Vec2) => {
    const hit = nearestWall(w)
    if (!hit) return
    const [a, b] = wallSegments(hit.room)[hit.wall]
    const width = mode === 'door' ? 0.8 : 1.2
    const offset = Math.max(0, Math.min(dist(a, b) - width, hit.t - width / 2))
    if (mode === 'door') store().addOpening(hit.room.id, { kind: 'door', wall: hit.wall, offset, width, height: 2.1, sill: 0, hinge: 'start', swing: 'in' })
    else store().addOpening(hit.room.id, { kind: 'window', wall: hit.wall, offset, width, height: 1.4, sill: 0.8 })
    store().setMode('select')
  }

  // ---------------------------------------------------------------- render helpers
  const fs = 12 * px
  const selectedRoom = selection?.kind === 'room' ? doc.rooms.find((r) => r.id === selection.id) : undefined
  const pathColor = (w: number) => (w < 0.6 ? COLORS.error : w < 0.9 ? COLORS.warning : COLORS.path)
  const issueZones = new Set(analysis.issues.filter((i) => i.kind === 'clearance' && i.area).map((i) => JSON.stringify(i.area)))

  const gridLines = useMemo(() => {
    const lines: { x1: number; y1: number; x2: number; y2: number; major: boolean }[] = []
    const step = vs.k > 40 ? 0.5 : 1
    const x0 = Math.floor(vb.x / step) * step
    const y0 = Math.floor(vb.y / step) * step
    for (let x = x0; x < vb.x + vb.w; x += step) lines.push({ x1: x, y1: vb.y, x2: x, y2: vb.y + vb.h, major: Math.abs(x - Math.round(x)) < 1e-6 })
    for (let y = y0; y < vb.y + vb.h; y += step) lines.push({ x1: vb.x, y1: y, x2: vb.x + vb.w, y2: y, major: Math.abs(y - Math.round(y)) < 1e-6 })
    return lines
  }, [vb.x, vb.y, vb.w, vb.h, vs.k])

  // Scale bar: pick a round length about 100 px long.
  const scaleBar = useMemo(() => {
    const target = 100 * px
    const candidates = units === 'metric' ? [0.1, 0.25, 0.5, 1, 2, 5] : [0.3048 / 2, 0.3048, 0.6096, 1.524, 3.048]
    const l = candidates.reduce((best, c) => (Math.abs(c - target) < Math.abs(best - target) ? c : best), candidates[0])
    return l
  }, [px, units])

  const pieceBoxes = placementsWithItems.map(({ p, item }) => bounds(footprint(item, p)))
  const dimsDone: { l: number; mid: Vec2 }[] = []
  // Dimension labels already drawn (world boxes): a wall's length never lands on top of another's.
  const labelBoxes: Box[] = []
  const boxesHit = (b: Box) => labelBoxes.some((o) => o.maxX > b.minX && o.minX < b.maxX && o.maxY > b.minY && o.minY < b.maxY)
  // The selected piece's width and depth, drawn beside it. Wall lengths near them step aside.
  const selDims = (() => {
    if (selection?.kind !== 'placement') return null
    const pi = placementsWithItems.find((x) => x.p.id === selection.id)
    if (!pi) return null
    const [c0, c1, , c3] = footprint(pi.item, pi.p)
    const wMid = scale(add(c0, c1), 0.5)
    const dMid = scale(add(c0, c3), 0.5)
    const t1 = add(wMid, scale(norm(sub(wMid, v(pi.p.x, pi.p.y))), 12 * px))
    const t2 = add(dMid, scale(norm(sub(dMid, v(pi.p.x, pi.p.y))), 12 * px))
    return { t1, t2, w: formatLength(pi.item.width, units), d: formatLength(pi.item.depth, units) }
  })()
  if (selDims)
    for (const [t, text] of [
      [selDims.t1, selDims.w],
      [selDims.t2, selDims.d],
    ] as const) {
      const r = (text.length * fs * 0.9 * 0.58) / 2 + 20 * px
      labelBoxes.push({ minX: t.x - r, maxX: t.x + r, minY: t.y - fs * 0.5 - 20 * px, maxY: t.y + fs * 0.5 + 20 * px })
    }
  // Labels a piece doesn't need: a lamp on a table, a rug under the sofa or the table, a pendant over the table.
  const solidBodies = placementsWithItems
    .filter(({ item, p }) => !item.floorLevel && !p.on && item.category !== 'pendant' && !isWallDecor(item.category))
    .map(({ item, p }) => ({ id: p.id, body: footprint(item, p) }))
  const unlabelled = new Set(
    placementsWithItems
      .filter(({ p, item }) => !!p.on || isWallDecor(item.category) || ((item.floorLevel || item.category === 'pendant') && solidBodies.some((o) => convexOverlap(footprint(item, p), o.body, 0.01))))
      .map(({ p }) => p.id),
  )
  return (
    <div className="plan-wrap" ref={wrapRef} onContextMenu={(e) => e.preventDefault()}>
      {title && <div className="plan-title">{title}</div>}
      <svg
        ref={svgRef}
        className={`plan mode-${mode}`}
        width={Math.max(1, size.w)}
        height={Math.max(1, size.h)}
        viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
        onWheel={onWheel}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={() => setCursor(null)}
        onDoubleClick={() => mode === 'draw-room' && finishDraft()}
        xmlns="http://www.w3.org/2000/svg"
        style={{ background: COLORS.paper, cursor: drag?.type === 'item' ? 'grabbing' : undefined }}
      >
        <rect x={vb.x} y={vb.y} width={vb.w} height={vb.h} fill={COLORS.paper} onPointerDown={onBackgroundDown} />
        <g pointerEvents="none">
          {gridLines.map(({ major, ...l }, i) => (
            <line key={i} {...l} stroke={major ? COLORS.gridMajor : COLORS.grid} strokeWidth={px} />
          ))}
        </g>

        {/* Scan underlays */}
        {visibleScans.map((ref) => {
          if (ref.underlay === 'off') return null
          const img = underlayOf(ref)
          if (!img) return null
          const u = img.underlay
          return (
            <g key={ref.id} transform={svgTransform(ref)} pointerEvents="none" opacity={ref.opacity}>
              <image href={img.href} x={u.u0} y={u.v0} width={u.width * u.pixel} height={u.height * u.pixel} preserveAspectRatio="none" style={{ imageRendering: ref.underlay === 'walls' ? 'pixelated' : 'auto' }} />
            </g>
          )
        })}

        {/* Rooms: floor, walls, labels */}
        {doc.rooms.map((room) => {
          const selected = selectedRoom?.id === room.id
          const outer = offsetPolygon(room.polygon, room.wallThickness)
          const d = `M${outer.map((p) => `${p.x},${p.y}`).join('L')}Z M${room.polygon.map((p) => `${p.x},${p.y}`).join('L')}Z`
          const areaText = units === 'metric' ? `${polygonArea(room.polygon).toFixed(1)} m²` : `${Math.round(polygonArea(room.polygon) * 10.7639)} ft²`
          // The name and area sit in a free spot of the room, or not at all: never under a piece.
          const c = roomLabelSpot(room.polygon, centroid(room.polygon), fs, Math.max(room.name.length, areaText.length) * fs * 0.58, pieceBoxes)
          return (
            <g key={room.id}>
              <polygon
                points={room.polygon.map((p) => `${p.x},${p.y}`).join(' ')}
                fill={COLORS.floor}
                fillOpacity={visibleScans.some((r) => r.underlay !== 'off') ? 0.35 : 1}
                onPointerDown={(e) => {
                  if (readOnly) return startPan(e)
                  if (mode !== 'select') return onBackgroundDown(e)
                  store().select({ kind: 'room', id: room.id })
                  startPan(e)
                }}
              />
              <path
                d={d}
                fillRule="evenodd"
                fill={selected ? COLORS.accent : COLORS.wall}
                style={{ cursor: readOnly ? undefined : 'move' }}
                onPointerDown={(e) => {
                  if (readOnly) return startPan(e)
                  if (mode === 'door' || mode === 'window') return onBackgroundDown(e)
                  if (mode !== 'select') return onBackgroundDown(e)
                  store().select({ kind: 'room', id: room.id })
                  begin({ type: 'room', roomId: room.id, last: toWorld(e) }, e)
                }}
              />
              {c && (
                <g pointerEvents="none" fill={COLORS.muted} fontSize={fs} textAnchor="middle" fontFamily="system-ui, sans-serif">
                  <text x={c.x} y={c.y - fs * 0.2} fontWeight={600} fill={COLORS.text} opacity={0.55}>
                    {room.name}
                  </text>
                  <text x={c.x} y={c.y + fs * 1.1} opacity={0.55}>
                    {areaText}
                  </text>
                </g>
              )}
            </g>
          )
        })}

        {/* Scan point clouds, and detections waiting for review */}
        {!readOnly && <PlanScanLayer px={px} fs={fs} />}

        {/* Openings drawn over walls */}
        {doc.rooms.map((room) =>
          room.openings.map((o) => {
            const [a, b] = openingEndpoints(room, o)
            const n = wallInwardNormal(room, o.wall)
            const t = room.wallThickness
            const poly = [add(a, scale(n, 0.005)), add(b, scale(n, 0.005)), add(b, scale(n, -t - 0.005)), add(a, scale(n, -t - 0.005))]
            const selected = selection?.kind === 'opening' && selection.id === o.id
            const swing = doorSwing(room, o)
            const mid1 = add(a, scale(n, -t / 2))
            const mid2 = add(b, scale(n, -t / 2))
            return (
              <g key={o.id}>
                <polygon
                  points={poly.map((p) => `${p.x},${p.y}`).join(' ')}
                  fill={COLORS.opening}
                  stroke={selected ? COLORS.accent : COLORS.wall}
                  strokeWidth={selected ? 2.5 * px : px}
                  style={{ cursor: readOnly ? undefined : 'ew-resize' }}
                  onPointerDown={(e) => {
                    if (readOnly) return startPan(e)
                    store().select({ kind: 'opening', roomId: room.id, id: o.id })
                    const [wa, wb] = wallSegments(room)[o.wall]
                    const tt = dot(sub(toWorld(e), wa), norm(sub(wb, wa)))
                    begin({ type: 'opening', roomId: room.id, id: o.id, grab: tt - o.offset }, e)
                  }}
                />
                {o.kind === 'window' && (
                  <g pointerEvents="none" stroke={selected ? COLORS.accent : COLORS.wall} strokeWidth={px}>
                    <line x1={mid1.x} y1={mid1.y} x2={mid2.x} y2={mid2.y} />
                    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
                  </g>
                )}
                {swing && (
                  <g pointerEvents="none" fill="none" stroke={selected ? COLORS.accent : COLORS.wall} strokeWidth={1.2 * px}>
                    <line x1={swing[0].x} y1={swing[0].y} x2={swing[swing.length - 1].x} y2={swing[swing.length - 1].y} strokeWidth={2.5 * px} />
                    <polyline points={swing.slice(1).map((p) => `${p.x},${p.y}`).join(' ')} strokeDasharray={`${4 * px} ${3 * px}`} />
                    {o.entrance && (
                      <text x={(a.x + b.x) / 2 - n.x * 0.45} y={(a.y + b.y) / 2 - n.y * 0.45 + fs * 0.35} fontSize={fs * 0.9} textAnchor="middle" stroke="none" fill={COLORS.muted} fontFamily="system-ui, sans-serif">
                        Entrance
                      </text>
                    )}
                  </g>
                )}
              </g>
            )
          }),
        )}

        {/* Wall dimensions */}
        <g pointerEvents="none" fontSize={fs * 0.92} fontFamily="system-ui, sans-serif" textAnchor="middle" fill={COLORS.text}>
          {doc.rooms.map((room) =>
            wallSegments(room).map(([a, b], i) => {
              const l = dist(a, b)
              if (l * vs.k < 40) return null
              const mid = scale(add(a, b), 0.5)
              // A wall two rooms share (the same length, just across the wall) is measured once.
              if (dimsDone.some((o) => Math.abs(o.l - l) < 0.02 && dist(o.mid, mid) < Math.max(0.3, room.wallThickness * 3))) return null
              dimsDone.push({ l, mid })
              // Outside the wall line; an inside wall takes whichever side is clear of furniture, or none.
              const n = wallInwardNormal(room, i)
              const u = norm(sub(b, a))
              const half = v(Math.abs(u.x) * fs * 1.6 + Math.abs(u.y) * fs * 0.6, Math.abs(u.y) * fs * 1.6 + Math.abs(u.x) * fs * 0.6)
              const clear = (q: Vec2) => pieceBoxes.every((o) => o.maxX < q.x - half.x || o.minX > q.x + half.x || o.maxY < q.y - half.y || o.minY > q.y + half.y)
              const text = formatLength(l, units)
              const along = (text.length * fs * 0.92 * 0.58) / 2 + 4 * px
              const across = fs * 0.6
              const hx = Math.abs(u.x) * along + Math.abs(u.y) * across
              const hy = Math.abs(u.y) * along + Math.abs(u.x) * across
              const boxAt = (q: Vec2): Box => ({ minX: q.x - hx, maxX: q.x + hx, minY: q.y - hy, maxY: q.y + hy })
              const m = [add(mid, scale(n, -(room.wallThickness + 10 * px))), add(mid, scale(n, 14 * px))].find((q) => clear(q) && !boxesHit(boxAt(q)))
              if (!m) return null
              labelBoxes.push(boxAt(m))
              let ang = (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI
              if (ang > 90) ang -= 180
              if (ang < -90) ang += 180
              return (
                <text key={`${room.id}:${i}`} x={m.x} y={m.y} transform={`rotate(${ang} ${m.x} ${m.y})`} dominantBaseline="middle" opacity={selectedRoom?.id === room.id ? 1 : 0.7}>
                  {text}
                </text>
              )
            }),
          )}
        </g>

        {/* Clearance zones */}
        {showClearance &&
          placementsWithItems.map(({ p, item }) =>
            clearanceZones(item, p).map((z) => {
              const blocked = issueZones.has(JSON.stringify(z.poly))
              return (
                <polygon
                  key={`${p.id}:${z.side}`}
                  points={z.poly.map((q) => `${q.x},${q.y}`).join(' ')}
                  fill={blocked ? COLORS.error : COLORS.accent}
                  fillOpacity={blocked ? 0.16 : 0.06}
                  stroke={blocked ? COLORS.error : COLORS.accent}
                  strokeOpacity={0.5}
                  strokeWidth={px}
                  strokeDasharray={`${4 * px} ${3 * px}`}
                  pointerEvents="none"
                />
              )
            }),
          )}

        {/* Items, floor-level ones (rugs) first */}
        {[...placementsWithItems]
          .sort((a, b) => Number(!!b.item.floorLevel) - Number(!!a.item.floorLevel) || Number(!!a.p.on) - Number(!!b.p.on))
          .map(({ p, item }) => {
            const selected = selection?.kind === 'placement' && selection.id === p.id
            const flag = analysis.flagged.get(p.id)
            const stroke = selected ? COLORS.accent : flag === 'error' ? COLORS.error : flag === 'warning' ? COLORS.warning : '#4a4740'
            const hw = item.width / 2
            const hd = item.depth / 2
            const deg = (p.rotation * 180) / Math.PI
            // A pendant hangs over the table: a small disc, so the table's name stays readable under it.
            const pendant = item.category === 'pendant'
            const sk = pendant ? 0.6 : 1
            if (isWallDecor(item.category)) {
              // Wall decor: a thin strip on the wall face with a small badge in front of it.
              const t = Math.max(item.category === 'curtains' ? 0.06 : 0.035, 4 * px)
              const grab = (e: RPointerEvent) => {
                if (readOnly || mode !== 'select') return onBackgroundDown(e)
                store().select({ kind: 'placement', id: p.id })
                begin({ type: 'item', id: p.id, grab: sub(toWorld(e), v(p.x, p.y)), moved: false }, e)
              }
              const r = 7 * px
              return (
                <g key={p.id} transform={`translate(${p.x} ${p.y}) rotate(${deg})`} style={{ cursor: readOnly ? undefined : drag?.type === 'item' ? 'grabbing' : 'grab' }} onPointerDown={grab}>
                  <rect x={-hw} y={-hd - 4 * px} width={item.width} height={t + 4 * px + 2 * r + 4 * px} fill="transparent" />
                  <rect x={-hw} y={-hd} width={item.width} height={t} rx={t / 2} fill={item.color} stroke={stroke} strokeWidth={(selected ? 2.5 : 1) * px} />
                  <circle cx={0} cy={-hd + t + r + 2 * px} r={r} fill="#fffaf1" stroke={stroke} strokeWidth={(selected ? 2 : 1) * px} />
                  <text x={0} y={-hd + t + r + 2 * px} fontSize={9 * px} textAnchor="middle" dominantBaseline="central" fill="#4a4740" pointerEvents="none">
                    {WALL_GLYPH[item.category] ?? '▣'}
                  </text>
                </g>
              )
            }
            return (
              <g key={p.id} transform={`translate(${p.x} ${p.y}) rotate(${deg})`}>
                <rect
                  x={-hw * sk}
                  y={-hd * sk}
                  width={item.width * sk}
                  height={item.depth * sk}
                  rx={item.category === 'rug' ? 0 : pendant ? hw * sk : Math.min(0.03, hw / 4)}
                  fill={item.color}
                  fillOpacity={item.floorLevel ? 0.45 : pendant ? 0.6 : 0.85}
                  stroke={stroke}
                  strokeWidth={(selected || flag ? 2.5 : 1) * px}
                  strokeDasharray={item.floorLevel ? `${5 * px} ${3 * px}` : undefined}
                  style={{ cursor: readOnly ? undefined : drag?.type === 'item' ? 'grabbing' : 'grab' }}
                  onPointerDown={(e) => {
                    if (readOnly || mode !== 'select') return onBackgroundDown(e)
                    store().select({ kind: 'placement', id: p.id })
                    const w = toWorld(e)
                    begin({ type: 'item', id: p.id, grab: sub(w, v(p.x, p.y)), moved: false }, e)
                  }}
                />
                {!item.floorLevel && !pendant && <line x1={-hw + 0.04} y1={hd - 0.04} x2={hw - 0.04} y2={hd - 0.04} stroke="#ffffff" strokeOpacity={0.8} strokeWidth={2 * px} pointerEvents="none" />}
                <ItemGlyph category={item.category} hw={hw} hd={hd} px={px} />
                {selected && !readOnly && (
                  <g>
                    <line x1={0} y1={hd} x2={0} y2={hd + 22 * px} stroke={COLORS.accent} strokeWidth={1.5 * px} pointerEvents="none" />
                    <circle
                      cx={0}
                      cy={hd + 26 * px}
                      r={6 * px}
                      fill="#ffffff"
                      stroke={COLORS.accent}
                      strokeWidth={2 * px}
                      style={{ cursor: 'crosshair' }}
                      onPointerDown={(e) => begin({ type: 'rotate', id: p.id }, e)}
                    />
                  </g>
                )}
              </g>
            )
          })}

        {/* Item names, over every piece (a pendant over the table doesn't hide the table's name) */}
        {placementsWithItems.map(({ p, item }) => {
          if (unlabelled.has(p.id)) return null
          const labelSize = Math.min(fs, (item.width / Math.max(4, item.name.length)) * 1.6)
          // (Rotations can be negative or past a full turn: read the turn as 0 to 2π.)
          const r = ((p.rotation % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
          return (
            <g key={`label:${p.id}`} transform={`translate(${p.x} ${p.y}) rotate(${(p.rotation * 180) / Math.PI})`}>
            <text
              pointerEvents="none"
              x={0}
              y={0}
              fontSize={labelSize}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="#ffffff"
              stroke="#000000"
              strokeOpacity={0.25}
              strokeWidth={labelSize * 0.12}
              paintOrder="stroke"
              fontFamily="system-ui, sans-serif"
              fontWeight={600}
              transform={Math.abs(Math.sin(r)) > 0.7 && item.depth > item.width ? `rotate(${r > Math.PI ? 90 : -90})` : Math.cos(r) < -0.7 ? 'rotate(180)' : undefined}
            >
              {item.name}
            </text>
            </g>
          )
        })}

        {/* Selected item dimensions */}
        {selDims && (
          <g pointerEvents="none" fontSize={fs * 0.9} fill={COLORS.accent} fontFamily="system-ui, sans-serif" textAnchor="middle" dominantBaseline="middle" fontWeight={600}>
            <text x={selDims.t1.x} y={selDims.t1.y}>{selDims.w}</text>
            <text x={selDims.t2.x} y={selDims.t2.y}>{selDims.d}</text>
          </g>
        )}

        {/* Walking paths */}
        {showPath &&
          analysis.paths.results.map((r) =>
            r.path.length > 1 ? (
              <g key={r.roomId} pointerEvents="none">
                <polyline points={r.path.map((p) => `${p.x},${p.y}`).join(' ')} fill="none" stroke={pathColor(r.narrowest)} strokeWidth={2 * px} strokeOpacity={0.7} strokeDasharray={`${1 * px} ${5 * px}`} strokeLinecap="round" />
                {r.pinch && r.narrowest < 1.2 && (
                  <g>
                    <circle cx={r.pinch.x} cy={r.pinch.y} r={4 * px} fill={pathColor(r.narrowest)} />
                    <text x={r.pinch.x + 7 * px} y={r.pinch.y - 6 * px} fontSize={fs * 0.85} fill={pathColor(r.narrowest)} fontFamily="system-ui, sans-serif" fontWeight={600}>
                      {formatLength(r.narrowest, units)}
                    </text>
                  </g>
                )}
              </g>
            ) : null,
          )}

        {/* Vertex handles of the selected room */}
        {selectedRoom && !readOnly && mode === 'select' && (
          <g>
            {wallSegments(selectedRoom).map(([a, b], i) => {
              const m = scale(add(a, b), 0.5)
              return (
                <circle
                  key={`m${i}`}
                  cx={m.x}
                  cy={m.y}
                  r={4 * px}
                  fill="#ffffff"
                  stroke={COLORS.accent}
                  strokeWidth={px}
                  style={{ cursor: 'copy' }}
                  onPointerDown={(e) => {
                    store().insertVertex(selectedRoom.id, i, m)
                    begin({ type: 'vertex', roomId: selectedRoom.id, index: i + 1 }, e)
                  }}
                >
                  <title>Drag to add a corner</title>
                </circle>
              )
            })}
            {selectedRoom.polygon.map((p, i) => (
              <rect
                key={`v${i}`}
                x={p.x - 5 * px}
                y={p.y - 5 * px}
                width={10 * px}
                height={10 * px}
                fill={COLORS.accent}
                stroke="#ffffff"
                strokeWidth={1.5 * px}
                style={{ cursor: 'move' }}
                onPointerDown={(e) => begin({ type: 'vertex', roomId: selectedRoom.id, index: i }, e)}
                onDoubleClick={(e) => {
                  e.stopPropagation()
                  store().deleteVertex(selectedRoom.id, i)
                }}
              >
                <title>Drag to move; double-click to delete</title>
              </rect>
            ))}
          </g>
        )}

        {/* Scan reference lengths and the one being drawn: only while measuring, or while that scan is
            selected (its inspector is open). Elsewhere, such as the onboarding room review, they'd be clutter. */}
        {scans
          .filter((r) => r.visible && (mode === 'scan-measure' || (selection?.kind === 'scan' && selection.id === r.id)))
          .map((ref) => (
            <g key={ref.id} pointerEvents="none" fontFamily="system-ui, sans-serif" fontSize={fs * 0.9} fontWeight={600}>
              {ref.references.map((r) => {
                const a = scanToPlan(ref, r.a)
                const b = scanToPlan(ref, r.b)
                const m = scale(add(a, b), 0.5)
                const measured = dist(a, b)
                const off = Math.abs(measured / r.length - 1)
                const color = off > 0.02 ? COLORS.warning : '#7a3fc2'
                return (
                  <g key={r.id}>
                    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={color} strokeWidth={2 * px} />
                    <circle cx={a.x} cy={a.y} r={3.5 * px} fill={color} />
                    <circle cx={b.x} cy={b.y} r={3.5 * px} fill={color} />
                    <text x={m.x} y={labelY(m.y)} textAnchor="middle" fill={color} stroke="#ffffff" strokeWidth={3 * px} paintOrder="stroke">
                      {formatLength(r.length, units)}
                      {off > 0.005 ? ` (scan ${formatLength(measured, units)})` : ''}
                    </text>
                  </g>
                )
              })}
            </g>
          ))}
        {scanDraft && (
          <g pointerEvents="none">
            {(() => {
              const live = measureLive
              const b = measureEnd
              return (
                <>
                  {live?.corner && cornerRing(live.p)}
                  {b && <line x1={scanDraft.a.x} y1={scanDraft.a.y} x2={b.x} y2={b.y} stroke="#7a3fc2" strokeWidth={2 * px} strokeDasharray={`${5 * px} ${3 * px}`} />}
                  <circle cx={scanDraft.a.x} cy={scanDraft.a.y} r={4 * px} fill="#7a3fc2" />
                  {b && <circle cx={b.x} cy={b.y} r={4 * px} fill="#7a3fc2" />}
                  {b && dist(scanDraft.a, b) > 0.05 && (
                    <text x={(scanDraft.a.x + b.x) / 2} y={labelY((scanDraft.a.y + b.y) / 2)} fontSize={fs * 0.9} textAnchor="middle" fill="#7a3fc2" fontFamily="system-ui, sans-serif" fontWeight={600} stroke="#ffffff" strokeWidth={3 * px} paintOrder="stroke">
                      {formatLength(dist(scanDraft.a, b), units)} on the scan
                    </text>
                  )}
                </>
              )
            })()}
          </g>
        )}
        {mode === 'scan-measure' && !scanDraft && cursor && activeScan && (() => {
          const hit = scanMagnet(cursor, activeScan)
          return hit ? (
            <>
              <circle cx={hit.p.x} cy={hit.p.y} r={4 * px} fill="#7a3fc2" pointerEvents="none" />
              {hit.corner && cornerRing(hit.p)}
            </>
          ) : null
        })()}
        {mode === 'scan-straighten' && straightenStart && cursor && (
          <g pointerEvents="none" stroke={COLORS.accent} strokeWidth={2 * px}>
            <line x1={straightenStart.x} y1={straightenStart.y} x2={cursor.x} y2={cursor.y} strokeDasharray={`${5 * px} ${3 * px}`} />
            <circle cx={straightenStart.x} cy={straightenStart.y} r={4 * px} fill={COLORS.accent} />
          </g>
        )}
        {mode === 'scan-move' && activeScan && (() => {
          const c = scanCenter(activeScan)
          return c ? <circle cx={c.x} cy={c.y} r={6 * px} fill="none" stroke={COLORS.accent} strokeWidth={2 * px} pointerEvents="none" /> : null
        })()}

        {/* Snap guides */}
        <g pointerEvents="none" stroke="#e0457b" strokeWidth={1.2 * px}>
          {guides.map((g, i) => (
            <line key={i} x1={g.a.x} y1={g.a.y} x2={g.b.x} y2={g.b.y} />
          ))}
        </g>

        {/* Room drawing draft */}
        {mode === 'draw-room' && (
          <g pointerEvents="none">
            {draft.length > 0 && <polyline points={[...draft, ...(draftPreview ? [draftPreview] : [])].map((p) => `${p.x},${p.y}`).join(' ')} fill="none" stroke={COLORS.accent} strokeWidth={2 * px} />}
            {draft.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r={(i === 0 ? 6 : 4) * px} fill={i === 0 ? '#ffffff' : COLORS.accent} stroke={COLORS.accent} strokeWidth={2 * px} />
            ))}
            {draftPreview && draft.length > 0 && (
              <text x={draftPreview.x + 8 * px} y={draftPreview.y - 8 * px} fontSize={fs} fill={COLORS.accent} fontFamily="system-ui, sans-serif" fontWeight={600}>
                {formatLength(dist(draft[draft.length - 1], draftPreview), units)}
              </text>
            )}
            {!draft.length && cursor && (() => {
              const p = traceSnap(null, cursor, false)
              return <circle cx={p.x} cy={p.y} r={4 * px} fill={COLORS.accent} />
            })()}
          </g>
        )}

        {/* Opening placement preview */}
        {openingPreview &&
          (() => {
            const [a, b] = wallSegments(openingPreview.room)[openingPreview.wall]
            const width = mode === 'door' ? 0.8 : 1.2
            const t = Math.max(0, Math.min(dist(a, b) - width, openingPreview.t - width / 2))
            const d = norm(sub(b, a))
            const s = add(a, scale(d, t))
            const e = add(a, scale(d, t + width))
            return <line x1={s.x} y1={s.y} x2={e.x} y2={e.y} stroke={COLORS.accent} strokeWidth={6 * px} strokeLinecap="round" pointerEvents="none" />
          })()}

        {/* Scale bar (not while sizing an unsized scan: its length would be a guess) */}
        {!(mode === 'scan-measure' && activeScan && !activeScan.references.length) && <g transform={`translate(${vb.x + 16 * px} ${vb.y + vb.h - 18 * px})`} pointerEvents="none" fontFamily="system-ui, sans-serif">
          <rect x={0} y={-4 * px} width={scaleBar} height={4 * px} fill={COLORS.text} />
          <text x={scaleBar + 6 * px} y={0} fontSize={fs * 0.9} fill={COLORS.text}>
            {formatLength(scaleBar, units)}
          </text>
        </g>}
      </svg>

      {!readOnly && mode === 'draw-room' && (
        <div className="draw-hint">
          {draft.length === 0 ? (
            <span>Click to place the first corner.</span>
          ) : (
            <>
              <span>Click the next corner, or type a length:</span>
              <input
                autoFocus
                value={lengthInput}
                placeholder={units === 'metric' ? 'e.g. 350 or 3.5m' : `e.g. 11'6"`}
                onChange={(e) => setLengthInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.stopPropagation()
                    if (lengthInput) applyTypedLength()
                    else finishDraft()
                  }
                }}
              />
              <span className="muted">Enter or double-click to finish · Shift for free angles · Esc to cancel</span>
            </>
          )}
        </div>
      )}
      {!readOnly && (mode === 'door' || mode === 'window') && <div className="draw-hint">Click a wall to add a {mode}. Esc to cancel.</div>}
      {!readOnly && mode === 'scan-measure' && (
        <div className="draw-hint">
          {!scanDraft || scanDraft.b ? 'Click one end of something you measured with a tape (wall to wall works best).' : 'Click the other end.'}
          {measureSkew && <strong> That line runs at a slant, not along a wall. Measure along a wall for the best fit.</strong>}
          <span className="muted"> Points snap to the scan’s walls; hold Alt to place freely. Esc when done.</span>
        </div>
      )}
      {!readOnly && mode === 'scan-straighten' && <div className="draw-hint">{straightenStart ? 'Click further along the same wall.' : 'Click two points along a long, straight wall of the scan.'} Esc to cancel.</div>}
      {!readOnly && mode === 'scan-move' && <div className="draw-hint">Drag to move the scan. Esc when done.</div>}
      {!readOnly && mode === 'draw-room' && visibleScans.length > 0 && draft.length === 0 && <div className="draw-hint draw-hint-sub">Corners snap to the scan’s walls. Hold Shift to place freely.</div>}
      {!readOnly && (
        <div className="plan-tools">
          <button onClick={() => fit()} title="Fit to view">
            Fit
          </button>
        </div>
      )}
    </div>
  )
}

/** Simple top-down symbols so items read at a glance. */
function ItemGlyph({ category, hw, hd, px }: { category: string; hw: number; hd: number; px: number }) {
  const s = { stroke: '#000000', strokeOpacity: 0.22, strokeWidth: px, fill: 'none', pointerEvents: 'none' as const }
  const entry = catalogEntry(category)
  void entry
  switch (category) {
    case 'sofa':
    case 'armchair': {
      const back = Math.min(0.2, hd * 0.4)
      const arm = Math.min(0.18, hw * 0.25)
      return (
        <g {...s}>
          <line x1={-hw} y1={-hd + back} x2={hw} y2={-hd + back} />
          <line x1={-hw + arm} y1={-hd + back} x2={-hw + arm} y2={hd} />
          <line x1={hw - arm} y1={-hd + back} x2={hw - arm} y2={hd} />
        </g>
      )
    }
    case 'bed': {
      const pillow = Math.min(0.35, hd * 0.2)
      return (
        <g {...s}>
          <rect x={-hw + 0.08} y={-hd + 0.06} width={hw - 0.12} height={pillow} rx={0.04} />
          <rect x={0.04} y={-hd + 0.06} width={hw - 0.12} height={pillow} rx={0.04} />
          <line x1={-hw} y1={-hd + pillow + 0.25} x2={hw} y2={-hd + pillow + 0.25} />
        </g>
      )
    }
    case 'dining-chair':
    case 'office-chair':
      return (
        <g {...s}>
          <line x1={-hw} y1={-hd + 0.08} x2={hw} y2={-hd + 0.08} />
        </g>
      )
    case 'plant':
    case 'floor-lamp':
    case 'table-lamp':
      return (
        <g {...s}>
          <circle cx={0} cy={0} r={Math.min(hw, hd) * 0.6} />
        </g>
      )
    default:
      return null
  }
}

/** Whether a plan point is over a scan's footprint. */
function insideScan(ref: ScanRef, p: Vec2) {
  const img = underlayOf(ref, 'walls')
  if (!img) return false
  const u = img.underlay
  const q = planToScan(ref, p)
  return q.x >= u.u0 && q.y >= u.v0 && q.x <= u.u0 + u.width * u.pixel && q.y <= u.v0 + u.height * u.pixel
}

export type { Doc, Layout }

type Box = { minX: number; minY: number; maxX: number; maxY: number }

/**
 * Where a room's name and area go: its centre if that's free, else the free spot nearest it
 * (inside the room, clear of every piece), or null when the room has no room for them.
 */
function roomLabelSpot(poly: Vec2[], c: Vec2, fs: number, w: number, boxes: Box[]): Vec2 | null {
  const free = (q: Vec2) => {
    const b = { minX: q.x - w / 2, maxX: q.x + w / 2, minY: q.y - fs * 1.1, maxY: q.y + fs * 1.5 }
    if (![v(b.minX, b.minY), v(b.maxX, b.minY), v(b.maxX, b.maxY), v(b.minX, b.maxY)].every((p) => pointInPolygon(p, poly))) return false
    return boxes.every((o) => o.maxX < b.minX || o.minX > b.maxX || o.maxY < b.minY || o.minY > b.maxY)
  }
  if (free(c)) return c
  const rb = bounds(poly)
  const tries: Vec2[] = []
  const N = 8
  for (let i = 0; i <= N; i++) for (let j = 0; j <= N; j++) tries.push(v(rb.minX + ((rb.maxX - rb.minX) * i) / N, rb.minY + ((rb.maxY - rb.minY) * j) / N))
  tries.sort((p, q) => dist(p, c) - dist(q, c))
  return tries.find(free) ?? null
}
