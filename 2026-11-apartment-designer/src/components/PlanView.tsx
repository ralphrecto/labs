import { useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import { useAnalysis } from '../analysis'
import { catalogEntry } from '../model/catalog'
import {
  add,
  bounds,
  centroid,
  clearanceZones,
  closestPointOnSegment,
  dist,
  doorSwing,
  dot,
  footprint,
  norm,
  offsetPolygon,
  openingEndpoints,
  polygonArea,
  scale,
  sub,
  v,
  wallInwardNormal,
  wallSegments,
} from '../model/geometry'
import { snapDrawPoint, snapPlacement, type SnapGuide } from '../model/snapping'
import type { Doc, Layout, Room, Vec2 } from '../model/types'
import { formatLength, parseLength } from '../model/units'
import { activeLayout, useStore } from '../store'

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

interface ViewState {
  cx: number
  cy: number
  /** Pixels per meter. */
  k: number
}

export interface PlanViewProps {
  /** Show a specific layout read-only (used by compare). Defaults to the active layout, editable. */
  layoutId?: string
  readOnly?: boolean
  svgRef?: React.RefObject<SVGSVGElement | null>
  title?: string
}

export function PlanView({ layoutId, readOnly, svgRef: externalRef, title }: PlanViewProps) {
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

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth || 1, h: el.clientHeight || 1 }))
    ro.observe(el)
    setSize({ w: el.clientWidth || 1, h: el.clientHeight || 1 })
    return () => ro.disconnect()
  }, [])

  const fit = () => {
    const pts = doc.rooms.flatMap((r) => r.polygon)
    if (!pts.length) {
      setView({ cx: 3, cy: 2, k: Math.min(size.w / 8, size.h / 6) })
      return
    }
    const b = bounds(pts)
    const w = b.maxX - b.minX + 1.6
    const h = b.maxY - b.minY + 1.6
    setView({ cx: (b.minX + b.maxX) / 2, cy: (b.minY + b.maxY) / 2, k: Math.min(size.w / w, size.h / h) })
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
  // Re-fit when the whole document is replaced (new apartment, import).
  const aptId = doc.apartment.id
  useEffect(() => {
    if (fitted.current) fit()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aptId])

  useEffect(() => {
    if (mode !== 'draw-room') {
      setDraft([])
      setLengthInput('')
    }
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

  const store = useStore.getState
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
      addDraftPoint(snapDrawPoint(draft[draft.length - 1] ?? null, w, snap.grid, !e.shiftKey))
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
      const others = placementsWithItems.filter((x) => x.p.id !== drag.id)
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
      store().addRoom(`Room ${doc.rooms.length + 1}`, draft)
      store().setMode('select')
    }
    setDraft([])
  }
  const draftPreview = draft.length && cursor ? snapDrawPoint(draft[draft.length - 1], cursor, snap.grid, true) : null
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
        style={{ background: COLORS.paper }}
      >
        <rect x={vb.x} y={vb.y} width={vb.w} height={vb.h} fill={COLORS.paper} onPointerDown={onBackgroundDown} />
        <g pointerEvents="none">
          {gridLines.map((l, i) => (
            <line key={i} {...l} stroke={l.major ? COLORS.gridMajor : COLORS.grid} strokeWidth={px} />
          ))}
        </g>

        {/* Rooms: floor, walls, labels */}
        {doc.rooms.map((room) => {
          const selected = selectedRoom?.id === room.id
          const outer = offsetPolygon(room.polygon, room.wallThickness)
          const d = `M${outer.map((p) => `${p.x},${p.y}`).join('L')}Z M${room.polygon.map((p) => `${p.x},${p.y}`).join('L')}Z`
          const c = centroid(room.polygon)
          return (
            <g key={room.id}>
              <polygon
                points={room.polygon.map((p) => `${p.x},${p.y}`).join(' ')}
                fill={COLORS.floor}
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
              <g pointerEvents="none" fill={COLORS.muted} fontSize={fs} textAnchor="middle" fontFamily="system-ui, sans-serif">
                <text x={c.x} y={c.y - fs * 0.2} fontWeight={600} fill={COLORS.text} opacity={0.55}>
                  {room.name}
                </text>
                <text x={c.x} y={c.y + fs * 1.1} opacity={0.55}>
                  {units === 'metric' ? `${polygonArea(room.polygon).toFixed(1)} m²` : `${Math.round(polygonArea(room.polygon) * 10.7639)} ft²`}
                </text>
              </g>
            </g>
          )
        })}

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
              const n = wallInwardNormal(room, i)
              const m = add(scale(add(a, b), 0.5), scale(n, 14 * px))
              let ang = (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI
              if (ang > 90) ang -= 180
              if (ang < -90) ang += 180
              return (
                <text key={`${room.id}:${i}`} x={m.x} y={m.y} transform={`rotate(${ang} ${m.x} ${m.y})`} dominantBaseline="middle" opacity={selectedRoom?.id === room.id ? 1 : 0.7}>
                  {formatLength(l, units)}
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
          .sort((a, b) => Number(!!b.item.floorLevel) - Number(!!a.item.floorLevel))
          .map(({ p, item }) => {
            const selected = selection?.kind === 'placement' && selection.id === p.id
            const flag = analysis.flagged.get(p.id)
            const stroke = selected ? COLORS.accent : flag === 'error' ? COLORS.error : flag === 'warning' ? COLORS.warning : '#4a4740'
            const hw = item.width / 2
            const hd = item.depth / 2
            const deg = (p.rotation * 180) / Math.PI
            const labelSize = Math.min(fs, (item.width / Math.max(4, item.name.length)) * 1.6)
            return (
              <g key={p.id} transform={`translate(${p.x} ${p.y}) rotate(${deg})`}>
                <rect
                  x={-hw}
                  y={-hd}
                  width={item.width}
                  height={item.depth}
                  rx={item.category === 'rug' ? 0 : Math.min(0.03, hw / 4)}
                  fill={item.color}
                  fillOpacity={item.floorLevel ? 0.45 : 0.85}
                  stroke={stroke}
                  strokeWidth={(selected || flag ? 2.5 : 1) * px}
                  strokeDasharray={item.floorLevel ? `${5 * px} ${3 * px}` : undefined}
                  style={{ cursor: readOnly ? undefined : 'grab' }}
                  onPointerDown={(e) => {
                    if (readOnly || mode !== 'select') return onBackgroundDown(e)
                    store().select({ kind: 'placement', id: p.id })
                    const w = toWorld(e)
                    begin({ type: 'item', id: p.id, grab: sub(w, v(p.x, p.y)), moved: false }, e)
                  }}
                />
                {!item.floorLevel && <line x1={-hw + 0.04} y1={hd - 0.04} x2={hw - 0.04} y2={hd - 0.04} stroke="#ffffff" strokeOpacity={0.8} strokeWidth={2 * px} pointerEvents="none" />}
                <ItemGlyph category={item.category} hw={hw} hd={hd} px={px} />
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
                  transform={Math.abs(Math.sin(p.rotation)) > 0.7 && item.depth > item.width ? `rotate(${p.rotation > Math.PI ? 90 : -90})` : Math.cos(p.rotation) < -0.7 ? 'rotate(180)' : undefined}
                >
                  {item.name}
                </text>
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

        {/* Selected item dimensions */}
        {selection?.kind === 'placement' &&
          (() => {
            const pi = placementsWithItems.find((x) => x.p.id === selection.id)
            if (!pi) return null
            const fp = footprint(pi.item, pi.p)
            const [c0, c1, , c3] = fp
            const wMid = scale(add(c0, c1), 0.5)
            const dMid = scale(add(c0, c3), 0.5)
            const out = norm(sub(wMid, v(pi.p.x, pi.p.y)))
            const out2 = norm(sub(dMid, v(pi.p.x, pi.p.y)))
            const t1 = add(wMid, scale(out, 12 * px))
            const t2 = add(dMid, scale(out2, 12 * px))
            return (
              <g pointerEvents="none" fontSize={fs * 0.9} fill={COLORS.accent} fontFamily="system-ui, sans-serif" textAnchor="middle" dominantBaseline="middle" fontWeight={600}>
                <text x={t1.x} y={t1.y}>{formatLength(pi.item.width, units)}</text>
                <text x={t2.x} y={t2.y}>{formatLength(pi.item.depth, units)}</text>
              </g>
            )
          })()}

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
            {!draft.length && cursor && <circle cx={snapDrawPoint(null, cursor, snap.grid, false).x} cy={snapDrawPoint(null, cursor, snap.grid, false).y} r={4 * px} fill={COLORS.accent} />}
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

        {/* Scale bar */}
        <g transform={`translate(${vb.x + 16 * px} ${vb.y + vb.h - 18 * px})`} pointerEvents="none" fontFamily="system-ui, sans-serif">
          <rect x={0} y={-4 * px} width={scaleBar} height={4 * px} fill={COLORS.text} />
          <text x={scaleBar + 6 * px} y={0} fontSize={fs * 0.9} fill={COLORS.text}>
            {formatLength(scaleBar, units)}
          </text>
        </g>
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
      {!readOnly && (
        <div className="plan-tools">
          <button onClick={fit} title="Fit to view">
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
      return (
        <g {...s}>
          <circle cx={0} cy={0} r={Math.min(hw, hd) * 0.6} />
        </g>
      )
    default:
      return null
  }
}

export type { Doc, Layout }
