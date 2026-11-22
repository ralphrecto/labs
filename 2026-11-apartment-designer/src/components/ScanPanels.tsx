import { useEffect, useMemo, useRef, useState } from 'react'
import { CATALOG, catalogEntry } from '../model/catalog'
import { polygonArea } from '../model/geometry'
import { formatLength } from '../model/units'
import { placementShift } from '../scan/apply'
import { parsePLY } from '../scan/ply'
import { reviewChoice, useScanUI, type ReviewObject } from '../scan/scanUi'
import { syntheticScan } from '../scan/synthetic'
import { topDownImage } from '../scan/underlay'
import { useStore } from '../store'
import { LengthField } from './fields'

// ---------------------------------------------------------------- left: start from a scan

export function ScanImportPanel() {
  const busy = useScanUI((s) => s.busy)
  const analyze = useScanUI((s) => s.analyze)
  const importSpatialLM = useScanUI((s) => s.importSpatialLM)
  const fileRef = useRef<HTMLInputElement>(null)
  const [text, setText] = useState('')
  const hasReal = useStore((s) => (s.doc.scans ?? []).some((r) => !/^sample/i.test(r.fileName ?? r.name)))
  return (
    <section className="panel">
      <h3>From a scan</h3>
      <p className="small muted">Turn a room scan into walls, doors and your existing furniture, then fix anything it got wrong.</p>
      <div className="btn-row">
        <button disabled={!!busy} onClick={() => fileRef.current?.click()} title="A point cloud or Gaussian splat export (.ply), in meters">
          Open scan (.ply)…
        </button>
        {!hasReal && (
          <button disabled={!!busy} onClick={() => analyze(syntheticScan())}>
            Try a sample scan
          </button>
        )}
      </div>
      {busy && <p className="small scan-busy">{busy}</p>}
      <details>
        <summary className="small">Paste SpatialLM output</summary>
        <p className="small muted">If you ran the scan through SpatialLM, paste its Wall / Door / Window / Bbox lines here instead.</p>
        <textarea className="scan-text" rows={4} value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.stopPropagation()} placeholder="wall_0=Wall(…)&#10;bbox_0=Bbox(bed,…)" />
        <div className="btn-row">
          <button
            disabled={!text.trim()}
            onClick={() => {
              importSpatialLM(text)
              setText('')
            }}
          >
            Import layout
          </button>
        </div>
      </details>
      <input
        ref={fileRef}
        type="file"
        accept=".ply"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0]
          e.target.value = ''
          if (!f) return
          try {
            const scan = parsePLY(await f.arrayBuffer(), f.name)
            await analyze(scan)
          } catch (err) {
            useStore.getState().notify(`Couldn't open ${f.name}: ${(err as Error).message}`)
          }
        }}
      />
    </section>
  )
}

// ---------------------------------------------------------------- right: review what was found

function confidenceBadge(o: ReviewObject) {
  if (o.category === 'box') return <span className="badge warning">unknown</span>
  if (o.confidence >= 0.6) return null
  return <span className="badge warning" title="The size fits more than one kind of furniture; check the type">check type</span>
}

export function ScanReviewPanel() {
  const review = useScanUI((s) => s.review)
  const ui = useScanUI.getState()
  const units = useStore((s) => s.doc.apartment.units)
  const rooms = useStore((s) => s.doc.rooms)
  const listRef = useRef<HTMLUListElement>(null)
  useEffect(() => {
    if (!review?.focusId) return
    listRef.current?.querySelector(`[data-id="${review.focusId}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [review?.focusId])
  if (!review) return null
  const { estimate, objects } = review
  const room = estimate.room
  const accepted = objects.filter((o) => o.accepted).length
  return (
    <section className="panel scan-review">
      <h3>Review scan</h3>
      {estimate.notes.map((n) => (
        <p key={n} className="small muted">
          {n}
        </p>
      ))}
      {room ? (
        <>
          <label className="field checkbox">
            <input type="checkbox" checked={review.includeRoom} onChange={(e) => ui.setReview({ includeRoom: e.target.checked })} />
            <span>
              Room outline · {polygonArea(room.polygon).toFixed(1)} m² · {room.polygon.length} walls · {room.openings.filter((o) => o.kind === 'door').length} doors · {room.openings.filter((o) => o.kind === 'window').length} windows · ceiling {formatLength(room.ceilingHeight, units)}
            </span>
          </label>
          {review.includeRoom && (
            <div className="grid2">
              <label className="field">
                <span>Put it</span>
                <select value={review.replaceRoomId ?? ''} onChange={(e) => ui.setReview({ replaceRoomId: e.target.value || undefined })}>
                  <option value="">As a new room</option>
                  {rooms.map((r) => (
                    <option key={r.id} value={r.id}>
                      In place of {r.name}
                    </option>
                  ))}
                </select>
              </label>
              {!review.replaceRoomId && (
                <label className="field">
                  <span>Name</span>
                  <input value={review.roomName} onChange={(e) => ui.setReview({ roomName: e.target.value })} onKeyDown={(e) => e.stopPropagation()} />
                </label>
              )}
            </div>
          )}
        </>
      ) : (
        <p className="small muted">No room outline. The furniture goes into your first room; move it into place.</p>
      )}
      <h4>
        Furniture · {accepted} of {objects.length}
      </h4>
      <ul className="scan-objects" ref={listRef}>
        {objects.map((o) => (
          <ScanObjectRow key={o.id} o={o} focused={review.focusId === o.id} />
        ))}
      </ul>
      <div className="btn-row">
        <button className="link small" onClick={() => ui.setAll(true)}>
          Check all
        </button>
        <button className="link small" onClick={() => ui.setAll(false)}>
          Uncheck all
        </button>
      </div>
      <div className="btn-row">
        <button className="primary" disabled={!accepted && !(review.includeRoom && room)} onClick={() => ui.apply()}>
          Add to plan
        </button>
        <button onClick={() => ui.cancel()}>Discard</button>
      </div>
      <p className="small muted">Click a box on the plan to find it here. Sizes are measured from the scan; change the type if it guessed wrong. Everything stays editable after you add it.</p>
    </section>
  )
}

function ScanObjectRow({ o, focused }: { o: ReviewObject; focused: boolean }) {
  const ui = useScanUI.getState()
  const units = useStore((s) => s.doc.apartment.units)
  const [opened, setOpen] = useState(false)
  const open = opened || focused
  const alts = o.alternatives.map((a) => a.category).filter((c) => c !== o.category)
  const rest = CATALOG.map((c) => c.category).filter((c) => c !== o.category && !alts.includes(c))
  return (
    <li data-id={o.id} className={`${focused ? 'focused' : ''} ${o.accepted ? '' : 'rejected'}`}>
      <div className="scan-row">
        <input type="checkbox" checked={o.accepted} onChange={(e) => ui.setObject(o.id, { accepted: e.target.checked })} title="Add this one" />
        <span className="swatch" style={{ background: o.color ?? catalogEntry(o.category).color }} />
        <select
          value={o.category}
          onChange={(e) => ui.setCategory(o.id, e.target.value)}
          onFocus={() => ui.setReview({ focusId: o.id })}
          title="What it is"
        >
          <option value={o.category}>{o.name}</option>
          {alts.length > 0 && (
            <optgroup label="Could also be">
              {alts.map((c) => (
                <option key={c} value={c}>
                  {catalogEntry(c).name}
                </option>
              ))}
            </optgroup>
          )}
          <optgroup label="Other">
            {rest.map((c) => (
              <option key={c} value={c}>
                {catalogEntry(c).name}
              </option>
            ))}
          </optgroup>
        </select>
      </div>
      <div className="scan-row scan-sub">
        <button className="link small scan-size" onClick={() => setOpen(!open)} title="Edit the size">
          {formatLength(o.width, units)} × {formatLength(o.depth, units)} × {formatLength(o.height, units)} high
        </button>
        {confidenceBadge(o)}
      </div>
      {open && (
        <div className="grid3">
          <LengthField label="Width" value={o.width} units={units} onChange={(width) => ui.setObject(o.id, { width })} min={0.05} />
          <LengthField label="Depth" value={o.depth} units={units} onChange={(depth) => ui.setObject(o.id, { depth })} min={0.05} />
          <LengthField label="Height" value={o.height} units={units} onChange={(height) => ui.setObject(o.id, { height })} min={0.01} />
        </div>
      )}
    </li>
  )
}

// ---------------------------------------------------------------- plan layer

/** Scan point clouds under the plan, plus the review preview (detected outline and boxes). */
export function PlanScanLayer({ px, fs }: { px: number; fs: number }) {
  const rooms = useStore((s) => s.doc.rooms)
  const doc = useStore((s) => s.doc)
  const show = useScanUI((s) => s.showUnderlay)
  const clouds = useScanUI((s) => s.clouds)
  const review = useScanUI((s) => s.review)
  const linked = rooms.filter((r) => r.scan)
  useEffect(() => {
    for (const r of linked) useScanUI.getState().ensureCloud(r.scan!.scanId)
  }, [linked.map((r) => r.scan!.scanId).join()]) // eslint-disable-line react-hooks/exhaustive-deps

  const shift = useMemo(() => (review ? placementShift(doc, review.estimate, reviewChoice(review)) : null), [doc, review])

  return (
    <g pointerEvents="none">
      {show &&
        linked.map((r) => {
          const cloud = clouds[r.scan!.scanId]
          const td = cloud && topDownImage(cloud, r.ceilingHeight)
          if (!td) return null
          return (
            <g key={r.id} transform={`translate(${r.scan!.x} ${r.scan!.y}) rotate(${(r.scan!.rotation * 180) / Math.PI})`} opacity={0.6}>
              <image href={td.url} x={td.x} y={td.z} width={td.w} height={td.h} preserveAspectRatio="none" style={{ imageRendering: 'pixelated' }} />
            </g>
          )
        })}
      {review && shift && <ReviewOverlay px={px} fs={fs} shift={shift} />}
    </g>
  )
}

function ReviewOverlay({ px, fs, shift }: { px: number; fs: number; shift: { x: number; y: number } }) {
  const review = useScanUI((s) => s.review)!
  const clouds = useScanUI((s) => s.clouds)
  const { estimate, objects } = review
  const t = estimate.transform
  const cloud = clouds[estimate.scanId]
  const td = cloud && topDownImage(cloud, estimate.room?.ceilingHeight ?? 2.6)
  const room = review.includeRoom ? estimate.room : null
  const accent = '#2f6fde'
  return (
    <g transform={`translate(${shift.x} ${shift.y})`}>
      {td && (
        <g transform={`translate(${t.x} ${t.y}) rotate(${(t.rotation * 180) / Math.PI})`} opacity={0.85}>
          <image href={td.url} x={td.x} y={td.z} width={td.w} height={td.h} preserveAspectRatio="none" />
        </g>
      )}
      {room && (
        <polygon
          points={room.polygon.map((p) => `${p.x},${p.y}`).join(' ')}
          fill="none"
          stroke={accent}
          strokeWidth={3 * px}
          strokeDasharray={`${8 * px} ${5 * px}`}
        />
      )}
      {room?.openings.map((o, i) => {
        const a = room.polygon[o.wall]
        const b = room.polygon[(o.wall + 1) % room.polygon.length]
        const L = Math.hypot(b.x - a.x, b.y - a.y)
        const d = { x: (b.x - a.x) / L, y: (b.y - a.y) / L }
        return (
          <line
            key={i}
            x1={a.x + d.x * o.offset}
            y1={a.y + d.y * o.offset}
            x2={a.x + d.x * (o.offset + o.width)}
            y2={a.y + d.y * (o.offset + o.width)}
            stroke={o.kind === 'door' ? '#d68a1f' : '#6fb3e6'}
            strokeWidth={7 * px}
            strokeLinecap="butt"
          />
        )
      })}
      {objects.map((o) => {
        const focused = review.focusId === o.id
        return (
          <g key={o.id} transform={`translate(${o.x} ${o.y}) rotate(${(o.rotation * 180) / Math.PI})`} pointerEvents="all" style={{ cursor: 'pointer' }}>
            <rect
              x={-o.width / 2}
              y={-o.depth / 2}
              width={o.width}
              height={o.depth}
              fill={o.accepted ? (o.color ?? catalogEntry(o.category).color) : '#ffffff'}
              fillOpacity={o.accepted ? 0.55 : 0.2}
              stroke={focused ? accent : o.accepted ? '#3b3a36' : '#8b877d'}
              strokeWidth={(focused ? 3 : 1.5) * px}
              strokeDasharray={o.accepted ? undefined : `${4 * px} ${3 * px}`}
              onPointerDown={(e) => {
                e.stopPropagation()
                useScanUI.getState().setReview({ focusId: o.id })
              }}
            />
            {/* Front edge */}
            <line x1={-o.width / 2 + 0.03} y1={o.depth / 2 - 0.03} x2={o.width / 2 - 0.03} y2={o.depth / 2 - 0.03} stroke="#ffffff" strokeWidth={2.5 * px} pointerEvents="none" />
            <text
              x={0}
              y={0}
              fontSize={Math.min(fs, (o.width / Math.max(4, o.name.length)) * 1.6)}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="#1f1e1b"
              fontFamily="system-ui, sans-serif"
              fontWeight={600}
              pointerEvents="none"
              transform={Math.cos(o.rotation) < -0.7 ? 'rotate(180)' : undefined}
            >
              {o.accepted ? o.name : `${o.name} (skip)`}
            </text>
          </g>
        )
      })}
    </g>
  )
}
