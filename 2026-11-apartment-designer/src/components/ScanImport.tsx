import { useRef, useState } from 'react'
import type { ScanRef, UpAxis } from '../model/types'
import { formatLength, parseLength, unitLabel } from '../model/units'
import { calibrate, detectFloorCeiling, detectUpAxis, planToScan, refLength, rotateAbout, straightenAngle, uprightPositions, wallBandPoints } from '../scan/align'
import { SCAN_ACCEPT, isScanFileName } from '../scan/importers'
import { bandOf, importScanFile, relinkScanFile, scanCenter, scanEntry, uprightOf, useScanRuntime } from '../scan/runtime'
import { SAMPLE_TRUTH, samplePly } from '../scan/sampleScan'
import { useStore } from '../store'
import { LengthField, TextField } from './fields'

const NO_SCANS: ScanRef[] = []
const fmtSize = (b: number) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`)

/** Where a new scan goes: to the right of the rooms already drawn, top-aligned with them. */
function placeNextToRooms() {
  const pts = useStore.getState().doc.rooms.flatMap((r) => r.polygon)
  if (!pts.length) return { x: 0, y: 0 }
  return { x: Math.max(...pts.map((p) => p.x)) + 0.8, y: Math.min(...pts.map((p) => p.y)) }
}

async function doImport(file: { name: string; type?: string; blob: Blob }, setBusy: (s: string | null) => void) {
  const s = useStore.getState()
  setBusy(`Importing ${file.name}…`)
  try {
    const ref = await importScanFile(file, placeNextToRooms)
    s.addScan(ref)
    s.setView(s.view === '3d' ? 'split' : s.view)
    const up = ref.upAxis === 'y' ? '' : ` It was stored ${ref.upAxis === '-y' ? 'upside down' : 'Z up'}, so it was turned upright.`
    s.notify(`Imported ${ref.name}: floor found and walls squared to the plan.${up} Next, set its scale.`)
  } catch (err) {
    s.notify(`Couldn't import ${file.name}: ${err instanceof Error ? err.message : String(err)}`)
  } finally {
    setBusy(null)
  }
}

export function ScansPanel() {
  const scans = useStore((s) => s.doc.scans ?? NO_SCANS)
  const selection = useStore((s) => s.selection)
  const entries = useScanRuntime((s) => s.entries)
  const [busy, setBusy] = useState<string | null>(null)
  const [over, setOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const s = useStore.getState()
  // Once the home has a real scan, this panel is for adding the next room, not trying things out.
  const hasReal = scans.some((r) => !/^sample/i.test(r.fileName ?? r.name))

  return (
    <section
      className="panel"
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = Array.from(e.dataTransfer.files).find((x) => isScanFileName(x.name))
        if (f) void doImport({ name: f.name, type: f.type, blob: f }, setBusy)
        else s.notify('Drop a PLY, SPZ, GLB or OBJ scan file.')
      }}
    >
      <h3>{hasReal ? 'Add a room' : 'Scans'}</h3>
      {scans.length > 0 && (
        <ul className="list layouts">
          {scans.map((r) => {
            const st = entries[r.id]?.status
            return (
              <li key={r.id} className={selection?.kind === 'scan' && selection.id === r.id ? 'active' : undefined}>
                <button className="layout-name" onClick={() => s.setUI({ selection: { kind: 'scan', id: r.id }, activeScanId: r.id })}>
                  {r.name} {st && st !== 'ready' && <span className={`badge ${st === 'loading' ? 'ok' : 'warning'}`}>{st === 'loading' ? 'loading' : st === 'missing' ? 'file missing' : 'error'}</span>}
                </button>
                <label className="small" title="Show this scan">
                  <input type="checkbox" checked={r.visible} onChange={(e) => s.updateScan(r.id, { visible: e.target.checked })} /> Show
                </label>
              </li>
            )
          })}
        </ul>
      )}
      {hasReal ? (
        <p className="muted small">
          Scan another room in Scaniverse and bring it in here. It lands beside your other rooms, ready to size and furnish.
        </p>
      ) : scans.length === 0 && (
        <p className="muted small">
          Import a room scan to trace your walls over it at true size. In Scaniverse, export a splat as <strong>PLY</strong> or <strong>SPZ</strong>, or a mesh as <strong>GLB</strong> or OBJ.
        </p>
      )}
      <div className={`drop-hint small muted${over ? ' over' : ''}`}>{busy ?? 'Drop a scan file here, or'}</div>
      <div className="btn-row">
        <button className="primary g-btn sm" disabled={!!busy} onClick={() => fileRef.current?.click()}>
          {hasReal ? 'Import another room…' : 'Import scan…'}
        </button>
        {!hasReal && <button className="g-btn sm ghost" disabled={!!busy} onClick={() => void doImport({ name: 'Sample living room.ply', type: 'application/octet-stream', blob: new Blob([samplePly() as BlobPart]) }, setBusy)} title="A synthetic splat scan of an L-shaped room, to try the tools">
          Try a sample
        </button>}
      </div>
      <input
        ref={fileRef}
        type="file"
        accept={SCAN_ACCEPT}
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void doImport({ name: f.name, type: f.type, blob: f }, setBusy)
          e.target.value = ''
        }}
      />
    </section>
  )
}

function Seg<T extends string>({ value, options, onChange }: { value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <span className="seg">
      {options.map(([v, label]) => (
        <button key={v} className={v === value ? 'on' : undefined} onClick={() => onChange(v)}>
          {label}
        </button>
      ))}
    </span>
  )
}

/** A slider whose drag is one undo step. */
function Range({ label, value, min, max, step, format, onChange }: { label: string; value: number; min: number; max: number; step: number; format: (v: number) => string; onChange: (v: number) => void }) {
  const s = useStore.getState()
  return (
    <label className="range-row">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onPointerDown={() => s.beginGesture()}
        onPointerUp={() => s.endGesture()}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <output>{format(value)}</output>
    </label>
  )
}

export function ScanInspector({ id }: { id: string }) {
  const ref = useStore((s) => (s.doc.scans ?? []).find((r) => r.id === id))
  const units = useStore((s) => s.doc.apartment.units)
  const mode = useStore((s) => s.mode)
  const activeScanId = useStore((s) => s.activeScanId)
  const scanDraft = useStore((s) => s.scanDraft)
  const entry = useScanRuntime((s) => s.entries[id])
  const s = useStore.getState()
  const relinkRef = useRef<HTMLInputElement>(null)
  if (!ref) return null
  const ready = entry?.status === 'ready'
  const anchor = () => scanCenter(ref) ?? { x: ref.x, y: ref.y }
  const update = (patch: Partial<ScanRef>) => s.updateScan(ref.id, patch)
  const startMode = (m: 'scan-measure' | 'scan-straighten' | 'scan-move' | 'draw-room') => {
    s.setUI({ activeScanId: ref.id })
    s.setMode(mode === m && activeScanId === ref.id ? 'select' : m)
  }
  const cal = calibrate(ref.references)
  const ceilingM = ref.ceiling != null ? ref.scale * (ref.ceiling - ref.floor) : null
  const fmtM = (v: number) => formatLength(v, units)

  return (
    <section className="panel">
      <h3>Scan</h3>
      <TextField label="Name" value={ref.name} onChange={(name) => update({ name })} />
      <p className="small muted">
        {ref.fileName} · {fmtSize(ref.fileSize)} · {ref.format === 'splat' ? 'Gaussian splat' : ref.format === 'mesh' ? 'mesh' : 'point cloud'}
        {entry?.loaded && <> · {entry.loaded.points.count.toLocaleString()} points</>}
      </p>
      {entry && entry.status !== 'ready' && <div className={`scan-status${entry.status === 'loading' ? '' : ' error'}`}>{entry.message ?? entry.status}</div>}
      {(entry?.status === 'missing' || entry?.status === 'error') && (
        <>
          <button onClick={() => relinkRef.current?.click()}>Open the scan file…</button>
          <input
            ref={relinkRef}
            type="file"
            accept={SCAN_ACCEPT}
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) relinkScanFile(ref, f).catch((err) => s.notify(`Couldn't open it: ${err instanceof Error ? err.message : err}`))
              e.target.value = ''
            }}
          />
        </>
      )}

      <h4>1 · Line it up</h4>
      <label className="field">
        <span>Up direction in the file</span>
        <span className="field-input">
          <select value={ref.upAxis} onChange={(e) => reorient(ref, e.target.value as UpAxis)}>
            <option value="y">Y up (Scaniverse, glTF)</option>
            <option value="-y">Y down</option>
            <option value="z">Z up</option>
            <option value="-z">Z down</option>
          </select>
          <button className="link" disabled={!ready} onClick={() => reorient(ref, detectUpAxis(scanEntry(ref.id)!.loaded!.points.positions))} title="Guess again from the scan">
            Detect
          </button>
        </span>
      </label>
      <div className="btn-row">
        <button disabled={!ready} onClick={() => autoStraighten(ref)} title="Turn the scan so its walls run along the plan's axes">
          Square up
        </button>
        <button disabled={!ready} className={mode === 'scan-straighten' && activeScanId === ref.id ? 'primary' : undefined} onClick={() => startMode('scan-straighten')} title="Click two points along a wall">
          Along a wall
        </button>
        <button disabled={!ready} onClick={() => update(rotateAbout(ref, -Math.PI / 2, anchor()))} title="Rotate 90° counterclockwise">
          ⟲ 90°
        </button>
        <button disabled={!ready} onClick={() => update(rotateAbout(ref, Math.PI / 2, anchor()))} title="Rotate 90° clockwise">
          ⟳ 90°
        </button>
        <button disabled={!ready} className={mode === 'scan-move' && activeScanId === ref.id ? 'primary' : undefined} onClick={() => startMode('scan-move')}>
          Move
        </button>
      </div>
      <Range
        label="Floor level"
        value={Math.round(ref.scale * (ref.floor - (ref.floorDetected ?? ref.floor)) * 100) / 100}
        min={-0.3}
        max={0.3}
        step={0.01}
        format={(v) => (v === 0 ? 'as found' : `${v > 0 ? '+' : ''}${Math.round(v * 100)} cm`)}
        onChange={(v) => update({ floor: (ref.floorDetected ?? ref.floor) + v / ref.scale, floorDetected: ref.floorDetected ?? ref.floor })}
      />

      <h4>2 · Set the scale</h4>
      {ref.references.length === 0 ? (
        <p className="small muted">
          A phone scan without LiDAR is only roughly to scale. Measure 2 or 3 long distances with a tape (whole walls, wall to wall, are best), then mark each one on the scan.
        </p>
      ) : (
        <ul className="list refs">
          {ref.references.map((r, i) => {
            const dev = cal?.deviations[i] ?? 0
            return (
              <li key={r.id}>
                <span style={{ flex: 1 }}>
                  <InlineLength value={r.length} units={units} onChange={(length) => s.updateScanReference(ref.id, r.id, { length }, anchor())} />
                  <span className="muted"> · scan alone {fmtM(refLength(r))}</span>
                </span>
                {ref.references.length > 1 && (
                  <span className={`dev${Math.abs(dev) > 0.02 ? ' bad' : ''}`} title="How far this length is from the others">
                    {dev >= 0 ? '+' : ''}
                    {(dev * 100).toFixed(1)}%
                  </span>
                )}
                <button className="link danger" onClick={() => s.deleteScanReference(ref.id, r.id, anchor())}>
                  ✕
                </button>
              </li>
            )
          })}
        </ul>
      )}
      {cal && (
        <p className={`small ${cal.spread > 0.02 ? '' : 'muted'}`}>
          {ref.references.length === 1
            ? `Scale ×${cal.scale.toFixed(3)} from one length. Add one or two more to cross-check.`
            : cal.spread > 0.02
              ? `⚠ These lengths disagree by up to ${(cal.spread * 100).toFixed(1)}%. Re-measure the highlighted one, or check its ends sit on the right walls. The scale uses the median (×${cal.scale.toFixed(3)}).`
              : `Scale ×${cal.scale.toFixed(3)} · ${ref.references.length} lengths agree within ${(cal.spread * 100).toFixed(1)}%.`}
        </p>
      )}
      {scanDraft?.scanId === ref.id && scanDraft.b ? (
        <ReferenceForm
          scanLength={Math.hypot(scanDraft.b.x - scanDraft.a.x, scanDraft.b.y - scanDraft.a.y)}
          onAdd={(length) => {
            const a = planToScan(ref, scanDraft.a)
            const b = planToScan(ref, scanDraft.b!)
            s.addScanReference(ref.id, { a, b, length }, anchor())
            s.setUI({ scanDraft: null })
          }}
          onCancel={() => s.setUI({ scanDraft: null })}
        />
      ) : (
        <div className="btn-row">
          <button disabled={!ready} className={mode === 'scan-measure' && activeScanId === ref.id ? 'primary' : undefined} onClick={() => startMode('scan-measure')}>
            {mode === 'scan-measure' && activeScanId === ref.id ? 'Done measuring' : ref.references.length ? 'Add a length' : 'Mark a measured length'}
          </button>
        </div>
      )}
      {ref.name.startsWith('Sample living room') && (
        <p className="small muted">
          Sample room tape measures: {SAMPLE_TRUTH.references.map((r) => `${r.label.toLowerCase()} ${fmtM(r.length)}`).join(', ')}.
        </p>
      )}

      <h4>3 · Trace the walls</h4>
      <p className="small muted">
        Click each inside corner. Corners snap to the walls in the scan{ceilingM ? `, and the room gets the scan's ${fmtM(ceilingM)} ceiling` : ''}.
      </p>
      <div className="btn-row">
        <button disabled={!ready} className={mode === 'draw-room' ? 'primary' : undefined} onClick={() => startMode('draw-room')}>
          {mode === 'draw-room' ? 'Stop tracing' : 'Trace a room'}
        </button>
      </div>

      <h4>Display</h4>
      <div className="range-row">
        <span>On the plan</span>
        <Seg
          value={ref.underlay}
          options={[
            ['walls', 'Wall outline'],
            ['top', 'Top view'],
            ['off', 'Off'],
          ]}
          onChange={(underlay) => update({ underlay })}
        />
      </div>
      <Range label="Opacity" value={ref.opacity} min={0.1} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(opacity) => update({ opacity })} />
      {ref.underlay === 'walls' && (
        <>
          <Range label="Outline from" value={ref.sliceLow} min={0} max={Math.max(0.1, ref.sliceHigh - 0.1)} step={0.05} format={fmtM} onChange={(sliceLow) => update({ sliceLow })} />
          <Range label="Outline to" value={ref.sliceHigh} min={ref.sliceLow + 0.1} max={3.5} step={0.05} format={fmtM} onChange={(sliceHigh) => update({ sliceHigh })} />
          <p className="small muted">Points between these heights draw the outline. Keep it above your furniture.</p>
        </>
      )}
      <Range label="Cut above" value={ref.cutHeight} min={0.3} max={4} step={0.05} format={fmtM} onChange={(cutHeight) => update({ cutHeight })} />
      <p className="small muted">The top view and the 3D view hide the scan above this height, so you can see in.</p>
      <div className="btn-row">
        <label className="small">
          <input type="checkbox" checked={ref.visible} onChange={(e) => update({ visible: e.target.checked })} /> Show this scan
        </label>
      </div>
      <div className="btn-row">
        <button
          className="danger"
          onClick={() => {
            s.deleteScan(ref.id)
            s.notify(`Scan "${ref.name}" removed. Press Ctrl+Z to undo.`)
          }}
        >
          Remove scan
        </button>
      </div>
    </section>
  )
}

function InlineLength({ value, units, onChange }: { value: number; units: 'metric' | 'imperial'; onChange: (v: number) => void }) {
  const [editing, setEditing] = useState(false)
  if (!editing)
    return (
      <button className="link" onClick={() => setEditing(true)} title="Click to correct the measured length">
        <strong>{formatLength(value, units)}</strong>
      </button>
    )
  return (
    <span style={{ display: 'inline-block', width: 110 }}>
      <LengthField
        label=""
        value={value}
        units={units}
        min={0.05}
        onChange={(v) => {
          onChange(v)
          setEditing(false)
        }}
      />
    </span>
  )
}

function ReferenceForm({ scanLength, onAdd, onCancel }: { scanLength: number; onAdd: (m: number) => void; onCancel: () => void }) {
  const units = useStore((s) => s.doc.apartment.units)
  const [text, setText] = useState('')
  const m = parseLength(text, units)
  const ok = m != null && m > 0.05
  return (
    <form
      className="scan-status"
      onSubmit={(e) => {
        e.preventDefault()
        if (ok) onAdd(m!)
      }}
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === 'Escape') onCancel()
      }}
    >
      <label className="field">
        <span>
          Tape-measured length {unitLabel(units) && `(${unitLabel(units)})`} · {formatLength(scanLength, units)} on the scan
        </span>
        <input autoFocus id="ref-length" value={text} placeholder={units === 'metric' ? 'e.g. 412 or 4.12m' : `e.g. 13'6"`} onChange={(e) => setText(e.target.value)} />
      </label>
      <div className="btn-row">
        <button type="submit" className="primary" disabled={!ok}>
          Use this length
        </button>
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  )
}

/** Change the up axis and re-find the floor and ceiling for it. */
function reorient(ref: ScanRef, upAxis: UpAxis) {
  const e = scanEntry(ref.id)
  const s = useStore.getState()
  if (!e?.loaded) return s.updateScan(ref.id, { upAxis })
  const { floor, ceiling } = detectFloorCeiling(uprightPositions(e.loaded.points.positions, upAxis))
  const next = { ...ref, upAxis, floor, ceiling, floorDetected: undefined }
  const yaw = straightenAngle(wallBandPoints(uprightPositions(e.loaded.points.positions, upAxis), next, 200_000), next.scale)
  s.updateScan(ref.id, { upAxis, floor, ceiling, floorDetected: undefined, yaw })
}

function autoStraighten(ref: ScanRef) {
  const band = bandOf(ref)
  if (!band || !uprightOf(ref)) return
  const target = straightenAngle(band, ref.scale)
  // Smallest turn that gets there (walls repeat every 90°).
  let delta = target - ref.yaw
  delta = ((((delta + Math.PI / 4) % (Math.PI / 2)) + Math.PI / 2) % (Math.PI / 2)) - Math.PI / 4
  const s = useStore.getState()
  if (Math.abs(delta) < 1e-4) return s.notify('The scan is already square to the plan.')
  s.updateScan(ref.id, rotateAbout(ref, delta, scanCenter(ref) ?? { x: ref.x, y: ref.y }))
}
