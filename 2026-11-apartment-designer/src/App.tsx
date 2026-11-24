import { useEffect, useRef, useState } from 'react'
import { useAnalysis } from './analysis'
import { CatalogPanel, Inspector, IssuesPanel, LayoutsPanel, LibraryPanel } from './components/Panels'
import { ScansPanel } from './components/ScanImport'
import { ensureScanLoaded } from './scan/runtime'
import { pruneScanFiles } from './scan/fileStore'
import { PlanView } from './components/PlanView'
import { ScanImportPanel, ScanReviewPanel } from './components/ScanPanels'
import { useScanUI } from './scan/scanUi'
import { exportPlanPNG } from './exporting'
import { Scene3D } from './three/Scene3D'
import { activeLayout, loadSample, newEmptyDoc, useStore, type Mode, type View } from './store'
import { allHomeScanIds, downloadJSON, parseDoc } from './storage'
import { parseLength, unitLabel } from './model/units'
import { flushEditRun, nudgePlacement, rotatePlacementSmart } from './game/editActions'

/** Set when built for an embedded preview where printing and downloads are unavailable. */
const EMBEDDED = import.meta.env.VITE_EMBED === '1'

export function RoomSizeForm({ onDone }: { onDone: () => void }) {
  const units = useStore((s) => s.doc.apartment.units)
  const [name, setName] = useState(`Room ${useStore.getState().doc.rooms.length + 1}`)
  const [w, setW] = useState(units === 'metric' ? '350' : `12'`)
  const [d, setD] = useState(units === 'metric' ? '300' : `10'`)
  const pw = parseLength(w, units)
  const pd = parseLength(d, units)
  const ok = !!pw && !!pd && pw > 0.3 && pd > 0.3
  return (
    <form
      className="popover"
      onSubmit={(e) => {
        e.preventDefault()
        if (!ok) return
        useStore.getState().addRect(name || 'Room', pw!, pd!)
        onDone()
      }}
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === 'Escape') onDone()
      }}
    >
      <label className="field">
        <span>Name</span>
        <input id="room-name" value={name} onChange={(e) => setName(e.target.value)} autoFocus />
      </label>
      <div className="grid2">
        <label className="field">
          <span>Width {unitLabel(units) && `(${unitLabel(units)})`}</span>
          <input id="room-width" className={pw ? undefined : 'bad'} value={w} onChange={(e) => setW(e.target.value)} />
        </label>
        <label className="field">
          <span>Depth {unitLabel(units) && `(${unitLabel(units)})`}</span>
          <input id="room-depth" className={pd ? undefined : 'bad'} value={d} onChange={(e) => setD(e.target.value)} />
        </label>
      </div>
      <p className="small muted">Inside measurements, wall to wall. It goes to the right of your other rooms; drag its walls to move it.</p>
      <div className="btn-row">
        <button type="submit" className="primary" disabled={!ok}>
          Add room
        </button>
        <button type="button" onClick={onDone}>
          Cancel
        </button>
      </div>
    </form>
  )
}

export function Toast() {
  const toast = useStore((s) => s.toast)
  return toast ? (
    <div className="toast" role="status">
      {toast}
    </div>
  ) : null
}

export function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') return
      const s = useStore.getState()
      const mod = e.metaKey || e.ctrlKey
      // Only more turns (R , .) and nudges (arrows) extend a run of edits into one undo step; any
      // other key (undo, duplicate, delete...) closes it first, so each stays its own step.
      const runKey = !mod && !e.altKey && (e.key.toLowerCase() === 'r' || ',.<>'.includes(e.key) || e.key.startsWith('Arrow'))
      const modifier = e.key === 'Shift' || e.key === 'Control' || e.key === 'Meta' || e.key === 'Alt'
      if (!runKey && !modifier) flushEditRun()
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) s.redo()
        else s.undo()
      } else if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault()
        s.redo()
      } else if (mod && e.key.toLowerCase() === 'd' && s.selection?.kind === 'placement') {
        e.preventDefault()
        s.duplicatePlacement(s.selection.id)
      } else if (!mod && !e.altKey && e.key.toLowerCase() === 'r' && s.selection?.kind === 'placement') {
        // A quarter turn (Shift turns back); , and . turn 15° for fine angles.
        rotatePlacementSmart(s.selection.id, ((e.shiftKey ? -1 : 1) * Math.PI) / 2)
      } else if (!mod && (e.key === ',' || e.key === '.' || e.key === '<' || e.key === '>') && s.selection?.kind === 'placement') {
        rotatePlacementSmart(s.selection.id, ((e.key === ',' || e.key === '<' ? -1 : 1) * Math.PI) / 12)
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (s.mode === 'draw-room') return
        if (s.selection?.kind === 'placement') s.deletePlacement(s.selection.id)
        else if (s.selection?.kind === 'opening') s.deleteOpening(s.selection.roomId, s.selection.id)
      } else if (e.key === 'Escape') {
        if (s.mode !== 'select') s.setMode('select')
        else s.select(null)
      } else if (s.selection?.kind === 'placement' && e.key.startsWith('Arrow')) {
        e.preventDefault()
        // Arrows follow the screen (in 3D they turn with the camera); a run of them is one undo step.
        const dx = e.key === 'ArrowLeft' ? -1 : e.key === 'ArrowRight' ? 1 : 0
        const dy = e.key === 'ArrowUp' ? -1 : e.key === 'ArrowDown' ? 1 : 0
        nudgePlacement(s.selection.id, { x: dx, y: dy }, e.shiftKey ? 0.1 : 0.01)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}

/** Show or hide scan point clouds under the plan and in 3D (only when there is a scan). */
function ScanToggle() {
  const hasScan = useStore((s) => s.doc.rooms.some((r) => r.scan))
  const show = useScanUI((s) => s.showUnderlay)
  if (!hasScan) return null
  return (
    <label title="Show the scan's points under the plan and in 3D">
      <input type="checkbox" checked={show} onChange={(e) => useScanUI.getState().setShowUnderlay(e.target.checked)} /> Scan
    </label>
  )
}

function Toolbar({ planRef }: { planRef: React.RefObject<SVGSVGElement | null> }) {
  const mode = useStore((s) => s.mode)
  const view = useStore((s) => s.view)
  const units = useStore((s) => s.doc.apartment.units)
  const snap = useStore((s) => s.snap)
  const showClearance = useStore((s) => s.showClearance)
  const showPath = useStore((s) => s.showPath)
  const canUndo = useStore((s) => s.past.length > 0)
  const canRedo = useStore((s) => s.future.length > 0)
  const s = useStore.getState()
  const fileRef = useRef<HTMLInputElement>(null)

  const modeBtn = (m: Mode, label: string, title: string) => (
    <button className={mode === m ? 'primary' : undefined} onClick={() => s.setMode(mode === m ? 'select' : m)} title={title}>
      {label}
    </button>
  )
  const viewBtn = (v: View, label: string) => (
    <button className={view === v ? 'primary' : undefined} onClick={() => s.setView(v)}>
      {label}
    </button>
  )

  const [roomForm, setRoomForm] = useState(false)
  return (
    <header className="toolbar">
      <div className="brand">Apartment designer</div>
      <div className="group">
        {modeBtn('select', 'Select', 'Select and move (Esc)')}
        <span className="popover-anchor">
          <button className={roomForm ? 'primary' : undefined} onClick={() => setRoomForm(!roomForm)} title="Add a rectangular room by typing its size">
            + Room
          </button>
          {roomForm && <RoomSizeForm onDone={() => setRoomForm(false)} />}
        </span>
        {modeBtn('draw-room', 'Draw room', 'Click corners; type lengths')}
        {modeBtn('door', 'Door', 'Click a wall to add a door')}
        {modeBtn('window', 'Window', 'Click a wall to add a window')}
      </div>
      <div className="group">
        <button disabled={!canUndo} onClick={() => s.undo()} title="Undo (Ctrl+Z)">
          Undo
        </button>
        <button disabled={!canRedo} onClick={() => s.redo()} title="Redo (Ctrl+Shift+Z)">
          Redo
        </button>
      </div>
      <div className="group">
        {viewBtn('split', 'Plan + 3D')}
        {viewBtn('plan', 'Plan')}
        {viewBtn('3d', '3D')}
      </div>
      <div className="group toggles">
        <label title="Snap to a grid">
          Grid
          <select value={snap.grid} onChange={(e) => s.setSnap({ grid: Number(e.target.value) })}>
            <option value={0}>off</option>
            <option value={0.01}>1 cm</option>
            <option value={0.05}>5 cm</option>
            <option value={0.1}>10 cm</option>
            <option value={0.0254}>1 in</option>
            <option value={0.1524}>6 in</option>
          </select>
        </label>
        <label title="Snap backs against walls">
          <input type="checkbox" checked={snap.walls} onChange={(e) => s.setSnap({ walls: e.target.checked })} /> Walls
        </label>
        <label title="Align edges with other items">
          <input type="checkbox" checked={snap.items} onChange={(e) => s.setSnap({ items: e.target.checked })} /> Edges
        </label>
        <label>
          <input type="checkbox" checked={showClearance} onChange={(e) => s.setUI({ showClearance: e.target.checked })} /> Clearance
        </label>
        <label>
          <input type="checkbox" checked={showPath} onChange={(e) => s.setUI({ showPath: e.target.checked })} /> Paths
        </label>
        <ScanToggle />
        <select value={units} onChange={(e) => s.setUnits(e.target.value as 'metric' | 'imperial')} title="Display units">
          <option value="metric">cm / m</option>
          <option value="imperial">ft / in</option>
        </select>
      </div>
      <div className="group">
        <details className="menu">
          <summary>File</summary>
          <div className="menu-items">
            {!EMBEDDED && (
              <>
            <button
              onClick={() => {
                const svg = planRef.current
                if (!svg) return s.notify('Switch to a view that shows the plan first.')
                const d = useStore.getState().doc
                exportPlanPNG(svg, `${d.apartment.name} · ${activeLayout(d).name}`)
              }}
            >
              Export plan as PNG
            </button>
                <button onClick={() => window.print()}>Print or save as PDF</button>
                <button onClick={() => downloadJSON(useStore.getState().doc)}>Download project (JSON)</button>
              </>
            )}
            <button onClick={() => fileRef.current?.click()}>Open project (JSON)…</button>
            <hr />
            <button
              onClick={() => {
                newEmptyDoc()
                s.notify('Started a new apartment. Press Ctrl+Z to get the previous one back.')
              }}
            >
              New empty apartment
            </button>
            <button
              onClick={() => {
                loadSample()
                s.notify('Loaded the sample apartment. Press Ctrl+Z to go back.')
              }}
            >
              Load sample apartment
            </button>
            {EMBEDDED && <p className="small muted menu-note">Exporting PNG, PDF and JSON works when you run the app from the repo. Here, your work is saved in this browser.</p>}
          </div>
        </details>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={async (e) => {
            const f = e.target.files?.[0]
            if (!f) return
            try {
              const d = parseDoc(await f.text())
              if (!d) throw new Error('bad')
              useStore.getState().replaceDoc(d)
            } catch {
              s.notify("That file isn't an apartment designer project.")
            }
            e.target.value = ''
          }}
        />
      </div>
    </header>
  )
}

function CompareView() {
  const doc = useStore((s) => s.doc)
  const compare = useStore((s) => s.compare)
  const setUI = useStore((s) => s.setUI)
  const ids = compare ?? [doc.layouts[0].id, doc.layouts[1]?.id ?? doc.layouts[0].id]
  const pick = (i: 0 | 1, id: string) => {
    const next: [string, string] = [...ids] as [string, string]
    next[i] = id
    setUI({ compare: next })
  }
  return (
    <div className="compare">
      {[0, 1].map((i) => {
        const layout = doc.layouts.find((l) => l.id === ids[i]) ?? doc.layouts[0]
        return (
          <div key={i} className="compare-col">
            <div className="compare-head">
              <select value={layout.id} onChange={(e) => pick(i as 0 | 1, e.target.value)}>
                {doc.layouts.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.name}
                  </option>
                ))}
              </select>
              <CompareSummary layoutId={layout.id} />
            </div>
            <PlanView layoutId={layout.id} readOnly />
          </div>
        )
      })}
    </div>
  )
}

function CompareSummary({ layoutId }: { layoutId: string }) {
  const doc = useStore((s) => s.doc)
  const layout = doc.layouts.find((l) => l.id === layoutId)!
  const { issues } = useAnalysis(doc, layout)
  const n = issues.filter((i) => i.severity !== 'info').length
  return <span className={`badge ${n ? 'warning' : 'ok'}`}>{n ? `${n} issue${n > 1 ? 's' : ''}` : 'All clear'}</span>
}

/** Load the files behind the document's scans, and clean up files no scan uses any more. */
export function useScanFiles() {
  const scans = useStore((s) => s.doc.scans)
  useEffect(() => {
    for (const r of scans ?? []) void ensureScanLoaded(r)
  }, [scans])
  useEffect(() => {
    // Keep the scan files of every saved home, not just the open one.
    void pruneScanFiles(allHomeScanIds((useStore.getState().doc.scans ?? []).map((r) => r.id)))
  }, [])
}

export default function App() {
  useShortcuts()
  useScanFiles()
  const view = useStore((s) => s.view)
  const layoutName = useStore((s) => activeLayout(s.doc).name)
  const planRef = useRef<SVGSVGElement>(null)
  return (
    <div className="app">
      <Toolbar planRef={planRef} />
      <aside className="left">
        <LayoutsPanel />
        <ScansPanel />
        <CatalogPanel />
        <LibraryPanel />
        <ScanImportPanel />
      </aside>
      <main className={`main view-${view}`}>
        {view === 'compare' ? (
          <CompareView />
        ) : (
          <>
            {view !== '3d' && (
              <div className="pane plan-pane">
                <PlanView svgRef={planRef} title={layoutName} />
              </div>
            )}
            {view !== 'plan' && (
              <div className="pane scene-pane">
                <Scene3D />
              </div>
            )}
          </>
        )}
      </main>
      <aside className="right">
        <ScanReviewPanel />
        <Inspector />
        <IssuesPanel />
        <Toast />
        <section className="panel help small muted">
          <strong>Shortcuts:</strong> drag to move · R / Shift+R rotate 15° · arrows nudge (Shift = 10 cm) · Ctrl+D duplicate · Delete remove · Ctrl+Z undo · Alt while dragging skips snapping · scroll to zoom, drag empty space to pan.
        </section>
      </aside>
    </div>
  )
}
