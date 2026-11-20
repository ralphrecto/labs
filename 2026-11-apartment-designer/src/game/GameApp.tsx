import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react'
import App, { useScanFiles, useShortcuts } from '../App'
import { useAnalysis } from '../analysis'
import { Inspector } from '../components/Panels'
import { PlanView } from '../components/PlanView'
import { ScanReviewPanel } from '../components/ScanPanels'
import { ScansPanel } from '../components/ScanImport'
import { catalogEntry } from '../model/catalog'
import { polygonArea } from '../model/geometry'
import { formatLength, parseLength, unitLabel } from '../model/units'
import { toScanData } from '../scan/align'
import { scanEntry } from '../scan/runtime'
import { useScanUI } from '../scan/scanUi'
import { activeLayout, loadSample, useStore } from '../store'
import { downloadJSON, parseDoc } from '../storage'
import { Scene3D, type SceneInset, type WallMode } from '../three/Scene3D'
import { BuyDrawer, ShopSheet } from './BuyDrawer'
import { ItemThumb } from './ItemThumb'
import { GoalsSheet, VibeMeter } from './Goals'
import { useGame, type GameMode } from './gameStore'
import * as I from './icons'
import { Onboarding } from './Onboarding'
import { StylePicker, useStylePicker } from './StylePicker'
import { useModal } from './useModal'
import { reduceMotion } from '../motion'
import './theme.css'
import './hud.css'
import './goals.css'

const EMBEDDED = import.meta.env.VITE_EMBED === '1'

/** Picks the game or, if the player asked for it in settings, the classic all-panels editor. */
export function Root() {
  const classic = useGame((s) => s.classic)
  return classic ? (
    <>
      <App />
      <button className="hud-back-to-game" onClick={() => useGame.getState().setClassic(false)}>
        ← Back to the game
      </button>
    </>
  ) : (
    <GameApp />
  )
}

function GameApp() {
  useShortcuts()
  useScanFiles()
  const phase = useGame((s) => s.phase)
  const mode = useGame((s) => s.mode)
  const sheet = useGame((s) => s.sheet)
  const review = useScanUI((s) => s.review)
  const selection = useStore((s) => s.selection)
  const storeMode = useStore((s) => s.mode)

  // Esc in the game also closes sheets.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && useGame.getState().sheet) useGame.getState().openSheet(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const pickingStyle = useStylePicker((s) => s.open)
  const tool = storeMode !== 'select'
  const showDrawer = phase === 'play' && mode === 'buy' && !selection && !tool && !review
  const gameRef = useRef<HTMLDivElement>(null)
  const inset = useHudInset(gameRef)
  return (
    <div ref={gameRef} className={`game mode-${mode}${tool ? ' tool-active' : ''}`}>
      <Stage inset={inset} />
      {phase === 'onboarding' ? (
        <Onboarding />
      ) : (
        <>
          {mode !== 'look' && <TopBar />}
          {mode === 'look' && <LookControls />}
          {mode === 'look' && <StylePicker className="sty-look" />}
          <CameraControls />
          {showDrawer && (
            // The drawer steps aside while the style picker is open, so on a phone the style tiles
            // and their names are never under it. Hidden rather than unmounted, so it comes back as it was.
            <div className={`hud-drawer-slot${pickingStyle ? ' tucked' : ''}`}>
              <BuyDrawer />
            </div>
          )}
          {mode === 'build' && !selection && !review && !tool && <BuildTray />}
          {tool && <ToolBanner />}
          {selection && !tool && mode !== 'look' && <ContextBar />}
          {review && (
            <div className="hud-review g-card">
              <ScanReviewPanel />
            </div>
          )}
          <Dock />
          <div className="hud-toast">
            <GameToast />
          </div>
        </>
      )}
      {sheet === 'goals' && <GoalsSheet />}
      {sheet === 'shop' && <ShopSheet />}
      {sheet === 'settings' && <SettingsSheet />}
      {sheet === 'layouts' && <DesignsSheet />}
      {sheet === 'details' && <DetailsSheet />}
      {sheet === 'room' && <NewRoomSheet />}
      {sheet === 'scan' && <ScanSheet />}
    </div>
  )
}

// ---------------------------------------------------------------- stage

export const useStageView = () => {
  const camera = useGame((s) => s.camera)
  const mode = useGame((s) => s.mode)
  const storeMode = useStore((s) => s.mode)
  // Drawing walls, doors, windows and scan tools work on the top-down plan.
  return storeMode !== 'select' ? 'plan' : mode === 'look' ? '3d' : camera
}

interface CamState {
  resetKey: number
  eye: boolean
  autoRotate: boolean
  wallMode: WallMode
}
// The Look spin starts off when the player asked their system to reduce motion; Spin still turns it on.
let camState: CamState = { resetKey: 0, eye: false, autoRotate: !reduceMotion, wallMode: 'auto' }
const GROUND = '#fff4e2'
const SKY = '#f3ece0'
const camListeners = new Set<() => void>()
function setCam(p: Partial<CamState>) {
  camState = { ...camState, ...p }
  camListeners.forEach((l) => l())
}
function useCam(): CamState {
  const [, force] = useState(0)
  useEffect(() => {
    const l = () => force((n) => n + 1)
    camListeners.add(l)
    return () => {
      camListeners.delete(l)
    }
  }, [])
  return camState
}

function Stage({ inset }: { inset: SceneInset }) {
  const view = useStageView()
  const phase = useGame((s) => s.phase)
  const mode = useGame((s) => s.mode)
  const cam = useCam()
  const style = useGame((s) => s.renderStyle)
  const control = useMemo(
    () => ({
      ...cam,
      ground: GROUND,
      sky: SKY,
      style,
      autoRotate: mode === 'look' && cam.autoRotate,
      eye: mode === 'look' && cam.eye,
      inset,
      hideFlags: mode === 'look',
    }),
    [cam, mode, style, inset],
  )
  return (
    <div
      className={`hud-stage view-${view}`}
      onDragOver={(e) => {
        if (e.dataTransfer.types.some((t) => t.startsWith('application/x-apt-'))) e.preventDefault()
      }}
      onDrop={(e) => {
        const itemId = e.dataTransfer.getData('application/x-apt-item')
        const cat = e.dataTransfer.getData('application/x-apt-catalog')
        if (!itemId && !cat) return
        e.preventDefault()
        if (itemId) useStore.getState().placeItem(itemId)
        else useStore.getState().addItemFromCatalog(catalogEntry(cat))
      }}
    >
      {phase === 'onboarding' ? null : view === '3d' ? <Scene3D control={control} /> : <PlanView />}
    </div>
  )
}

/** HUD pieces along the bottom and the top of the screen that cover the stage. */
const BOTTOM_HUD = ['.buy-drawer', '.hud-ctx-wrap', '.hud-tray', '.hud-dock', '.hud-review']
const TOP_HUD = ['.hud-top', '.hud-look-bar']

/**
 * How much of the stage the HUD covers at the top and bottom (CSS px), kept up to date as the
 * drawer, selection bar and tray come and go, so the 3D view can centre the home in what's left.
 */
function useHudInset(root: RefObject<HTMLDivElement | null>): SceneInset {
  const [inset, setInset] = useState<SceneInset>({ top: 0, bottom: 0 })
  useLayoutEffect(() => {
    const game = root.current
    if (!game) return
    let raf = 0
    const measure = () => {
      raf = 0
      const W = game.clientWidth
      const H = game.clientHeight
      let bottom = 0
      let top = 0
      for (const sel of BOTTOM_HUD) {
        const el = game.querySelector<HTMLElement>(sel)
        // Side panels (the scan review on wide screens) don't cover the middle of the stage.
        if (!el || el.offsetWidth < W * 0.6) continue
        bottom = Math.max(bottom, H - offsetTopIn(el, game))
      }
      for (const sel of TOP_HUD) {
        const el = game.querySelector<HTMLElement>(sel)
        if (el) top = Math.max(top, offsetTopIn(el, game) + el.offsetHeight)
      }
      setInset((was) => (Math.abs(was.top - top) < 4 && Math.abs(was.bottom - bottom) < 4 ? was : { top, bottom }))
    }
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(measure)
    }
    measure()
    const mo = new MutationObserver(schedule)
    mo.observe(game, { childList: true, subtree: true })
    window.addEventListener('resize', schedule)
    return () => {
      mo.disconnect()
      window.removeEventListener('resize', schedule)
      if (raf) cancelAnimationFrame(raf)
    }
  }, [root])
  return inset
}

/** An element's top edge inside `root`, ignoring transforms (so opening animations don't count). */
function offsetTopIn(el: HTMLElement, root: HTMLElement) {
  let y = 0
  let n: HTMLElement | null = el
  while (n && n !== root) {
    y += n.offsetTop
    n = n.offsetParent as HTMLElement | null
  }
  return y
}

const DRAG_HINT = /\s*Drag it into place\.$/
const DRAG_HINT_KEY = 'apartment-designer:drag-hint-count'
const DRAG_HINT_TIMES = 3

/**
 * The game's toast. "Drag it into place" after buying something is only said the first few times;
 * after that the toast just names what was placed.
 */
function GameToast() {
  const toast = useStore((s) => s.toast)
  const [hint, setHint] = useState<{ toast: string; keep: boolean } | null>(null)
  useEffect(() => {
    if (!toast || !DRAG_HINT.test(toast)) return
    let n = DRAG_HINT_TIMES
    try {
      n = Number(localStorage.getItem(DRAG_HINT_KEY)) || 0
      if (n < DRAG_HINT_TIMES) localStorage.setItem(DRAG_HINT_KEY, String(n + 1))
    } catch {
      // Private mode: show the hint every time rather than never.
      n = 0
    }
    setHint({ toast, keep: n < DRAG_HINT_TIMES })
  }, [toast])
  if (!toast) return null
  const text = DRAG_HINT.test(toast) && !(hint?.toast === toast && hint.keep) ? toast.replace(DRAG_HINT, '') : toast
  return (
    <div className="toast" role="status">
      {text}
    </div>
  )
}

// ---------------------------------------------------------------- top bar

function TopBar() {
  const name = useStore((s) => s.doc.apartment.name)
  const layoutName = useStore((s) => activeLayout(s.doc).name)
  const multi = useStore((s) => s.doc.layouts.length > 1)
  const canUndo = useStore((s) => s.past.length > 0)
  const canRedo = useStore((s) => s.future.length > 0)
  const s = useStore.getState()
  const open = useGame((g) => g.openSheet)
  return (
    <header className="hud-top">
      <button className="hud-home" onClick={() => open('layouts')} aria-label="Your designs">
        <span className="hud-home-icon">
          <I.IconHome />
        </span>
        <span className="hud-home-text">
          <strong>{name}</strong>
          {multi && <small>{layoutName}</small>}
        </span>
      </button>
      <div className="hud-center">
        <VibeMeter />
      </div>
      <div className="hud-right">
        {/* On phones these two move down beside the dock (hud.css). */}
        <div className="hud-undo">
          <button className="g-icon sm" disabled={!canUndo} onClick={() => s.undo()} aria-label="Undo">
            <I.IconUndo />
          </button>
          <button className="g-icon sm" disabled={!canRedo} onClick={() => s.redo()} aria-label="Redo">
            <I.IconRedo />
          </button>
        </div>
        <button className="g-icon sm" onClick={() => open('settings')} aria-label="Settings">
          <I.IconGear />
        </button>
      </div>
    </header>
  )
}

// ---------------------------------------------------------------- dock

const MODES: { m: GameMode; label: string; icon: ReactNode }[] = [
  { m: 'build', label: 'Build', icon: <I.IconHammer /> },
  { m: 'buy', label: 'Furnish', icon: <I.IconSofa /> },
  { m: 'look', label: 'Look', icon: <I.IconEye /> },
]

function Dock() {
  const mode = useGame((s) => s.mode)
  return (
    <nav className="hud-dock" aria-label="Game mode">
      {MODES.map(({ m, label, icon }) => (
        <button
          key={m}
          className={`hud-mode hud-mode-${m}${mode === m ? ' on' : ''}`}
          onClick={() => {
            const st = useStore.getState()
            if (st.mode !== 'select') st.setMode('select')
            st.select(null)
            useGame.getState().setMode(m)
            if (m === 'look') setCam({ autoRotate: !reduceMotion, eye: false, resetKey: camState.resetKey + 1 })
          }}
          aria-pressed={mode === m}
        >
          <span className="hud-mode-icon">{icon}</span>
          <span className="hud-mode-label">{label}</span>
        </button>
      ))}
    </nav>
  )
}

const CAM_LABELS_KEY = 'apartment-designer:cam-labels-seen'

/** Names next to the camera buttons on a first visit, until one is used or a few seconds pass. */
function useFirstRunLabels() {
  const [show, setShow] = useState(() => {
    try {
      return localStorage.getItem(CAM_LABELS_KEY) !== '1'
    } catch {
      return false
    }
  })
  const done = () => {
    setShow(false)
    try {
      localStorage.setItem(CAM_LABELS_KEY, '1')
    } catch {
      // Private mode: fine, they just show again next visit.
    }
  }
  useEffect(() => {
    if (!show) return
    const t = setTimeout(done, 9000)
    return () => clearTimeout(t)
  }, [show])
  return [show, done] as const
}

function CameraControls() {
  const view = useStageView()
  const mode = useGame((s) => s.mode)
  const tool = useStore((s) => s.mode !== 'select')
  const [labels, seen] = useFirstRunLabels()
  if (mode === 'look' || tool) return null
  const planLabel = view === '3d' ? 'Top-down plan' : '3D view'
  return (
    <div className={`hud-cam${labels ? ' labelled' : ''}`} onPointerDown={labels ? seen : undefined}>
      <div className="hud-cam-row">
        {labels && <span className="hud-cam-label">{view === '3d' ? 'Plan' : '3D'}</span>}
        <button className="g-icon sm" onClick={() => useGame.getState().setCamera(view === '3d' ? 'plan' : '3d')} aria-label={planLabel} title={planLabel}>
          {view === '3d' ? <I.IconMap /> : <I.IconCube />}
        </button>
      </div>
      {view === '3d' && (
        <div className="hud-cam-row">
          {labels && <span className="hud-cam-label">Recenter</span>}
          <button className="g-icon sm" onClick={() => setCam({ resetKey: camState.resetKey + 1 })} aria-label="Recenter the view" title="Recenter the view">
            <I.IconHome />
          </button>
        </div>
      )}
      {view === '3d' && (
        <div className="hud-cam-row">
          {labels && <span className="hud-cam-label">Style</span>}
          <StylePicker />
        </div>
      )}
    </div>
  )
}

function LookControls() {
  const cam = useCam()
  return (
    <div className="hud-look-bar">
      <div className="hud-look" role="group" aria-label="View">
        <button className={`g-chip${!cam.eye ? ' on' : ''}`} aria-pressed={!cam.eye} onClick={() => setCam({ eye: false, resetKey: cam.resetKey + 1 })}>
          <I.IconCube width={16} height={16} /> Dollhouse
        </button>
        <button className={`g-chip${cam.eye ? ' on' : ''}`} aria-pressed={cam.eye} onClick={() => setCam({ eye: true, autoRotate: false })}>
          <I.IconWalk width={16} height={16} /> Walk in
        </button>
      </div>
      {!cam.eye && (
        <button
          className={`hud-spin${cam.autoRotate ? ' on' : ''}`}
          aria-pressed={cam.autoRotate}
          aria-label={cam.autoRotate ? 'Stop spinning' : 'Spin the view'}
          title={cam.autoRotate ? 'Stop spinning' : 'Spin the view'}
          onClick={() => setCam({ autoRotate: !cam.autoRotate })}
        >
          <I.IconRotate width={20} height={20} />
        </button>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- build mode

function BuildTray() {
  const s = useStore.getState()
  const hasScan = useStore((st) => (st.doc.scans ?? []).length > 0)
  const open = useGame((g) => g.openSheet)
  const tools: { key: string; label: string; icon: ReactNode; on: () => void }[] = [
    { key: 'room', label: 'Room', icon: <I.IconRoom />, on: () => open('room') },
    { key: 'walls', label: 'Draw walls', icon: <I.IconPen />, on: () => s.setMode('draw-room') },
    { key: 'door', label: 'Door', icon: <I.IconDoor />, on: () => s.setMode('door') },
    { key: 'window', label: 'Window', icon: <I.IconWindow />, on: () => s.setMode('window') },
    { key: 'scan', label: hasScan ? 'Scans' : 'Add scan', icon: <I.IconScan />, on: () => open('scan') },
  ]
  return (
    <div className="hud-tray" role="toolbar" aria-label="Build tools">
      {tools.map((t) => (
        <button key={t.key} className="hud-tool" onClick={t.on}>
          <span className="hud-tool-icon">{t.icon}</span>
          <span>{t.label}</span>
        </button>
      ))}
    </div>
  )
}

/** While a plan tool is active: what to do, and a big Done button. */
function ToolBanner() {
  const mode = useStore((s) => s.mode)
  const label: Record<string, string> = {
    'draw-room': 'Tap each corner of the room. Tap the first corner again to close it.',
    door: 'Tap a wall to put a door there.',
    window: 'Tap a wall to put a window there.',
    'scan-measure': 'Tap both ends of something you measured, then type its real length.',
    'scan-straighten': 'Tap two points along a long straight wall.',
    'scan-move': 'Drag the scan into place.',
  }
  return (
    <div className="hud-toolbanner g-card">
      <span>{label[mode] ?? 'Editing'}</span>
      <button className="g-btn primary sm" onClick={() => useStore.getState().setMode('select')}>
        <I.IconCheck width={16} height={16} /> Done
      </button>
    </div>
  )
}

// ---------------------------------------------------------------- selection

const SWATCHES = ['#7d8fa3', '#a3907d', '#c9c3b6', '#8a6b4f', '#3d3d3d', '#5f8f5a', '#c7a98a', '#e2d6b8', '#b5534a', '#e0b04f', '#5b7bb5', '#f2efe8']

function ContextBar() {
  const sel = useStore((s) => s.selection)!
  const doc = useStore((s) => s.doc)
  const layout = activeLayout(doc)
  const { issues } = useAnalysis(doc, layout)
  const [paint, setPaint] = useState(false)
  const s = useStore.getState()
  const open = useGame((g) => g.openSheet)
  const units = doc.apartment.units
  useEffect(() => setPaint(false), [sel])

  let thumb: ReactNode = null
  let title = ''
  let sub = ''
  let actions: ReactNode = null
  let warn: string | null = null

  if (sel.kind === 'placement') {
    const p = layout.placements.find((x) => x.id === sel.id)
    const item = p && doc.items.find((i) => i.id === p.itemId)
    if (!p || !item) return null
    thumb = item.imageUrl ? <img src={item.imageUrl} alt="" /> : <ItemThumb category={item.category} color={item.color} size={44} />
    title = item.name
    sub = `${formatLength(item.width, units)} × ${formatLength(item.depth, units)}`
    warn = issues.find((i) => i.placementIds[0] === p.id && i.severity !== 'info')?.message ?? null
    actions = (
      <>
        <Act label="Rotate" onClick={() => s.rotatePlacement(p.id, Math.PI / 4)}>
          <I.IconRotate />
        </Act>
        <Act label="Copy" onClick={() => s.duplicatePlacement(p.id)}>
          <I.IconCopy />
        </Act>
        <Act label="Color" on={paint} onClick={() => setPaint(!paint)}>
          <I.IconPalette />
        </Act>
        <Act label="Details" onClick={() => open('details')}>
          <I.IconMore />
        </Act>
        <Act label="Remove" danger onClick={() => s.deletePlacement(p.id)}>
          <I.IconTrash />
        </Act>
      </>
    )
    if (paint) warn = null
  } else if (sel.kind === 'room') {
    const r = doc.rooms.find((x) => x.id === sel.id)
    if (!r) return null
    const a = polygonArea(r.polygon)
    thumb = (
      <span className="hud-ctx-roomicon">
        <I.IconRoom />
      </span>
    )
    title = r.name
    sub = units === 'metric' ? `${a.toFixed(1)} m²` : `${Math.round(a * 10.7639)} ft²`
    actions = (
      <>
        <Act label="Door" onClick={() => s.setMode('door')}>
          <I.IconDoor />
        </Act>
        <Act label="Window" onClick={() => s.setMode('window')}>
          <I.IconWindow />
        </Act>
        <Act label="Details" onClick={() => open('details')}>
          <I.IconMore />
        </Act>
        <Act
          label="Remove"
          danger
          onClick={() => {
            s.deleteRoom(r.id)
            s.notify(`Removed ${r.name}. Undo brings it back.`)
          }}
        >
          <I.IconTrash />
        </Act>
      </>
    )
  } else if (sel.kind === 'opening') {
    const r = doc.rooms.find((x) => x.id === sel.roomId)
    const o = r?.openings.find((x) => x.id === sel.id)
    if (!r || !o) return null
    thumb = <span className="hud-ctx-roomicon">{o.kind === 'door' ? <I.IconDoor /> : <I.IconWindow />}</span>
    title = o.kind === 'door' ? 'Door' : 'Window'
    sub = `${formatLength(o.width, units)} wide · ${r.name}`
    actions = (
      <>
        <Act label="Details" onClick={() => open('details')}>
          <I.IconMore />
        </Act>
        <Act label="Remove" danger onClick={() => s.deleteOpening(r.id, o.id)}>
          <I.IconTrash />
        </Act>
      </>
    )
  } else if (sel.kind === 'scan') {
    const r = (doc.scans ?? []).find((x) => x.id === sel.id)
    if (!r) return null
    thumb = (
      <span className="hud-ctx-roomicon">
        <I.IconScan />
      </span>
    )
    title = r.name
    sub = 'Scan'
    actions = (
      <Act label="Details" onClick={() => open('details')}>
        <I.IconMore />
      </Act>
    )
  }

  return (
    <div className="hud-ctx-wrap">
      {warn && <div className="hud-ctx-warn">{warn}</div>}
      {paint && sel.kind === 'placement' && (
        <div className="hud-paint g-card">
          {SWATCHES.map((c) => (
            <button
              key={c}
              style={{ background: c }}
              aria-label={`Color ${c}`}
              onClick={() => {
                const p = layout.placements.find((x) => x.id === sel.id)
                if (p) s.updateItem(p.itemId, { color: c })
              }}
            />
          ))}
        </div>
      )}
      <div className="hud-ctx g-card" role="toolbar" aria-label="Selected">
        <div className="hud-ctx-id">
          <span className="hud-ctx-thumb">{thumb}</span>
          <span className="hud-ctx-text">
            <strong>{title}</strong>
            <small>{sub}</small>
          </span>
        </div>
        <div className="hud-ctx-actions">{actions}</div>
        <button className="hud-ctx-done" onClick={() => s.select(null)} aria-label="Done">
          <I.IconCheck />
        </button>
      </div>
    </div>
  )
}

function Act({ label, onClick, children, danger, on }: { label: string; onClick: () => void; children: ReactNode; danger?: boolean; on?: boolean }) {
  return (
    <button className={`hud-act${danger ? ' danger' : ''}${on ? ' on' : ''}`} onClick={onClick} aria-label={label} title={label}>
      {children}
      <span>{label}</span>
    </button>
  )
}

// ---------------------------------------------------------------- sheets

function Sheet({ title, sub, children, className }: { title: string; sub?: string; children: ReactNode; className?: string }) {
  const close = () => useGame.getState().openSheet(null)
  const ref = useRef<HTMLDivElement>(null)
  useModal(ref, close)
  return (
    <div className="g-scrim" onPointerDown={(e) => e.target === e.currentTarget && close()}>
      <div ref={ref} className={`g-sheet hud-sheet ${className ?? ''}`} role="dialog" aria-modal="true" aria-label={title} onKeyDown={(e) => e.stopPropagation()}>
        <button className="g-icon sm hud-sheet-close" onClick={close} aria-label="Close">
          <I.IconClose />
        </button>
        <h2>{title}</h2>
        {sub && <p className="g-sub">{sub}</p>}
        {children}
      </div>
    </div>
  )
}

function DetailsSheet() {
  const sel = useStore((s) => s.selection)
  useEffect(() => {
    if (!sel) useGame.getState().openSheet(null)
  }, [sel])
  return (
    <Sheet title="Details" sub="Exact sizes and the fine print." className="hud-classic">
      <Inspector />
    </Sheet>
  )
}

function ScanSheet() {
  const scans = useStore((s) => s.doc.scans ?? [])
  const busy = useScanUI((s) => s.busy)
  const s = useStore.getState()
  const close = () => useGame.getState().openSheet(null)
  return (
    <Sheet title="Your scans" sub="Bring in a Scaniverse scan of your place, set its true size, and let the game find your furniture." className="hud-classic">
      {scans.map((r) => {
        const loaded = scanEntry(r.id)?.loaded
        return (
          <div key={r.id} className="hud-scanrow g-card">
            <span className="hud-ctx-roomicon">
              <I.IconScan />
            </span>
            <strong>{r.name}</strong>
            <div className="hud-scanrow-actions">
              <button
                className="g-btn sm"
                onClick={() => {
                  close()
                  useGame.getState().setCamera('plan')
                  s.setUI({ activeScanId: r.id, selection: { kind: 'scan', id: r.id } })
                  s.setMode('scan-measure')
                }}
              >
                <I.IconRuler width={16} height={16} /> Set size
              </button>
              <button
                className="g-btn sm primary"
                disabled={!loaded || !!busy}
                onClick={() => {
                  if (!loaded) return
                  close()
                  void useScanUI.getState().analyze(toScanData(r, loaded.points))
                }}
              >
                <I.IconSparkle width={16} height={16} /> Find furniture
              </button>
            </div>
          </div>
        )
      })}
      <ScansPanel />
    </Sheet>
  )
}

function NewRoomSheet() {
  const units = useStore((s) => s.doc.apartment.units)
  const count = useStore((s) => s.doc.rooms.length)
  const [name, setName] = useState(count ? `Room ${count + 1}` : 'Living room')
  const [w, setW] = useState(units === 'metric' ? '400' : `13'`)
  const [d, setD] = useState(units === 'metric' ? '350' : `11'`)
  const pw = parseLength(w, units)
  const pd = parseLength(d, units)
  const ok = !!pw && !!pd && pw > 0.3 && pd > 0.3
  const u = unitLabel(units)
  return (
    <Sheet title="Add a room" sub="Inside size, wall to wall. You can drag its walls later.">
      <form
        className="hud-form"
        onSubmit={(e) => {
          e.preventDefault()
          if (!ok) return
          useStore.getState().addRect(name || 'Room', pw!, pd!)
          useGame.getState().openSheet(null)
        }}
      >
        <label className="g-field">
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <div className="hud-form-row">
          <label className="g-field">
            Width {u && `(${u})`}
            <input value={w} inputMode="decimal" onChange={(e) => setW(e.target.value)} />
          </label>
          <label className="g-field">
            Depth {u && `(${u})`}
            <input value={d} inputMode="decimal" onChange={(e) => setD(e.target.value)} />
          </label>
        </div>
        <div className="hud-chips">
          {['Living room', 'Bedroom', 'Kitchen', 'Office', 'Dining room', 'Bathroom'].map((n) => (
            <button type="button" key={n} className={`g-chip${name === n ? ' on' : ''}`} onClick={() => setName(n)}>
              {n}
            </button>
          ))}
        </div>
        <button type="submit" className="g-btn primary big" disabled={!ok}>
          Build it
        </button>
      </form>
    </Sheet>
  )
}

function DesignsSheet() {
  const doc = useStore((s) => s.doc)
  const s = useStore.getState()
  const [editing, setEditing] = useState<string | null>(null)
  return (
    <Sheet title={doc.apartment.name} sub="Try different designs for the same home and flip between them.">
      <label className="g-field hud-homename">
        Home name
        <input value={doc.apartment.name} onChange={(e) => s.renameApartment(e.target.value)} />
      </label>
      <div className="hud-designs">
        {doc.layouts.map((l) => {
          const active = l.id === doc.activeLayoutId
          return (
            <div key={l.id} className={`hud-design g-card${active ? ' on' : ''}`}>
              <button className="hud-design-main" onClick={() => s.setActiveLayout(l.id)}>
                {editing === l.id ? (
                  <input
                    autoFocus
                    defaultValue={l.name}
                    onClick={(e) => e.stopPropagation()}
                    onBlur={(e) => {
                      s.renameLayout(l.id, e.target.value || l.name)
                      setEditing(null)
                    }}
                    onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                  />
                ) : (
                  <strong>{l.name}</strong>
                )}
                <small>
                  {l.placements.length} piece{l.placements.length === 1 ? '' : 's'}
                  {active && ' · playing'}
                </small>
              </button>
              <div className="hud-design-actions">
                <button className="g-icon sm" aria-label="Rename" onClick={() => setEditing(l.id)}>
                  <I.IconPen />
                </button>
                <button className="g-icon sm" aria-label="Copy design" onClick={() => s.duplicateLayout(l.id)}>
                  <I.IconCopy />
                </button>
                {doc.layouts.length > 1 && (
                  <button className="g-icon sm danger" aria-label="Delete design" onClick={() => s.deleteLayout(l.id)}>
                    <I.IconTrash />
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>
      <button className="g-btn primary" onClick={() => s.addLayout()}>
        <I.IconPlus width={18} height={18} /> New empty design
      </button>
    </Sheet>
  )
}

function SettingsSheet() {
  const units = useStore((s) => s.doc.apartment.units)
  const snap = useStore((s) => s.snap)
  const showClearance = useStore((s) => s.showClearance)
  const showPath = useStore((s) => s.showPath)
  const scan3d = useStore((s) => s.scan3d)
  const hasScans = useStore((s) => (s.doc.scans ?? []).length > 0)
  const cam = useCam()
  const s = useStore.getState()
  const g = useGame.getState()
  const fileRef = useRef<HTMLInputElement>(null)
  return (
    <Sheet title="Settings">
      <div className="hud-settings">
        <div className="g-toggle">
          Units
          <span className="hud-seg">
            <button className={units === 'imperial' ? 'on' : ''} onClick={() => s.setUnits('imperial')}>
              ft / in
            </button>
            <button className={units === 'metric' ? 'on' : ''} onClick={() => s.setUnits('metric')}>
              cm / m
            </button>
          </span>
        </div>
        <label className="g-toggle">
          Show space each piece needs
          <input type="checkbox" checked={showClearance} onChange={(e) => s.setUI({ showClearance: e.target.checked })} />
        </label>
        <label className="g-toggle">
          Show walking paths
          <input type="checkbox" checked={showPath} onChange={(e) => s.setUI({ showPath: e.target.checked })} />
        </label>
        <label className="g-toggle">
          Snap furniture to walls
          <input type="checkbox" checked={snap.walls} onChange={(e) => s.setSnap({ walls: e.target.checked })} />
        </label>
        <label className="g-toggle">
          Line up with other furniture
          <input type="checkbox" checked={snap.items} onChange={(e) => s.setSnap({ items: e.target.checked })} />
        </label>
        <label className="g-toggle">
          Snap to a 5 cm grid
          <input type="checkbox" checked={snap.grid > 0} onChange={(e) => s.setSnap({ grid: e.target.checked ? 0.05 : 0 })} />
        </label>
        <div className="g-toggle">
          Walls in 3D
          <span className="hud-seg">
            {(['auto', 'low', 'full'] as WallMode[]).map((w) => (
              <button key={w} className={cam.wallMode === w ? 'on' : ''} onClick={() => setCam({ wallMode: w })}>
                {w === 'auto' ? 'Cutaway' : w === 'low' ? 'Low' : 'Full'}
              </button>
            ))}
          </span>
        </div>
        {hasScans && (
          <div className="g-toggle">
            Scan in 3D
            <span className="hud-seg">
              {(
                [
                  ['with-model', 'Both'],
                  ['scan-only', 'Scan'],
                  ['off', 'Off'],
                ] as const
              ).map(([v, l]) => (
                <button key={v} className={scan3d === v ? 'on' : ''} onClick={() => s.setUI({ scan3d: v })}>
                  {l}
                </button>
              ))}
            </span>
          </div>
        )}
      </div>
      <h3 className="hud-h3">Home</h3>
      <div className="hud-settings-btns">
        <button className="g-btn" onClick={() => g.setPhase('onboarding')}>
          <I.IconScan width={18} height={18} /> Move into a new place
        </button>
        <button
          className="g-btn"
          onClick={() => {
            loadSample()
            g.openSheet(null)
            s.notify('Loaded the sample home. Undo to go back.')
          }}
        >
          <I.IconHome width={18} height={18} /> Load the sample home
        </button>
        {!EMBEDDED && (
          <button className="g-btn" onClick={() => downloadJSON(useStore.getState().doc)}>
            Save to a file
          </button>
        )}
        <button className="g-btn" onClick={() => fileRef.current?.click()}>
          Open a saved file
        </button>
        <button className="g-btn ghost" onClick={() => g.setClassic(true)}>
          <I.IconLayers width={18} height={18} /> Classic editor (all the tools)
        </button>
      </div>
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
            g.openSheet(null)
          } catch {
            s.notify("That file isn't a saved home.")
          }
          e.target.value = ''
        }}
      />
    </Sheet>
  )
}
