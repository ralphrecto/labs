import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode, type RefObject } from 'react'
import App, { useScanFiles, useShortcuts } from '../App'
import { useAnalysis } from '../analysis'
import { Inspector } from '../components/Panels'
import { PlanView } from '../components/PlanView'
import { ScanReviewPanel } from '../components/ScanPanels'
import { ScansPanel } from '../components/ScanImport'
import { catalogEntry, itemFromCatalog } from '../model/catalog'
import { orientedRect, pointInPolygon, polygonArea, v } from '../model/geometry'
import type { Item } from '../model/types'
import { viewFrame } from '../three/viewFrame'
import { formatLength, parseLength, unitLabel } from '../model/units'
import { toScanData } from '../scan/align'
import { scanEntry } from '../scan/runtime'
import { useScanUI } from '../scan/scanUi'
import { activeLayout, openDocAsHome, useStore, type PlaceAt } from '../store'
import { DesignsSheetBody, RemixPicker } from './Designs'
import { downloadJSON, parseDoc } from '../storage'
import { camMoved, Scene3D, type SceneInset, type WallMode } from '../three/Scene3D'
import { BuyDrawer, ShopSheet, visibleGroups } from './BuyDrawer'
import { dropSpotAt, flushEditRun, rotatePlacementSmart } from './editActions'
import { isWallDecor } from '../model/wallDecor'
import { ItemThumb } from './ItemThumb'
import { GoalsSheet, LevelUpCard, VibeMeter } from './Goals'
import { unpackAndCelebrate } from './unpack'
import { useGame, type GameMode } from './gameStore'
import { familyOf } from './swatches'
import * as I from './icons'
import { Onboarding } from './Onboarding'
import { PhotoShutter } from './PhotoShutter'
import { StylePicker, useStylePicker, useStylesUnlocked } from './StylePicker'
import { useModal } from './useModal'
import { announce, LiveRegion } from './announce'
import { reduceMotion } from '../motion'
import { usePageZoomGuard } from '../gestureGuard'
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
  const gesturing = useStore((s) => s.gesture)

  // Esc in the game also closes sheets.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && useGame.getState().sheet) useGame.getState().openSheet(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const pickingStyle = useStylePicker((s) => s.open)
  const stylesOk = useStylesUnlocked()
  // Phones show a compact drawer (one row of small thumbnails); tapping the open tab again
  // expands it to the full cards and back (hud.css). The camera isn't refitted for it.
  const [drawerExpanded, setDrawerExpanded] = useState(false)
  const tool = storeMode !== 'select'
  // In Furnish the drawer stays open while a piece is selected (its actions ride in a slim strip
  // above it), so placing piece after piece needs no "Done" in between.
  const showDrawer = phase === 'play' && mode === 'buy' && !tool && !review
  const view = useStageView()
  const cam = useCam()
  const gameRef = useRef<HTMLDivElement>(null)
  // No pinch zooms the page anywhere in the game, iOS Safari's gesture events included (theme.css).
  usePageZoomGuard(gameRef)
  useBackFromDreamPhoto(gameRef)
  // The 3D view frames the home around the HUD only when the screen really changes (mode, view,
  // recenter, scan review, window size), never as the drawer and selection bar come and go.
  // On a phone, the compact drawer leaves a small stage: frame the selected piece's room instead
  // of the whole home (and the whole home again on deselect). Desktop never refits on select.
  const focusRoom = useFocusRoom(showDrawer && view === '3d', selection?.kind === 'placement' ? selection.id : null)
  const inset = useHudInset(gameRef, `${phase}:${mode}:${view}:${cam.resetKey}:${cam.eye}:${!!review}:${focusRoom}`)
  useGameShortcuts()
  const homeName = useStore((s) => s.doc.apartment.name)
  return (
    <div ref={gameRef} className={`game mode-${mode}${tool ? ' tool-active' : ''}${gesturing ? ' gesturing' : ''}`}>
      <LiveRegion />
      {/* The top bar is the page's banner, so it sits outside main (the order on screen is CSS's). */}
      {phase !== 'onboarding' && mode !== 'look' && <TopBar />}
      {/* The scene and the HUD that works on it; display: contents, so the layout is unchanged. */}
      <main className="game-main">
      <h1 className="sr-only">{homeName || 'Homebody'}</h1>
      <Stage inset={inset} focusRoom={focusRoom} />
      {phase === 'onboarding' ? (
        <Onboarding />
      ) : (
        <>
          {mode === 'look' && <LookControls />}
          {mode === 'look' && <PhotoShutter />}
          {mode === 'look' && stylesOk && <StylePicker className="sty-look" />}
          <CameraControls />
          {showDrawer && (
            // The drawer steps aside while the style picker is open, so on a phone the style tiles
            // and their names are never under it. Hidden rather than unmounted, so it comes back as it was.
            <div
              className={`hud-drawer-slot${pickingStyle ? ' tucked' : ''}${drawerExpanded ? ' expanded' : ''}`}
              onClickCapture={(e) => {
                if (!(e.target instanceof Element)) return
                // With a piece selected, a tab is the way back to shopping: it lets the piece go.
                if (e.target.closest('.buy-tab') && useStore.getState().selection) useStore.getState().select(null)
                else if (e.target.closest('.buy-tab.on')) setDrawerExpanded((x) => !x)
              }}
            >
              <BuyDrawer top={selection ? <ContextBar slim /> : null} />
            </div>
          )}
          {mode === 'build' && !selection && !review && !tool && <BuildTray />}
          {tool && <ToolBanner />}
          {selection && !tool && mode !== 'look' && !showDrawer && <ContextBar />}
          <ShortcutHelp />
          {review && (
            <div className="hud-review g-card">
              <ScanReviewPanel />
            </div>
          )}
          <Dock />
          <div className="hud-toast">
            <GameToast />
          </div>
          <SaveFailedToast />
          <LevelUpCard />
        </>
      )}
      </main>
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

const PHONE_QUERY = '(max-width: 700px) and (orientation: portrait)'
function usePhone() {
  const [phone, setPhone] = useState(() => typeof window !== 'undefined' && !!window.matchMedia?.(PHONE_QUERY).matches)
  useEffect(() => {
    const mq = window.matchMedia?.(PHONE_QUERY)
    if (!mq) return
    const on = () => setPhone(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return phone
}

/** The room of the selected piece while a phone furnishes with the compact drawer, else null. */
function useFocusRoom(drawer: boolean, placementId: string | null): string | null {
  const phone = usePhone()
  return useMemo(() => {
    if (!phone || !drawer || !placementId) return null
    // Read once at selection: dragging the piece into the next room doesn't swing the camera.
    const doc = useStore.getState().doc
    const p = activeLayout(doc).placements.find((x) => x.id === placementId)
    return (p && doc.rooms.find((r) => pointInPolygon(v(p.x, p.y), r.polygon))?.id) ?? null
  }, [phone, drawer, placementId])
}

/** The drawer card being dragged (its data can't be read during dragover, so it's noted at dragstart). */
type Dragged = { itemId: string; cat: string }
function draggedItem(d: Dragged): Item | null {
  if (d.itemId) return useStore.getState().doc.items.find((i) => i.id === d.itemId) ?? null
  return d.cat ? itemFromCatalog(catalogEntry(d.cat), 'drop-ghost') : null
}
/** Where a drawer card dropped at a screen point lands, or null to fall back to smart placement. */
function dropAt(item: Item, clientX: number, clientY: number): PlaceAt | null {
  const pt = viewFrame.toFloor?.(clientX, clientY)
  return pt ? dropSpotAt(item, pt) : null
}
/** The landing footprint's corners in stage pixels. */
function ghostPoints(item: Item, at: PlaceAt, stage: HTMLElement | null) {
  const toScreen = viewFrame.toScreen
  if (!toScreen || !stage) return null
  const doc = useStore.getState().doc
  const host = at.on ? activeLayout(doc).placements.find((p) => p.id === at.on) : undefined
  const h = host ? (doc.items.find((i) => i.id === host.itemId)?.height ?? 0) : 0
  const r = stage.getBoundingClientRect()
  const pts = orientedRect(v(at.x, at.y), item.width / 2, item.depth / 2, at.rotation ?? 0).map((c) => toScreen(c, h))
  return pts.every((q) => q) ? pts.map((q) => ({ x: q!.x - r.left, y: q!.y - r.top })) : null
}

function Stage({ inset, focusRoom }: { inset: SceneInset; focusRoom: string | null }) {
  const stageRef = useRef<HTMLDivElement>(null)
  const dragged = useRef<Dragged | null>(null)
  const [ghost, setGhost] = useState<{ x: number; y: number }[] | null>(null)
  useEffect(() => {
    // The drawer sets the card's data in its own dragstart handler, which runs before this one.
    const start = (e: DragEvent) => {
      const dt = e.dataTransfer
      dragged.current = dt ? { itemId: dt.getData('application/x-apt-item'), cat: dt.getData('application/x-apt-catalog') } : null
    }
    const end = () => {
      dragged.current = null
      setGhost(null)
    }
    window.addEventListener('dragstart', start)
    window.addEventListener('dragend', end)
    return () => {
      window.removeEventListener('dragstart', start)
      window.removeEventListener('dragend', end)
    }
  }, [])
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
      fullFrame: mode === 'look',
      focusRoom: mode === 'buy' ? focusRoom : null,
    }),
    [cam, mode, style, inset, focusRoom],
  )
  return (
    <div
      ref={stageRef}
      className={`hud-stage view-${view}`}
      onDragOver={(e) => {
        if (!e.dataTransfer.types.some((t) => t.startsWith('application/x-apt-'))) return
        e.preventDefault()
        e.dataTransfer.dropEffect = 'copy'
        const d = dragged.current
        const item = d && draggedItem(d)
        const spot = item ? dropAt(item, e.clientX, e.clientY) : null
        setGhost(item && spot ? ghostPoints(item, spot, stageRef.current) : null)
      }}
      onDragLeave={(e) => {
        if (!(e.relatedTarget instanceof Node && e.currentTarget.contains(e.relatedTarget))) setGhost(null)
      }}
      onDrop={(e) => {
        const itemId = e.dataTransfer.getData('application/x-apt-item')
        const cat = e.dataTransfer.getData('application/x-apt-catalog')
        setGhost(null)
        if (!itemId && !cat) return
        e.preventDefault()
        const st = useStore.getState()
        const item = draggedItem({ itemId, cat })
        // Dropped on a room: it lands under the pointer (snapped, nudged clear). Elsewhere it goes
        // to a spot that suits it, as a tap on the card does.
        const at = item ? dropAt(item, e.clientX, e.clientY) : null
        const res = itemId ? st.placeItem(itemId, at ?? undefined) : st.addItemFromCatalog(catalogEntry(cat), at ?? undefined)
        if (res.ok && item) {
          const doc = useStore.getState().doc
          const p = activeLayout(doc).placements.find((x) => x.id === res.placementId)
          const room = res.roomName ?? (p && doc.rooms.find((r) => pointInPolygon(v(p.x, p.y), r.polygon))?.name)
          st.notify(`Placed ${itemId ? `“${item.name}”` : `${/^[aeiou]/i.test(item.name) ? 'an' : 'a'} ${item.name.toLowerCase()}`}${room ? ` in the ${room}` : ''}.`)
        }
      }}
    >
      {phase === 'onboarding' ? null : view === '3d' ? <Scene3D control={control} /> : <PlanView inset={inset} />}
      {phase === 'play' && mode !== 'look' && <SceneItems />}
      {ghost && (
        <svg className="hud-drop-ghost" aria-hidden>
          <polygon points={ghost.map((q) => `${q.x},${q.y}`).join(' ')} />
        </svg>
      )}
    </div>
  )
}

/** HUD pieces along the bottom and the top of the screen that cover the stage. */
const BOTTOM_HUD = ['.buy-drawer', '.hud-ctx-wrap', '.hud-tray', '.hud-dock', '.hud-review']
/** After a framing change, how long the HUD may settle (pieces mounting) before the inset is held. */
const INSET_SETTLE_MS = 700
/** How long after the window closes a late HUD change (fonts, drawer layout) may still be measured. */
const LATE_SETTLE_MS = 1500
const TOP_HUD = ['.hud-top', '.goal-next', '.hud-look-bar']
/** When the furniture drawer is a right-hand column (keep in step with buy.css). */
const DESKTOP_SIDE_DRAWER = '(min-width: 1100px) and (min-height: 700px), (min-width: 1000px) and (min-height: 640px) and (orientation: landscape)'
/** The slim selection bar above the drawer: its height and gap (hud.css). */
const CTX_BAR_H = 52
/** Height kept for the slim selection strip at the top left of the stage beside the side drawer. */
const SIDE_CTX_H = 56

/**
 * How much of the stage the HUD covers at the top and bottom (CSS px), so the 3D view can centre
 * the home in what's left. It's measured when `frameKey` changes or the window resizes, and held
 * after the HUD settles: a selection bar or drawer coming and going doesn't move the camera.
 */
function useHudInset(root: RefObject<HTMLDivElement | null>, frameKey: string): SceneInset {
  const [inset, setInset] = useState<SceneInset>({ top: 0, bottom: 0 })
  const openUntil = useRef(0)
  const remeasure = useRef<() => void>(() => {})
  useLayoutEffect(() => {
    openUntil.current = performance.now() + INSET_SETTLE_MS
    remeasure.current()
  }, [frameKey])
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
      // Laptops and desktops (buy.css): the drawer is a column down the right, and the selection
      // strip rides at the top left of the stage, so they count at the right and the top instead.
      const g = game.getBoundingClientRect()
      const side = window.matchMedia(DESKTOP_SIDE_DRAWER).matches
      const drawer = side ? game.querySelector<HTMLElement>('.hud-drawer-slot:not(.tucked) .buy-drawer') : null
      for (const sel of BOTTOM_HUD) {
        const el = game.querySelector<HTMLElement>(sel)
        if (!el || (drawer && (el === drawer || drawer.contains(el)))) continue
        // Side panels (the scan review on wide screens) don't cover the middle of the stage;
        // centred bars (dock, selection bar) do, however narrow they are.
        const r = el.getBoundingClientRect()
        const mid = g.left + W / 2
        if (r.right < mid - W * 0.1 || r.left > mid + W * 0.1) continue
        bottom = Math.max(bottom, H - offsetTopIn(el, game))
      }
      for (const sel of TOP_HUD) {
        const el = game.querySelector<HTMLElement>(sel)
        if (el) top = Math.max(top, offsetTopIn(el, game) + el.offsetHeight)
      }
      // On a phone the camera buttons run down the right edge of the stage: frame the home beside them.
      let right = 0
      const camEl = game.querySelector<HTMLElement>('.hud-cam')
      if (camEl && W <= 700 && H > W) right = Math.max(0, W - (camEl.getBoundingClientRect().left - g.left) + 4)
      if (drawer) {
        right = Math.max(right, W - (drawer.getBoundingClientRect().left - g.left))
        // Room for the selection strip at the top left, whether or not something is selected now.
        top = Math.max(top, offsetTopIn(drawer, game) + SIDE_CTX_H)
      }
      // With the drawer along the bottom, a selection bar will ride above it: the plan keeps that clear too.
      const ctx = !side && game.querySelector('.hud-drawer-slot:not(.tucked) .buy-drawer') && !game.querySelector('.hud-ctx-wrap') ? CTX_BAR_H : 0
      if (performance.now() > openUntil.current) return
      setInset((was) =>
        Math.abs(was.top - top) < 4 && Math.abs(was.bottom - bottom) < 4 && Math.abs((was.right ?? 0) - right) < 4 && (was.ctx ?? 0) === ctx ? was : { top, bottom, right, ctx },
      )
    }
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(measure)
    }
    // A resize event only reopens the measuring window when the stage's size really changed
    // (mobile browsers and devtools fire plain resizes that would otherwise reframe the camera).
    let size = `${game.clientWidth}x${game.clientHeight}`
    const onResize = () => {
      const now = `${game.clientWidth}x${game.clientHeight}`
      if (now === size) return
      size = now
      openUntil.current = performance.now() + INSET_SETTLE_MS
      schedule()
    }
    // The HUD can settle after the window closes: web fonts loading late, or the drawer still
    // laying out. Measure again then (shortly after a framing change only, so a drawer that opens
    // or grows later still doesn't move the camera).
    const reopen = () => {
      if (performance.now() > openUntil.current + LATE_SETTLE_MS) return
      openUntil.current = Math.max(openUntil.current, performance.now() + 200)
      schedule()
    }
    let live = true
    document.fonts?.ready.then(() => live && reopen())
    const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(reopen)
    let watched: Element | null = null
    const watchDrawer = () => {
      const el = game.querySelector('.buy-drawer')
      if (el === watched || !ro) return
      if (watched) ro.unobserve(watched)
      watched = el
      if (el) ro.observe(el)
    }
    remeasure.current = schedule
    measure()
    watchDrawer()
    const mo = new MutationObserver(() => {
      watchDrawer()
      schedule()
    })
    mo.observe(game, { childList: true, subtree: true })
    window.addEventListener('resize', onResize)
    return () => {
      live = false
      mo.disconnect()
      ro?.disconnect()
      window.removeEventListener('resize', onResize)
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
  // Read out through the page's one live region (announce.tsx); the toast itself stays quiet.
  useEffect(() => announce(toast), [toast])
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
    <div className="toast" aria-hidden="true">
      {text}
    </div>
  )
}

/**
 * The placed pieces as a keyboard list beside the canvas (visually hidden until focused): Tab
 * walks through them, focusing one selects it, and then the editing keys work on it (arrows nudge,
 * R turns, Delete removes, Enter or Esc lets go).
 */
function SceneItems() {
  const doc = useStore((s) => s.doc)
  const selection = useStore((s) => s.selection)
  const layout = activeLayout(doc)
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const rows = layout.placements.flatMap((p) => {
    const item = items.get(p.itemId)
    if (!item) return []
    const room = doc.rooms.find((r) => pointInPolygon(v(p.x, p.y), r.polygon))?.name
    return [{ id: p.id, label: `${item.name}${room ? `, ${room}` : ''}` }]
  })
  if (!rows.length) return null
  const select = (id: string) => {
    const s = useStore.getState()
    if (!(s.selection?.kind === 'placement' && s.selection.id === id)) s.select({ kind: 'placement', id })
  }
  return (
    <ul className="scene-items" aria-label={`Pieces in the home, ${rows.length}. Focus one to select it; arrows move, R turns, Delete removes`}>
      {rows.map((r) => (
        <li key={r.id}>
          <button type="button" aria-pressed={selection?.kind === 'placement' && selection.id === r.id} onFocus={() => select(r.id)} onClick={() => select(r.id)}>
            {r.label}
          </button>
        </li>
      ))}
    </ul>
  )
}

/**
 * A save that didn't reach storage: says so, and stays up (the store retries on every change)
 * until a save goes through. The backup is the same JSON file as Settings' download.
 */
function SaveFailedToast() {
  const failed = useStore((s) => s.saveFailed)
  useEffect(() => {
    if (failed) announce("Couldn't save. Your device storage is full. A backup can be downloaded.", 'assertive')
  }, [failed])
  if (!failed) return null
  return (
    <div className="dz-save-failed g-card">
      <span>Couldn't save. Your device storage is full.</span>
      <button className="g-btn sm primary" onClick={() => downloadJSON(useStore.getState().doc)}>
        Download a backup
      </button>
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
  // Remix just unlocked (round 16): Your designs, where it lives, glows "new" until it's used.
  const freshRemix = useGame((g) => g.freshUnlocks.includes('remix'))
  return (
    <header className="hud-top">
      <button className={`hud-home${freshRemix ? ' g-new' : ''}`} onClick={() => open('layouts')} aria-label={freshRemix ? 'Your designs (new: Remix)' : 'Your designs'}>
        <span className="dz-hud-icon-col">
          <span className="hud-home-icon">
            <I.IconDesigns />
          </span>
          {/* Phones hide the name beside the icon; the design's name sits under it instead. */}
          <small className="dz-hud-design" aria-hidden>
            {layoutName}
          </small>
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
          <button className="g-icon sm" disabled={!canUndo} onClick={() => (flushEditRun(), s.undo())} aria-label="Undo" title="Undo (Ctrl+Z)">
            <I.IconUndo />
          </button>
          <button className="g-icon sm" disabled={!canRedo} onClick={() => (flushEditRun(), s.redo())} aria-label="Redo" title="Redo (Ctrl+Shift+Z)">
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

const MODES: { m: GameMode; label: string; icon: ReactNode; key: string }[] = [
  { m: 'build', label: 'Build', icon: <I.IconHammer />, key: 'B' },
  { m: 'buy', label: 'Furnish', icon: <I.IconSofa />, key: 'F' },
  { m: 'look', label: 'Look', icon: <I.IconEye />, key: 'L' },
]

function switchMode(m: GameMode) {
  const st = useStore.getState()
  if (st.mode !== 'select') st.setMode('select')
  st.select(null)
  useGame.getState().setMode(m)
  if (m === 'look') setCam({ autoRotate: !reduceMotion, eye: false, resetKey: camState.resetKey + 1 })
}

function Dock() {
  const mode = useGame((s) => s.mode)
  return (
    <nav className="hud-dock" aria-label="Game mode">
      {MODES.map(({ m, label, icon, key }) => (
        <button
          key={m}
          className={`hud-mode hud-mode-${m}${mode === m ? ' on' : ''}`}
          onClick={() => switchMode(m)}
          title={`${label} (${key})`}
          aria-pressed={mode === m}
        >
          <span className="hud-mode-icon">{icon}</span>
          <span className="hud-mode-label">{label}</span>
          <kbd className="hud-mode-key" aria-hidden="true">
            {key}
          </kbd>
        </button>
      ))}
    </nav>
  )
}

const CAM_LABELS_KEY = 'apartment-designer:cam-labels-seen'

/**
 * Names next to the camera buttons on a first visit, until one is used or a few seconds pass. On
 * phones they are small captions under the buttons instead (hud.css), clear of the Next chip.
 */
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

function subscribeCamMoved(l: () => void) {
  camMoved.listeners.add(l)
  return () => {
    camMoved.listeners.delete(l)
  }
}

function CameraControls() {
  const view = useStageView()
  const mode = useGame((s) => s.mode)
  const tool = useStore((s) => s.mode !== 'select')
  const [labels, seen] = useFirstRunLabels()
  const stylesOk = useStylesUnlocked()
  const moved = useSyncExternalStore(subscribeCamMoved, () => camMoved.value)
  if (mode === 'look' || tool) return null
  // One View button: it swaps 3D and the plan, but once the player has dragged the 3D view away
  // from its framing, the first tap brings it back (a crosshair), and the next one goes to the plan.
  const recenterFirst = view === '3d' && moved
  const viewLabel = recenterFirst ? 'Recenter the view' : view === '3d' ? 'Top-down plan' : '3D view'
  return (
    <div className={`hud-cam${labels ? ' labelled' : ''}`} onPointerDown={labels ? seen : undefined}>
      <div className="hud-cam-row">
        {labels && <span className="hud-cam-label">{recenterFirst ? 'Recenter' : view === '3d' ? 'Plan' : '3D'}</span>}
        <button
          className="g-icon sm hud-view"
          onClick={() => (recenterFirst ? setCam({ resetKey: camState.resetKey + 1 }) : useGame.getState().setCamera(view === '3d' ? 'plan' : '3d'))}
          aria-label={viewLabel}
          title={`${viewLabel} (${recenterFirst ? 'Home' : 'P'})`}
        >
          {recenterFirst ? <I.IconCrosshair /> : view === '3d' ? <I.IconMap /> : <I.IconCube />}
        </button>
      </div>
      {view === '3d' && stylesOk && (
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

/** Frame finishes for framed wall pieces (wall art, mirrors): written to item.frameColor. */
const FRAME_FINISHES = [
  { name: 'Black', color: '#2b2a28' },
  { name: 'White', color: '#f2efe8' },
  { name: 'Oak', color: '#b88a5a' },
  { name: 'Brass', color: '#c9a24a' },
]
const FRAMED = new Set(['wall-art', 'mirror'])
const SWATCHES = ['#7d8fa3', '#a3907d', '#c9c3b6', '#8a6b4f', '#3d3d3d', '#5f8f5a', '#c7a98a', '#e2d6b8', '#b5534a', '#e0b04f', '#5b7bb5', '#f2efe8']

function ContextBar({ slim = false }: { slim?: boolean }) {
  const sel = useStore((s) => s.selection)!
  const doc = useStore((s) => s.doc)
  const layout = activeLayout(doc)
  const { issues } = useAnalysis(doc, layout)
  const [paint, setPaint] = useState(false)
  const s = useStore.getState()
  const open = useGame((g) => g.openSheet)
  const units = doc.apartment.units
  useEffect(() => setPaint(false), [sel])
  const selPlacement = sel.kind === 'placement' ? layout.placements.find((x) => x.id === sel.id) : undefined
  const selItem = selPlacement && doc.items.find((i) => i.id === selPlacement.itemId)
  const framedItem = selItem && FRAMED.has(selItem.category) ? selItem : undefined

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
    // A packed moving box: Unpack comes first; copying or painting a box is beside the point.
    const packed = !!item.unpack?.length && !item.unpacked
    actions = (
      <>
        {packed && (
          <Act label="Unpack" filled onClick={() => unpackAndCelebrate(p.id)}>
            <I.IconBox />
          </Act>
        )}
        {/* Wall decor always faces into the room: nothing to turn. */}
        {!isWallDecor(item?.category) && (
          <Act label="Rotate" tip="Rotate 90°" keys="R" onClick={() => rotatePlacementSmart(p.id, Math.PI / 2)}>
            <I.IconRotate />
          </Act>
        )}
        {!packed && (
          <Act label="Copy" keys="Ctrl+D" onClick={() => s.duplicatePlacement(p.id)}>
            <I.IconCopy />
          </Act>
        )}
        {!packed && (
          <Act label="Color" on={paint} onClick={() => setPaint(!paint)}>
            <I.IconPalette />
          </Act>
        )}
        <Act label="Details" onClick={() => open('details')}>
          <I.IconMore />
        </Act>
        <Act label="Remove" keys="Del" danger onClick={() => s.deletePlacement(p.id)}>
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
    <div className={`hud-ctx-wrap${slim ? ' slim' : ''}`}>
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
                if (!p) return
                s.updateItem(p.itemId, { color: c })
                // Painting teaches the drawer the player's taste: the next picks lead with this colour.
                const fam = familyOf(doc.items.find((i) => i.id === p.itemId)?.category ?? '')
                if (fam) useGame.getState().setTaste(fam, c)
              }}
            />
          ))}
          {framedItem && (
            <div className="hud-paint-frames" role="group" aria-label="Frame finish" style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', gap: 8, paddingTop: 4 }}>
              <small style={{ flex: 1, fontWeight: 600, opacity: 0.75 }}>Frame</small>
              {FRAME_FINISHES.map((f) => (
                <button
                  key={f.name}
                  style={{ background: f.color, borderRadius: 8, outline: framedItem.frameColor === f.color ? '2px solid #e0603a' : undefined }}
                  aria-label={`${f.name} frame`}
                  aria-pressed={framedItem.frameColor === f.color}
                  title={`${f.name} frame`}
                  onClick={() => s.updateItem(framedItem.id, { frameColor: f.color })}
                />
              ))}
            </div>
          )}
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
        {!slim && (
          <button className="hud-ctx-done" onClick={() => s.select(null)} aria-label="Done" title="Done (Enter)">
            <I.IconCheck />
          </button>
        )}
      </div>
    </div>
  )
}

function Act({ label, tip, keys, onClick, children, danger, on, filled }: { label: string; tip?: string; keys?: string; onClick: () => void; children: ReactNode; danger?: boolean; on?: boolean; filled?: boolean }) {
  return (
    <button className={`hud-act${danger ? ' danger' : ''}${on ? ' on' : ''}${filled ? ' filled' : ''}`} onClick={onClick} aria-label={label} title={keys ? `${tip ?? label} (${keys})` : (tip ?? label)}>
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
      <Inspector game />
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
                className={`g-btn sm ${s.doc.rooms.some((rm) => rm.scan?.scanId === r.id) || s.doc.items.some((it) => it.scanId === r.id) ? 'ghost' : 'primary'}`}
                disabled={!loaded || !!busy}
                onClick={() => {
                  if (!loaded) return
                  close()
                  void useScanUI.getState().analyze(toScanData(r, loaded.points))
                }}
              >
                <I.IconSparkle width={16} height={16} />{' '}
                {s.doc.rooms.some((rm) => rm.scan?.scanId === r.id) || s.doc.items.some((it) => it.scanId === r.id) ? 'Find again' : 'Find furniture'}
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
  const name = useStore((s) => s.doc.apartment.name)
  const s = useStore.getState()
  return (
    <Sheet title={name} sub="Try different designs for the same home and flip between them.">
      <DesignsSheetBody />
      <div className="dz-actions">
        <RemixPicker />
        <button className="g-btn ghost" onClick={() => s.addLayout()}>
          <I.IconPlus width={18} height={18} /> New empty design
        </button>
      </div>
      <HomeFileLinks />
    </Sheet>
  )
}

/** Whether the URL asks for the Classic editor door (`?classic`): the full tool kit stays out of the game's way. */
const CLASSIC_DOOR = typeof location !== 'undefined' && new URLSearchParams(location.search).has('classic')
/** A mouse and keyboard: the shortcut list is worth offering. */
const HAS_KEYBOARD = typeof matchMedia !== 'undefined' && matchMedia('(hover: hover) and (pointer: fine)').matches

function SettingsSheet() {
  const units = useStore((s) => s.doc.apartment.units)
  const scan3d = useStore((s) => s.scan3d)
  const hasScans = useStore((s) => (s.doc.scans ?? []).length > 0)
  const s = useStore.getState()
  const g = useGame.getState()
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
        {HAS_KEYBOARD && (
          <button
            className="g-toggle hud-settings-row"
            onClick={() => {
              g.openSheet(null)
              useShortcutHelp.set(true)
            }}
          >
            Keyboard shortcuts <kbd>?</kbd>
          </button>
        )}
      </div>
      {CLASSIC_DOOR && (
        <div className="hud-settings-btns">
          <button className="g-btn ghost" onClick={() => g.setClassic(true)}>
            <I.IconLayers width={18} height={18} /> Classic editor (all the tools)
          </button>
        </div>
      )}
    </Sheet>
  )
}

/** Save the home to a file, or open one: two quiet links at the foot of Your designs. */
function HomeFileLinks() {
  const fileRef = useRef<HTMLInputElement>(null)
  const s = useStore.getState()
  const g = useGame.getState()
  return (
    <div className="dz-file-links">
      {!EMBEDDED && (
        <button className="g-link" onClick={() => downloadJSON(useStore.getState().doc)}>
          Save to a file
        </button>
      )}
      <button className="g-link" onClick={() => fileRef.current?.click()}>
        Open a saved file
      </button>
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
            openDocAsHome(d)
            g.openSheet(null)
            s.notify(`Opened ${d.apartment.name}. Your other homes are in Your designs.`)
          } catch {
            s.notify("That file isn't a saved home.")
          }
          e.target.value = ''
        }}
      />
    </div>
  )
}

// ---------------------------------------------------------------- keyboard

const SHORTCUTS: [string, string][] = [
  ['B / F / L', 'Build, Furnish, Look'],
  ['P', 'Top-down plan or 3D'],
  ['Home', 'Recenter the view'],
  ['1 to 6', 'Furniture drawer tabs'],
  ['R / Shift+R', 'Turn 90°'],
  [', / .', 'Turn 15°'],
  ['Arrows', 'Nudge 1 cm (Shift: 10 cm)'],
  ['Ctrl+D', 'Copy'],
  ['Del', 'Remove'],
  ['Drag floor', 'Orbit'],
  ['Right-drag', 'Pan'],
  ['Wheel', 'Zoom'],
  ['Esc', 'Close / deselect'],
  ['Enter', 'Done'],
  ['Ctrl+Z / Ctrl+Shift+Z', 'Undo, redo'],
  ['?', 'This list'],
]

const useShortcutHelp = createFlag()

/** Game keys on top of the editing ones in useShortcuts: modes, camera, drawer tabs, Done, help. */
function useGameShortcuts() {
  useEffect(() => {
    // A click anywhere ends a run of nudges, so it stays its own undo step.
    const onDown = () => flushEditRun()
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const g = useGame.getState()
      const st = useStore.getState()
      if (g.phase !== 'play' || g.sheet) return
      const k = e.key
      if (k === '?') {
        useShortcutHelp.toggle()
        return
      }
      if (k === 'Escape' && useShortcutHelp.get()) return useShortcutHelp.set(false)
      // Esc leaves Look for Furnish (the dock has no Done of its own there).
      if (k === 'Escape' && g.mode === 'look') return switchMode('buy')
      if (st.mode !== 'select') return
      const lower = k.toLowerCase()
      const mode = MODES.find((m) => m.key.toLowerCase() === lower)
      if (mode) switchMode(mode.m)
      else if (lower === 'p' && g.mode !== 'look') g.setCamera(g.camera === '3d' ? 'plan' : '3d')
      else if (k === 'Home') {
        e.preventDefault()
        setCam({ resetKey: camState.resetKey + 1 })
      } else if (/^[1-7]$/.test(k) && g.mode === 'buy') {
        const tab = visibleGroups(st.doc.items, g.buyTab)[Number(k) - 1]
        if (tab) g.setBuyTab(tab.key)
      }
      else if (k === 'Enter' && st.selection && t.tagName !== 'BUTTON' && t.tagName !== 'A') st.select(null)
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown, true)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown, true)
    }
  }, [])
}

/** A tiny shared on/off flag with a hook, for the shortcut list. */
function createFlag() {
  let on = false
  const subs = new Set<() => void>()
  const set = (v: boolean) => {
    on = v
    subs.forEach((f) => f())
  }
  function useFlag() {
    const [, force] = useState(0)
    useEffect(() => {
      const f = () => force((n) => n + 1)
      subs.add(f)
      return () => void subs.delete(f)
    }, [])
    return on
  }
  return Object.assign(useFlag, { get: () => on, set, toggle: () => set(!on) })
}

/** The list of keys, opened with "?" or from Settings (desktop). */
function ShortcutHelp() {
  const open = useShortcutHelp()
  return (
    <>
      {open && (
        <div className="hud-keys g-card" role="dialog" aria-label="Keyboard shortcuts" onClick={() => useShortcutHelp.set(false)}>
          <strong>Keyboard shortcuts</strong>
          <dl>
            {SHORTCUTS.map(([k, what]) => (
              <div key={k}>
                <dt>
                  <kbd>{k}</kbd>
                </dt>
                <dd>{what}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </>
  )
}

/**
 * Round 16: the Dream home's "Take its photo" ends in Look mode, which has no top bar and no chip,
 * so the game looked over. Once that photo's card is closed (Done, or a tap outside), the game goes
 * back to Furnish, where the Remix chip waits. Watches the card from outside PhotoShutter.
 */
function useBackFromDreamPhoto(ref: RefObject<HTMLDivElement | null>) {
  const photoRequest = useGame((s) => s.photoRequest)
  const armed = useRef(false)
  useEffect(() => {
    if (photoRequest) armed.current = true
  }, [photoRequest])
  useEffect(() => {
    const el = ref.current
    if (!el || typeof MutationObserver === 'undefined') return
    let seen = false
    const obs = new MutationObserver(() => {
      if (!armed.current) return
      const open = !!el.querySelector('.photo-sheet')
      if (open) seen = true
      else if (seen) {
        seen = false
        armed.current = false
        if (useGame.getState().mode === 'look') useGame.getState().setMode('buy')
      }
    })
    obs.observe(el, { childList: true, subtree: true })
    return () => obs.disconnect()
  }, [ref])
}
