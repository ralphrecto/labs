import { create } from 'zustand'
import type { RenderStyle } from '../three/style'

// Game-level UI state: which phase the player is in and which mode the HUD shows.
// Document edits still go through `useStore` (src/store.ts); this store never touches the doc.

/** onboarding: the first-run "move in" flow. play: the design game itself. */
export type Phase = 'onboarding' | 'play'
/** build: walls, rooms, doors, windows, scans. buy: furniture drawer. look: just admire it, no chrome. */
export type GameMode = 'build' | 'buy' | 'look'
/** What the main stage shows. */
export type Camera = '3d' | 'plan'

export type Sheet = null | 'settings' | 'goals' | 'layouts' | 'shop' | 'scan' | 'details' | 'room'

interface GameState {
  phase: Phase
  mode: GameMode
  camera: Camera
  /** A modal sheet over the stage (settings, goals list, saved designs, shop-from-link, scans). */
  sheet: Sheet
  /** Buy drawer category (a CATEGORY_GROUPS key in BuyDrawer, or 'mine' for owned/imported items). */
  buyTab: string
  /** Shows the advanced controls (exact sizes, snapping, overlays) in context menus. */
  advanced: boolean
  /** Player chose the old all-panels editor from settings. */
  classic: boolean
  /** How the 3D view is drawn (Cartoon, Lifelike, Scan, Clay, Blueprint), switched like map layers. */
  renderStyle: RenderStyle
  /**
   * Best level reached (with no errors) per design, by bestLevelKey (apartment id and layout id;
   * older saves have bare apartment ids). The HUD never shows less.
   */
  bestLevel: Record<string, number>
  /** The player was away a while (WELCOME_BACK_MS) before this visit: greet them once. */
  welcomeBack: boolean
  /** The catalog piece the "Next" chip pointed at; the drawer gives its card a nudge until it's placed. */
  buyHint: { category: string; roomId?: string } | null
  /** The goal to scroll to when the goals sheet opens. */
  goalFocus: string | null
  /** A level just reached, for the celebration card. */
  levelUp: { key: number; index: number; name: string } | null

  setPhase: (p: Phase) => void
  setMode: (m: GameMode) => void
  setCamera: (c: Camera) => void
  openSheet: (s: Sheet) => void
  setBuyTab: (t: string) => void
  setAdvanced: (a: boolean) => void
  setClassic: (c: boolean) => void
  setRenderStyle: (s: RenderStyle) => void
  /** Remember a level reached in a home; lower levels are ignored. */
  noteLevel: (homeId: string, level: number) => void
  /** "Show me" for a goal: the drawer on a tab with a piece to point at, or the goals sheet at the goal. */
  showGoal: (goal: { id: string; tab?: string; item?: string; placeIn?: string }) => void
  setBuyHint: (hint: { category: string; roomId?: string } | null) => void
  celebrateLevel: (index: number, name: string) => void
  clearLevelUp: () => void
  consumeWelcomeBack: () => void
}

const ONBOARDED_KEY = 'apartment-designer:onboarded'
const CLASSIC_KEY = 'apartment-designer:classic'
const STYLE_KEY = 'apartment-designer:render-style'
const BEST_LEVEL_KEY = 'apartment-designer:best-level'
const LAST_VISIT_KEY = 'apartment-designer:last-visit'
/** Away this long counts as coming back. */
export const WELCOME_BACK_MS = 6 * 60 * 60 * 1000
const STYLE_IDS: RenderStyle[] = ['toon', 'real', 'scan', 'clay', 'blueprint']

function readStyle(): RenderStyle {
  try {
    const v = localStorage.getItem(STYLE_KEY) as RenderStyle | null
    return v && STYLE_IDS.includes(v) ? v : 'toon'
  } catch {
    return 'toon'
  }
}

function readBestLevels(): Record<string, number> {
  try {
    const v = JSON.parse(localStorage.getItem(BEST_LEVEL_KEY) ?? '{}')
    return v && typeof v === 'object' ? (v as Record<string, number>) : {}
  } catch {
    return {}
  }
}

/**
 * Whether this visit comes after a long break, and from now on remember this as the last visit.
 * The time is refreshed whenever the page is hidden, so a tab left open still counts its break.
 */
function checkVisit(now = Date.now()): boolean {
  let away = false
  try {
    const last = Number(localStorage.getItem(LAST_VISIT_KEY))
    away = last > 0 && now - last >= WELCOME_BACK_MS
    localStorage.setItem(LAST_VISIT_KEY, String(now))
  } catch {
    // Private mode: no welcome back.
  }
  return away
}

function noteVisit() {
  try {
    localStorage.setItem(LAST_VISIT_KEY, String(Date.now()))
  } catch {
    // ignore
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') noteVisit()
    // Back to a tab left open for hours: that's a return visit too.
    else if (checkVisit()) useGame.setState({ welcomeBack: true })
  })
  window.addEventListener('pagehide', noteVisit)
}

function readOnboarded(): boolean {
  try {
    return localStorage.getItem(ONBOARDED_KEY) === '1'
  } catch {
    return false
  }
}

export function markOnboarded(done = true) {
  try {
    if (done) localStorage.setItem(ONBOARDED_KEY, '1')
    else localStorage.removeItem(ONBOARDED_KEY)
  } catch {
    // Private mode: the flow simply shows again next time.
  }
}

export const useGame = create<GameState>((set) => ({
  phase: readOnboarded() ? 'play' : 'onboarding',
  mode: 'buy',
  camera: '3d',
  sheet: null,
  buyTab: 'living',
  advanced: false,
  renderStyle: readStyle(),
  bestLevel: readBestLevels(),
  // A first-time player still in the move-in flow has nothing to come back to.
  welcomeBack: checkVisit() && readOnboarded(),
  buyHint: null,
  goalFocus: null,
  levelUp: null,
  classic: (() => {
    try {
      return localStorage.getItem(CLASSIC_KEY) === '1'
    } catch {
      return false
    }
  })(),

  setPhase: (phase) => {
    if (phase === 'play') markOnboarded()
    set({ phase, sheet: null })
  },
  setMode: (mode) => set({ mode }),
  setCamera: (camera) => set({ camera }),
  openSheet: (sheet) => set(sheet === 'goals' ? { sheet } : { sheet, goalFocus: null }),
  setBuyTab: (buyTab) => set({ buyTab }),
  setAdvanced: (advanced) => set({ advanced }),
  setClassic: (classic) => {
    try {
      localStorage.setItem(CLASSIC_KEY, classic ? '1' : '0')
    } catch {
      // ignore
    }
    set({ classic, sheet: null })
  },
  setRenderStyle: (renderStyle) => {
    try {
      localStorage.setItem(STYLE_KEY, renderStyle)
    } catch {
      // Private mode: the choice lasts for this visit.
    }
    set({ renderStyle })
  },
  noteLevel: (homeId, level) =>
    set((s) => {
      if ((s.bestLevel[homeId] ?? 0) >= level) return s
      const bestLevel = { ...s.bestLevel, [homeId]: level }
      try {
        localStorage.setItem(BEST_LEVEL_KEY, JSON.stringify(bestLevel))
      } catch {
        // Private mode: remembered for this visit.
      }
      return { bestLevel }
    }),
  showGoal: (goal) =>
    set(
      goal.tab
        ? { mode: 'buy', buyTab: goal.tab, buyHint: goal.item ? { category: goal.item, roomId: goal.placeIn } : null, sheet: null, goalFocus: null }
        : { sheet: 'goals', goalFocus: goal.id },
    ),
  setBuyHint: (buyHint) => set({ buyHint }),
  celebrateLevel: (index, name) => set({ levelUp: { key: Date.now(), index, name } }),
  clearLevelUp: () => set({ levelUp: null }),
  consumeWelcomeBack: () => set({ welcomeBack: false }),
}))
