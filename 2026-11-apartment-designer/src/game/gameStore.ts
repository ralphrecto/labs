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

  setPhase: (p: Phase) => void
  setMode: (m: GameMode) => void
  setCamera: (c: Camera) => void
  openSheet: (s: Sheet) => void
  setBuyTab: (t: string) => void
  setAdvanced: (a: boolean) => void
  setClassic: (c: boolean) => void
  setRenderStyle: (s: RenderStyle) => void
}

const ONBOARDED_KEY = 'apartment-designer:onboarded'
const CLASSIC_KEY = 'apartment-designer:classic'
const STYLE_KEY = 'apartment-designer:render-style'
const STYLE_IDS: RenderStyle[] = ['toon', 'real', 'scan', 'clay', 'blueprint']

function readStyle(): RenderStyle {
  try {
    const v = localStorage.getItem(STYLE_KEY) as RenderStyle | null
    return v && STYLE_IDS.includes(v) ? v : 'toon'
  } catch {
    return 'toon'
  }
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
  openSheet: (sheet) => set({ sheet }),
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
}))
