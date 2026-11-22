import { create } from 'zustand'
import type { RenderStyle } from '../three/style'
import type { Family } from './swatches'
import type { Doc } from '../model/types'
import { localDay, todayVisit, VISIT_BY_ID } from './visits'

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
  /** `title`: the goal it came from ("A table to eat at"), headed on the picker. */
  buyHint: { category: string; roomId?: string; near?: string; title?: string } | null
  /** The goal to scroll to when the goals sheet opens. */
  goalFocus: string | null
  /** A level just reached, for the celebration card. */
  levelUp: { key: number; index: number; name: string } | null
  /** Evening light in the 3D view (lamps glow, the sun goes down): a reward from the Cozy level up. */
  evening: boolean
  /** When set, Look mode's shutter takes a photo as soon as it's showing (the Dream home card's "Take its photo"). */
  photoRequest: number
  /** Pieces that just came out of a moving box (they pop in one by one) and where the box stood (for the confetti). */
  unpacked: { key: number; ids: string[]; at?: { x: number; y: number } } | null
  /** The colour last picked per material family (wood, fabric, light); the hint picker offers it first. */
  taste: Partial<Record<Family, string>>

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
  showGoal: (goal: { id: string; title?: string; tab?: string; item?: string; placeIn?: string; near?: string }) => void
  setBuyHint: (hint: { category: string; roomId?: string; near?: string; title?: string } | null) => void
  celebrateLevel: (index: number, name: string) => void
  clearLevelUp: () => void
  consumeWelcomeBack: () => void
  setEvening: (on: boolean) => void
  /** Switch to Look mode and have the shutter take a photo there. */
  requestPhoto: () => void
  consumePhotoRequest: () => void
  setUnpacked: (u: GameState['unpacked']) => void
  /** Remember the colour of a placed pick for its family. */
  setTaste: (family: Family, color: string) => void
  /** Today's visitor and the stickers collected so far (one per visit ever finished). */
  visit: VisitState
  /** On a new local day (or the first time), pick today's visitor for this home. */
  rollVisit: (doc: Doc, day?: string, met?: (visitId: string) => boolean) => void
  /** Today's visitor (this one: a stale id from before a new day is ignored) is happy: mark the day done and collect the sticker. */
  completeVisit: (visitId: string) => void
}

export interface VisitState {
  /** The local day this visit is for ("2026-11-22"); empty before the first roll. */
  date: string
  visitId: string
  done: boolean
  stickers: string[]
}

const ONBOARDED_KEY = 'apartment-designer:onboarded'
const CLASSIC_KEY = 'apartment-designer:classic'
const STYLE_KEY = 'apartment-designer:render-style'
const BEST_LEVEL_KEY = 'apartment-designer:best-level'
const LAST_VISIT_KEY = 'apartment-designer:last-visit'
const TASTE_KEY = 'apartment-designer:taste'
export const VISIT_KEY = 'apartment-designer:visit'

export function readVisit(): VisitState {
  const empty: VisitState = { date: '', visitId: '', done: false, stickers: [] }
  try {
    const v = JSON.parse(localStorage.getItem(VISIT_KEY) ?? 'null')
    if (!v || typeof v !== 'object') return empty
    const stickers: string[] = Array.isArray(v.stickers) ? v.stickers.filter((x: unknown): x is string => typeof x === 'string' && !!VISIT_BY_ID[x]) : []
    // Yesterday's visitor has gone home: only the stickers carry over (so no stars come and go on load).
    if (v.date !== localDay() || typeof v.visitId !== 'string' || !VISIT_BY_ID[v.visitId]) return { ...empty, stickers }
    return { date: v.date, visitId: v.visitId, done: !!v.done, stickers }
  } catch {
    return empty
  }
}

function saveVisit(v: VisitState) {
  try {
    localStorage.setItem(VISIT_KEY, JSON.stringify(v))
  } catch {
    // Private mode: today's visitor lasts for this visit.
  }
}

function readTaste(): Partial<Record<Family, string>> {
  try {
    const v = JSON.parse(localStorage.getItem(TASTE_KEY) ?? '{}')
    return v && typeof v === 'object' ? (v as Partial<Record<Family, string>>) : {}
  } catch {
    return {}
  }
}
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
/** How long the last break was, for the welcome back ("10 days away!"). */
export let lastAwayMs = 0

function checkVisit(now = Date.now()): boolean {
  let away = false
  try {
    const last = Number(localStorage.getItem(LAST_VISIT_KEY))
    away = last > 0 && now - last >= WELCOME_BACK_MS
    if (away) lastAwayMs = now - last
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
  evening: false,
  photoRequest: 0,
  unpacked: null,
  taste: readTaste(),
  visit: readVisit(),
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
        ? { mode: 'buy', buyTab: goal.tab, buyHint: goal.item ? { category: goal.item, roomId: goal.placeIn, near: goal.near, title: goal.title } : null, sheet: null, goalFocus: null }
        : { sheet: 'goals', goalFocus: goal.id },
    ),
  setBuyHint: (buyHint) => set({ buyHint }),
  celebrateLevel: (index, name) => set({ levelUp: { key: Date.now(), index, name } }),
  clearLevelUp: () => set({ levelUp: null }),
  consumeWelcomeBack: () => set({ welcomeBack: false }),
  setEvening: (evening) => set({ evening }),
  requestPhoto: () => set({ mode: 'look', sheet: null, levelUp: null, photoRequest: Date.now() }),
  consumePhotoRequest: () => set({ photoRequest: 0 }),
  setUnpacked: (unpacked) => set({ unpacked }),
  setTaste: (family, color) =>
    set((s) => {
      if (s.taste[family] === color) return s
      const taste = { ...s.taste, [family]: color }
      try {
        localStorage.setItem(TASTE_KEY, JSON.stringify(taste))
      } catch {
        // Private mode: remembered for this visit.
      }
      return { taste }
    }),
  rollVisit: (doc, day = localDay(), met) =>
    set((s) => {
      if (s.visit.date === day && s.visit.visitId) return s
      const visit = { date: day, visitId: todayVisit(doc, day, s.visit.stickers, met).id, done: false, stickers: s.visit.stickers }
      saveVisit(visit)
      return { visit }
    }),
  completeVisit: (visitId) =>
    set((s) => {
      if (!s.visit.visitId || s.visit.visitId !== visitId || s.visit.done) return s
      const stickers = s.visit.stickers.includes(s.visit.visitId) ? s.visit.stickers : [...s.visit.stickers, s.visit.visitId]
      const visit = { ...s.visit, done: true, stickers }
      saveVisit(visit)
      return { visit }
    }),
}))

// ---- Steady stars while a gesture is open (r13-b) ----

/**
 * The score the vibe meter shows: while a drag or an edit run is open it keeps the score from the
 * moment the gesture began, so stars, level and pills never flicker mid-move; once the gesture
 * ends the live score shows again and the meter settles once, on the net change.
 */
export function heldScore<T>(live: T, held: T | null, gesturing: boolean): T {
  return gesturing && held !== null ? held : live
}

/** The one net change a settled gesture (or a single edit) shows: a cheer, a soft loss, or nothing. */
export function settleStars(before: number, after: number): { kind: 'gain' | 'loss' | 'none'; delta: number } {
  if (after > before) return { kind: 'gain', delta: after - before }
  if (after < before) return { kind: 'loss', delta: before - after }
  return { kind: 'none', delta: 0 }
}
