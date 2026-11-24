// Goals HUD: the vibe meter pill (stars, level, progress, celebrations), the next-goal chip, the
// level-up card and the goals sheet.
import { announce } from './announce'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from 'react'
import { useAnalysis } from '../analysis'
import { catalogEntry, itemFromCatalog } from '../model/catalog'
import { findDrop } from '../model/placement'
import type { Doc } from '../model/types'
import { activeLayout, useStore } from '../store'
import { heldScore, lastAwayMs, settleStars, useGame, VISIT_KEY } from './gameStore'
import { bankableLevel, bestLevelKey, levelBlocker, CATEGORY_LABEL, CATEGORY_ORDER, LEVELS, levelProgress, MIN_PIECES, nextGoal, packedBoxIds, scoreHome, type Goal, type HomeScore } from './goals'
import { unpackAndCelebrate } from './unpack'
import { albumFor, TWIST_EMOJI, useAlbum } from './album'
import { IconCheck, IconClose, IconSparkle, IconStar } from './icons'
import { useModal } from './useModal'
import { openRemixPicker } from './Designs'
import { DRAG_PERK, goalMove, localDay, twistMove, TWISTS, VISIT_BY_ID, VISIT_LEVEL, VISIT_STARS, VISITS, visitLeads, type Visit } from './goals'
import { reduceMotion } from '../motion'
import { listHomes } from '../storage'
import type { Issue } from '../model/checks'
import './feel.css'

/**
 * The home's score as it is now (an honest level: a layout with snags reads as what it is), plus
 * the best level reached in this design. The best only gates celebrations (no second level-up for
 * a level already reached) and shows as a small "Best" note. Each design keeps its own best.
 */
export function useHomeScore(): HomeScore & { best: number } {
  const doc = useStore((s) => s.doc)
  const layout = activeLayout(doc)
  const analysis = useAnalysis(doc, layout)
  const visitState = useGame((s) => s.visit)
  const visitIn = useMemo(() => (visitState.visitId ? { id: visitState.visitId, done: visitState.done } : null), [visitState.visitId, visitState.done])
  // The drag lesson's star belongs to the home it was nailed in (round 16).
  const dragPerk = useGame((s) => !!s.dragStar && s.dragStar === doc.apartment.id)
  const perks = useMemo(() => (dragPerk ? { drag: true } : null), [dragPerk])
  const raw = useMemo(() => scoreHome(doc, layout, analysis, visitIn, perks), [doc, layout, analysis, visitIn, perks])
  lastIssues = analysis.issues
  // A new local day brings a new visitor (one the home doesn't already please, when there is one).
  // While move-in day's boxes are still the first thing to do, the visitor waits: unpacking would
  // otherwise please a visitor the player never got to meet.
  const unpacking = raw.goals.some((g) => g.id === 'unpack' && !g.done && g.urgent)
  useEffect(() => {
    if (unpacking) return
    useGame.getState().rollVisit(doc, undefined, (id) => !!scoreHome(doc, layout, analysis, { id }).goals.find((g) => g.visit === id)?.done)
  }, [doc, layout, analysis, unpacking])
  // Today's visitor is happy: the sticker is theirs to keep, even if the room changes later.
  const metId = raw.goals.find((g) => g.visit && g.done)?.visit
  useEffect(() => {
    if (metId) useGame.getState().completeVisit(metId)
  }, [metId])
  const key = bestLevelKey(doc)
  // Saves from before per-design levels kept one best per home: it carries over to the first design only.
  const legacy = doc.layouts[0]?.id === layout.id ? doc.apartment.id : null
  const best = useGame((s) => s.bestLevel[key] ?? (legacy ? s.bestLevel[legacy] : undefined) ?? 0)
  const bank = bankableLevel(raw)
  useEffect(() => {
    if (bank > best) useGame.getState().noteLevel(key, bank)
  }, [key, bank, best])
  // A remix's first Dream home is noted on the design (the Designs sheet's ribbon, the card's count).
  const remixDreamDue = bank === LEVELS.length - 1 && !!layout.remix && !layout.remix.dreamAt
  useEffect(() => {
    if (remixDreamDue) useStore.getState().markRemixDream(layout.id)
  }, [remixDreamDue, layout.id])
  return useMemo(() => ({ ...raw, best: Math.max(best, bank) }), [raw, best, bank])
}

/**
 * Dry-runs placement for the Next chip: whether a catalog piece has a free spot in the home right
 * now. Cached per doc; pieces the store just refused (while the chip's hint was up) count as not
 * fitting until the home changes.
 */
function useFits(doc: Doc, refused: Set<string>) {
  const cache = useMemo(() => new Map<string, boolean>(), [doc])
  return useCallback(
    (category: string, near?: string) => {
      if (refused.has(category)) return false
      const key = near ? `${category}|${near}` : category
      let ok = cache.get(key)
      if (ok === undefined) {
        try {
          ok = findDrop(doc, activeLayout(doc), itemFromCatalog(catalogEntry(category), '__probe'), { nextTo: near }).ok
        } catch {
          ok = true
        }
        cache.set(key, ok)
      }
      return ok
    },
    [cache, doc, refused],
  )
}

/** The Next goal, never one whose piece has nowhere to go. */
export function useNextGoal(score: HomeScore, refused: Set<string> = NO_REFUSALS, skipped: Set<string> = NO_REFUSALS): Goal | undefined {
  const doc = useStore((s) => s.doc)
  const fits = useFits(doc, refused)
  return useMemo(() => nextGoal(score, fits, skipped), [score, fits, skipped])
}
const NO_REFUSALS = new Set<string>()

/** The latest layout issues, so Next can point at the piece a "move it" goal is about. */
let lastIssues: Issue[] = []
/** Goals fixed by moving a piece, and the issue kind that names that piece. */
const MOVE_GOAL: Record<string, Issue['kind']> = { 'no-overlap': 'overlap', 'in-walls': 'wall', 'doors-clear': 'door', walkways: 'path' }

/** Phones get a compact level-up banner under the pill instead of the centred card. */
const PHONE = '(max-width: 700px)'
function usePhone() {
  const [phone, setPhone] = useState(() => typeof matchMedia === 'function' && matchMedia(PHONE).matches)
  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const m = matchMedia(PHONE)
    const on = () => setPhone(m.matches)
    m.addEventListener?.('change', on)
    return () => m.removeEventListener?.('change', on)
  }, [])
  return phone
}

/** The goal that earned the latest level, shown in the level-up ("+2★ A cozy corner to sleep"). */
let levelGoalLine = ''
/** What each level opens up, said on its level-up card (and to screen readers). */
const UNLOCK_LINE: Record<string, string | undefined> = {
  'Moving in': 'Styles unlocked. Try one with the style button.',
  Cozy: 'Evening light, visitors and bonus goals unlocked.',
  'Designer home': 'Remix unlocked: tap the glowing designs button, top left.',
}

/** The level-up's unlock line, only the first time that level is reached anywhere (round 16). */
const unlockLine = (lv: { name: string; unlock?: boolean }) => (lv.unlock ? UNLOCK_LINE[lv.name] : undefined)

/** Goals finished in two steps (a desk, then a chair): the first step gets a "½" and a nudge. */
const TWO_STEP = new Set(['work', 'eat', 'nook'])
/** How close (m) the drag lesson's piece must land to its ring. */
const LESSON_REACH = 0.2
const piece = (cat: string) => cat.replace(/^(office|dining)-chair$/, 'chair').replace(/-/g, ' ')
const ago = (ms: number) => {
  const d = Math.round(ms / 86400000)
  if (d >= 1) return `${d} day${d === 1 ? '' : 's'} away!`
  const h = Math.max(1, Math.round(ms / 3600000))
  return `${h} hours away!`
}

const SPARKS = 10
/** A level drop shows only once it has lasted this long (an Undo or a quick fix never demotes). */
const LEVEL_DROP_MS = 2000
const LOSS_MS = 6000
/** A tapped chip steps aside this long at most (round 16), then offers itself again. */
const CHIP_BACK_MS = 8000
const PLACED_TOAST = /^Placed /
/** The level-up card waits until placing has paused this long, so it never lands mid-burst. */
const LEVEL_UP_QUIET_MS = 1200

function Stars({ n, filled = true }: { n: number; filled?: boolean }) {
  return (
    <span className={`goal-stars${filled ? ' on' : ''}`} aria-label={`${n} star${n === 1 ? '' : 's'}`}>
      {Array.from({ length: n }, (_, i) => (
        <IconStar key={i} />
      ))}
    </span>
  )
}

/**
 * Yesterday's visitor, if they were pleased (their stars went home with them): read before today's
 * visitor is rolled, so the welcome back can say why the stars dropped overnight.
 */
let goneVisit: Visit | undefined = (() => {
  try {
    const v = JSON.parse(localStorage.getItem(VISIT_KEY) ?? 'null')
    return v && v.date && v.date !== localDay() && v.done ? VISIT_BY_ID[v.visitId] : undefined
  } catch {
    return undefined
  }
})()

/** Today's visitor, for the welcome card and the goals sheet. */
function useVisit(): { visit?: Visit; done: boolean; stickers: string[] } {
  const v = useGame((s) => s.visit)
  return { visit: VISIT_BY_ID[v.visitId], done: v.done, stickers: v.stickers }
}

type Banner =
  | { key: number; kind: 'goal'; goal: Goal; more: number }
  /** `line`: "Sample apartment · Move-in day · Cozy" (the home's name only with more than one home; the stars sit in the badge). `top`: at the top level, lead with a variation. */
  | { key: number; kind: 'welcome'; goal?: Goal; home: string; away?: string; line: string; stars: number; top: boolean; visit?: { id: string; text: string }; gone?: string }

/** Homes saved on this device (the welcome card names the home only when there's more than one). */
function homeCount() {
  try {
    return listHomes().length
  } catch {
    return 1
  }
}

/** At the top level there's nothing left to climb in this design: a remix (with a twist) is the next game. */
function tryVariation() {
  openRemixPicker()
}

export function VibeMeter() {
  const live = useHomeScore()
  // Mid-gesture (a drag, a run of nudges) the meter holds still on the score from before it began;
  // when the gesture ends it settles once, with one net gain or loss for the whole move.
  const gesturing = useStore((s) => s.gesture)
  const held = useRef<typeof live | null>(null)
  if (!gesturing) held.current = live
  const score = heldScore(live, held.current, gesturing)
  const designKey = useStore((s) => bestLevelKey(s.doc))
  const homeName = useStore((s) => s.doc.apartment.name)
  const designName = useStore((s) => activeLayout(s.doc).name)
  const placed = useStore((s) => activeLayout(s.doc).placements.length)
  const remixTwist = useStore((s) => {
    const r = activeLayout(s.doc).remix
    return r && !r.dreamAt ? r.twist : null
  })
  const top = score.level.index >= LEVELS.length - 1
  const scoreRef = useRef(score)
  scoreRef.current = score
  const welcomeBack = useGame((s) => s.welcomeBack)
  const levelUp = useGame((s) => s.levelUp)
  const sheet = useGame((s) => s.sheet)
  const phone = usePhone()
  const prev = useRef<{ key: string; stars: number; level: number; done: Set<string>; progress: Map<string, number>; visit?: string } | null>(null)
  const [pop, setPop] = useState<{ key: number; delta: number; half?: boolean; text?: string; tick?: string } | null>(null)
  const [banner, setBanner] = useState<Banner | null>(null)
  const [shown, setShown] = useState<string | null>(null)
  // Pieces the store refused while the chip's hint was up: Next stops suggesting them until the home changes.
  const [refused, setRefused] = useState<Set<string>>(NO_REFUSALS)
  // Goals a piece placed from the chip didn't finish or bring closer (round 16): Next moves on from
  // them in this design rather than offering the same piece again and again.
  const [skipped, setSkipped] = useState<Set<string>>(NO_REFUSALS)
  /** The pick goal the chip was last tapped for, to judge the next piece placed. */
  const tried = useRef<{ id: string; progress: number; placed: number; key: string } | null>(null)
  const next = useNextGoal(score, refused, skipped)
  const nextRef = useRef(next)
  nextRef.current = next
  // When the Next goal moves on to another piece, a hint left from the old one is stale.
  const nextKey = next ? `${next.id}:${next.item ?? ''}` : ''
  const lastNextKey = useRef(nextKey)
  useEffect(() => {
    if (lastNextKey.current === nextKey) return
    lastNextKey.current = nextKey
    const hint = useGame.getState().buyHint
    if (hint && hint.category !== nextRef.current?.item) useGame.getState().setBuyHint(null)
  }, [nextKey])
  const toast = useStore((s) => s.toast)
  // A refusal ("The bedroom's full...") while the hint is up: bring the chip back with something that fits.
  useEffect(() => {
    const hint = useGame.getState().buyHint
    if (!toast || PLACED_TOAST.test(toast) || !hint || shown === null) return
    setRefused((r) => new Set([...r, hint.category]))
    setShown(null)
    // Only when a toast arrives.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toast])
  // The drag lesson (round 15): dropped within 20 cm of the glowing ring, it's a "Nice placement!";
  // gone (deleted, another design), the ring goes too.
  const lesson = useGame((s) => s.dragLesson)
  const lessonAt = useStore((s) => (lesson ? activeLayout(s.doc).placements.find((p) => p.id === lesson.id) : undefined))
  useEffect(() => {
    if (!lesson || gesturing) return
    if (!lessonAt) {
      useGame.getState().setDragLesson(null)
      return
    }
    if (Math.hypot(lessonAt.x - lesson.x, lessonAt.y - lesson.y) > LESSON_REACH) return
    useGame.getState().setDragLesson(null)
    // Round 16: nailing it earns a star (once ever, for this home), cheered like any goal.
    const starred = useGame.getState().awardDragStar(useStore.getState().doc.apartment.id)
    useStore.getState().notify(starred ? `Nice placement! +${DRAG_PERK.stars}★ ✨ Drag any piece to try a new spot.` : 'Nice placement! ✨ Drag any piece to try a new spot.')
    if (!starred) setPop({ key: Date.now(), delta: 0, text: 'Nice!' })
  }, [lesson, lessonAt, gesturing])
  /** Stars just lost, with the goal that broke: a soft pill with Undo, never a cheer. */
  const [loss, setLoss] = useState<{ key: number; delta: number; title: string } | null>(null)
  /** The level the pill shows. Going up shows at once; a drop waits LEVEL_DROP_MS in case it's undone. */
  const [shownLevel, setShownLevel] = useState(score.level.index)
  const levelKey = useRef(designKey)
  useEffect(() => {
    const switched = levelKey.current !== designKey
    levelKey.current = designKey
    if (switched || score.level.index >= shownLevel) {
      setShownLevel(score.level.index)
      return
    }
    const t = setTimeout(() => setShownLevel(score.level.index), LEVEL_DROP_MS)
    return () => clearTimeout(t)
  }, [score.level.index, designKey, shownLevel])
  const levelTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const pendingLevel = useRef<{ index: number; name: string } | null>(null)
  /** The highest level celebrated (or banked before this visit) per design. Kept apart from the
   * banked best, which a passive effect raises as soon as a level is reached: a level-up that got
   * cancelled (a loss, a burst of goals) is still celebrated once the level is back. */
  const celebrated = useRef<Record<string, number>>({})
  const latestLevel = useRef(score.level.index)
  latestLevel.current = bankableLevel(score)
  useEffect(() => () => clearTimeout(levelTimer.current), [])

  // Layout effect, so a "Placed a sofa" toast that a goal or level-up replaces never paints.
  useLayoutEffect(() => {
    const done = new Set(score.goals.filter((g) => g.done).map((g) => g.id))
    const progress = new Map(score.goals.map((g) => [g.id, g.progress ?? 0]))
    const before = prev.current
    const visitGoal = score.goals.find((g) => g.visit)
    prev.current = { key: designKey, stars: score.stars, level: score.level.index, done, progress, visit: visitGoal?.id }
    // First render, or another design or home: no fanfare.
    if (!before || before.key !== designKey) {
      if (before) setSkipped(NO_REFUSALS)
      tried.current = null
      celebrated.current[designKey] = Math.max(celebrated.current[designKey] ?? 0, useGame.getState().bestLevel[designKey] ?? 0, bankableLevel(score))
      clearTimeout(levelTimer.current)
      pendingLevel.current = null
      return
    }
    // The walkway's pinch marker goes once the way is clear.
    if (useGame.getState().pinch && !score.goals.some((g) => g.id === 'walkways' && g.snag)) useGame.getState().setPinch(null)
    // Any change to the home brings the chip back (with whatever is next now).
    setShown(null)
    setRefused(NO_REFUSALS)
    // The first piece placed after a tap on a pick goal: if that goal is no closer, Next moves on.
    const t = tried.current
    const count = activeLayout(useStore.getState().doc).placements.length
    if (t && t.key === designKey && count > t.placed) {
      tried.current = null
      const g = score.goals.find((x) => x.id === t.id)
      if (g && !g.done && (g.progress ?? 0) <= t.progress + 1e-6) setSkipped((s) => new Set([...s, t.id]))
    }
    // A visitor who arrives already happy (a new day) is no headline: only one the player pleases is.
    const fresh = score.goals.filter((g) => g.done && !before.done.has(g.id) && !(g.visit && before.visit !== g.id))
    // The drawer's hint pill ("Tap Plant → Bathroom") points at a goal's piece: once that goal is
    // done (it just finished, or every goal asking for that piece there is done), the pill goes.
    const hint = useGame.getState().buyHint
    if (hint) {
      const asks = (g: Goal) => g.item === hint.category && (!hint.roomId || !g.placeIn || g.placeIn === hint.roomId)
      const behind = score.goals.filter(asks)
      if (fresh.some(asks) || (behind.length > 0 && behind.every((g) => g.done))) useGame.getState().setBuyHint(null)
    }
    // A level never celebrated in this design, on a layout without errors, is celebrated, even when
    // it comes with a burst of goals (a template, a paste) that gets no banner of its own. The
    // level-up waits for a pause in placing (each change pushes it back), so the drop shows first.
    const seen = celebrated.current[designKey] ?? 0
    const leveled = score.level.index > before.level && bankableLevel(score) > seen
    if (leveled) pendingLevel.current = { index: score.level.index, name: score.level.name }
    if (pendingLevel.current) {
      clearTimeout(levelTimer.current)
      levelTimer.current = setTimeout(() => {
        const lv = pendingLevel.current
        pendingLevel.current = null
        // Undone (or a snag appeared) while it waited: not yet; it's celebrated when it's back.
        if (!lv || latestLevel.current < lv.index) return
        celebrated.current[designKey] = Math.max(celebrated.current[designKey] ?? 0, lv.index)
        // The level-up carries the goal line; the toast steps aside for it.
        useStore.setState({ toast: null })
        useGame.getState().celebrateLevel(lv.index, lv.name)
      }, LEVEL_UP_QUIET_MS)
    }
    if (fresh.length > 4) return
    // A new visitor (a new day) changes the stars on offer, not the player's work: no cheer, no loss.
    if (before.visit !== visitGoal?.id) return
    const net = settleStars(before.stars, score.stars)
    // The drag lesson's piece sits a little aside on purpose: no loss pill while it's waiting there.
    if (net.kind === 'loss' && useGame.getState().dragLesson) return
    if (net.kind === 'loss') {
      // A loss: say what broke, offer Undo, and clear any cheer (banner, pop, pending level-up).
      const broke = score.goals.filter((g) => !g.done && before.done.has(g.id)).sort((a, b) => b.stars - a.stars)[0]
      setLoss({ key: Date.now(), delta: net.delta, title: broke?.title ?? '' })
      setPop(null)
      setBanner((b) => (b?.kind === 'goal' ? null : b))
      clearTimeout(levelTimer.current)
      pendingLevel.current = null
      if (useGame.getState().levelUp) useGame.getState().clearLevelUp()
      return
    }
    // A remix's pill counts goals, not stars (round 16): each goal finished ticks it, visibly.
    const rx = activeLayout(useStore.getState().doc).remix
    const ticks = rx && !rx.dreamAt ? fresh.filter((g) => !g.visit).length : 0
    if (net.kind === 'gain') {
      setLoss(null)
      setPop({ key: Date.now(), delta: net.delta, ...(ticks ? { tick: ticks > 1 ? `✓ ${ticks} goals` : '✓ Goal' } : {}) })
    }
    const toast = useStore.getState().toast
    const placedToast = !!toast && PLACED_TOAST.test(toast)
    // First half of a two-step goal (a desk without its chair, a table without a chair).
    // Only the goal the player is following (round 15): an armchair placed for the talk corner must
    // not say "Now add a floor lamp beside it" for the reading nook.
    const hintGoal = useGame.getState().hintGoal
    const half = placedToast && hintGoal ? score.goals.find((g) => g.id === hintGoal && TWO_STEP.has(g.id) && !g.done && (g.progress ?? 0) > (before.progress.get(g.id) ?? 0) && g.item) : undefined
    // A room's "feels just right" that tips over as a side effect is no headline when the piece
    // just placed is half of something: the nudge to finish it ("Now add a chair") says more.
    const incidental = fresh.length > 0 && fresh.every((g) => g.id.startsWith('density:'))
    const pleased = fresh.find((g) => g.visit)
    if (pleased) {
      // Today's visitor loved it: always the banner (with its sticker), never just a toast.
      levelGoalLine = `+${pleased.stars}★ ${VISIT_BY_ID[pleased.visit!]?.who ?? 'Your visitor'} loved it!`
      if (placedToast) useStore.setState({ toast: null })
      setBanner({ key: Date.now(), kind: 'goal', goal: pleased, more: fresh.length - 1 })
    } else if (fresh.length && !(half && incidental)) {
      const best = [...fresh].sort((a, b) => b.stars - a.stars)[0]
      // One short line per action ("+1★ Every room has something"): the room shows in 3D and the
      // pill's pop has the total.
      const line = ticks ? `✓ ${best.title}` : `+${best.stars}★ ${best.title}`
      levelGoalLine = line
      if (placedToast) useStore.getState().notify(line)
      else setBanner({ key: Date.now(), kind: 'goal', goal: best, more: fresh.length - 1 })
    } else if (half) {
      // Say what finishes it (the pill's pop still counts any incidental star).
      const what = toast!.replace(/^Placed (an? )?/, '').replace(/ in the .*$/, '').replace(/\.$/, '')
      useStore.getState().notify(`${what.charAt(0).toUpperCase()}${what.slice(1)} placed. Now add a ${piece(half.item!)} beside it.`)
      setPop({ key: Date.now(), delta: score.stars - before.stars, half: true })
    }
  }, [score, designKey])

  // Welcome back, once, after a long break: the goal banner with what to do next.
  useEffect(() => {
    if (!welcomeBack) return
    useGame.getState().consumeWelcomeBack()
    const sc = scoreRef.current
    const vs = useGame.getState().visit
    const vd = VISIT_BY_ID[vs.visitId]
    // A visitor pleased yesterday took their stars home: say so, so the drop isn't a mystery.
    const gone = goneVisit ?? (vs.date && vs.date !== localDay() && vs.done ? VISIT_BY_ID[vs.visitId] : undefined)
    goneVisit = undefined
    const today = gone && vd && vd.id !== gone.id ? vd : undefined
    // Next only leads with the visitor from Cozy on (or when it's one pick away): say when.
    const hasVisitGoal = sc.goals.some((g) => g.visit)
    const later = !vs.done && (hasVisitGoal ? !visitLeads(sc) : sc.level.index < VISIT_LEVEL)
    setBanner({
      key: Date.now(),
      kind: 'welcome',
      goal: nextRef.current,
      home: homeName,
      away: lastAwayMs ? ago(lastAwayMs) : undefined,
      line: `${homeCount() > 1 ? `${homeName} · ` : ''}${designName} · ${sc.level.name}`,
      stars: sc.stars,
      // At the top level a remix comes before any bonus goal (only core goals beat it).
      top: sc.level.index >= LEVELS.length - 1 && (!nextRef.current || (!!nextRef.current.bonus && !nextRef.current.visit)),
      visit:
        vd && (!gone || today)
          ? { id: vd.id, text: vs.done ? `${vd.emoji} ${vd.who} loved it! Sticker collected` : `${vd.emoji} ${vd.line}${later ? ' (once this design is Cozy)' : ''}` }
          : undefined,
      gone: gone ? `${gone.emoji} ${gone.who} went home${today ? ` · today: ${today.emoji} +${VISIT_STARS}★` : ''}` : undefined,
    })
    // Only when the flag is raised; the score at that moment is the one to suggest from.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [welcomeBack])

  // A tap that went nowhere (round 16): the chip comes back when the goals sheet closes, or after
  // CHIP_BACK_MS, rather than staying hidden until the home changes.
  useEffect(() => {
    if (shown === null) return
    const t = setTimeout(() => setShown(null), CHIP_BACK_MS)
    return () => clearTimeout(t)
  }, [shown])
  const sheetWas = useRef(sheet)
  useEffect(() => {
    if (sheetWas.current && !sheet) setShown(null)
    sheetWas.current = sheet
  }, [sheet])
  useEffect(() => {
    if (!loss) return
    const t = setTimeout(() => setLoss(null), LOSS_MS)
    return () => clearTimeout(t)
  }, [loss])
  useEffect(() => {
    if (!pop) return
    const t = setTimeout(() => setPop(null), 1300)
    return () => clearTimeout(t)
  }, [pop])
  // The welcome back stays until it's tapped or a piece is placed; goal banners pass by.
  useEffect(() => {
    if (!banner || banner.kind === 'welcome') return
    const t = setTimeout(() => setBanner(null), 2600)
    return () => clearTimeout(t)
  }, [banner])
  // Screen readers hear these through the page's one live region (announce.tsx); the visual
  // banners, pops and cards come and go without roles of their own.
  useEffect(() => {
    if (!banner) return
    if (banner.kind === 'goal') {
      const who = banner.goal.visit ? `${VISIT_BY_ID[banner.goal.visit]?.who ?? 'Your visitor'} loved it! ` : banner.more ? `${banner.more + 1} goals complete. ` : 'Goal complete. '
      announce(`${who}${banner.goal.title}${banner.goal.visit ? '. Sticker collected.' : `, +${banner.goal.stars}★.`}`)
    } else if (banner.kind === 'welcome') {
      const next = banner.top ? 'Remix this home.' : banner.goal ? `Next: ${banner.goal.title}, +${banner.goal.stars}★.` : ''
      announce(`Welcome back!${banner.away ? ` ${banner.away}` : ''} ${banner.line}. ${banner.stars}★. ${next} ${banner.gone ?? ''} ${banner.visit?.text ?? ''}`)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [banner?.key])
  useEffect(() => {
    if (loss) announce(`Lost ${loss.delta}★${loss.title ? `: ${loss.title}` : ''}. Undo is available.`)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loss?.key])
  useEffect(() => {
    if (!levelUp) return
    const dream = levelUp.index >= LEVELS.length - 1
    announce(
      `${dream ? 'You did it! Dream home reached.' : `Level up! ${levelUp.name}, level ${levelUp.index + 1} of ${LEVELS.length}.`} ${levelGoalLine}${unlockLine(levelUp) ? ` ${unlockLine(levelUp)}` : ''}`,
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [levelUp?.key])
  // Every settled change in stars: "+2 stars, 16 stars, Cozy". Switching homes or designs isn't one.
  const spoken = useRef({ key: designKey, stars: score.stars })
  useEffect(() => {
    if (gesturing) return
    const was = spoken.current
    spoken.current = { key: designKey, stars: score.stars }
    const d = score.stars - was.stars
    if (was.key !== designKey || !d) return
    announce(`${d > 0 ? '+' : '−'}${Math.abs(d)}★, ${score.stars}★, ${score.level.name}`)
  }, [score.stars, score.level.name, designKey, gesturing])
  const placedBefore = useRef(placed)
  useEffect(() => {
    if (placed > placedBefore.current) setBanner((b) => (b?.kind === 'welcome' ? null : b))
    placedBefore.current = placed
  }, [placed])

  // While a level drop is held back, the pill keeps the old level (with an empty bar).
  const lagging = shownLevel !== score.level.index
  const levelName = lagging ? LEVELS[Math.min(shownLevel, LEVELS.length - 1)] : score.level.name
  const fill = lagging ? 0 : levelProgress(score)
  // One quiet chip with the next goal. Tapping it steps it aside until the home changes.
  const chip = next && !(top && next.bonus && !next.visit) && !banner && !levelUp && !sheet && !loss && shown !== next.id ? next : null
  const bestName = score.best > score.level.index ? LEVELS[score.best] : null
  useAlbum((s) => s.rev)
  const homeId = useStore((s) => s.doc.apartment.id)
  const layoutId = useStore((s) => s.doc.activeLayoutId)
  const photographed = top && albumFor(homeId).some((p) => p.layoutId === layoutId)
  // A remix (round 15) climbs toward its own Dream home: the pill shows that climb ("Remix: 3 of 5")
  // and the twist, never the original's stars dropping away.
  const remixing = !!remixTwist
  const remixLabel = remixing ? `Remix: ${score.level.index} of ${LEVELS.length - 1}` : ''
  const remixGoals = remixing ? score.goals.filter((g) => g.done && !g.visit).length : 0
  return (
    <div className="goal-meter-wrap">
      <button
        type="button"
        className={`goal-meter${pop ? ' popping' : ''}`}
        onClick={() => useGame.getState().openSheet('goals')}
        aria-label={remixing ? `${remixLabel}, ${levelName}, ${remixGoals} goals done. Open goals` : `${score.stars} stars, ${levelName}. Open goals`}
      >
        <span className={`goal-meter-star${remixing ? ' remix' : ''}`} key={pop?.key}>
          {remixing ? <span aria-hidden>{TWIST_EMOJI[remixTwist!] ?? '↻'}</span> : <IconStar />}
          <b>{remixing ? `${score.level.index}/${LEVELS.length - 1}` : score.stars}</b>
        </span>
        <span className="goal-meter-body">
          <span className="goal-meter-level">
            {remixing ? remixLabel : levelName}
            {remixing ? (
              <small className="goal-meter-best">
                {TWISTS[remixTwist!].name} · {levelName} ·{' '}
                <span className="goal-meter-tick" key={remixGoals} aria-label={`${remixGoals} goals done`}>
                  ✓{remixGoals}
                </span>
              </small>
            ) : bestName && <small className="goal-meter-best">Best: {bestName}</small>}
          </span>
          <span className="goal-meter-bar">
            <i style={{ width: `${Math.round(fill * 100)}%` }} />
          </span>
        </span>
      </button>
      {pop && (
        <div className="goal-burst" key={`burst${pop.key}`} aria-hidden>
          {Array.from({ length: SPARKS }, (_, i) => (
            <i key={i} style={{ '--a': `${(360 / SPARKS) * i + (i % 2) * 12}deg`, '--d': `${i % 2 ? 34 : 48}px` } as CSSProperties} />
          ))}
          <span className="goal-plus">{pop.tick ?? (pop.text ? `✨ ${pop.text}` : `${pop.half && !pop.delta ? '½' : `+${pop.delta}`} ★`)}</span>
        </div>
      )}
      {loss && (
        <div className="feel-loss" key={`loss${loss.key}`}>
          <span className="feel-loss-text">
            <b>−{loss.delta}★</b>
            {loss.title && <span> {loss.title}</span>}
          </span>
          <button
            type="button"
            className="feel-loss-undo"
            onClick={() => {
              setLoss(null)
              useStore.getState().undo()
            }}
          >
            Undo
          </button>
        </div>
      )}
      {top && (!next || (next.bonus && !next.visit)) && !banner && !levelUp && !sheet && !loss && (
        photographed ? (
          <button type="button" className="goal-next" onClick={tryVariation} aria-label="Top level reached. Remix this home with a twist">
            <span className="goal-next-text">
              <b>Remix this home</b>
            </span>
            <span className="goal-next-stars">★</span>
          </button>
        ) : (
          // Round 15: until this design has a photo in the album (even after a reload), the Dream
          // home's photo comes first; then Remix.
          <button
            type="button"
            className="goal-next"
            onClick={() => {
              useGame.getState().setEvening(true)
              useGame.getState().requestPhoto()
            }}
            aria-label="Dream home reached. Take its photo"
          >
            <span className="goal-next-text">
              <b>Take its photo</b>
            </span>
            <span className="goal-next-stars">📷</span>
          </button>
        )
      )}
      {chip && (
        <button
          type="button"
          className="goal-next"
          key={chip.id}
          onClick={() => {
            setShown(chip.id)
            tried.current = chip.item && chip.tab ? { id: chip.id, progress: chip.progress ?? 0, placed, key: designKey } : null
            showGoal(chip)
          }}
          aria-label={`Next goal: ${chip.title}, ${chip.stars} star${chip.stars === 1 ? '' : 's'}. Show me`}
        >
          <span className="goal-next-text">
            Next: <b>{chip.title}</b>
          </span>
          <span className="goal-next-stars">+{chip.stars} ★</span>
        </button>
      )}
      {banner?.kind === 'goal' && (
        <div className="goal-banner" key={`banner${banner.key}`} aria-hidden="true">
          <span className="goal-banner-check">
            <IconCheck />
          </span>
          {banner.goal.visit ? (
            <span className="goal-banner-text">
              <small>{VISIT_BY_ID[banner.goal.visit]?.who ?? 'Your visitor'} loved it!</small>
              {banner.goal.title}
            </span>
          ) : (
            <span className="goal-banner-text">
              <small>{banner.more ? `${banner.more + 1} goals complete` : 'Goal complete'}</small>
              {banner.goal.title}
            </span>
          )}
          {banner.goal.visit ? (
            <span className="goal-sticker pop" aria-label="Sticker collected">
              {VISIT_BY_ID[banner.goal.visit]?.emoji}
            </span>
          ) : (
            <Stars n={banner.goal.stars} />
          )}
        </div>
      )}
      {banner?.kind === 'welcome' && (
        <button
          type="button"
          className="goal-banner welcome dz-stay"
          key={`banner${banner.key}`}
          onClick={() => {
            setBanner(null)
            if (banner.top) tryVariation()
            else if (banner.goal) useGame.getState().showGoal({ id: banner.goal.id })
            else if (banner.visit) useGame.getState().showGoal({ id: `visit:${banner.visit.id}` })
          }}
        >
          <span className="goal-banner-check" aria-label={`${banner.stars} stars`}>
            <IconStar />
            {banner.stars}
          </span>
          <span className="goal-banner-text">
            <small>
              Welcome back!
              {banner.away ? ` ${banner.away}` : ''}
            </small>
            <span className="dz-welcome-line">{banner.line}</span>
            <span className="goal-welcome-next">
              {banner.top ? (
                'Remix this home'
              ) : banner.goal ? (
                <>
                  Next: {banner.goal.title} <span className="goal-nowrap">(+{banner.goal.stars} ★)</span>
                </>
              ) : (
                'Your home missed you.'
              )}
            </span>
            {banner.gone && <span className="goal-visit-line">{banner.gone}</span>}
            {banner.visit && <span className="goal-visit-line">{banner.visit.text}</span>}
          </span>
        </button>
      )}
      {phone && levelUp && levelUp.index < LEVELS.length - 1 && (
        <button type="button" className="goal-banner goal-levelup-mini" key={`lv${levelUp.key}`} onClick={() => useGame.getState().clearLevelUp()}>
          <span className="goal-banner-check">
            <IconStar />
          </span>
          <span className="goal-banner-text">
            <small>Level up! {levelUp.index + 1} of {LEVELS.length}</small>
            {levelUp.name}
            {levelGoalLine && <em>{levelGoalLine}</em>}
            {unlockLine(levelUp) && <em>{unlockLine(levelUp)}</em>}
          </span>
        </button>
      )}
    </div>
  )
}

/** Ends the wait set up by showGoal's unpack branch (selecting the next box after an unpack). */
let stopUnpackFlow = () => {}

/** Opens the drawer on the goal's tab (out of any tool or selection), or the goals sheet at the goal. */
function showGoal(g: Goal) {
  // Unpack your boxes: select the box (its bar offers Unpack), rather than opening the drawer.
  // Tapped again with that box already selected, the chip unpacks it and moves on to the next
  // packed box, so every box gets opened, not only the first.
  if (g.id === 'unpack' && g.near) {
    const st = useStore.getState()
    if (st.mode !== 'select') st.setMode('select')
    useGame.setState({ mode: 'buy', sheet: null, buyHint: null })
    const sel = st.selection
    if (sel?.kind === 'placement' && sel.id === g.near) {
      unpackAndCelebrate(g.near)
      const d = useStore.getState().doc
      const next = [...packedBoxIds(d, activeLayout(d))][0]
      if (next) useStore.setState({ selection: { kind: 'placement', id: next } })
      return
    }
    useStore.setState({ selection: { kind: 'placement', id: g.near } })
    // The box is selected now, so the only step left is its bar's Unpack.
    st.notify('Tap Unpack!')
    // Once that box is unpacked (from its bar), select the next packed box, if any.
    stopUnpackFlow()
    stopUnpackFlow = useGame.subscribe((s, prev) => {
      if (!s.unpacked || s.unpacked === prev.unpacked) return
      stopUnpackFlow()
      const d = useStore.getState().doc
      const next = [...packedBoxIds(d, activeLayout(d))][0]
      if (next) useStore.setState({ selection: { kind: 'placement', id: next } })
    })
    return
  }
  // A remix twist that is a move (Open floor, Fresh start, Talk corner): the chip moves the piece
  // itself to a clean spot that meets the twist (round 16), so Next alone can finish any twist.
  // The piece is selected first, so the move reads as "this one went there"; dragging still works.
  if (g.id.startsWith('twist:') && !g.tab && g.near && activeLayout(useStore.getState().doc).placements.some((p) => p.id === g.near)) {
    const st = useStore.getState()
    if (st.mode !== 'select') st.setMode('select')
    useGame.setState({ mode: 'buy', sheet: null, buyHint: null })
    st.select({ kind: 'placement', id: g.near })
    const docAt = st.doc
    // After a paint: the search scores the home a few times over.
    setTimeout(() => {
      const s = useStore.getState()
      if (s.doc !== docAt) return
      const m = twistMove(s.doc, activeLayout(s.doc))
      if (!m) {
        if (g.hint) s.notify(g.hint)
        return
      }
      const p = activeLayout(s.doc).placements.find((x) => x.id === m.id)
      const name = s.doc.items.find((i) => i.id === p?.itemId)?.name.toLowerCase() ?? 'piece'
      s.updatePlacement(m.id, { x: m.x, y: m.y, rotation: m.rotation })
      s.select({ kind: 'placement', id: m.id })
      const twist = g.id.slice(6) as keyof typeof TWISTS
      s.notify(m.done ? `${TWIST_EMOJI[twist] ?? '↻'} Twist done! Moved the ${name}.` : `Moved the ${name}. ${TWISTS[twist]?.name ?? 'Twist'}: a step closer.`)
    }, 40)
    return
  }
  // A goal fixed by moving a piece (an overlap, a piece in a wall, a blocked door): select that piece.
  const kind = !g.tab ? MOVE_GOAL[g.id] : undefined
  const issue = kind ? lastIssues.find((i) => i.kind === kind && i.severity !== 'info' && i.placementIds.length) : undefined
  // Or a goal about a piece you already have (a coffee table to move in front of the sofa, a desk
  // to move out of the bedroom to make room for a bed): select that piece, never buy another.
  const st0 = useStore.getState()
  const moveId = issue ? issue.placementIds[0] : !g.tab && g.near && activeLayout(st0.doc).placements.some((p) => p.id === g.near) ? g.near : undefined
  if (moveId) {
    const st = st0
    if (st.mode !== 'select') st.setMode('select')
    useGame.setState({ mode: 'buy', sheet: null, buyHint: null })
    st.select({ kind: 'placement', id: moveId })
    // A pinched walkway (round 16): show the walking paths, so the pinch is plain to see.
    if (g.id === 'walkways') {
      if (!st.showPath) st.setUI({ showPath: true })
      if (g.at) useGame.getState().setPinch({ x: g.at.x, y: g.at.y })
    }
    if (g.hint) st.notify(g.hint)
    // Round 16: as with a twist, the chip moves the piece itself when a clean spot finishes the
    // goal (or brings it closer) at no cost in stars; otherwise the piece stays selected to drag.
    if (issue || moveId !== g.near) return
    const docAt = st.doc
    setTimeout(() => {
      const s = useStore.getState()
      if (s.doc !== docAt) return
      const m = goalMove(s.doc, activeLayout(s.doc), g.id)
      if (!m) return
      const p = activeLayout(s.doc).placements.find((x) => x.id === m.id)
      const name = s.doc.items.find((i) => i.id === p?.itemId)?.name.toLowerCase() ?? 'piece'
      // A dining table moves with its chairs (round 17): one step to undo.
      if (m.also?.length) s.beginGesture()
      s.updatePlacement(m.id, { x: m.x, y: m.y, rotation: m.rotation })
      for (const q of m.also ?? []) s.updatePlacement(q.id, { x: q.x, y: q.y, rotation: q.rotation })
      if (m.also?.length) s.endGesture()
      s.select({ kind: 'placement', id: m.id })
      const room = /^Make room for (.+)$/.exec(g.title)
      s.notify(room ? `Moved the ${name}: room for ${room[1]} now.` : m.done ? `Moved the ${name}. ${g.id === 'walkways' ? 'The way is clear!' : `${g.title} ✓`}` : `Moved the ${name}: a step closer.`)
    }, 40)
    return
  }
  if (g.tab) {
    const st = useStore.getState()
    if (st.mode !== 'select') st.setMode('select')
    if (st.selection) st.select(null)
  }
  useGame.getState().showGoal(g)
}

/** Taps this soon after the Dream home card appears are ignored (ms). */
const DREAM_TAP_GUARD_MS = 1500
const CONFETTI = ['var(--g-coral)', 'var(--g-sun)', 'var(--g-sage)', 'var(--g-sky)', 'var(--g-plum)']

/** The level-up moment: a card with confetti, over the HUD, for a few seconds. Tap to close. */
export function LevelUpCard() {
  const levelUp = useGame((s) => s.levelUp)
  const phone = usePhone()
  const dream = !!levelUp && levelUp.index >= LEVELS.length - 1
  const remixDone = useRemixDone(dream)
  // While the pointer is over it or focus is in it, the card stays (WCAG 2.2.1): reading it takes time.
  const [held, setHeld] = useState({ hover: false, focus: false })
  // Only the Dream card holds: the plain level-up covers the whole stage, so a resting mouse would keep it up forever.
  const holding = dream && (held.hover || held.focus)
  useEffect(() => {
    if (!levelUp || holding) return
    // The Dream home card stays a while: it has a button to press.
    const t = setTimeout(() => useGame.getState().clearLevelUp(), dream ? 12000 : 3600)
    return () => clearTimeout(t)
  }, [levelUp, dream, holding])
  // The Dream home card takes focus (it has buttons to press) and gives it back when it closes.
  const cardRef = useRef<HTMLDivElement>(null)
  const showing = !!levelUp && dream
  useEffect(() => {
    if (!showing) return
    const back = document.activeElement as HTMLElement | null
    cardRef.current?.focus({ preventScroll: true })
    return () => {
      if (back?.isConnected && (!document.activeElement || document.activeElement === document.body || cardRef.current?.contains(document.activeElement))) back.focus({ preventScroll: true })
      setHeld({ hover: false, focus: false })
    }
  }, [showing, levelUp?.key])
  // On phones a level-up is a compact banner under the pill (VibeMeter), so the room stays in view.
  // The top level gets the full card everywhere: it's the moment the whole game builds to.
  if (!levelUp || (phone && !dream)) return null
  // The Dream home card ignores taps for a moment (a tap meant for the Next chip mustn't close it),
  // then closes only from the backdrop or its close button.
  const close = (e: ReactMouseEvent) => {
    if (!dream) return useGame.getState().clearLevelUp()
    if (Date.now() - levelUp.key < DREAM_TAP_GUARD_MS) return
    const t = e.target as HTMLElement
    if (t === e.currentTarget || t.closest('.goal-confetti')) useGame.getState().clearLevelUp()
  }
  return (
    <div
      ref={cardRef}
      className={`goal-levelup${dream ? ' feel-dream' : ''}`}
      key={levelUp.key}
      role={dream ? 'dialog' : undefined}
      aria-label={dream ? (remixDone ? 'Remix complete' : 'Dream home reached') : undefined}
      // Read out through the live region (VibeMeter); the brief card is decoration for sighted players.
      aria-hidden={dream ? undefined : true}
      tabIndex={dream ? -1 : undefined}
      onClick={close}
      onMouseEnter={() => setHeld((h) => ({ ...h, hover: true }))}
      onMouseLeave={() => setHeld((h) => ({ ...h, hover: false }))}
      onFocus={() => setHeld((h) => ({ ...h, focus: true }))}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setHeld((h) => ({ ...h, focus: false }))
      }}
      onKeyDown={(e) => {
        // Esc closes; Enter closes too unless it's pressing one of the card's buttons.
        if (e.key === 'Escape' || (e.key === 'Enter' && e.target === e.currentTarget)) {
          e.preventDefault()
          e.stopPropagation()
          useGame.getState().clearLevelUp()
        }
      }}
    >
      <div className="goal-confetti" aria-hidden>
        {Array.from({ length: reduceMotion ? 0 : dream ? 48 : 24 }, (_, i) => (
          <i
            key={i}
            style={{
              left: `${(i * 37) % 100}%`,
              background: CONFETTI[i % CONFETTI.length],
              animationDelay: `${(i % 7) * 0.1}s`,
              animationDuration: `${1.6 + ((i * 13) % 10) / 10}s`,
              transform: `rotate(${i * 47}deg)`,
            }}
          />
        ))}
      </div>
      <div className="goal-levelup-card">
        <div className="goal-levelup-badge">
          <IconStar />
          <IconSparkle className="goal-spark s1" />
          <IconSparkle className="goal-spark s2" />
          <IconSparkle className="goal-spark s3" />
        </div>
        {dream && (
          <button
            type="button"
            className="g-icon sm goal-close"
            aria-label="Close"
            onClick={(e) => {
              e.stopPropagation()
              if (Date.now() - levelUp.key >= DREAM_TAP_GUARD_MS) useGame.getState().clearLevelUp()
            }}
          >
            <IconClose />
          </button>
        )}
        {remixDone ? (
          <>
            <small>Remix complete!</small>
            <strong>{remixDone.name}</strong>
            <em
              className="goal-levelup-goal"
              style={{ display: 'inline-block', alignSelf: 'center', padding: '2px 10px', borderRadius: 999, background: '#dcefd9', color: '#2f6b2a', fontStyle: 'normal', fontWeight: 600 }}
            >
              Twist ✓ {remixDone.twist}
            </em>
            <span>That's your {ordinal(remixDone.count)} Dream home!</span>
          </>
        ) : (
          <>
            <small>{dream ? 'You did it!' : 'Level up!'}</small>
            <strong>{levelUp.name}</strong>
            <span>
              {dream ? `Every level reached: ${LEVELS.length} of ${LEVELS.length}` : `Level ${levelUp.index + 1} of ${LEVELS.length}`}
            </span>
            {levelGoalLine && <em className="goal-levelup-goal">{levelGoalLine}</em>}
            {unlockLine(levelUp) && <em className="goal-levelup-goal">{unlockLine(levelUp)}</em>}
          </>
        )}
        {dream && (
          <button
            type="button"
            className="g-btn primary feel-dream-photo"
            onClick={(e) => {
              e.stopPropagation()
              // The dream photo is taken at dusk, with the lamps on (evening unlocks at Cozy).
              if (levelUp.index >= LEVELS.indexOf('Cozy')) useGame.getState().setEvening(true)
              useGame.getState().requestPhoto()
            }}
          >
            Take its photo
          </button>
        )}
        {remixDone && (
          <button
            type="button"
            className="g-btn ghost"
            onClick={(e) => {
              e.stopPropagation()
              useGame.getState().clearLevelUp()
              useGame.getState().openSheet('layouts')
            }}
          >
            Compare
          </button>
        )}
      </div>
    </div>
  )
}

/**
 * The Dream home card for a remix: its name, its twist and how many Dream homes this home has now
 * (remixes that reached it, plus other designs whose best is Dream home).
 */
function useRemixDone(dream: boolean) {
  const doc = useStore((s) => s.doc)
  const bestLevel = useGame((s) => s.bestLevel)
  const layout = activeLayout(doc)
  if (!dream || !layout.remix) return null
  const top = LEVELS.length - 1
  const count = Math.max(
    1,
    doc.layouts.filter((l) => (l.remix ? !!l.remix.dreamAt || l.id === layout.id : (bestLevel[`${doc.apartment.id}:${l.id}`] ?? 0) >= top)).length,
  )
  return { name: layout.name, twist: TWISTS[layout.remix.twist]?.title ?? '', count }
}

const ordinal = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? 'th' : (['th', 'st', 'nd', 'rd'][n % 10] ?? 'th')}`

function GoalRow({ g, focus }: { g: Goal; focus?: boolean }) {
  const ref = useRef<HTMLLIElement>(null)
  useEffect(() => {
    if (focus) ref.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [focus])
  const cls = `goal-row${g.done ? ' done' : ''}${focus ? ' focus' : ''}`
  const body = (
    <>
      <span className="goal-check">{g.done && <IconCheck />}</span>
      <span className="goal-main">
        <span className="goal-title">
          {g.title}
          {g.visit ? <span className="goal-bonus visit">Visitor</span> : g.bonus && <span className="goal-bonus">Bonus</span>}
        </span>
        {!g.done && g.hint && <span className="goal-hint">{g.hint}</span>}
        {!g.done && (g.progress ?? 0) > 0 && (
          <span className="goal-progress">
            <i style={{ width: `${Math.round((g.progress ?? 0) * 100)}%` }} />
          </span>
        )}
      </span>
      <Stars n={g.stars} filled={g.done} />
    </>
  )
  // Done goals (and the folded "unlocks after" line) are just a record; an open goal is a button
  // that does what Next would: opens the drawer on its piece, or selects the piece to move.
  if (g.done || g.id === 'locked')
    return (
      <li ref={ref} className={cls}>
        {body}
      </li>
    )
  return (
    <li ref={ref} className="goal-row-item">
      <button type="button" className={`${cls} open`} onClick={() => showGoal(g)} aria-label={`${g.title}, ${g.stars} star${g.stars === 1 ? '' : 's'}. Show me`}>
        {body}
      </button>
    </li>
  )
}

/** Goals still waiting on a few pieces of furniture, as one line rather than one row each. */
function lockedRow(locked: Goal[]): Goal {
  return {
    id: 'locked',
    title: `Unlocks after ${MIN_PIECES} pieces`,
    hint: locked.map((g) => g.title).join(' · '),
    done: false,
    progress: Math.max(...locked.map((g) => g.progress ?? 0)),
    stars: locked.reduce((a, g) => a + g.stars, 0),
    category: locked[0].category,
  }
}

export function GoalsSheet() {
  const score = useHomeScore()
  const focus = useGame((s) => s.goalFocus)
  const close = () => useGame.getState().openSheet(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  const fill = levelProgress(score)
  const sheetRef = useRef<HTMLDivElement>(null)
  useModal(sheetRef, close)
  const focused = score.goals.find((g) => g.id === focus)
  const focusId = focused?.locked ? 'locked' : focus
  // One name for the level everywhere: the pill's. The headline only differs when there's nothing yet or a snag.
  const headline = score.headline === score.level.name ? null : score.headline
  // At the top level the bonus goals are what's left to play for.
  const bonus = score.goals.filter((g) => g.bonus)
  const bonusLeft = bonus.filter((g) => !g.done).length
  const bonusStars = bonus.reduce((a, g) => a + (g.done ? g.stars : 0), 0)
  const bonusMax = bonus.reduce((a, g) => a + g.stars, 0)
  // A remix's twist: named under the level, and pinned above the other goals.
  const twist = score.goals.find((g) => g.id.startsWith('twist:'))
  const twistInfo = twist ? TWISTS[twist.id.slice(6) as keyof typeof TWISTS] : undefined
  // Today's visitor: a pill under the level, its own section, and the stickers collected so far.
  const { visit: visitDef, stickers } = useVisit()
  // Visitors and stickers arrive with Cozy, as the Next chip has it.
  const reached = Math.max(score.level.index, score.best)
  const visitGoal = reached >= VISIT_LEVEL ? score.goals.find((g) => g.visit) : undefined
  // Before Cozy the open bonus goals wait behind one "Bonus goals (n)" row, unless one is being shown.
  const [bonusOpen, setBonusOpen] = useState(false)
  const foldBonus = reached < VISIT_LEVEL && !bonusOpen && !focused?.bonus
  const folded = foldBonus ? score.goals.filter((g) => g.bonus && !g.done && !g.visit && !g.urgent && g !== twist) : []
  return (
    <div className="g-scrim" onClick={close}>
      <div ref={sheetRef} className="g-sheet goal-sheet" role="dialog" aria-modal="true" aria-label="Goals" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="g-icon sm goal-close" onClick={close} aria-label="Close">
          <IconClose />
        </button>
        <div className="goal-hero">
          <div className="goal-hero-badge">
            <IconStar />
            <b>{score.stars}</b>
          </div>
          <div className="goal-hero-text">
            <div className="goal-headline">{headline ?? score.level.name}</div>
            <div className="goal-hero-level">
              Level {score.level.index + 1} of {LEVELS.length}
              {headline && (
                <>
                  : <b>{score.level.name}</b>
                </>
              )}
              {score.best > score.level.index && ` · Best: ${LEVELS[score.best]}`}
            </div>
            {twist && twistInfo && (
              <div className={`goal-twist-pill${twist.done ? ' done' : ''}`}>
                ↻ Twist: {twistInfo.title} · {twist.done ? 'done ✓' : `${Math.round((twist.progress ?? 0) * 100)}%`}
              </div>
            )}
            {visitGoal && visitDef && (
              <div className={`goal-visit-pill${visitGoal.done ? ' done' : ''}`}>
                {visitDef.emoji} {visitDef.who} · {visitGoal.done ? 'loved it ✓' : 'visiting today'}
              </div>
            )}
            <div className="goal-meter-bar big">
              <i style={{ width: `${Math.round(fill * 100)}%` }} />
            </div>
            <div className="goal-hero-sub">
              {!score.level.next
                ? bonusLeft
                  ? `Top level reached · ${bonusLeft} bonus goal${bonusLeft === 1 ? '' : 's'} left (★${bonusStars}/${bonusMax})`
                  : 'Top level reached. What a home!'
                : levelBlocker(score)}
            </div>
            {!score.level.next && (
              // At Dream home the chip may be today's visitor: the remix stays one tap away here.
              <button type="button" className="goal-hero-remix" onClick={tryVariation}>
                Remix this home
              </button>
            )}
          </div>
        </div>
        {twist && (
          <section className="goal-group goal-cat-twist">
            <h3>Today’s twist</h3>
            <ul>
              <GoalRow g={twist} focus={twist.id === focusId} />
            </ul>
          </section>
        )}
        {visitGoal && visitDef && (
          <section className="goal-group goal-cat-visit">
            <h3>Today’s visitor</h3>
            <p className="goal-visit-ask">
              {visitDef.emoji} {visitDef.line}
            </p>
            <ul>
              <GoalRow g={visitGoal} focus={visitGoal.id === focusId} />
            </ul>
            <div className="goal-stickers" aria-label={`Stickers: ${stickers.length} of ${VISITS.length}`}>
              <span className="goal-stickers-label">Stickers:</span>
              {stickers.length ? (
                stickers.map((id) => (
                  <span key={id} className="goal-sticker" title={VISIT_BY_ID[id]?.who}>
                    {VISIT_BY_ID[id]?.emoji}
                  </span>
                ))
              ) : (
                <span className="goal-stickers-none">none yet</span>
              )}
              <span className="goal-stickers-count">
                · {stickers.length} of {VISITS.length}
              </span>
            </div>
          </section>
        )}
        {CATEGORY_ORDER.map((cat) => {
          const goals = score.goals.filter((g) => g.category === cat && g !== twist && !g.visit && !folded.includes(g))
          if (!goals.length) return null
          const locked = goals.filter((g) => g.locked && !g.done)
          const rows = locked.length > 1 ? [...goals.filter((g) => !locked.includes(g) && !g.done), lockedRow(locked), ...goals.filter((g) => g.done)] : goals
          return (
            <section key={cat} className={`goal-group goal-cat-${cat}`}>
              <h3>{CATEGORY_LABEL[cat]}</h3>
              <ul>
                {rows.map((g) => (
                  <GoalRow key={g.id} g={g} focus={g.id === focusId} />
                ))}
              </ul>
            </section>
          )
        })}
        {folded.length > 0 && (
          <section className="goal-group goal-cat-bonus">
            <ul>
              <li className="goal-row-item">
                <button type="button" className="goal-row open goal-bonus-fold" onClick={() => setBonusOpen(true)} aria-expanded={false}>
                  <span className="goal-check" />
                  <span className="goal-main">
                    <span className="goal-title">Bonus goals ({folded.length})</span>
                    <span className="goal-hint">{folded.map((g) => g.title).join(' · ')}</span>
                  </span>
                  <Stars n={folded.reduce((a, g) => a + g.stars, 0)} filled={false} />
                </button>
              </li>
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}
