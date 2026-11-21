// Goals HUD: the vibe meter pill (stars, level, progress, celebrations), the next-goal chip, the
// level-up card and the goals sheet.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from 'react'
import { useAnalysis } from '../analysis'
import { catalogEntry, itemFromCatalog } from '../model/catalog'
import { findDrop } from '../model/placement'
import type { Doc } from '../model/types'
import { activeLayout, useStore } from '../store'
import { lastAwayMs, useGame } from './gameStore'
import { bankableLevel, bestLevelKey, CATEGORY_LABEL, CATEGORY_ORDER, LEVELS, levelProgress, MIN_PIECES, nextGoal, packedBoxIds, scoreHome, type Goal, type HomeScore } from './goals'
import { unpackAndCelebrate } from './unpack'
import { IconCheck, IconClose, IconSparkle, IconStar } from './icons'
import { useModal } from './useModal'
import { openRemixPicker } from './Designs'
import { TWISTS } from './goals'
import { reduceMotion } from '../motion'
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
  const raw = useMemo(() => scoreHome(doc, layout, analysis), [doc, layout, analysis])
  lastIssues = analysis.issues
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
    (category: string) => {
      if (refused.has(category)) return false
      let ok = cache.get(category)
      if (ok === undefined) {
        try {
          ok = findDrop(doc, activeLayout(doc), itemFromCatalog(catalogEntry(category), '__probe')).ok
        } catch {
          ok = true
        }
        cache.set(category, ok)
      }
      return ok
    },
    [cache, doc, refused],
  )
}

/** The Next goal, never one whose piece has nowhere to go. */
export function useNextGoal(score: HomeScore, refused: Set<string> = NO_REFUSALS): Goal | undefined {
  const doc = useStore((s) => s.doc)
  const fits = useFits(doc, refused)
  return useMemo(() => nextGoal(score, fits), [score, fits])
}
const NO_REFUSALS = new Set<string>()

/** The latest layout issues, so Next can point at the piece a "move it" goal is about. */
let lastIssues: Issue[] = []
/** Goals fixed by moving a piece, and the issue kind that names that piece. */
const MOVE_GOAL: Record<string, Issue['kind']> = { 'no-overlap': 'overlap', 'in-walls': 'wall', 'doors-clear': 'door' }

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

/** Goals finished in two steps (a desk, then a chair): the first step gets a "½" and a nudge. */
const TWO_STEP = new Set(['work', 'eat', 'nook'])
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

type Banner =
  | { key: number; kind: 'goal'; goal: Goal; more: number }
  /** `line`: "Sample apartment · Move-in day · Cozy, 14★". `top`: at the top level, lead with a variation. */
  | { key: number; kind: 'welcome'; goal?: Goal; home: string; away?: string; line: string; top: boolean }

/** At the top level there's nothing left to climb in this design: a remix (with a twist) is the next game. */
function tryVariation() {
  openRemixPicker()
}

export function VibeMeter() {
  const score = useHomeScore()
  const designKey = useStore((s) => bestLevelKey(s.doc))
  const homeName = useStore((s) => s.doc.apartment.name)
  const designName = useStore((s) => activeLayout(s.doc).name)
  const placed = useStore((s) => activeLayout(s.doc).placements.length)
  const top = score.level.index >= LEVELS.length - 1
  const scoreRef = useRef(score)
  scoreRef.current = score
  const welcomeBack = useGame((s) => s.welcomeBack)
  const levelUp = useGame((s) => s.levelUp)
  const sheet = useGame((s) => s.sheet)
  const phone = usePhone()
  const prev = useRef<{ key: string; stars: number; level: number; done: Set<string>; progress: Map<string, number> } | null>(null)
  const [pop, setPop] = useState<{ key: number; delta: number; half?: boolean } | null>(null)
  const [banner, setBanner] = useState<Banner | null>(null)
  const [shown, setShown] = useState<string | null>(null)
  // Pieces the store refused while the chip's hint was up: Next stops suggesting them until the home changes.
  const [refused, setRefused] = useState<Set<string>>(NO_REFUSALS)
  const next = useNextGoal(score, refused)
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
  useEffect(() => () => clearTimeout(levelTimer.current), [])

  // Layout effect, so a "Placed a sofa" toast that a goal or level-up replaces never paints.
  useLayoutEffect(() => {
    const done = new Set(score.goals.filter((g) => g.done).map((g) => g.id))
    const progress = new Map(score.goals.map((g) => [g.id, g.progress ?? 0]))
    const before = prev.current
    prev.current = { key: designKey, stars: score.stars, level: score.level.index, done, progress }
    // First render, or another design or home: no fanfare.
    if (!before || before.key !== designKey) {
      clearTimeout(levelTimer.current)
      pendingLevel.current = null
      return
    }
    // Any change to the home brings the chip back (with whatever is next now).
    setShown(null)
    setRefused(NO_REFUSALS)
    const fresh = score.goals.filter((g) => g.done && !before.done.has(g.id))
    // The drawer's hint pill ("Tap Plant → Bathroom") points at a goal's piece: once that goal is
    // done (it just finished, or every goal asking for that piece there is done), the pill goes.
    const hint = useGame.getState().buyHint
    if (hint) {
      const asks = (g: Goal) => g.item === hint.category && (!hint.roomId || !g.placeIn || g.placeIn === hint.roomId)
      const behind = score.goals.filter(asks)
      if (fresh.some(asks) || (behind.length > 0 && behind.every((g) => g.done))) useGame.getState().setBuyHint(null)
    }
    if (fresh.length > 4) return
    if (score.stars < before.stars) {
      // A loss: say what broke, offer Undo, and clear any cheer (banner, pop, pending level-up).
      const broke = score.goals.filter((g) => !g.done && before.done.has(g.id)).sort((a, b) => b.stars - a.stars)[0]
      setLoss({ key: Date.now(), delta: before.stars - score.stars, title: broke?.title ?? '' })
      setPop(null)
      setBanner((b) => (b?.kind === 'goal' ? null : b))
      clearTimeout(levelTimer.current)
      pendingLevel.current = null
      if (useGame.getState().levelUp) useGame.getState().clearLevelUp()
      return
    }
    if (score.stars > before.stars) setLoss(null)
    if (score.stars > before.stars) setPop({ key: Date.now(), delta: score.stars - before.stars })
    // Only a level never reached before in this design, on a layout without errors, is celebrated
    // (the best is banked in a passive effect, after this one, so the store still holds the old best).
    const banked = useGame.getState().bestLevel[designKey] ?? 0
    const leveled = score.level.index > before.level && bankableLevel(score) > banked
    // The level-up waits for a pause in placing (each change pushes it back), so the drop shows
    // first and a burst of placements isn't covered up halfway.
    if (leveled) pendingLevel.current = { index: score.level.index, name: score.level.name }
    if (pendingLevel.current) {
      clearTimeout(levelTimer.current)
      levelTimer.current = setTimeout(() => {
        const lv = pendingLevel.current
        pendingLevel.current = null
        if (!lv) return
        // The level-up carries the goal line; the toast steps aside for it.
        useStore.setState({ toast: null })
        useGame.getState().celebrateLevel(lv.index, lv.name)
      }, LEVEL_UP_QUIET_MS)
    }
    const toast = useStore.getState().toast
    const placedToast = !!toast && PLACED_TOAST.test(toast)
    if (fresh.length) {
      const best = [...fresh].sort((a, b) => b.stars - a.stars)[0]
      // One short line per action ("+1★ Every room has something"): the room shows in 3D and the
      // pill's pop has the total.
      const line = `+${best.stars}★ ${best.title}`
      levelGoalLine = line
      if (placedToast) useStore.getState().notify(line)
      else setBanner({ key: Date.now(), kind: 'goal', goal: best, more: fresh.length - 1 })
    } else if (placedToast) {
      // First half of a two-step goal (a desk without its chair): say what finishes it.
      const half = score.goals.find((g) => TWO_STEP.has(g.id) && !g.done && (g.progress ?? 0) > (before.progress.get(g.id) ?? 0) && g.item)
      if (half) {
        const what = toast!.replace(/^Placed (an? )?/, '').replace(/ in the .*$/, '').replace(/\.$/, '')
        useStore.getState().notify(`${what.charAt(0).toUpperCase()}${what.slice(1)} placed. Now add a ${piece(half.item!)} beside it.`)
        setPop({ key: Date.now(), delta: 0, half: true })
      }
    }
  }, [score, designKey])

  // Welcome back, once, after a long break: the goal banner with what to do next.
  useEffect(() => {
    if (!welcomeBack) return
    useGame.getState().consumeWelcomeBack()
    const sc = scoreRef.current
    setBanner({
      key: Date.now(),
      kind: 'welcome',
      goal: nextRef.current,
      home: homeName,
      away: lastAwayMs ? ago(lastAwayMs) : undefined,
      line: `${homeName} · ${designName} · ${sc.level.name}, ${sc.stars}★`,
      // At the top level a remix comes before any bonus goal (only core goals beat it).
      top: sc.level.index >= LEVELS.length - 1 && (!nextRef.current || !!nextRef.current.bonus),
    })
    // Only when the flag is raised; the score at that moment is the one to suggest from.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [welcomeBack])

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
  const chip = next && !(top && next.bonus) && !banner && !levelUp && !sheet && !loss && shown !== next.id ? next : null
  const bestName = score.best > score.level.index ? LEVELS[score.best] : null
  return (
    <div className="goal-meter-wrap">
      <button
        type="button"
        className={`goal-meter${pop ? ' popping' : ''}`}
        onClick={() => useGame.getState().openSheet('goals')}
        aria-label={`${score.stars} stars, ${levelName}. Open goals`}
      >
        <span className="goal-meter-star" key={pop?.key}>
          <IconStar />
          <b>{score.stars}</b>
        </span>
        <span className="goal-meter-body">
          <span className="goal-meter-level">
            {levelName}
            {bestName && <small className="goal-meter-best">Best: {bestName}</small>}
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
          <span className="goal-plus">{pop.half ? '½' : `+${pop.delta}`} ★</span>
        </div>
      )}
      {loss && (
        <div className="feel-loss" key={`loss${loss.key}`} role="status">
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
      {top && (!next || next.bonus) && !banner && !levelUp && !sheet && !loss && (
        <button type="button" className="goal-next" onClick={tryVariation} aria-label="Top level reached. Remix this home with a twist">
          <span className="goal-next-text">
            <b>Remix this home</b>
          </span>
          <span className="goal-next-stars">★</span>
        </button>
      )}
      {chip && (
        <button
          type="button"
          className="goal-next"
          key={chip.id}
          onClick={() => {
            setShown(chip.id)
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
        <div className="goal-banner" key={`banner${banner.key}`} role="status">
          <span className="goal-banner-check">
            <IconCheck />
          </span>
          <span className="goal-banner-text">
            <small>{banner.more ? `${banner.more + 1} goals complete` : 'Goal complete'}</small>
            {banner.goal.title}
          </span>
          <Stars n={banner.goal.stars} />
        </div>
      )}
      {banner?.kind === 'welcome' && (
        <button
          type="button"
          className="goal-banner welcome dz-stay"
          key={`banner${banner.key}`}
          role="status"
          onClick={() => {
            setBanner(null)
            if (banner.top) tryVariation()
            else if (banner.goal) useGame.getState().showGoal({ id: banner.goal.id })
          }}
        >
          <span className="goal-banner-check">
            <IconStar />
          </span>
          <span className="goal-banner-text">
            <small>
              Welcome back!
              {banner.away ? ` ${banner.away}` : ''}
            </small>
            <span className="dz-welcome-line">{banner.line}</span>
            {banner.top ? 'Remix this home' : banner.goal ? `Next: ${banner.goal.title} (+${banner.goal.stars} ★)` : 'Your home missed you.'}
          </span>
        </button>
      )}
      {phone && levelUp && levelUp.index < LEVELS.length - 1 && (
        <button type="button" className="goal-banner goal-levelup-mini" key={`lv${levelUp.key}`} role="status" onClick={() => useGame.getState().clearLevelUp()}>
          <span className="goal-banner-check">
            <IconStar />
          </span>
          <span className="goal-banner-text">
            <small>Level up! {levelUp.index + 1} of {LEVELS.length}</small>
            {levelUp.name}
            {levelUp.name === 'Cozy' ? <em>Evening light unlocked. Try it in Look.</em> : levelGoalLine && <em>{levelGoalLine}</em>}
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
    if (g.hint) st.notify(g.hint)
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
    if (g.hint) st.notify(g.hint)
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
  useEffect(() => {
    if (!levelUp) return
    // The Dream home card stays a while: it has a button to press.
    const t = setTimeout(() => useGame.getState().clearLevelUp(), dream ? 12000 : 3600)
    return () => clearTimeout(t)
  }, [levelUp, dream])
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
    <div className={`goal-levelup${dream ? ' feel-dream' : ''}`} key={levelUp.key} role="status" onClick={close}>
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
            <em className="goal-levelup-goal">Twist ✓ {remixDone.twist}</em>
            <span>Your {ordinal(remixDone.count)} Dream home here</span>
          </>
        ) : (
          <>
            <small>{dream ? 'You did it!' : 'Level up!'}</small>
            <strong>{levelUp.name}</strong>
            <span>
              {dream ? `Every level reached: ${LEVELS.length} of ${LEVELS.length}` : `Level ${levelUp.index + 1} of ${LEVELS.length}`}
            </span>
            {levelGoalLine && <em className="goal-levelup-goal">{levelGoalLine}</em>}
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
  return (
    <li ref={ref} className={`goal-row${g.done ? ' done' : ''}${focus ? ' focus' : ''}`}>
      <span className="goal-check">{g.done && <IconCheck />}</span>
      <div className="goal-main">
        <div className="goal-title">
          {g.title}
          {g.bonus && <span className="goal-bonus">Bonus</span>}
        </div>
        {!g.done && g.hint && <div className="goal-hint">{g.hint}</div>}
        {!g.done && (g.progress ?? 0) > 0 && (
          <div className="goal-progress">
            <i style={{ width: `${Math.round((g.progress ?? 0) * 100)}%` }} />
          </div>
        )}
      </div>
      <Stars n={g.stars} filled={g.done} />
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
            <div className="goal-meter-bar big">
              <i style={{ width: `${Math.round(fill * 100)}%` }} />
            </div>
            <div className="goal-hero-sub">
              {!score.level.next
                ? bonusLeft
                  ? `Top level reached · ${bonusLeft} bonus goal${bonusLeft === 1 ? '' : 's'} left (★${bonusStars}/${bonusMax})`
                  : 'Top level reached. What a home!'
                : score.level.toNext > 0
                  ? `${score.level.toNext} more ★ to ${score.level.next}`
                  : `Fix the snag to reach ${score.level.next}`}
            </div>
          </div>
        </div>
        {CATEGORY_ORDER.map((cat) => {
          const goals = score.goals.filter((g) => g.category === cat)
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
      </div>
    </div>
  )
}
