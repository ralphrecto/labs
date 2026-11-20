// Goals HUD: the vibe meter pill (stars, level, progress, celebrations), the next-goal chip, the
// level-up card and the goals sheet.
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { useAnalysis } from '../analysis'
import { activeLayout, useStore } from '../store'
import { useGame } from './gameStore'
import { bankableLevel, bestLevelKey, CATEGORY_LABEL, CATEGORY_ORDER, LEVELS, levelProgress, MIN_PIECES, nextGoal, scoreHome, withBestLevel, type Goal, type HomeScore } from './goals'
import { IconCheck, IconClose, IconSparkle, IconStar } from './icons'
import { useModal } from './useModal'

/**
 * The home's score, with the level held at the best one reached in this design (it never drops).
 * Each design keeps its own best, so a fresh "what if" design starts from its own score.
 */
export function useHomeScore(): HomeScore {
  const doc = useStore((s) => s.doc)
  const layout = activeLayout(doc)
  const analysis = useAnalysis(doc, layout)
  const raw = useMemo(() => scoreHome(doc, layout, analysis), [doc, layout, analysis])
  const key = bestLevelKey(doc)
  // Saves from before per-design levels kept one best per home: it carries over to the first design only.
  const legacy = doc.layouts[0]?.id === layout.id ? doc.apartment.id : null
  const best = useGame((s) => s.bestLevel[key] ?? (legacy ? s.bestLevel[legacy] : undefined) ?? 0)
  const bank = bankableLevel(raw)
  useEffect(() => {
    if (bank > best) useGame.getState().noteLevel(key, bank)
  }, [key, bank, best])
  return useMemo(() => withBestLevel(raw, Math.max(best, bank)), [raw, best, bank])
}

const SPARKS = 10
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

type Banner = { key: number; kind: 'goal'; goal: Goal; more: number } | { key: number; kind: 'welcome'; goal?: Goal; home: string }

export function VibeMeter() {
  const score = useHomeScore()
  const designKey = useStore((s) => bestLevelKey(s.doc))
  const homeName = useStore((s) => s.doc.apartment.name)
  const welcomeBack = useGame((s) => s.welcomeBack)
  const levelUp = useGame((s) => s.levelUp)
  const sheet = useGame((s) => s.sheet)
  const prev = useRef<{ key: string; stars: number; level: number; done: Set<string> } | null>(null)
  const [pop, setPop] = useState<{ key: number; delta: number } | null>(null)
  const [banner, setBanner] = useState<Banner | null>(null)
  const [shown, setShown] = useState<string | null>(null)
  const next = nextGoal(score)
  const levelTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const pendingLevel = useRef<{ index: number; name: string } | null>(null)
  useEffect(() => () => clearTimeout(levelTimer.current), [])

  // Layout effect, so a "Placed a sofa" toast that a goal or level-up replaces never paints.
  useLayoutEffect(() => {
    const done = new Set(score.goals.filter((g) => g.done).map((g) => g.id))
    const before = prev.current
    prev.current = { key: designKey, stars: score.stars, level: score.level.index, done }
    // First render, or another design or home: no fanfare.
    if (!before || before.key !== designKey) {
      clearTimeout(levelTimer.current)
      pendingLevel.current = null
      return
    }
    // Any change to the home brings the chip back (with whatever is next now).
    setShown(null)
    const fresh = score.goals.filter((g) => g.done && !before.done.has(g.id))
    if (fresh.length > 4) return
    if (score.stars > before.stars) setPop({ key: Date.now(), delta: score.stars - before.stars })
    const leveled = score.level.index > before.level
    // The level-up card waits for a pause in placing (each change pushes it back), so a burst of
    // placements isn't covered up halfway.
    if (leveled) pendingLevel.current = { index: score.level.index, name: score.level.name }
    if (pendingLevel.current) {
      clearTimeout(levelTimer.current)
      levelTimer.current = setTimeout(() => {
        const lv = pendingLevel.current
        pendingLevel.current = null
        if (lv) useGame.getState().celebrateLevel(lv.index, lv.name)
      }, LEVEL_UP_QUIET_MS)
    }
    if (fresh.length) {
      const best = [...fresh].sort((a, b) => b.stars - a.stars)[0]
      const toast = useStore.getState().toast
      // One line per action: "Placed a bed in the Bedroom · Goal: A restful bed +2★" rather than
      // the goal banner replacing the room confirmation.
      if (toast && PLACED_TOAST.test(toast) && !toast.includes(' · Goal: ')) {
        const more = fresh.length > 1 ? ` (+${fresh.length - 1} more)` : ''
        useStore.getState().notify(`${toast.replace(/\.$/, '')} · Goal: ${best.title} +${best.stars}★${more}`)
      } else setBanner({ key: Date.now(), kind: 'goal', goal: best, more: fresh.length - 1 })
    }
  }, [score, designKey])

  // Welcome back, once, after a long break: the goal banner with what to do next.
  useEffect(() => {
    if (!welcomeBack) return
    useGame.getState().consumeWelcomeBack()
    setBanner({ key: Date.now(), kind: 'welcome', goal: nextGoal(score), home: homeName })
    // Only when the flag is raised; the score at that moment is the one to suggest from.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [welcomeBack])

  useEffect(() => {
    if (!pop) return
    const t = setTimeout(() => setPop(null), 1300)
    return () => clearTimeout(t)
  }, [pop])
  useEffect(() => {
    if (!banner) return
    const t = setTimeout(() => setBanner(null), banner.kind === 'welcome' ? 6000 : 2600)
    return () => clearTimeout(t)
  }, [banner])

  const fill = levelProgress(score)
  // One quiet chip with the next goal. Tapping it steps it aside until the home changes.
  const chip = next && !banner && !levelUp && !sheet && shown !== next.id ? next : null
  return (
    <div className="goal-meter-wrap">
      <button
        type="button"
        className={`goal-meter${pop ? ' popping' : ''}`}
        onClick={() => useGame.getState().openSheet('goals')}
        aria-label={`${score.stars} stars, ${score.level.name}. Open goals`}
      >
        <span className="goal-meter-star" key={pop?.key}>
          <IconStar />
          <b>{score.stars}</b>
        </span>
        <span className="goal-meter-body">
          <span className="goal-meter-level">{score.level.name}</span>
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
          <span className="goal-plus">+{pop.delta} ★</span>
        </div>
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
            <small>Goal complete{banner.more ? ` (+${banner.more} more)` : ''}</small>
            {banner.goal.title}
          </span>
          <Stars n={banner.goal.stars} />
        </div>
      )}
      {banner?.kind === 'welcome' && (
        <button
          type="button"
          className="goal-banner welcome"
          key={`banner${banner.key}`}
          role="status"
          onClick={() => {
            setBanner(null)
            if (banner.goal) useGame.getState().showGoal({ id: banner.goal.id })
          }}
        >
          <span className="goal-banner-check">
            <IconStar />
          </span>
          <span className="goal-banner-text">
            <small>Welcome back{banner.goal ? '!' : ` to ${banner.home}!`}</small>
            {banner.goal ? `Next: ${banner.goal.title} (+${banner.goal.stars} ★)` : 'Your home missed you.'}
          </span>
        </button>
      )}
    </div>
  )
}

/** Opens the drawer on the goal's tab (out of any tool or selection), or the goals sheet at the goal. */
function showGoal(g: Goal) {
  if (g.tab) {
    const st = useStore.getState()
    if (st.mode !== 'select') st.setMode('select')
    if (st.selection) st.select(null)
  }
  useGame.getState().showGoal(g)
}

const CONFETTI = ['var(--g-coral)', 'var(--g-sun)', 'var(--g-sage)', 'var(--g-sky)', 'var(--g-plum)']

/** The level-up moment: a card with confetti, over the HUD, for a few seconds. Tap to close. */
export function LevelUpCard() {
  const levelUp = useGame((s) => s.levelUp)
  useEffect(() => {
    if (!levelUp) return
    const t = setTimeout(() => useGame.getState().clearLevelUp(), 3600)
    return () => clearTimeout(t)
  }, [levelUp])
  if (!levelUp) return null
  return (
    <div className="goal-levelup" key={levelUp.key} role="status" onClick={() => useGame.getState().clearLevelUp()}>
      <div className="goal-confetti" aria-hidden>
        {Array.from({ length: 24 }, (_, i) => (
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
        <small>Level up!</small>
        <strong>{levelUp.name}</strong>
        <span>
          Level {levelUp.index + 1} of {LEVELS.length}
        </span>
      </div>
    </div>
  )
}

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
            </div>
            <div className="goal-meter-bar big">
              <i style={{ width: `${Math.round(fill * 100)}%` }} />
            </div>
            <div className="goal-hero-sub">
              {!score.level.next
                ? 'Top level reached. What a home!'
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
