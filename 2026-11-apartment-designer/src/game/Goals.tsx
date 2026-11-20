// Goals HUD: the vibe meter pill (stars, level, progress, celebrations) and the goals sheet.
import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { useAnalysis } from '../analysis'
import { activeLayout, useStore } from '../store'
import { useGame } from './gameStore'
import { bankableLevel, CATEGORY_LABEL, CATEGORY_ORDER, levelProgress, scoreHome, withBestLevel, type Goal, type HomeScore } from './goals'
import { IconCheck, IconClose, IconStar } from './icons'
import { useModal } from './useModal'

/** The home's score, with the level held at the best one reached in this home (it never drops). */
export function useHomeScore(): HomeScore {
  const doc = useStore((s) => s.doc)
  const layout = activeLayout(doc)
  const analysis = useAnalysis(doc, layout)
  const raw = useMemo(() => scoreHome(doc, layout, analysis), [doc, layout, analysis])
  const home = doc.apartment.id
  const best = useGame((s) => s.bestLevel[home] ?? 0)
  const bank = bankableLevel(raw)
  useEffect(() => {
    if (bank > best) useGame.getState().noteLevel(home, bank)
  }, [home, bank, best])
  return useMemo(() => withBestLevel(raw, Math.max(best, bank)), [raw, best, bank])
}

const SPARKS = 10

function Stars({ n, filled = true }: { n: number; filled?: boolean }) {
  return (
    <span className={`goal-stars${filled ? ' on' : ''}`} aria-label={`${n} star${n === 1 ? '' : 's'}`}>
      {Array.from({ length: n }, (_, i) => (
        <IconStar key={i} />
      ))}
    </span>
  )
}

export function VibeMeter() {
  const score = useHomeScore()
  const prevStars = useRef<number | null>(null)
  const prevDone = useRef<Set<string> | null>(null)
  const [pop, setPop] = useState<{ key: number; delta: number } | null>(null)
  const [banner, setBanner] = useState<{ key: number; goal: Goal; more: number } | null>(null)

  useEffect(() => {
    const done = new Set(score.goals.filter((g) => g.done).map((g) => g.id))
    const before = prevDone.current
    const was = prevStars.current
    prevDone.current = done
    prevStars.current = score.stars
    // First render, or a whole new document loaded: no fanfare.
    if (before === null || was === null) return
    const fresh = score.goals.filter((g) => g.done && !before.has(g.id))
    if (fresh.length > 4) return
    const timers: ReturnType<typeof setTimeout>[] = []
    if (score.stars > was) {
      setPop({ key: Date.now(), delta: score.stars - was })
      timers.push(setTimeout(() => setPop(null), 1300))
    }
    if (fresh.length) {
      const best = [...fresh].sort((a, b) => b.stars - a.stars)[0]
      setBanner({ key: Date.now(), goal: best, more: fresh.length - 1 })
      timers.push(setTimeout(() => setBanner(null), 2600))
    }
    return () => timers.forEach(clearTimeout)
  }, [score])

  const fill = levelProgress(score)
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
      {banner && (
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
    </div>
  )
}

function GoalRow({ g }: { g: Goal }) {
  return (
    <li className={`goal-row${g.done ? ' done' : ''}`}>
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

export function GoalsSheet() {
  const score = useHomeScore()
  const close = () => useGame.getState().openSheet(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  const fill = levelProgress(score)
  const sheetRef = useRef<HTMLDivElement>(null)
  useModal(sheetRef, close)
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
            <div className="goal-headline">{score.headline}</div>
            <div className="goal-hero-level">
              Level {score.level.index + 1}: <b>{score.level.name}</b>
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
          return (
            <section key={cat} className={`goal-group goal-cat-${cat}`}>
              <h3>{CATEGORY_LABEL[cat]}</h3>
              <ul>
                {goals.map((g) => (
                  <GoalRow key={g.id} g={g} />
                ))}
              </ul>
            </section>
          )
        })}
      </div>
    </div>
  )
}
