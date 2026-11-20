import { useEffect, useRef, useState } from 'react'
import { create } from 'zustand'
import { useStore } from '../store'
import { RENDER_STYLES, type RenderStyle } from '../three/style'
import { useGame } from './gameStore'
import './style-picker.css'

// The "map layers" control: a small round button showing the current look. Tapping it fans out
// one illustrated tile per rendering style.

/** A tiny room corner (wall, floor, sofa, plant) drawn the way the style draws it. */
export function StyleSwatch({ id }: { id: RenderStyle }) {
  const ink = id === 'toon' ? '#3d2c25' : id === 'blueprint' ? '#e8f2ff' : 'none'
  const sw = id === 'toon' ? 1.6 : id === 'blueprint' ? 1.2 : 0
  const fill = (toon: string, real: string, clay: string) => (id === 'toon' ? toon : id === 'real' ? real : id === 'clay' ? clay : id === 'blueprint' ? 'rgba(93,149,224,0.25)' : 'none')
  const bg = id === 'scan' ? '#1d2129' : id === 'blueprint' ? '#1c4a8a' : id === 'toon' ? '#fde9cf' : id === 'real' ? '#e9edf0' : '#e7e3dd'
  return (
    <svg viewBox="0 0 64 64" aria-hidden className="sty-swatch">
      <defs>
        <linearGradient id="sty-real-wall" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#f3efe9" />
          <stop offset="1" stopColor="#d9d3ca" />
        </linearGradient>
        <linearGradient id="sty-real-floor" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#c99a68" />
          <stop offset="1" stopColor="#a87849" />
        </linearGradient>
        <linearGradient id="sty-real-sofa" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#7f93ad" />
          <stop offset="1" stopColor="#56677f" />
        </linearGradient>
        <radialGradient id="sty-clay" cx="0.35" cy="0.3" r="0.9">
          <stop offset="0" stopColor="#fbf9f6" />
          <stop offset="1" stopColor="#d6d0c7" />
        </radialGradient>
      </defs>
      <rect width="64" height="64" fill={bg} />
      {id === 'scan' ? (
        <g>
          {SCAN_DOTS.map(([x, y, c], i) => (
            <circle key={i} cx={x} cy={y} r={1.5} fill={c} opacity={0.9} />
          ))}
        </g>
      ) : (
        <g stroke={ink} strokeWidth={sw} strokeLinejoin="round">
          {id === 'blueprint' && (
            <g stroke="#3a6fb5" strokeWidth={0.6}>
              {[8, 16, 24, 32, 40, 48, 56].map((k) => (
                <path key={k} d={`M${k} 0V64M0 ${k}H64`} />
              ))}
            </g>
          )}
          {/* back wall and side wall */}
          <path d="M6 10 L40 4 L40 36 L6 44 Z" fill={id === 'real' ? 'url(#sty-real-wall)' : fill('#f6d7b0', '', id === 'clay' ? 'url(#sty-clay)' : '')} />
          <path d="M40 4 L58 12 L58 44 L40 36 Z" fill={fill('#efc597', '#d8d1c7', '#e2ddd5')} />
          {/* floor */}
          <path d="M6 44 L40 36 L58 44 L24 58 Z" fill={id === 'real' ? 'url(#sty-real-floor)' : fill('#d9a874', '', '#ebe6de')} />
          {id === 'real' && <ellipse cx="25" cy="46" rx="15" ry="3.5" fill="rgba(40,25,15,0.25)" />}
          {/* sofa */}
          <path d="M12 34 L36 29 L36 40 L12 46 Z" fill={id === 'real' ? 'url(#sty-real-sofa)' : fill('#ff7a59', '', id === 'clay' ? 'url(#sty-clay)' : '')} />
          <path d="M12 40 L36 35 L40 38 L16 44 Z" fill={fill('#ff9a7d', '#8ea2bb', '#f4f1ec')} />
          {/* plant */}
          <path d="M47 38 L52 38 L51 45 L48 45 Z" fill={fill('#c47a4e', '#9b6a4c', '#e6e1d9')} />
          <circle cx="49.5" cy="32" r="6" fill={fill('#6cc08b', '#5d8a5f', id === 'clay' ? 'url(#sty-clay)' : '')} />
          {id === 'toon' && <path d="M45 30 q2 -3 5 -2" stroke="#fff" strokeWidth={1.2} fill="none" opacity={0.8} />}
        </g>
      )}
    </svg>
  )
}

const SCAN_DOTS: [number, number, string][] = (() => {
  // A fixed speckle pattern that reads as a point cloud of the same corner.
  const out: [number, number, string][] = []
  let s = 3
  const r = () => ((s = (s * 16807) % 2147483647) / 2147483647)
  for (let i = 0; i < 150; i++) {
    const x = 4 + r() * 56
    const y = 4 + r() * 56
    const floor = y > 40 + (x < 40 ? -(x - 6) * 0.23 : (x - 40) * 0.44) - 4
    const sofa = x > 12 && x < 36 && y > 30 && y < 45
    const c = sofa ? '#d98a6a' : floor ? '#b88b5d' : x > 40 ? '#cfc6b8' : '#e8dfd2'
    out.push([x, y, c])
  }
  return out
})()

/**
 * Whether a style picker is open. The HUD tucks the furniture drawer away meanwhile, so on a phone
 * the tiles and their names aren't hidden under it.
 */
export const useStylePicker = create<{ open: boolean; setOpen: (open: boolean) => void }>((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
}))

export function StylePicker({ className = '' }: { className?: string }) {
  const style = useGame((s) => s.renderStyle)
  const hasScan = useStore((s) => (s.doc.scans ?? []).length > 0)
  const [open, setOpen] = useState(false)
  useEffect(() => {
    useStylePicker.getState().setOpen(open)
  }, [open])
  useEffect(() => () => useStylePicker.getState().setOpen(false), [])
  const [hint, setHint] = useState<RenderStyle | null>(null)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('pointerdown', close)
    window.addEventListener('keydown', esc)
    return () => {
      window.removeEventListener('pointerdown', close)
      window.removeEventListener('keydown', esc)
    }
  }, [open])

  useEffect(() => {
    if (!hint) return
    const t = setTimeout(() => setHint(null), 1800)
    return () => clearTimeout(t)
  }, [hint])

  const pick = (id: RenderStyle) => {
    useGame.getState().setRenderStyle(id)
    setOpen(false)
    setHint(id)
  }
  const current = RENDER_STYLES.find((s) => s.id === style)!
  const hintInfo = hint && RENDER_STYLES.find((s) => s.id === hint)

  return (
    <div className={`sty ${className}`} ref={ref}>
      <button className={`sty-btn${open ? ' open' : ''}`} onClick={() => setOpen((o) => !o)} aria-label={`Style: ${current.label}. Change how the room is drawn`} aria-expanded={open} title="Change the style">
        <StyleSwatch id={style} />
        <span className="sty-badge" aria-hidden>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.6} strokeLinecap="round" strokeLinejoin="round">
            <path d="m12 3 9 5-9 5-9-5zM3 13l9 5 9-5" />
          </svg>
        </span>
      </button>
      {open && (
        <div className="sty-pop g-card" role="menu" aria-label="Styles">
          {RENDER_STYLES.map((s, i) => {
            const disabled = s.id === 'scan' && !hasScan
            return (
              <button
                key={s.id}
                role="menuitemradio"
                aria-checked={style === s.id}
                className={`sty-tile${style === s.id ? ' on' : ''}`}
                style={{ animationDelay: `${i * 35}ms` }}
                disabled={disabled}
                title={disabled ? 'Add a scan of your place to see it here' : s.blurb}
                onClick={() => pick(s.id)}
              >
                <span className="sty-tile-art">
                  <StyleSwatch id={s.id} />
                </span>
                <span className="sty-tile-label">{s.label}</span>
              </button>
            )
          })}
        </div>
      )}
      {hintInfo && !open && (
        <div className="sty-hint" key={hintInfo.id} role="status">
          <b>{hintInfo.label}</b> {hintInfo.blurb}
        </div>
      )}
    </div>
  )
}
