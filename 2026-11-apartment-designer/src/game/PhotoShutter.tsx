// Look mode's photo moment: one shutter button that snaps the 3D view, frames it as a polaroid
// with the home's name, level and stars, and offers to share or save it.
import { useEffect, useState } from 'react'
import { flushSync } from 'react-dom'
import { useStore } from '../store'
import { useHomeScore } from './Goals'
import { reduceMotion } from '../motion'
import './photo.css'

interface Snap {
  url: string
  blob: Blob | null
  file: string
}

const roundRect = (g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) => {
  g.beginPath()
  g.moveTo(x + r, y)
  g.arcTo(x + w, y, x + w, y + h, r)
  g.arcTo(x + w, y + h, x, y + h, r)
  g.arcTo(x, y + h, x, y, r)
  g.arcTo(x, y, x + w, y, r)
  g.closePath()
}

/** Draw the polaroid: the shot cropped to a 4:3-ish window, a paper frame and the caption. */
export function composePolaroid(shot: HTMLCanvasElement, name: string, level: string, stars: number, date = new Date()): HTMLCanvasElement {
  const W = 1080
  const pad = 48
  const photoW = W - pad * 2
  const photoH = Math.round(photoW * 0.92)
  const capH = 262
  const out = document.createElement('canvas')
  out.width = W
  out.height = pad + photoH + capH
  const g = out.getContext('2d')!
  g.fillStyle = '#fffdf8'
  g.fillRect(0, 0, out.width, out.height)
  // Cover-crop the shot into the photo window.
  const sw = shot.width
  const sh = shot.height
  const k = Math.max(photoW / sw, photoH / sh)
  const cw = photoW / k
  const ch = photoH / k
  g.save()
  roundRect(g, pad, pad, photoW, photoH, 10)
  g.clip()
  g.drawImage(shot, (sw - cw) / 2, (sh - ch) / 2, cw, ch, pad, pad, photoW, photoH)
  g.restore()
  const font = getComputedStyle(document.documentElement).getPropertyValue('--g-font').trim() || 'system-ui, sans-serif'
  g.fillStyle = '#3b2f26'
  g.textBaseline = 'alphabetic'
  g.font = `800 62px ${font}`
  const y0 = pad + photoH + 92
  let title = name
  while (g.measureText(title).width > photoW && title.length > 4) title = title.slice(0, -2)
  if (title !== name) title = title.trimEnd() + '…'
  g.fillText(title, pad + 4, y0)
  // the date, hand-written small under the home's name, then the level
  g.font = `700 34px ${font}`
  g.fillStyle = '#a89282'
  g.fillText(date.toLocaleDateString(undefined, { day: 'numeric', month: 'long', year: 'numeric' }), pad + 4, y0 + 52)
  g.font = `700 40px ${font}`
  g.fillStyle = '#8a7465'
  g.fillText(level, pad + 4, y0 + 112)
  const starTxt = `★ ${stars}`
  g.font = `800 48px ${font}`
  const tw = g.measureText(starTxt).width
  g.fillStyle = '#f3b53c'
  roundRect(g, W - pad - tw - 44, y0 + 64, tw + 44, 66, 33)
  g.fill()
  g.fillStyle = '#5a3a08'
  g.fillText(starTxt, W - pad - tw - 22, y0 + 114)
  return out
}

const fileName = (name: string) => `${name.replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '-').toLowerCase() || 'home'}.png`

/** The shutter (only in Look mode) and the polaroid card it opens. */
export function PhotoShutter() {
  const name = useStore((s) => s.doc.apartment.name)
  const score = useHomeScore()
  const [snap, setSnap] = useState<Snap | null>(null)
  const [flash, setFlash] = useState(0)
  useEffect(() => () => void (snap && URL.revokeObjectURL(snap.url)), [snap])

  const take = () => {
    const canvas = document.querySelector<HTMLCanvasElement>('.hud-stage canvas')
    if (!canvas) return
    // The renderer keeps its drawing buffer (preserveDrawingBuffer), so the last frame is still there.
    const level = `Level ${score.level.index + 1} · ${score.level.name}`
    // Flash first, on the frame of the tap: composing the polaroid (reading the WebGL canvas back)
    // is slow on phones, so it waits until the flash has painted. The DOM flash never touches the
    // 3D canvas, and the renderer keeps its drawing buffer, so the shot is the room as it was.
    if (!reduceMotion) flushSync(() => setFlash((f) => f + 1))
    const compose = () => {
      const card = composePolaroid(canvas, name, level, score.stars)
      card.toBlob((blob) => {
        const url = blob ? URL.createObjectURL(blob) : card.toDataURL('image/png')
        setSnap({ url, blob, file: fileName(name) })
      }, 'image/png')
    }
    if (reduceMotion) compose()
    else requestAnimationFrame(() => requestAnimationFrame(compose))
  }

  const file = snap?.blob ? new File([snap.blob], snap.file, { type: 'image/png' }) : null
  const canShare = !!file && typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] })
  const share = async () => {
    if (!file) return
    try {
      await navigator.share({ files: [file], title: name })
    } catch {
      /* cancelled */
    }
  }
  const download = () => {
    if (!snap) return
    const a = document.createElement('a')
    a.href = snap.url
    a.download = snap.file
    a.click()
  }

  return (
    <>
      {!snap && <button className="photo-shutter" onClick={take} aria-label="Take a photo" title="Take a photo">
        <svg viewBox="0 0 24 24" width={22} height={22} fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M4 8h3l1.6-2.4h6.8L17 8h3v11H4z" />
          <circle cx="12" cy="13.2" r="3.4" />
        </svg>
      </button>}
      {flash > 0 && <div key={flash} className="photo-flash" aria-hidden />}
      {snap && (
        <div className="photo-sheet" role="dialog" aria-label="Your photo" onClick={() => setSnap(null)}>
          <div className="photo-card" onClick={(e) => e.stopPropagation()}>
            <img src={snap.url} alt={`${name}, ${score.level.name}, ${score.stars} stars`} />
            <div className="photo-actions">
              {canShare ? (
                <button className="g-btn primary" onClick={share}>
                  Share
                </button>
              ) : null}
              <button className={`g-btn${canShare ? '' : ' primary'}`} onClick={download}>
                Save
              </button>
              <button className="g-btn ghost" onClick={() => setSnap(null)}>
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
