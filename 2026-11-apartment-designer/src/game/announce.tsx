import { useEffect, useState } from 'react'

/**
 * One always-present pair of live regions for screen readers. Toasts, banners, level-ups and star
 * changes come and go on screen; a live region that is created at the same moment as its text is
 * often not read, so the visual pieces stay quiet and their words are sent here instead.
 */
export type Politeness = 'polite' | 'assertive'

/** Read "★" as "stars": "+2★" → "+2 stars", "1★" → "1 star", a lone ★ → "stars". */
export function spellStars(text: string): string {
  return text
    .replace(/(\d+)\s*★/g, (_, n: string) => `${n} star${n === '1' ? '' : 's'}`)
    .replace(/½\s*★/g, 'half a star')
    .replace(/★/g, 'stars')
    .replace(/−/g, '-')
    .replace(/\s+/g, ' ')
    .trim()
}

type Listener = (p: Politeness, text: string) => void
const listeners = new Set<Listener>()
const pending: Record<Politeness, string[]> = { polite: [], assertive: [] }
let timer: ReturnType<typeof setTimeout> | undefined
/** Messages sent within this window are read together, in order, as one announcement. */
export const ANNOUNCE_BATCH_MS = 150

function flush() {
  timer = undefined
  for (const p of ['assertive', 'polite'] as const) {
    const text = pending[p].join(' ')
    pending[p] = []
    if (text) listeners.forEach((l) => l(p, text))
  }
}

/** Say something to screen reader users (nothing visible). Repeats of the same text are dropped within a batch. */
export function announce(text: string | null | undefined, politeness: Politeness = 'polite') {
  const t = text ? spellStars(text) : ''
  if (!t) return
  const q = pending[politeness]
  if (!q.includes(t)) q.push(t)
  if (!timer) timer = setTimeout(flush, ANNOUNCE_BATCH_MS)
}

/** Subscribe to announcements (the live region does; tests too). */
export function onAnnounce(l: Listener) {
  listeners.add(l)
  return () => void listeners.delete(l)
}

/** Clears a region a while after it speaks, so the same words said again are read again. */
const CLEAR_MS = 5000

export function LiveRegion() {
  const [text, setText] = useState<Record<Politeness, string>>({ polite: '', assertive: '' })
  useEffect(() => {
    const clears: Partial<Record<Politeness, ReturnType<typeof setTimeout>>> = {}
    const off = onAnnounce((p, t) => {
      clearTimeout(clears[p])
      // Empty first, then the words on the next tick: a repeat of the last message is still a change.
      setText((s) => ({ ...s, [p]: '' }))
      clears[p] = setTimeout(() => {
        setText((s) => ({ ...s, [p]: t }))
        clears[p] = setTimeout(() => setText((s) => ({ ...s, [p]: '' })), CLEAR_MS)
      }, 50)
    })
    return () => {
      off()
      Object.values(clears).forEach((c) => clearTimeout(c))
    }
  }, [])
  return (
    <div className="sr-only">
      <div role="status" aria-live="polite" aria-atomic="true" data-testid="live-polite">
        {text.polite}
      </div>
      <div role="alert" aria-live="assertive" aria-atomic="true" data-testid="live-assertive">
        {text.assertive}
      </div>
    </div>
  )
}
