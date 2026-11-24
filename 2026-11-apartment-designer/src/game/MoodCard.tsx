// Mood moments (round 17): the one card that asks how a room should feel, and the soft light a
// chosen mood casts in the 3D view. The logic (which room, which goal, which colours) is in moods.ts.
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import * as THREE from 'three'
import { activeLayout, useStore } from '../store'
import type { Doc, MoodId, Room } from '../model/types'
import { bestLevelKey } from './goals'
import { useGame } from './gameStore'
import { dueMoment, momentKey, moodGoals, moodLightSpot, moodOf, MOOD_IDS, MOODS, pickMood, piecesOf, withMood } from './moods'
import { setPickMoodSource } from './swatches'
import { announce } from './announce'
import { reduceMotion } from '../motion'
import './moods.css'

// The pick cards take their room's mood (or the design's latest) without the drawer knowing:
// picksFor asks this when it isn't given a mood.
setPickMoodSource(() => {
  const doc = useStore.getState().doc
  return pickMood(activeLayout(doc), useGame.getState().buyHint?.roomId)
})

/** How long the game stays calm (no card, sheet or drag) before a due moment shows. */
export const MOMENT_DELAY_MS = 1400
/** Pieces to place after one mood card before the next due one shows, so two never come back to back. */
export const MOMENT_GAP_PIECES = 3

/** How long the view stays on a room just given a mood before it steps back to the whole home. */
export const MOOD_FOCUS_MS = 2600

// The room a mood was just chosen for: the 3D view eases to it for a moment (not with reduced
// motion), so the player sees the light change. A tiny store of its own; GameApp reads it.
let focusRoom: string | null = null
let focusTimer: ReturnType<typeof setTimeout> | undefined
const focusListeners = new Set<() => void>()
function setMoodFocus(roomId: string | null) {
  focusRoom = roomId
  focusListeners.forEach((f) => f())
}
/** The room the view is easing to after a mood was chosen, or null. */
export function useMoodFocus(): string | null {
  return useSyncExternalStore(
    (f) => {
      focusListeners.add(f)
      return () => void focusListeners.delete(f)
    },
    () => focusRoom,
  )
}
function focusMoodRoom(roomId: string) {
  if (reduceMotion) return
  clearTimeout(focusTimer)
  setMoodFocus(roomId)
  focusTimer = setTimeout(() => setMoodFocus(null), MOOD_FOCUS_MS)
}

/** A mood card closed (chosen or waved off): the next due one waits a few pieces. */
function restMoments() {
  useGame.setState({ moodQuiet: activeLayout(useStore.getState().doc).placements.length + MOMENT_GAP_PIECES })
}

/** Set a room's mood on the active design (undoable, like any edit), and say what it brings. */
export function chooseMood(roomId: string, mood: MoodId) {
  const st = useStore.getState()
  const room = st.doc.rooms.find((r) => r.id === roomId)
  if (!room) return
  const before = moodOf(activeLayout(st.doc), roomId)
  st.update((d: Doc) => ({ ...d, layouts: d.layouts.map((l) => (l.id === d.activeLayoutId ? withMood(l, roomId, mood) : l)) }))
  const m = MOODS[mood]
  const where = room.name.toLowerCase()
  if (before !== mood) focusMoodRoom(roomId)
  // The choice shows at once: the drawer opens on the mood goal's first piece, its pick cards in
  // the mood's colours (unless the room has what the mood asks for already).
  const d = useStore.getState().doc
  const l = activeLayout(d)
  const g = moodGoals(d, l, piecesOf(d, l)).find((x) => x.roomId === roomId)
  if (g && !g.done && g.item && g.tab && before !== mood) {
    if (st.selection) st.select(null)
    useGame.getState().showGoal({ id: g.id, title: g.title, tab: g.tab, item: g.item, placeIn: g.placeIn })
  }
  // A change swaps the room's mood goal for the new mood's (no loss pill for it: see Goals).
  st.notify(
    before === mood
      ? `${m.emoji} Your ${where} stays ${m.short.toLowerCase()}.`
      : g?.done
        ? `${m.emoji} ${m.short} ${where}: it's all there ★`
        : before
          ? `${m.emoji} Now ${m.short.toLowerCase()}! New goal: ${m.adds} ★`
          : `${m.emoji} ${m.short} ${where}! Goal: ${m.adds} ★`,
  )
}

/** The room whose card is showing: one the player asked to change, or a due moment once things are calm. */
function useMomentRoom(): { room: Room; asked: boolean } | null {
  const doc = useStore((s) => s.doc)
  const layout = activeLayout(doc)
  const best = useGame((s) => s.bestLevel[bestLevelKey(doc)] ?? 0)
  const skips = useGame((s) => s.moodSkips)
  const ask = useGame((s) => s.moodAsk)
  const calm = useGame((s) => s.phase === 'play' && s.mode !== 'look' && !s.sheet && !s.levelUp)
  const still = useStore((s) => !s.gesture && s.mode === 'select')
  const quiet = useGame((s) => s.moodQuiet)
  const rested = layout.placements.length >= quiet
  const due = useMemo(() => (rested ? dueMoment(doc, layout, best, piecesOf(doc, layout), new Set(skips)) : null), [doc, layout, best, skips, rested])
  const [ready, setReady] = useState<string | null>(null)
  const dueId = due?.id ?? null
  useEffect(() => {
    if (!dueId || !calm || !still) return setReady(null)
    const t = setTimeout(() => setReady(dueId), MOMENT_DELAY_MS)
    return () => clearTimeout(t)
  }, [dueId, calm, still])
  const asked = ask ? doc.rooms.find((r) => r.id === ask) : undefined
  const showing = asked && calm ? asked.id : due && ready === due.id && calm && still ? due.id : null
  // A selected piece's action bar sits where the card lands (round 17 review): the card clears the
  // selection as it opens, so the two never overlap.
  useEffect(() => {
    if (showing && useStore.getState().selection) useStore.getState().select(null)
  }, [showing])
  if (asked && calm) return { room: asked, asked: true }
  if (due && ready === due.id && calm && still) return { room: due, asked: false }
  return null
}

/** The mood moment card: one question, three moods with a tiny preview each, and "Not now". */
export function MoodCard() {
  const moment = useMomentRoom()
  const doc = useStore((s) => s.doc)
  const cardRef = useRef<HTMLDivElement>(null)
  const roomId = moment?.room.id
  const current = moodOf(activeLayout(doc), roomId)
  useEffect(() => {
    if (!moment) return
    announce(`Mood moment: how should the ${moment.room.name.toLowerCase()} feel?`)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roomId])
  if (!moment) return null
  const { room, asked } = moment
  const where = room.name.toLowerCase()
  const close = () => {
    restMoments()
    if (asked) useGame.getState().askMood(null)
    else useGame.getState().skipMood(momentKey(doc, activeLayout(doc), room.id))
  }
  return (
    <div
      ref={cardRef}
      className="mood-card g-card"
      role="dialog"
      aria-label={`How should the ${where} feel?`}
      key={room.id}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          e.stopPropagation()
          close()
        }
      }}
    >
      <button type="button" className="mood-close" onClick={close} aria-label="Not now" title="Not now">
        ×
      </button>
      <div className="mood-kicker">{asked ? 'Change the mood' : 'Mood moment'}</div>
      <h2 className="mood-q">How should the {where} feel?</h2>
      <div className="mood-options">
        {MOOD_IDS.map((id) => {
          const m = MOODS[id]
          return (
            <button
              key={id}
              type="button"
              className={`mood-opt mood-${id}${current === id ? ' on' : ''}`}
              aria-pressed={current === id}
              aria-label={`${m.name}: ${m.blurb}. Adds the goal ${m.goal.title}`}
              onClick={() => {
                chooseMood(room.id, id)
                restMoments()
                useGame.getState().askMood(null)
              }}
            >
              <MoodPreview id={id} />
              <span className="mood-name">{m.name}</span>
              <small className="mood-blurb">{m.blurb}</small>
              <small className="mood-adds">+ {m.adds}</small>
            </button>
          )
        })}
      </div>
      <p className="mood-foot">{current ? `Now ${MOODS[current].short.toLowerCase()}. A new mood brings a new ★ goal of its own.` : 'Shapes its colours, its light and a ★ goal. You can change it later.'}</p>
    </div>
  )
}

/** A tiny room in the mood's colours: wall, floor, a sofa, a lamp's glow and an accent. */
export function MoodPreview({ id }: { id: MoodId }) {
  const p = MOODS[id].preview
  return (
    <svg className="mood-prev" viewBox="0 0 96 60" aria-hidden>
      <rect width="96" height="44" fill={p.wall} />
      <rect y="44" width="96" height="16" fill={p.floor} />
      <circle cx="78" cy="22" r="18" fill={p.glow} opacity="0.85" />
      <rect x="76" y="20" width="4" height="26" rx="1" fill="#3d3d3d" opacity="0.7" />
      <path d="M70 20 h16 l-3 -8 h-10 z" fill={p.accent} />
      {id === 'bold' ? <rect x="24" y="8" width="30" height="18" rx="2" fill={p.accent} stroke="#2d2620" strokeWidth="1.5" /> : id === 'airy' ? <rect x="30" y="6" width="16" height="22" rx="8" fill="#e8f2ff" stroke="#c7a98a" strokeWidth="2" /> : <rect x="28" y="10" width="20" height="14" rx="2" fill="#e8a27c" stroke="#8a6b4f" strokeWidth="2" />}
      <rect x="14" y="32" width="50" height="14" rx="5" fill={p.sofa} />
      <rect x="10" y="30" width="10" height="17" rx="4" fill={p.sofa} />
      <rect x="58" y="30" width="10" height="17" rx="4" fill={p.sofa} />
      {id === 'airy' && <path d="M8 46 q2 -14 6 -18 q1 8 -2 18 z" fill={p.accent} />}
      {id === 'snug' && <ellipse cx="38" cy="53" rx="12" ry="4" fill={p.accent} />}
      {id === 'bold' && <circle cx="40" cy="54" r="4" fill="#b5534a" />}
    </svg>
  )
}

/** A soft round glow (white in the middle, clear at the edge), tinted per room by its material. */
let glowTex: THREE.Texture | null = null
function glowTexture(): THREE.Texture | null {
  if (glowTex || typeof document === 'undefined') return glowTex
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')
  if (!g) return null
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64)
  r.addColorStop(0, 'rgba(255,255,255,1)')
  r.addColorStop(0.55, 'rgba(255,255,255,0.55)')
  r.addColorStop(1, 'rgba(255,255,255,0)')
  g.fillStyle = r
  g.fillRect(0, 0, 128, 128)
  glowTex = new THREE.CanvasTexture(c)
  glowTex.colorSpace = THREE.SRGBColorSpace
  return glowTex
}

/** The room's floor, faintly washed in its mood's colour from the middle out (so a mood reads in daylight too). */
function MoodGlow({ room, color, at, reach, evening }: { room: Room; color: string; at: { x: number; y: number }; reach: number; evening: boolean }) {
  const geom = useMemo(() => {
    const g = new THREE.ShapeGeometry(new THREE.Shape(room.polygon.map((p) => new THREE.Vector2(p.x, -p.y))))
    // The glow's centre at the light, fading out by its reach.
    const pos = g.getAttribute('position')
    const uv = new Float32Array(pos.count * 2)
    const R = reach * 1.15
    for (let i = 0; i < pos.count; i++) {
      uv[i * 2] = (pos.getX(i) - at.x) / (2 * R) + 0.5
      uv[i * 2 + 1] = (pos.getY(i) + at.y) / (2 * R) + 0.5
    }
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
    g.rotateX(-Math.PI / 2)
    return g
  }, [room.polygon, at.x, at.y, reach])
  useEffect(() => () => geom.dispose(), [geom])
  const tex = glowTexture()
  if (!tex) return null
  return (
    <mesh geometry={geom} position={[0, 0.004, 0]} renderOrder={1} raycast={() => null}>
      <meshBasicMaterial map={tex} color={color} transparent opacity={evening ? 0.2 : 0.38} depthWrite={false} polygonOffset polygonOffsetFactor={-1} polygonOffsetUnits={-1} toneMapped={false} />
    </mesh>
  )
}

/** Each mood room's soft light and floor glow in the 3D view (rendered inside the Canvas). */
export function MoodLights() {
  const doc = useStore((s) => s.doc)
  const evening = useGame((s) => s.evening)
  const layout = activeLayout(doc)
  const lights = Object.entries(layout.moods ?? {}).flatMap(([roomId, id]) => {
    const room = doc.rooms.find((r) => r.id === roomId)
    const m = MOODS[id]
    return room && m ? [{ room, m, ...moodLightSpot(room) }] : []
  })
  return (
    <>
      {lights.map(({ room, m, at, height, reach }) => (
        <group key={room.id}>
          <pointLight position={[at.x, height, at.y]} color={m.light.color} intensity={evening ? m.light.intensity : m.light.day} distance={reach * 1.4} decay={1.2} castShadow={false} />
          <MoodGlow room={room} color={m.light.floor} at={at} reach={reach} evening={evening} />
        </group>
      ))}
    </>
  )
}
