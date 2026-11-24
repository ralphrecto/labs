// Mood moments (round 17): the one card that asks how a room should feel, and the soft light a
// chosen mood casts in the 3D view. The logic (which room, which goal, which colours) is in moods.ts.
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { activeLayout, useStore } from '../store'
import type { Doc, MoodId, Opening, Room, Vec2 } from '../model/types'
import { analyze } from '../analysis'
import { placedItems } from '../model/checks'
import { pointInPolygon, wallInwardNormal, wallSegments } from '../model/geometry'
import { sharedDoorways } from '../three/walkCollision'
import type { WallMode } from '../three/Scene3D'
import { bestLevelKey, scoreHome } from './goals'
import { useGame } from './gameStore'
import { changeMood, dueMoment, momentKey, moodGoalId, moodGoals, moodLightSpot, moodOf, moodsDoneIn, MOOD_IDS, MOODS, pickMood, piecesOf } from './moods'
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
  const l0 = activeLayout(st.doc)
  const before = moodOf(l0, roomId)
  // A finished mood is remembered on the design (round 18), so changing it keeps the room's star.
  const scored = !!before && !!scoreHome(st.doc, l0, analyze(st.doc, l0)).goals.find((g) => g.id === moodGoalId(roomId))?.done
  st.update((d: Doc) => ({ ...d, layouts: d.layouts.map((l) => (l.id === d.activeLayoutId ? changeMood(d, l, roomId, mood, scored) : l)) }))
  const m = MOODS[mood]
  const where = room.name.toLowerCase()
  if (before !== mood) focusMoodRoom(roomId)
  // The choice shows at once: the drawer opens on the mood goal's first piece, its pick cards in
  // the mood's colours (unless the room has what the mood asks for already).
  const d = useStore.getState().doc
  const l = activeLayout(d)
  // The new mood's own pieces (even when the room keeps its star from a mood finished before).
  const g = moodGoals(d, l, piecesOf(d, l), { live: true }).find((x) => x.roomId === roomId)
  const kept = moodsDoneIn(l, roomId).length > 0
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
        : before && kept
          ? `${m.emoji} Now ${m.short.toLowerCase()}! Your ★ stays. For the full look: ${m.adds}`
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
  // The room's mood star is earned (round 18): changing the mood keeps it.
  const layoutNow = activeLayout(doc)
  const earned = !!current && (moodsDoneIn(layoutNow, room.id).length > 0 || !!moodGoals(doc, layoutNow, piecesOf(doc, layoutNow), { live: true }).find((g) => g.roomId === room.id)?.done)
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
      <p className="mood-foot">{current ? (earned ? `Now ${MOODS[current].short.toLowerCase()}. Its ★ stays whichever mood you pick.` : `Now ${MOODS[current].short.toLowerCase()}. A new mood brings a new ★ goal of its own.`) : 'Shapes its colours, its light and a ★ goal. You can change it later.'}</p>
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

/** Soft round glows, tinted per room by their material: `glow` is white in the middle and clear at the edge; `ring` is a halo, clear in the middle (round a rug). */
const glowTex: Record<'glow' | 'ring', THREE.Texture | null> = { glow: null, ring: null }
function glowTexture(kind: 'glow' | 'ring' = 'glow'): THREE.Texture | null {
  if (glowTex[kind] || typeof document === 'undefined') return glowTex[kind]
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')
  if (!g) return null
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64)
  const stops: [number, number][] = kind === 'glow' ? [[0, 1], [0.55, 0.55], [1, 0]] : [[0, 0.12], [0.5, 0.3], [0.72, 1], [1, 0]]
  for (const [at, a] of stops) r.addColorStop(at, `rgba(255,255,255,${a})`)
  g.fillStyle = r
  g.fillRect(0, 0, 128, 128)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  glowTex[kind] = t
  return t
}

/** A soft glow on the room's floor, centred at `at` and fading out by `radius`, cut to the room's outline (so it never shows past its walls). */
function MoodGlow({ room, color, at, radius, opacity, ring }: { room: Room; color: string; at: Vec2; radius: number; opacity: number; ring?: boolean }) {
  const geom = useMemo(() => {
    const g = new THREE.ShapeGeometry(new THREE.Shape(room.polygon.map((p) => new THREE.Vector2(p.x, -p.y))))
    const pos = g.getAttribute('position')
    const uv = new Float32Array(pos.count * 2)
    for (let i = 0; i < pos.count; i++) {
      uv[i * 2] = (pos.getX(i) - at.x) / (2 * radius) + 0.5
      uv[i * 2 + 1] = (pos.getY(i) + at.y) / (2 * radius) + 0.5
    }
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
    g.rotateX(-Math.PI / 2)
    return g
  }, [room.polygon, at.x, at.y, radius])
  useEffect(() => () => geom.dispose(), [geom])
  const tex = glowTexture(ring ? 'ring' : 'glow')
  if (!tex) return null
  return (
    <mesh geometry={geom} position={[0, ring ? 0.012 : 0.004, 0]} renderOrder={1} raycast={() => null}>
      <meshBasicMaterial map={tex} color={color} transparent opacity={opacity} depthWrite={false} polygonOffset polygonOffsetFactor={-1} polygonOffsetUnits={-1} toneMapped={false} />
    </mesh>
  )
}

/** The mood light: a spot aimed straight down whose cone meets the floor inside the room (round 18). */
function MoodSpot({ at, height, radius, color, intensity }: { at: Vec2; height: number; radius: number; color: string; intensity: number }) {
  const target = useMemo(() => new THREE.Object3D(), [])
  target.position.set(at.x, 0, at.y)
  const angle = Math.min(Math.PI / 3, Math.atan(radius / height))
  return (
    <>
      <primitive object={target} />
      <spotLight position={[at.x, height, at.y]} target={target} color={color} intensity={intensity} angle={angle} penumbra={0.85} distance={height / Math.cos(angle) + 0.25} decay={1} castShadow={false} />
    </>
  )
}

/** A soft wash on a wall's inner face, kept off its doors and windows. */
interface WallWash {
  wi: number
  geom: THREE.BufferGeometry
  /** A point on the wall's outer face and its outward normal: the camera past it means the wall is cut down. */
  ox: number
  oz: number
  nx: number
  nz: number
  length: number
}

function wallWashes(room: Room, doorways: Opening[] | undefined): WallWash[] {
  const top = room.ceilingHeight * 0.92
  return wallSegments(room).map(([a, b], wi) => {
    const L = Math.hypot(b.x - a.x, b.y - a.y)
    const d = { x: (b.x - a.x) / (L || 1), y: (b.y - a.y) / (L || 1) }
    const n = wallInwardNormal(room, wi)
    const ops = [...room.openings, ...(doorways ?? [])].filter((o) => o.wall === wi).sort((x, y) => x.offset - y.offset)
    // Solid parts of the wall (s0..s1 along it, y0..y1 up it), like RoomMesh's wall pieces.
    const parts: [number, number, number, number][] = []
    let cursor = 0
    for (const o of ops) {
      if (o.offset > cursor) parts.push([cursor, o.offset, 0, top])
      const s0 = Math.max(cursor, o.offset)
      const s1 = Math.min(L, o.offset + o.width)
      if (s1 > s0) {
        if (o.sill > 0.05) parts.push([s0, s1, 0, Math.min(top, o.sill)])
        if (o.sill + o.height < top - 0.05) parts.push([s0, s1, o.sill + o.height, top])
      }
      cursor = Math.max(cursor, o.offset + o.width)
    }
    if (cursor < L) parts.push([cursor, L, 0, top])
    const pos: number[] = []
    const uv: number[] = []
    const eps = 0.006
    const at = (s: number, y: number) => {
      pos.push(a.x + d.x * s + n.x * eps, y, a.y + d.y * s + n.y * eps)
      uv.push(s / (L || 1), y / top)
    }
    for (const [s0, s1, y0, y1] of parts) {
      if (s1 - s0 < 1e-3 || y1 - y0 < 1e-3) continue
      at(s0, y0); at(s1, y0); at(s1, y1)
      at(s0, y0); at(s1, y1); at(s0, y1)
    }
    const geom = new THREE.BufferGeometry()
    geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    geom.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2))
    const t = room.wallThickness
    return { wi, geom, ox: a.x - n.x * t, oz: a.y - n.y * t, nx: -n.x, nz: -n.y, length: L }
  })
}

/**
 * The mood's wall washes: every standing wall (Airy) or one accent wall (Bold: the wall its art
 * hangs on, else its longest; when that wall is cut down for the view, the next in line). A wall
 * cut down (the camera beyond it, or Low walls) shows none, so no glow floats where a wall was.
 */
function MoodWalls({ room, doorways, color, opacity, which, order, wallMode }: { room: Room; doorways: Opening[] | undefined; color: string; opacity: number; which: 'all' | 'accent'; order: number[]; wallMode: WallMode }) {
  const washes = useMemo(() => wallWashes(room, doorways), [room, doorways])
  useEffect(() => () => washes.forEach((w) => w.geom.dispose()), [washes])
  const refs = useRef<(THREE.Mesh | null)[]>([])
  useFrame(({ camera }) => {
    const c = camera.position
    const stands = (w: WallWash) => wallMode === 'full' || (wallMode === 'auto' && (c.x - w.ox) * w.nx + (c.z - w.oz) * w.nz <= 0)
    const lead = which === 'accent' ? order.find((wi) => stands(washes[wi])) : undefined
    washes.forEach((w, i) => {
      const m = refs.current[i]
      if (m) m.visible = which === 'all' ? stands(w) : i === lead
    })
  })
  const tex = glowTexture()
  if (!tex) return null
  return (
    <>
      {washes.map((w, i) => (
        <mesh key={w.wi} ref={(m) => void (refs.current[i] = m)} geometry={w.geom} renderOrder={1} raycast={() => null} visible={false}>
          <meshBasicMaterial map={tex} color={color} transparent opacity={opacity} depthWrite={false} side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={-1} polygonOffsetUnits={-1} toneMapped={false} />
        </mesh>
      ))}
    </>
  )
}

/** The walls in accent order: the one nearest the room's wall art first, then longest first. */
function accentOrder(room: Room, art: Vec2[]): number[] {
  const segs = wallSegments(room)
  const distTo = (p: Vec2, [a, b]: [Vec2, Vec2]) => {
    const dx = b.x - a.x
    const dy = b.y - a.y
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy || 1)))
    return Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy)
  }
  const artWalls = art.map((p) => segs.map((s, i) => [distTo(p, s), i] as const).sort((x, y) => x[0] - y[0])[0][1])
  const byLength = segs.map((s, i) => [Math.hypot(s[1].x - s[0].x, s[1].y - s[0].y), i] as const).sort((x, y) => y[0] - x[0]).map((x) => x[1])
  return [...new Set([...artWalls, ...byLength])]
}

/**
 * Each mood room's light in the 3D view (rendered inside the Canvas), all kept inside the room
 * (round 18): a spot aimed down whose pool stays off the walls, a floor glow cut to the room's
 * outline, and the mood's own touch: Snug a warm amber pool, Airy a bright white wash on its
 * walls, Bold a cool blue glow on its accent wall and round its rug.
 */
export function MoodLights({ wallMode = 'auto' }: { wallMode?: WallMode }) {
  const doc = useStore((s) => s.doc)
  const evening = useGame((s) => s.evening)
  const layout = activeLayout(doc)
  const doorways = useMemo(() => sharedDoorways(doc.rooms), [doc.rooms])
  const placed = useMemo(() => placedItems(doc, layout), [doc, layout])
  const lights = Object.entries(layout.moods ?? {}).flatMap(([roomId, id]) => {
    const room = doc.rooms.find((r) => r.id === roomId)
    const m = MOODS[id]
    return room && m ? [{ room, m, ...moodLightSpot(room) }] : []
  })
  return (
    <>
      {lights.map(({ room, m, at, height, reach, inner }) => {
        const L = m.light
        const pool = Math.max(0.5, inner * L.pool * 0.95)
        const here = placed.filter(({ p }) => pointInPolygon({ x: p.x, y: p.y }, room.polygon))
        const rugs = L.rug ? here.filter(({ item }) => item.category === 'rug') : []
        const art = L.wall?.walls === 'accent' ? here.filter(({ item }) => item.category === 'wall-art' || item.category === 'mirror').map(({ p }) => ({ x: p.x, y: p.y })) : []
        return (
          <group key={room.id}>
            <MoodSpot at={at} height={height} radius={pool} color={L.color} intensity={evening ? L.intensity : L.day} />
            <MoodGlow room={room} color={L.floor} at={at} radius={Math.min(reach * 1.15, pool * 1.6)} opacity={evening ? L.floorOpacity.evening : L.floorOpacity.day} />
            {rugs.map(({ p, item }) => (
              <MoodGlow key={p.id} room={room} color={L.rug!} at={{ x: p.x, y: p.y }} radius={Math.max(item.width, item.depth) * 0.72} opacity={evening ? 0.3 : 0.4} ring />
            ))}
            {L.wall && <MoodWalls room={room} doorways={doorways.get(room.id)} color={L.wall.color} opacity={evening ? L.wall.opacity.evening : L.wall.opacity.day} which={L.wall.walls} order={L.wall.walls === 'accent' ? accentOrder(room, art) : []} wallMode={wallMode} />}
          </group>
        )
      })}
    </>
  )
}
