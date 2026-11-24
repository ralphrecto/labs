// Mood moments (round 17): every level or so from Lived-in, one card asks how a room should feel.
// The answer is stored on the design (Layout.moods) and shapes the home from then on: the room's
// pick cards come in the mood's colours (swatches.MOOD_TONES), it gets a bonus goal of its own
// worth a star, and a soft light of the mood's colour in the 3D view. Pure logic, no React.
import { BEDROOM, DINING, LIVING, OFFICE, UTILITY_ROOM } from '../model/placement'
import { centroid, pointInPolygon, polygonArea } from '../model/geometry'
import { placedItems } from '../model/checks'
import type { Doc, Layout, MoodId, Room, Vec2 } from '../model/types'
import { MOOD_TONES } from './swatches'
import type { Goal } from './goals'

export type { MoodId } from '../model/types'

export interface Mood {
  id: MoodId
  /** On the card: "Bright & airy". */
  name: string
  /** One word, for toasts and goal titles: "Airy". */
  short: string
  emoji: string
  /** One line under the name on the card. */
  blurb: string
  /** What its goal asks for, in a few words, for the card on phones: "mirror + plant". */
  adds: string
  /**
   * The mood's bonus goal: the pieces it asks for in the room (one of each need's categories, in
   * the mood's colours when `tone`), and its copy. The first category is the one the chip suggests:
   * rugs, wall decor, lamps and plants, which find a spot in almost any room. A `tone` need the
   * room already has a piece for, in another colour (an oak rug), is met by painting that piece
   * `paint` rather than buying another.
   */
  goal: { title: string; hint: string; needs: { cats: string[]; tone?: boolean; paint?: string; paintName?: string }[] }
  /** The tiny preview: wall, floor, sofa and lamp glow colours. */
  preview: { wall: string; floor: string; sofa: string; glow: string; accent: string }
  /**
   * The room's mood light in the 3D view (round 18: a spot light aimed down whose cone stays inside
   * the room, so nothing spills into a neighbouring hallway): its colour, `intensity` in the evening
   * and a stronger `day` one so it still reads against daylight, and how wide its pool is (0..1 of
   * the room's inner radius).
   * Each mood also has its own touch by day: `floor` tints the pool on the floor (`floorOpacity`),
   * and `wall` washes walls with a soft glow: every wall for Airy (white, lifting) and Snug (amber,
   * mostly in the evening, where the pool alone would leave its walls dark), the accent wall
   * (its art, else its longest) for Bold (cool blue, with a blue glow round its rug).
   */
  light: { color: string; intensity: number; day: number; pool: number; floor: string; floorOpacity: { day: number; evening: number }; wall?: { color: string; walls: 'all' | 'accent'; opacity: { day: number; evening: number } }; rug?: string }
}

export const MOODS: Record<MoodId, Mood> = {
  airy: {
    id: 'airy',
    name: 'Bright & airy',
    short: 'Airy',
    emoji: '☀️',
    blurb: 'Pale woods, linen, green',
    adds: 'mirror + plant',
    goal: { title: 'Airy: a mirror and a plant', hint: 'A mirror bounces the daylight round; a plant brings the outside in.', needs: [{ cats: ['mirror'] }, { cats: ['plant'] }] },
    preview: { wall: '#f4f1ea', floor: '#d9c4a5', sofa: '#e2d6b8', glow: '#e8f2ff', accent: '#5f8f5a' },
    light: { color: '#f7faff', intensity: 14, day: 22, pool: 1, floor: '#ffffff', floorOpacity: { day: 0.5, evening: 0.22 }, wall: { color: '#ffffff', walls: 'all', opacity: { day: 0.34, evening: 0.2 } } },
  },
  snug: {
    id: 'snug',
    name: 'Warm & snug',
    short: 'Snug',
    emoji: '🕯️',
    blurb: 'Walnut, honey, soft light',
    adds: 'rug + lamp',
    goal: { title: 'Snug: a soft rug and warm light', hint: 'A soft rug underfoot and a lamp for a warm pool of light.', needs: [{ cats: ['rug'] }, { cats: ['floor-lamp', 'table-lamp'] }] },
    preview: { wall: '#ead9c3', floor: '#8a6b4f', sofa: '#b5534a', glow: '#ffc27a', accent: '#e0b04f' },
    light: { color: '#ffa04d', intensity: 12, day: 20, pool: 0.8, floor: '#ffa24f', floorOpacity: { day: 0.4, evening: 0.24 }, wall: { color: '#ffa857', walls: 'all', opacity: { day: 0.12, evening: 0.42 } } },
  },
  bold: {
    id: 'bold',
    name: 'Bold & playful',
    short: 'Bold',
    emoji: '🎨',
    blurb: 'Blue, terracotta, big art',
    adds: 'art + bold rug',
    goal: { title: 'Bold: big art and a bold rug', hint: 'Art on the wall and a rug in a bold colour (blue, terracotta, honey) make the room sing. Or paint yours.', needs: [{ cats: ['wall-art'] }, { cats: ['rug'], tone: true, paint: '#5b7bb5', paintName: 'cornflower blue' }] },
    preview: { wall: '#f1e6d8', floor: '#c7a98a', sofa: '#5b7bb5', glow: '#ffe0bd', accent: '#e0b04f' },
    light: { color: '#ffe6cc', intensity: 9, day: 12, pool: 0.9, floor: '#ffe0bd', floorOpacity: { day: 0.16, evening: 0.1 }, wall: { color: '#4f7fe0', walls: 'accent', opacity: { day: 0.42, evening: 0.3 } }, rug: '#4f7fe0' },
  },
}

export const MOOD_IDS = Object.keys(MOODS) as MoodId[]

/** The levels (LEVELS indices) that open a mood moment: Lived-in, Cozy, Designer home... */
export const MOMENT_LEVELS = [2, 3, 4]
/** ...once the home has this many pieces (not counting moving boxes), so the first isn't on move-in day. */
export const MOMENT_PIECES = [7, 11, 15]

/** The goal id of a room's mood goal. */
export const moodGoalId = (roomId: string) => `mood:${roomId}`

/** The drawer tab each mood piece is on (goals.TAB_OF has no wall decor). */
const MOOD_TAB: Record<string, string> = { mirror: 'wall', 'wall-art': 'wall', plant: 'decor', 'floor-lamp': 'decor', 'table-lamp': 'decor', rug: 'decor', ottoman: 'living', armchair: 'living', bench: 'dining' }


const WET = /bath|\bwc\b|toilet|powder|shower|lavatory/i
const MIN_AREA = 3

/** A placed piece as the mood goals see it (goals.scoreHome's counted pieces fit this). */
export interface MoodPiece {
  /** Placement id, for its colour (a `tone` need). */
  id?: string
  category: string
  room?: { id: string }
}

/** A room's mood on this design, if it has one. */
export const moodOf = (layout: Layout | undefined, roomId: string | undefined): MoodId | undefined =>
  roomId && layout?.moods?.[roomId] && MOODS[layout.moods[roomId]] ? layout.moods[roomId] : undefined

/** The design's latest mood (the last room given one), for picks not tied to a room. */
export function latestMood(layout: Layout | undefined): MoodId | undefined {
  const all = Object.values(layout?.moods ?? {}).filter((m) => MOODS[m])
  return all[all.length - 1]
}

/** The mood that colours a pick: its room's, else the design's latest. */
export const pickMood = (layout: Layout | undefined, roomId?: string): MoodId | undefined => moodOf(layout, roomId) ?? latestMood(layout)

/** The design with this room's mood set (or cleared with `undefined`); the room moves to the end, so it's the latest. */
export function withMood(layout: Layout, roomId: string, mood: MoodId | undefined): Layout {
  const rest = Object.fromEntries(Object.entries(layout.moods ?? {}).filter(([k]) => k !== roomId))
  const moods = mood ? { ...rest, [roomId]: mood } : rest
  return { ...layout, moods: Object.keys(moods).length ? moods : undefined }
}

/**
 * The design with this room's mood changed (round 18): a mood the room has finished is remembered
 * (`Layout.moodsDone`), so the room's star stays when the player tries another mood. One star per
 * room at most: switching back and forth can't add any. `scored` is whether goals.scoreHome counts
 * the room's mood goal done (it leaves out cramped or packed pieces, which piecesOf counts).
 */
export function changeMood(doc: Doc, layout: Layout, roomId: string, mood: MoodId | undefined, scored = true): Layout {
  const was = moodOf(layout, roomId)
  const live = scored && was && moodGoals(doc, layout, piecesOf(doc, layout), { live: true }).find((g) => g.roomId === roomId)?.done
  const done = layout.moodsDone?.[roomId] ?? []
  const keep = live && !done.includes(was) ? { ...layout, moodsDone: { ...layout.moodsDone, [roomId]: [...done, was] } } : layout
  return withMood(keep, roomId, mood)
}

/** The moods a room finished before its mood was changed (Layout.moodsDone). */
export const moodsDoneIn = (layout: Layout, roomId: string): MoodId[] => (layout.moodsDone?.[roomId] ?? []).filter((m) => MOODS[m])

/**
 * The mood goals for a design (one per room with a mood and still in the plan): a bonus worth a
 * star, done when the room has one of each piece group the mood asks for. Like a perk it is left
 * out of every level bar and cap (`perk: 'mood'`), so a mood adds a star but never holds a level.
 */
export function moodGoals(doc: Doc, layout: Layout, pieces: MoodPiece[], opts: { live?: boolean } = {}): Goal[] {
  const out: Goal[] = []
  if (!layout.moods) return out
  const items = new Map(doc.items.map((i) => [i.id, i.color]))
  const colorOf = new Map(layout.placements.map((p) => [p.id, items.get(p.itemId)]))
  for (const [roomId, id] of Object.entries(layout.moods)) {
    const mood = MOODS[id]
    const room = doc.rooms.find((r) => r.id === roomId)
    if (!mood || !room) continue
    const here = pieces.filter((p) => p.room?.id === roomId)
    const toned = (p: MoodPiece) => inTone(id, colorOf.get(p.id ?? ''))
    const have = mood.goal.needs.map((n) => here.some((p) => n.cats.includes(p.category) && (!n.tone || toned(p))))
    const missing = mood.goal.needs.find((_, i) => !have[i])
    const where = room.name.toLowerCase()
    // A mood this room finished before (round 18) keeps its star: the goal stays done while the
    // new mood's pieces come in (the card still opens the drawer on them), never a second star.
    const kept = !opts.live && !!missing && moodsDoneIn(layout, roomId).length > 0
    if (kept) {
      const was = MOODS[moodsDoneIn(layout, roomId).slice(-1)[0]]
      out.push({
        id: moodGoalId(roomId),
        title: mood.goal.title,
        hint: `Your ${where} keeps its ★ from feeling ${was.short.toLowerCase()}. For the full ${mood.short.toLowerCase()} look: ${mood.adds}.`,
        done: true,
        progress: 1,
        stars: 1,
        roomId,
        category: 'style',
        bonus: true,
        perk: 'mood',
      })
      continue
    }
    const done = !missing
    // A rug already down in another colour (round 17 review): the goal asks to paint it, not to
    // squeeze a second rug in; the chip paints it (Goals.showGoal).
    const repaint = missing?.tone && missing.paint ? here.find((p) => p.id && missing.cats.includes(p.category)) : undefined
    out.push({
      id: moodGoalId(roomId),
      title: mood.goal.title,
      hint: done ? `Your ${where} feels ${mood.short.toLowerCase()}. Select the room to change its mood.` : repaint ? `Paint the ${where}'s ${repaint.category.replace(/-/g, ' ')} ${missing.paintName ?? 'a bold colour'} (tap Next, or Color on its bar).` : `${mood.goal.hint} In the ${where}.`,
      done,
      progress: have.filter(Boolean).length / have.length,
      stars: 1,
      roomId,
      category: 'style',
      bonus: true,
      perk: 'mood',
      ...(repaint ? { near: repaint.id } : missing ? { item: missing.cats[0], tab: MOOD_TAB[missing.cats[0]] ?? 'decor', placeIn: roomId } : {}),
    })
  }
  return out
}

/** A colour that reads as the mood (one of its tones, or close to one). */
export const inTone = (id: MoodId, color: string | undefined): boolean => !!color && MOOD_TONES[id].palette.some((t) => near(t, color))

/**
 * The mood whose colours a pick must wear to count (round 17): when the hinted goal is a room's
 * mood goal whose missing piece is a `tone` need of this category (Bold's rug), a piece of yours
 * in another colour (an oak rug waiting in a box, the remix source's rug) isn't offered first.
 */
export function toneNeeded(doc: Doc, layout: Layout, goalId: string | undefined, category: string): MoodId | undefined {
  if (!goalId?.startsWith('mood:')) return undefined
  const roomId = goalId.slice(5)
  const id = moodOf(layout, roomId)
  if (!id) return undefined
  const g = moodGoals(doc, layout, piecesOf(doc, layout)).find((x) => x.id === goalId)
  if (!g || g.done || g.item !== category) return undefined
  return MOODS[id].goal.needs.some((n) => n.tone && n.cats.includes(category)) ? id : undefined
}

/**
 * The paint a mood goal asks for (a Bold room's oak rug, painted blue): the placement and colour,
 * or null when the goal isn't a repaint.
 */
export function moodRepaint(doc: Doc, layout: Layout, goalId: string): { placementId: string; color: string; name: string } | null {
  const id = moodOf(layout, goalId.startsWith('mood:') ? goalId.slice(5) : undefined)
  const g = id && moodGoals(doc, layout, piecesOf(doc, layout)).find((x) => x.id === goalId)
  const need = id ? MOODS[id].goal.needs.find((n) => n.tone && n.paint) : undefined
  return g && !g.done && g.near && !g.tab && need?.paint ? { placementId: g.near, color: need.paint, name: need.paintName ?? 'a new colour' } : null
}

/**
 * The design with one placement repainted: its item takes the colour, or, when another placement
 * (here or in another design) shares that item, the placement gets a repainted copy of it.
 */
export function repaintDoc(doc: Doc, placementId: string, color: string, newId: string): Doc {
  const layout = doc.layouts.find((l) => l.id === doc.activeLayoutId)
  const p = layout?.placements.find((x) => x.id === placementId)
  const item = p && doc.items.find((i) => i.id === p.itemId)
  if (!layout || !p || !item) return doc
  const shared = doc.layouts.some((l) => l.placements.some((x) => x.itemId === item.id && (x.id !== placementId || l.id !== layout.id)))
  if (!shared) return { ...doc, items: doc.items.map((i) => (i.id === item.id ? { ...i, color } : i)) }
  const copy = { ...item, id: newId, color, clearance: { ...item.clearance } }
  return {
    ...doc,
    items: [...doc.items, copy],
    layouts: doc.layouts.map((l) => (l.id === layout.id ? { ...l, placements: l.placements.map((x) => (x.id === placementId ? { ...x, itemId: newId } : x)) } : l)),
  }
}

/** A mood goal just finished, as its own headline: "🕯️ Your living room feels snug". */
export function moodDoneTitle(doc: Doc, layout: Layout, goal: Goal): string | undefined {
  if (goal.perk !== 'mood' || !goal.roomId) return undefined
  const id = moodOf(layout, goal.roomId)
  const room = doc.rooms.find((r) => r.id === goal.roomId)
  return id && room ? `${MOODS[id].emoji} Your ${room.name.toLowerCase()} feels ${MOODS[id].short.toLowerCase()}` : undefined
}

/** Rooms a mood suits, best first: living rooms, then bedrooms, then dining rooms and offices, then the rest. */
export function moodRooms(doc: Doc, pieces: MoodPiece[]): Room[] {
  const count = (r: Room) => pieces.filter((p) => p.room?.id === r.id).length
  const rank = (r: Room) => (LIVING.test(r.name) ? 0 : BEDROOM.test(r.name) ? 1 : DINING.test(r.name) || OFFICE.test(r.name) ? 2 : 3)
  return doc.rooms
    .filter((r) => polygonArea(r.polygon) >= MIN_AREA && !WET.test(r.name) && (!UTILITY_ROOM.test(r.name) || LIVING.test(r.name)) && count(r) > 0)
    .sort((a, b) => rank(a) - rank(b) || count(b) - count(a))
}

/** Where skipped moments are remembered: per design and room. */
export const momentKey = (doc: Doc, layout: Layout, roomId: string) => `${doc.apartment.id}:${layout.id}:${roomId}`

/**
 * The mood moment due now, if any: one per level in MOMENT_LEVELS reached (with MOMENT_PIECES placed), each for the next
 * room without a mood (or a skipped card). A skipped room isn't asked again on this design; a
 * remix keeps its source's moods, so it only asks about rooms still without one.
 */
export function dueMoment(doc: Doc, layout: Layout, level: number, pieces: MoodPiece[], skipped: ReadonlySet<string>): Room | null {
  const n = pieces.filter((p) => p.category !== 'box').length
  const due = MOMENT_LEVELS.filter((l, i) => level >= l && n >= MOMENT_PIECES[i]).length
  if (!due) return null
  const rooms = moodRooms(doc, pieces)
  const handled = rooms.filter((r) => moodOf(layout, r.id) || skipped.has(momentKey(doc, layout, r.id))).length
  if (handled >= due) return null
  return rooms.find((r) => !moodOf(layout, r.id) && !skipped.has(momentKey(doc, layout, r.id))) ?? null
}

/**
 * Where a room's mood light hangs (plan x/y, a little below the ceiling) and how far its pool may
 * reach: `inner` is the distance from that spot to the nearest wall (round 18). The spot is the
 * room's most inner point (not its centroid, which can sit outside an L-shaped room), so a cone of
 * light no wider than `inner` at the floor never touches a wall or leaks into the next room.
 */
export function moodLightSpot(room: Room): { at: Vec2; height: number; reach: number; inner: number } {
  const { at, inner } = innerPoint(room.polygon)
  // A pool of light in its own room, not a wash over the whole home.
  const reach = Math.max(1.8, Math.sqrt(polygonArea(room.polygon)) * 0.7)
  return { at, height: Math.max(1.8, room.ceilingHeight - 0.45), reach, inner }
}

/** The point of a polygon farthest from its edges (a grid search, refined once), and that distance. */
export function innerPoint(poly: Vec2[]): { at: Vec2; inner: number } {
  const c = centroid(poly)
  let best = { at: c, inner: pointInPolygon(c, poly) ? edgeDistance(c, poly) : 0 }
  const xs = poly.map((p) => p.x)
  const ys = poly.map((p) => p.y)
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  for (let pass = 0; pass < 2; pass++) {
    const step = Math.max(x1 - x0, y1 - y0) / 24
    if (!(step > 0)) break
    for (let x = x0 + step / 2; x < x1; x += step)
      for (let y = y0 + step / 2; y < y1; y += step) {
        const p = { x, y }
        if (!pointInPolygon(p, poly)) continue
        const d = edgeDistance(p, poly)
        // Prefer the centroid's side on near ties, so a plain rectangle keeps its light in the middle.
        if (d > best.inner + 1e-3) best = { at: p, inner: d }
      }
    ;[x0, x1, y0, y1] = [best.at.x - step, best.at.x + step, best.at.y - step, best.at.y + step]
  }
  return best
}

function edgeDistance(p: Vec2, poly: Vec2[]): number {
  let d = Infinity
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]
    const b = poly[(i + 1) % poly.length]
    const dx = b.x - a.x
    const dy = b.y - a.y
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy || 1)))
    d = Math.min(d, Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy))
  }
  return d
}

/** The mood's colours for its card preview swatches. */
export const moodSwatches = (id: MoodId) => MOOD_TONES[id].palette.slice(0, 3)

/** Two colours close enough to read as the same tone. */
function near(a: string, b: string): boolean {
  const rgb = (h: string) => (/^#[0-9a-f]{6}$/i.test(h) ? [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) : null)
  const x = rgb(a)
  const y = rgb(b)
  return !!x && !!y && Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]) < 20
}

/** Placed pieces and the room each stands in (by its centre), for choosing a moment's room. */
export function piecesOf(doc: Doc, layout: Layout): MoodPiece[] {
  return placedItems(doc, layout).map(({ p, item }) => ({ id: p.id, category: item.category, room: doc.rooms.find((r) => pointInPolygon({ x: p.x, y: p.y }, r.polygon)) }))
}
