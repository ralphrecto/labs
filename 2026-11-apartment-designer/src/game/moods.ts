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
  /**
   * The mood's bonus goal: the pieces it asks for in the room (one of each need's categories, in
   * the mood's colours when `tone`), and its copy. The first category is the one the chip suggests:
   * rugs, wall decor, lamps and plants, which find a spot in almost any room.
   */
  goal: { title: string; hint: string; needs: { cats: string[]; tone?: boolean }[] }
  /** The tiny preview: wall, floor, sofa and lamp glow colours. */
  preview: { wall: string; floor: string; sofa: string; glow: string; accent: string }
  /** The room's mood light in the 3D view (a soft point light). */
  light: { color: string; intensity: number }
}

export const MOODS: Record<MoodId, Mood> = {
  airy: {
    id: 'airy',
    name: 'Bright & airy',
    short: 'Airy',
    emoji: '☀️',
    blurb: 'Pale woods, linen, green',
    goal: { title: 'Airy: a mirror and a plant', hint: 'A mirror bounces the daylight round; a plant brings the outside in.', needs: [{ cats: ['mirror'] }, { cats: ['plant'] }] },
    preview: { wall: '#f4f1ea', floor: '#d9c4a5', sofa: '#e2d6b8', glow: '#e8f2ff', accent: '#5f8f5a' },
    light: { color: '#dcecff', intensity: 5 },
  },
  snug: {
    id: 'snug',
    name: 'Warm & snug',
    short: 'Snug',
    emoji: '🕯️',
    blurb: 'Walnut, honey, soft light',
    goal: { title: 'Snug: a soft rug and warm light', hint: 'A soft rug underfoot and a lamp for a warm pool of light.', needs: [{ cats: ['rug'] }, { cats: ['floor-lamp', 'table-lamp'] }] },
    preview: { wall: '#ead9c3', floor: '#8a6b4f', sofa: '#b5534a', glow: '#ffc27a', accent: '#e0b04f' },
    light: { color: '#ffb36b', intensity: 8 },
  },
  bold: {
    id: 'bold',
    name: 'Bold & playful',
    short: 'Bold',
    emoji: '🎨',
    blurb: 'Blue, terracotta, big art',
    goal: { title: 'Bold: big art and a bold rug', hint: 'Art on the wall and a rug in a bold colour (blue, terracotta, honey) make the room sing. Or paint yours.', needs: [{ cats: ['wall-art'] }, { cats: ['rug'], tone: true }] },
    preview: { wall: '#f1e6d8', floor: '#c7a98a', sofa: '#5b7bb5', glow: '#ffd1e0', accent: '#e0b04f' },
    light: { color: '#ff9ec0', intensity: 5 },
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
 * The mood goals for a design (one per room with a mood and still in the plan): a bonus worth a
 * star, done when the room has one of each piece group the mood asks for. Like a perk it is left
 * out of every level bar and cap (`perk: 'mood'`), so a mood adds a star but never holds a level.
 */
export function moodGoals(doc: Doc, layout: Layout, pieces: MoodPiece[]): Goal[] {
  const out: Goal[] = []
  if (!layout.moods) return out
  const items = new Map(doc.items.map((i) => [i.id, i.color]))
  const colorOf = new Map(layout.placements.map((p) => [p.id, items.get(p.itemId)]))
  for (const [roomId, id] of Object.entries(layout.moods)) {
    const mood = MOODS[id]
    const room = doc.rooms.find((r) => r.id === roomId)
    if (!mood || !room) continue
    const here = pieces.filter((p) => p.room?.id === roomId)
    const toned = (p: MoodPiece) => {
      const c = colorOf.get(p.id ?? '')
      return !!c && MOOD_TONES[id].palette.some((t) => near(t, c))
    }
    const have = mood.goal.needs.map((n) => here.some((p) => n.cats.includes(p.category) && (!n.tone || toned(p))))
    const missing = mood.goal.needs.find((_, i) => !have[i])
    const done = !missing
    const where = room.name.toLowerCase()
    out.push({
      id: moodGoalId(roomId),
      title: mood.goal.title,
      hint: done ? `Your ${where} feels ${mood.short.toLowerCase()}. Select the room to change its mood.` : `${mood.goal.hint} In the ${where}.`,
      done,
      progress: have.filter(Boolean).length / have.length,
      stars: 1,
      roomId,
      category: 'style',
      bonus: true,
      perk: 'mood',
      ...(missing ? { item: missing.cats[0], tab: MOOD_TAB[missing.cats[0]] ?? 'decor', placeIn: roomId } : {}),
    })
  }
  return out
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
 * remix starts without moods, so it asks afresh.
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

/** Where a room's mood light hangs (plan x/y, a little below the ceiling) and how far it reaches. */
export function moodLightSpot(room: Room): { at: Vec2; height: number; reach: number } {
  const at = centroid(room.polygon)
  const reach = Math.max(2.5, Math.sqrt(polygonArea(room.polygon)) * 1.1)
  return { at, height: Math.max(1.8, room.ceilingHeight - 0.45), reach }
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
