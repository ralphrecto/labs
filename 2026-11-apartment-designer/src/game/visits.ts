// Visitor of the day: a friendly guest drops by with one small wish for the home (Grandma wants a
// comfy chair by a lamp). A daily bonus goal worth two stars; it never holds a level back. Pure logic.
import { polygonGap } from '../model/checks'
import { dist, dot, norm, polygonArea, rot, sub, v } from '../model/geometry'
import { BEDROOM, DINING, NEXT_TO_REACH, UTILITY_ROOM } from '../model/placement'
import type { Doc, Room, Vec2 } from '../model/types'

/** A piece of furniture as the visit checks see it (scoreHome's own record fits). */
export interface VisitPiece {
  id: string
  category: string
  center: Vec2
  rotation: number
  hw: number
  hd: number
  body: Vec2[]
  host?: string
  room?: Room
}

export interface VisitCtx {
  /** Pieces that count (inside a room, free of errors, room to use them). */
  pieces: VisitPiece[]
  /** Every placed piece, for "move it" hints. */
  all: VisitPiece[]
  /** Whether a scored goal is done (undefined when the home has no such goal). */
  goal: (id: string) => boolean | undefined
}

export interface VisitCheck {
  done: boolean
  progress: number
  hint: string
  tab?: string
  item?: string
  near?: string
}

export interface Visit {
  id: string
  emoji: string
  who: string
  /** The goal's title ("A comfy chair by a lamp"). */
  title: string
  /** The welcome card's line ("Grandma's visiting: she needs a comfy chair near a lamp"). */
  line: string
  /** Whether the visit makes sense in this home (a dinner needs somewhere to eat). */
  applies?: (doc: Doc) => boolean
  check: (c: VisitCtx) => VisitCheck
}

/** Stars a visit is worth. */
export const VISIT_STARS = 2
/** A lamp this close (m, edge to edge) to a seat lights it. */
const LAMP_REACH = 1.0
const CHAIR_TO_TABLE = 0.6
const TALK_DIST = 3.2

const SEATS = ['sofa', 'armchair']
const cats = (c: VisitCtx, ...k: string[]) => c.pieces.filter((p) => k.includes(p.category))
const any = (c: VisitCtx, ...k: string[]) => c.all.filter((p) => k.includes(p.category))
const front = (p: VisitPiece) => rot(v(0, 1), p.rotation)
const livable = (doc: Doc) => doc.rooms.filter((r) => !UTILITY_ROOM.test(r.name) && polygonArea(r.polygon) >= 3)
/** Seats a piece offers to a group: a long sofa takes three. */
const capacity = (p: VisitPiece) => (p.category === 'sofa' ? (p.hw * 2 >= 1.8 ? 3 : 2) : p.category === 'bench' ? 2 : 1)
const step = (hint: string, progress: number, item?: string, tab?: string, near?: string): VisitCheck => ({ done: false, progress, hint, item, tab: item ? tab : undefined, near })

export const VISITS: Visit[] = [
  {
    id: 'grandma',
    emoji: '👵',
    who: 'Grandma',
    title: 'A comfy chair by a lamp',
    line: "Grandma's visiting: she needs a comfy chair near a lamp",
    check: (c) => {
      const seats = cats(c, ...SEATS)
      const lamps = cats(c, 'floor-lamp', 'table-lamp')
      const lit = seats.find((s) => lamps.some((l) => polygonGap(s.body, l.body) <= LAMP_REACH))
      if (lit) return { done: true, progress: 1, hint: 'Grandma has her reading spot.' }
      if (!seats.length) return step('Add an armchair for Grandma, then a lamp beside it.', 0, 'armchair', 'living', any(c, ...SEATS)[0]?.id)
      return step('Put a floor lamp right beside a seat so Grandma can read.', 0.5, 'floor-lamp', 'decor', seats[0].id)
    },
  },
  {
    id: 'book-club',
    emoji: '📚',
    who: 'Book club',
    title: 'Four seats for book club',
    line: 'Book club is coming over: they need four seats in one room',
    applies: (doc) => !doc.rooms.length || livable(doc).some((r) => !BEDROOM.test(r.name)),
    check: (c) => {
      const seats = cats(c, ...SEATS, 'ottoman')
      const byRoom = new Map<string, { n: number; near: string }>()
      for (const s of seats) {
        const k = s.room?.id ?? '-'
        const cur = byRoom.get(k)
        byRoom.set(k, { n: (cur?.n ?? 0) + capacity(s), near: cur?.near ?? s.id })
      }
      const best = [...byRoom.values()].sort((a, b) => b.n - a.n)[0]
      const n = best?.n ?? 0
      if (n >= 4) return { done: true, progress: 1, hint: 'Everyone has a seat for the chat.' }
      return step(`${n} of 4 seats in one room. Add an armchair beside the others.`, n / 4, 'armchair', 'living', best?.near)
    },
  },
  {
    id: 'sister',
    emoji: '👯',
    who: 'Your sister',
    title: 'A place for your sister to sleep',
    line: "Your sister's staying over: she needs a sofa or a guest bed",
    applies: (doc) => !doc.rooms.length || livable(doc).some((r) => polygonArea(r.polygon) >= 7),
    check: (c) => {
      const beds = cats(c, 'bed')
      const sofas = cats(c, 'sofa').filter((s) => s.hw * 2 >= 1.8)
      const spots = beds.length + sofas.length
      if (spots >= 2) return { done: true, progress: 1, hint: 'Sweet dreams, sis.' }
      if (!sofas.length) return step('A long sofa makes a cozy guest bed. Add one.', spots / 2, 'sofa', 'living', beds[0]?.id)
      return step('Add a bed, so you both have somewhere to sleep.', 0.5, 'bed', 'bedroom')
    },
  },
  {
    id: 'dog',
    emoji: '🐶',
    who: 'A friend with a dog',
    title: 'A rug and room to roam',
    line: 'A friend with a dog is dropping by: a soft rug and clear paths, please',
    check: (c) => {
      const rugs = cats(c, 'rug')
      const paths = c.goal('walkways') !== false
      if (rugs.length && paths) return { done: true, progress: 1, hint: 'Zoomies approved.' }
      if (!rugs.length) return step('Add a rug for the dog to flop on.', paths ? 0.5 : 0, 'rug', 'decor', any(c, 'rug')[0]?.id ?? cats(c, ...SEATS)[0]?.id)
      return step('Clear a wide path through the home so the dog can run.', 0.5)
    },
  },
  {
    id: 'movie',
    emoji: '🍿',
    who: 'Movie night crew',
    title: 'Movie night seating',
    line: 'Movie night crew is coming: they need a sofa facing the TV',
    check: (c) => {
      const sofas = cats(c, 'sofa')
      const tvs = cats(c, 'tv-stand')
      if (sofas.length && tvs.length && c.goal('sofa-tv')) return { done: true, progress: 1, hint: 'Popcorn time.' }
      if (!sofas.length) return step('Add a sofa for the movie crew.', tvs.length ? 0.4 : 0, 'sofa', 'living', any(c, 'sofa')[0]?.id)
      if (!tvs.length) return step('Add a TV stand across from the sofa.', 0.4, 'tv-stand', 'living', sofas[0].id)
      return step('Turn the sofa to face the TV, with nothing in between.', 0.7, undefined, undefined, sofas[0].id)
    },
  },
  {
    id: 'wfh',
    emoji: '💻',
    who: 'Work-from-home pal',
    title: 'A desk, a chair and a lamp',
    line: 'Your work-from-home pal is coming: a desk with a chair and a lamp',
    check: (c) => {
      const desks = cats(c, 'desk')
      const chairs = cats(c, 'office-chair', 'dining-chair')
      const lamps = cats(c, 'table-lamp', 'floor-lamp')
      const seated = desks.filter((d) => chairs.some((ch) => polygonGap(d.body, ch.body) <= CHAIR_TO_TABLE))
      const lit = seated.find((d) => lamps.some((l) => l.host === d.id || (!l.host && polygonGap(d.body, l.body) <= NEXT_TO_REACH)))
      if (lit) return { done: true, progress: 1, hint: 'Ready for a productive day.' }
      if (!desks.length) return step('Add a desk for your pal.', 0, 'desk', 'work', any(c, 'desk')[0]?.id)
      if (!seated.length) return step('Pull an office chair up to the desk.', 0.35, 'office-chair', 'work', desks[0].id)
      return step('Set a table lamp on the desk.', 0.7, 'table-lamp', 'decor', seated[0].id)
    },
  },
  {
    id: 'dinner',
    emoji: '🍝',
    who: 'Dinner guests',
    title: 'Dinner for four',
    line: 'Dinner guests tonight: four seats at the table',
    applies: (doc) => !doc.rooms.length || livable(doc).some((r) => DINING.test(r.name) || (!BEDROOM.test(r.name) && polygonArea(r.polygon) >= 20)),
    check: (c) => {
      const tables = cats(c, 'dining-table')
      const chairs = cats(c, 'dining-chair', 'bench')
      const at = tables.map((t) => ({ t, n: chairs.filter((ch) => polygonGap(t.body, ch.body) <= CHAIR_TO_TABLE).reduce((a, ch) => a + capacity(ch), 0) })).sort((a, b) => b.n - a.n)[0]
      if (at && at.n >= 4) return { done: true, progress: 1, hint: 'Dinner is served.' }
      if (!at) return step('Add a dining table for your guests.', 0, 'dining-table', 'dining', any(c, 'dining-table')[0]?.id)
      return step(`${at.n} of 4 seats at the table. Pull up another chair.`, 0.25 + at.n / 8, 'dining-chair', 'dining', at.t.id)
    },
  },
  {
    id: 'aunt',
    emoji: '🪴',
    who: 'Your plant-loving aunt',
    title: 'Three plants for your aunt',
    line: 'Your plant-loving aunt is visiting: she hopes to see three plants',
    check: (c) => {
      const n = cats(c, 'plant').length
      if (n >= 3) return { done: true, progress: 1, hint: 'A little jungle.' }
      return step(`${n} of 3 plants. Add ${3 - n === 1 ? 'one more' : `${3 - n} more`}.`, n / 3, 'plant', 'decor')
    },
  },
  {
    id: 'neighbour',
    emoji: '🫖',
    who: 'Your neighbour',
    title: 'Two seats for tea',
    line: 'Your neighbour is popping in for tea: two seats that face each other',
    check: (c) => {
      const seats = cats(c, ...SEATS, 'dining-chair')
      const facing = (a: VisitPiece, b: VisitPiece) =>
        a.room?.id === b.room?.id && dist(a.center, b.center) <= TALK_DIST && dot(front(a), norm(sub(b.center, a.center))) > 0.3 && dot(front(b), norm(sub(a.center, b.center))) > 0.3
      const ok = seats.some((a, i) => seats.some((b, j) => j > i && facing(a, b)))
      if (ok) return { done: true, progress: 1, hint: 'The kettle is on.' }
      const soft = cats(c, ...SEATS)
      if (soft.length < 2) return step('Add an armchair across from your seat.', soft.length / 3, 'armchair', 'living', soft[0]?.id)
      const arm = soft.find((s) => s.category === 'armchair') ?? soft[1]
      return step('Turn a chair to face the other seat, close enough to chat.', 0.7, undefined, undefined, arm.id)
    },
  },
  {
    id: 'cousins',
    emoji: '🧸',
    who: 'Little cousins',
    title: 'A play rug by the seats',
    line: 'The little cousins are coming: a soft rug under the seats to play on',
    check: (c) => {
      const seats = cats(c, ...SEATS)
      const rugs = cats(c, 'rug')
      if (seats.some((s) => rugs.some((r) => polygonGap(s.body, r.body) <= 0.01))) return { done: true, progress: 1, hint: 'Playtime!' }
      if (!seats.length) return step('Add a sofa, then a rug in front of it.', 0, 'sofa', 'living')
      if (!rugs.length) return step('Add a rug under the front of the sofa.', 0.5, 'rug', 'decor', seats[0].id)
      return step('Slide the rug under the seats.', 0.7, undefined, undefined, rugs[0].id)
    },
  },
]

export const VISIT_BY_ID: Record<string, Visit> = Object.fromEntries(VISITS.map((x) => [x.id, x]))

/** The local calendar day, "2026-11-22". */
export function localDay(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

function hash(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}

/** Visits that make sense in this home. */
export function applicableVisits(doc: Doc): Visit[] {
  return VISITS.filter((x) => !x.applies || x.applies(doc))
}

/**
 * Today's visitor: picked by the date among the visits that suit this home, preferring ones whose
 * sticker isn't collected yet and that the home doesn't already please (`met`). The same day and
 * home always give the same visitor.
 */
export function todayVisit(doc: Doc, date: string | Date = new Date(), stickers: string[] = [], met?: (id: string) => boolean): Visit {
  const day = typeof date === 'string' ? date : localDay(date)
  const ok = applicableVisits(doc)
  // Prefer a new sticker, and a wish the home doesn't already grant (something to do today).
  const tiers = [ok.filter((x) => !stickers.includes(x.id) && !met?.(x.id)), ok.filter((x) => !stickers.includes(x.id)), ok.filter((x) => !met?.(x.id)), ok, VISITS]
  const pool = tiers.find((t) => t.length)!
  return pool[hash(day) % pool.length]
}
