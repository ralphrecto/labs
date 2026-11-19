// Home score: interior-design rules of thumb, checked against the real plan geometry,
// turned into goals, stars, a level and a "vibe" number for the game HUD. Pure logic, no React.
import type { Analysis } from '../analysis'
import { placedItems } from '../model/checks'
import { closestPointOnSegment, convexOverlap, dist, dot, lerp, norm, openingEndpoints, pointInPolygon, polygonArea, polygonInsideRoom, rot, sub, v } from '../model/geometry'
import type { Doc, Layout, Room, Vec2 } from '../model/types'

export type GoalCategory = 'basics' | 'comfort' | 'flow' | 'style'

export interface Goal {
  id: string
  title: string
  hint: string
  done: boolean
  /** 0..1 */
  progress?: number
  /** Reward, 1-3. */
  stars: number
  roomId?: string
  category: GoalCategory
}

export interface HomeScore {
  stars: number
  maxStars: number
  level: { name: string; index: number; next?: string; toNext: number }
  goals: Goal[]
  /** 0..100 */
  vibe: number
  headline: string
}

export const LEVELS = ['Empty shell', 'Moving in', 'Lived-in', 'Cozy', 'Designer home', 'Dream home'] as const
export const CATEGORY_ORDER: GoalCategory[] = ['basics', 'flow', 'comfort', 'style']
export const CATEGORY_LABEL: Record<GoalCategory, string> = { basics: 'The basics', flow: 'Easy flow', comfort: 'Comfort', style: 'Style' }

/** Rule-of-thumb numbers, in meters. */
export const RULES = {
  nightstandGap: 0.6,
  coffeeMin: 0.35,
  coffeeMax: 0.5,
  densityMin: 0.15,
  densityMax: 0.45,
  deskWindow: 1.5,
  tvAngleDeg: 20,
  chairToTable: 0.6,
  /** Rooms smaller than this (closets) are left out of the per-room goals. */
  minRoomArea: 3,
}

const SEATING = new Set(['sofa', 'armchair'])
const STORAGE = new Set(['dresser', 'wardrobe', 'bookshelf', 'sideboard', 'console-table'])
const UTILITY_ROOM = /bath|wc|toilet|closet|laundry|storage|utility|pantry|hall|entry|corridor/i

interface P {
  id: string
  category: string
  name: string
  center: Vec2
  rotation: number
  hw: number
  hd: number
  body: Vec2[]
  floor: boolean
  room?: Room
  inside: boolean
}

const front = (p: P) => rot(v(0, 1), p.rotation)
/** A point in p's local frame (x to its side, +y out of its front). */
const toLocal = (p: P, q: Vec2) => rot(sub(q, p.center), -p.rotation)

/** Smallest distance between two convex polygons (0 when they touch or overlap). */
export function polygonGap(a: Vec2[], b: Vec2[]): number {
  if (convexOverlap(a, b, 0)) return 0
  let best = Infinity
  for (const [pts, poly] of [
    [a, b],
    [b, a],
  ] as const)
    for (const pt of pts)
      for (let i = 0; i < poly.length; i++) best = Math.min(best, closestPointOnSegment(pt, poly[i], poly[(i + 1) % poly.length]).distance)
  return best
}

const clamp01 = (x: number) => Math.max(0, Math.min(1, x))
const cm = (m: number) => `${Math.round(m * 100)} cm`

/** Stars needed for each of LEVELS, given how many stars the current goals offer. */
export function levelThresholds(maxStars: number): number[] {
  const raw = [0, 1, Math.ceil(maxStars * 0.25), Math.ceil(maxStars * 0.5), Math.ceil(maxStars * 0.75), maxStars]
  const out: number[] = []
  raw.forEach((t, i) => out.push(i ? Math.max(t, out[i - 1] + 1) : 0))
  return out
}

/** 0..1 progress from the current level's threshold toward the next one. */
export function levelProgress(score: HomeScore): number {
  if (score.level.next === undefined) return 1
  const t = levelThresholds(score.maxStars)
  const from = t[score.level.index]
  const to = t[score.level.index + 1]
  return to > from ? clamp01((score.stars - from) / (to - from)) : 1
}

export function scoreHome(doc: Doc, layout: Layout, analysis: Analysis): HomeScore {
  const placed: P[] = placedItems(doc, layout).map(({ p, item, body }) => {
    const center = v(p.x, p.y)
    const room = doc.rooms.find((r) => pointInPolygon(center, r.polygon))
    return {
      id: p.id,
      category: item.category,
      name: item.name,
      center,
      rotation: p.rotation,
      hw: item.width / 2,
      hd: item.depth / 2,
      body,
      floor: !!item.floorLevel,
      room,
      inside: !!room && polygonInsideRoom(body, room.polygon),
    }
  })
  const of = (...cats: string[]) => placed.filter((x) => cats.includes(x.category))
  const inRoom = (room: Room) => placed.filter((x) => x.room?.id === room.id)
  const issuesOf = (kind: string) => analysis.issues.filter((i) => i.kind === kind && i.severity !== 'info')
  const goals: Goal[] = []
  const add = (g: Omit<Goal, 'progress'> & { progress?: number }) => goals.push({ ...g, progress: g.done ? 1 : g.progress })
  const rooms = doc.rooms.filter((r) => polygonArea(r.polygon) >= RULES.minRoomArea)
  const anything = placed.some((x) => !x.floor)

  // ---- Basics -------------------------------------------------------------
  const beds = of('bed')
  add({
    id: 'sleep',
    title: 'Somewhere to sleep',
    hint: beds.length ? 'Move the bed fully inside a room.' : 'Add a bed from the Bedroom tab.',
    done: beds.some((b) => b.inside),
    progress: beds.length ? 0.5 : 0,
    stars: 2,
    category: 'basics',
  })
  const seats = of(...SEATING)
  add({
    id: 'sit',
    title: 'Somewhere to sit',
    hint: seats.length ? 'Move your seat fully inside a room.' : 'Add a sofa or an armchair.',
    done: seats.some((s) => s.inside),
    progress: seats.length ? 0.5 : 0,
    stars: 1,
    category: 'basics',
  })
  const tables = of('dining-table')
  const chairs = of('dining-chair', 'bench')
  const servedTable = tables.some((t) => chairs.some((c) => polygonGap(t.body, c.body) <= RULES.chairToTable))
  add({
    id: 'eat',
    title: 'A table to eat at',
    hint: !tables.length ? 'Add a dining table and a chair or two.' : 'Pull a chair up to the dining table.',
    done: servedTable,
    progress: tables.length ? 0.5 : chairs.length ? 0.25 : 0,
    stars: 2,
    category: 'basics',
  })
  add({
    id: 'storage',
    title: 'A place for your things',
    hint: 'Add a dresser, wardrobe, bookshelf or sideboard.',
    done: placed.some((x) => STORAGE.has(x.category) && x.inside),
    stars: 1,
    category: 'basics',
  })
  if (rooms.length) {
    const filled = rooms.filter((r) => inRoom(r).length > 0)
    const empty = rooms.filter((r) => !inRoom(r).length)
    add({
      id: 'rooms-used',
      title: 'Every room has something',
      hint: empty.length ? `Nothing in the ${empty[0].name} yet. A plant works anywhere!` : '',
      done: filled.length === rooms.length,
      progress: filled.length / rooms.length,
      stars: 1,
      category: 'basics',
    })
  }

  // ---- Flow ---------------------------------------------------------------
  const overlaps = issuesOf('overlap')
  add({
    id: 'no-overlap',
    title: 'No furniture pile-ups',
    hint: anything ? (overlaps.length ? `${overlaps[0].message}. Nudge them apart.` : '') : 'Place some furniture first.',
    done: anything && overlaps.length === 0,
    stars: 1,
    category: 'flow',
  })
  const walls = issuesOf('wall')
  add({
    id: 'in-walls',
    title: 'Everything inside the walls',
    hint: anything ? (walls.length ? `${walls[0].message}.` : '') : 'Place some furniture first.',
    done: anything && walls.length === 0,
    stars: 1,
    category: 'flow',
  })
  const doors = issuesOf('door')
  add({
    id: 'doors-clear',
    title: 'Doors swing freely',
    hint: anything ? (doors.length ? `${doors[0].message}.` : '') : 'Place some furniture first.',
    done: anything && doors.length === 0,
    stars: 1,
    category: 'flow',
  })
  const paths = issuesOf('path')
  add({
    id: 'walkways',
    title: 'Clear walkways',
    hint: !analysis.paths.entrance
      ? 'Mark your front door in Build mode so paths can be checked.'
      : !anything
        ? 'Place some furniture first.'
        : paths.length
          ? `${paths[0].message}.`
          : '',
    done: analysis.paths.entrance && anything && paths.length === 0,
    progress: analysis.paths.results.length ? analysis.paths.results.filter((r) => r.reachable && r.narrowest >= 0.6).length / analysis.paths.results.length : 0,
    stars: 2,
    category: 'flow',
  })
  for (const room of rooms) {
    if (UTILITY_ROOM.test(room.name)) continue
    const area = polygonArea(room.polygon)
    const used = inRoom(room)
      .filter((x) => !x.floor)
      .reduce((a, x) => a + polygonArea(x.body), 0)
    const f = used / area
    const pct = Math.round(f * 100)
    add({
      id: `density:${room.id}`,
      title: `${room.name} feels just right`,
      hint:
        f >= RULES.densityMin && f <= RULES.densityMax
          ? `Furniture covers ${pct}% of the floor. Just right.`
          : f < RULES.densityMin
          ? `Furniture covers ${pct}% of the floor; add a few pieces (aim for 15–45%).`
          : `Furniture covers ${pct}% of the floor; it feels crowded (aim for 15–45%).`,
      done: f >= RULES.densityMin && f <= RULES.densityMax,
      progress: f < RULES.densityMin ? f / RULES.densityMin : clamp01(1 - (f - RULES.densityMax) / RULES.densityMax),
      stars: 1,
      roomId: room.id,
      category: 'flow',
    })
  }

  // ---- Comfort ------------------------------------------------------------
  for (const bed of beds) {
    const nearby = of('nightstand', 'side-table')
      .map((n) => {
        const l = toLocal(bed, n.center)
        return { n, gap: polygonGap(bed.body, n.body), side: Math.abs(l.x) >= bed.hw * 0.9, head: l.y <= 0 }
      })
      .filter((x) => x.side && x.head)
    const best = nearby.reduce((m, x) => Math.min(m, x.gap), Infinity)
    const roomName = bed.room?.name
    add({
      id: `nightstand:${bed.id}`,
      title: `Nightstand by the ${bed.name.toLowerCase()}${roomName ? ` (${roomName})` : ''}`,
      hint: Number.isFinite(best)
        ? `Your nightstand is ${cm(best)} from the bed; slide it within ${cm(RULES.nightstandGap)} of the side, by the pillows.`
        : 'Put a nightstand right beside the head of the bed.',
      done: best <= RULES.nightstandGap,
      progress: Number.isFinite(best) ? 0.5 : 0,
      stars: 1,
      roomId: bed.room?.id,
      category: 'comfort',
    })
  }
  const sofas = of('sofa')
  for (const sofa of sofas.slice(0, 2)) {
    const ahead = of('coffee-table', 'ottoman')
      .map((t) => ({ t, l: toLocal(sofa, t.center), gap: polygonGap(sofa.body, t.body) }))
      .filter((x) => x.l.y > sofa.hd && Math.abs(x.l.x) < sofa.hw)
      .sort((a, b) => a.gap - b.gap)[0]
    const gap = ahead?.gap
    add({
      id: `coffee:${sofa.id}`,
      title: 'Coffee table within reach',
      hint:
        gap === undefined
          ? 'Put a coffee table in front of the sofa, 35–50 cm from the seat.'
          : gap < RULES.coffeeMin
            ? `It's ${cm(gap)} from the sofa; give your knees room (35–50 cm).`
            : `It's ${cm(gap)} from the sofa; pull it closer (35–50 cm).`,
      done: gap !== undefined && gap >= RULES.coffeeMin - 0.005 && gap <= RULES.coffeeMax + 0.005,
      progress: gap === undefined ? 0 : 0.6,
      stars: 1,
      roomId: sofa.room?.id,
      category: 'comfort',
    })
  }
  for (const room of rooms) {
    const seating = inRoom(room).filter((x) => SEATING.has(x.category))
    if (!seating.length) continue
    add({
      id: `lamp:${room.id}`,
      title: `Soft light in the ${room.name}`,
      hint: `Add a floor lamp near the seats in the ${room.name}.`,
      done: inRoom(room).some((x) => x.category === 'floor-lamp'),
      stars: 1,
      roomId: room.id,
      category: 'comfort',
    })
  }
  const windows: Vec2[] = doc.rooms.flatMap((r) =>
    r.openings.filter((o) => o.kind === 'window').map((o) => {
      const [a, b] = openingEndpoints(r, o)
      return lerp(a, b, 0.5)
    }),
  )
  const desks = of('desk')
  if (desks.length && windows.length) {
    const best = Math.min(...desks.flatMap((d) => windows.map((w) => dist(d.center, w) - d.hd)))
    add({
      id: 'desk-window',
      title: 'Desk with daylight',
      hint: `Your desk is ${best.toFixed(1)} m from the nearest window; move it within ${RULES.deskWindow} m.`,
      done: best <= RULES.deskWindow,
      progress: clamp01(RULES.deskWindow / Math.max(best, 0.01)),
      stars: 1,
      category: 'comfort',
    })
  }

  // ---- Style --------------------------------------------------------------
  add({
    id: 'plant',
    title: 'Bring in some greenery',
    hint: 'Add a plant. Corners and windows love them.',
    done: of('plant').some((x) => x.inside),
    stars: 1,
    category: 'style',
  })
  if (sofas.length || of('coffee-table').length) {
    const rugs = of('rug')
    const anchors = of('sofa', 'coffee-table', 'armchair')
    add({
      id: 'rug',
      title: 'A rug to anchor the seating',
      hint: rugs.length ? 'Slide the rug under the coffee table or the front of the sofa.' : 'Add a rug under the sofa and coffee table.',
      done: rugs.some((r) => anchors.some((a) => convexOverlap(r.body, a.body, 0.01))),
      progress: rugs.length ? 0.5 : 0,
      stars: 1,
      category: 'style',
    })
  }
  const armchairs = of('armchair')
  const nook = armchairs.some((a) => of('floor-lamp', 'side-table', 'bookshelf').some((x) => polygonGap(a.body, x.body) <= 0.5))
  add({
    id: 'nook',
    title: 'A cozy reading corner',
    hint: armchairs.length ? 'Tuck a floor lamp, side table or bookshelf next to the armchair.' : 'Add an armchair with a lamp or bookshelf beside it.',
    done: nook,
    progress: armchairs.length ? 0.5 : 0,
    stars: 2,
    category: 'style',
  })
  const tvs = of('tv-stand')
  if (sofas.length && tvs.length) {
    let bestDeg = 180
    for (const s of sofas)
      for (const t of tvs) {
        const c = Math.max(-1, Math.min(1, dot(front(s), norm(sub(t.center, s.center)))))
        bestDeg = Math.min(bestDeg, (Math.acos(c) * 180) / Math.PI)
      }
    add({
      id: 'sofa-tv',
      title: 'Sofa faces the TV',
      hint: bestDeg <= RULES.tvAngleDeg ? 'Movie night is ready.' : `The sofa is turned ${Math.round(bestDeg)}° away from the TV; rotate it to face the screen.`,
      done: bestDeg <= RULES.tvAngleDeg,
      progress: clamp01(1 - bestDeg / 180),
      stars: 1,
      category: 'style',
    })
  }

  // ---- Totals -------------------------------------------------------------
  const stars = goals.reduce((a, g) => a + (g.done ? g.stars : 0), 0)
  const maxStars = goals.reduce((a, g) => a + g.stars, 0)
  const thresholds = levelThresholds(maxStars)
  let index = 0
  for (let i = 0; i < thresholds.length; i++) if (stars >= thresholds[i]) index = i
  const level = {
    name: LEVELS[index],
    index,
    next: LEVELS[index + 1],
    toNext: index + 1 < thresholds.length ? thresholds[index + 1] - stars : 0,
  }

  const errors = analysis.issues.filter((i) => i.severity === 'error').length
  const warnings = analysis.issues.filter((i) => i.severity === 'warning').length
  const vibe = maxStars ? Math.round(Math.max(0, Math.min(100, (100 * stars) / maxStars - 8 * errors - 3 * warnings))) : 0

  const headline =
    !anything
      ? 'Let’s move in!'
      : errors > 0
        ? 'Almost! Fix a snag'
        : (['Fresh start', 'Settling in', 'Homey', 'Cozy!', 'So stylish!', 'Dream home!'] as const)[index]

  // Next achievable first: unfinished goals by progress, then basics before style, then cheaper first.
  const rank = (g: Goal) => CATEGORY_ORDER.indexOf(g.category)
  goals.sort((a, b) => Number(a.done) - Number(b.done) || (b.progress ?? 0) - (a.progress ?? 0) || rank(a) - rank(b) || a.stars - b.stars)

  return { stars, maxStars, level, goals, vibe, headline }
}
