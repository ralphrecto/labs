// Home score: interior-design rules of thumb, checked against the real plan geometry,
// turned into goals, stars, a level and a "vibe" number for the game HUD. Pure logic, no React.
import type { Analysis } from '../analysis'
import { COMFORTABLE_WALKWAY, MIN_WALKWAY, placedItems, walkingPaths, type PathReport } from '../model/checks'
import { BEDROOM, DINING, UTILITY_ROOM } from '../model/placement'
import { add as add2, closestPointOnSegment, convexOverlap, dist, dot, lerp, norm, openingEndpoints, pointInPolygon, polygonArea, polygonInsideRoom, rot, scale, sub, v, wallInwardNormal, wallSegments } from '../model/geometry'
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
  /**
   * Bonus goals only show up once certain pieces exist (a nightstand goal per bed...). They add
   * stars but never raise the bar for a level, so buying furniture can't cost you a level.
   */
  bonus?: boolean
  /** Waiting on something else first (a few pieces of furniture): the sheet folds these into one line. */
  locked?: boolean
  /** The furniture drawer tab that helps with this goal ("Show me"), and the catalog piece to point at. */
  tab?: string
  item?: string
  /** The room that piece should go in, when the goal is about a particular room. */
  placeIn?: string
}

export interface HomeScore {
  stars: number
  maxStars: number
  /** Stars on offer from the core goals, which exist whatever furniture you have. Levels are based on these. */
  coreMax: number
  /** Stars needed for each of LEVELS. */
  thresholds: number[]
  level: { name: string; index: number; next?: string; toNext: number }
  goals: Goal[]
  /** 0..100 */
  vibe: number
  /** Error-level issues (overlaps, walls, doors, unreachable rooms) in the layout. */
  errors: number
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
  /** Designer norms for walkways: main circulation, and the minimum for secondary paths (bathroom, hall). */
  mainWalkway: COMFORTABLE_WALKWAY,
  minWalkway: MIN_WALKWAY,
  /** A lamp this close to a seat lights it. */
  lampReach: 1.0,
  /** The bed's head counts as against a wall within this gap. */
  headboardGap: 0.1,
}

const SEATING = new Set(['sofa', 'armchair'])
const STORAGE = new Set(['dresser', 'wardrobe', 'bookshelf', 'sideboard', 'console-table'])
/** Flow goals only count once the home has this many solid pieces. */
export const MIN_PIECES = 4
/** A room this big with no walls between living and eating has space for a dining table. */
const OPEN_PLAN_AREA = 20
/** Smallest room that takes a bed with space to walk round it. */
const BED_ROOM_AREA = 7

/**
 * The front door when none is marked: the only door that leads outside (one that doesn't open
 * into another room), or the only door at all. Null when it can't be told.
 */
export function inferEntrance(doc: Doc): { roomId: string; openingId: string } | null {
  const doors = doc.rooms.flatMap((room) => room.openings.filter((o) => o.kind === 'door').map((o) => ({ room, o })))
  if (doors.some((d) => d.o.entrance)) return null
  if (doors.length === 1) return { roomId: doors[0].room.id, openingId: doors[0].o.id }
  const outside = doors.filter(({ room, o }) => {
    const [a, b] = openingEndpoints(room, o)
    const out = sub(scale(add2(a, b), 0.5), scale(wallInwardNormal(room, o.wall), 0.35))
    return !doc.rooms.some((r) => r.id !== room.id && pointInPolygon(out, r.polygon))
  })
  return outside.length === 1 ? { roomId: outside[0].room.id, openingId: outside[0].o.id } : null
}

/** Walking paths, from the marked front door or, failing that, the one inferEntrance finds. */
function pathsWithEntrance(doc: Doc, layout: Layout, paths: PathReport): PathReport {
  if (paths.entrance) return paths
  const e = inferEntrance(doc)
  if (!e) return paths
  const rooms = doc.rooms.map((r) => (r.id === e.roomId ? { ...r, openings: r.openings.map((o) => (o.id === e.openingId ? { ...o, entrance: true } : o)) } : r))
  return walkingPaths({ ...doc, rooms }, layout)
}

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
  /** Inside a room and free of errors (overlaps, walls, door swings): only these count toward goals. */
  clean: boolean
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

/** Fractions of the core stars needed for each of LEVELS (level 2 takes one star). Early levels come quickly; bonus stars help with the last ones. */
export const LEVEL_FRACTIONS = [0, 0, 0.35, 0.75, 0.9, 1]

/**
 * Stars needed for each of LEVELS, from the stars the core goals offer. The core goals depend on
 * the floor plan, not the furniture, so the bar doesn't move when you buy something; bonus goals
 * only add stars.
 */
export function levelThresholds(coreMax: number): number[] {
  const out: number[] = []
  LEVEL_FRACTIONS.forEach((f, i) => out.push(i === 0 ? 0 : Math.max(i === 1 ? 1 : Math.ceil(coreMax * f), out[i - 1] + 1)))
  return out
}

function levelAt(stars: number, thresholds: number[], force?: number): HomeScore['level'] {
  let index = 0
  for (let i = 0; i < thresholds.length; i++) if (stars >= thresholds[i]) index = i
  if (force !== undefined) index = Math.min(LEVELS.length - 1, Math.max(0, force))
  return {
    name: LEVELS[index],
    index,
    next: LEVELS[index + 1],
    toNext: index + 1 < thresholds.length ? Math.max(0, thresholds[index + 1] - stars) : 0,
  }
}

/** 0..1 progress from the current level's threshold toward the next one. */
export function levelProgress(score: HomeScore): number {
  if (score.level.next === undefined) return 1
  const t = score.thresholds
  const from = t[score.level.index]
  const to = t[score.level.index + 1]
  return to > from ? clamp01((score.stars - from) / (to - from)) : 1
}

/**
 * The level shown to the player, given the best level reached so far in this home. It never drops
 * below the best (buying, moving or removing furniture never costs a level), and while the layout
 * has errors it doesn't rise past the best either: a broken layout can't earn a level.
 */
export function withBestLevel(score: HomeScore, best: number): HomeScore {
  const index = score.errors > 0 ? Math.max(best, Math.min(score.level.index, best)) : Math.max(best, score.level.index)
  if (index === score.level.index) return score
  const level = levelAt(score.stars, score.thresholds, index)
  return { ...score, level, headline: score.headline === score.level.name ? level.name : score.headline }
}

/**
 * The goal to suggest next: the unfinished one closest to done that the player can act on now
 * (goals waiting on something else are skipped), favouring ones the furniture drawer finishes.
 */
export function nextGoal(score: HomeScore): Goal | undefined {
  const open = score.goals.filter((g) => !g.done && !g.locked)
  // score.goals is already ranked (closest to done, basics first). A goal a piece of furniture
  // finishes is the easiest next step, so the first of those wins; then the first core goal.
  return open.find((g) => g.tab) ?? open.find((g) => !g.bonus) ?? open[0]
}

/** Where the best level is remembered: per design, so designs of one home can be compared. */
export const bestLevelKey = (doc: Doc) => `${doc.apartment.id}:${doc.activeLayoutId}`

/** The level to remember as the best reached: only a layout without errors banks a level. */
export function bankableLevel(score: HomeScore): number {
  return score.errors > 0 ? 0 : score.level.index
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
      clean: false,
    }
  })
  for (const x of placed) x.clean = x.inside && analysis.flagged.get(x.id) !== 'error'
  /** Pieces of these categories that count (inside a room, no errors). */
  const of = (...cats: string[]) => placed.filter((x) => x.clean && cats.includes(x.category))
  /** Any pieces of these categories, for hints. */
  const ofAny = (...cats: string[]) => placed.filter((x) => cats.includes(x.category))
  const inRoom = (room: Room) => placed.filter((x) => x.clean && x.room?.id === room.id)
  const issuesOf = (kind: string) => analysis.issues.filter((i) => i.kind === kind && i.severity !== 'info')
  const goals: Goal[] = []
  const add = (g: Omit<Goal, 'progress'> & { progress?: number }) => goals.push({ ...g, progress: g.done ? 1 : g.progress })
  const rooms = doc.rooms.filter((r) => polygonArea(r.polygon) >= RULES.minRoomArea)
  const anything = placed.some((x) => !x.floor)
  const solidPieces = placed.filter((x) => !x.floor)
  const cleanShare = solidPieces.length ? solidPieces.filter((x) => x.clean).length / solidPieces.length : 0
  /** Flow goals grow with the furniture: clean share, scaled until the home has a few pieces. */
  const flowProgress = cleanShare * Math.min(1, solidPieces.length / MIN_PIECES)
  const furnished = solidPieces.length >= MIN_PIECES
  const fewHint = `Place at least ${MIN_PIECES} pieces of furniture first.`
  const bedrooms = rooms.filter((r) => BEDROOM.test(r.name) && !UTILITY_ROOM.test(r.name))
  const livable = rooms.filter((r) => !UTILITY_ROOM.test(r.name))
  // Goals that need a kind of room only show when the plan has one. With no rooms yet, show them all.
  // A studio (one livable room, or one named so) takes the bed anywhere.
  const studio = !bedrooms.length && (livable.length === 1 || livable.some((r) => /studio|loft/i.test(r.name)))
  const canSleep = !rooms.length || bedrooms.length > 0 || (studio && livable.some((r) => polygonArea(r.polygon) >= BED_ROOM_AREA))
  const canEat = !rooms.length || livable.some((r) => DINING.test(r.name) || (!bedrooms.includes(r) && polygonArea(r.polygon) >= OPEN_PLAN_AREA))
  const paths = pathsWithEntrance(doc, layout, analysis.paths)

  // ---- Basics -------------------------------------------------------------
  const beds = of('bed')
  // With a bedroom in the plan, the bed belongs there (a studio takes it anywhere).
  const sleepBeds = bedrooms.length ? beds.filter((b) => b.room && bedrooms.includes(b.room)) : beds
  if (canSleep) add({
    id: 'sleep',
    title: bedrooms.length || !rooms.length ? 'Somewhere to sleep' : 'A cozy corner to sleep',
    tab: 'bedroom',
    item: 'bed',
    hint: !ofAny('bed').length
      ? bedrooms.length || !rooms.length
        ? 'Add a bed from the Bedroom tab.'
        : 'Add a bed from the Bedroom tab. In a one-room home it can go anywhere.'
      : beds.length && bedrooms.length
        ? `Move the bed into the ${bedrooms[0].name}.`
        : 'Move the bed fully inside a room, clear of other furniture and doors.',
    done: sleepBeds.length > 0,
    progress: beds.length ? 0.6 : ofAny('bed').length ? 0.4 : 0,
    stars: 2,
    category: 'basics',
  })
  const seats = of(...SEATING)
  add({
    id: 'sit',
    title: 'Somewhere to sit',
    hint: ofAny(...SEATING).length ? 'Move your seat fully inside a room, clear of other furniture.' : 'Add a sofa or an armchair.',
    tab: 'living',
    item: 'sofa',
    done: seats.length > 0,
    progress: ofAny(...SEATING).length ? 0.5 : 0,
    stars: 1,
    category: 'basics',
  })
  const tables = of('dining-table')
  const chairs = of('dining-chair', 'bench')
  const servedTable = tables.some((t) => chairs.some((c) => polygonGap(t.body, c.body) <= RULES.chairToTable))
  if (canEat) add({
    id: 'eat',
    title: 'A table to eat at',
    hint: !tables.length ? 'Add a dining table and a chair or two.' : 'Pull a chair up to the dining table.',
    tab: 'dining',
    item: tables.length ? 'dining-chair' : 'dining-table',
    done: servedTable,
    progress: tables.length ? 0.5 : chairs.length ? 0.25 : 0,
    stars: 2,
    category: 'basics',
  })
  add({
    id: 'storage',
    title: 'A place for your things',
    hint: 'Add a dresser, wardrobe, bookshelf or sideboard.',
    tab: bedrooms.length ? 'bedroom' : 'work',
    item: bedrooms.length ? 'dresser' : 'bookshelf',
    done: placed.some((x) => STORAGE.has(x.category) && x.clean),
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
      tab: 'decor',
      item: 'plant',
      placeIn: empty[0]?.id,
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
    hint: overlaps.length ? `${overlaps[0].message}. Nudge them apart.` : furnished ? '' : fewHint,
    done: furnished && overlaps.length === 0,
    locked: !furnished && !overlaps.length,
    progress: overlaps.length ? flowProgress * 0.9 : flowProgress,
    stars: 1,
    category: 'flow',
  })
  const walls = issuesOf('wall')
  add({
    id: 'in-walls',
    title: 'Everything inside the walls',
    hint: walls.length ? `${walls[0].message}.` : furnished ? '' : fewHint,
    done: furnished && walls.length === 0,
    locked: !furnished && !walls.length,
    progress: walls.length ? flowProgress * 0.9 : flowProgress,
    stars: 1,
    category: 'flow',
  })
  const doors = issuesOf('door')
  add({
    id: 'doors-clear',
    title: 'Doors swing freely',
    hint: doors.length ? `${doors[0].message}.` : furnished ? '' : fewHint,
    done: furnished && doors.length === 0,
    locked: !furnished && !doors.length,
    progress: doors.length ? flowProgress * 0.9 : flowProgress,
    stars: 1,
    category: 'flow',
  })
  // Designers plan 90 cm for the main routes (to living spaces) and at least 60 cm for secondary ones.
  const routes = paths.results.map((r) => {
    const need = UTILITY_ROOM.test(r.roomName) ? RULES.minWalkway : RULES.mainWalkway
    return { r, need, ok: r.reachable && r.narrowest >= need - 0.005 }
  })
  const tight = routes.find((x) => !x.ok)
  // One room with no door to walk in from has no routes to check.
  if (paths.entrance || rooms.length > 1 || !rooms.length) add({
    id: 'walkways',
    title: 'Clear walkways',
    locked: paths.entrance && !furnished && !tight,
    hint: !paths.entrance
      ? 'Mark your front door in Build mode so paths can be checked.'
      : tight
        ? !tight.r.reachable
          ? `There's no way to walk to the ${tight.r.roomName}; clear a path.`
          : `The way to the ${tight.r.roomName} narrows to ${cm(tight.r.narrowest)}; keep ${cm(tight.need)} free.`
        : furnished
          ? ''
          : fewHint,
    done: paths.entrance && furnished && !tight,
    progress: routes.length ? (routes.filter((x) => x.ok).length / routes.length) * Math.min(1, solidPieces.length / MIN_PIECES) : 0,
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
      tab: f < RULES.densityMin ? (bedrooms.includes(room) ? 'bedroom' : 'living') : undefined,
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
      tab: Number.isFinite(best) ? undefined : 'bedroom',
      item: 'nightstand',
      done: best <= RULES.nightstandGap,
      progress: Number.isFinite(best) ? 0.5 : 0,
      stars: 1,
      roomId: bed.room?.id,
      category: 'comfort',
      bonus: true,
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
      tab: gap === undefined ? 'living' : undefined,
      item: 'coffee-table',
      done: gap !== undefined && gap >= RULES.coffeeMin - 0.005 && gap <= RULES.coffeeMax + 0.005,
      progress: gap === undefined ? 0 : 0.6,
      stars: 1,
      roomId: sofa.room?.id,
      category: 'comfort',
      bonus: true,
    })
  }
  // Soft light: a floor lamp near a seat, in a living space (not a bathroom or hall).
  const livingSeats = of(...SEATING).filter((x) => x.room && !UTILITY_ROOM.test(x.room.name))
  const lamps = of('floor-lamp')
  const lampGap = livingSeats.flatMap((st) => lamps.filter((l) => l.room?.id === st.room?.id).map((l) => polygonGap(st.body, l.body)))
  const nearestLamp = lampGap.length ? Math.min(...lampGap) : Infinity
  add({
    id: 'light',
    title: 'Soft light by the seats',
    hint: !livingSeats.length
      ? 'Add a sofa or armchair, then a floor lamp beside it.'
      : Number.isFinite(nearestLamp)
        ? `Your lamp is ${cm(nearestLamp)} from the seats; bring it within ${cm(RULES.lampReach)}.`
        : `Add a floor lamp near the seats in the ${livingSeats[0].room!.name}.`,
    tab: !livingSeats.length ? 'living' : Number.isFinite(nearestLamp) ? undefined : 'decor',
    item: !livingSeats.length ? 'armchair' : 'floor-lamp',
    done: nearestLamp <= RULES.lampReach,
    progress: Number.isFinite(nearestLamp) ? 0.6 : lamps.length ? 0.3 : 0,
    stars: 1,
    category: 'comfort',
  })
  // A restful bed: headboard against a solid wall, not under a window.
  const headboard = (b: P) => {
    if (!b.room) return { wall: false, window: false }
    const back = add2(b.center, rot(v(0, -b.hd), b.rotation))
    const segs = wallSegments(b.room)
    let wi = -1
    let best = Infinity
    segs.forEach(([a, c], i) => {
      const d = closestPointOnSegment(back, a, c).distance
      if (d < best) {
        best = d
        wi = i
      }
    })
    if (best > RULES.headboardGap) return { wall: false, window: false }
    const left = add2(b.center, rot(v(-b.hw, -b.hd), b.rotation))
    const right = add2(b.center, rot(v(b.hw, -b.hd), b.rotation))
    const [a, c] = segs[wi]
    const dir = norm(sub(c, a))
    const t0 = Math.min(dot(sub(left, a), dir), dot(sub(right, a), dir))
    const t1 = Math.max(dot(sub(left, a), dir), dot(sub(right, a), dir))
    const window = b.room.openings.some((o) => o.kind === 'window' && o.wall === wi && o.offset < t1 && o.offset + o.width > t0)
    return { wall: true, window }
  }
  const bedChecks = sleepBeds.map((b) => ({ b, ...headboard(b) }))
  const goodBed = bedChecks.find((x) => x.wall && !x.window)
  if (canSleep) add({
    id: 'headboard',
    title: 'A restful bed',
    tab: sleepBeds.length ? undefined : 'bedroom',
    item: 'bed',
    hint: !sleepBeds.length
      ? bedrooms.length
        ? 'Put a bed in the bedroom first.'
        : 'Add a bed first.'
      : bedChecks.some((x) => x.wall)
        ? 'Your headboard is under a window; move the bed to a solid wall.'
        : 'Push the head of the bed against a wall.',
    done: !!goodBed,
    progress: bedChecks.some((x) => x.wall) ? 0.6 : sleepBeds.length ? 0.3 : 0,
    stars: 1,
    category: 'comfort',
  })
  // A spot to work: a desk with a chair pulled up to it.
  const desksAll = of('desk')
  const worked = desksAll.some((dk) => of('office-chair', 'dining-chair').some((c) => polygonGap(dk.body, c.body) <= RULES.chairToTable && toLocal(dk, c.center).y > 0))
  add({
    id: 'work',
    title: 'A spot to work',
    hint: !desksAll.length ? 'Add a desk and a chair from the Work tab.' : 'Pull a chair up to the front of the desk.',
    tab: 'work',
    item: desksAll.length ? 'office-chair' : 'desk',
    done: worked,
    progress: desksAll.length ? 0.5 : 0,
    stars: 1,
    category: 'comfort',
  })
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
      bonus: true,
    })
  }

  // ---- Style --------------------------------------------------------------
  add({
    id: 'plant',
    title: 'Bring in some greenery',
    hint: 'Add a plant. Corners and windows love them.',
    tab: 'decor',
    item: 'plant',
    done: of('plant').some((x) => x.inside),
    stars: 1,
    category: 'style',
  })
  if (sofas.length || of('coffee-table').length) {
    // The rug and the seat it anchors share a room that isn't a bathroom or hall.
    const rugs = of('rug').filter((r) => r.room && !UTILITY_ROOM.test(r.room.name))
    const anchors = of('sofa', 'coffee-table', 'armchair')
    add({
      id: 'rug',
      title: 'A rug to anchor the seating',
      hint: ofAny('rug').length ? 'Slide the rug under the coffee table or the front of the sofa.' : 'Add a rug under the sofa and coffee table.',
      tab: ofAny('rug').length ? undefined : 'decor',
      item: 'rug',
      done: rugs.some((r) => anchors.some((a) => a.room?.id === r.room?.id && convexOverlap(r.body, a.body, 0.01))),
      progress: ofAny('rug').length ? 0.5 : 0,
      stars: 1,
      category: 'style',
      bonus: true,
    })
  }
  const armchairs = of('armchair')
  const nook = armchairs.some((a) => of('floor-lamp', 'side-table', 'bookshelf').some((x) => polygonGap(a.body, x.body) <= 0.5))
  add({
    id: 'nook',
    title: 'A cozy reading corner',
    hint: ofAny('armchair').length ? 'Tuck a floor lamp, side table or bookshelf next to the armchair.' : 'Add an armchair with a lamp or bookshelf beside it.',
    tab: ofAny('armchair').length ? 'decor' : 'living',
    item: ofAny('armchair').length ? 'floor-lamp' : 'armchair',
    done: nook,
    progress: ofAny('armchair').length ? 0.5 : 0,
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
      bonus: true,
    })
  }

  // ---- Totals -------------------------------------------------------------
  const stars = goals.reduce((a, g) => a + (g.done ? g.stars : 0), 0)
  const maxStars = goals.reduce((a, g) => a + g.stars, 0)
  const coreMax = goals.reduce((a, g) => a + (g.bonus ? 0 : g.stars), 0)
  const thresholds = levelThresholds(coreMax)
  const level = levelAt(stars, thresholds)

  const errors = analysis.issues.filter((i) => i.severity === 'error').length
  const warnings = analysis.issues.filter((i) => i.severity === 'warning').length
  const vibe = maxStars ? Math.round(Math.max(0, Math.min(100, (100 * stars) / maxStars - 8 * errors - 3 * warnings))) : 0

  const headline =
    !anything
      ? 'Let’s move in!'
      : errors > 0
        ? 'Almost! Fix a snag'
        : level.name

  // Next achievable first: unfinished goals by progress, then basics before style, then cheaper first.
  const rank = (g: Goal) => CATEGORY_ORDER.indexOf(g.category)
  goals.sort((a, b) => Number(a.done) - Number(b.done) || (b.progress ?? 0) - (a.progress ?? 0) || rank(a) - rank(b) || a.stars - b.stars)

  return { stars, maxStars, coreMax, thresholds, level, goals, vibe, errors, headline }
}
