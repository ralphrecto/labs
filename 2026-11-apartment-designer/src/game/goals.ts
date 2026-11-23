// Home score: interior-design rules of thumb, checked against the real plan geometry,
// turned into goals, stars, a level and a "vibe" number for the game HUD. Pure logic, no React.
import { analyze, type Analysis } from '../analysis'
import { COMFORTABLE_WALKWAY, MIN_WALKWAY, placedItems, walkingPaths, type PathReport } from '../model/checks'
import { BEDROOM, BEDROOM_ONLY, DINING, LIVING, NEXT_TO_REACH, nookTarget, OFFICE, polygonGap, UTILITY_ROOM, VIEW_OK } from '../model/placement'
import { add as add2, closestPointOnSegment, convexOverlap, orientedRect, dist, dot, lerp, norm, openingEndpoints, pointInPolygon, polygonArea, polygonInsideRoom, rot, scale, sub, v, wallInwardNormal, wallSegments } from '../model/geometry'
import type { Doc, Layout, Room, Vec2 } from '../model/types'
import { centroid, normalizeAngle } from '../model/geometry'
import type { Placement, TwistId } from '../model/types'
import { catalogEntry, itemFromCatalog } from '../model/catalog'
import { findDrop } from '../model/placement'
import { isWallDecor } from '../model/wallDecor'
import { VISIT_BY_ID, VISIT_STARS } from './visits'

export { VISITS, VISIT_BY_ID, VISIT_STARS, applicableVisits, localDay, todayVisit, type Visit } from './visits'

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
  /**
   * Something in the layout is broken (an overlap, a blocked door, a tight walkway, a TV the sofa
   * can't see): Next points here before suggesting anything to buy.
   */
  snag?: boolean
  /** The placement the suggested piece belongs beside (the armchair a reading lamp is for). */
  near?: string
  /**
   * The piece in the way when the suggested one has nowhere to go (the desk filling the bedroom
   * before a bed): Next turns the goal into "make room" and selects it.
   */
  makeRoom?: { id: string; name: string; room?: string }
  /** Next suggests this before anything else (unpacking the moving boxes on move-in day). */
  urgent?: boolean
  /**
   * Today's visitor (Grandma wants a comfy chair by a lamp): a bonus that adds stars but is left
   * out of every level bar and cap, so it can lift a level but never hold one back.
   */
  visit?: string
  /**
   * A one-time reward for a skill learned (round 16: the drag lesson's star). Like a visitor, it
   * adds a star but is left out of every level bar and cap, so it can lift a level, never hold one.
   */
  perk?: string
}

/** Rewards earned once per home, as scoreHome takes them: `drag` once the drag lesson was nailed here. */
export interface Perks {
  drag?: boolean
}
/** The drag lesson's reward (round 16). */
export const DRAG_PERK = { id: 'perk:drag', title: 'Drag like a pro', stars: 1 }

/** Today's visitor, as scoreHome takes it: which visit, and whether it was already done today. */
export interface VisitInput {
  id: string
  done?: boolean
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
  /**
   * The plan is a living space with no bedroom (one scanned living room, say): Next starts with
   * the sofa, coffee table, rug, lamp and plant rather than a desk.
   */
  livingFirst?: boolean
  /**
   * The highest level the goals allow right now (snags hold it at Cozy, open core goals or warnings
   * at Designer home): when it is no higher than the level, more stars alone won't reach the next.
   */
  cap?: number
}

export const LEVELS = ['Empty shell', 'Moving in', 'Lived-in', 'Cozy', 'Designer home', 'Dream home'] as const
export const CATEGORY_ORDER: GoalCategory[] = ['basics', 'flow', 'comfort', 'style']
export const CATEGORY_LABEL: Record<GoalCategory, string> = { basics: 'The basics', flow: 'Easy flow', comfort: 'Comfort', style: 'Style' }

/** Rule-of-thumb numbers, in meters. */
export const RULES = {
  nightstandGap: 0.6,
  /** A bedside lamp within this (m, edge to edge) of the bed's side lights it. */
  bedsideLamp: 0.6,
  /** Knee room in front of the sofa: designers quote 35–45 cm, and anything up to 60 cm still reaches. */
  coffeeMin: 0.35,
  coffeeMax: 0.6,
  densityMin: 0.15,
  /** Designers aim for about a third of the floor; up to 40% still feels lived-in, not crowded. */
  densityMax: 0.4,
  densityMaxBedroom: 0.42,
  /** Free floor along the side of a bed to get in (both sides for a double or bigger). */
  bedSide: 0.6,
  /** TV viewing distance as a multiple of the screen diagonal (the stand's width stands in for it). */
  tvDistMin: 1.5,
  tvDistMax: 2.5,
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
/** Pieces you use from the front or sides: short of that space, they don't count toward goals. */
const NEEDS_ROOM = new Set(['bed', 'wardrobe', 'dresser', 'desk'])
const STORAGE = new Set(['dresser', 'wardrobe', 'bookshelf', 'sideboard', 'console-table'])
/** Flow goals only count once the home has this many solid pieces. */
export const MIN_PIECES = 4
/** Unpacking stays Next's first suggestion while the home has fewer pieces than this (boxes aside). */
export const UNPACK_URGENT_PIECES = 8
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
  /** Placement id of the piece this stands on (table lamps). */
  host?: string
  room?: Room
  inside: boolean
  /** Inside a room and free of errors (overlaps, walls, door swings): only these count toward goals. */
  clean: boolean
}

const front = (p: P) => rot(v(0, 1), p.rotation)
/** A point in p's local frame (x to its side, +y out of its front). */
const toLocal = (p: P, q: Vec2) => rot(sub(q, p.center), -p.rotation)

export { polygonGap }

const clamp01 = (x: number) => Math.max(0, Math.min(1, x))
const cm = (m: number) => `${Math.round(m * 100)} cm`

/**
 * Fractions of the core stars needed for each of LEVELS (level 2 takes one star). Early levels come
 * quickly; Designer home takes nearly every core star, and Dream home all of them plus some bonus
 * stars (DREAM_BONUS_STARS) and a layout with no warnings.
 */
export const LEVEL_FRACTIONS = [0, 0, 0.35, 0.75, 0.92, 1]
/** Bonus stars Dream home also needs (or all the bonus stars on offer, when there are fewer). */
export const DREAM_BONUS_STARS = 3
/** Bonus or style goals Dream home also needs done, on top of every core goal. */
export const DREAM_EXTRA_GOALS = 2
/**
 * Share of the home's bonus goals Dream home also asks for (round 15: the sample reached Dream home
 * in about a dozen taps, so the game ended at minute three). It always leaves a couple of bonus goals
 * spare, so one piece that won't fit is never a dead end.
 */
export const DREAM_BONUS_SHARE = 0.85
export const DREAM_BONUS_SPARE = 2
/** How many bonus goals (of `count` on offer) Dream home asks for. */
export function dreamBonusGoals(count: number): number {
  return Math.max(0, Math.min(count - DREAM_BONUS_SPARE, Math.ceil(count * DREAM_BONUS_SHARE)))
}
/** The highest level a layout with warnings (a blocked clearance, say) can reach: Designer home. */
export const WARNING_LEVEL_CAP = 4
/** The highest level a layout with snags (errors, a tight walkway, a blocked door) can reach: Cozy. */
export const SNAG_LEVEL_CAP = 3
/** Goals about the layout working at all: while one is open the level stays capped, and Next points at it first. */
export const FIX_FIRST = ['no-overlap', 'in-walls', 'doors-clear', 'walkways', 'sofa-tv']
/** The level from which Next offers today's visitor (Cozy). */
export const VISIT_LEVEL = 3
const CAPPING = new Set(['no-overlap', 'in-walls', 'doors-clear', 'walkways'])

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

/** The furniture drawer tab each catalog piece lives on. */
export const TAB_OF: Record<string, string> = {
  sofa: 'living', armchair: 'living', 'coffee-table': 'living', 'side-table': 'living', ottoman: 'living', 'tv-stand': 'living', 'console-table': 'living',
  bed: 'bedroom', nightstand: 'bedroom', dresser: 'bedroom', wardrobe: 'bedroom',
  'dining-table': 'dining', 'dining-chair': 'dining', bench: 'dining', sideboard: 'dining',
  desk: 'work', 'office-chair': 'work', bookshelf: 'work',
  plant: 'decor', 'floor-lamp': 'decor', 'table-lamp': 'decor', rug: 'decor', box: 'decor',
}
/** Other pieces that finish the same goal, tried in order when the suggested one has no room. */
const STAND_INS: Record<string, string[]> = {
  dresser: ['wardrobe', 'bookshelf', 'sideboard', 'console-table'],
  wardrobe: ['dresser', 'bookshelf', 'sideboard', 'console-table'],
  bookshelf: ['sideboard', 'console-table', 'dresser', 'wardrobe'],
  sofa: ['armchair'],
  nightstand: ['side-table'],
  'dining-chair': ['bench'],
  'table-lamp': ['floor-lamp'],
}
const STAND_IN_GOALS: Record<string, string[]> = { storage: ['bookshelf', 'sideboard', 'console-table', 'wardrobe', 'dresser'], sit: STAND_INS.sofa }
/** Goals that make a living room cozy, suggested first when the home is a living space. */
const LIVING_GOAL = /^(sit|coffee:|rug|light|plant|nook|rooms-used|sofa-tv|tv-distance)/

/** Placements in the layout that are moving boxes still packed. */
export function packedBoxIds(doc: Doc, layout: Layout): Set<string> {
  const packed = new Set(doc.items.filter((i) => i.unpack?.length && !i.unpacked).map((i) => i.id))
  return new Set(layout.placements.filter((p) => packed.has(p.itemId)).map((p) => p.id))
}

/**
 * The goal to suggest next: the unfinished one closest to done that the player can act on now
 * (goals waiting on something else are skipped), favouring ones the furniture drawer finishes.
 * `fits` dry-runs placement: a goal whose piece has nowhere to go suggests a stand-in that fits
 * (a bookshelf when the bedroom has no room for a dresser), or gives way to the next goal, so the
 * chip never leads to a refusal.
 */
export function nextGoal(score: HomeScore, fits?: (category: string, near?: string) => boolean): Goal | undefined {
  const open = score.goals.filter((g) => !g.done && !g.locked)
  const urgent = open.find((g) => g.urgent && g.near)
  if (urgent) return urgent
  // Fix what's broken before buying more: errors, a tight walkway or a blocked door, then a TV the sofa can't see.
  for (const id of FIX_FIRST) {
    const g = open.find((x) => x.id === id && x.snag)
    if (g) return g
  }
  // A remix's twist is the day's headline: once the big pieces are in (somewhere to sleep and sit),
  // Next leads with it rather than replaying the original step for step. A twist done by moving a
  // piece points at that piece; one that needs a piece suggests it when it fits.
  const twist = open.find((g) => g.id.startsWith('twist:'))
  if (twist && !open.some((g) => !g.bonus && (g.id === 'sleep' || g.id === 'sit'))) {
    if (twist.item && twist.tab ? !fits || fits(twist.item) : !!twist.near) return twist
  }
  // Today's visitor waits until the home is Cozy (or nothing core is left), then leads: a piece it
  // needs is suggested when it fits, a piece to move is selected, otherwise it joins the pool.
  const visit = open.find((g) => g.visit)
  const visitReady = visitLeads(score)
  if (visit && visitReady && !open.some((g) => g.snag)) {
    if (visit.item && visit.tab ? !fits || fits(visit.item) : !!visit.near && !visit.tab) return visit
  }
  const live = visit && !visitReady ? open.filter((g) => g !== visit) : open
  const lived = score.livingFirst ? [...live.filter((g) => LIVING_GOAL.test(g.id)), ...live.filter((g) => !LIVING_GOAL.test(g.id))] : live
  // The stars for the next level are in but a core goal holds it (levelBlocker names it): core goals
  // come first, so Next never wanders off to bonus goals that can't lift the level.
  const blocked = !!score.level.next && score.level.toNext === 0
  const pool = blocked ? [...lived.filter((g) => !g.bonus), ...lived.filter((g) => g.bonus)] : lived
  if (blocked) {
    const core = pickFrom(pool.filter((g) => !g.bonus), fits)
    if (core) return core
    const held = pool.find((g) => !g.bonus && (!g.tab || !g.item))
    if (held) return held
  }
  return pickFrom(pool, fits)
}

/**
 * The goal to suggest from a ranked pool: score.goals is already ranked (closest to done, basics
 * first). A goal a piece of furniture finishes is the easiest next step, so the first of those
 * wins; then the first core goal.
 */
function pickFrom(pool: Goal[], fits?: (category: string, near?: string) => boolean): Goal | undefined {
  for (const g of pool) {
    if (!g.tab || !g.item) continue
    if (!fits || fits(g.item, g.near)) return g
    // A side table elsewhere in the room wouldn't light that side of the bed: no stand-in.
    if (g.id.startsWith('bedside-light:') && g.item === 'nightstand') continue
    const alt = [g.item, ...(STAND_IN_GOALS[g.id] ?? []), ...(STAND_INS[g.item] ?? [])].find((c) => c !== g.item && fits(c))
    if (alt) return { ...g, item: alt, tab: TAB_OF[alt] ?? g.tab, placeIn: undefined }
    // A core goal (somewhere to sleep) whose piece has nowhere to go: say what to move, rather
    // than skipping on to bonus goals and leaving the level out of reach.
    if (!g.bonus && g.makeRoom) return makeRoomGoal(g)
  }
  const rest = pool.filter((g) => !g.tab || !g.item)
  const pick = rest.find((g) => !g.bonus) ?? rest[0] ?? (fits ? undefined : pool[0])
  if (pick) return pick
  // Nothing on offer fits anywhere: point at the goals sheet (its hints) rather than a refusal.
  const g = pool.find((x) => !x.bonus) ?? pool[0]
  return g && { ...g, tab: undefined, item: undefined }
}

/**
 * Whether Next may lead with today's visitor: from Cozy on (or once no core goal is left), or
 * sooner when the visitor is one pick or one move away (a lamp beside the armchair Grandma has).
 */
export function visitLeads(score: HomeScore): boolean {
  const open = score.goals.filter((g) => !g.done && !g.locked)
  const visit = open.find((g) => g.visit)
  if (!visit) return false
  const oneAway = (!!(visit.item && visit.tab) || (!!visit.near && !visit.tab)) && (visit.progress ?? 0) >= 0.5
  return score.level.index >= VISIT_LEVEL || !open.some((g) => !g.bonus) || oneAway
}

/** "Make room for a bed": the goal turned into moving the piece in the way (selected by the chip). */
function makeRoomGoal(g: Goal): Goal {
  const what = g.item!.replace(/-/g, ' ')
  const a = /^[aeiou]/.test(what) ? 'an' : 'a'
  const m = g.makeRoom!
  return {
    ...g,
    title: `Make room for ${a} ${what}`,
    hint: `No room for ${a} ${what}. Move the ${m.name}${m.room ? ` out of the ${m.room}` : ''} to make space.`,
    tab: undefined,
    item: undefined,
    placeIn: undefined,
    near: m.id,
  }
}

/**
 * What stands between the stars and the next level when the stars are already there (toNext is
 * 0): a snag, an open core goal ("Finish “A place to eat”"), or the extras Dream home asks for.
 */
export function levelBlocker(score: HomeScore): string {
  const next = score.level.next ?? ''
  const open = score.goals.filter((g) => !g.done && !g.visit)
  const core = open.find((g) => !g.bonus)
  const gate = score.errors > 0 || open.some((g) => g.snag) ? 'Fix the snag' : core ? `Finish “${core.title}”` : 'Finish a bonus goal'
  if (score.level.toNext > 0) {
    const more = `${score.level.toNext} more ★`
    // Stars alone won't do it when a gate (a snag, an open core goal) holds the level: say both.
    const held = score.cap !== undefined && score.cap <= score.level.index
    return held ? `${more} and ${gate.charAt(0).toLowerCase()}${gate.slice(1)} to reach ${next}` : `${more} to ${next}`
  }
  return `${gate} to reach ${next}`
}

/** Where the best level is remembered: per design, so designs of one home can be compared. */
export const bestLevelKey = (doc: Doc) => `${doc.apartment.id}:${doc.activeLayoutId}`

/** The level to remember as the best reached: only a layout without errors banks a level. */
export function bankableLevel(score: HomeScore): number {
  return score.errors > 0 ? 0 : score.level.index
}

export function scoreHome(doc: Doc, layout: Layout, analysis: Analysis, visit?: VisitInput | null, perks?: Perks | null): HomeScore {
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
      // Wall decor takes no floor: like a rug, it isn't a solid piece and adds nothing to density.
      floor: !!item.floorLevel || isWallDecor(item.category),
      host: p.on,
      room,
      inside: !!room && polygonInsideRoom(body, room.polygon),
      clean: false,
    }
  })
  // A piece that is short of its own clearance (a bed with no room to get in) has a snag: it earns
  // nothing until it's moved, so a goal and a level-up never land on a piece the game flags.
  // (A coffee table inside the sofa's knee room is the point of a coffee table, so only a wall in
  // the way counts there; beds, wardrobes, dressers and desks need their space whatever blocks it.)
  const catOf = new Map(placed.map((x) => [x.id, x.category]))
  // (A dresser or wardrobe is only flagged short of MIN_DRAWER_FRONT, so 75–90 cm is no snag.)
  // A coffee table 35–45 cm from the sofa is within designers' range, though the clearance check
  // asks 45 cm: here it neither blocks the coffee goal nor counts as a warning holding the level.
  const byPid = new Map(placed.map((x) => [x.id, x]))
  const kneeRoomOk = (id: string) => {
    const m = /^clear:([^:]+):front:([^:]+)$/.exec(id)
    const seat = m && byPid.get(m[1])
    const table = m && byPid.get(m[2])
    return !!seat && !!table && SEATING.has(seat.category) && (table.category === 'coffee-table' || table.category === 'ottoman') && polygonGap(seat.body, table.body) >= RULES.coffeeMin - 0.005
  }
  const issues = analysis.issues.filter((i) => i.severity === 'error' || !kneeRoomOk(i.id))
  const clearanceSnag = (id: string) => issues.some((i) => i.kind === 'clearance' && i.severity !== 'info' && i.placementIds[0] === id)
  const cramped = new Set(
    issues
      .filter((i) => i.kind === 'clearance' && i.severity !== 'info' && (i.id.startsWith('clearwall:') || NEEDS_ROOM.has(catOf.get(i.placementIds[0]) ?? '')))
      .map((i) => i.placementIds[0]),
  )
  for (const x of placed) x.clean = x.inside && analysis.flagged.get(x.id) !== 'error'
  /** Pieces of these categories that count (inside a room, no errors, room to use them). */
  const of = (...cats: string[]) => placed.filter((x) => x.clean && !cramped.has(x.id) && cats.includes(x.category))
  /** Any pieces of these categories, for hints. */
  const ofAny = (...cats: string[]) => placed.filter((x) => cats.includes(x.category))
  const inRoom = (room: Room) => placed.filter((x) => x.clean && x.room?.id === room.id)
  const issuesOf = (kind: string) => issues.filter((i) => i.kind === kind && i.severity !== 'info')
  const goals: Goal[] = []
  const add = (g: Omit<Goal, 'progress'> & { progress?: number }) => goals.push({ ...g, progress: g.done ? 1 : g.progress })
  const rooms = doc.rooms.filter((r) => polygonArea(r.polygon) >= RULES.minRoomArea)
  const anything = placed.some((x) => !x.floor)
  // Packed moving boxes are clutter, not furniture: they don't count toward MIN_PIECES.
  const packedIds = packedBoxIds(doc, layout)
  const solidPieces = placed.filter((x) => !x.floor && !packedIds.has(x.id))
  const cleanShare = solidPieces.length ? solidPieces.filter((x) => x.clean).length / solidPieces.length : 0
  /** Flow goals grow with the furniture: clean share, scaled until the home has a few pieces. */
  const flowProgress = cleanShare * Math.min(1, solidPieces.length / MIN_PIECES)
  const furnished = solidPieces.length >= MIN_PIECES
  const fewHint = `Place at least ${MIN_PIECES} pieces of furniture first.`
  const bedrooms = rooms.filter((r) => BEDROOM.test(r.name) && !UTILITY_ROOM.test(r.name))
  const livable = rooms.filter((r) => !UTILITY_ROOM.test(r.name))
  // Goals that need a kind of room only show when the plan has one. With no rooms yet, show them all.
  // A studio takes the bed anywhere: a room named so, a single room with a bed in it already (a
  // scan that found one), or a single room that isn't named for something else. One room called
  // "Living room" is a living room: no bed goals.
  const named = (r: Room) => /studio|loft|bedsit/i.test(r.name)
  const single = livable.length === 1 ? livable[0] : undefined
  const studio =
    !bedrooms.length &&
    (livable.some(named) || (!!single && (ofAny('bed').length > 0 || !(LIVING.test(single.name) || DINING.test(single.name) || OFFICE.test(single.name)))))
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
    // A bed you already have that doesn't count yet is moved, not bought again.
    tab: ofAny('bed').length ? undefined : 'bedroom',
    item: ofAny('bed').length ? undefined : 'bed',
    near: ofAny('bed')[0]?.id,
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
    tab: ofAny(...SEATING).length ? undefined : 'living',
    item: ofAny(...SEATING).length ? undefined : 'sofa',
    near: ofAny(...SEATING)[0]?.id,
    done: seats.length > 0,
    progress: ofAny(...SEATING).length ? 0.5 : 0,
    stars: 1,
    category: 'basics',
  })
  const tables = of('dining-table')
  // A chair jammed against a wall or another piece (no room to pull it out) doesn't count.
  const chairs = of('dining-chair', 'bench').filter((c) => !clearanceSnag(c.id))
  const servedTable = tables.some((t) => chairs.some((c) => polygonGap(t.body, c.body) <= RULES.chairToTable))
  // Chairs at the table that don't count (no room to pull them out): the table is the problem, so
  // Next points at moving it off the wall rather than buying yet another chair.
  const stuckChairs = !servedTable && tables.length > 0 && ofAny('dining-chair', 'bench').some((c) => tables.some((t) => polygonGap(t.body, c.body) <= RULES.chairToTable))
  if (canEat) add({
    id: 'eat',
    title: 'A table to eat at',
    hint: !tables.length
      ? 'Add a dining table and a chair or two.'
      : stuckChairs
        ? 'The chairs have no room to pull out. Move the dining table off the wall.'
        : 'Pull a chair up to the dining table.',
    tab: stuckChairs ? undefined : 'dining',
    item: stuckChairs ? undefined : tables.length ? 'dining-chair' : 'dining-table',
    near: stuckChairs ? tables[0].id : undefined,
    done: servedTable,
    progress: tables.length ? 0.5 : chairs.length ? 0.25 : 0,
    stars: 2,
    category: 'basics',
  })
  // Dinner for two: two chairs (or a bench) pulled up to the same table. Later than the first
  // chair (a comfort goal), so the table works first; Dream home wants both seats.
  const seatsAt = (t: P) => chairs.filter((c) => polygonGap(t.body, c.body) <= RULES.chairToTable).reduce((a, c) => a + (c.category === 'bench' ? 2 : 1), 0)
  const mostSeats = tables.reduce((m, t) => Math.max(m, seatsAt(t)), 0)
  const seatTable = tables.slice().sort((a, b) => seatsAt(b) - seatsAt(a))[0]
  if (canEat) add({
    id: 'eat-two',
    title: 'Seats for two at the table',
    hint: !tables.length
      ? 'Once the dining table is in, pull up two chairs.'
      : mostSeats >= 2
        ? 'Dinner for two.'
        : stuckChairs
          ? 'Move the dining table off the wall so two chairs fit.'
          : mostSeats === 1
            ? 'Pull a second chair up to the table, so someone can join you.'
            : 'Pull two chairs up to the dining table.',
    tab: servedTable ? 'dining' : undefined,
    item: servedTable ? 'dining-chair' : undefined,
    near: seatTable?.id,
    done: mostSeats >= 2,
    progress: mostSeats >= 2 ? 1 : servedTable ? 0.5 : 0,
    stars: 1,
    category: 'comfort',
  })
  add({
    id: 'storage',
    title: 'A place for your things',
    hint: ofAny(...STORAGE).length ? 'Move your storage fully inside a room, with room to open it.' : 'Add a dresser, wardrobe, bookshelf or sideboard.',
    tab: ofAny(...STORAGE).length ? undefined : bedrooms.length ? 'bedroom' : 'work',
    item: ofAny(...STORAGE).length ? undefined : bedrooms.length ? 'dresser' : 'bookshelf',
    near: ofAny(...STORAGE)[0]?.id,
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
    snag: overlaps.length > 0,
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
    snag: walls.length > 0,
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
    snag: doors.length > 0,
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
    snag: paths.entrance && !!tight,
    progress: routes.length ? (routes.filter((x) => x.ok).length / routes.length) * Math.min(1, solidPieces.length / MIN_PIECES) : 0,
    stars: 2,
    category: 'flow',
  })
  for (const room of rooms) {
    if (UTILITY_ROOM.test(room.name)) continue
    const area = polygonArea(room.polygon)
    // A desk chair pulled up to its desk tucks under it at night: it doesn't crowd the room.
    const desksHere = inRoom(room).filter((x) => x.category === 'desk')
    const tucked = (x: (typeof placed)[number]) => x.category === 'office-chair' && desksHere.some((dk) => polygonGap(dk.body, x.body) <= RULES.chairToTable)
    const used = inRoom(room)
      .filter((x) => !x.floor && !tucked(x))
      .reduce((a, x) => a + polygonArea(x.body), 0)
    const f = used / area
    const pct = Math.round(f * 100)
    const max = bedrooms.includes(room) ? RULES.densityMaxBedroom : RULES.densityMax
    const aim = `aim for ${Math.round(RULES.densityMin * 100)}–${Math.round(max * 100)}%`
    add({
      id: `density:${room.id}`,
      title: `${room.name} feels just right`,
      hint:
        f >= RULES.densityMin && f <= max
          ? `Furniture covers ${pct}% of the floor. Just right.`
          : f < RULES.densityMin
          ? `Furniture covers ${pct}% of the floor; it still feels bare (${aim}).`
          : `Furniture covers ${pct}% of the floor; it feels crowded (${aim}). Take something out.`,
      done: f >= RULES.densityMin && f <= max,
      progress: f < RULES.densityMin ? f / RULES.densityMin : clamp01(1 - (f - max) / max),
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
      item: Number.isFinite(best) ? undefined : 'nightstand',
      near: Number.isFinite(best) ? nearby.slice().sort((a, b) => a.gap - b.gap)[0].n.id : bed.id,
      done: best <= RULES.nightstandGap,
      progress: Number.isFinite(best) ? 0.5 : 0,
      stars: 1,
      roomId: bed.room?.id,
      category: 'comfort',
      bonus: true,
    })
  }
  // A reading light on each side of a double bed (one side of a single): lights out without getting up.
  for (const bed of beds) {
    const stands = of('nightstand', 'side-table').map((n) => ({ n, l: toLocal(bed, n.center), gap: polygonGap(bed.body, n.body) }))
    const near = stands.filter((x) => x.l.y <= 0 && x.gap <= RULES.nightstandGap && Math.abs(x.l.x) >= bed.hw * 0.9)
    const twin = 2 * bed.hw < 1.2
    const nearest = near.slice().sort((a, b) => a.gap - b.gap)[0]
    // A side with no room for a real nightstand (against a wall, or another piece in the way) gets
    // no light goal: the goal never suggests a nightstand the game would then refuse. The same spots
    // findDrop tries (a real nightstand's size, 3 or 10 cm off the bed, by the pillows), and findDrop
    // itself has the last word (doors, walkways, the rest of its rules).
    const ns = catalogEntry('nightstand')
    const solid = placed.filter((x) => x.id !== bed.id && !x.floor && !x.host)
    const open = (sd: number) =>
      !!bed.room &&
      [0.03, 0.1].some((gap) => {
        const body = orientedRect(bed.center, ns.width / 2, ns.depth / 2, bed.rotation, v(sd * (bed.hw + gap + ns.width / 2), -bed.hd + ns.depth / 2))
        return polygonInsideRoom(body, bed.room!.polygon) && !solid.some((x) => convexOverlap(body, x.body))
      })
    let canDrop: boolean | undefined
    const dropOk = () => (canDrop ??= findDrop(doc, layout, itemFromCatalog(ns, '__bedside-probe'), { nextTo: bed.id }).ok)
    const sides = twin ? [nearest ? Math.sign(nearest.l.x) || 1 : open(1) || !open(-1) ? 1 : -1] : [-1, 1]
    for (const sd of sides) {
      const stand = near.filter((x) => Math.sign(x.l.x) === sd).sort((a, b) => a.gap - b.gap)[0]?.n
      if (!stand && ((!twin && !open(sd)) || !dropOk())) continue
      const lit = of('table-lamp', 'floor-lamp').some((l) => {
        const q = toLocal(bed, l.center)
        return q.x * sd >= bed.hw * 0.9 && q.y <= 0 && polygonGap(bed.body, l.body) <= RULES.bedsideLamp
      })
      // Sides are named as seen from the foot of the bed, facing the headboard: local +x is the right.
      const name = twin ? '' : sd > 0 ? 'right' : 'left'
      add({
        id: `bedside-light:${bed.id}:${sd}`,
        title: twin ? 'A reading light by the bed' : `Reading light, ${name} side of the bed`,
        hint: lit
          ? 'Lights out without getting up.'
          : stand
            ? `Set a table lamp on the ${name ? `${name} ` : ''}nightstand.`
            : `Put a nightstand ${name ? `on the ${name} side` : 'by the bed'}, then a lamp on it.`,
        tab: lit ? undefined : stand ? 'decor' : 'bedroom',
        item: lit ? undefined : stand ? 'table-lamp' : 'nightstand',
        near: stand ? stand.id : bed.id,
        done: lit,
        progress: lit ? 1 : stand ? 0.5 : 0,
        stars: 1,
        roomId: bed.room?.id,
        category: 'comfort',
        bonus: true,
      })
    }
  }
  const sofas = of('sofa')
  for (const sofa of sofas.slice(0, 2)) {
    const ahead = of('coffee-table', 'ottoman')
      .map((t) => ({ t, l: toLocal(sofa, t.center), gap: polygonGap(sofa.body, t.body), blocks: issues.some((i) => i.id === `clear:${sofa.id}:front:${t.id}` && i.severity !== 'info') }))
      .filter((x) => x.l.y > sofa.hd && Math.abs(x.l.x) < sofa.hw)
      .sort((a, b) => a.gap - b.gap)[0]
    const gap = ahead?.gap
    const tooClose = gap !== undefined && (ahead!.blocks || gap < RULES.coffeeMin - 0.005)
    // A coffee table already in the home (just not in front of this sofa) is moved, never bought again.
    const spare = gap === undefined ? ofAny('coffee-table').sort((a, b) => polygonGap(sofa.body, a.body) - polygonGap(sofa.body, b.body))[0] : undefined
    const range = `${cm(RULES.coffeeMin)}–${cm(RULES.coffeeMax)}`
    add({
      id: `coffee:${sofa.id}`,
      title: 'Coffee table within reach',
      hint:
        gap === undefined
          ? spare
            ? `Move the coffee table in front of the sofa, ${range} from the seat.`
            : `Put a coffee table in front of the sofa, ${range} from the seat.`
          : tooClose
            ? `It's ${cm(gap)} from the sofa; give your knees room (${range}).`
            : `It's ${cm(gap)} from the sofa; pull it closer (${range}).`,
      tab: gap === undefined && !spare ? 'living' : undefined,
      item: gap === undefined && !spare ? 'coffee-table' : undefined,
      // Buying: the sofa it goes in front of. Moving: the table to move.
      near: gap !== undefined ? ahead!.t.id : spare ? spare.id : sofa.id,
      done: gap !== undefined && !tooClose && gap <= RULES.coffeeMax + 0.005,
      progress: gap === undefined ? 0 : 0.6,
      stars: 1,
      roomId: sofa.room?.id,
      category: 'comfort',
      bonus: true,
    })
  }
  // Soft light: a floor lamp near a seat, in a living space (not a bathroom or hall).
  const livingSeats = of(...SEATING).filter((x) => x.room && !UTILITY_ROOM.test(x.room.name))
  const lamps = of('floor-lamp', 'table-lamp')
  const lampGap = livingSeats.flatMap((st) => lamps.filter((l) => l.room?.id === st.room?.id).map((l) => polygonGap(st.body, l.body)))
  const nearestLamp = lampGap.length ? Math.min(...lampGap) : Infinity
  add({
    id: 'light',
    title: 'Soft light by the seats',
    hint: !livingSeats.length
      ? 'Add a sofa or armchair, then a floor lamp beside it.'
      : nearestLamp <= RULES.lampReach
        ? 'A warm pool of light by the sofa.'
        : Number.isFinite(nearestLamp)
        ? `Your lamp is ${cm(nearestLamp)} from the seats; bring it within ${cm(RULES.lampReach)}.`
        : `Add a floor lamp or table lamp near the seats in the ${livingSeats[0].room!.name}.`,
    tab: !livingSeats.length ? (ofAny(...SEATING).length ? undefined : 'living') : Number.isFinite(nearestLamp) ? undefined : 'decor',
    item: !livingSeats.length ? (ofAny(...SEATING).length ? undefined : 'armchair') : Number.isFinite(nearestLamp) ? undefined : 'floor-lamp',
    near: livingSeats.length ? livingSeats[0].id : undefined,
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
    tab: sleepBeds.length || ofAny('bed').length ? undefined : 'bedroom',
    item: sleepBeds.length || ofAny('bed').length ? undefined : 'bed',
    near: sleepBeds[0]?.id ?? ofAny('bed')[0]?.id,
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
  // Room to get into bed: 60 cm of free floor along the side (both sides for a double or bigger),
  // below the nightstands.
  const solids = placed.filter((x) => !x.floor && !x.host)
  // Checked on every bed in the right room, cramped or not: this goal is the one that says why.
  const accessBeds = placed.filter((x) => x.clean && x.category === 'bed' && (!bedrooms.length || (!!x.room && bedrooms.includes(x.room))))
  for (const b of accessBeds) {
    const along = Math.max(0.3, 2 * b.hd - 0.45)
    const sideFree = (sd: number) => {
      const strip = orientedRect(b.center, RULES.bedSide / 2, along / 2, b.rotation, v(sd * (b.hw + RULES.bedSide / 2), b.hd - along / 2))
      return !!b.room && polygonInsideRoom(strip, b.room.polygon, 0.01) && !solids.some((o) => o.id !== b.id && convexOverlap(strip, o.body, 0.005))
    }
    const free = [-1, 1].filter(sideFree).length
    const twin = b.hw * 2 < 1.2
    const need = twin ? 1 : 2
    add({
      id: `bed-access:${b.id}`,
      title: 'Room to get into bed',
      hint:
        free >= need
          ? 'Plenty of room to get in.'
          : twin
            ? `Keep ${cm(RULES.bedSide)} free along one side of the bed.`
            : `Keep ${cm(RULES.bedSide)} free along both sides of the bed${free ? '; one side is blocked' : ''}.`,
      done: free >= need,
      progress: free / need,
      stars: 1,
      roomId: b.room?.id,
      category: 'comfort',
      bonus: true,
    })
  }
  // A spot to work: a desk with a chair pulled up to it.
  const desksAll = of('desk')
  const worked = desksAll.some((dk) => of('office-chair', 'dining-chair').some((c) => polygonGap(dk.body, c.body) <= RULES.chairToTable && toLocal(dk, c.center).y > 0))
  add({
    id: 'work',
    title: 'A spot to work',
    hint: !desksAll.length ? 'Add a desk and a chair from the Work tab.' : 'Pull a chair up to the front of the desk.',
    tab: !desksAll.length && ofAny('desk').length ? undefined : 'work',
    item: desksAll.length ? 'office-chair' : ofAny('desk').length ? undefined : 'desk',
    near: desksAll[0]?.id ?? ofAny('desk')[0]?.id,
    done: worked,
    progress: desksAll.length ? 0.5 : 0,
    stars: 1,
    category: 'comfort',
  })
  // A lamp on (or right by) each desk.
  for (const desk of desksAll) {
    const lit = of('table-lamp').some((l) => l.host === desk.id) || of('floor-lamp', 'table-lamp').some((l) => !l.host && polygonGap(desk.body, l.body) <= NEXT_TO_REACH)
    add({
      id: `desk-light:${desk.id}`,
      title: 'A lamp on the desk',
      hint: lit ? 'Good light to work by.' : 'Set a table lamp on the desk, at the back corner.',
      tab: lit ? undefined : 'decor',
      item: lit ? undefined : 'table-lamp',
      near: desk.id,
      done: lit,
      progress: lit ? 1 : 0,
      stars: 1,
      roomId: desk.room?.id,
      category: 'comfort',
      bonus: true,
    })
  }
  // A pendant hung over each dining table: a warm pool of light on the meal.
  for (const table of of('dining-table')) {
    const lit = of('pendant').some((l) => l.host === table.id)
    add({
      id: `dining-light:${table.id}`,
      title: 'A pendant over the table',
      hint: lit ? 'Dinner in a warm pool of light.' : 'Hang a pendant light over the dining table.',
      tab: lit ? undefined : 'decor',
      item: lit ? undefined : 'pendant',
      near: table.id,
      done: lit,
      progress: lit ? 1 : 0,
      stars: 1,
      roomId: table.room?.id,
      category: 'comfort',
      bonus: true,
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
      bonus: true,
    })
  }

  // ---- Style --------------------------------------------------------------
  add({
    id: 'plant',
    title: 'Bring in some greenery',
    hint: ofAny('plant').length ? 'Move your plant fully inside a room.' : 'Add a plant. Corners and windows love them.',
    tab: ofAny('plant').length ? undefined : 'decor',
    item: ofAny('plant').length ? undefined : 'plant',
    near: ofAny('plant')[0]?.id,
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
      item: ofAny('rug').length ? undefined : 'rug',
      near: ofAny('rug')[0]?.id,
      done: rugs.some((r) => anchors.some((a) => a.room?.id === r.room?.id && convexOverlap(r.body, a.body, 0.01))),
      progress: ofAny('rug').length ? 0.5 : 0,
      stars: 1,
      category: 'style',
      bonus: true,
    })
  }
  const armchairs = of('armchair')
  const nookFor = ofAny('armchair').length ? nookTarget(doc, layout) : undefined
  // Only a real lamp makes it a reading corner; a side table or bookshelf beside the chair is a start.
  // A seat facing the armchair across a conversation corner shares its light: a lamp beside either counts.
  const nookSeats = [...armchairs, ...of('sofa').filter((s) => armchairs.some((a) => talkPair(a, s).ok))]
  const nook = nookSeats.some((a) => of('floor-lamp', 'table-lamp').some((x) => polygonGap(a.body, x.body) <= NEXT_TO_REACH))
  // A side table already beside the reading chair: the lamp goes on it.
  const nookTable = armchairs.flatMap((a) => of('side-table').filter((x) => polygonGap(a.body, x.body) <= NEXT_TO_REACH))[0]
  const nookStart = armchairs.some((a) => of('side-table', 'bookshelf').some((x) => polygonGap(a.body, x.body) <= NEXT_TO_REACH))
  add({
    id: 'nook',
    title: 'A cozy reading corner',
    hint: !ofAny('armchair').length
      ? 'Add an armchair with a lamp beside it.'
      : nookStart
        ? 'Nice start. Now stand a lamp next to the armchair to read by.'
        : 'Stand a floor lamp or table lamp next to the armchair.',
    tab: ofAny('armchair').length ? 'decor' : 'living',
    item: ofAny('armchair').length ? (nookTable ? 'table-lamp' : 'floor-lamp') : 'armchair',
    near: nookTable ? nookTable.id : nookFor,
    done: nook,
    progress: nookStart ? 0.7 : ofAny('armchair').length ? 0.5 : 0,
    stars: 2,
    category: 'style',
  })
  const tvs = of('tv-stand')
  if (sofas.length && tvs.length) {
    let bestDeg = 180
    let pair: [P, P] = [sofas[0], tvs[0]]
    for (const s of sofas)
      for (const t of tvs) {
        const c = Math.max(-1, Math.min(1, dot(front(s), norm(sub(t.center, s.center)))))
        const deg = (Math.acos(c) * 180) / Math.PI
        if (deg < bestDeg) {
          bestDeg = deg
          pair = [s, t]
        }
      }
    // Nothing solid (a desk, a chair) stands between the sofa and the screen.
    const [vs, vt] = pair
    const reach = toLocal(vs, vt.center).y - vt.hd - vs.hd
    const view = reach > 0.05 ? orientedRect(vs.center, vs.hw - 0.05, reach / 2, vs.rotation, v(0, vs.hd + reach / 2)) : null
    const blocker = view && bestDeg <= RULES.tvAngleDeg ? placed.find((x) => !x.floor && x.id !== vs.id && x.id !== vt.id && !VIEW_OK.has(x.category) && convexOverlap(view, x.body, 0.005)) : undefined
    const faces = bestDeg <= RULES.tvAngleDeg
    add({
      id: 'sofa-tv',
      title: 'Sofa faces the TV',
      hint: blocker
        ? `The ${blocker.name.toLowerCase()} is in the way between the sofa and the TV; move it aside.`
        : faces
          ? 'Movie night is ready.'
          : `The sofa is turned ${Math.round(bestDeg)}° away from the TV; rotate it to face the screen.`,
      done: faces && !blocker,
      snag: !faces || !!blocker,
      progress: clamp01(1 - bestDeg / 180) * (blocker ? 0.8 : 1),
      stars: 1,
      category: 'style',
      bonus: true,
    })
  }

  if (sofas.length && of('rug').length) {
    // The rug spans the sofa's width and at least its front legs stand on it.
    const fits = sofas.some((s) =>
      of('rug').some((r) => {
        const xs = r.body.map((q) => toLocal(s, q).x)
        const wide = Math.max(...xs) - Math.min(...xs) >= 2 * s.hw - 0.02
        const legs = [-1, 1].every((sd) => pointInPolygon(add2(s.center, rot(v(sd * (s.hw - 0.08), s.hd - 0.08), s.rotation)), r.body))
        return wide && legs
      }),
    )
    add({
      id: 'rug-size',
      title: 'A rug big enough',
      hint: fits ? 'The rug holds the seating together.' : 'Pick a rug wider than the sofa, with at least the front legs on it.',
      done: fits,
      progress: fits ? 1 : 0.5,
      stars: 1,
      category: 'style',
      bonus: true,
    })
  }
  if (sofas.length && tvs.length) {
    // Screen diagonal ≈ the stand's width; sit 1.5–2.5 diagonals away.
    const fit = sofas.flatMap((s) =>
      tvs.map((t) => {
        const d = dist(s.center, t.center)
        const diag = 2 * t.hw
        return { d, lo: diag * RULES.tvDistMin, hi: diag * RULES.tvDistMax }
      }),
    )
    const best = fit.reduce((m, x) => {
      const off = (y: typeof x) => Math.max(0, y.lo - y.d, y.d - y.hi)
      return off(x) < off(m) ? x : m
    }, fit[0])
    const ok = best.d >= best.lo - 0.005 && best.d <= best.hi + 0.005
    add({
      id: 'tv-distance',
      title: 'Comfortable TV distance',
      hint: ok
        ? `${best.d.toFixed(1)} m from the screen. Just right.`
        : `The sofa is ${best.d.toFixed(1)} m from the TV; for that screen sit ${best.lo.toFixed(1)}–${best.hi.toFixed(1)} m away.`,
      done: ok,
      progress: ok ? 1 : clamp01(1 - Math.max(best.lo - best.d, best.d - best.hi) / best.lo),
      stars: 1,
      category: 'style',
      bonus: true,
    })
  }

  // Your palette: three or more pieces painted away from their catalog colour, in a few colour
  // families (or neighbours plus one accent). A bonus that names the palette it finds.
  const itemOf = new Map(doc.items.map((i) => [i.id, i]))
  const itemAt = new Map(layout.placements.map((p) => [p.id, itemOf.get(p.itemId)]))
  const paintable = placed.filter((x) => x.inside && !packedIds.has(x.id) && x.category !== 'box')
  const painted = paintable.flatMap((x) => {
    const it = itemAt.get(x.id)
    const def = catalogEntry(x.category)
    return it && def.category === x.category && sameColor(it.color, def.color) === false ? [{ x, color: it.color }] : []
  })
  if (furnished || painted.length) {
    const pal = paletteOf(painted.map((q) => q.color))
    const odd = pal.odd ? painted.find((q) => hueFamily(q.color) === pal.odd)?.x : undefined
    const plain = paintable.filter((x) => !painted.some((q) => q.x === x)).sort((a, b) => polygonArea(b.body) - polygonArea(a.body))[0]
    const ok = painted.length >= PALETTE_PIECES && pal.ok
    add({
      id: 'palette',
      title: painted.length >= 2 ? `Your palette: ${pal.name}` : 'Your palette',
      hint: ok
        ? `A palette all your own: ${pal.name}.`
        : painted.length < PALETTE_PIECES
          ? `${painted.length} of ${PALETTE_PIECES} pieces painted. Tap a piece and pick a swatch.`
          : `Too many colours at once. Keep to ${PALETTE_FAMILIES} colour families (or neighbours plus one accent): repaint the ${odd?.name.toLowerCase() ?? 'odd one out'}.`,
      near: ok ? undefined : painted.length < PALETTE_PIECES ? plain?.id : odd?.id,
      done: ok,
      progress: ok ? 1 : (Math.min(painted.length, PALETTE_PIECES) / PALETTE_PIECES) * (pal.ok ? 1 : 0.8),
      stars: 1,
      category: 'style',
      bonus: true,
    })
  }

  // ---- Wall decor (bonus only: nothing on a wall is ever needed for a level) ----
  // Art or a mirror on a wall of the living room or a bedroom, once there's a sofa or bed to hang it over.
  const decorRooms = rooms.filter((r) => !UTILITY_ROOM.test(r.name) && (LIVING.test(r.name) || BEDROOM.test(r.name)))
  const anchored = decorRooms.filter((r) => of('sofa', 'bed').some((x) => x.room?.id === r.id))
  // Round 15: one per room with a sofa or bed (the living room and the bedroom each want their own),
  // so a home has a few more small, cozy wins on the way to Dream home.
  anchored.forEach((r, i) => {
    const hung = i === 0 ? ofAny('wall-art', 'mirror') : []
    const ok = i === 0
      ? of('wall-art', 'mirror').some((x) => decorRooms.some((d) => d.id === x.room?.id) && (anchored.length === 1 || x.room?.id === r.id))
      : of('wall-art', 'mirror').some((x) => x.room?.id === r.id)
    const where = r.name.toLowerCase()
    add({
      id: i === 0 ? 'wall-art' : `wall-art:${r.id}`,
      title: i === 0 ? 'Something on the walls' : `Art in the ${where}`,
      hint: ok ? 'Your walls have some personality.' : hung.length ? 'Hang your art in the living room or a bedroom.' : i === 0 ? 'Hang some art or a mirror. Over the sofa or bed is a classic.' : `Hang some art or a mirror in the ${where} too.`,
      tab: ok || hung.length ? undefined : 'wall',
      item: ok || hung.length ? undefined : 'wall-art',
      placeIn: ok || hung.length ? undefined : r.id,
      near: ok ? undefined : hung[0]?.id,
      done: ok,
      stars: 1,
      category: 'style',
      bonus: true,
    })
  })
  // Curtains on a bedroom window soften the light: a cozy little extra. Round 15: the living room's
  // window (with a sofa there) asks for its own pair too.
  const curtainRooms = bedrooms.filter((r) => r.openings.some((o) => o.kind === 'window') && of('bed').some((x) => x.room?.id === r.id))
  if (curtainRooms.length) {
    const hung = ofAny('curtains')
    const ok = of('curtains').some((x) => curtainRooms.some((r) => r.id === x.room?.id))
    add({
      id: 'curtains',
      title: 'Curtains in the bedroom',
      hint: ok ? 'Soft light for slow mornings.' : hung.length ? 'Hang the curtains on a bedroom window.' : 'Hang curtains on the bedroom window for softer light.',
      tab: ok || hung.length ? undefined : 'wall',
      item: ok || hung.length ? undefined : 'curtains',
      placeIn: ok || hung.length ? undefined : curtainRooms[0].id,
      near: ok ? undefined : hung[0]?.id,
      done: ok,
      stars: 1,
      category: 'comfort',
      bonus: true,
    })
  }
  for (const r of rooms.filter((r) => LIVING.test(r.name) && r.openings.some((o) => o.kind === 'window') && of('sofa').some((x) => x.room?.id === r.id)).slice(0, 1)) {
    const ok = of('curtains').some((x) => x.room?.id === r.id)
    add({
      id: `curtains:${r.id}`,
      title: `Curtains in the ${r.name.toLowerCase()}`,
      hint: ok ? 'The evening light goes golden.' : `Hang curtains on the ${r.name.toLowerCase()} window: evenings feel snug.`,
      tab: ok ? undefined : 'wall',
      item: ok ? undefined : 'curtains',
      placeIn: ok ? undefined : r.id,
      done: ok,
      stars: 1,
      category: 'comfort',
      bonus: true,
    })
  }

  // A shelf of books where you sit (round 15): one more small, personal touch on the way to Dream home.
  const shelfRoom = decorRooms.find((r) => of('sofa', 'armchair').some((x) => x.room?.id === r.id))
  if (shelfRoom) {
    const ok = of('wall-shelf', 'bookshelf').some((x) => decorRooms.some((r) => r.id === x.room?.id))
    add({
      id: 'books',
      title: 'A shelf for your books',
      hint: ok ? 'Your favourite books, close at hand.' : `Put up a shelf in the ${shelfRoom.name.toLowerCase()} for books and little things.`,
      tab: ok ? undefined : 'wall',
      item: ok ? undefined : 'wall-shelf',
      placeIn: ok ? undefined : shelfRoom.id,
      done: ok,
      stars: 1,
      category: 'style',
      bonus: true,
    })
  }

  // Unpack your boxes: a bonus, and the cheapest first win on move-in day (ranked first by Next
  // while the home has fewer than UNPACK_URGENT_PIECES pieces besides the boxes).
  const boxes = doc.items.filter((i) => i.unpack?.length)
  if (boxes.length)
    add({
      id: 'unpack',
      title: 'Unpack your boxes',
      hint: 'Tap a moving box, then Unpack.',
      category: 'basics',
      bonus: true,
      stars: boxes.length,
      done: boxes.every((b) => b.unpacked),
      progress: boxes.filter((b) => b.unpacked).length / boxes.length,
      near: placed.find((x) => packedIds.has(x.id))?.id,
      // Stays first while any box is still packed and the home is still sparse, so the second box
      // gets unpacked too (not only the first).
      urgent: solidPieces.length < UNPACK_URGENT_PIECES && packedIds.size > 0,
    })

  // ---- Remix twist ------------------------------------------------------
  // A remix's twist is a core style goal: while it's open the design stays below Dream home.
  const twist = layout.remix ? twistGoal(doc, layout, placed, of, inRoom, rooms, solidPieces) : null
  if (twist) add(twist)

  // ---- Visitor of the day ---------------------------------------------------
  const vdef = visit ? VISIT_BY_ID[visit.id] : undefined
  if (vdef) {
    const counted = placed.filter((x) => x.clean && !cramped.has(x.id) && !packedIds.has(x.id))
    const byId = new Map(goals.map((g) => [g.id, g.done]))
    const c = vdef.check({ pieces: counted, all: placed, goal: (id) => byId.get(id) })
    const done = c.done || !!visit!.done
    add({
      id: `visit:${vdef.id}`,
      title: vdef.title,
      hint: done ? `${vdef.who} loved it!` : c.hint,
      tab: done ? undefined : c.tab,
      item: done ? undefined : c.item,
      near: done ? undefined : c.near,
      done,
      progress: c.progress,
      stars: VISIT_STARS,
      category: 'comfort',
      bonus: true,
      visit: vdef.id,
    })
  }

  // ---- Perks ----------------------------------------------------------------
  // The drag lesson nailed (a piece dragged onto its glowing spot): a star, once, for this home.
  if (perks?.drag) {
    add({
      id: DRAG_PERK.id,
      title: DRAG_PERK.title,
      hint: 'You dragged a piece onto its glowing spot. Drag any piece to try a new spot.',
      done: true,
      progress: 1,
      stars: DRAG_PERK.stars,
      category: 'style',
      bonus: true,
      perk: 'drag',
    })
  }

  // What to move when a core goal's piece has nowhere to go: the biggest piece in the room it
  // belongs in (the bedroom for a bed), pieces that could live in another room first.
  // With no room to aim at (a desk can go anywhere), never ask to move a bedroom-only piece (the
  // bed) out of its bedroom: a spare piece (a second armchair) goes first, then a single movable
  // living piece (the dining table), the big anchors and bedroom pieces last.
  const kinds = new Map<string, number>()
  for (const x of solidPieces) kinds.set(x.category, (kinds.get(x.category) ?? 0) + 1)
  const moveCost = (x: P, room: unknown) =>
    room
      ? Number(BEDROOM_ONLY.has(x.category))
      : BEDROOM_ONLY.has(x.category)
        ? 3
        : ANCHORS.has(x.category) && x.category !== 'dining-table'
          ? 2
          : (kinds.get(x.category) ?? 0) > 1
            ? 0
            : 1
  for (const g of goals) {
    if (g.bonus || !g.item || !g.tab) continue
    const room = g.placeIn ? rooms.find((r) => r.id === g.placeIn) : BEDROOM_ONLY.has(g.item) ? bedrooms[0] : undefined
    const big = solidPieces
      .filter((x) => !x.host && x.category !== g.item && (!room || x.room?.id === room.id))
      .sort((a, b) => moveCost(a, room) - moveCost(b, room) || polygonArea(b.body) - polygonArea(a.body))[0]
    if (big) g.makeRoom = { id: big.id, name: big.name.toLowerCase(), room: big.room?.name }
  }

  // ---- Totals -------------------------------------------------------------
  const stars = goals.reduce((a, g) => a + (g.done ? g.stars : 0), 0)
  const maxStars = goals.reduce((a, g) => a + g.stars, 0)
  const coreMax = goals.reduce((a, g) => a + (g.bonus ? 0 : g.stars), 0)
  const thresholds = levelThresholds(coreMax)
  const errors = analysis.issues.filter((i) => i.severity === 'error').length
  // A layout with snags can't be more than Cozy, however many stars it has.
  const capped = errors > 0 || goals.some((g) => g.snag && CAPPING.has(g.id))
  const warnings = issues.filter((i) => i.severity === 'warning').length
  // The visitor's stars (and a perk's) count toward the level, but never toward what a level asks for.
  const bonusMax = goals.reduce((a, g) => a + (g.bonus && !g.visit && !g.perk ? g.stars : 0), 0)
  const bonusStars = goals.reduce((a, g) => a + (g.bonus && !g.visit && !g.perk && g.done ? g.stars : 0), 0)
  const uncapped = levelAt(stars, thresholds)
  // Stars alone race to the top (bonus goals add stars too), so the top levels also need goals done:
  // Designer home every core flow goal, Dream home every core goal (basics, comfort and style) as
  // well, plus DREAM_EXTRA_GOALS bonus or style goals.
  const coreOpen = (...cats: GoalCategory[]) => goals.some((g) => !g.bonus && !g.done && cats.includes(g.category))
  const extras = goals.filter((g) => g.done && !g.visit && !g.perk && (g.bonus || g.category === 'style')).length
  // Only bonus goals a pick can finish (or already done) count toward what Dream home asks for: a
  // goal done by moving a piece, or by colour, is a nice extra but never a gate.
  const bonusGoals = goals.filter((g) => g.bonus && !g.visit && !g.perk && (g.done || (!!g.item && !!g.tab)))
  const bonusDone = bonusGoals.filter((g) => g.done).length
  const bonusNeed = dreamBonusGoals(bonusGoals.length)
  // Snags hold the level at Cozy (so does an open flow goal, or more than one open basics or comfort goal); warnings on screen, an open snag goal
  // (a blocked view of the TV), an open basics, comfort or style goal or too few extras hold it at Designer home.
  // Round 15: Designer home also asks for half of Dream home's bonus goals, so the climb has steps.
  const cap = capped || coreOpen('flow') || goals.filter((g) => !g.bonus && !g.done && (g.category === 'basics' || g.category === 'comfort')).length > 1 || bonusDone < Math.ceil(bonusNeed / 2)
    ? SNAG_LEVEL_CAP
    : warnings > 0 || goals.some((g) => g.snag) || bonusStars < Math.min(DREAM_BONUS_STARS, bonusMax) || coreOpen('basics', 'comfort', 'style') || extras < DREAM_EXTRA_GOALS || bonusDone < bonusNeed
      ? WARNING_LEVEL_CAP
      : LEVELS.length - 1
  const level = uncapped.index > cap ? levelAt(stars, thresholds, cap) : uncapped

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

  const livingFirst = !!rooms.length && !canSleep && livable.some((r) => LIVING.test(r.name) && !named(r))
  return { stars, maxStars, coreMax, thresholds, level, goals, vibe, errors, headline, livingFirst, cap }
}

// ---- Remix: twists -------------------------------------------------------

/** The big pieces a remix keeps: everything else goes back to the drawer to be placed again. */
export const ANCHORS = new Set(['bed', 'sofa', 'tv-stand', 'wardrobe', 'dining-table'])

/** The twist challenges, in the order the picker offers them. */
export const TWISTS: Record<TwistId, { name: string; title: string; hint: string }> = {
  'talk-corner': { name: 'Talk corner', title: 'Talk corner', hint: 'Turn the sofa and an armchair to face each other, under 3 m apart.' },
  'open-floor': { name: 'Open floor', title: 'Open floor', hint: 'Keep the living room’s middle clear (4+ pieces in the room).' },
  'window-seat': { name: 'Window seat', title: 'Sofa faces the window', hint: 'Turn the sofa to look out of a window in its room.' },
  'fresh-spots': { name: 'Fresh start', title: 'Everything in a new spot', hint: 'Move each big piece at least 80 cm from its old spot.' },
  'bed-swap': { name: 'Bed swap', title: 'Bed on a different wall', hint: 'Put the bed’s head against a different wall.' },
  'less-is-more': { name: 'Less is more', title: 'Dream home in 10 pieces', hint: 'Reach Dream home with 10 pieces of furniture or fewer.' },
}
/** Two seats this close (centre to centre, m) and facing each other make a conversation corner. */
export const TALK_DIST = 3.0
export const TWIST_IDS = Object.keys(TWISTS) as TwistId[]

/**
 * Twists the plan can take: a window-seat needs a window in a room with a sofa (or a living room),
 * a talk corner an armchair that fits, a bed swap a bedroom.
 */
export function applicableTwists(doc: Doc, fits?: (category: string) => boolean): TwistId[] {
  const layout = doc.layouts.find((l) => l.id === doc.activeLayoutId) ?? doc.layouts[0]
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const sofaIn = (r: Room) => (layout?.placements ?? []).some((p) => items.get(p.itemId)?.category === 'sofa' && pointInPolygon(v(p.x, p.y), r.polygon))
  return TWIST_IDS.filter((t) => {
    if (t === 'window-seat') return doc.rooms.some((r) => r.openings.some((o) => o.kind === 'window') && (sofaIn(r) || LIVING.test(r.name)))
    if (t === 'talk-corner') return talkCornerFits(doc, layout) ?? (!fits || fits('armchair'))
    if (t === 'bed-swap') return doc.rooms.some((r) => BEDROOM.test(r.name) && !UTILITY_ROOM.test(r.name))
    return true
  }).filter((t) => !(layout && ALREADY_MET_CHECK.has(t) && twistMetBy(doc, layout, t)))
}

/**
 * Whether a remix of `layout` has room for a conversation corner: an armchair facing a sofa
 * (turned toward it, under TALK_DIST away, clear of the TV's sightline) on clear floor among the
 * big pieces a remix keeps (a lamp beside either seat then makes the reading corner). Null when there's no sofa to try.
 */
function talkCornerFits(doc: Doc, layout: Layout | undefined): boolean | null {
  const spot = talkCornerSpot(doc, layout)
  return spot === undefined ? null : !!spot
}

/** The spot talkCornerFits found for the armchair (null: none; undefined: no sofa to try). */
export function talkCornerSpot(doc: Doc, layout: Layout | undefined): { x: number; y: number; rotation: number } | null | undefined {
  if (!layout) return undefined
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const kept = layout.placements.flatMap((p) => {
    const it = items.get(p.itemId)
    return it && ANCHORS.has(it.category) ? [{ p, it, body: orientedRect(v(p.x, p.y), it.width / 2, it.depth / 2, p.rotation) }] : []
  })
  const sofas = kept.filter((k) => k.it.category === 'sofa')
  if (!sofas.length) return undefined
  const chair = catalogEntry('armchair')
  const tv = kept.some((k) => k.it.category === 'tv-stand')
  for (const s of sofas) {
    const sc = v(s.p.x, s.p.y)
    // With a TV, the floor in front of the sofa is the view: the chair goes to one side, angled in.
    const view = tv ? orientedRect(sc, s.it.width / 2, 2, s.p.rotation, v(0, s.it.depth / 2 + 2)) : null
    const side0 = s.it.width / 2 + chair.width / 2 + 0.05
    const offs = tv ? [side0, side0 + 0.3] : [0, 0.7, side0]
    for (const d of [1.4, 1.8, 2.2, 2.6])
      for (const o of offs)
        for (const side of o ? [1, -1] : [1]) {
          const c = add2(sc, rot(v(side * o, d), s.p.rotation))
          if (dist(c, sc) > TALK_DIST - 0.1) continue
          const dir = sub(sc, c)
          const at = Math.atan2(-dir.x, dir.y)
          const straight = s.p.rotation + Math.PI
          // Turned toward the sofa's middle, straight across from it, or in between.
          for (const r of [at, straight, (at + straight + (Math.abs(at - straight) > Math.PI ? 2 * Math.PI : 0)) / 2]) {
            const rotation = normalizeAngle(r)
            const body = orientedRect(c, chair.width / 2, chair.depth / 2, rotation)
            if (dot(rot(v(0, 1), rotation), rot(v(0, 1), s.p.rotation)) >= -0.55 || dot(rot(v(0, 1), rotation), dir) <= 0) continue
            const room = doc.rooms.find((rm) => pointInPolygon(c, rm.polygon))
            if (!room || !polygonInsideRoom(body, room.polygon) || kept.some((k) => convexOverlap(body, k.body))) continue
            if (view && convexOverlap(body, view)) continue
            return { x: c.x, y: c.y, rotation }
          }
        }
  }
  return null
}

/** Open floor's middle: 1 m around the centre of the living room (or the sofa's room). */
export const OPEN_MIDDLE = 1.0
/** Whether a piece's footprint reaches into the open-floor middle around `c`. */
export function inOpenMiddle(c: Vec2, body: Vec2[], margin = OPEN_MIDDLE): boolean {
  return pointInPolygon(c, body) || body.some((q, i) => closestPointOnSegment(c, q, body[(i + 1) % body.length]).distance < margin)
}
/** The room an Open floor remix keeps clear, and its centre (as the twist goal reads it). */
export function openFloorCentre(doc: Doc, layout: Layout): { room: Room; c: Vec2 } | null {
  if (layout.remix?.twist !== 'open-floor') return null
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const sofa = layout.placements.find((p) => items.get(p.itemId)?.category === 'sofa')
  const room = doc.rooms.find((r) => LIVING.test(r.name)) ?? (sofa && doc.rooms.find((r) => pointInPolygon(v(sofa.x, sofa.y), r.polygon))) ?? doc.rooms[0]
  return room ? { room, c: centroid(room.polygon) } : null
}

/** Where the Next chip moves a piece: the placement, its new spot, and whether its goal is done after. */
export interface ChipMove {
  id: string
  x: number
  y: number
  rotation: number
  /** The goal the move is for, done after it (or a step closer). */
  done: boolean
}
type Spot = { x: number; y: number; rotation: number }

/**
 * The best clean spot to move placement `pid` to, for the Next chip (round 16). Candidates are a
 * slide along its own wall (turned as it is), the drop search's spots in its own room and the
 * nearest free spots there (on a 20 cm grid, as it is or a quarter turn); other living spaces only
 * when its room has none. `meets` filters spots cheaply; each survivor is scored as the home would
 * be, and `accept` judges the result. A move never adds a warning or a snag. The most stars wins,
 * then one that finishes the goal, then the roomiest floor left for the pieces still to come (a
 * desk and its chair), then the shortest move. Up to a dozen spots per room are scored.
 */
function bestMove(
  doc: Doc,
  layout: Layout,
  pid: string,
  meets: (body: Vec2[], c: Vec2) => boolean,
  accept: (after: HomeScore) => { ok: boolean; done: boolean },
  before: HomeScore,
  a0: Analysis,
): ChipMove | null {
  const p = layout.placements.find((x) => x.id === pid)
  const item = p && doc.items.find((i) => i.id === p.itemId)
  if (!p || !item || p.on || isWallDecor(item.category)) return null
  const warnings = (a: Analysis) => a.issues.filter((i) => i.severity !== 'info').length
  const base = warnings(a0)
  const snagged = new Set(before.goals.filter((g) => g.snag).map((g) => g.id))
  const others: Layout = { ...layout, placements: layout.placements.filter((x) => x.id !== p.id && x.on !== p.id) }
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const solids = others.placements.flatMap((x) => {
    const it = items.get(x.itemId)
    return it && !it.floorLevel && !x.on && !isWallDecor(it.category) ? [orientedRect(v(x.x, x.y), it.width / 2, it.depth / 2, x.rotation)] : []
  })
  const bodyAt = (q: Spot) => orientedRect(v(q.x, q.y), item.width / 2, item.depth / 2, q.rotation)
  /** Spots on a 30 cm grid of the room where a desk and its chair (1.4 × 1.3 m) would still stand clear. */
  const roomy = (room: Room, body: Vec2[]) => {
    const xs = room.polygon.map((q) => q.x)
    const ys = room.polygon.map((q) => q.y)
    let n = 0
    for (let x = Math.min(...xs) + 0.7; x <= Math.max(...xs) - 0.7; x += 0.3)
      for (let y = Math.min(...ys) + 0.65; y <= Math.max(...ys) - 0.65; y += 0.3) {
        const box = orientedRect(v(x, y), 0.7, 0.65, 0)
        if (polygonInsideRoom(box, room.polygon) && !convexOverlap(box, body) && !solids.some((b) => convexOverlap(box, b))) n++
      }
    return n
  }
  type Cand = { spot: Spot; stars: number; done: boolean; roomy: number; d: number }
  let best: Cand | null = null
  const better = (a: Cand, b: Cand) => (a.stars !== b.stars ? a.stars > b.stars : a.done !== b.done ? a.done : a.roomy !== b.roomy ? a.roomy > b.roomy : a.d < b.d)
  const consider = (spot: Spot, room?: Room) => {
    const dx = spot.x - p.x
    const dy = spot.y - p.y
    const l2: Layout = { ...layout, placements: layout.placements.map((x) => (x.id === p.id ? { ...x, ...spot } : x.on === p.id ? { ...x, x: x.x + dx, y: x.y + dy } : x)) }
    const a2 = analyze(doc, l2)
    if (warnings(a2) > base) return
    const s2 = scoreHome(doc, l2, a2)
    if (s2.goals.some((g) => g.snag && !snagged.has(g.id))) return
    const verdict = accept(s2)
    if (!verdict.ok) return
    const c: Cand = { spot, stars: s2.stars, done: verdict.done, roomy: room ? roomy(room, bodyAt(spot)) : 0, d: Math.hypot(dx, dy) }
    if (!best || better(c, best)) best = c
  }
  const home = doc.rooms.find((r) => pointInPolygon(v(p.x, p.y), r.polygon))
  const seen = new Set<string>()
  const tryRoom = (room: Room) => {
    const fits = (q: Spot) => {
      const body = bodyAt(q)
      return polygonInsideRoom(body, room.polygon) && !solids.some((b) => convexOverlap(body, b)) && meets(body, v(q.x, q.y))
    }
    const near = (a: Spot, b: Spot) => Math.hypot(a.x - p.x, a.y - p.y) - Math.hypot(b.x - p.x, b.y - p.y)
    const side = rot(v(1, 0), p.rotation)
    const slides: Spot[] = []
    if (room === home)
      for (let k = 0.1; k <= 4 && slides.length < 3; k += 0.1)
        for (const sgn of [1, -1]) {
          const q = { x: p.x + side.x * k * sgn, y: p.y + side.y * k * sgn, rotation: p.rotation }
          if (slides.length < 3 && fits(q)) slides.push(q)
        }
    const c = centroid(room.polygon)
    const drops: Spot[] = []
    for (const target of [c, ...room.polygon.map((q) => lerp(q, c, 0.25)), ...wallSegments(room).map(([a, b]) => lerp(lerp(a, b, 0.5), c, 0.2))]) {
      const r = findDrop({ ...doc, rooms: [room] }, others, item, { near: target })
      if (r.ok && fits(r.spot)) drops.push({ x: r.spot.x, y: r.spot.y, rotation: r.spot.rotation })
    }
    const grid: Spot[] = []
    if (room === home) {
      const xs = room.polygon.map((q) => q.x)
      const ys = room.polygon.map((q) => q.y)
      for (let x = Math.min(...xs) + 0.1; x < Math.max(...xs); x += 0.2)
        for (let y = Math.min(...ys) + 0.1; y < Math.max(...ys); y += 0.2)
          for (const rotation of [p.rotation, normalizeAngle(p.rotation + Math.PI / 2)]) if (fits({ x, y, rotation })) grid.push({ x, y, rotation })
    }
    let tries = 0
    for (const q of [...slides, ...drops.sort(near), ...grid.sort(near).slice(0, 4)]) {
      const key = `${q.x.toFixed(2)}:${q.y.toFixed(2)}:${normalizeAngle(q.rotation).toFixed(2)}`
      if (seen.has(key) || tries >= 12) continue
      seen.add(key)
      tries++
      consider({ x: q.x, y: q.y, rotation: q.rotation }, room)
    }
  }
  if (home) tryRoom(home)
  for (const room of doc.rooms) {
    if (best) break
    if (room !== home && !UTILITY_ROOM.test(room.name)) tryRoom(room)
  }
  const b = best as Cand | null
  return b ? { id: p.id, x: b.spot.x, y: b.spot.y, rotation: b.spot.rotation, done: b.done } : null
}

/**
 * A move that advances a remix's twist in one tap (round 16): the piece the twist goal points at
 * (`near`), taken to a clean spot that meets the twist: out of the living room's middle for Open
 * floor, 80 cm+ from its old spot for Fresh start, facing the sofa for Talk corner. The twist must
 * be done after, or this piece no longer in its way and the goal closer. Null when the twist isn't
 * a move or nothing fits (the chip then selects the piece to drag, as before).
 */
export function twistMove(doc: Doc, layout: Layout): ChipMove | null {
  const t = layout.remix?.twist
  if (!t || (t !== 'open-floor' && t !== 'fresh-spots' && t !== 'talk-corner')) return null
  const a0 = analyze(doc, layout)
  const before = scoreHome(doc, layout, a0)
  const goal = before.goals.find((g) => g.id === `twist:${t}`)
  if (!goal || goal.done || !goal.near) return null
  const p = layout.placements.find((x) => x.id === goal.near)
  if (!p) return null
  const accept = (after: HomeScore) => {
    const g2 = after.goals.find((g) => g.id === goal.id)
    if (!g2) return { ok: false, done: false }
    return { ok: g2.done || (g2.near !== p.id && (g2.progress ?? 0) > (goal.progress ?? 0) + 1e-6), done: g2.done }
  }
  if (t === 'talk-corner') {
    // The armchair goes where it faces the sofa (the spot a new armchair would take).
    const others = { ...layout, placements: layout.placements.filter((x) => x.id !== p.id) }
    const at = talkCornerSpot(doc, others)
    if (!at) return null
    const l2: Layout = { ...layout, placements: layout.placements.map((x) => (x.id === p.id ? { ...x, ...at } : x)) }
    const a2 = analyze(doc, l2)
    const v2 = accept(scoreHome(doc, l2, a2))
    const warn = (a: Analysis) => a.issues.filter((i) => i.severity !== 'info').length
    return v2.ok && warn(a2) <= warn(a0) ? { id: p.id, ...at, done: v2.done } : null
  }
  const open = openFloorCentre(doc, layout)
  const src = doc.layouts.find((l) => l.id === layout.remix!.from)?.placements.find((x) => x.itemId === p.itemId)
  const meets = (body: Vec2[], c: Vec2) => {
    if (t === 'fresh-spots') return !src || dist(c, v(src.x, src.y)) >= 0.85
    return !open || !pointInPolygon(c, open.room.polygon) || !inOpenMiddle(open.c, body, OPEN_MIDDLE + 0.05)
  }
  return bestMove(doc, layout, p.id, meets, accept, before, a0)
}

/** Twists finished in this home: some remix with that twist has reached Dream home. */
export function finishedTwists(doc: Doc): Set<TwistId> {
  return new Set(doc.layouts.flatMap((l) => (l.remix?.dreamAt ? [l.remix.twist] : [])))
}

/**
 * The (up to) three twists the picker offers today: never one a remix is still playing, twists no
 * design has finished first, each group turned round by the day so the picker changes day to day.
 */
export function pickTwists(doc: Doc, applicable: TwistId[], day = Math.floor(Date.now() / 864e5), n = 3): TwistId[] {
  const playing = new Set(doc.layouts.flatMap((l) => (l.remix && !l.remix.dreamAt ? [l.remix.twist] : [])))
  const done = finishedTwists(doc)
  const turn = (xs: TwistId[]) => (xs.length ? xs.map((_, i) => xs[(i + day) % xs.length]) : xs)
  const offer = applicable.filter((t) => !playing.has(t))
  return [...turn(offer.filter((t) => !done.has(t))), ...turn(offer.filter((t) => done.has(t)))].slice(0, n)
}

/** Twists a design can already satisfy as it stands: no challenge if the source already does it. */
const ALREADY_MET_CHECK = new Set<TwistId>(['window-seat', 'talk-corner', 'open-floor'])

/** Whether `layout`, scored as if it were a remix with this twist, already has the twist done. */
function twistMetBy(doc: Doc, layout: Layout, t: TwistId): boolean {
  try {
    const probe: Layout = { ...layout, remix: { from: layout.id, twist: t, anchors: 0 } }
    return !!scoreHome(doc, probe, analyze(doc, probe)).goals.find((g) => g.id === `twist:${t}`)?.done
  } catch {
    return false
  }
}

/** A remix of `source`: only its anchor pieces, with new placement ids, standing on nothing. */
export function remixedLayout(doc: Doc, sourceId: string, twist: TwistId, newId: () => string): Layout | null {
  const src = doc.layouts.find((l) => l.id === sourceId)
  if (!src) return null
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const placements: Placement[] = src.placements
    .filter((p) => ANCHORS.has(items.get(p.itemId)?.category ?? ''))
    .map(({ on: _on, ...p }) => ({ ...p, id: newId() }))
  const taken = doc.layouts.map((l) => l.name)
  const base = TWISTS[twist].name
  let name = base
  for (let n = 2; taken.includes(name); n++) name = `${base} ${n}`
  return { id: newId(), name, placements, remix: { from: src.id, twist, anchors: placements.length } }
}

/**
 * Two designs with the same pieces in the same spots (an untouched copy): compared to the
 * centimetre and a hundredth of a radian.
 */
export function sameArrangement(a: Layout, b: Layout): boolean {
  const key = (l: Layout) => l.placements.map((p) => `${p.itemId}:${p.x.toFixed(2)}:${p.y.toFixed(2)}:${normalizeAngle(p.rotation).toFixed(2)}`).sort().join('|')
  return a.placements.length === b.placements.length && key(a) === key(b)
}

/** The wall of its room a point is nearest to, as "roomId:wall". */
function nearestWall(doc: Doc, q: Vec2): string {
  const room = doc.rooms.find((r) => pointInPolygon(q, r.polygon))
  if (!room) return ''
  let best = 0
  let d = Infinity
  wallSegments(room).forEach(([a, b], i) => {
    const c = closestPointOnSegment(q, a, b).distance
    if (c < d) [best, d] = [i, c]
  })
  return `${room.id}:${best}`
}

/** Two seats facing each other (not back to back), close enough to talk. */
function talkPair(a: P, b: P): { ok: boolean; progress: number; facing: boolean; far: boolean } {
  const d = dist(a.center, b.center)
  const facingDot = dot(front(a), front(b))
  const toward = dot(front(a), sub(b.center, a.center)) > 0
  const facing = facingDot < -0.5 && toward
  const far = d > TALK_DIST
  const progress = 0.3 + 0.7 * clamp01((1 - facingDot) / 1.5) * (toward ? 1 : 0.5) * clamp01(TALK_DIST / Math.max(d, 0.01))
  return { ok: facing && !far, progress, facing, far }
}

/** The twist goal of a remix, from what scoreHome already built. */
function twistGoal(doc: Doc, layout: Layout, placed: P[], of: (...c: string[]) => P[], inRoom: (r: Room) => P[], rooms: Room[], solidPieces: P[]): Goal | null {
  const remix = layout.remix
  if (!remix || !TWISTS[remix.twist]) return null
  const t = remix.twist
  const src = doc.layouts.find((l) => l.id === remix.from)
  const srcOf = (p: P) => {
    const me = layout.placements.find((x) => x.id === p.id)
    const s = me && src?.placements.find((x) => x.itemId === me.itemId)
    return s ? { center: v(s.x, s.y), rotation: s.rotation } : null
  }
  let done = false
  let progress = 0
  let hint = TWISTS[t].hint
  /** The piece to turn or move (Next selects it), and a catalog piece the twist needs, if any. */
  let near: string | undefined
  let item: string | undefined
  let placeIn: string | undefined
  if (t === 'window-seat') {
    for (const s of of('sofa')) {
      if (!s.room) continue
      for (const o of s.room.openings) {
        if (o.kind !== 'window') continue
        const [a, b] = openingEndpoints(s.room, o)
        const d = dot(front(s), norm(sub(lerp(a, b, 0.5), s.center)))
        if (d >= progress || !near) near = s.id
        progress = Math.max(progress, clamp01(d))
        if (d > 0.7) done = true
      }
    }
    near ??= of('sofa')[0]?.id
    const sofa = placed.find((x) => x.id === near)
    if (!done && sofa) hint = `Turn the ${sofa.name.toLowerCase()} to look out of a window in the ${sofa.room?.name.toLowerCase() ?? 'room'}.`
  } else if (t === 'fresh-spots') {
    const anchors = placed.filter((x) => ANCHORS.has(x.category))
    const moved = anchors.filter((x) => {
      const s = srcOf(x)
      return !s || dist(s.center, x.center) >= 0.8
    })
    done = !src || (anchors.length > 0 && moved.length === anchors.length)
    progress = anchors.length ? moved.length / anchors.length : 0
    const stay = anchors.find((x) => !moved.includes(x))
    if (!done && stay) {
      near = stay.id
      const left = anchors.length - moved.length
      hint = `Move the ${stay.name.toLowerCase()} at least 80 cm from its old spot${left > 1 ? ` (${left} big pieces still to move)` : ''}.`
    }
  } else if (t === 'bed-swap') {
    near = of('bed')[0]?.id
    done = of('bed').some((b) => {
      const s = srcOf(b)
      if (!s) return true
      const turn = normalizeAngle(b.rotation - s.rotation)
      return Math.min(turn, 2 * Math.PI - turn) >= Math.PI / 2 - 1e-3 || nearestWall(doc, b.center) !== nearestWall(doc, s.center)
    })
    progress = done ? 1 : 0
  } else if (t === 'talk-corner') {
    const seats = of('sofa', 'armchair')
    progress = Math.min(seats.length, 1) * 0.3
    /** The pair closest to talking, and what's still wrong with it. */
    let bestPair: { a: P; b: P; facing: boolean; far: boolean } | undefined
    let bestP = -1
    for (let i = 0; i < seats.length; i++)
      for (let j = i + 1; j < seats.length; j++) {
        const [a, b] = [seats[i], seats[j]]
        const pr = talkPair(a, b)
        if (pr.ok) done = true
        if (pr.progress > bestP) [bestP, bestPair] = [pr.progress, { a, b, facing: pr.facing, far: pr.far }]
        progress = Math.max(progress, pr.progress)
      }
    // Point at the armchair to turn or move (or the sofa), or ask for an armchair when there's only one seat.
    const pairChair = bestPair && ([bestPair.a, bestPair.b].find((x) => x.category === 'armchair') ?? bestPair.b)
    near = (pairChair ?? of('armchair')[0] ?? seats[0])?.id
    if (!done && seats.length < 2) {
      item = 'armchair'
      hint = `Add an armchair, then turn it and the sofa to face each other, under ${TALK_DIST} m apart.`
    } else if (!done && bestPair && pairChair) {
      const other = pairChair === bestPair.a ? bestPair.b : bestPair.a
      const what = pairChair.name.toLowerCase()
      hint = !bestPair.facing
        ? `Turn the ${what} to face the ${other.name.toLowerCase()}${bestPair.far ? `, then bring it under ${TALK_DIST} m away` : ''}.`
        : `Move the ${what} closer to the ${other.name.toLowerCase()}, under ${TALK_DIST} m away.`
    }
  } else if (t === 'open-floor') {
    const room = rooms.find((r) => LIVING.test(r.name)) ?? of('sofa')[0]?.room ?? rooms[0]
    if (room) {
      const c = centroid(room.polygon)
      const pieces = inRoom(room).filter((x) => !x.floor)
      const near0 = (x: P) => inOpenMiddle(c, x.body)
      const clear = pieces.filter((x) => !near0(x))
      done = pieces.length >= 4 && clear.length === pieces.length
      progress = pieces.length ? (clear.length / pieces.length) * Math.min(1, pieces.length / 4) : 0
      near = pieces.find((x) => near0(x))?.id
      if (!done && clear.length < pieces.length) hint = `Move ${pieces.length - clear.length === 1 ? 'one piece' : `${pieces.length - clear.length} pieces`} out of the middle of the ${room.name.toLowerCase()} (1 m around its centre).`
      else if (!done) {
        // Too few pieces with the middle already clear: Next suggests one for the room's edges
        // (round 16: pickSpot keeps it out of the middle), so the chip alone can finish the twist.
        item = 'plant'
        placeIn = room.id
        const left = 4 - pieces.length
        hint = `Add ${left === 1 ? 'one more piece' : `${left} more pieces`} around the edges of the ${room.name.toLowerCase()}, keeping its middle clear.`
      }
    }
  } else if (t === 'less-is-more') {
    const n = solidPieces.length
    done = n <= 10
    progress = n ? Math.min(1, 10 / n) : 1
    hint = `${n} of 10 pieces. ${TWISTS[t].hint}`
  }
  return {
    id: `twist:${t}`,
    title: TWISTS[t].title,
    hint,
    category: 'style',
    stars: 3,
    done,
    progress: done ? 1 : clamp01(progress),
    ...(done ? {} : { near, ...(item ? { item, tab: TAB_OF[item] } : {}), ...(placeIn ? { placeIn } : {}) }),
  }
}

// ---- Palette ---------------------------------------------------------------

/** Painted pieces the palette goal wants, and the colour families it may use. */
export const PALETTE_PIECES = 3
export const PALETTE_FAMILIES = 3

/** Names for the paint swatches (and the warm pick colours), for naming a palette. */
const COLOR_NAMES: [string, string][] = [
  ['#7d8fa3', 'Slate'], ['#a3907d', 'Taupe'], ['#c9c3b6', 'Stone'], ['#8a6b4f', 'Walnut'], ['#3d3d3d', 'Charcoal'], ['#5f8f5a', 'Sage'],
  ['#c7a98a', 'Oak'], ['#e2d6b8', 'Sand'], ['#b5534a', 'Terracotta'], ['#e0b04f', 'Honey'], ['#5b7bb5', 'Cornflower'], ['#f2efe8', 'Linen'],
  ['#ffffff', 'White'], ['#111111', 'Black'], ['#d98a9a', 'Rose'], ['#6f5a8f', 'Plum'], ['#3f7f7a', 'Teal'], ['#2f4a6b', 'Navy'],
]

function rgb(hex: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) return null
  const h = m[1].length === 3 ? m[1].replace(/./g, (c) => c + c) : m[1]
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number]
}

/** Whether two colours are the same paint (undefined when either can't be read). */
export function sameColor(a: string, b: string): boolean | undefined {
  const x = rgb(a)
  const y = rgb(b)
  if (!x || !y) return undefined
  return x.every((c, i) => Math.abs(c - y[i]) <= 2)
}

/** The nearest swatch name for a colour ("Sage"). */
export function colorName(hex: string): string {
  const c = rgb(hex)
  if (!c) return 'Custom'
  let best = COLOR_NAMES[0][1]
  let d = Infinity
  for (const [h, name] of COLOR_NAMES) {
    const q = rgb(h)!
    const e = (c[0] - q[0]) ** 2 * 0.3 + (c[1] - q[1]) ** 2 * 0.59 + (c[2] - q[2]) ** 2 * 0.11
    if (e < d) [d, best] = [e, name]
  }
  return best
}

/**
 * A colour's hue family: "neutral" for greys, beiges and muted tones (they go with anything), else
 * one of twelve 30° slices of the colour wheel ("h0" red, "h1" orange... "h7" blue).
 */
export function hueFamily(hex: string): string {
  const c = rgb(hex)
  if (!c) return 'neutral'
  const [r, g, b] = c.map((x) => x / 255)
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const d = max - min
  if (d < 0.18) return 'neutral'
  const h = max === r ? 60 * (((g - b) / d + 6) % 6) : max === g ? 60 * ((b - r) / d + 2) : 60 * ((r - g) / d + 4)
  return `h${Math.round(h / 30) % 12}`
}

/**
 * Whether a set of paint colours makes a palette: at most PALETTE_FAMILIES hue families (neutrals
 * are free), or neighbouring hues plus one accent. Named from its most-used colours ("Sage &
 * Terracotta"); `odd` is the family to repaint when it doesn't hold together.
 */
export function paletteOf(colors: string[]): { ok: boolean; name: string; odd?: string } {
  const count = new Map<string, number>()
  for (const c of colors) count.set(colorName(c), (count.get(colorName(c)) ?? 0) + 1)
  const names = [...count.entries()].sort((a, b) => b[1] - a[1]).map(([n]) => n).slice(0, 3)
  const name = names.length > 1 ? `${names.slice(0, -1).join(', ')} & ${names[names.length - 1]}` : (names[0] ?? '')
  const fams = new Map<string, number>()
  for (const c of colors) {
    const f = hueFamily(c)
    if (f !== 'neutral') fams.set(f, (fams.get(f) ?? 0) + 1)
  }
  if (fams.size <= PALETTE_FAMILIES) return { ok: true, name }
  // Neighbours (within 60° of each other) plus one accent.
  const slots = [...fams.keys()].map((f) => Number(f.slice(1)))
  const span = (xs: number[]) => Math.min(...xs.map((s) => Math.max(...xs.map((t) => (t - s + 12) % 12))))
  const accent = slots.find((a) => span(slots.filter((s) => s !== a)) <= 2)
  if (accent !== undefined) return { ok: true, name }
  const odd = [...fams.entries()].sort((a, b) => a[1] - b[1])[0][0]
  return { ok: false, name, odd }
}
