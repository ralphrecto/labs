// Paint swatches and the "Pick one" variants the hint picker offers for each hinted category.
// Pure data, so tests can check every pick against the starter catalog.
import type { MoodId } from '../model/types'

/** The paint swatches of the selected piece's bar (kept in step with GameApp's list). */
export const SWATCHES = ['#7d8fa3', '#a3907d', '#c9c3b6', '#8a6b4f', '#3d3d3d', '#5f8f5a', '#c7a98a', '#e2d6b8', '#b5534a', '#e0b04f', '#5b7bb5', '#f2efe8']

/** Warm colours the picks are drawn in. */
export const WARM = {
  walnut: '#8a6b4f',
  oak: '#c7a98a',
  charcoal: '#3d3d3d',
  linen: '#f2efe8',
  sage: '#5f8f5a',
  terracotta: '#b5534a',
  honey: '#e0b04f',
  taupe: '#a3907d',
} as const

/** A material family the player's taste is remembered for. */
export type Family = 'wood' | 'fabric' | 'light'

/** How a thumbnail tells one variant from another: a round top, a taller body, or drawn smaller. */
export interface ThumbLook {
  round?: boolean
  tall?: boolean
  /** Drawn wider than usual (a landscape print). */
  wide?: boolean
  /** Drawn at this size (1 = the category's usual thumbnail), from the ground up. */
  scale?: number
}

/** One card on the picker: a starter catalog piece, a short label and the colour it's placed in. */
export interface Pick {
  /** A StarterProduct id (src/model/starterCatalog.ts). */
  id: string
  label: string
  color: string
  /** Its thumbnail's look, when the shape or size is the difference (Round, Tall, Medium). */
  look?: ThumbLook
}

const { walnut, oak, charcoal, linen, sage, terracotta, honey, taupe } = WARM

/** 2-3 variants per category the Next chip points at. Every id is a starter piece of that same category. */
export const PICKS: Record<string, Pick[]> = {
  'coffee-table': [
    { id: 'coffee-rect', label: 'Long', color: walnut },
    { id: 'coffee-round', label: 'Round', color: oak, look: { round: true } },
    { id: 'coffee-small', label: 'Small', color: charcoal, look: { scale: 0.78 } },
  ],
  armchair: [
    { id: 'armchair', label: 'Classic', color: taupe },
    { id: 'accent-chair', label: 'Accent', color: honey },
    { id: 'swivel', label: 'Swivel', color: sage },
  ],
  'dining-table': [
    { id: 'dining-round', label: 'Round', color: walnut, look: { round: true } },
    { id: 'dining-4', label: 'Seats 4', color: oak },
    { id: 'bistro', label: 'Bistro', color: charcoal },
  ],
  // "Bench" (dining-bench) is a bench, not a dining chair, so it wouldn't finish a chair goal.
  'dining-chair': [
    { id: 'dining-chair', label: 'Chair', color: walnut },
    { id: 'dining-armchair', label: 'Carver', color: linen },
  ],
  // One task chair in a few colours: the desk's chair gets pick cards like the dining chairs.
  'office-chair': [
    { id: 'office-chair', label: 'Black', color: charcoal },
    { id: 'office-chair', label: 'Linen', color: linen },
    { id: 'office-chair', label: 'Sage', color: sage },
  ],
  nightstand: [
    { id: 'nightstand', label: 'Standard', color: walnut },
    { id: 'nightstand-sm', label: 'Slim', color: linen, look: { scale: 0.85 } },
  ],
  dresser: [
    { id: 'dresser-3', label: '3-drawer', color: oak },
    { id: 'dresser-6', label: 'Wide', color: walnut },
    { id: 'chest-tall', label: 'Tall', color: linen, look: { tall: true } },
  ],
  desk: [
    { id: 'desk-sm', label: 'Writing', color: oak },
    { id: 'desk', label: 'Classic', color: walnut },
    { id: 'desk-standing', label: 'Standing', color: charcoal },
  ],
  rug: [
    { id: 'rug-6x9', label: '6×9', color: oak },
    { id: 'rug-8x10', label: '8×10', color: terracotta },
    { id: 'rug-round', label: 'Round', color: taupe },
  ],
  'floor-lamp': [
    { id: 'floor-lamp', label: 'Floor', color: linen },
    { id: 'arc-lamp', label: 'Arc', color: charcoal },
  ],
  'table-lamp': [
    { id: 'table-lamp', label: 'Linen', color: linen },
    { id: 'table-lamp', label: 'Honey', color: honey },
  ],
  pendant: [
    { id: 'pendant', label: 'Linen', color: linen },
    { id: 'pendant', label: 'Brass', color: honey },
    { id: 'pendant', label: 'Black', color: charcoal },
  ],
  plant: [
    { id: 'plant-md', label: 'Medium', color: sage, look: { scale: 0.7 } },
    { id: 'plant-lg', label: 'Tall', color: sage },
  ],
  // Art goes up over the sofa or the bed, where a designer hangs about two thirds of its width:
  // the large canvas first, a framed print and a landscape print for a narrower wall.
  'wall-art': [
    { id: 'art-lg', label: 'Large', color: '#7fa7c9', look: { wide: true } },
    { id: 'art-md', label: 'Framed', color: '#e8a27c' },
    { id: 'art-land', label: 'Landscape', color: '#8fae84', look: { wide: true, scale: 0.8 } },
  ],
  sofa: [
    { id: 'sofa-apt', label: 'Compact', color: taupe },
    { id: 'sofa-3', label: '3-seat', color: sage },
    { id: 'sofa-2', label: 'Loveseat', color: terracotta },
  ],
}

const FAMILY: Record<string, Family> = {
  'coffee-table': 'wood', 'dining-table': 'wood', nightstand: 'wood', dresser: 'wood', desk: 'wood', 'side-table': 'wood',
  sofa: 'fabric', armchair: 'fabric', 'dining-chair': 'fabric', rug: 'fabric',
  'floor-lamp': 'light', 'table-lamp': 'light', pendant: 'light',
}

/** The material family a category's colour counts toward, if any (plants have none). */
export const familyOf = (category: string): Family | undefined => FAMILY[category]

/**
 * How a room's mood (round 17) tints its picks: `palette` is the colours that read as the mood,
 * `map` swaps a pick's usual colour for the mood's take on it (walnut turns oak in an airy room).
 */
export const MOOD_TONES: Record<MoodId, { palette: string[]; map: Record<string, string> }> = {
  airy: {
    palette: [linen, oak, '#e2d6b8', '#c9c3b6', sage, '#7fa7c9'],
    map: { [walnut]: oak, [charcoal]: linen, [taupe]: '#c9c3b6', [terracotta]: '#e2d6b8', [honey]: linen, '#e8a27c': '#7fa7c9' },
  },
  snug: {
    palette: [walnut, honey, terracotta, taupe, '#e8a27c'],
    map: { [oak]: walnut, [linen]: honey, [charcoal]: walnut, [sage]: terracotta, '#7fa7c9': '#e8a27c' },
  },
  bold: {
    palette: [terracotta, honey, '#5b7bb5', charcoal, '#e8a27c'],
    map: { [taupe]: '#5b7bb5', [linen]: honey, [oak]: terracotta, [walnut]: charcoal, '#8fae84': '#5b7bb5' },
  },
}

/** Where picksFor looks up the mood of the room a pick is for, when it isn't given (set by MoodCard). */
let moodSource: (() => MoodId | undefined) | null = null
export function setPickMoodSource(fn: (() => MoodId | undefined) | null) {
  moodSource = fn
}

/**
 * A room's mood applied to a category's picks: each pick takes the mood's colour for it (unless a
 * sibling already wears that colour), and the first pick always wears one of the mood's colours,
 * so the glowing card suits the room. The order (and so the first pick's shape and size) is kept:
 * a mood never changes what fits where. Colour-only variants (a lamp in Linen or Honey) are named
 * for their colour: they keep it, and the one in the mood's palette leads instead.
 * Picks in the mood's palette are marked `mood`.
 */
export function moodPicks(list: Pick[], mood: MoodId): (Pick & { mood?: boolean })[] {
  const tone = MOOD_TONES[mood]
  const twins = list.length > 1 && list.every((p) => p.id === list[0].id)
  if (twins) {
    const marked = list.map((p) => ({ ...p, mood: tone.palette.includes(p.color) }))
    return [...marked.filter((p) => p.mood), ...marked.filter((p) => !p.mood)]
  }
  const used = new Set(list.map((p) => p.color))
  const out = list.map((p) => {
    const to = tone.map[p.color]
    let color = p.color
    if (to && !used.has(to)) {
      used.add(to)
      color = to
    }
    return { ...p, color }
  })
  if (out[0] && !tone.palette.includes(out[0].color)) {
    const free = tone.palette.find((c) => !out.some((p) => p.color === c))
    if (free) out[0] = { ...out[0], color: free }
  }
  return out.map((p) => ({ ...p, mood: tone.palette.includes(p.color) }))
}

/**
 * The picks for a category, the one in the player's remembered colour first (marked `taste`).
 * Order is otherwise kept, so the first pick is stable for a player with no taste yet.
 * In a room with a mood (round 17) the picks are drawn in the mood's colours (moodPicks), and the
 * lead pick is shown as "your style" when no remembered colour leads.
 */
export function picksFor(category: string, taste: Partial<Record<Family, string>> = {}, mood: MoodId | undefined = moodSource?.()): (Pick & { taste?: boolean; mood?: boolean })[] {
  const base = PICKS[category] ?? []
  const list = mood ? moodPicks(base, mood) : base
  const fam = familyOf(category)
  const liked = fam ? taste[fam] : undefined
  const k = liked ? list.findIndex((p) => p.color === liked) : -1
  if (k >= 0) return [{ ...list[k], taste: true }, ...list.filter((_, i) => i !== k)]
  if (mood && list[0] && (list[0] as { mood?: boolean }).mood) return [{ ...list[0], taste: true }, ...list.slice(1)]
  return list
}
