// Paint swatches and the "Pick one" variants the hint picker offers for each hinted category.
// Pure data, so tests can check every pick against the starter catalog.

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

/** One card on the picker: a starter catalog piece, a short label and the colour it's placed in. */
export interface Pick {
  /** A StarterProduct id (src/model/starterCatalog.ts). */
  id: string
  label: string
  color: string
}

const { walnut, oak, charcoal, linen, sage, terracotta, honey, taupe } = WARM

/** 2-3 variants per category the Next chip points at. Every id is a starter piece of that same category. */
export const PICKS: Record<string, Pick[]> = {
  'coffee-table': [
    { id: 'coffee-rect', label: 'Long', color: walnut },
    { id: 'coffee-round', label: 'Round', color: oak },
    { id: 'coffee-small', label: 'Small', color: charcoal },
  ],
  armchair: [
    { id: 'armchair', label: 'Classic', color: taupe },
    { id: 'accent-chair', label: 'Accent', color: honey },
    { id: 'swivel', label: 'Swivel', color: sage },
  ],
  'dining-table': [
    { id: 'dining-round', label: 'Round', color: walnut },
    { id: 'dining-4', label: 'Seats 4', color: oak },
    { id: 'bistro', label: 'Bistro', color: charcoal },
  ],
  // "Bench" (dining-bench) is a bench, not a dining chair, so it wouldn't finish a chair goal.
  'dining-chair': [
    { id: 'dining-chair', label: 'Chair', color: walnut },
    { id: 'dining-armchair', label: 'Armchair', color: linen },
  ],
  nightstand: [
    { id: 'nightstand', label: 'Standard', color: walnut },
    { id: 'nightstand-sm', label: 'Slim', color: linen },
  ],
  dresser: [
    { id: 'dresser-3', label: '3-drawer', color: oak },
    { id: 'dresser-6', label: 'Wide', color: walnut },
    { id: 'chest-tall', label: 'Tall', color: linen },
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
    { id: 'plant-md', label: 'Medium', color: sage },
    { id: 'plant-lg', label: 'Tall', color: sage },
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
 * The picks for a category, the one in the player's remembered colour first (marked `taste`).
 * Order is otherwise kept, so the first pick is stable for a player with no taste yet.
 */
export function picksFor(category: string, taste: Partial<Record<Family, string>> = {}): (Pick & { taste?: boolean })[] {
  const list = PICKS[category] ?? []
  const fam = familyOf(category)
  const liked = fam ? taste[fam] : undefined
  const k = liked ? list.findIndex((p) => p.color === liked) : -1
  if (k < 0) return list
  return [{ ...list[k], taste: true }, ...list.filter((_, i) => i !== k)]
}
