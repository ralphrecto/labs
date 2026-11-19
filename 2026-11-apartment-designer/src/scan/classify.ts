import { catalogEntry } from '../model/catalog'
import type { CategoryGuess } from './estimate'

// Guesses a furniture category from a box's size. A scan only tells us the footprint and
// height, so this is a size prior: it gets beds, sofas, wardrobes and shelves right most of
// the time and leaves the rest for the person to relabel. A server model (SpatialLM, SAM 3)
// can replace it by filling in categories directly.

/** Which side of the box is its back: a long side (sofa, desk), a short side (bed headboard), or either. */
export type BackSide = 'long' | 'short' | 'any'

interface Prototype {
  category: string
  /** [min, max] of the longer footprint side, shorter side, and height, in meters. */
  long: [number, number]
  short: [number, number]
  height: [number, number]
  back: BackSide
  /** [min, max] of long / short, for shapes that size alone can't tell apart (a lamp vs a narrow shelf). */
  aspect?: [number, number]
  /** Prior weight for categories that match almost any size (plants). */
  prior?: number
}

const PROTOTYPES: Prototype[] = [
  { category: 'bed', long: [1.85, 2.3], short: [0.95, 2.05], height: [0.3, 0.7], back: 'short' },
  // A bed with a tall headboard: only wide ones, so it doesn't swallow sofas.
  { category: 'bed', long: [1.85, 2.3], short: [1.3, 2.05], height: [0.7, 1.3], back: 'short' },
  { category: 'sofa', long: [1.4, 3.3], short: [0.75, 1.1], height: [0.6, 1.0], back: 'long' },
  { category: 'armchair', long: [0.7, 1.05], short: [0.7, 1.0], height: [0.65, 1.1], back: 'any' },
  { category: 'coffee-table', long: [0.6, 1.4], short: [0.4, 0.9], height: [0.28, 0.52], back: 'long' },
  { category: 'tv-stand', long: [1.0, 2.2], short: [0.3, 0.55], height: [0.35, 0.7], back: 'long' },
  { category: 'nightstand', long: [0.35, 0.65], short: [0.3, 0.5], height: [0.42, 0.75], back: 'long' },
  { category: 'dresser', long: [0.75, 1.8], short: [0.38, 0.56], height: [0.72, 1.3], back: 'long' },
  { category: 'wardrobe', long: [0.8, 2.6], short: [0.5, 0.72], height: [1.65, 2.45], back: 'long' },
  { category: 'dining-table', long: [0.9, 2.4], short: [0.75, 1.2], height: [0.7, 0.95], back: 'any' },
  { category: 'dining-chair', long: [0.4, 0.6], short: [0.38, 0.58], height: [0.75, 1.05], back: 'any' },
  { category: 'desk', long: [1.0, 2.0], short: [0.5, 0.85], height: [0.7, 1.35], back: 'long' },
  { category: 'office-chair', long: [0.55, 0.78], short: [0.55, 0.75], height: [0.85, 1.3], back: 'any' },
  { category: 'bookshelf', long: [0.4, 1.3], short: [0.2, 0.45], height: [1.0, 2.3], back: 'long', aspect: [1.6, 8] },
  { category: 'floor-lamp', long: [0.2, 0.55], short: [0.2, 0.55], height: [1.3, 1.95], back: 'any', aspect: [1, 1.6] },
  { category: 'plant', long: [0.25, 0.8], short: [0.25, 0.8], height: [0.3, 1.8], back: 'any', prior: 0.5 },
]

/** 1 inside [lo, hi], falling off smoothly outside (about 15% of the bound per standard deviation). */
function fit(value: number, [lo, hi]: [number, number]) {
  if (value >= lo && value <= hi) return 1
  const edge = value < lo ? lo : hi
  const sigma = 0.15 * edge + 0.02
  const d = (value - edge) / sigma
  return Math.exp(-0.5 * d * d)
}

export function backSide(category: string): BackSide {
  return PROTOTYPES.find((p) => p.category === category)?.back ?? 'any'
}

const BED_SIZES = [
  { name: 'Twin bed', width: 0.97 },
  { name: 'Full bed', width: 1.37 },
  { name: 'Queen bed', width: 1.52 },
  { name: 'King bed', width: 1.93 },
]

/** A display name for a category at a given size ("Queen bed"). */
export function displayName(category: string, width: number, depth: number) {
  if (category === 'box') return 'Unknown object'
  if (category === 'bed') {
    const w = Math.min(width, depth)
    return BED_SIZES.reduce((best, b) => (Math.abs(b.width - w) < Math.abs(best.width - w) ? b : best)).name
  }
  return catalogEntry(category).name
}

export interface Classification {
  category: string
  name: string
  confidence: number
  alternatives: CategoryGuess[]
}

/** Rank categories for a box with footprint a × b (any order) and height h. */
export function classifyBox(a: number, b: number, h: number): Classification {
  const long = Math.max(a, b)
  const short = Math.min(a, b)
  const scored = PROTOTYPES.map((p) => ({
    category: p.category,
    name: displayName(p.category, long, short),
    score: fit(long, p.long) * fit(short, p.short) * fit(h, p.height) * (p.aspect ? fit(long / short, p.aspect) : 1) * (p.prior ?? 1),
  }))
    .sort((x, y) => y.score - x.score)
    // One entry per category (the bed has two prototypes).
    .filter((s, i, all) => all.findIndex((o) => o.category === s.category) === i)
  const [best, second] = scored
  const alternatives = scored.filter((s) => s.score > 0.05).slice(0, 4)
  if (best.score < 0.2) {
    return { category: 'box', name: 'Unknown object', confidence: 0, alternatives }
  }
  // High when the best fits well and clearly beats the runner-up.
  const confidence = best.score * (best.score / (best.score + (second?.score ?? 0)))
  return { category: best.category, name: best.name, confidence: Math.round(confidence * 100) / 100, alternatives }
}
