import type { Clearance, Item } from './types'

export interface CatalogEntry {
  category: string
  name: string
  width: number
  depth: number
  height: number
  clearance: Clearance
  color: string
  floorLevel?: boolean
  /** Size presets, e.g. bed sizes. */
  presets?: { name: string; width: number; depth: number }[]
}

const c = (front = 0, back = 0, left = 0, right = 0): Clearance => ({ front, back, left, right })

// Clearances follow common interior design rules of thumb; every item can override them.
export const CATALOG: CatalogEntry[] = [
  { category: 'sofa', name: 'Sofa', width: 2.1, depth: 0.92, height: 0.85, clearance: c(0.45), color: '#7d8fa3' },
  { category: 'armchair', name: 'Armchair', width: 0.85, depth: 0.85, height: 0.85, clearance: c(0.6), color: '#a3907d' },
  { category: 'coffee-table', name: 'Coffee table', width: 1.1, depth: 0.6, height: 0.45, clearance: c(), color: '#8a6b4f' },
  { category: 'tv-stand', name: 'TV stand', width: 1.6, depth: 0.45, height: 0.55, clearance: c(), color: '#5b5b5b' },
  {
    category: 'bed',
    name: 'Bed',
    width: 1.52,
    depth: 2.03,
    height: 0.6,
    clearance: c(0.6, 0, 0.6, 0.6),
    color: '#c9c3b6',
    presets: [
      { name: 'Twin', width: 0.97, depth: 1.91 },
      { name: 'Full', width: 1.37, depth: 1.91 },
      { name: 'Queen', width: 1.52, depth: 2.03 },
      { name: 'King', width: 1.93, depth: 2.03 },
    ],
  },
  { category: 'nightstand', name: 'Nightstand', width: 0.5, depth: 0.4, height: 0.55, clearance: c(), color: '#8a6b4f' },
  { category: 'dresser', name: 'Dresser', width: 1.2, depth: 0.5, height: 0.8, clearance: c(0.9), color: '#9a7b5f' },
  { category: 'wardrobe', name: 'Wardrobe', width: 1.0, depth: 0.6, height: 2.0, clearance: c(0.9), color: '#b9a58f' },
  { category: 'dining-table', name: 'Dining table', width: 1.6, depth: 0.9, height: 0.75, clearance: c(), color: '#8a6b4f' },
  { category: 'dining-chair', name: 'Dining chair', width: 0.45, depth: 0.5, height: 0.9, clearance: c(0, 0.75), color: '#6e5a48' },
  { category: 'desk', name: 'Desk', width: 1.4, depth: 0.7, height: 0.75, clearance: c(0.9), color: '#d2c4ae' },
  { category: 'office-chair', name: 'Office chair', width: 0.65, depth: 0.65, height: 1.05, clearance: c(), color: '#3d3d3d' },
  { category: 'bookshelf', name: 'Bookshelf', width: 0.8, depth: 0.35, height: 1.8, clearance: c(0.6), color: '#b08d68' },
  { category: 'rug', name: 'Rug', width: 2.4, depth: 1.7, height: 0.01, clearance: c(), color: '#c7a98a', floorLevel: true },
  { category: 'floor-lamp', name: 'Floor lamp', width: 0.4, depth: 0.4, height: 1.6, clearance: c(), color: '#e2d6b8' },
  { category: 'plant', name: 'Plant', width: 0.45, depth: 0.45, height: 1.1, clearance: c(), color: '#5f8f5a' },
  { category: 'box', name: 'Box', width: 1, depth: 1, height: 1, clearance: c(), color: '#9aa4ad' },
]

export function catalogEntry(category: string): CatalogEntry {
  return CATALOG.find((e) => e.category === category) ?? CATALOG[CATALOG.length - 1]
}

export function itemFromCatalog(entry: CatalogEntry, id: string, overrides: Partial<Item> = {}): Item {
  return {
    id,
    name: entry.name,
    category: entry.category,
    width: entry.width,
    depth: entry.depth,
    height: entry.height,
    clearance: { ...entry.clearance },
    color: entry.color,
    floorLevel: entry.floorLevel,
    source: 'catalog',
    ...overrides,
  }
}
