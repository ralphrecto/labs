// A starter catalog of generic furniture at typical retail sizes, so a plan can be furnished
// before you've picked real products. Each entry builds on a base category in catalog.ts for
// its look and default clearances. Sizes are in centimeters here and converted to meters.

import { catalogEntry, itemFromCatalog } from './catalog'
import type { Item } from './types'

export interface StarterProduct {
  id: string
  name: string
  /** Base category in catalog.ts. */
  category: string
  /** Group shown in the catalog browser. */
  group: string
  width: number
  depth: number
  height: number
  color?: string
  /** Extra search words. */
  tags?: string
}

type Row = [id: string, name: string, category: string, w: number, d: number, h: number, extra?: { color?: string; tags?: string }]

const groups: Record<string, Row[]> = {
  'Sofas & seating': [
    ['sofa-3', '3-seat sofa', 'sofa', 213, 94, 86],
    ['sofa-2', 'Loveseat', 'sofa', 152, 89, 86, { tags: '2-seat two seater' }],
    ['sofa-apt', 'Apartment sofa', 'sofa', 183, 89, 84, { tags: 'small compact' }],
    ['sofa-4', '4-seat sofa', 'sofa', 254, 99, 86, { tags: 'large' }],
    ['sofa-sleeper', 'Sleeper sofa', 'sofa', 203, 97, 89, { tags: 'sofa bed pull-out' }],
    ['sofa-deep', 'Deep-seat sofa', 'sofa', 236, 109, 81],
    ['sectional-l', 'L-shaped sectional (main piece)', 'sofa', 264, 97, 84, { tags: 'corner sectional' }],
    ['chaise', 'Chaise (sectional add-on)', 'sofa', 81, 163, 84, { tags: 'sectional chaise lounge' }],
    ['armchair', 'Armchair', 'armchair', 84, 86, 84],
    ['accent-chair', 'Accent chair', 'armchair', 71, 76, 79],
    ['recliner', 'Recliner', 'armchair', 89, 97, 104, { tags: 'reclining' }],
    ['swivel', 'Swivel chair', 'armchair', 79, 81, 79],
    ['ottoman-sq', 'Square ottoman', 'ottoman', 61, 61, 43, { tags: 'footstool pouf' }],
    ['ottoman-rect', 'Storage ottoman', 'ottoman', 91, 46, 43],
    ['pouf', 'Pouf', 'ottoman', 46, 46, 41],
  ],
  Tables: [
    ['coffee-rect', 'Coffee table', 'coffee-table', 122, 61, 43],
    ['coffee-small', 'Small coffee table', 'coffee-table', 91, 51, 43],
    ['coffee-round', 'Round coffee table', 'coffee-table', 91, 91, 41, { tags: 'circle' }],
    ['side', 'Side table', 'side-table', 51, 51, 56, { tags: 'end table' }],
    ['side-c', 'C-shaped side table', 'side-table', 46, 30, 61],
    ['console', 'Console table', 'console-table', 122, 36, 79, { tags: 'entryway sofa table' }],
    ['console-narrow', 'Narrow console', 'console-table', 91, 25, 76, { tags: 'hallway' }],
  ],
  'Dining': [
    ['dining-4', 'Dining table, seats 4', 'dining-table', 122, 76, 76],
    ['dining-6', 'Dining table, seats 6', 'dining-table', 183, 91, 76],
    ['dining-8', 'Dining table, seats 8', 'dining-table', 229, 102, 76],
    ['dining-round', 'Round dining table, seats 4', 'dining-table', 107, 107, 76, { tags: 'circle' }],
    ['bistro', 'Bistro table, seats 2', 'dining-table', 71, 71, 76, { tags: 'small' }],
    ['dining-chair', 'Dining chair', 'dining-chair', 46, 53, 86],
    ['dining-armchair', 'Dining armchair', 'dining-chair', 58, 56, 81],
    ['counter-stool', 'Counter stool', 'dining-chair', 43, 46, 91, { tags: 'bar stool kitchen island' }],
    ['dining-bench', 'Dining bench', 'bench', 152, 38, 46],
  ],
  Bedroom: [
    ['bed-twin', 'Twin bed', 'bed', 99, 196, 91, { tags: 'single' }],
    ['bed-twinxl', 'Twin XL bed', 'bed', 99, 206, 91],
    ['bed-full', 'Full bed', 'bed', 140, 196, 102, { tags: 'double' }],
    ['bed-queen', 'Queen bed', 'bed', 158, 213, 107],
    ['bed-king', 'King bed', 'bed', 198, 213, 107],
    ['bed-cal', 'California king bed', 'bed', 188, 223, 107],
    ['bed-platform-q', 'Queen platform bed (low)', 'bed', 163, 216, 36, { tags: 'low profile' }],
    ['daybed', 'Daybed', 'bed', 102, 201, 86, { tags: 'trundle' }],
    ['crib', 'Crib', 'bed', 74, 135, 89, { tags: 'baby nursery' }],
    ['nightstand', 'Nightstand', 'nightstand', 51, 41, 61, { tags: 'bedside table' }],
    ['nightstand-sm', 'Small nightstand', 'nightstand', 41, 36, 56],
    ['dresser-6', '6-drawer dresser', 'dresser', 152, 46, 81],
    ['dresser-3', '3-drawer dresser', 'dresser', 81, 46, 81],
    ['chest-tall', 'Tall chest of drawers', 'dresser', 81, 46, 122, { tags: 'tallboy' }],
    ['wardrobe-2', '2-door wardrobe', 'wardrobe', 100, 58, 201, { tags: 'closet armoire' }],
    ['wardrobe-3', '3-door wardrobe', 'wardrobe', 150, 58, 201],
    ['bed-bench', 'End-of-bed bench', 'bench', 122, 41, 46],
  ],
  'Living & storage': [
    ['tv-stand', 'Media console', 'tv-stand', 152, 41, 56, { tags: 'tv stand' }],
    ['tv-stand-lg', 'Large media console', 'tv-stand', 183, 46, 61],
    ['sideboard', 'Sideboard', 'sideboard', 152, 46, 81, { tags: 'buffet credenza' }],
    ['bookcase-tall', 'Tall bookcase', 'bookshelf', 80, 30, 202, { tags: 'bookshelf' }],
    ['bookcase-low', 'Low bookcase', 'bookshelf', 80, 30, 106],
    ['cube-4x4', 'Cube shelf, 4×4', 'bookshelf', 147, 39, 147, { tags: 'kallax' }],
    ['cube-2x4', 'Cube shelf, 2×4', 'bookshelf', 77, 39, 147],
    ['shoe-cabinet', 'Shoe cabinet', 'sideboard', 80, 30, 100, { tags: 'entryway' }],
    ['entry-bench', 'Entryway bench', 'bench', 91, 36, 46, { tags: 'shoe bench' }],
  ],
  Office: [
    ['desk', 'Desk', 'desk', 122, 61, 76],
    ['desk-sm', 'Writing desk', 'desk', 102, 51, 76, { tags: 'small' }],
    ['desk-standing', 'Standing desk', 'desk', 152, 76, 74, { tags: 'sit stand adjustable' }],
    ['desk-l', 'Corner desk (long side)', 'desk', 152, 61, 76, { tags: 'l-shaped' }],
    ['office-chair', 'Office chair', 'office-chair', 66, 66, 107, { tags: 'task chair' }],
    ['file-cabinet', 'Filing cabinet', 'bookshelf', 38, 51, 66, { tags: 'file drawers' }],
  ],
  'Rugs': [
    ['rug-5x8', "Rug 5' × 8'", 'rug', 244, 152, 1, { tags: '150 x 240' }],
    ['rug-6x9', "Rug 6' × 9'", 'rug', 274, 183, 1],
    ['rug-8x10', "Rug 8' × 10'", 'rug', 305, 244, 1, { tags: '240 x 300' }],
    ['rug-9x12', "Rug 9' × 12'", 'rug', 366, 274, 1],
    ['rug-round', "Round rug 6'", 'rug', 183, 183, 1, { tags: 'circle' }],
    ['runner', "Runner 2.5' × 8'", 'rug', 244, 76, 1, { tags: 'hallway' }],
  ],
  'Lights & plants': [
    ['floor-lamp', 'Floor lamp', 'floor-lamp', 40, 40, 160],
    ['arc-lamp', 'Arc floor lamp', 'floor-lamp', 100, 40, 200, { tags: 'arched' }],
    ['table-lamp', 'Table lamp', 'table-lamp', 30, 30, 55, { tags: 'bedside desk reading' }],
    ['pendant', 'Pendant light', 'pendant', 45, 45, 30, { tags: 'dining ceiling hanging' }],
    ['plant-lg', 'Large floor plant', 'plant', 60, 60, 160, { tags: 'fiddle leaf fig tree' }],
    ['plant-md', 'Medium plant', 'plant', 45, 45, 100],
  ],
  'Appliances & other': [
    ['fridge', 'Refrigerator', 'box', 91, 76, 178, { color: '#d9dcdf', tags: 'fridge kitchen' }],
    ['washer', 'Washer or dryer', 'box', 69, 76, 99, { color: '#e6e8ea', tags: 'laundry' }],
    ['piano', 'Upright piano', 'box', 150, 60, 125, { color: '#2b2522' }],
    ['tv-65', '65" TV on a stand', 'tv-stand', 145, 30, 90, { color: '#1f1f1f', tags: 'television' }],
  ],
}

export const STARTER_CATALOG: StarterProduct[] = Object.entries(groups).flatMap(([group, rows]) =>
  rows.map(([id, name, category, w, d, h, extra]) => ({ id, name, category, group, width: w / 100, depth: d / 100, height: h / 100, ...extra })),
)

export const STARTER_GROUPS = Object.keys(groups)

// "8' × 10'", "8 x 10" and "8x10" all search the same.
const simplify = (s: string) => s.toLowerCase().replace(/['"′″]/g, '').replace(/\s*[x×]\s*(?=\d)/g, 'x')

export function searchStarter(query: string, group?: string): StarterProduct[] {
  const words = simplify(query).split(/\s+/).filter(Boolean)
  return STARTER_CATALOG.filter((p) => {
    if (group && p.group !== group) return false
    const hay = simplify(`${p.name} ${p.category} ${p.group} ${p.tags ?? ''}`)
    return words.every((w) => hay.includes(w))
  })
}

export function itemFromStarter(p: StarterProduct, id: string): Item {
  const base = catalogEntry(p.category)
  return itemFromCatalog(base, id, { name: p.name, width: p.width, depth: p.depth, height: p.height, color: p.color ?? base.color })
}
