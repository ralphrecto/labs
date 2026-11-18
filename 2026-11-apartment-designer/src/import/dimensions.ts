// Reads furniture dimensions out of free text: product pages, spec sheets, pasted listings.
// Everything comes back in meters. Shared by the browser and the import server, so this file
// has no imports.

export interface Dims {
  width?: number
  depth?: number
  height?: number
}

export interface DimsMatch extends Dims {
  /** The text the dimensions were read from. */
  text: string
  /** Rough confidence from 0 to 1: labeled values with units score highest. */
  score: number
}

type Unit = 'mm' | 'cm' | 'm' | 'in' | 'ft'
type Axis = 'w' | 'd' | 'h' | 'l'

const TO_M: Record<Unit, number> = { mm: 0.001, cm: 0.01, m: 1, in: 0.0254, ft: 0.3048 }

/** Normalize quotes, fractions and separators so the patterns below stay simple. */
export function normalizeText(s: string): string {
  return s
    .replace(/&quot;|&#34;|&#x22;/gi, '"')
    .replace(/&#39;|&#x27;|&apos;/gi, "'")
    .replace(/&times;|&#215;|&#xd7;/gi, ' x ')
    .replace(/&nbsp;|&#160;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/[‘’′]/g, "'")
    .replace(/[“”″]|''/g, '"')
    .replace(/[×✕✖]/g, ' x ')
    .replace(/(\d)\s*½/g, '$1.5')
    .replace(/(\d)\s*¼/g, '$1.25')
    .replace(/(\d)\s*¾/g, '$1.75')
    .replace(/(\d+)\s+(\d)\/(\d{1,2})(?!\d)/g, (_, a, n, d) => String(+a + +n / +d))
    .replace(/(\d),(\d{3})(?!\d)/g, '$1$2') // 1,200 mm
    .replace(/(\d),(\d{1,2})(?!\d)/g, '$1.$2') // 85,5 cm
    .replace(/[ \t]+/g, ' ')
}

const NUM = String.raw`(\d+(?:\.\d+)?|\.\d+)`
const UNIT = String.raw`(mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m(?![a-z])|met(?:er|re)s?|in(?:ch(?:es)?)?\.?|"|ft\.?|feet|foot|')`

function unitOf(raw: string | undefined): Unit | undefined {
  if (!raw) return undefined
  const u = raw.toLowerCase().replace(/\.$/, '')
  if (u.startsWith('mm') || u.startsWith('milli')) return 'mm'
  if (u.startsWith('cm') || u.startsWith('centi')) return 'cm'
  if (u === 'm' || u.startsWith('met')) return 'm'
  if (u.startsWith('in') || u === '"') return 'in'
  if (u.startsWith('f') || u === "'") return 'ft'
  return undefined
}

function axisOf(raw: string): Axis | undefined {
  const a = raw.toLowerCase()
  if (/^w/.test(a)) return 'w'
  if (/^d/.test(a)) return 'd'
  if (/^(h|t)/.test(a)) return 'h'
  if (/^l/.test(a)) return 'l'
  return undefined
}

const plausible = (m: number) => m >= 0.01 && m <= 8

/**
 * A length and its unit, as written: `84"`, `213 cm`, `6' 8"`, `6 ft 8 in`.
 * Returns the value in meters, or the bare number when there's no unit.
 */
function readLength(s: string): { value: number; unit?: Unit } | null {
  const ftIn = s.match(new RegExp(String.raw`^\s*${NUM}\s*(?:'|ft\.?|feet|foot)\s*${NUM}\s*(?:"|in(?:ch(?:es)?)?\.?)?\s*$`, 'i'))
  if (ftIn) return { value: +ftIn[1] * TO_M.ft + +ftIn[2] * TO_M.in, unit: 'in' }
  const m = s.match(new RegExp(String.raw`^\s*${NUM}\s*${UNIT}?\s*$`, 'i'))
  if (!m) return null
  const unit = unitOf(m[2])
  return { value: unit ? +m[1] * TO_M[unit] : +m[1], unit }
}

/** Guess a unit for bare numbers from the rest of the text. */
function guessUnit(values: number[], context: string): Unit {
  const ctx = context.toLowerCase()
  const metric = (ctx.match(/\d\s*(cm|mm)\b/g) ?? []).length
  const imperial = (ctx.match(/\d\s*("|in\b|inch|inches)/g) ?? []).length
  if (Math.max(...values) > 400) return 'mm'
  if (metric > imperial) return Math.max(...values) > 300 ? 'mm' : 'cm'
  if (imperial > metric) return 'in'
  // No hints. Furniture over 100 units on its longest side is almost always centimeters.
  return Math.max(...values) > 100 ? 'cm' : 'in'
}

/** Words that mean "this is a part of the item, not the whole thing". */
const PART_WORDS = /(seat|arm|leg|back|cushion|drawer|shelf|shelves|interior|inside|opening|clearance|package|packaged|box|carton|shipping|mattress|headboard|footboard|top thickness|tabletop thickness|cord|shade|base)\s*$/i

function assign(axes: Partial<Record<Axis, number>>, category?: string): Dims {
  const { w, d, h, l } = axes
  const out: Dims = { height: h }
  if (l != null && w != null && d == null) {
    // "L x W": for beds the length runs front to back; for tables and rugs it's the long side.
    if (category === 'bed') {
      out.width = w
      out.depth = l
    } else {
      out.width = Math.max(l, w)
      out.depth = Math.min(l, w)
    }
  } else {
    out.width = w ?? l
    out.depth = d ?? (w != null ? l : undefined)
  }
  return out
}

const count = (d: Dims) => [d.width, d.depth, d.height].filter((x) => x != null).length

/**
 * Inline expressions like `84"W x 38"D x 34"H`, `W 213 x D 96 x H 85 cm`,
 * `32.25" H x 84" W x 38" D`, or unlabeled `213 x 96 x 85 cm` (read as width, depth, height).
 */
function inlineMatches(text: string, category?: string): DimsMatch[] {
  const out: DimsMatch[] = []
  const LABEL = String.raw`(w(?:idth|ide)?|d(?:epth|eep)?|h(?:eight|igh)?|l(?:ength|ong)?|t(?:all)?)\.?`
  // A part: optional label, number, optional unit (or ft-in), optional label.
  const PART = String.raw`(?:${LABEL}\s*:?\s*)?${NUM}\s*(?:${UNIT})?(?:\s*${NUM}\s*(?:"|in(?:ch(?:es)?)?\.?))?(?:\s*${LABEL}(?![a-z]))?`
  const SEP = String.raw`\s*(?:x|by|\*|/)\s*`
  const re = new RegExp(String.raw`(?<![\w.])${PART}(?:${SEP}${PART}){1,2}(?:\s*${UNIT})?(?![\w])`, 'gi')
  for (const m of text.matchAll(re)) {
    const raw = m[0]
    const before = text.slice(Math.max(0, (m.index ?? 0) - 30), m.index)
    if (PART_WORDS.test(before)) continue
    const parts = raw.split(new RegExp(SEP, 'i'))
    // A unit written once at the end applies to every part ("213 x 96 x 85 cm").
    const tail = raw.match(new RegExp(String.raw`${UNIT}\s*$`, 'i'))
    const tailUnit = tail ? unitOf(tail[1]) : undefined
    const vals: { value: number; unit?: Unit; axis?: Axis }[] = []
    for (const p of parts) {
      const lead = p.match(new RegExp(String.raw`^\s*${LABEL}\s*:?\s*(?=\d|\.)`, 'i'))
      const trail = p.match(new RegExp(String.raw`(?<=[\d"'.a-z ])\s*${LABEL}\s*$`, 'i'))
      let body = p
      if (lead) body = body.slice(lead[0].length)
      if (trail && (!lead || trail.index! > 0)) {
        // Don't treat a unit like "in" or "m" as an axis label.
        const t = trail[1].toLowerCase()
        if (!/^(in|m|mm|cm|ft)$/.test(t)) body = body.slice(0, trail.index)
      }
      const len = readLength(body)
      if (!len) {
        vals.length = 0
        break
      }
      const axisRaw = lead?.[1] ?? (trail && !/^(in|m|mm|cm|ft)$/i.test(trail[1]) ? trail[1] : undefined)
      vals.push({ ...len, axis: axisRaw ? axisOf(axisRaw) : undefined })
    }
    if (vals.length < 2) continue
    const labeled = vals.every((v) => v.axis)
    const anyUnit = vals.some((v) => v.unit) || !!tailUnit
    // Unlabeled pairs are too ambiguous (could be a price range or a ratio) unless they have units.
    if (!labeled && vals.length === 2 && !anyUnit) continue
    const unitless = vals.filter((v) => !v.unit)
    const fallback = tailUnit ?? (unitless.length ? guessUnit(unitless.map((v) => v.value), text) : 'cm')
    const meters = vals.map((v) => (v.unit ? v.value : v.value * TO_M[fallback]))
    if (!meters.every(plausible)) continue
    let dims: Dims
    if (labeled) {
      const axes: Partial<Record<Axis, number>> = {}
      vals.forEach((v, i) => (axes[v.axis!] ??= meters[i]))
      dims = assign(axes, category)
    } else if (vals.length === 3) {
      dims = { width: meters[0], depth: meters[1], height: meters[2] }
    } else {
      // Two unlabeled values with units: a footprint, longer side as width (rugs, tables).
      dims = { width: Math.max(meters[0], meters[1]), depth: Math.min(meters[0], meters[1]) }
    }
    const score = (labeled ? 0.6 : 0.35) + (anyUnit ? 0.25 : 0) + (count(dims) === 3 ? 0.1 : 0) + (/overall|dimension|size/i.test(before) ? 0.05 : 0)
    out.push({ ...dims, text: raw.trim(), score: Math.min(1, score) })
  }
  return out
}

/** Spec-sheet style lines: `Overall Width - Side to Side: 84''`, `Depth: 96 cm`, `Height 34 in`. */
function labeledLines(text: string, category?: string): DimsMatch | null {
  const re = new RegExp(String.raw`(overall\s+|product\s+|total\s+|assembled\s+)?(width|depth|height|length)\b([^\d\n]{0,40}?)${NUM}\s*(${UNIT})?(?:\s*${NUM}\s*(?:"|in(?:ch(?:es)?)?\.?))?`, 'gi')
  const best: Partial<Record<Axis, { value: number; unit?: Unit; overall: boolean; raw: string }>> = {}
  for (const m of text.matchAll(re)) {
    const before = text.slice(Math.max(0, (m.index ?? 0) - 25), m.index)
    const between = m[3] ?? ''
    const qualifier = between.replace(/side to side|front to back|top to bottom/gi, '')
    if (PART_WORDS.test(before) || /(seat|arm|leg|back|min|max|adjust|package|box|shipping|drawer|shelf|interior|cushion)/i.test(qualifier)) continue
    const axis = axisOf(m[2])!
    const unit = unitOf(m[5])
    let value = +m[4]
    if (m[6] && unit === 'ft') value = value * TO_M.ft + +m[6] * TO_M.in
    else if (unit) value *= TO_M[unit]
    const overall = !!m[1] || /overall|side to side|front to back|top to bottom/i.test(between)
    const prev = best[axis]
    if (!prev || (overall && !prev.overall)) best[axis] = { value, unit: m[6] && unit === 'ft' ? 'in' : unit, overall, raw: m[0] }
  }
  const found = Object.values(best)
  if (found.length < 2) return null
  const unitless = found.filter((f) => !f.unit)
  const fallback = unitless.length ? guessUnit(unitless.map((f) => f.value), text) : 'cm'
  const axes: Partial<Record<Axis, number>> = {}
  for (const [k, f] of Object.entries(best)) axes[k as Axis] = f.unit ? f.value : f.value * TO_M[fallback]
  if (!Object.values(axes).every((x) => plausible(x!))) return null
  const dims = assign(axes, category)
  const score = 0.55 + (unitless.length ? 0 : 0.25) + (count(dims) === 3 ? 0.1 : 0) + (found.some((f) => f.overall) ? 0.05 : 0)
  return { ...dims, text: found.map((f) => f.raw.trim()).join('; '), score: Math.min(1, score) }
}

/** Find the most likely overall dimensions in a block of text. */
export function parseDimensions(input: string, opts: { category?: string } = {}): DimsMatch | null {
  const text = normalizeText(input)
  const candidates = [...inlineMatches(text, opts.category)]
  const lines = labeledLines(text, opts.category)
  if (lines) candidates.push(lines)
  if (!candidates.length) return null
  // Highest score wins; ties go to the first one on the page (usually the overall size).
  return candidates.reduce((a, b) => (b.score > a.score ? b : a))
}

/** Map a product name or description to one of the app's furniture categories. */
export function guessCategory(text: string): string | undefined {
  const t = text.toLowerCase()
  const rules: [RegExp, string][] = [
    [/\b(sofa|console|entryway|hall)\s+table\b|\bconsole\b/, 'console-table'],
    [/\b(coffee|cocktail)\s+table\b/, 'coffee-table'],
    [/\b(side|end|accent|bedside)\s+table\b/, 'side-table'],
    [/\bnight\s?stand\b|\bbedside\b/, 'nightstand'],
    [/\b(tv|media|entertainment)\s*(stand|console|unit|center|cabinet)\b/, 'tv-stand'],
    [/\b(dining|kitchen|breakfast)\s+table\b|\bdining set\b/, 'dining-table'],
    [/\b(office|desk|task|gaming)\s+chair\b/, 'office-chair'],
    [/\b(dining|kitchen|side)\s+chair\b|\bstool\b/, 'dining-chair'],
    [/\b(sectional|sofa|couch|loveseat|settee|futon|chaise)\b/, 'sofa'],
    [/\b(armchair|accent chair|lounge chair|recliner|club chair|wingback|rocking chair|swivel chair|chair)\b/, 'armchair'],
    [/\b(ottoman|pouf|pouffe|footstool)\b/, 'ottoman'],
    [/\bbench\b/, 'bench'],
    [/\b(bed|bed frame|headboard|daybed|bunk|crib)\b/, 'bed'],
    [/\b(wardrobe|armoire|closet)\b/, 'wardrobe'],
    [/\b(dresser|chest of drawers|drawer chest|tallboy)\b/, 'dresser'],
    [/\b(sideboard|buffet|credenza|cabinet)\b/, 'sideboard'],
    [/\b(bookcase|bookshelf|shelving|shelf unit|etagere|étagère)\b/, 'bookshelf'],
    [/\bdesk\b/, 'desk'],
    [/\btable\b/, 'dining-table'],
    [/\b(rug|carpet|runner|mat)\b/, 'rug'],
    [/\b(lamp|floor light|torchiere)\b/, 'floor-lamp'],
    [/\b(plant|planter|tree|ficus|monstera)\b/, 'plant'],
  ]
  return rules.find(([re]) => re.test(t))?.[1]
}
