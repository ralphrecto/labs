import type { Units } from './types'

const IN = 0.0254
const FT = 0.3048

/** Format a length in meters for display. */
export function formatLength(m: number, units: Units, opts: { precise?: boolean } = {}): string {
  if (!Number.isFinite(m)) return '–'
  if (units === 'metric') {
    if (Math.abs(m) >= 1 && !opts.precise) return `${(m).toFixed(2)} m`
    return `${Math.round(m * 1000) / 10} cm`
  }
  const totalIn = m / IN
  const sign = totalIn < 0 ? '-' : ''
  const abs = Math.abs(totalIn)
  // Round to the nearest quarter inch.
  const q = Math.round(abs * 4) / 4
  const ft = Math.floor(q / 12)
  const inch = q - ft * 12
  const inStr = Number.isInteger(inch) ? `${inch}` : inch.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')
  if (ft === 0) return `${sign}${inStr}″`
  return `${sign}${ft}′ ${inStr}″`
}

/** A short form for inputs: "350" (cm) or "138" (in). */
export function toInputValue(m: number, units: Units): string {
  if (units === 'metric') return String(Math.round(m * 1000) / 10)
  const q = Math.round((m / IN) * 4) / 4
  if (Math.abs(q) < 24) return `${q}"`
  const ft = Math.floor(q / 12)
  const inch = q - ft * 12
  return inch ? `${ft}' ${inch}"` : `${ft}'`
}

export function unitLabel(units: Units) {
  return units === 'metric' ? 'cm' : ''
}

/**
 * Parse a typed length into meters. Bare numbers are centimeters (metric) or inches (imperial).
 * Accepts: 350, 350cm, 3.5m, 35mm, 138, 138in, 11'6", 11' 6, 11ft 6in, 11.5', 11 ft.
 */
export function parseLength(input: string, units: Units): number | null {
  const s = input.trim().toLowerCase().replace(/[’′]/g, "'").replace(/[”″]/g, '"').replace(/,/g, '.')
  if (!s) return null
  const num = '(-?\\d+(?:\\.\\d+)?|\\.\\d+)'
  let m: RegExpMatchArray | null
  if ((m = s.match(new RegExp(`^${num}\\s*mm$`)))) return +m[1] / 1000
  if ((m = s.match(new RegExp(`^${num}\\s*cm$`)))) return +m[1] / 100
  if ((m = s.match(new RegExp(`^${num}\\s*m$`)))) return +m[1]
  if ((m = s.match(new RegExp(`^${num}\\s*(?:in|"|inch|inches)$`)))) return +m[1] * IN
  if ((m = s.match(new RegExp(`^${num}\\s*(?:ft|'|feet|foot)\\s*(?:${num}\\s*(?:in|"|inch|inches)?)?$`)))) {
    return +m[1] * FT + (m[2] ? +m[2] * IN : 0)
  }
  if ((m = s.match(new RegExp(`^${num}$`)))) return units === 'metric' ? +m[1] / 100 : +m[1] * IN
  return null
}

const NUM_ONLY = /^\s*(-?\d+(?:[.,]\d+)?|[.,]\d+)\s*$/

/** What a bare number could mean, in meters, for each unit a player might have had in mind. */
function bareCandidates(n: number, units: Units | 'any'): { unit: string; m: number }[] {
  const metric = [
    { unit: 'm', m: n },
    { unit: 'cm', m: n / 100 },
  ]
  const imperial = [
    { unit: 'ft', m: n * FT },
    { unit: 'in', m: n * IN },
  ]
  return units === 'metric' ? metric : units === 'imperial' ? imperial : [...metric, { unit: 'mm', m: n / 1000 }, ...imperial]
}

const closest = <T extends { m: number }>(cands: T[], near: number): T =>
  cands.reduce((best, c) => (Math.abs(Math.log(c.m / near)) < Math.abs(Math.log(best.m / near)) ? c : best))

export interface LengthReading {
  /** Meters. */
  m: number
  /** True when the player typed a unit (m, cm, ', " ...); false when it was inferred. */
  explicit: boolean
}

/**
 * Parse a typed length when we already know roughly how long it should be (for example the same
 * length measured on a phone scan). A number with a unit is read as written. A bare number is
 * read in whichever unit of the chosen system (m or cm for metric, ft or in for imperial) puts
 * it closest to `near`, so "5.2" next to a 4.99 m scan length is 5.2 m and "17" is 17 ft.
 * Without a usable `near`, falls back to `parseLength`.
 */
export function parseLengthNear(input: string, units: Units, near: number): LengthReading | null {
  const bare = input.match(NUM_ONLY)
  if (!bare) {
    const m = parseLength(input, units)
    return m == null ? null : { m, explicit: true }
  }
  const n = +bare[1].replace(',', '.')
  if (!(near > 0) || !Number.isFinite(near) || !(n > 0)) {
    const m = parseLength(input, units)
    return m == null ? null : { m, explicit: false }
  }
  return { m: closest(bareCandidates(n, units), near).m, explicit: false }
}

/** True when a typed length would rescale a scan length by an unbelievable amount (outside about 0.3x to 3x). */
export function isWildScale(m: number, near: number, lo = 0.3, hi = 3): boolean {
  if (!(near > 0) || !(m > 0)) return false
  const r = m / near
  return r < lo || r > hi
}

/**
 * For a length that looks wildly off, what the player most likely meant: the same number in
 * the unit (any system) that lands closest to `near`, and which system that unit is in.
 * Null when no reading is believable.
 */
export function suggestLength(input: string, near: number): { m: number; units: Units } | null {
  const s = input.trim().toLowerCase().replace(/,/g, '.')
  const num = s.match(/^(\d+(?:\.\d+)?|\.\d+)/)
  if (!num || !(near > 0)) return null
  const n = +num[1]
  if (!(n > 0)) return null
  const best = closest(bareCandidates(n, 'any'), near)
  return isWildScale(best.m, near) ? null : { m: best.m, units: best.unit === 'ft' || best.unit === 'in' ? 'imperial' : 'metric' }
}

/** Several lengths in one consistent unit: "2.12 × 0.92 m", "212 × 92 cm" or "6′ 11″ × 3′". */
export function formatLengths(ms: number[], units: Units): string {
  if (!ms.length || ms.some((m) => !Number.isFinite(m))) return ms.map((m) => formatLength(m, units)).join(' × ')
  if (units === 'imperial') return ms.map((m) => formatLength(m, units)).join(' × ')
  const big = Math.max(...ms.map(Math.abs)) >= 1
  return big ? `${ms.map((m) => m.toFixed(2)).join(' × ')} m` : `${ms.map((m) => Math.round(m * 1000) / 10).join(' × ')} cm`
}
