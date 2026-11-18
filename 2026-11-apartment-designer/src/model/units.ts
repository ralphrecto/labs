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
