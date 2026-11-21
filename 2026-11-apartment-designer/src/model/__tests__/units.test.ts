import { describe, expect, it } from 'vitest'
import { formatLengths, isWildScale, parseLength, parseLengthNear, suggestLength } from '../units'

describe('parseLengthNear', () => {
  const scan = 4.99 // meters measured on the scan

  it('reads a bare metric number as m or cm, whichever is closer to the scan', () => {
    expect(parseLengthNear('5.2', 'metric', scan)).toEqual({ m: 5.2, explicit: false })
    expect(parseLengthNear('520', 'metric', scan)!.m).toBeCloseTo(5.2)
    expect(parseLengthNear('5,2', 'metric', scan)!.m).toBeCloseTo(5.2)
    expect(parseLengthNear('92', 'metric', 0.9)!.m).toBeCloseTo(0.92)
    expect(parseLengthNear('1.5', 'metric', 1.4)!.m).toBeCloseTo(1.5)
  })

  it('reads a bare imperial number as ft or in, whichever is closer', () => {
    expect(parseLengthNear('17', 'imperial', scan)!.m).toBeCloseTo(17 * 0.3048)
    expect(parseLengthNear('200', 'imperial', scan)!.m).toBeCloseTo(200 * 0.0254)
    expect(parseLengthNear('36', 'imperial', 0.9)!.m).toBeCloseTo(36 * 0.0254)
  })

  it('keeps a typed unit as written', () => {
    expect(parseLengthNear('5.2cm', 'metric', scan)).toMatchObject({ m: expect.closeTo(0.052), explicit: true })
    expect(parseLengthNear('5.2 m', 'imperial', scan)!.m).toBeCloseTo(5.2)
    expect(parseLengthNear("17'", 'imperial', scan)!.m).toBeCloseTo(17 * 0.3048)
    expect(parseLengthNear(`13'6"`, 'imperial', scan)!.m).toBeCloseTo(13.5 * 0.3048)
    expect(parseLengthNear(`13′ 6″`, 'metric', scan)!.m).toBeCloseTo(13.5 * 0.3048)
    expect(parseLengthNear('17 ft', 'metric', scan)!.explicit).toBe(true)
  })

  it('returns null for things that are not lengths', () => {
    expect(parseLengthNear('abc', 'metric', scan)).toBeNull()
    expect(parseLengthNear('', 'metric', scan)).toBeNull()
    expect(parseLengthNear('5..2', 'metric', scan)).toBeNull()
  })

  it('falls back to the editor rule without a scan length', () => {
    expect(parseLengthNear('350', 'metric', 0)!.m).toBeCloseTo(parseLength('350', 'metric')!)
    expect(parseLengthNear('138', 'imperial', NaN)!.m).toBeCloseTo(parseLength('138', 'imperial')!)
  })
})

describe('scale guard', () => {
  it('flags lengths far from the scan', () => {
    expect(isWildScale(0.052, 4.99)).toBe(true)
    expect(isWildScale(52, 4.99)).toBe(true)
    expect(isWildScale(5.2, 4.99)).toBe(false)
    expect(isWildScale(2, 4.99)).toBe(false)
  })

  it('suggests the unit the player probably meant', () => {
    expect(suggestLength('5.2cm', 4.99)).toEqual({ m: 5.2, units: 'metric' })
    expect(suggestLength('520 mm', 4.99)!.m).toBeCloseTo(5.2)
    expect(suggestLength('17 in', 4.99)).toMatchObject({ m: expect.closeTo(17 * 0.3048), units: 'imperial' })
    expect(suggestLength('17', 4.99)!.units).toBe('imperial')
    expect(suggestLength('abc', 4.99)).toBeNull()
    expect(suggestLength('90000', 4.99)).toBeNull()
  })
})

describe('formatLengths', () => {
  it('uses one unit for every dimension', () => {
    expect(formatLengths([2.12, 0.915], 'metric')).toBe('2.12 × 0.92 m')
    expect(formatLengths([0.6, 0.45], 'metric')).toBe('60 × 45 cm')
    expect(formatLengths([1.8288, 0.9144], 'imperial')).toBe('6′ 0″ × 3′ 0″')
  })
})
