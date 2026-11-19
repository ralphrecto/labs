import { describe, expect, it } from 'vitest'
import { CATALOG } from '../../model/catalog'
import { STARTER_CATALOG, searchStarter } from '../../model/starterCatalog'
import { guessCategory, parseDimensions } from '../dimensions'
import { draftFromInfo, itemFromDraft } from '../draft'
import { extractFromHtml, mergeInfo, nameFromUrl } from '../productPage'

const IN = 0.0254
const cm = (x: number | undefined) => (x == null ? undefined : Math.round(x * 1000) / 10)
const dims = (s: string, category?: string) => {
  const d = parseDimensions(s, { category })
  return d && [cm(d.width), cm(d.depth), cm(d.height)]
}

describe('parseDimensions', () => {
  it('reads labeled inline sizes in any order', () => {
    expect(dims('84"W x 38"D x 34"H')).toEqual([213.4, 96.5, 86.4])
    expect(dims('Overall: 32.25" H x 84" W x 38" D')).toEqual([213.4, 96.5, 81.9])
    expect(dims('Dimensions: W 213 x D 96 x H 85 cm')).toEqual([213, 96, 85])
    expect(parseDimensions('72 1/2" W x 18" D x 30" H')?.width).toBeCloseTo(72.5 * IN)
  })
  it('reads unlabeled width x depth x height with a trailing unit', () => {
    expect(dims('Size 213 x 96 x 85 cm')).toEqual([213, 96, 85])
    expect(dims('Product size: 1200 x 600 x 750 mm')).toEqual([120, 60, 75])
    expect(dims('84 x 38 x 34 in.')).toEqual([213.4, 96.5, 86.4])
  })
  it('reads spec-sheet lines and skips seat and package sizes', () => {
    const wayfair = "Seat Height - Floor to Seat: 18''\nOverall Width - Side to Side: 84''\nOverall Depth - Front to Back: 38''\nOverall Height - Top to Bottom: 34''\nPackage Width: 90''"
    expect(dims(wayfair)).toEqual([213.4, 96.5, 86.4])
    expect(dims('Seat height 45 cm. Width 180 cm, Depth 85 cm, Height 80 cm')).toEqual([180, 85, 80])
  })
  it('handles feet and inches, and rug footprints', () => {
    expect(dims(`Measures 6' 8" W x 3' 2" D`)).toEqual([203.2, 96.5, undefined])
    expect(dims("8' x 10' area rug")).toEqual([304.8, 243.8, undefined])
  })
  it('maps length to depth for beds and to width for tables', () => {
    expect(dims('Length 84 in x Width 64 in x Height 40 in', 'bed')).toEqual([162.6, 213.4, 101.6])
    expect(dims('L 180 x W 90 x H 75 cm', 'dining-table')).toEqual([180, 90, 75])
  })
  it('ignores things that are not sizes', () => {
    expect(parseDimensions('Price $199 - $299')).toBeNull()
    expect(parseDimensions('Ships in 3 x 2 boxes')).toBeNull()
  })
})

describe('guessCategory', () => {
  it('maps product names to app categories', () => {
    expect(guessCategory('Harbor 3-Seat Sofa')).toBe('sofa')
    expect(guessCategory('Walnut coffee table')).toBe('coffee-table')
    expect(guessCategory('Mid-century accent chair')).toBe('armchair')
    expect(guessCategory('Queen platform bed frame')).toBe('bed')
    expect(guessCategory('Oak sofa table')).toBe('console-table')
    expect(guessCategory('Wool rug 8x10')).toBe('rug')
    expect(guessCategory('Ergonomic office chair')).toBe('office-chair')
    expect(guessCategory('Something else')).toBeUndefined()
  })
})

const jsonLdPage = `<!doctype html><html><head><title>Harbor Sofa | Example Store</title>
<meta property="og:image" content="https://cdn.example.com/og.jpg">
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"BreadcrumbList"},{"@type":"Product","name":"Harbor 3-Seat Sofa","image":["https://cdn.example.com/harbor.jpg"],"brand":{"@type":"Brand","name":"Example"},
"offers":{"@type":"Offer","price":"1299.00","priceCurrency":"USD"},
"width":{"@type":"QuantitativeValue","value":84,"unitCode":"INH"},"depth":{"@type":"QuantitativeValue","value":38,"unitCode":"INH"},"height":{"@type":"QuantitativeValue","value":34,"unitCode":"INH"}}]}</script>
</head><body><h1>Harbor 3-Seat Sofa</h1><p>Seat height: 18"</p></body></html>`

const specPage = `<html><head><meta property="og:title" content="Linnea Queen Bed Frame">
<meta property="product:price:amount" content="649"></head><body>
<h1>Linnea Queen Bed Frame</h1><p>A low platform bed in solid oak.</p>
<model-viewer src="/assets/models/linnea-queen.glb" ar></model-viewer>
<table><tr><th>Overall Width</th><td>64&quot;</td></tr><tr><th>Overall Length</th><td>86&quot;</td></tr><tr><th>Overall Height</th><td>14&quot;</td></tr><tr><th>Headboard Height</th><td>40&quot;</td></tr></table>
</body></html>`

describe('extractFromHtml', () => {
  it('reads schema.org product data', () => {
    const info = extractFromHtml(jsonLdPage, 'https://www.example.com/p/harbor')
    expect(info.name).toBe('Harbor 3-Seat Sofa')
    expect(info.price).toBe(1299)
    expect(info.currency).toBe('USD')
    expect(info.brand).toBe('Example')
    expect(info.imageUrl).toBe('https://cdn.example.com/harbor.jpg')
    expect(info.category).toBe('sofa')
    expect(info.dimsSource).toBe('listed')
    expect(info.dims.width).toBeCloseTo(84 * IN)
    expect(info.dims.depth).toBeCloseTo(38 * IN)
    expect(info.dims.height).toBeCloseTo(34 * IN)
  })
  it('reads a spec table, the price meta tag and a published 3D model', () => {
    const info = extractFromHtml(specPage, 'https://shop.example.com/beds/linnea')
    expect(info.name).toBe('Linnea Queen Bed Frame')
    expect(info.price).toBe(649)
    expect(info.category).toBe('bed')
    expect(info.glbUrl).toBe('https://shop.example.com/assets/models/linnea-queen.glb')
    expect(cm(info.dims.width)).toBe(162.6)
    expect(cm(info.dims.depth)).toBe(218.4)
    expect(cm(info.dims.height)).toBe(35.6)
  })
  it('says when there is no size', () => {
    const info = extractFromHtml('<title>Nice lamp</title><p>Lovely.</p>')
    expect(info.name).toBe('Nice lamp')
    expect(info.dims.width).toBeUndefined()
    expect(info.notes?.length).toBe(1)
  })
})

describe('nameFromUrl', () => {
  it('turns a product slug into a name', () => {
    expect(nameFromUrl('https://www.example.com/us/en/p/kivik-sofa-tibbleby-beige-gray-s69440533/')).toBe('Kivik sofa tibbleby beige gray')
    expect(nameFromUrl('https://shop.example.com/products/harbor-3-seat-sofa.html?variant=12')).toBe('Harbor 3 seat sofa')
    expect(nameFromUrl('not a url')).toBeUndefined()
  })
})

describe('mergeInfo', () => {
  it('keeps listed sizes from the parser and fills gaps from the model', () => {
    const parsed = { name: 'Harbor Sofa | Store', dims: { width: 2.13, depth: 0.96 }, dimsSource: 'listed' as const, dimsText: '84"W x 38"D', price: 999 }
    const model = { name: 'Harbor Sofa', category: 'sofa', dims: { width: 2.2, depth: 1, height: 0.86 }, dimsSource: 'listed' as const, price: 1 }
    const m = mergeInfo(parsed, model)
    expect(m.name).toBe('Harbor Sofa')
    expect(m.dims).toEqual({ width: 2.13, depth: 0.96, height: 0.86 })
    expect(m.price).toBe(999)
    expect(m.dimsText).toBe('84"W x 38"D')
  })
  it('uses the model size when the parser found nothing', () => {
    const m = mergeInfo({ dims: {} }, { name: 'Chair', dims: { width: 0.8, depth: 0.8, height: 0.9 }, dimsSource: 'estimated' })
    expect(m.dims.width).toBe(0.8)
    expect(m.dimsSource).toBe('estimated')
  })
})

describe('drafts', () => {
  it('fills missing sizes from the category and marks them estimated', () => {
    const d = draftFromInfo({ name: 'Oak desk', dims: { width: 1.5 }, dimsSource: 'listed' }, 'link')
    expect(d.category).toBe('desk')
    expect(d.width).toBe(1.5)
    expect(d.depth).toBe(0.7)
    expect(d.estimated).toBe(true)
    expect(d.notes.join(' ')).toMatch(/No depth or height listed/)
  })
  it('makes an item with the category clearances and the import source', () => {
    const d = draftFromInfo({ name: 'Harbor sofa', dims: { width: 2.1, depth: 0.95, height: 0.85 }, dimsSource: 'listed', glbUrl: 'https://x/y.glb', price: 999 }, 'link')
    expect(d.estimated).toBe(false)
    const item = itemFromDraft(d)
    expect(item.source).toBe('link')
    expect(item.clearance.front).toBe(0.45)
    expect(item.glbUrl).toBe('https://x/y.glb')
    expect(item.price).toBe(999)
    expect(item.sizeEstimated).toBe(false)
  })
})

describe('starter catalog', () => {
  it('uses known categories, unique ids and sensible sizes', () => {
    const cats = new Set(CATALOG.map((c) => c.category))
    const ids = new Set(STARTER_CATALOG.map((p) => p.id))
    expect(ids.size).toBe(STARTER_CATALOG.length)
    for (const p of STARTER_CATALOG) {
      expect(cats.has(p.category), p.id).toBe(true)
      expect(p.width).toBeGreaterThan(0.2)
      expect(p.width).toBeLessThan(4)
      expect(p.height).toBeLessThan(2.5)
    }
  })
  it('searches by name and tags', () => {
    expect(searchStarter('queen').map((p) => p.id)).toContain('bed-queen')
    expect(searchStarter("8x10").map((p) => p.id)).toEqual(['rug-8x10'])
    expect(searchStarter("8' x 10'").map((p) => p.id)).toEqual(['rug-8x10'])
    expect(searchStarter('bar stool').map((p) => p.id)).toEqual(['counter-stool'])
  })
})
