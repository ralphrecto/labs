// Pulls a product's name, size, price, picture and 3D model out of a retailer page's HTML.
// Uses structured data first (schema.org Product in JSON-LD, Open Graph tags), then reads
// dimensions from the visible text. Shared by the browser and the import server.

import { guessCategory, normalizeText, parseDimensions, type Dims } from './dimensions.ts'

export interface ProductInfo {
  name?: string
  url?: string
  imageUrl?: string
  price?: number
  currency?: string
  brand?: string
  category?: string
  color?: string
  dims: Dims
  /** Where the size came from: listed on the page, or estimated from a photo or typical sizes. */
  dimsSource?: 'listed' | 'estimated'
  /** The text the size was read from, shown to the user so they can check it. */
  dimsText?: string
  /** A ready-made 3D model the retailer publishes (GLB), scaled to the listed size in the app. */
  glbUrl?: string
  notes?: string[]
}

const UNIT_CODES: Record<string, number> = { INH: 0.0254, CMT: 0.01, MMT: 0.001, MTR: 1, FOT: 0.3048 }

function absolute(u: string | undefined, base?: string): string | undefined {
  if (!u) return undefined
  try {
    return new URL(u.trim(), base).toString()
  } catch {
    return undefined
  }
}

function decodeEntities(s: string): string {
  return s
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(+n))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n, 16)))
    .replace(/&quot;/g, '"')
    .replace(/&apos;|&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
}

/** Visible text of a page, with block elements on their own lines. */
export function htmlToText(html: string): string {
  return decodeEntities(
    html
      .replace(/<(script|style|noscript|svg|template)[\s\S]*?<\/\1>/gi, ' ')
      .replace(/<!--[\s\S]*?-->/g, ' ')
      .replace(/<(br|\/p|\/div|\/li|\/tr|\/h\d|\/dd|\/table)[^>]*>/gi, '\n')
      // Keep a spec table's label and value on one line: "Overall Width: 64 in".
      .replace(/<\/(th|dt)>/gi, ': ')
      .replace(/<\/td>/gi, ' ')
      .replace(/<[^>]+>/g, ' '),
  )
    .replace(/[ \t\f\v]+/g, ' ')
    .replace(/\s*\n\s*/g, '\n')
    .replace(/\n{2,}/g, '\n')
    .trim()
}

function metaContent(html: string, key: string): string | undefined {
  const k = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const a = html.match(new RegExp(`<meta[^>]+(?:property|name|itemprop)=["']${k}["'][^>]*content=["']([^"']*)["']`, 'i'))
  const b = html.match(new RegExp(`<meta[^>]+content=["']([^"']*)["'][^>]*(?:property|name|itemprop)=["']${k}["']`, 'i'))
  const v = (a ?? b)?.[1]
  return v ? decodeEntities(v).trim() : undefined
}

type Json = Record<string, unknown>

function jsonLdBlocks(html: string): unknown[] {
  const out: unknown[] = []
  for (const m of html.matchAll(/<script[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)) {
    try {
      out.push(JSON.parse(m[1].trim()))
    } catch {
      // Some sites put several objects or trailing commas in one block; skip what doesn't parse.
    }
  }
  return out
}

function findProduct(node: unknown, depth = 0): Json | undefined {
  if (!node || typeof node !== 'object' || depth > 6) return undefined
  if (Array.isArray(node)) {
    for (const n of node) {
      const p = findProduct(n, depth + 1)
      if (p) return p
    }
    return undefined
  }
  const o = node as Json
  const type = o['@type']
  const types = Array.isArray(type) ? type : [type]
  if (types.some((t) => typeof t === 'string' && /^(Product|ProductGroup|IndividualProduct)$/i.test(t))) return o
  for (const key of ['@graph', 'mainEntity', 'itemListElement', 'item', 'hasVariant']) {
    const p = findProduct(o[key], depth + 1)
    if (p) return p
  }
  return undefined
}

const str = (v: unknown): string | undefined => (typeof v === 'string' ? v : typeof v === 'number' ? String(v) : undefined)

function imageOf(v: unknown): string | undefined {
  if (typeof v === 'string') return v
  if (Array.isArray(v)) return imageOf(v[0])
  if (v && typeof v === 'object') return str((v as Json).url) ?? str((v as Json).contentUrl)
  return undefined
}

function quantity(v: unknown): number | undefined {
  if (v == null) return undefined
  if (typeof v === 'number') return undefined // no unit, can't trust it
  if (typeof v === 'string') {
    const d = parseDimensions(`width ${v} height 1 cm`)
    return d?.width
  }
  if (typeof v === 'object') {
    const o = v as Json
    const value = Number(o.value)
    if (!Number.isFinite(value)) return undefined
    const code = str(o.unitCode)?.toUpperCase()
    const text = str(o.unitText)?.toLowerCase()
    const f = (code && UNIT_CODES[code]) ?? (text ? { in: 0.0254, inch: 0.0254, inches: 0.0254, cm: 0.01, mm: 0.001, m: 1, ft: 0.3048 }[text] : undefined)
    return f ? value * f : undefined
  }
  return undefined
}

function offerPrice(offers: unknown): { price?: number; currency?: string } {
  const o = (Array.isArray(offers) ? offers[0] : offers) as Json | undefined
  if (!o || typeof o !== 'object') return {}
  const raw = o.price ?? o.lowPrice ?? (o.priceSpecification as Json | undefined)?.price
  const price = raw != null ? Number(String(raw).replace(/[^\d.]/g, '')) : undefined
  return { price: Number.isFinite(price) && price! > 0 ? price : undefined, currency: str(o.priceCurrency) }
}

function findGlb(html: string, base?: string): string | undefined {
  const m = html.match(/(?:src|href|data-src|"url"|"glb"|"modelUrl")\s*[=:]\s*["']([^"'\s]+?\.glb(?:\?[^"'\s]*)?)["']/i) ?? html.match(/(https?:\/\/[^"'\s<>]+?\.glb)(?:[?"'\s<>])/i)
  return absolute(m?.[1]?.replace(/\\\//g, '/'), base)
}

/** Read everything useful from a product page. Fields that aren't found are left undefined. */
export function extractFromHtml(html: string, pageUrl?: string): ProductInfo {
  const notes: string[] = []
  const product = jsonLdBlocks(html).map((b) => findProduct(b)).find(Boolean)
  const info: ProductInfo = { url: pageUrl, dims: {}, notes }

  if (product) {
    info.name = str(product.name)
    info.imageUrl = absolute(imageOf(product.image), pageUrl)
    const brand = product.brand
    info.brand = typeof brand === 'string' ? brand : str((brand as Json | undefined)?.name)
    Object.assign(info, offerPrice(product.offers))
    const color = str(product.color)
    if (color) info.color = color
    const dims: Dims = { width: quantity(product.width), depth: quantity(product.depth), height: quantity(product.height) }
    const props = Array.isArray(product.additionalProperty) ? (product.additionalProperty as Json[]) : []
    const propText = props.map((p) => `${str(p.name) ?? ''}: ${str(p.value) ?? ''}${str(p.unitText) ? ' ' + str(p.unitText) : ''}`).join('\n')
    const fromProps = propText ? parseDimensions(propText) : null
    for (const k of ['width', 'depth', 'height'] as const) dims[k] ??= fromProps?.[k]
    if (dims.width || dims.depth) {
      info.dims = dims
      info.dimsSource = 'listed'
      info.dimsText = fromProps?.text ?? 'Product data on the page'
    }
    const desc = str(product.description)
    if (!info.dims.width && desc) {
      const d = parseDimensions(desc)
      if (d) {
        info.dims = { width: d.width, depth: d.depth, height: d.height }
        info.dimsSource = 'listed'
        info.dimsText = d.text
      }
    }
  }

  info.name ??= metaContent(html, 'og:title') ?? metaContent(html, 'twitter:title') ?? decodeEntities(html.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1] ?? '').trim()
  if (!info.name) info.name = undefined
  info.imageUrl ??= absolute(metaContent(html, 'og:image') ?? metaContent(html, 'twitter:image'), pageUrl)
  if (info.price == null) {
    const p = Number(metaContent(html, 'product:price:amount') ?? metaContent(html, 'og:price:amount') ?? metaContent(html, 'price'))
    if (Number.isFinite(p) && p > 0) info.price = p
    info.currency ??= metaContent(html, 'product:price:currency') ?? metaContent(html, 'og:price:currency')
  }
  info.glbUrl = findGlb(html, pageUrl)

  const text = htmlToText(html)
  info.category = guessCategory(`${info.name ?? ''}`) ?? guessCategory(text.slice(0, 2000))

  // Visible text, when structured data had no size or only part of it.
  const known = [info.dims.width, info.dims.depth, info.dims.height].filter(Boolean).length
  if (known < 3) {
    const d = parseDimensions(text, { category: info.category })
    const n = d ? [d.width, d.depth, d.height].filter(Boolean).length : 0
    if (d && n > known) {
      info.dims = { width: d.width, depth: d.depth, height: d.height }
      info.dimsSource = 'listed'
      info.dimsText = d.text
    }
  }
  if (!info.dims.width) notes.push("Couldn't find the size on the page.")
  return info
}

/** The page text worth sending to a language model: dimension-ish lines first, then the rest. */
export function pageDigest(html: string, maxChars = 12000): string {
  const text = normalizeText(htmlToText(html))
  const lines = text.split('\n').filter((l) => l.length > 1)
  const keyLines = lines.filter((l) => /\b(width|depth|height|length|dimension|size|overall|measure|\d\s*(cm|mm|in|")\b)/i.test(l))
  const head = keyLines.join('\n').slice(0, maxChars / 2)
  return `${head}\n---\n${lines.join('\n')}`.slice(0, maxChars)
}

/** True when the parser found enough that a model doesn't need to read the page. */
export function isComplete(info: ProductInfo): boolean {
  const { width, depth, height } = info.dims
  return !!(info.name && width && depth && height)
}

/**
 * Combine what the parser read with what a model read. Sizes the parser read from the page win
 * (they're exact quotes); the model fills the gaps and usually has the cleaner name.
 */
export function mergeInfo(parsed: ProductInfo, model: ProductInfo): ProductInfo {
  const parsedCount = [parsed.dims.width, parsed.dims.depth, parsed.dims.height].filter(Boolean).length
  const useParsedDims = parsed.dimsSource === 'listed' && parsedCount >= 2
  const dims = useParsedDims
    ? { width: parsed.dims.width ?? model.dims.width, depth: parsed.dims.depth ?? model.dims.depth, height: parsed.dims.height ?? model.dims.height }
    : model.dims
  return {
    ...parsed,
    name: model.name ?? parsed.name,
    category: model.category ?? parsed.category,
    color: model.color ?? parsed.color,
    price: parsed.price ?? model.price,
    currency: parsed.currency ?? model.currency,
    imageUrl: parsed.imageUrl ?? model.imageUrl,
    dims,
    dimsSource: useParsedDims ? 'listed' : model.dimsSource,
    dimsText: useParsedDims ? parsed.dimsText : model.dimsText,
    notes: [...(model.notes ?? [])],
  }
}

/** A readable name from a product URL's slug, for when the page itself can't be read. */
export function nameFromUrl(raw: string): string | undefined {
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    return undefined
  }
  const segments = url.pathname.split('/').filter(Boolean).map((s) => decodeURIComponent(s).replace(/\.(html?|php|aspx?)$/i, ''))
  // The slug is the longest segment made of words, skipping ids and short paths like /us/en/p/.
  const words = (s: string) => s.split(/[-_+]+/).filter((w) => (/[a-z]/i.test(w) || /^\d{1,3}$/.test(w)) && !/^[a-z]?\d{4,}[a-z\d]*$/i.test(w))
  const best = segments.map(words).filter((w) => w.length >= 2).sort((a, b) => b.length - a.length)[0]
  if (!best) return undefined
  const name = best.join(' ')
  return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase()
}
