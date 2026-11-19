// An imported item before it goes on the plan: what the importer found, with gaps filled from
// the category's typical size, for the user to check and edit.

import { catalogEntry, itemFromCatalog } from '../model/catalog'
import { uid } from '../model/sample'
import type { Item, ItemSource } from '../model/types'
import { guessCategory } from './dimensions'
import type { ProductInfo } from './productPage'

export interface Draft {
  source: ItemSource
  name: string
  category: string
  width: number
  depth: number
  height: number
  color: string
  price?: number
  link?: string
  imageUrl?: string
  glbUrl?: string
  useGlb: boolean
  owned: boolean
  estimated: boolean
  dimsText?: string
  notes: string[]
}

/** Turn what an importer found into an editable draft, filling gaps from the category's defaults. */
export function draftFromInfo(info: ProductInfo, source: ItemSource, extra: Partial<Draft> = {}): Draft {
  const category = info.category ?? guessCategory(info.name ?? '') ?? 'box'
  const base = catalogEntry(category)
  const notes = [...(info.notes ?? [])]
  const { width, depth, height } = info.dims
  const missing = (['width', 'depth', 'height'] as const).filter((k) => !info.dims[k])
  if (missing.length && missing.length < 3) notes.push(`No ${missing.join(' or ')} listed, so ${missing.length > 1 ? 'those use' : 'that uses'} a typical ${base.name.toLowerCase()}.`)
  if (missing.length === 3) notes.push(`No size found, so this starts at a typical ${base.name.toLowerCase()}. Enter the real size below.`)
  return {
    source,
    name: info.name?.trim() || base.name,
    category,
    width: width ?? base.width,
    depth: depth ?? base.depth,
    height: height ?? base.height,
    color: info.color && /^#[0-9a-f]{6}$/i.test(info.color) ? info.color : base.color,
    price: info.price,
    link: info.url,
    imageUrl: info.imageUrl,
    glbUrl: info.glbUrl,
    useGlb: !!info.glbUrl,
    owned: false,
    estimated: info.dimsSource !== 'listed' || missing.length > 0,
    dimsText: info.dimsSource === 'listed' ? info.dimsText : undefined,
    notes,
    ...extra,
  }
}

export function itemFromDraft(d: Draft): Item {
  const base = catalogEntry(d.category)
  return itemFromCatalog(base, uid(), {
    name: d.name,
    width: d.width,
    depth: d.depth,
    height: d.height,
    color: d.color,
    price: d.price,
    link: d.link,
    imageUrl: d.imageUrl,
    glbUrl: d.useGlb ? d.glbUrl : undefined,
    owned: d.owned,
    sizeEstimated: d.estimated,
    source: d.source,
  })
}
