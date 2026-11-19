// Buy mode: the furniture drawer above the mode dock, and the "shop from a link or photo" sheet.
import { useEffect, useMemo, useRef, useState, type DragEvent } from 'react'
import { CATALOG, itemFromCatalog, type CatalogEntry } from '../model/catalog'
import { uid } from '../model/sample'
import type { Item, Units } from '../model/types'
import { formatLength } from '../model/units'
import { activeLayout, useStore } from '../store'
import { ImportDialog, Photo } from '../components/AddDialogs'
import { useGame } from './gameStore'
import { IconCamera, IconLink, IconStar } from './icons'
import { ItemThumb } from './ItemThumb'
import './buy.css'

export const CATEGORY_GROUPS: { key: string; label: string; emoji?: string; categories: string[] }[] = [
  { key: 'living', label: 'Living', emoji: '🛋️', categories: ['sofa', 'armchair', 'coffee-table', 'side-table', 'ottoman', 'tv-stand', 'console-table'] },
  { key: 'bedroom', label: 'Bedroom', emoji: '🛏️', categories: ['bed', 'nightstand', 'dresser', 'wardrobe'] },
  { key: 'dining', label: 'Dining', emoji: '🍽️', categories: ['dining-table', 'dining-chair', 'bench', 'sideboard'] },
  { key: 'work', label: 'Work', emoji: '💻', categories: ['desk', 'office-chair', 'bookshelf'] },
  { key: 'decor', label: 'Decor', emoji: '🪴', categories: ['rug', 'floor-lamp', 'plant', 'box'] },
  { key: 'mine', label: 'Mine', emoji: '⭐', categories: [] },
]

/** The category drawn on each tab chip. */
const TAB_ICON: Record<string, string> = { living: 'sofa', bedroom: 'bed', dining: 'dining-chair', work: 'desk', decor: 'plant' }

/** Mime type the stage reads on drop: the catalog category of the dragged card. */
export const CATALOG_DRAG_TYPE = 'application/x-apt-catalog'
/** Set alongside it when a "Mine" card is dragged: the id of the item to place another copy of. */
export const ITEM_DRAG_TYPE = 'application/x-apt-item'

const ENTRIES = new Map(CATALOG.map((e) => [e.category, e]))
/** "Box" reads as a generic custom piece in the drawer. */
const displayName = (e: CatalogEntry) => (e.category === 'box' ? 'Custom box' : e.name)

/** A compact footprint for a card: "210 × 92 cm" or "83 × 36 in". */
export function footprint(width: number, depth: number, units: Units): string {
  if (units === 'metric') {
    const n = (m: number) => formatLength(m, units, { precise: true }).replace(/\s*cm$/, '')
    return `${n(width)} × ${n(depth)} cm`
  }
  const n = (m: number) => Math.round(m / 0.0254)
  return `${n(width)} × ${n(depth)} in`
}

/** Items that are yours rather than generic catalog pieces: imported from a link or photo, scanned, or marked owned. */
export function isMine(i: Item): boolean {
  return !!(i.owned || i.link || i.imageUrl || i.scanId || i.source === 'link' || i.source === 'photo' || i.source === 'scan')
}

const article = (name: string) => (/^[aeiou]/i.test(name) ? 'an' : 'a')

export function BuyDrawer() {
  const tab = useGame((s) => s.buyTab)
  const setTab = useGame((s) => s.setBuyTab)
  const units = useStore((s) => s.doc.apartment.units)
  const items = useStore((s) => s.doc.items)
  const placements = useStore((s) => activeLayout(s.doc).placements)
  const mine = useMemo(() => items.filter(isMine), [items])
  const counts = useMemo(() => {
    const m = new Map<string, number>()
    for (const p of placements) m.set(p.itemId, (m.get(p.itemId) ?? 0) + 1)
    return m
  }, [placements])

  const group = CATEGORY_GROUPS.find((g) => g.key === tab) ?? CATEGORY_GROUPS[0]
  const [popped, setPopped] = useState<{ key: string; n: number } | null>(null)
  const popTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(popTimer.current), [])
  const celebrate = (key: string) => {
    setPopped((p) => ({ key, n: (p?.n ?? 0) + 1 }))
    clearTimeout(popTimer.current)
    popTimer.current = setTimeout(() => setPopped(null), 900)
  }

  const place = (entry: CatalogEntry, preset?: { name: string; width: number; depth: number }) => {
    const st = useStore.getState()
    if (preset) st.addItem(itemFromCatalog(entry, uid(), { name: `${preset.name} bed`, width: preset.width, depth: preset.depth }))
    else st.addItemFromCatalog(entry) // adds the item and a placement at the default drop point, and selects it
    const name = preset ? `${preset.name.toLowerCase()} bed` : displayName(entry).toLowerCase()
    st.notify(`Placed ${article(name)} ${name}. Drag it into place.`)
    celebrate(entry.category + (preset ? `:${preset.name}` : ''))
  }
  const placeMine = (item: Item) => {
    const st = useStore.getState()
    st.placeItem(item.id) // adds a placement of the same item and selects it
    st.notify(`Placed another “${item.name}”.`)
    celebrate(item.id)
  }
  const shop = () => useGame.getState().openSheet('shop')

  const onDrag = (e: DragEvent, category: string, itemId?: string) => {
    e.dataTransfer.effectAllowed = 'copy'
    e.dataTransfer.setData(CATALOG_DRAG_TYPE, category)
    if (itemId) e.dataTransfer.setData(ITEM_DRAG_TYPE, itemId)
    const thumb = (e.currentTarget as HTMLElement).querySelector('.buy-thumb')
    if (thumb) e.dataTransfer.setDragImage(thumb, 48, 48)
  }

  const isPopped = (key: string) => popped?.key === key

  return (
    <div className="buy-drawer" role="region" aria-label="Buy furniture">
      <div className="buy-tabs" role="tablist" aria-label="Furniture categories">
        {CATEGORY_GROUPS.map((g) => (
          <button key={g.key} role="tab" aria-selected={g.key === group.key} aria-label={g.label} className={`buy-tab ${g.key === group.key ? 'on' : ''}`} onClick={() => setTab(g.key)}>
            <span className="buy-tab-icon">
              {g.key === 'mine' ? <IconStar /> : <ItemThumb category={TAB_ICON[g.key] ?? g.categories[0]} color={ENTRIES.get(TAB_ICON[g.key] ?? g.categories[0])?.color ?? '#9aa4ad'} size={34} />}
              {g.key === 'mine' && mine.length > 0 && <span className="buy-badge">{mine.length}</span>}
            </span>
            <span className="buy-tab-label">{g.label}</span>
          </button>
        ))}
      </div>

      <div className="buy-grid" key={group.key} role="tabpanel" aria-label={group.label}>
        {group.key === 'mine' ? (
          <>
            <ShopCard onClick={shop} i={0} />
            {mine.length === 0 && (
              <div className="buy-card buy-empty" style={{ ['--i' as string]: 1 }}>
                <IconStar />
                <p>Pieces you add from a link, a photo or a scan live here.</p>
              </div>
            )}
            {mine.map((item, k) => (
              <div key={item.id} className={`buy-card ${isPopped(item.id) ? 'pop' : ''}`} style={{ ['--i' as string]: k + 1 }} draggable onDragStart={(e) => onDrag(e, item.category, item.id)}>
                <button className="buy-hit" onClick={() => placeMine(item)} aria-label={`Place another ${item.name}`}>
                  <span className="buy-thumb">
                    {item.imageUrl ? (
                      <Photo className="buy-photo" src={item.imageUrl} fallback={<ItemThumb category={item.category} color={item.color} size={78} />} />
                    ) : (
                      <ItemThumb category={item.category} color={item.color} size={78} />
                    )}
                    {counts.get(item.id) ? (
                      <span className="buy-count" aria-label={`${counts.get(item.id)} placed`}>
                        ×{counts.get(item.id)}
                      </span>
                    ) : null}
                    <span className="buy-tag">{item.scanId || item.source === 'scan' ? 'Scanned' : item.owned ? 'Owned' : item.price ? `$${Math.round(item.price)}` : 'Wishlist'}</span>
                  </span>
                  <span className="buy-name">{item.name}</span>
                  <span className="buy-size">{footprint(item.width, item.depth, units)}</span>
                </button>
                {isPopped(item.id) && <span key={popped?.n} className="buy-plus" aria-hidden>+1</span>}
              </div>
            ))}
          </>
        ) : (
          <>
            {group.categories.map((cat, k) => {
              const entry = ENTRIES.get(cat)
              if (!entry) return null
              const name = displayName(entry)
              return (
                <div key={cat} className={`buy-card ${isPopped(cat) || popped?.key.startsWith(cat + ':') ? 'pop' : ''}`} style={{ ['--i' as string]: k }} draggable onDragStart={(e) => onDrag(e, cat)}>
                  <button className="buy-hit" onClick={() => place(entry)} aria-label={`Place ${article(name)} ${name.toLowerCase()}`}>
                    <span className="buy-thumb">
                      <ItemThumb category={cat} color={entry.color} size={78} />
                    </span>
                    <span className="buy-name">{name}</span>
                    <span className="buy-size">{footprint(entry.width, entry.depth, units)}</span>
                  </button>
                  {entry.presets && (
                    <span className="buy-presets">
                      {entry.presets.map((pr) => (
                        <button key={pr.name} className="buy-preset" onClick={() => place(entry, pr)} aria-label={`Place a ${pr.name.toLowerCase()} bed, ${footprint(pr.width, pr.depth, units)}`} title={footprint(pr.width, pr.depth, units)}>
                          {pr.name}
                        </button>
                      ))}
                    </span>
                  )}
                  {(isPopped(cat) || popped?.key.startsWith(cat + ':')) && (
                    <span key={popped?.n} className="buy-plus" aria-hidden>
                      +1
                    </span>
                  )}
                </div>
              )
            })}
            <ShopCard onClick={shop} i={group.categories.length} />
          </>
        )}
      </div>
    </div>
  )
}

function ShopCard({ onClick, i }: { onClick: () => void; i: number }) {
  return (
    <div className="buy-card buy-shopcard" style={{ ['--i' as string]: i }}>
      <button className="buy-hit" onClick={onClick} aria-label="Shop from a link or photo">
        <span className="buy-thumb">
          <span className="buy-shop-icons">
            <IconLink />
            <IconCamera />
          </span>
        </span>
        <span className="buy-name">Shop from a link or photo</span>
      </button>
    </div>
  )
}

/** The import flow (link, photo, pasted text) dressed as a game sheet. Rendered when sheet === 'shop'. */
export function ShopSheet() {
  const close = () => useGame.getState().openSheet(null)
  return (
    <div className="buy-shop">
      <ImportDialog onClose={close} />
    </div>
  )
}
