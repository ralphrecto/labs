// Buy mode: the furniture drawer above the mode dock, and the "shop from a link or photo" sheet.
import { useEffect, useMemo, useRef, useState, type DragEvent, type ReactNode } from 'react'
import { CATALOG, itemFromCatalog, type CatalogEntry } from '../model/catalog'
import { checkLayout } from '../model/checks'
import { centroid } from '../model/geometry'
import { findDrop, type DropResult } from '../model/placement'
import { STARTER_CATALOG } from '../model/starterCatalog'
import { uid } from '../model/sample'
import type { Doc, Item, Room, Units } from '../model/types'
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
  { key: 'decor', label: 'Decor', emoji: '🪴', categories: ['plant', 'floor-lamp', 'rug', 'box'] },
  { key: 'mine', label: 'Mine', emoji: '⭐', categories: [] },
]

/** The category drawn on each tab chip. */
const TAB_ICON: Record<string, string> = { living: 'sofa', bedroom: 'bed', dining: 'dining-chair', work: 'desk', decor: 'plant' }

/** Mime type the stage reads on drop: the catalog category of the dragged card. */
export const CATALOG_DRAG_TYPE = 'application/x-apt-catalog'
/** Set alongside it when a "Mine" card is dragged: the id of the item to place another copy of. */
export const ITEM_DRAG_TYPE = 'application/x-apt-item'

const ENTRIES = new Map(CATALOG.map((e) => [e.category, e]))

/** A size on a card's chip row: a short label and the piece it places. */
interface SizeChip {
  label: string
  /** Name of the placed piece ("Loveseat", "Rug 8' × 10'"). */
  name: string
  width: number
  depth: number
  height?: number
}

/** Size chips per category, from the retail sizes in starterCatalog.ts: [chip label, starter id]. */
const CHIP_IDS: Record<string, [string, string][]> = {
  sofa: [['2-seat', 'sofa-2'], ['Compact', 'sofa-apt'], ['3-seat', 'sofa-3'], ['L-sofa', 'sectional-l']],
  'dining-table': [['Bistro', 'bistro'], ['Round', 'dining-round'], ['4-seat', 'dining-4'], ['6-seat', 'dining-6']],
  desk: [['Small', 'desk-sm'], ['Mid', 'desk'], ['Large', 'desk-standing']],
  rug: [['5×8', 'rug-5x8'], ['6×9', 'rug-6x9'], ['8×10', 'rug-8x10'], ['9×12', 'rug-9x12']],
  'floor-lamp': [['Floor', 'floor-lamp'], ['Table', 'table-lamp']],
}
const STARTER = new Map(STARTER_CATALOG.map((p) => [p.id, p]))

function sizeChips(entry: CatalogEntry): SizeChip[] {
  // Beds keep their own presets (Twin to King), named "Queen bed" and so on.
  if (entry.presets) return entry.presets.map((p) => ({ label: p.name, name: `${p.name} bed`, width: p.width, depth: p.depth }))
  return (CHIP_IDS[entry.category] ?? []).flatMap(([label, id]) => {
    const p = STARTER.get(id)
    return p ? [{ label, name: p.name, width: p.width, depth: p.depth, height: p.height }] : []
  })
}
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

/**
 * A free spot for `item` in one room: first the usual search pulled toward that room (it respects
 * every door and walkway), then a search of that room alone, kept only if it trips no checks.
 */
function dropIn(doc: Doc, item: Item, room: Room): DropResult | null {
  const layout = activeLayout(doc)
  const near = findDrop(doc, layout, item, { near: centroid(room.polygon) })
  if (near.ok && near.spot.roomId === room.id) return near
  const only = findDrop({ ...doc, rooms: [room] }, layout, item)
  if (!only.ok) return null
  const id = '__try'
  const trial = { ...layout, placements: [...layout.placements, { id, itemId: item.id, x: only.spot.x, y: only.spot.y, rotation: only.spot.rotation }] }
  const issues = checkLayout({ ...doc, items: [...doc.items, item], layouts: doc.layouts.map((l) => (l.id === layout.id ? trial : l)) }, trial)
  return issues.some((i) => i.severity !== 'info' && i.placementIds.includes(id)) ? null : only
}

const article = (name: string) => (/^[aeiou]/i.test(name) ? 'an' : 'a')

/** `top` rides just above the drawer: the selected piece's actions, so the drawer can stay open. */
export function BuyDrawer({ top }: { top?: ReactNode } = {}) {
  const tab = useGame((s) => s.buyTab)
  const setTab = useGame((s) => s.setBuyTab)
  const hint = useGame((s) => s.buyHint)
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
  const gridRef = useRef<HTMLDivElement>(null)
  // Bring the card the "Next" chip pointed at into view.
  useEffect(() => {
    const grid = gridRef.current
    const card = hint ? grid?.querySelector<HTMLElement>('.buy-card.hinted') : null
    if (!grid || !card) return
    // Scroll only the grid (scrollIntoView would also nudge the fixed game layer), by layout
    // offsets so the cards' entrance animation doesn't throw the numbers off.
    const left = card.offsetLeft - grid.offsetLeft
    const top = card.offsetTop - grid.offsetTop
    const x = left < grid.scrollLeft || left + card.offsetWidth > grid.scrollLeft + grid.clientWidth ? Math.max(0, left - 12) : grid.scrollLeft
    const y = top < grid.scrollTop || top + card.offsetHeight > grid.scrollTop + grid.clientHeight ? Math.max(0, top - 8) : grid.scrollTop
    if (x !== grid.scrollLeft || y !== grid.scrollTop) grid.scrollTo({ left: x, top: y })
  }, [hint, tab])
  const [popped, setPopped] = useState<{ key: string; n: number } | null>(null)
  const popTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(popTimer.current), [])
  const celebrate = (key: string) => {
    setPopped((p) => ({ key, n: (p?.n ?? 0) + 1 }))
    clearTimeout(popTimer.current)
    popTimer.current = setTimeout(() => setPopped(null), 900)
  }

  const place = (entry: CatalogEntry, preset?: SizeChip) => {
    const st = useStore.getState()
    const item = preset
      ? itemFromCatalog(entry, uid(), { name: preset.name, width: preset.width, depth: preset.depth, ...(preset.height ? { height: preset.height } : {}) })
      : itemFromCatalog(entry, uid())
    // A goal about one room ("Nothing in the Bathroom yet") puts the piece it pointed at in that room, if it fits there.
    const room = hint?.category === entry.category && hint.roomId ? st.doc.rooms.find((r) => r.id === hint.roomId) : undefined
    const there = room ? dropIn(st.doc, item, room) : null
    // Otherwise the item goes to a free spot that suits it and is selected; when nothing fits, the store toasts why.
    let res = there?.ok ? st.addItem(item, there.spot) : st.addItem(item)
    if (res.ok && there?.ok) {
      if (Math.abs(there.spot.rotation) > 1e-3) st.updatePlacement(res.placementId, { rotation: there.spot.rotation })
      res = { ...res, roomName: room!.name }
    }
    if (!res.ok) return
    if (hint) useGame.getState().setBuyHint(null)
    const name = (preset ? preset.name : displayName(entry)).replace(/^(\w)(?=[a-z])/, (c) => c.toLowerCase())
    st.notify(`Placed ${article(name)} ${name}${res.roomName ? ` in the ${res.roomName}` : ''}.`)
    celebrate(entry.category + (preset ? `:${preset.label}` : ''))
  }
  const placeMine = (item: Item) => {
    const st = useStore.getState()
    const res = st.placeItem(item.id) // adds a placement of the same item and selects it
    if (!res.ok) return
    st.notify(`Placed another “${item.name}”${res.roomName ? ` in the ${res.roomName}` : ''}.`)
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
      {top}
      <div className="buy-tabs" role="tablist" aria-label="Furniture categories">
        {CATEGORY_GROUPS.map((g, gi) => (
          <button key={g.key} role="tab" aria-selected={g.key === group.key} aria-label={g.label} title={`${g.label} (${gi + 1})`} className={`buy-tab ${g.key === group.key ? 'on' : ''}`} onClick={() => {
              setTab(g.key)
              if (hint) useGame.getState().setBuyHint(null)
            }}>
            <span className="buy-tab-icon">
              {g.key === 'mine' ? <IconStar /> : <ItemThumb category={TAB_ICON[g.key] ?? g.categories[0]} color={ENTRIES.get(TAB_ICON[g.key] ?? g.categories[0])?.color ?? '#9aa4ad'} size={34} />}
              {g.key === 'mine' && mine.length > 0 && <span className="buy-badge">{mine.length}</span>}
            </span>
            <span className="buy-tab-label">{g.label}</span>
          </button>
        ))}
      </div>

      <div className="buy-grid" ref={gridRef} key={group.key} role="tabpanel" aria-label={group.label}>
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
              const chips = sizeChips(entry)
              return (
                <div key={cat} className={`buy-card ${isPopped(cat) || popped?.key.startsWith(cat + ':') ? 'pop' : ''}${hint?.category === cat ? ' hinted' : ''}`} style={{ ['--i' as string]: k }} draggable onDragStart={(e) => onDrag(e, cat)}>
                  <button className="buy-hit" onClick={() => place(entry)} aria-label={`Place ${article(name)} ${name.toLowerCase()}`}>
                    <span className="buy-thumb">
                      <ItemThumb category={cat} color={entry.color} size={78} />
                    </span>
                    <span className="buy-name">{name}</span>
                    <span className="buy-size">{footprint(entry.width, entry.depth, units)}</span>
                  </button>
                  {chips.length > 0 && (
                    <span className="buy-presets">
                      {chips.map((pr) => (
                        <button key={pr.label} className="buy-preset" onClick={() => place(entry, pr)} aria-label={`Place ${article(pr.name)} ${pr.name.toLowerCase()}, ${footprint(pr.width, pr.depth, units)}`} title={`${pr.name}, ${footprint(pr.width, pr.depth, units)}`}>
                          {pr.label}
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
