// Buy mode: the furniture drawer above the mode dock, and the "shop from a link or photo" sheet.
import { useEffect, useMemo, useRef, useState, type DragEvent, type ReactNode } from 'react'
import { CATALOG, itemFromCatalog, type CatalogEntry } from '../model/catalog'
import { checkLayout } from '../model/checks'
import { centroid } from '../model/geometry'
import { findDrop, nookTarget, type DropResult } from '../model/placement'
import { STARTER_CATALOG } from '../model/starterCatalog'
import { uid } from '../model/sample'
import type { Doc, Item, Room, Units } from '../model/types'
import { formatLength } from '../model/units'
import { activeLayout, defaultDrop, useStore } from '../store'
import { analyze } from '../analysis'
import { scoreHome } from './goals'
import { ImportDialog, Photo } from '../components/AddDialogs'
import { useGame } from './gameStore'
import { IconCamera, IconLink, IconStar } from './icons'
import { ItemThumb } from './ItemThumb'
import { familyOf, picksFor, type Pick, type ThumbLook } from './swatches'
import './buy.css'
import './feel.css'

export const CATEGORY_GROUPS: { key: string; label: string; emoji?: string; categories: string[] }[] = [
  { key: 'living', label: 'Living', emoji: '🛋️', categories: ['sofa', 'armchair', 'coffee-table', 'side-table', 'ottoman', 'tv-stand', 'console-table'] },
  { key: 'bedroom', label: 'Bedroom', emoji: '🛏️', categories: ['bed', 'nightstand', 'dresser', 'wardrobe'] },
  { key: 'dining', label: 'Dining', emoji: '🍽️', categories: ['dining-table', 'dining-chair', 'bench', 'sideboard'] },
  { key: 'work', label: 'Work', emoji: '💻', categories: ['desk', 'office-chair', 'bookshelf'] },
  { key: 'decor', label: 'Decor', emoji: '🪴', categories: ['plant', 'floor-lamp', 'table-lamp', 'pendant', 'rug', 'box'] },
  { key: 'wall', label: 'Walls', emoji: '🖼️', categories: ['wall-art', 'mirror', 'wall-shelf', 'curtains', 'wall-clock'] },
  { key: 'mine', label: 'Mine', emoji: '⭐', categories: [] },
]

/** The category drawn on each tab chip. */
const TAB_ICON: Record<string, string> = { living: 'sofa', bedroom: 'bed', dining: 'dining-chair', work: 'desk', decor: 'plant', wall: 'wall-art' }

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
  /** The colour it's placed in (a "Pick one" card); otherwise the category's own. */
  color?: string
  /** How a "Pick one" card's thumbnail tells it apart (round, tall, smaller). */
  look?: ThumbLook
}

/** Size chips per category, from the retail sizes in starterCatalog.ts: [chip label, starter id]. */
const CHIP_IDS: Record<string, [string, string][]> = {
  sofa: [['2-seat', 'sofa-2'], ['Compact', 'sofa-apt'], ['3-seat', 'sofa-3'], ['L-sofa', 'sectional-l']],
  'dining-table': [['Bistro', 'bistro'], ['Round', 'dining-round'], ['4-seat', 'dining-4'], ['6-seat', 'dining-6']],
  desk: [['Small', 'desk-sm'], ['Mid', 'desk'], ['Large', 'desk-standing']],
  rug: [['5×8', 'rug-5x8'], ['6×9', 'rug-6x9'], ['8×10', 'rug-8x10'], ['9×12', 'rug-9x12']],
  'floor-lamp': [['Floor', 'floor-lamp'], ['Arc', 'arc-lamp']],
  armchair: [['Classic', 'armchair'], ['Accent', 'accent-chair'], ['Recliner', 'recliner'], ['Swivel', 'swivel']],
  'coffee-table': [['Classic', 'coffee-rect'], ['Round', 'coffee-round'], ['Small', 'coffee-small']],
  ottoman: [['Square', 'ottoman-sq'], ['Pouf', 'pouf'], ['Storage', 'ottoman-rect']],
  'side-table': [['Square', 'side'], ['C-shape', 'side-c']],
  'wall-art': [['Small', 'art-sm'], ['Medium', 'art-md'], ['Wide', 'art-lg']],
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
/** A "Pick one" card as a preset: the starter piece's name and size, in the pick's colour. */
export function pickChip(pick: Pick): SizeChip | null {
  const p = STARTER.get(pick.id)
  return p ? { label: pick.label, name: p.name, width: p.width, depth: p.depth, height: p.height, color: pick.color, look: pick.look } : null
}

/** The item a preset places: its name, size and colour over the category's defaults. */
export function presetItem(entry: CatalogEntry, id: string, preset?: SizeChip): Item {
  if (!preset) return itemFromCatalog(entry, id)
  return itemFromCatalog(entry, id, {
    name: preset.name,
    width: preset.width,
    depth: preset.depth,
    ...(preset.height ? { height: preset.height } : {}),
    ...(preset.color ? { color: preset.color } : {}),
  })
}

/** " in the Living room" for a placement toast; nothing in a one-room home, where it can only go there. */
export function inRoom(roomName?: string): string {
  return roomName && useStore.getState().doc.rooms.length > 1 ? ` in the ${roomName}` : ''
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

/**
 * Items that are yours rather than generic catalog pieces: imported from a link or photo, scanned, or
 * marked owned. An emptied moving box isn't (its item stays only so the unpack goal can count it).
 */
export function isMine(i: Item): boolean {
  if (i.unpacked) return false
  return !!(i.owned || i.link || i.imageUrl || i.scanId || i.source === 'link' || i.source === 'photo' || i.source === 'scan')
}

/** The "Mine" tab's badge: your pieces still waiting to be placed (owned, in no placement; no moving boxes). */
export function waitingCount(items: Item[], placed: (id: string) => boolean): number {
  return items.filter((i) => isMine(i) && i.owned && !i.unpack?.length && !placed(i.id)).length
}

/**
 * Pieces that finish the same goal as the hinted category (the seating goal takes a sofa or an
 * armchair), so "From your boxes" offers any of them. Mirrors the goal checks in goals.ts.
 */
const STORAGE = ['dresser', 'wardrobe', 'bookshelf', 'sideboard', 'console-table']
export const ACCEPTS: Record<string, string[]> = {
  sofa: ['sofa', 'armchair'],
  dresser: STORAGE,
  wardrobe: STORAGE,
  bookshelf: STORAGE,
  nightstand: ['nightstand', 'side-table'],
  'dining-chair': ['dining-chair', 'bench'],
}

/** A waiting piece's pick label: "From your boxes" when a box held its kind, otherwise "From Mine". */
export function pickLabel(item: Item, items: Item[]): string {
  return items.some((b) => b.unpacked && b.unpack?.includes(item.category)) ? 'From your boxes' : 'From Mine'
}

/** A boxed piece of yours (owned, not placed) that would finish a goal asking for `category`: its own kind first. */
export function boxedFor(category: string, items: Item[], placed: (id: string) => boolean): Item | undefined {
  const ok = ACCEPTS[category] ?? [category]
  const waiting = items.filter((i) => isMine(i) && i.owned && !i.unpack?.length && !placed(i.id) && ok.includes(i.category))
  return waiting.find((i) => i.category === category) ?? waiting[0]
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

/** Pieces the Next chip puts beside the piece its goal is about (a reading lamp by the armchair). */
const BESIDE = new Set(['floor-lamp', 'table-lamp', 'side-table'])

type Hint = { category: string; roomId?: string; near?: string; title?: string }
type Spot = { x: number; y: number; rotation: number; on?: string; roomName?: string }

/**
 * Where a tap on a "Pick one" card (or the hinted card) puts `item`: in the goal's room, beside the
 * goal's piece, or (spot undefined) wherever the store's default drop finds. `fail` is a reason to
 * toast when the goal named a place that has no room.
 */
export function pickSpot(doc: Doc, item: Item, hint: Hint | null): { spot?: Spot; fail?: string } {
  const layout = activeLayout(doc)
  const room = hint?.category === item.category && hint.roomId ? doc.rooms.find((r) => r.id === hint.roomId) : undefined
  const there = room ? dropIn(doc, item, room) : null
  if (there?.ok) return { spot: { ...there.spot, roomName: room!.name } }
  const nearId = hint?.category === item.category ? hint.near ?? (BESIDE.has(item.category) ? nookTarget(doc, layout) : undefined) : undefined
  const beside = nearId ? findDrop(doc, layout, item, { nextTo: nearId }) : null
  if (beside && !beside.ok && hint?.near) return { fail: beside.reason }
  if (beside?.ok) return { spot: beside.spot }
  return hint?.category === item.category ? cleanDrop(doc, item) : {}
}

/** Warnings and errors a trial placement `id` of `item` at `spot` would carry. */
function trialIssues(doc: Doc, item: Item, spot: Spot): number {
  const layout = activeLayout(doc)
  const id = '__try'
  const trial = { ...layout, placements: [...layout.placements, { id, itemId: item.id, x: spot.x, y: spot.y, rotation: spot.rotation, ...(spot.on ? { on: spot.on } : {}) }] }
  const issues = checkLayout({ ...doc, items: [...doc.items, item], layouts: doc.layouts.map((l) => (l.id === layout.id ? trial : l)) }, trial)
  return issues.filter((i) => i.severity !== 'info' && i.placementIds.includes(id)).length
}

/**
 * The store's default drop, unless it wedges the piece in with a clearance warning (an armchair
 * squeezed beside the sofa) while a search pulled toward a room's middle finds a clean spot.
 */
function cleanDrop(doc: Doc, item: Item): { spot?: Spot } {
  const first = defaultDrop(doc, item)
  if (!first.ok || !trialIssues(doc, item, first.spot)) return {}
  // Same room only: the room it picked is the one the goal wants it in (the living room's seats).
  const room = doc.rooms.find((r) => r.id === first.spot.roomId)
  if (!room) return {}
  const alt = defaultDrop(doc, item, centroid(room.polygon))
  if (alt.ok && alt.spot.roomId === room.id && !trialIssues(doc, item, alt.spot)) return { spot: alt.spot }
  const only = findDrop({ ...doc, rooms: [room] }, activeLayout(doc), item, { near: centroid(room.polygon) })
  if (only.ok && !trialIssues(doc, item, only.spot)) return { spot: { ...only.spot, roomName: room.name } }
  return {}
}

/** How a piece placed by a pick card would leave the home: its stars, warnings, and goals it would undo. */
export interface PickTrial {
  ok: boolean
  stars: number
  warnings: number
  /** Goals done now that the piece would undo (a long coffee table squeezing a walkway). */
  broken: number
}

/** Dry-run a pick card: place `item` exactly as a tap would, then score the home. */
export function trialPick(doc: Doc, item: Item, hint: Hint | null): PickTrial {
  const layout = activeLayout(doc)
  const before = scoreHome(doc, layout, analyze(doc, layout))
  const warn = (a: ReturnType<typeof analyze>) => a.issues.filter((i) => i.severity !== 'info').length
  const p = pickSpot(doc, item, hint)
  const drop = p.fail ? null : p.spot ? { ok: true as const, spot: p.spot } : defaultDrop(doc, item)
  if (!drop?.ok) return { ok: false, stars: before.stars, warnings: Infinity, broken: 0 }
  const s = drop.spot
  const l2 = { ...layout, placements: [...layout.placements, { id: '__pick', itemId: item.id, x: s.x, y: s.y, rotation: s.rotation, ...(s.on ? { on: s.on } : {}) }] }
  const d2 = { ...doc, items: [...doc.items, item], layouts: doc.layouts.map((l) => (l.id === layout.id ? l2 : l)) }
  const a2 = analyze(d2, l2)
  const after = scoreHome(d2, l2, a2)
  const doneAfter = new Set(after.goals.filter((g) => g.done).map((g) => g.id))
  const broken = before.goals.filter((g) => g.done && !g.visit && !doneAfter.has(g.id)).length
  return { ok: true, stars: after.stars, warnings: warn(a2) - warn(analyze(doc, layout)), broken }
}

/**
 * The pick variants worth offering: ones that would undo a goal or add a warning go, unless every
 * one does; then only the least bad stays. Order is kept.
 */
export function goodPicks<T>(cands: T[], trial: (c: T) => PickTrial): T[] {
  if (cands.length < 2) return cands
  const t = cands.map(trial)
  const clean = cands.filter((_, i) => t[i].ok && t[i].broken === 0 && t[i].warnings <= 0)
  if (clean.length) return clean
  const cost = (x: PickTrial) => (x.ok ? 0 : 1000) + 10 * x.broken + x.warnings - x.stars / 100
  let best = 0
  t.forEach((x, i) => {
    if (cost(x) < cost(t[best])) best = i
  })
  return [cands[best]]
}

const article = (name: string) => (/^[aeiou]/i.test(name) ? 'an' : 'a')

/** `top` rides just above the drawer: the selected piece's actions, so the drawer can stay open. */
export function BuyDrawer({ top }: { top?: ReactNode } = {}) {
  const tab = useGame((s) => s.buyTab)
  const setTab = useGame((s) => s.setBuyTab)
  const hint = useGame((s) => s.buyHint)
  // The selected piece's action bar rides where the "Pick one" row floats, so the row steps aside.
  const selected = useStore((s) => !!s.selection)
  const units = useStore((s) => s.doc.apartment.units)
  const hintRoom = useStore((s) => (hint?.roomId ? s.doc.rooms.find((r) => r.id === hint.roomId)?.name : undefined))
  const hintEntry = hint?.category ? ENTRIES.get(hint.category) : undefined
  const items = useStore((s) => s.doc.items)
  const placements = useStore((s) => activeLayout(s.doc).placements)
  const mine = useMemo(() => items.filter(isMine), [items])
  const counts = useMemo(() => {
    const m = new Map<string, number>()
    for (const p of placements) m.set(p.itemId, (m.get(p.itemId) ?? 0) + 1)
    return m
  }, [placements])
  const waiting = useMemo(() => waitingCount(items, (id) => !!counts.get(id)), [items, counts])

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

  /** `chose`: the player picked this colour on a "Pick one" card (not a plain tap on the hinted card), so it's remembered as their taste. */
  const place = (entry: CatalogEntry, preset?: SizeChip, chose = false) => {
    const st = useStore.getState()
    const item = presetItem(entry, uid(), preset)
    // A goal about one room puts the piece in that room if it fits; a goal about one piece puts it
    // beside that piece; a piece the goal named a place for that has no room there says so.
    const { spot, fail } = pickSpot(st.doc, item, hint)
    if (fail) {
      st.notify(fail)
      return
    }
    // Otherwise the item goes to a free spot that suits it and is selected; when nothing fits, the store toasts why.
    let res = spot ? st.addItem(item, spot) : st.addItem(item)
    if (res.ok && spot) res = { ...res, roomName: spot.roomName }
    if (!res.ok) return
    if (hint) useGame.getState().setBuyHint(null)
    // Taste memory: the colour picked for a family (wood, fabric, light) is offered first next time.
    const fam = chose && preset?.color ? familyOf(entry.category) : undefined
    if (fam) useGame.getState().setTaste(fam, preset!.color!)
    const name = (preset ? preset.name : displayName(entry)).replace(/^(\w)(?=[a-z])/, (c) => c.toLowerCase())
    st.notify(`Placed ${article(name)} ${name}${inRoom(res.roomName)}.`)
    celebrate(entry.category + (preset ? `:${preset.label}` : ''))
  }
  const placeMine = (item: Item) => {
    const st = useStore.getState()
    const res = st.placeItem(item.id) // adds a placement of the same item and selects it
    if (!res.ok) return
    if (hint) useGame.getState().setBuyHint(null)
    st.notify(`Placed ${counts.get(item.id) ? 'another' : 'your'} “${item.name}”${inRoom(res.roomName)}.`)
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

  // "Pick one": 2-3 variants of the hinted piece, each placed in one tap. A piece of that kind
  // still waiting in your boxes (owned, not placed) comes first, so Next never sells you a copy.
  const taste = useGame((s) => s.taste)
  const doc = useStore((s) => s.doc)
  // Each variant is dry-run where a tap would put it: one that would undo a goal (a long coffee
  // table closing a walkway) or wedge in with a warning isn't offered while a cleaner one is.
  const vetted = useMemo(() => {
    if (!hintEntry) return { all: 0, chips: [] }
    const all = picksFor(hintEntry.category, taste).flatMap((pk) => {
      const chip = pickChip(pk)
      return chip ? [{ chip, taste: !!pk.taste }] : []
    })
    if (all.length < 2) return { all: all.length, chips: all }
    try {
      return { all: all.length, chips: goodPicks(all, (c) => trialPick(doc, presetItem(hintEntry, '__pick_item', c.chip), hint)) }
    } catch {
      return { all: all.length, chips: all }
    }
  }, [doc, hint, hintEntry, taste])
  const pickCards: PickCard[] = []
  let pickCount = 0
  let firstPick: (() => void) | undefined
  if (hintEntry) {
    const boxed = boxedFor(hintEntry.category, items, (id) => !!counts.get(id))
    if (boxed) pickCards.push({ key: boxed.id, label: pickLabel(boxed, items), name: boxed.name, category: boxed.category, color: boxed.color, mine: true, onClick: () => placeMine(boxed) })
    const chips = vetted.chips
    pickCount = vetted.all
    for (const { chip, taste: liked } of chips.slice(0, 3 - pickCards.length))
      pickCards.push({ key: `${hintEntry.category}:${chip.label}`, label: chip.label, name: chip.name, category: hintEntry.category, color: chip.color, look: chip.look, taste: liked, onClick: () => place(hintEntry, chip, true) })
    // A plain tap on the hinted card places the first card without counting it as a chosen colour.
    const first = pickCards[0]
    const firstChip = chips[0]?.chip
    if (vetted.all >= 2 && first) firstPick = first.mine ? first.onClick : firstChip ? () => place(hintEntry, firstChip) : undefined
  }
  const showPicker = pickCount >= 2

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
              {g.key === 'mine' && waiting > 0 && (
                <span className="buy-badge" aria-label={`${waiting} waiting to be placed`}>
                  {waiting}
                </span>
              )}
            </span>
            <span className="buy-tab-label">{g.label}</span>
          </button>
        ))}
      </div>

      {hintEntry && showPicker && !selected && <HintPicker title={displayName(hintEntry)} goal={hint?.title} room={hintRoom} cards={pickCards} popped={popped?.key} />}
      {hintEntry && !showPicker && !selected && (
        <div className="feel-hint-pill" role="status">
          Tap <b>{displayName(hintEntry)}</b>
          {hintRoom ? (
            <>
              {' '}
              <span aria-label="in the">→</span> <b>{hintRoom}</b>
            </>
          ) : (
            ' to place it'
          )}
        </div>
      )}
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
                  <button className="buy-hit" onClick={() => (hint?.category === cat && firstPick ? firstPick() : place(entry))} aria-label={`Place ${article(name)} ${name.toLowerCase()}`}>
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

interface PickCard {
  key: string
  label: string
  /** The placed piece's name, for the card's spoken label. */
  name: string
  category: string
  color?: string
  look?: ThumbLook
  /** In the colour the player picked last for this family. */
  taste?: boolean
  /** A piece of yours from a moving box, placed rather than bought. */
  mine?: boolean
  onClick: () => void
}

/** The "Pick one" row floating above the drawer while the Next chip's hint is up. */
function HintPicker({ title, goal, room, cards, popped }: { title: string; goal?: string; room?: string; cards: PickCard[]; popped?: string }) {
  return (
    <div className="feel-pick" role="group" aria-label={`${goal ? `${goal}: p` : 'P'}ick one ${title.toLowerCase()}${room ? ` for the ${room}` : ''}`}>
      <div className="feel-pick-head">
        {goal ? (
          <>
            <b>{goal}</b> · pick one
          </>
        ) : (
          'Pick one'
        )}
        {room ? (
          <>
            {' '}
            <span aria-hidden>→</span> <b>{room}</b>
          </>
        ) : null}
      </div>
      <div className="feel-pick-row">
        {cards.map((c) => (
          <button
            key={c.key}
            type="button"
            className={`feel-pick-card${c.mine ? ' mine' : ''}${popped === c.key ? ' pop' : ''}`}
            onClick={c.onClick}
            aria-label={c.mine ? `Place your ${c.name.toLowerCase()} from your boxes` : `Place ${article(c.name)} ${c.name.toLowerCase()}, ${c.label}${c.taste ? ', your style' : ''}`}
            title={c.mine ? c.name : `${c.name}, ${c.label}`}
          >
            <span className="feel-pick-thumb">
              <ItemThumb category={c.category} color={c.color ?? '#a3907d'} size={40} look={c.look} />
              {c.taste && <span className="feel-pick-dot" title="Your style" aria-hidden />}
            </span>
            <span className="feel-pick-label">{c.label}</span>
          </button>
        ))}
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
