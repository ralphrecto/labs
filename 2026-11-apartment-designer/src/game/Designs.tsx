// "Your designs": the homes you've moved into and the designs for the open one, as little
// top-down postcards. Rename and delete sit behind a "…" menu, and deleting asks first.
import { useEffect, useMemo, useRef, useState } from 'react'
import { analyze } from '../analysis'
import type { Doc, Layout } from '../model/types'
import { deleteHome, restoreHome, switchHome, useStore } from '../store'
import { listHomes, type HomeEntry } from '../storage'
import { scoreHome } from './goals'
import { create } from 'zustand'
import { catalogEntry, itemFromCatalog } from '../model/catalog'
import { findDrop } from '../model/placement'
import { activeLayout } from '../store'
import { useGame } from './gameStore'
import { applicableTwists, finishedTwists, levelProgress, pickTwists, sameArrangement, TWIST_IDS, TWISTS, LEVELS } from './goals'
import * as I from './icons'
import './designs.css'

/** A tiny floor plan: room outlines plus a rectangle per piece, in the piece's own color. */
export function DesignThumb({ doc, layout, size = 64 }: { doc: Doc; layout?: Layout; size?: number }) {
  const pts = doc.rooms.flatMap((r) => r.polygon)
  if (!pts.length)
    return (
      <svg className="dz-thumb" width={size} height={size} viewBox="0 0 10 10" aria-hidden>
        <rect x="2" y="2" width="6" height="6" rx="0.6" fill="none" stroke="var(--g-line)" strokeWidth="0.5" strokeDasharray="1 0.8" />
      </svg>
    )
  const minX = Math.min(...pts.map((p) => p.x))
  const maxX = Math.max(...pts.map((p) => p.x))
  const minY = Math.min(...pts.map((p) => p.y))
  const maxY = Math.max(...pts.map((p) => p.y))
  const span = Math.max(maxX - minX, maxY - minY, 0.5)
  const pad = span * 0.08
  const vb = `${minX - pad + (span - (maxX - minX)) / -2} ${minY - pad + (span - (maxY - minY)) / -2} ${span + pad * 2} ${span + pad * 2}`
  const items = new Map(doc.items.map((i) => [i.id, i]))
  const sw = span / 50
  return (
    <svg className="dz-thumb" width={size} height={size} viewBox={vb} aria-hidden>
      {doc.rooms.map((r) => (
        <polygon key={r.id} points={r.polygon.map((p) => `${p.x},${p.y}`).join(' ')} fill="#fff" stroke="#8a7a68" strokeWidth={sw * 1.6} strokeLinejoin="round" />
      ))}
      {(layout?.placements ?? []).map((p) => {
        const it = items.get(p.itemId)
        if (!it) return null
        return (
          <rect
            key={p.id}
            x={-it.width / 2}
            y={-it.depth / 2}
            width={it.width}
            height={it.depth}
            rx={Math.min(it.width, it.depth) * 0.15}
            fill={it.color || '#c9b8a3'}
            stroke="rgba(45,38,32,0.45)"
            strokeWidth={sw * 0.6}
            transform={`translate(${p.x} ${p.y}) rotate(${(p.rotation * 180) / Math.PI})`}
          />
        )
      })}
    </svg>
  )
}

type MenuItem = { label: string; icon?: React.ReactNode; danger?: boolean; confirm?: string; onPick: () => void }

/** A small "…" button with a popover menu. An item with `confirm` asks once before it runs. */
function MoreMenu({ label, items }: { label: string; items: MenuItem[] }) {
  const [open, setOpen] = useState(false)
  const [asking, setAsking] = useState<MenuItem | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const close = () => {
    setOpen(false)
    setAsking(null)
  }
  useEffect(() => {
    if (!open) return
    const off = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) close()
    }
    document.addEventListener('pointerdown', off)
    return () => document.removeEventListener('pointerdown', off)
  }, [open])
  return (
    <div className="dz-more" ref={ref}>
      <button className="g-icon sm dz-more-btn" aria-label={label} aria-expanded={open} onClick={() => (open ? close() : setOpen(true))}>
        <I.IconMore />
      </button>
      {open && (
        <div className="dz-menu g-card" role="menu">
          {asking ? (
            <>
              <p className="dz-ask">{asking.confirm}</p>
              <div className="dz-ask-btns">
                <button className="g-btn sm ghost" onClick={close}>
                  Keep it
                </button>
                <button
                  className="g-btn sm dz-danger"
                  onClick={() => {
                    asking.onPick()
                    close()
                  }}
                >
                  {asking.label}
                </button>
              </div>
            </>
          ) : (
            items.map((m) => (
              <button
                key={m.label}
                role="menuitem"
                className={`dz-menu-item${m.danger ? ' danger' : ''}`}
                onClick={() => {
                  if (m.confirm) return setAsking(m)
                  m.onPick()
                  close()
                }}
              >
                {m.icon}
                {m.label}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}

function NameInput({ value, onDone }: { value: string; onDone: (v: string) => void }) {
  return (
    <input
      className="dz-rename"
      autoFocus
      defaultValue={value}
      aria-label="New name"
      onClick={(e) => e.stopPropagation()}
      onBlur={(e) => onDone(e.target.value.trim() || value)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        if (e.key === 'Escape') onDone(value)
      }}
    />
  )
}

const ago = (t: number) => {
  const m = Math.round((Date.now() - t) / 60000)
  if (m < 2) return 'just now'
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} h ago`
  const d = Math.round(h / 24)
  return d === 1 ? 'yesterday' : `${d} days ago`
}

/** Stars and level for each design of a home, and which one scores best (only with more than one). */
export function useDesignScores(doc: Doc) {
  return useMemo(() => {
    const scores = new Map(
      doc.layouts.map((l) => {
        try {
          const sc = scoreHome(doc, l, analyze(doc, l))
          const twist = l.remix ? sc.goals.find((g) => g.id === `twist:${l.remix!.twist}`) : undefined
          return [
            l.id,
            {
              stars: sc.stars,
              level: sc.level.name,
              progress: levelProgress(sc),
              twistDone: !!twist?.done,
              dream: !!l.remix?.dreamAt || (!l.remix && sc.level.index === LEVELS.length - 1),
              /** An untouched copy: the older design it matches. */
              sameAs: doc.layouts.slice(0, doc.layouts.indexOf(l)).find((o) => sameArrangement(o, l))?.name,
            },
          ] as const
        } catch {
          return [l.id, null] as const
        }
      }),
    )
    let best: string | null = null
    let top = 0
    if (doc.layouts.length > 1)
      for (const l of doc.layouts) {
        // An untouched copy never takes the crown from its source, and a finished remix has its
        // own badge (its twist, ticked) rather than racing the others on stars: Less is more can't win that.
        if (scores.get(l.id)?.sameAs || (l.remix && (scores.get(l.id)?.twistDone || l.remix.dreamAt))) continue
        const st = scores.get(l.id)?.stars ?? 0
        if (st > top) [best, top] = [l.id, st]
      }
    return { scores, best }
  }, [doc.rooms, doc.items, doc.layouts, doc.apartment.units])
}

function Crown() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden>
      <path d="M2 12.5h12l1-8-4 3-3-5-3 5-4-3z" fill="currentColor" stroke="currentColor" strokeWidth="1" strokeLinejoin="round" />
    </svg>
  )
}

/** Every home you've moved into, the open one first. Tap one to go there. */
export function HomesStrip({ onRenameHome }: { onRenameHome: () => void }) {
  const doc = useStore((s) => s.doc)
  // The index changes on save; re-read it whenever the open document changes.
  const [tick, setTick] = useState(0)
  // The home just deleted, for "Deleted X · Undo" (it waits in the trash for a week).
  const [deleted, setDeleted] = useState<{ id: string; name: string } | null>(null)
  const homes = useMemo<HomeEntry[]>(() => {
    const saved = listHomes()
    const here = saved.find((h) => h.id === doc.apartment.id) ?? { id: doc.apartment.id, name: doc.apartment.name, updatedAt: Date.now() }
    return [{ ...here, name: doc.apartment.name }, ...saved.filter((h) => h.id !== doc.apartment.id)]
  }, [doc.apartment.id, doc.apartment.name, tick])
  const many = homes.length > 1
  return (
    <>
    {deleted && (
      <div className="dz-undo" role="status">
        <span>Deleted {deleted.name}</span>
        <button
          className="g-btn sm"
          onClick={() => {
            const ok = restoreHome(deleted.id)
            setDeleted(null)
            setTick((t) => t + 1)
            useStore.getState().notify(ok ? `${deleted.name} is back.` : `Couldn't bring back ${deleted.name}.`)
          }}
        >
          Undo
        </button>
      </div>
    )}
    <div className="dz-homes" aria-label="Your homes">
      {homes.map((h) => {
        const here = h.id === doc.apartment.id
        return (
          <div key={h.id} className={`dz-home${here ? ' on' : ''}`}>
            <button
              className="dz-home-main"
              aria-current={here || undefined}
              onClick={() => {
                if (here) return
                if (!switchHome(h.id)) useStore.getState().notify("Couldn't open that home.")
                else useStore.getState().notify(`Welcome back to ${h.name}.`)
              }}
            >
              <I.IconHome width={18} height={18} />
              <span>
                <strong>{h.name}</strong>
                <small>{here ? 'You are here' : `Visited ${ago(h.updatedAt)}`}</small>
              </span>
            </button>
            <MoreMenu
              label={`More for ${h.name}`}
              items={[
                ...(here ? [{ label: 'Rename home', icon: <I.IconPen width={16} height={16} />, onPick: onRenameHome }] : []),
                ...(many
                  ? [
                      {
                        label: 'Delete home',
                        icon: <I.IconTrash width={16} height={16} />,
                        danger: true,
                        confirm: `Delete ${h.name} and all its designs? You can undo it right after.`,
                        onPick: () => {
                          if (deleteHome(h.id)) setDeleted({ id: h.id, name: h.name })
                          else useStore.getState().notify(`Couldn't delete ${h.name}.`)
                          setTick((t) => t + 1)
                        },
                      },
                    ]
                  : []),
              ]}
            />
          </div>
        )
      })}
    </div>
    </>
  )
}

export function DesignsSheetBody() {
  const doc = useStore((s) => s.doc)
  const s = useStore.getState()
  const [editing, setEditing] = useState<string | null>(null)
  const [renamingHome, setRenamingHome] = useState(false)
  const { scores, best } = useDesignScores(doc)
  return (
    <>
      <h3 className="hud-h3">Your homes</h3>
      {renamingHome ? (
        <NameInput
          value={doc.apartment.name}
          onDone={(v) => {
            s.renameApartment(v)
            setRenamingHome(false)
          }}
        />
      ) : (
        <HomesStrip onRenameHome={() => setRenamingHome(true)} />
      )}
      <h3 className="hud-h3">Designs for {doc.apartment.name}</h3>
      <DreamCount scores={scores} />
      <div className="dz-designs">
        {doc.layouts.map((l) => {
          const active = l.id === doc.activeLayoutId
          const n = l.placements.length
          const sc = scores.get(l.id)
          return (
            <div key={l.id} className={`dz-design g-card${active ? ' on' : ''}`}>
              {l.remix && sc?.dream && !sc.sameAs && <span className="dz-dream">Dream ★</span>}
              <button className="dz-design-main" onClick={() => s.setActiveLayout(l.id)} aria-pressed={active}>
                <DesignThumb doc={doc} layout={l} />
                <span className="dz-design-text">
                  {editing === l.id ? (
                    <NameInput
                      value={l.name}
                      onDone={(v) => {
                        s.renameLayout(l.id, v)
                        setEditing(null)
                      }}
                    />
                  ) : (
                    <strong>{l.name}</strong>
                  )}
                  <small>
                    {sc?.sameAs ? (
                      <span className="dz-same">Same as {sc.sameAs} · move something to make it yours</span>
                    ) : (
                      sc && (
                        <span className="dz-score">
                          ★{sc.stars} · {l.remix?.dreamAt ? 'Dream reached' : sc.level}
                          {l.remix && !sc.dream && (
                            <span className="dz-bar" aria-hidden>
                              <i style={{ width: `${Math.round(sc.progress * 100)}%` }} />
                            </span>
                          )}
                        </span>
                      )
                    )}
                    {l.remix && <RemixBadge l={l} done={!!sc?.twistDone} />}
                    {best === l.id && (
                      <span className="dz-crown" title="Your best design">
                        <Crown />
                        Best
                      </span>
                    )}
                    <span>
                      {n} piece{n === 1 ? '' : 's'}
                      {active && ' · playing'}
                    </span>
                  </small>
                </span>
              </button>
              <MoreMenu
                label={`More for ${l.name}`}
                items={[
                  { label: 'Rename', icon: <I.IconPen width={16} height={16} />, onPick: () => setEditing(l.id) },
                  { label: 'Make a copy', icon: <I.IconCopy width={16} height={16} />, onPick: () => s.duplicateLayout(l.id) },
                  ...(doc.layouts.length > 1
                    ? [
                        {
                          label: 'Delete',
                          icon: <I.IconTrash width={16} height={16} />,
                          danger: true,
                          confirm: `Delete “${l.name}”? Its furniture arrangement goes with it.`,
                          onPick: () => {
                            s.deleteLayout(l.id)
                            s.notify(`Deleted ${l.name}. Undo brings it back.`)
                          },
                        },
                      ]
                    : []),
                ]}
              />
            </div>
          )
        })}
      </div>
    </>
  )
}

// ---- Remix ----------------------------------------------------------------

/** Whether the twist picker is showing in the Designs sheet. */
const useRemixPicker = create<{ open: boolean }>(() => ({ open: false }))

// Closing the Designs sheet folds the picker away again.
useGame.subscribe((s, prev) => {
  if (prev.sheet === 'layouts' && s.sheet !== 'layouts') useRemixPicker.setState({ open: false })
})

/** Opens the Designs sheet on the twist picker (the HUD's "Remix this home" chip). */
export function openRemixPicker() {
  useRemixPicker.setState({ open: true })
  useGame.getState().openSheet('layouts')
}

/** "↻ Twist: Sofa faces the window ✓" (the "Dream ★" ribbon sits in the card's corner). */
function RemixBadge({ l, done }: { l: Layout; done: boolean }) {
  const t = TWISTS[l.remix!.twist]
  if (!t) return null
  return (
    <>
      <span className={`dz-twist${done ? ' done' : ''}`}>
        ↻ Twist: {t.title}
        {done && ' ✓'}
      </span>
      {!!l.remix!.dreamAt && (
        <span className="dz-crown dz-twist-won" title={`${t.name} twist finished`}>
          {TWIST_EMOJI[l.remix!.twist] ?? '↻'} ✓
        </span>
      )}
    </>
  )
}

/** "2 of 3 designs are Dream homes", once there is more than one design and at least one Dream home. */
function DreamCount({ scores }: { scores: Map<string, { dream: boolean; sameAs?: string } | null> }) {
  // An untouched copy is the same design again: it counts toward neither side.
  const own = [...scores.values()].filter((x) => !x?.sameAs)
  const total = own.length
  const n = own.filter((x) => x?.dream).length
  if (total < 2 || n === 0) return null
  return (
    <p className="dz-dreams">
      {n} of {total} designs {n === 1 ? 'is a Dream home' : 'are Dream homes'}
    </p>
  )
}

/** A small picture for each twist in the picker. */
const TWIST_EMOJI: Record<string, string> = {
  'talk-corner': '💬',
  'open-floor': '🕺',
  'window-seat': '🪟',
  'fresh-spots': '🔀',
  'bed-swap': '🛏️',
  'less-is-more': '🍃',
}

/**
 * "Remix this home": the primary button of the Designs sheet, which opens a small picker of three
 * twists. Picking one makes the remix (the big pieces only) and closes the sheet.
 */
export function RemixPicker() {
  const open = useRemixPicker((s) => s.open)
  const doc = useStore((s) => s.doc)
  const twists = useMemo(() => {
    if (!open) return []
    const fits = (c: string) => {
      try {
        return findDrop(doc, activeLayout(doc), itemFromCatalog(catalogEntry(c), '__probe')).ok
      } catch {
        return true
      }
    }
    return pickTwists(doc, applicableTwists(doc, fits))
  }, [open, doc])
  const finished = useMemo(() => finishedTwists(doc), [doc.layouts])
  if (!open)
    return (
      <button className="g-btn primary" onClick={() => useRemixPicker.setState({ open: true })}>
        <I.IconCopy width={18} height={18} /> Remix this home
      </button>
    )
  return (
    <div className="dz-picker g-card" role="group" aria-label="Pick a twist">
      <p className="dz-picker-h">
        Remix <b>{activeLayout(doc).name}</b>: keep the big pieces, rebuild around them, with a twist.
      </p>
      {twists.map((t) => (
        <button
          key={t}
          className="dz-picker-opt"
          onClick={() => {
            useStore.getState().remixLayout(doc.activeLayoutId, t)
            useRemixPicker.setState({ open: false })
            useGame.getState().openSheet(null)
          }}
        >
          <strong>
            {TWIST_EMOJI[t] ? `${TWIST_EMOJI[t]} ` : ''}
            {TWISTS[t].name}
            {finished.has(t) && <span className="dz-picker-done" title="You finished this twist before"> ✓</span>}
          </strong>
          <small>{TWISTS[t].hint}</small>
        </button>
      ))}
      {finished.size > 0 && (
        <p className="dz-picker-count">
          {finished.size} of {TWIST_IDS.length} twists done
        </p>
      )}
      <button className="g-btn sm ghost" onClick={() => useRemixPicker.setState({ open: false })}>
        Not now
      </button>
    </div>
  )
}
