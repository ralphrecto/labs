// "Your designs": the homes you've moved into and the designs for the open one, as little
// top-down postcards. Rename and delete sit behind a "…" menu, and deleting asks first.
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Doc, Layout } from '../model/types'
import { deleteHome, switchHome, useStore } from '../store'
import { listHomes, type HomeEntry } from '../storage'
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

/** Every home you've moved into, the open one first. Tap one to go there. */
export function HomesStrip({ onRenameHome }: { onRenameHome: () => void }) {
  const doc = useStore((s) => s.doc)
  // The index changes on save; re-read it whenever the open document changes.
  const [tick, setTick] = useState(0)
  const homes = useMemo<HomeEntry[]>(() => {
    const saved = listHomes()
    const here = saved.find((h) => h.id === doc.apartment.id) ?? { id: doc.apartment.id, name: doc.apartment.name, updatedAt: Date.now() }
    return [{ ...here, name: doc.apartment.name }, ...saved.filter((h) => h.id !== doc.apartment.id)]
  }, [doc.apartment.id, doc.apartment.name, tick])
  const many = homes.length > 1
  return (
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
                        confirm: `Delete ${h.name} and all its designs? This can't be undone.`,
                        onPick: () => {
                          deleteHome(h.id)
                          setTick((t) => t + 1)
                          useStore.getState().notify(`Deleted ${h.name}.`)
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
  )
}

export function DesignsSheetBody() {
  const doc = useStore((s) => s.doc)
  const s = useStore.getState()
  const [editing, setEditing] = useState<string | null>(null)
  const [renamingHome, setRenamingHome] = useState(false)
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
      <div className="dz-designs">
        {doc.layouts.map((l) => {
          const active = l.id === doc.activeLayoutId
          const n = l.placements.length
          return (
            <div key={l.id} className={`dz-design g-card${active ? ' on' : ''}`}>
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
                    {n} piece{n === 1 ? '' : 's'}
                    {active && ' · playing'}
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
