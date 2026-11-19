import { useState } from 'react'
import { useAnalysis } from '../analysis'
import { CATALOG, catalogEntry } from '../model/catalog'
import { ARCH, COMFORTABLE_WALKWAY, MIN_WALKWAY } from '../model/checks'
import { dist, polygonArea, wallSegments } from '../model/geometry'
import type { Item } from '../model/types'
import { formatLength } from '../model/units'
import { activeLayout, useStore } from '../store'
import { CatalogBrowser, ImportDialog, Photo } from './AddDialogs'
import { LengthField, NumberField, TextField } from './fields'
import { ScanInspector } from './ScanImport'

// ---------------------------------------------------------------- left: catalog, library, layouts

export function CatalogPanel() {
  const add = useStore((s) => s.addItemFromCatalog)
  const [dialog, setDialog] = useState<'catalog' | 'import' | null>(null)
  return (
    <section className="panel">
      <h3>Add furniture</h3>
      <div className="add-actions">
        <button className="primary" onClick={() => setDialog('catalog')} title="Browse generic furniture at typical sizes">
          Browse catalog
        </button>
        <button onClick={() => setDialog('import')} title="Add a real product from a store link, a photo, or pasted dimensions">
          From link or photo
        </button>
      </div>
      {dialog === 'catalog' && <CatalogBrowser onClose={() => setDialog(null)} />}
      {dialog === 'import' && <ImportDialog onClose={() => setDialog(null)} />}
      <p className="small muted">Or add a basic shape and set its size:</p>
      <div className="catalog-grid">
        {CATALOG.map((e) => (
          <button key={e.category} className="catalog-btn" onClick={() => add(e)} title={`Add a ${e.name.toLowerCase()}`}>
            <span className="swatch" style={{ background: e.color }} />
            {e.name}
          </button>
        ))}
      </div>
    </section>
  )
}

export function LibraryPanel() {
  const doc = useStore((s) => s.doc)
  const placeItem = useStore((s) => s.placeItem)
  const layout = activeLayout(doc)
  const used = new Set(layout.placements.map((p) => p.itemId))
  const unplaced = doc.items.filter((i) => !used.has(i.id))
  const units = doc.apartment.units
  const toBuy = doc.items.filter((i) => used.has(i.id) && !i.owned)
  const total = toBuy.reduce((s, i) => s + (i.price ?? 0), 0)
  return (
    <section className="panel">
      <h3>Your items</h3>
      {unplaced.length === 0 ? (
        <p className="muted small">Every item in your library is in this layout.</p>
      ) : (
        <>
          <p className="muted small">Not in this layout:</p>
          <ul className="list">
            {unplaced.map((i) => (
              <li key={i.id}>
                <span className="lib-item">
                  {i.imageUrl && <Photo className="mini-thumb" src={i.imageUrl} />}
                  <span>
                    {i.name} <span className="muted">· {formatLength(i.width, units)} × {formatLength(i.depth, units)}</span>
                  </span>
                </span>
                <button className="link" onClick={() => placeItem(i.id)}>
                  Place
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
      {toBuy.length > 0 && (
        <div className="shopping">
          <p className="small">
            <strong>To buy in this layout:</strong> {toBuy.length} item{toBuy.length > 1 ? 's' : ''}
            {total > 0 && <> · ${total.toLocaleString()}</>}
          </p>
        </div>
      )}
    </section>
  )
}

export function LayoutsPanel() {
  const doc = useStore((s) => s.doc)
  const s = useStore.getState()
  const view = useStore((st) => st.view)
  const compare = useStore((st) => st.compare)
  const [renaming, setRenaming] = useState<string | null>(null)
  return (
    <section className="panel">
      <h3>Layouts</h3>
      <ul className="list layouts">
        {doc.layouts.map((l) => (
          <li key={l.id} className={l.id === doc.activeLayoutId ? 'active' : undefined}>
            {renaming === l.id ? (
              <input
                autoFocus
                defaultValue={l.name}
                onBlur={(e) => {
                  s.renameLayout(l.id, e.target.value || l.name)
                  setRenaming(null)
                }}
                onKeyDown={(e) => {
                  e.stopPropagation()
                  if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
                }}
              />
            ) : (
              <button className="layout-name" onClick={() => s.setActiveLayout(l.id)} onDoubleClick={() => setRenaming(l.id)} title="Click to open, double-click to rename">
                {l.name} <span className="muted">· {l.placements.length}</span>
              </button>
            )}
            <span className="row-actions">
              <button className="link" onClick={() => s.duplicateLayout(l.id)} title="Duplicate this layout to try a variant">
                Copy
              </button>
              {doc.layouts.length > 1 && (
                <button className="link danger" onClick={() => notifyDeleted(`Layout "${l.name}"`) && s.deleteLayout(l.id)}>
                  Delete
                </button>
              )}
            </span>
          </li>
        ))}
      </ul>
      <div className="btn-row">
        <button onClick={() => s.addLayout()}>New layout</button>
        {doc.layouts.length > 1 && (
          <button
            className={view === 'compare' ? 'primary' : undefined}
            onClick={() => {
              if (view === 'compare') return s.setView('split')
              const other = doc.layouts.find((l) => l.id !== doc.activeLayoutId)!
              s.setUI({ view: 'compare', compare: compare ?? [doc.activeLayoutId, other.id] })
            }}
          >
            {view === 'compare' ? 'Exit compare' : 'Compare'}
          </button>
        )}
      </div>
    </section>
  )
}

// Every delete can be undone, so deletes don't ask first; they say how to undo instead.
function notifyDeleted(what: string) {
  useStore.getState().notify(`${what} deleted. Press Ctrl+Z to undo.`)
  return true
}

// ---------------------------------------------------------------- right: inspector + issues

export function Inspector() {
  const selection = useStore((s) => s.selection)
  const doc = useStore((s) => s.doc)
  if (!selection)
    return (
      <section className="panel">
        <h3>Apartment</h3>
        <ApartmentInfo />
      </section>
    )
  if (selection.kind === 'placement') return <PlacementInspector id={selection.id} />
  if (selection.kind === 'room') {
    const room = doc.rooms.find((r) => r.id === selection.id)
    return room ? <RoomInspector id={room.id} /> : null
  }
  if (selection.kind === 'scan') return <ScanInspector id={selection.id} />
  return <OpeningInspector roomId={selection.roomId} id={selection.id} />
}

function ApartmentInfo() {
  const doc = useStore((s) => s.doc)
  const s = useStore.getState()
  const units = doc.apartment.units
  const area = doc.rooms.reduce((a, r) => a + polygonArea(r.polygon), 0)
  return (
    <>
      <TextField label="Name" value={doc.apartment.name} onChange={(n) => s.renameApartment(n)} />
      <p className="small muted">
        {doc.rooms.length} room{doc.rooms.length === 1 ? '' : 's'} · {units === 'metric' ? `${area.toFixed(1)} m²` : `${Math.round(area * 10.7639)} ft²`} · {doc.items.length} items
      </p>
      <ul className="list">
        {doc.rooms.map((r) => (
          <li key={r.id}>
            <button className="link" onClick={() => s.select({ kind: 'room', id: r.id })}>
              {r.name}
            </button>
          </li>
        ))}
      </ul>
      <p className="small muted">Select a room, wall opening or piece of furniture to edit it. Mark one door as the entrance to get walking path checks.</p>
    </>
  )
}

function PlacementInspector({ id }: { id: string }) {
  const doc = useStore((s) => s.doc)
  const s = useStore.getState()
  const units = doc.apartment.units
  const p = activeLayout(doc).placements.find((x) => x.id === id)
  const item = p && doc.items.find((i) => i.id === p.itemId)
  if (!p || !item) return null
  const upd = (patch: Partial<Item>) => s.updateItem(item.id, patch)
  const entry = catalogEntry(item.category)
  const usedIn = doc.layouts.filter((l) => l.placements.some((x) => x.itemId === item.id)).length
  const otherLayouts = usedIn - 1
  return (
    <section className="panel">
      <h3>{item.name}</h3>
      {item.imageUrl && <Photo className="inspector-photo" src={item.imageUrl} />}
      <TextField label="Name" value={item.name} onChange={(name) => upd({ name })} />
      <label className="field">
        <span>Type</span>
        <select value={item.category} onChange={(e) => upd({ category: e.target.value, floorLevel: catalogEntry(e.target.value).floorLevel })}>
          {CATALOG.map((c) => (
            <option key={c.category} value={c.category}>
              {c.name}
            </option>
          ))}
        </select>
      </label>
      {entry.presets && (
        <div className="presets">
          {entry.presets.map((pr) => (
            <button key={pr.name} className={Math.abs(pr.width - item.width) < 0.01 && Math.abs(pr.depth - item.depth) < 0.01 ? 'primary small' : 'small'} onClick={() => upd({ width: pr.width, depth: pr.depth })}>
              {pr.name}
            </button>
          ))}
        </div>
      )}
      <div className="grid3">
        <LengthField label="Width" value={item.width} units={units} onChange={(width) => upd({ width, sizeEstimated: undefined })} min={0.01} />
        <LengthField label="Depth" value={item.depth} units={units} onChange={(depth) => upd({ depth, sizeEstimated: undefined })} min={0.01} />
        <LengthField label="Height" value={item.height} units={units} onChange={(height) => upd({ height, sizeEstimated: undefined })} min={0.005} />
      </div>
      {item.sizeEstimated && (
        <p className="small">
          <span className="badge warning">Estimated size</span> Measure the real piece and type its size here.
        </p>
      )}
      <div className="grid2">
        <label className="field">
          <span>Rotation</span>
          <span className="field-input">
            <input
              value={Math.round((p.rotation * 180) / Math.PI)}
              onChange={(e) => {
                const n = Number(e.target.value)
                if (Number.isFinite(n)) s.updatePlacement(p.id, { rotation: ((((n % 360) + 360) % 360) * Math.PI) / 180 })
              }}
              onKeyDown={(e) => e.stopPropagation()}
            />
            <em>°</em>
          </span>
        </label>
        <label className="field">
          <span>Color</span>
          <input type="color" value={item.color} onChange={(e) => upd({ color: e.target.value })} />
        </label>
      </div>
      <details open>
        <summary>Clearance</summary>
        <div className="grid2">
          {(['front', 'back', 'left', 'right'] as const).map((side) => (
            <LengthField key={side} label={side[0].toUpperCase() + side.slice(1)} value={item.clearance[side]} units={units} onChange={(m) => upd({ clearance: { ...item.clearance, [side]: m } })} />
          ))}
        </div>
      </details>
      <details>
        <summary>Shopping</summary>
        <label className="field checkbox">
          <input type="checkbox" checked={!!item.owned} onChange={(e) => upd({ owned: e.target.checked })} />
          <span>I already own this</span>
        </label>
        <NumberField label="Price" value={item.price} onChange={(price) => upd({ price })} suffix="$" />
        <TextField label="Link" value={item.link ?? ''} onChange={(link) => upd({ link: link || undefined })} />
        <TextField label="3D model URL (GLB)" value={item.glbUrl ?? ''} onChange={(glbUrl) => upd({ glbUrl: glbUrl || undefined })} />
      </details>
      <div className="btn-row">
        <button onClick={() => s.rotatePlacement(p.id, Math.PI / 12)}>Rotate 15°</button>
        <button onClick={() => s.rotatePlacement(p.id, Math.PI / 2)}>90°</button>
        <button onClick={() => s.duplicatePlacement(p.id)}>Duplicate</button>
      </div>
      <div className="btn-row">
        <button onClick={() => s.deletePlacement(p.id)}>Remove from layout</button>
        <button className="danger" onClick={() => notifyDeleted(`"${item.name}"`) && s.deleteItem(item.id)}>
          Delete item
        </button>
      </div>
      <p className="small muted">
        Size and clearance belong to the item, so they're shared by every layout that uses it
        {otherLayouts > 0 ? ` (${otherLayouts} other layout${otherLayouts > 1 ? 's' : ''} here). Delete item removes it from all of them.` : '.'}
      </p>
    </section>
  )
}

function RoomInspector({ id }: { id: string }) {
  const doc = useStore((s) => s.doc)
  const s = useStore.getState()
  const units = doc.apartment.units
  const room = doc.rooms.find((r) => r.id === id)
  if (!room) return null
  return (
    <section className="panel">
      <h3>{room.name}</h3>
      <TextField label="Name" value={room.name} onChange={(name) => s.updateRoom(room.id, { name })} />
      <div className="grid2">
        <LengthField label="Ceiling height" value={room.ceilingHeight} units={units} onChange={(ceilingHeight) => s.updateRoom(room.id, { ceilingHeight })} min={1} />
        <LengthField label="Wall thickness" value={room.wallThickness} units={units} onChange={(wallThickness) => s.updateRoom(room.id, { wallThickness })} min={0.02} />
      </div>
      <h4>Walls</h4>
      <ol className="walls">
        {wallSegments(room).map(([a, b], i) => (
          <li key={i}>
            <LengthField label={`Wall ${i + 1}`} value={dist(a, b)} units={units} onChange={(l) => s.setWallLength(room.id, i, l)} min={0.05} />
          </li>
        ))}
      </ol>
      <p className="small muted">Changing a wall's length moves the next corner along it. Drag corners on the plan, drag a wall's midpoint dot to add a corner, double-click a corner to delete it.</p>
      <h4>Doors and windows</h4>
      <ul className="list">
        {room.openings.map((o) => (
          <li key={o.id}>
            <button className="link" onClick={() => s.select({ kind: 'opening', roomId: room.id, id: o.id })}>
              {o.kind === 'door' ? (o.entrance ? 'Entrance door' : 'Door') : 'Window'} on wall {o.wall + 1}
            </button>
            <span className="muted">{formatLength(o.width, units)}</span>
          </li>
        ))}
      </ul>
      <div className="btn-row">
        <button onClick={() => s.setMode('door')}>Add door</button>
        <button onClick={() => s.setMode('window')}>Add window</button>
      </div>
      <div className="btn-row">
        <button className="danger" onClick={() => notifyDeleted(`Room "${room.name}"`) && s.deleteRoom(room.id)}>
          Delete room
        </button>
      </div>
    </section>
  )
}

function OpeningInspector({ roomId, id }: { roomId: string; id: string }) {
  const doc = useStore((s) => s.doc)
  const s = useStore.getState()
  const units = doc.apartment.units
  const room = doc.rooms.find((r) => r.id === roomId)
  const o = room?.openings.find((x) => x.id === id)
  if (!room || !o) return null
  const wallLen = (() => {
    const [a, b] = wallSegments(room)[o.wall]
    return dist(a, b)
  })()
  const upd = (patch: Parameters<typeof s.updateOpening>[2]) => s.updateOpening(room.id, o.id, patch)
  return (
    <section className="panel">
      <h3>
        {o.kind === 'door' ? 'Door' : 'Window'} · {room.name}
      </h3>
      <div className="grid2">
        <LengthField label="Width" value={o.width} units={units} onChange={(width) => upd({ width: Math.min(width, wallLen), offset: Math.min(o.offset, Math.max(0, wallLen - width)) })} min={0.2} />
        <LengthField label="Height" value={o.height} units={units} onChange={(height) => upd({ height })} min={0.2} />
        <LengthField label="From wall start" value={o.offset} units={units} onChange={(offset) => upd({ offset: Math.min(offset, wallLen - o.width) })} />
        {o.kind === 'window' && <LengthField label="Sill height" value={o.sill} units={units} onChange={(sill) => upd({ sill })} />}
      </div>
      {o.kind === 'door' && (
        <>
          <div className="btn-row">
            <button onClick={() => upd({ hinge: o.hinge === 'end' ? 'start' : 'end' })}>Flip hinge</button>
            <button onClick={() => upd({ swing: o.swing === 'out' ? 'in' : 'out' })}>Swing {o.swing === 'out' ? 'in' : 'out'}</button>
          </div>
          <label className="field checkbox">
            <input type="checkbox" checked={!!o.entrance} onChange={(e) => upd({ entrance: e.target.checked })} />
            <span>Front door (start of walking paths)</span>
          </label>
        </>
      )}
      <p className="small muted">Drag it along the wall on the plan.</p>
      <div className="btn-row">
        <button className="danger" onClick={() => s.deleteOpening(room.id, o.id)}>
          Delete {o.kind}
        </button>
      </div>
    </section>
  )
}

export function IssuesPanel() {
  const doc = useStore((s) => s.doc)
  const layout = activeLayout(doc)
  const { issues, paths } = useAnalysis(doc, layout)
  const units = doc.apartment.units
  const select = useStore((s) => s.select)
  const errors = issues.filter((i) => i.severity === 'error').length
  const warnings = issues.filter((i) => i.severity === 'warning').length
  return (
    <section className="panel">
      <h3>
        Checks{' '}
        <span className={`badge ${errors ? 'error' : warnings ? 'warning' : 'ok'}`}>{errors + warnings === 0 ? 'All clear' : `${errors + warnings}`}</span>
      </h3>
      {issues.length === 0 && <p className="small muted">No overlaps, blocked clearances or tight paths.</p>}
      <ul className="issues">
        {issues.map((i) => (
          <li key={i.id} className={i.severity}>
            <button className="link" onClick={() => i.placementIds[0] && select({ kind: 'placement', id: i.placementIds[0] })}>
              {i.message}
            </button>
          </li>
        ))}
      </ul>
      <h4>Walking paths</h4>
      {!paths.entrance ? (
        <p className="small muted">Select a door and tick "Front door" to check walking paths.</p>
      ) : (
        <ul className="paths">
          {paths.results.map((r) => (
            <li key={r.roomId}>
              <span>{r.roomName}</span>
              <span className={!r.reachable ? 'error' : r.narrowest < MIN_WALKWAY ? 'error' : r.narrowest < COMFORTABLE_WALKWAY ? 'warning' : 'ok'}>
                {!r.reachable ? 'blocked' : r.narrowest >= ARCH ? 'clear' : `narrowest ${formatLength(r.narrowest, units)}`}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
