import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { CATALOG, catalogEntry } from '../model/catalog'
import { uid } from '../model/sample'
import { itemFromStarter, searchStarter, STARTER_GROUPS, type StarterProduct } from '../model/starterCatalog'
import type { Units } from '../model/types'
import { formatLength } from '../model/units'
import { importLink, importPhoto, importStatus, proxiedModelUrl, type ImportStatus } from '../import/client'
import { guessCategory, parseDimensions } from '../import/dimensions'
import { preparePhoto } from '../import/photo'
import { nameFromUrl } from '../import/productPage'
import { draftFromInfo, itemFromDraft, type Draft } from '../import/draft'
import { useStore } from '../store'
import { LengthField, NumberField, TextField } from './fields'

export function Modal({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        className={`modal ${wide ? 'wide' : ''}`}
        role="dialog"
        aria-label={title}
        onKeyDown={(e) => {
          // Keep typing in the dialog from triggering plan shortcuts.
          e.stopPropagation()
          if (e.key === 'Escape') onClose()
        }}
      >
        <header>
          <h2>{title}</h2>
          <button className="link" onClick={onClose} aria-label="Close">
            Close
          </button>
        </header>
        {children}
      </div>
    </div>
  )
}

/** A product photo that quietly disappears if the store doesn't let us show it. */
export function Photo({ src, className, fallback = null }: { src: string; className?: string; fallback?: ReactNode }) {
  const [failed, setFailed] = useState<string | null>(null)
  if (failed === src) return <>{fallback}</>
  return <img className={className} src={src} alt="" referrerPolicy="no-referrer" onError={() => setFailed(src)} />
}

const size = (w: number, d: number, h: number, units: Units) => `${formatLength(w, units)} × ${formatLength(d, units)} × ${formatLength(h, units)}`

// ---------------------------------------------------------------- starter catalog

export function CatalogBrowser({ onClose }: { onClose: () => void }) {
  const units = useStore((s) => s.doc.apartment.units)
  const [query, setQuery] = useState('')
  const [group, setGroup] = useState<string | undefined>()
  const results = useMemo(() => searchStarter(query, group), [query, group])
  const add = (p: StarterProduct) => {
    useStore.getState().addItem(itemFromStarter(p, uid()))
    useStore.getState().notify(`Added ${p.name.toLowerCase()}. Drag it into place.`)
    onClose()
  }
  const grouped = STARTER_GROUPS.map((g) => [g, results.filter((r) => r.group === g)] as const).filter(([, rs]) => rs.length)
  return (
    <Modal title="Furniture catalog" onClose={onClose} wide>
      <div className="catalog-search">
        <input autoFocus placeholder="Search: queen bed, 8x10 rug, desk…" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && results[0] && add(results[0])} />
      </div>
      <div className="chips">
        <button className={!group ? 'primary small' : 'small'} onClick={() => setGroup(undefined)}>
          All
        </button>
        {STARTER_GROUPS.map((g) => (
          <button key={g} className={group === g ? 'primary small' : 'small'} onClick={() => setGroup(group === g ? undefined : g)}>
            {g}
          </button>
        ))}
      </div>
      <div className="catalog-results">
        {grouped.length === 0 && <p className="muted">Nothing matches. Try a simpler word, or add it from a link.</p>}
        {grouped.map(([g, rs]) => (
          <section key={g}>
            <h4>{g}</h4>
            <ul className="product-list">
              {rs.map((p) => (
                <li key={p.id}>
                  <button className="product" onClick={() => add(p)} title={`Add a ${p.name.toLowerCase()}`}>
                    <span className="swatch" style={{ background: p.color ?? catalogEntry(p.category).color }} />
                    <span className="product-name">{p.name}</span>
                    <span className="muted small">{size(p.width, p.depth, p.height, units)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      <p className="small muted">Sizes are typical retail sizes (width × depth × height). Every size is editable after you add it.</p>
    </Modal>
  )
}

// ---------------------------------------------------------------- import from a link, photo or text

type Tab = 'link' | 'photo' | 'text'

export function ImportDialog({ onClose, initialTab = 'link' }: { onClose: () => void; initialTab?: Tab }) {
  const [tab, setTab] = useState<Tab>(initialTab)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [status, setStatus] = useState<ImportStatus | null>(null)
  useEffect(() => {
    importStatus().then(setStatus)
  }, [])
  const add = () => {
    if (!draft) return
    useStore.getState().addItem(itemFromDraft(draft))
    useStore.getState().notify(`Added "${draft.name}". Drag it into place.`)
    onClose()
  }
  return (
    <Modal title="Add from a link, photo or text" onClose={onClose} wide>
      <div className="tabs" role="tablist">
        {(
          [
            ['link', 'Product link'],
            ['photo', 'Photo'],
            ['text', 'Paste text'],
          ] as const
        ).map(([t, label]) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'tab active' : 'tab'} onClick={() => setTab(t)}>
            {label}
          </button>
        ))}
      </div>
      <div className="import-body">
        <div className="import-source">
          {tab === 'link' && <LinkTab status={status} onDraft={setDraft} />}
          {tab === 'photo' && <PhotoTab status={status} onDraft={setDraft} />}
          {tab === 'text' && <TextTab onDraft={setDraft} />}
        </div>
        <div className="import-draft">{draft ? <DraftForm draft={draft} onChange={setDraft} onAdd={add} /> : <p className="muted small">The item shows up here for you to check before it goes on the plan.</p>}</div>
      </div>
    </Modal>
  )
}

function ServerNote({ status, need }: { status: ImportStatus | null; need: 'server' | 'llm' }) {
  if (!status) return null
  if (!status.server)
    return (
      <p className="note small">
        {need === 'llm' ? "Photos can't be recognized" : "Pages can't be read"} automatically here.
        {need === 'llm' ? ' You can still add the photo and type its size.' : ' Paste the size from the page in the box below, or type it on the right.'}
      </p>
    )
  if (need === 'llm' && !status.llm)
    return (
      <p className="note small">
        Photo recognition isn't available here. You can still add the photo and type its size.
      </p>
    )
  return null
}

function LinkTab({ status, onDraft }: { status: ImportStatus | null; onDraft: (d: Draft) => void }) {
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pasted, setPasted] = useState('')
  const offline = status && !status.server
  const read = async () => {
    const u = url.trim()
    if (!u) return
    setError(null)
    const full = /^https?:\/\//i.test(u) ? u : `https://${u}`
    if (offline) {
      const name = nameFromUrl(full)
      onDraft(draftFromInfo({ url: full, name, category: guessCategory(name ?? ''), dims: {} }, 'link'))
      return
    }
    setBusy(true)
    try {
      const info = await importLink(full)
      onDraft(draftFromInfo({ ...info, glbUrl: info.glbUrl && proxiedModelUrl(info.glbUrl) }, 'link'))
    } catch (e) {
      setError((e as Error).message)
      const name = nameFromUrl(full)
      onDraft(draftFromInfo({ url: full, name, category: guessCategory(name ?? ''), dims: {} }, 'link'))
    } finally {
      setBusy(false)
    }
  }
  const applyPasted = (text: string) => {
    setPasted(text)
    const d = parseDimensions(text)
    if (!d) return
    const name = nameFromUrl(url.trim())
    onDraft(draftFromInfo({ url: url.trim() || undefined, name, category: guessCategory(`${name ?? ''} ${text}`), dims: d, dimsSource: 'listed', dimsText: d.text }, 'link'))
  }
  return (
    <>
      <p className="small muted">Paste a link to a product page. The name, size, price and picture are read from the page, and the store's 3D model is used when it publishes one.</p>
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault()
          read()
        }}
      >
        <input autoFocus type="url" inputMode="url" placeholder="https://www.example.com/products/sofa" value={url} onChange={(e) => setUrl(e.target.value)} />
        <button type="submit" className="primary" disabled={busy || !url.trim()}>
          {busy ? 'Reading…' : 'Read page'}
        </button>
      </form>
      {busy && <p className="small muted">Reading the page. If the store blocks it, Claude reads it instead, which takes up to a minute.</p>}
      {error && <p className="error small">{error}</p>}
      <ServerNote status={status} need="server" />
      {(offline || error) && (
        <label className="field">
          <span>Size from the page</span>
          <textarea rows={3} placeholder={'e.g. 84"W x 38"D x 34"H'} value={pasted} onChange={(e) => applyPasted(e.target.value)} />
        </label>
      )}
    </>
  )
}

function PhotoTab({ status, onDraft }: { status: ImportStatus | null; onDraft: (d: Draft) => void }) {
  const [thumb, setThumb] = useState<string | null>(null)
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const photo = useRef<{ upload: string; thumbnail: string; color: string } | null>(null)

  const recognize = async () => {
    const p = photo.current
    if (!p) return
    const fallback = () => draftFromInfo({ dims: {}, category: guessCategory(note) ?? 'armchair', name: note.trim() || undefined, color: p.color }, 'photo', { imageUrl: p.thumbnail })
    if (!status?.llm) return onDraft(fallback())
    setBusy(true)
    setError(null)
    try {
      const info = await importPhoto(p.upload, note.trim() || undefined)
      onDraft(draftFromInfo({ ...info, color: info.color ?? p.color }, 'photo', { imageUrl: p.thumbnail }))
    } catch (e) {
      setError((e as Error).message)
      onDraft(fallback())
    } finally {
      setBusy(false)
    }
  }

  const choose = async (file: File | undefined) => {
    if (!file) return
    setError(null)
    try {
      photo.current = await preparePhoto(file)
      setThumb(photo.current.thumbnail)
      await recognize()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <>
      <p className="small muted">Add a photo of a piece you own or are considering, or a screenshot of a listing. The type and a size estimate are filled in for you to check; listed sizes in a screenshot are read exactly.</p>
      <div
        className={`dropzone ${over ? 'over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          choose(e.dataTransfer.files[0])
        }}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
      >
        {thumb ? <img src={thumb} alt="Your photo" /> : <span>Drop a photo here, or click to choose one</span>}
        <input ref={input} type="file" accept="image/*" hidden onChange={(e) => choose(e.target.files?.[0])} />
      </div>
      <label className="field">
        <span>Anything to know? (optional)</span>
        <input placeholder="e.g. mid-century armchair, about 80 cm wide" value={note} onChange={(e) => setNote(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && recognize()} />
      </label>
      {thumb && (
        <button onClick={recognize} disabled={busy}>
          {busy ? 'Looking at the photo…' : status?.llm ? 'Look again with this note' : 'Use this note'}
        </button>
      )}
      {error && <p className="error small">{error}</p>}
      <ServerNote status={status} need="llm" />
    </>
  )
}

function TextTab({ onDraft }: { onDraft: (d: Draft) => void }) {
  const [text, setText] = useState('')
  const [found, setFound] = useState<string | null>(null)
  const parse = (t: string) => {
    setText(t)
    const lines = t.split('\n').map((l) => l.trim()).filter(Boolean)
    const d = parseDimensions(t)
    setFound(d?.text ?? null)
    if (!lines.length) return
    const url = t.match(/https?:\/\/\S+/)?.[0]
    // The first line that isn't the size or a link is usually the product name.
    const nameLine = lines.find((l) => !/https?:\/\//.test(l) && (!d || !l.includes(d.text)) && l.length < 120)
    const category = guessCategory(nameLine ?? '') ?? guessCategory(t)
    onDraft(draftFromInfo({ name: nameLine, url, category, dims: d ? { width: d.width, depth: d.depth, height: d.height } : {}, dimsSource: d ? 'listed' : undefined, dimsText: d?.text }, 'custom'))
  }
  return (
    <>
      <p className="small muted">Paste the product name and its dimensions from anywhere: a listing, an email, a spec sheet. Works without the import server.</p>
      <textarea autoFocus rows={8} placeholder={'Harbor 3-seat sofa\nOverall: 84" W x 38" D x 34" H'} value={text} onChange={(e) => parse(e.target.value)} />
      {text && <p className="small muted">{found ? <>Size read from: “{found}”</> : 'No size found yet. Include width, depth and height, e.g. 213 x 96 x 85 cm.'}</p>}
    </>
  )
}

function DraftForm({ draft, onChange, onAdd }: { draft: Draft; onChange: (d: Draft) => void; onAdd: () => void }) {
  const units = useStore((s) => s.doc.apartment.units)
  const upd = (p: Partial<Draft>) => onChange({ ...draft, ...p })
  const updSize = (p: Partial<Draft>) => onChange({ ...draft, ...p, estimated: false })
  return (
    <div className="draft">
      <div className="draft-head">
        {draft.imageUrl ? (
          <Photo className="thumb" src={draft.imageUrl} fallback={<span className="thumb swatch-lg" style={{ background: draft.color }} />} />
        ) : (
          <span className="thumb swatch-lg" style={{ background: draft.color }} />
        )}
        <div>
          <TextField label="Name" value={draft.name} onChange={(name) => upd({ name })} />
          <label className="field">
            <span>Type</span>
            <select
              value={draft.category}
              onChange={(e) => {
                const category = e.target.value
                // An estimated size follows the type: a typical bed, not a typical armchair.
                const base = catalogEntry(category)
                upd(draft.estimated && !draft.dimsText ? { category, width: base.width, depth: base.depth, height: base.height } : { category })
              }}
            >
              {CATALOG.map((c) => (
                <option key={c.category} value={c.category}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div className="grid3">
        <LengthField label="Width" value={draft.width} units={units} onChange={(width) => updSize({ width })} min={0.01} />
        <LengthField label="Depth" value={draft.depth} units={units} onChange={(depth) => updSize({ depth })} min={0.01} />
        <LengthField label="Height" value={draft.height} units={units} onChange={(height) => updSize({ height })} min={0.005} />
      </div>
      {draft.estimated ? (
        <p className="badge warning">Estimated size. Check it against the real piece before you rely on it.</p>
      ) : draft.dimsText ? (
        <p className="small muted">From the listing: “{draft.dimsText}”</p>
      ) : null}
      {draft.notes.map((n, i) => (
        <p key={i} className="small muted">
          {n}
        </p>
      ))}
      <div className="grid2">
        <NumberField label="Price" value={draft.price} onChange={(price) => upd({ price })} suffix="$" />
        <label className="field">
          <span>Color</span>
          <input type="color" value={draft.color} onChange={(e) => upd({ color: e.target.value })} />
        </label>
      </div>
      {draft.glbUrl && (
        <label className="field checkbox">
          <input type="checkbox" checked={draft.useGlb} onChange={(e) => upd({ useGlb: e.target.checked })} />
          <span>Use the store's 3D model (scaled to this size)</span>
        </label>
      )}
      <label className="field checkbox">
        <input type="checkbox" checked={draft.owned} onChange={(e) => upd({ owned: e.target.checked })} />
        <span>I already own this</span>
      </label>
      <div className="btn-row">
        <button className="primary" onClick={onAdd}>
          Add to plan
        </button>
      </div>
    </div>
  )
}
