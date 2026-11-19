import { useEffect, useRef, useState, type ReactNode } from 'react'
import { PlanView } from '../components/PlanView'
import { formatLength, parseLength } from '../model/units'
import { calibrate, planToScan, toScanData } from '../scan/align'
import { SCAN_ACCEPT, isScanFileName } from '../scan/importers'
import { importScanFile, scanCenter, scanEntry, useScanRuntime } from '../scan/runtime'
import { SAMPLE_TRUTH, samplePly } from '../scan/sampleScan'
import { placementShift } from '../scan/apply'
import { reviewChoice, useScanUI, type ReviewObject } from '../scan/scanUi'
import { loadSample, newEmptyDoc, useStore } from '../store'
import { useGame } from './gameStore'
import { IconCheck, IconHome, IconRuler, IconScan, IconSparkle } from './icons'
import './onboarding.css'

// First-run "move in" flow: welcome → how to scan + import → scale from one tape length →
// find rooms and furniture → welcome home. Everything goes through the same store actions as
// the editor's Scans panel and scan review, so what the player does here is ordinary edits.

type Step = 'welcome' | 'scan' | 'measure' | 'detect' | 'done'
const STEPS: Step[] = ['welcome', 'scan', 'measure', 'detect', 'done']

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))

/** Where a new scan goes: right of the rooms already drawn (same rule as the Scans panel). */
function placeNextToRooms() {
  const pts = useStore.getState().doc.rooms.flatMap((r) => r.polygon)
  if (!pts.length) return { x: 0, y: 0 }
  return { x: Math.max(...pts.map((p) => p.x)) + 0.8, y: Math.min(...pts.map((p) => p.y)) }
}

function finish() {
  const s = useStore.getState()
  if (s.mode !== 'select') s.setMode('select')
  s.setUI({ scanDraft: null, selection: null })
  useGame.getState().setMode('buy')
  useGame.getState().setPhase('play')
}

export function Onboarding() {
  const [step, setStep] = useState<Step>('welcome')
  const [scanId, setScanId] = useState<string | null>(null)
  const idx = STEPS.indexOf(step)

  const back = () => {
    if (step === 'scan') setStep('welcome')
    else if (step === 'measure') setStep('scan')
    else if (step === 'detect') {
      useScanUI.getState().cancel()
      setStep('measure')
    }
  }
  const canBack = step === 'scan' || step === 'measure' || step === 'detect'

  return (
    <div className="onb" role="dialog" aria-label="Move in">
      <div className="onb-bg" aria-hidden />
      <header className="onb-top">
        <button className={`g-icon sm onb-back${canBack ? '' : ' hide'}`} onClick={back} aria-label="Back" disabled={!canBack} tabIndex={canBack ? 0 : -1}>
          <IconChevron />
        </button>
        <Progress idx={idx} />
        <span className="onb-top-spacer" />
      </header>
      <main className="onb-stage" key={step}>
        {step === 'welcome' && <Welcome onScan={() => setStep('scan')} />}
        {step === 'scan' &&
          <ScanStep
            onImported={(id) => {
              setScanId(id)
              setStep('measure')
            }}
          />
        }
        {step === 'measure' && scanId && <MeasureStep scanId={scanId} onNext={() => setStep('detect')} />}
        {step === 'detect' && scanId && <DetectStep scanId={scanId} onNext={() => setStep('done')} />}
        {step === 'done' && <Done />}
      </main>
    </div>
  )
}

// ---------------------------------------------------------------- chrome

const STEP_NAMES = ['Hello', 'Scan', 'Size', 'Rooms', 'Home']

/** A row of pips that fill in, with a little house at the end. */
function Progress({ idx }: { idx: number }) {
  return (
    <div className="onb-progress" role="progressbar" aria-valuemin={1} aria-valuemax={STEPS.length} aria-valuenow={idx + 1} aria-valuetext={STEP_NAMES[idx]}>
      {STEPS.slice(0, -1).map((s, i) => (
        <span key={s} className={`onb-pip${i < idx ? ' done' : i === idx ? ' now' : ''}`} />
      ))}
      <span className={`onb-pip-home${idx === STEPS.length - 1 ? ' now' : ''}`}>
        <IconHome />
      </span>
    </div>
  )
}

function IconChevron() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.6} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="m15 5-7 7 7 7" />
    </svg>
  )
}

function Title({ kicker, children, sub }: { kicker?: string; children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="onb-title">
      {kicker && <div className="onb-kicker">{kicker}</div>}
      <h1>{children}</h1>
      {sub && <p>{sub}</p>}
    </div>
  )
}

// ---------------------------------------------------------------- 1 · welcome

function Welcome({ onScan }: { onScan: () => void }) {
  return (
    <div className="onb-welcome">
      <HouseArt />
      <Title sub="Scan your place, then make it beautiful.">Let’s move in</Title>
      <div className="onb-actions">
        <button className="g-btn primary big onb-wide" onClick={onScan}>
          <IconScan /> Scan my apartment
        </button>
        <div className="onb-alt">
          <button
            className="onb-link"
            onClick={() => {
              loadSample()
              finish()
            }}
          >
            Try a sample home
          </button>
          <span aria-hidden>·</span>
          <button
            className="onb-link"
            onClick={() => {
              newEmptyDoc()
              useStore.getState().addRect('Living room', 4, 3.5)
              finish()
            }}
          >
            Start with an empty room
          </button>
        </div>
      </div>
    </div>
  )
}

/** A cozy little house with a lit window and a puff of smoke. */
function HouseArt({ small }: { small?: boolean }) {
  return (
    <svg className={`onb-house${small ? ' sm' : ''}`} viewBox="0 0 200 160" aria-hidden>
      <ellipse cx="100" cy="148" rx="78" ry="9" fill="rgba(70,50,30,0.10)" />
      <g className="onb-smoke">
        <circle cx="142" cy="30" r="6" fill="#fff" opacity="0.9" />
        <circle cx="150" cy="18" r="8" fill="#fff" opacity="0.7" />
        <circle cx="160" cy="6" r="5" fill="#fff" opacity="0.5" />
      </g>
      <rect x="132" y="36" width="16" height="30" rx="3" fill="#c65f45" />
      <path d="M100 22 172 80H28z" fill="var(--g-coral)" stroke="var(--g-coral-dk)" strokeWidth="4" strokeLinejoin="round" />
      <rect x="42" y="76" width="116" height="70" rx="6" fill="var(--g-paper)" stroke="#e6d9c6" strokeWidth="3" />
      <rect x="58" y="92" width="34" height="28" rx="5" fill="var(--g-sun)" className="onb-window" />
      <path d="M75 92v28M58 106h34" stroke="#e6a52c" strokeWidth="3" />
      <rect x="110" y="98" width="30" height="48" rx="5" fill="var(--g-sage)" />
      <circle cx="133" cy="123" r="2.6" fill="#fff" />
      <path d="M52 146c4-12 14-12 18 0M150 146c3-9 10-9 13 0" fill="#7cbf8e" />
    </svg>
  )
}

// ---------------------------------------------------------------- 2 · how to scan + import

function ScanStep({ onImported }: { onImported: (id: string) => void }) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<{ name: string; message: string } | null>(null)
  const [over, setOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  // The player can go back while a big scan is still decoding; don't jump forward after that.
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])
  const progress = useScanRuntime((s) => Object.values(s.entries).find((e) => e.status === 'loading')?.message)

  const run = async (file: { name: string; type?: string; blob: Blob }) => {
    setError(null)
    setBusy(file.name)
    try {
      // A first scan is the whole home: start from a clean apartment.
      newEmptyDoc()
      const [ref] = await Promise.all([importScanFile(file, placeNextToRooms), wait(900)])
      useStore.getState().addScan(ref)
      if (alive.current) onImported(ref.id)
    } catch (err) {
      setError({ name: file.name, message: errText(err) })
      setBusy(null)
    }
  }
  const pick = (f: File | undefined) => {
    if (!f) return
    if (!isScanFileName(f.name)) return setError({ name: f.name, message: 'That isn’t a scan file. Look for one ending in .ply, .spz, .glb or .obj.' })
    void run({ name: f.name, type: f.type, blob: f })
  }

  if (busy)
    return (
      <div className="onb-center">
        <ScanningArt />
        <Title kicker="Unpacking boxes" sub={progress ?? 'Reading your scan…'}>
          Moving your scan in…
        </Title>
        <p className="onb-note">Big scans can take a few seconds. Finding the floor and squaring up the walls.</p>
      </div>
    )

  return (
    <div className="onb-scroll">
      <Title kicker="Step 1 · Scan" sub="Use the free Scaniverse app on your iPhone. No LiDAR needed.">
        Scan your place
      </Title>
      <ol className="onb-tips">
        <Tip n={1} art={<TipArtSplat />} title="Choose Splat mode">
          Start a new scan and pick <b>Splat</b>. It builds your room from the camera alone.
        </Tip>
        <Tip n={2} art={<TipArtWalk />} title="Walk the edges, slowly">
          Phone at chest height, circle the room along the walls. Catch the floor and every corner.
        </Tip>
        <Tip n={3} art={<TipArtExport />} title="Export PLY or SPZ">
          Share › Export as <b>PLY</b> or <b>SPZ</b>, then AirDrop it here or Save to Files.
        </Tip>
      </ol>

      <div
        className={`onb-drop${over ? ' over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          pick(Array.from(e.dataTransfer.files).find((x) => isScanFileName(x.name)) ?? e.dataTransfer.files[0])
        }}
      >
        {error ? (
          <div className="onb-error" role="alert">
            <div className="onb-error-face" aria-hidden>
              :(
            </div>
            <strong>Hmm, that scan didn’t unpack.</strong>
            <span>
              {error.name}: {error.message}
            </span>
          </div>
        ) : (
          <span className="onb-drop-hint">Drop your scan here, or</span>
        )}
        <button className="g-btn primary big onb-wide" onClick={() => fileRef.current?.click()}>
          {error ? 'Try another file' : 'Choose scan file'}
        </button>
        <span className="onb-formats">PLY · SPZ · GLB · OBJ</span>
      </div>
      <div className="onb-alt">
        <button className="onb-link" onClick={() => void run({ name: 'Sample living room.ply', type: 'application/octet-stream', blob: new Blob([samplePly() as BlobPart]) })}>
          No scan handy? Use our sample scan
        </button>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept={SCAN_ACCEPT}
        hidden
        onChange={(e) => {
          pick(e.target.files?.[0])
          e.target.value = ''
        }}
      />
    </div>
  )
}

function Tip({ n, art, title, children }: { n: number; art: ReactNode; title: string; children: ReactNode }) {
  return (
    <li className="onb-tip g-card" style={{ animationDelay: `${n * 70}ms` }}>
      <div className="onb-tip-art">{art}</div>
      <div>
        <div className="onb-tip-title">
          <span className="onb-tip-n">{n}</span>
          {title}
        </div>
        <p>{children}</p>
      </div>
    </li>
  )
}

function TipArtSplat() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden>
      <rect x="18" y="6" width="28" height="52" rx="7" fill="var(--g-ink)" />
      <rect x="21" y="11" width="22" height="42" rx="4" fill="#fdf3e4" />
      {[
        [27, 22, 4, 'var(--g-coral)'],
        [36, 27, 5, 'var(--g-sky)'],
        [29, 34, 6, 'var(--g-sun)'],
        [37, 40, 3.5, 'var(--g-sage)'],
        [27, 44, 3, 'var(--g-plum)'],
      ].map(([x, y, r, c], i) => (
        <circle key={i} cx={x} cy={y} r={r} fill={c as string} opacity={0.85} />
      ))}
    </svg>
  )
}

function TipArtWalk() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden>
      <rect x="8" y="8" width="48" height="48" rx="6" fill="#fdf3e4" stroke="#d9c7ad" strokeWidth="3" />
      <rect x="15" y="15" width="34" height="34" rx="5" fill="none" stroke="var(--g-coral)" strokeWidth="2.5" strokeDasharray="4 4" className="onb-walk-path" />
      <circle cx="15" cy="15" r="5" fill="var(--g-sky)" className="onb-walker" />
    </svg>
  )
}

function TipArtExport() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden>
      <path d="M16 26h-4v30h40V26h-4" fill="none" stroke="var(--g-ink)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M32 40V8M22 18l10-10 10 10" fill="none" stroke="var(--g-coral)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="20" y="44" width="24" height="7" rx="3.5" fill="var(--g-sun)" />
    </svg>
  )
}

/** A house outline with a scanning beam sweeping over it. */
function ScanningArt() {
  return (
    <div className="onb-scanning" aria-hidden>
      <HouseArt small />
      <div className="onb-beam" />
    </div>
  )
}

// ---------------------------------------------------------------- 3 · size it up

function MeasureStep({ scanId, onNext }: { scanId: string; onNext: () => void }) {
  const ref = useStore((s) => (s.doc.scans ?? []).find((r) => r.id === scanId))
  const units = useStore((s) => s.doc.apartment.units)
  const scanDraft = useStore((s) => s.scanDraft)
  const ready = useScanRuntime((s) => s.entries[scanId]?.status === 'ready')
  const [text, setText] = useState('')

  // Put the plan into the real measure mode for this scan while the step is open.
  useEffect(() => {
    const s = useStore.getState()
    s.setUI({ activeScanId: scanId, selection: null })
    s.setMode('scan-measure')
    return () => {
      const st = useStore.getState()
      st.setMode('select')
      st.setUI({ scanDraft: null })
    }
  }, [scanId])

  if (!ref) return null
  const draft = scanDraft?.scanId === scanId ? scanDraft : null
  const cal = calibrate(ref.references)
  const m = parseLength(text, units)
  const ok = m != null && m > 0.05
  const fmt = (v: number) => formatLength(v, units)

  const addLength = () => {
    if (!ok || !draft?.b) return
    const s = useStore.getState()
    const anchor = scanCenter(ref) ?? { x: ref.x, y: ref.y }
    s.addScanReference(ref.id, { a: planToScan(ref, draft.a), b: planToScan(ref, draft.b), length: m! }, anchor)
    s.setUI({ scanDraft: null })
    setText('')
  }

  const hint = !ready ? 'Loading your scan…' : draft?.b ? 'Nice! Now type its real length' : !draft ? (ref.references.length ? 'Tap the ends of another length' : 'Tap one end of something you measured') : 'Now tap the other end'
  const isSample = ref.name.startsWith('Sample living room')

  return (
    <div className="onb-measure">
      <Title kicker="Step 2 · Size it up" sub="Phone scans don’t know their true size. Give us one real length, measured with a tape. A whole wall is best.">
        How big is it really?
      </Title>

      <div className="onb-plan g-card">
        <PlanView />
        <div className={`onb-plan-hint${draft && !draft.b ? ' second' : ''}`}>
          <IconRuler /> {hint}
        </div>
        <div className="onb-units" role="group" aria-label="Units">
          <button className={`g-chip${units === 'metric' ? ' on' : ''}`} onClick={() => useStore.getState().setUnits('metric')}>
            m
          </button>
          <button className={`g-chip${units === 'imperial' ? ' on' : ''}`} onClick={() => useStore.getState().setUnits('imperial')}>
            ft
          </button>
        </div>
      </div>

      <div className="onb-measure-foot">
        {draft?.b ? (
          <form
            className="onb-length"
            onSubmit={(e) => {
              e.preventDefault()
              addLength()
            }}
            onKeyDown={(e) => {
              e.stopPropagation()
              if (e.key === 'Escape') useStore.getState().setUI({ scanDraft: null })
            }}
          >
            <label htmlFor="onb-length-input">
              How long is it really? <span>({fmt(Math.hypot(draft.b.x - draft.a.x, draft.b.y - draft.a.y))} on the scan)</span>
            </label>
            <div className="onb-length-row">
              <input
                id="onb-length-input"
                autoFocus
                inputMode="decimal"
                autoComplete="off"
                value={text}
                placeholder={units === 'metric' ? '412 cm or 4.12m' : `13' 6"`}
                onChange={(e) => setText(e.target.value)}
              />
              <button type="submit" className="g-btn primary" disabled={!ok}>
                <IconCheck /> Use it
              </button>
            </div>
            <button type="button" className="onb-link" onClick={() => useStore.getState().setUI({ scanDraft: null })}>
              Oops, tap again
            </button>
          </form>
        ) : (
          <>
            {ref.references.length > 0 && (
              <div className="onb-refs">
                {ref.references.map((r, i) => {
                  const dev = cal?.deviations[i] ?? 0
                  const bad = ref.references.length > 1 && Math.abs(dev) > 0.02
                  return (
                    <span key={r.id} className={`onb-ref${bad ? ' bad' : ''}`}>
                      <IconRuler /> {fmt(r.length)}
                      {ref.references.length > 1 && (
                        <small>
                          {dev >= 0 ? '+' : ''}
                          {(dev * 100).toFixed(1)}%
                        </small>
                      )}
                      <button aria-label="Remove this length" onClick={() => useStore.getState().deleteScanReference(ref.id, r.id, scanCenter(ref) ?? { x: ref.x, y: ref.y })}>
                        ×
                      </button>
                    </span>
                  )
                })}
              </div>
            )}
            {cal && <Confidence count={ref.references.length} spread={cal.spread} />}
            {!cal && isSample && <p className="onb-note">Psst, sample tape measures: {SAMPLE_TRUTH.references.map((r) => `${r.label.toLowerCase()} ${fmt(r.length)}`).join(', ')}.</p>}
            <div className="onb-row">
              <button className="g-btn ghost" onClick={onNext}>
                {cal ? 'That’s enough' : 'Skip for now'}
              </button>
              <button className="g-btn primary big" disabled={!cal} onClick={onNext}>
                Next
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function Confidence({ count, spread }: { count: number; spread: number }) {
  if (count === 1)
    return (
      <p className="onb-conf">
        <span className="onb-stars" aria-label="1 of 3 stars">
          ★<i>★★</i>
        </span>
        Sized! Add one more length to double-check it.
      </p>
    )
  const good = spread <= 0.02
  const stars = good ? (count >= 3 ? 3 : 2) : 1
  return (
    <p className={`onb-conf${good ? ' good' : ' warn'}`}>
      <span className="onb-stars" aria-label={`${stars} of 3 stars`}>
        {'★'.repeat(stars)}
        <i>{'★'.repeat(3 - stars)}</i>
      </span>
      {good ? `Your lengths agree within ${(spread * 100).toFixed(1)}%. Spot on!` : `These disagree by ${(spread * 100).toFixed(1)}%. Re-measure the red one, or check its ends sit on the walls.`}
    </p>
  )
}

// ---------------------------------------------------------------- 4 · find your rooms

function DetectStep({ scanId, onNext }: { scanId: string; onNext: () => void }) {
  const review = useScanUI((s) => s.review)
  const units = useStore((s) => s.doc.apartment.units)
  const [phase, setPhase] = useState<'ask' | 'running' | 'done' | 'failed'>(() => (useScanUI.getState().review?.estimate.scanId === scanId ? 'done' : 'ask'))
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setPhase('running')
    setError(null)
    try {
      const ref = (useStore.getState().doc.scans ?? []).find((r) => r.id === scanId)
      const points = scanEntry(scanId)?.loaded?.points
      if (!ref || !points) throw new Error('The scan isn’t loaded any more.')
      await Promise.all([useScanUI.getState().analyze(toScanData(ref, points)), wait(1400)])
      const r = useScanUI.getState().review
      if (!r || (!r.estimate.room && !r.estimate.objects.length)) setPhase('failed')
      else {
        followShift()
        setPhase('done')
      }
    } catch (e) {
      setError(errText(e))
      setPhase('failed')
    }
  }
  // Start right away: the player already said yes by getting here.
  const started = useRef(false)
  useEffect(() => {
    if (started.current || phase !== 'ask') return
    started.current = true
    void run()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const skip = () => {
    useScanUI.getState().cancel()
    onNext()
  }
  // Applying a review moves what it found next to the existing rooms (to the origin in an empty
  // home). Move the scan by the same amount so its points stay under the room and furniture.
  const followShift = () => {
    const r = useScanUI.getState().review
    const s = useStore.getState()
    const ref = (s.doc.scans ?? []).find((x) => x.id === scanId)
    const d = r && ref ? placementShift(s.doc, r.estimate, reviewChoice(r)) : null
    if (!d || !ref) return
    const dx = d.x - moved.current.x
    const dy = d.y - moved.current.y
    if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) return
    s.updateScan(scanId, { x: ref.x + dx, y: ref.y + dy })
    moved.current = d
  }
  const moved = useRef({ x: 0, y: 0 })

  if (phase === 'ask' || phase === 'running')
    return (
      <div className="onb-center">
        <DetectArt />
        <Title kicker="Step 3 · Rooms" sub="Looking for walls, doors and your furniture…">
          Peeking around…
        </Title>
        <button className="g-btn ghost sm" onClick={skip}>
          Skip this
        </button>
      </div>
    )

  if (phase === 'failed' || !review)
    return (
      <div className="onb-center">
        <div className="onb-shrug" aria-hidden>
          ¯\_(ツ)_/¯
        </div>
        <Title kicker="Step 3 · Rooms" sub="Your scan is in, at the right size. You can trace the walls over it yourself in Build mode.">
          Couldn’t spot the walls this time
        </Title>
        {error && <p className="onb-note">{error}</p>}
        <div className="onb-actions">
          <button className="g-btn primary big onb-wide" onClick={skip}>
            Trace walls myself later
          </button>
          <button className="onb-link" onClick={() => void run()}>
            Try detecting again
          </button>
        </div>
      </div>
    )

  const { estimate, objects } = review
  const accepted = objects.filter((o) => o.accepted)
  const room = estimate.room
  const roomOn = review.includeRoom && !!room
  const nothing = !roomOn && accepted.length === 0

  return (
    <div className="onb-detect">
      <Title kicker="Step 3 · Rooms" sub={summary(!!room, objects)}>
        Here’s what we found
      </Title>
      <div className="onb-preview g-card">
        <PlanView />
      </div>
      <div className="onb-found g-card">
        {room && (
          <label className="g-toggle onb-found-row room">
            <span>
              <span className="onb-found-ico">
                <IconHome />
              </span>
              {review.roomName || 'Room'}
              <small>
                {roomDetails(room.openings.map((o) => o.kind), formatLength(room.ceilingHeight, units))}
              </small>
            </span>
            <input type="checkbox" checked={review.includeRoom} onChange={(e) => useScanUI.getState().setReview({ includeRoom: e.target.checked })} />
          </label>
        )}
        {objects.map((o) => (
          <label key={o.id} className="g-toggle onb-found-row">
            <span>
              <span className="onb-found-ico" style={{ background: o.color ?? undefined }} />
              {o.category === 'box' ? 'Something (not sure)' : o.name}
              <small>
                {formatLength(o.width, units)} × {formatLength(o.depth, units)}
              </small>
            </span>
            <input type="checkbox" checked={o.accepted} onChange={(e) => useScanUI.getState().setObject(o.id, { accepted: e.target.checked })} />
          </label>
        ))}
      </div>
      <div className="onb-row sticky">
        <button className="g-btn ghost" onClick={skip}>
          Trace walls myself
        </button>
        <button
          className="g-btn primary big"
          disabled={nothing}
          onClick={() => {
            followShift()
            useScanUI.getState().apply()
            onNext()
          }}
        >
          <IconCheck /> Looks good
        </button>
      </div>
    </div>
  )
}

function roomDetails(kinds: string[], ceiling: string) {
  const n = (k: string) => kinds.filter((x) => x === k).length
  const plural = (c: number, w: string) => (c ? `${c} ${w}${c === 1 ? '' : 's'}` : null)
  return [plural(n('door'), 'door'), plural(n('window'), 'window'), `${ceiling} ceiling`].filter(Boolean).join(' · ')
}

/** "Found 1 room and 6 pieces of furniture: sofa, bed and 2 chairs." */
function summary(room: boolean, objects: ReviewObject[]) {
  const known = objects.filter((o) => o.category !== 'box')
  const counts = new Map<string, number>()
  for (const o of known) {
    const k = o.name.toLowerCase()
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  const names = [...counts].map(([k, n]) => (n > 1 ? `${n} ${k}s` : k))
  const list = names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}` : names[0]
  const n = objects.length
  const parts = [room ? '1 room' : null, n ? `${n} piece${n === 1 ? '' : 's'} of furniture` : null].filter(Boolean)
  if (!parts.length) return 'Nothing yet.'
  return `Found ${parts.join(' and ')}${list ? `: ${list}` : ''}. Switch off anything that isn’t yours.`
}

function DetectArt() {
  return (
    <div className="onb-detecting" aria-hidden>
      <svg viewBox="0 0 160 120">
        <rect x="14" y="14" width="132" height="92" rx="8" fill="var(--g-paper)" stroke="#d9c7ad" strokeWidth="4" />
        <rect className="onb-find f1" x="26" y="26" width="46" height="22" rx="5" fill="var(--g-sky)" />
        <rect className="onb-find f2" x="96" y="22" width="38" height="46" rx="5" fill="var(--g-plum)" />
        <circle className="onb-find f3" cx="56" cy="80" r="13" fill="var(--g-sun)" />
        <rect className="onb-find f4" x="92" y="82" width="30" height="16" rx="4" fill="var(--g-sage)" />
      </svg>
      <svg className="onb-glass" viewBox="0 0 48 48">
        <circle cx="20" cy="20" r="13" fill="rgba(255,255,255,0.55)" stroke="var(--g-ink)" strokeWidth="4.5" />
        <path d="m30 30 12 12" stroke="var(--g-ink)" strokeWidth="6" strokeLinecap="round" />
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------- 5 · done

const CONFETTI = ['var(--g-coral)', 'var(--g-sun)', 'var(--g-sage)', 'var(--g-sky)', 'var(--g-plum)']

function Done() {
  return (
    <div className="onb-center onb-done">
      <div className="onb-confetti" aria-hidden>
        {Array.from({ length: 28 }, (_, i) => (
          <i
            key={i}
            style={{
              left: `${(i * 37) % 100}%`,
              background: CONFETTI[i % CONFETTI.length],
              animationDelay: `${(i % 7) * 0.12}s`,
              animationDuration: `${1.8 + ((i * 13) % 10) / 10}s`,
              transform: `rotate(${i * 47}deg)`,
            }}
          />
        ))}
      </div>
      <div className="onb-done-art">
        <HouseArt />
        <IconSparkle className="onb-spark s1" />
        <IconSparkle className="onb-spark s2" />
        <IconSparkle className="onb-spark s3" />
      </div>
      <Title kicker="Keys are yours" sub="Tap Buy to furnish, Build to change walls.">
        Welcome home!
      </Title>
      <div className="onb-actions">
        <button className="g-btn primary big onb-wide" onClick={finish}>
          Start designing
        </button>
      </div>
    </div>
  )
}
