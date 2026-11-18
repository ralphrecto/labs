import { useEffect, useState } from 'react'
import type { Units } from '../model/types'
import { formatLength, parseLength, toInputValue, unitLabel } from '../model/units'

export function LengthField({
  label,
  value,
  units,
  onChange,
  min = 0,
}: {
  label: string
  value: number
  units: Units
  onChange: (m: number) => void
  min?: number
}) {
  const [text, setText] = useState(toInputValue(value, units))
  const [bad, setBad] = useState(false)
  useEffect(() => {
    setText(toInputValue(value, units))
    setBad(false)
  }, [value, units])
  const commit = () => {
    const m = parseLength(text, units)
    if (m == null || m < min) {
      setBad(true)
      return
    }
    setBad(false)
    if (Math.abs(m - value) > 1e-6) onChange(m)
  }
  return (
    <label className="field" title={formatLength(value, units, { precise: true })}>
      <span>{label}</span>
      <span className="field-input">
        <input
          className={bad ? 'bad' : undefined}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commit()
            e.stopPropagation()
          }}
        />
        <em>{unitLabel(units)}</em>
      </span>
    </label>
  )
}

export function TextField({ label, value, onChange }: { label: string; value: string; onChange: (s: string) => void }) {
  const [text, setText] = useState(value)
  useEffect(() => setText(value), [value])
  return (
    <label className="field">
      <span>{label}</span>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => text !== value && onChange(text)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && text !== value) onChange(text)
          e.stopPropagation()
        }}
      />
    </label>
  )
}

export function NumberField({ label, value, onChange, suffix }: { label: string; value: number | undefined; onChange: (n: number | undefined) => void; suffix?: string }) {
  const [text, setText] = useState(value == null ? '' : String(value))
  useEffect(() => setText(value == null ? '' : String(value)), [value])
  const commit = () => {
    if (text.trim() === '') return onChange(undefined)
    const n = Number(text)
    if (Number.isFinite(n)) onChange(n)
  }
  return (
    <label className="field">
      <span>{label}</span>
      <span className="field-input">
        <input value={text} onChange={(e) => setText(e.target.value)} onBlur={commit} onKeyDown={(e) => { if (e.key === 'Enter') commit(); e.stopPropagation() }} />
        {suffix && <em>{suffix}</em>}
      </span>
    </label>
  )
}
