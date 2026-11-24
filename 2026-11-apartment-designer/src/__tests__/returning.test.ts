import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { emptyDoc, sampleDoc } from '../model/sample'

class MemoryStorage {
  m = new Map<string, string>()
  /** When set, writes throw like a full quota. */
  full = false
  getItem(k: string) {
    return this.m.has(k) ? this.m.get(k)! : null
  }
  setItem(k: string, v: string) {
    if (this.full) throw new DOMException('Quota exceeded', 'QuotaExceededError')
    this.m.set(k, String(v))
  }
  removeItem(k: string) {
    this.m.delete(k)
  }
  clear() {
    this.m.clear()
  }
  key(i: number) {
    return [...this.m.keys()][i] ?? null
  }
  get length() {
    return this.m.size
  }
}

let ls: MemoryStorage
beforeEach(() => {
  ls = new MemoryStorage()
  vi.stubGlobal('localStorage', ls)
  vi.resetModules()
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
})

describe('saving when storage is full', () => {
  it('saveDoc reports failure', async () => {
    const st = await import('../storage')
    const d = sampleDoc()
    expect(st.saveDoc(d)).toBe(true)
    ls.full = true
    expect(st.saveDoc(d)).toBe(false)
  })

  it('keeps the unsaved change, flags it, and retries on the next change until it lands', async () => {
    const { useStore, flushSave } = await import('../store')
    const st = await import('../storage')
    const id = useStore.getState().doc.apartment.id
    flushSave()
    ls.full = true
    useStore.getState().renameApartment('Full flat')
    vi.advanceTimersByTime(400)
    expect(useStore.getState().saveFailed).toBe(true)
    expect(st.loadHome(id)?.apartment.name).not.toBe('Full flat')
    // Space frees up; the next change retries and both land.
    ls.full = false
    useStore.getState().setUnits('imperial')
    vi.advanceTimersByTime(400)
    expect(useStore.getState().saveFailed).toBe(false)
    expect(st.loadHome(id)?.apartment.name).toBe('Full flat')
    expect(st.loadHome(id)?.apartment.units).toBe('imperial')
  })

  it('a pending save is retried by flushSave (pagehide) too', async () => {
    const { useStore, flushSave } = await import('../store')
    const st = await import('../storage')
    const id = useStore.getState().doc.apartment.id
    ls.full = true
    useStore.getState().renameApartment('Later')
    expect(flushSave()).toBe(false)
    ls.full = false
    expect(flushSave()).toBe(true)
    expect(st.loadHome(id)?.apartment.name).toBe('Later')
  })
})

describe('designs and undo', () => {
  it('switching, adding or copying a design never makes an undo entry', async () => {
    const { useStore } = await import('../store')
    const s = () => useStore.getState()
    const first = s().doc.activeLayoutId
    s().renameApartment('Edited')
    expect(s().past).toHaveLength(1)
    s().duplicateLayout(first)
    const copy = s().doc.activeLayoutId
    expect(copy).not.toBe(first)
    expect(s().past).toHaveLength(0)
    s().setActiveLayout(first)
    expect(s().past).toHaveLength(0)
    s().setActiveLayout(copy)
    s().renameLayout(copy, 'Changed')
    s().undo()
    expect(s().doc.activeLayoutId).toBe(copy)
    s().undo()
    expect(s().doc.activeLayoutId).toBe(copy)
    s().addLayout()
    expect(s().past).toHaveLength(0)
  })

  it('names copies after their source', async () => {
    const { copyName } = await import('../store')
    expect(copyName('Move-in day', ['Move-in day'])).toBe('Move-in day 2')
    expect(copyName('Move-in day 2', ['Move-in day', 'Move-in day 2'])).toBe('Move-in day 3')
    expect(sampleDoc().layouts[0].name).toBe('Move-in day')
  })
})

describe('deleting a home', () => {
  it('moves it to the trash, where Undo brings it back', async () => {
    const st = await import('../storage')
    const a = sampleDoc()
    const b = emptyDoc()
    st.saveDoc(a)
    st.saveDoc(b)
    expect(st.deleteHome(a.apartment.id)).toBe(true)
    expect(st.loadHome(a.apartment.id)).toBeNull()
    expect(st.listHomes()).toHaveLength(1)
    expect(JSON.parse(ls.getItem(`apartment-designer:trash:${a.apartment.id}`)!).deletedAt).toBeGreaterThan(0)
    expect(st.restoreHome(a.apartment.id)).toBe(true)
    expect(st.loadHome(a.apartment.id)?.items.length).toBe(a.items.length)
    expect(st.listHomes()).toHaveLength(2)
    expect(ls.getItem(`apartment-designer:trash:${a.apartment.id}`)).toBeNull()
  })

  it('purges the trash after a week, with the home’s best levels', async () => {
    const st = await import('../storage')
    const a = sampleDoc()
    const b = emptyDoc()
    st.saveDoc(a)
    st.saveDoc(b)
    const levels = { [`${a.apartment.id}:${a.activeLayoutId}`]: 3, [a.apartment.id]: 2, [`${b.apartment.id}:x`]: 1 }
    ls.setItem('apartment-designer:best-level', JSON.stringify(levels))
    const t0 = Date.now()
    st.deleteHome(a.apartment.id, t0)
    st.purgeTrash(t0 + 6 * 86400000)
    expect(st.trashedHomeIds()).toEqual([a.apartment.id])
    st.purgeTrash(t0 + 8 * 86400000)
    expect(st.trashedHomeIds()).toEqual([])
    expect(st.restoreHome(a.apartment.id)).toBe(false)
    expect(JSON.parse(ls.getItem('apartment-designer:best-level')!)).toEqual({ [`${b.apartment.id}:x`]: 1 })
  })

  it('renames the old sample design "Current" when loading', async () => {
    const st = await import('../storage')
    const a = sampleDoc()
    a.layouts[0].name = 'Current'
    expect(st.parseDoc(JSON.stringify(a))?.layouts[0].name).toBe('Move-in day')
  })
})
