import { beforeEach, describe, expect, it, vi } from 'vitest'
import { emptyDoc, sampleDoc } from '../model/sample'

class MemoryStorage {
  m = new Map<string, string>()
  getItem(k: string) {
    return this.m.has(k) ? this.m.get(k)! : null
  }
  setItem(k: string, v: string) {
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
})

describe('storage migration', () => {
  it('moves the single legacy save into the first home and keeps the legacy slot', async () => {
    const legacy = sampleDoc()
    legacy.apartment.name = 'My flat'
    ls.setItem('apartment-designer:doc:v1', JSON.stringify(legacy))
    const st = await import('../storage')
    const homes = st.listHomes()
    expect(homes.map((h) => h.name)).toEqual(['My flat'])
    expect(st.currentHomeId()).toBe(legacy.apartment.id)
    expect(st.loadDoc()?.apartment.name).toBe('My flat')
    expect(ls.getItem('apartment-designer:doc:v1')).not.toBeNull()
    expect(ls.getItem(`apartment-designer:home:${legacy.apartment.id}`)).not.toBeNull()
  })

  it('starts empty without a legacy save, and ignores a broken one', async () => {
    const st = await import('../storage')
    expect(st.listHomes()).toEqual([])
    expect(st.loadDoc()).toBeNull()
    ls.clear()
    ls.setItem('apartment-designer:doc:v1', '{"nope":1}')
    expect(st.loadDoc()).toBeNull()
  })

  it('keeps every home in its own slot and never deletes the last one', async () => {
    const st = await import('../storage')
    const a = sampleDoc()
    const b = emptyDoc()
    st.saveDoc(a)
    st.saveDoc(b)
    expect(st.listHomes()).toHaveLength(2)
    expect(st.loadHome(a.apartment.id)?.items.length).toBe(a.items.length)
    expect(st.deleteHome(a.apartment.id)).toBe(true)
    expect(st.loadHome(a.apartment.id)).toBeNull()
    expect(st.deleteHome(b.apartment.id)).toBe(false)
    expect(st.listHomes()).toHaveLength(1)
  })

  it('remembers scans of homes that are not open', async () => {
    const st = await import('../storage')
    const a = emptyDoc()
    a.scans = [{ id: 'scan-1' } as never]
    st.saveDoc(a)
    st.saveDoc(sampleDoc())
    expect(st.allHomeScanIds(['scan-2']).sort()).toEqual(['scan-1', 'scan-2'])
  })
})

describe('multiple homes in the store', () => {
  it('loading the sample keeps the scanned home, and switching back finds it intact', async () => {
    const mine = sampleDoc()
    mine.apartment.name = 'My flat'
    ls.setItem('apartment-designer:doc:v1', JSON.stringify(mine))
    const store = await import('../store')
    const st = await import('../storage')
    const { useStore } = store
    expect(useStore.getState().doc.apartment.name).toBe('My flat')
    useStore.getState().addRect('Study', 3, 3)
    const rooms = useStore.getState().doc.rooms.length

    store.loadSample()
    expect(useStore.getState().doc.apartment.name).toBe('Sample apartment')
    expect(useStore.getState().past).toHaveLength(0)
    expect(st.listHomes().map((h) => h.name).sort()).toEqual(['My flat', 'Sample apartment'])
    // A reload opens the sample home: it is current.
    expect(st.currentHomeId()).toBe(useStore.getState().doc.apartment.id)

    expect(store.switchHome(mine.apartment.id)).toBe(true)
    expect(useStore.getState().doc.apartment.name).toBe('My flat')
    expect(useStore.getState().doc.rooms.length).toBe(rooms)
    expect(st.currentHomeId()).toBe(mine.apartment.id)

    // A second sample gets its own name; a blank home being left is dropped.
    store.loadSample()
    expect(useStore.getState().doc.apartment.name).toBe('Sample apartment 2')
    store.newEmptyDoc()
    store.newEmptyDoc()
    expect(st.listHomes()).toHaveLength(4)

    // Deleting the open home moves to another one first.
    const blank = useStore.getState().doc.apartment.id
    expect(store.deleteHome(blank)).toBe(true)
    expect(useStore.getState().doc.apartment.id).not.toBe(blank)
    expect(st.listHomes()).toHaveLength(3)
  })

  it('gives new designs and copies friendly names', async () => {
    const { designName, DESIGN_NAMES } = await import('../store')
    expect(designName([])).toBe(DESIGN_NAMES[0])
    expect(designName([DESIGN_NAMES[0]])).toBe(DESIGN_NAMES[1])
    expect(designName(DESIGN_NAMES)).toBe('Idea 2')
  })
})
