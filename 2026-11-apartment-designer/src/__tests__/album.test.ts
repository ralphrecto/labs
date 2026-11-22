import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AlbumPhoto } from '../storage'

class MemoryStorage {
  m = new Map<string, string>()
  /** Max total characters before setItem throws (a fake quota). */
  quota = Infinity
  getItem(k: string) {
    return this.m.has(k) ? this.m.get(k)! : null
  }
  setItem(k: string, v: string) {
    const size = [...this.m].reduce((n, [kk, vv]) => n + (kk === k ? 0 : vv.length), 0) + String(v).length
    if (size > this.quota) throw new DOMException('full', 'QuotaExceededError')
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

const photo = (n: number, extra: Partial<AlbumPhoto> = {}): AlbumPhoto => ({
  id: `p${n}`,
  homeId: 'h1',
  layoutId: 'l1',
  layoutName: 'Design 1',
  date: 1000 + n,
  level: 'Level 4 · Cozy',
  stars: 9,
  evening: false,
  src: `data:image/jpeg;base64,${'x'.repeat(100)}`,
  ...extra,
})

describe('photo album storage', () => {
  it('adds photos newest first, per home', async () => {
    const st = await import('../storage')
    expect(st.listAlbum('h1')).toEqual([])
    expect(st.addAlbumPhoto(photo(1))).toBe(true)
    expect(st.addAlbumPhoto(photo(2))).toBe(true)
    expect(st.addAlbumPhoto(photo(3, { homeId: 'h2' }))).toBe(true)
    expect(st.listAlbum('h1').map((p) => p.id)).toEqual(['p2', 'p1'])
    expect(st.listAlbum('h2').map((p) => p.id)).toEqual(['p3'])
  })

  it('caps the album, dropping the oldest unpinned photo first', async () => {
    const st = await import('../storage')
    st.addAlbumPhoto(photo(0, { pinned: true, twist: 'open-floor' }))
    for (let i = 1; i <= st.ALBUM_CAP + 3; i++) st.addAlbumPhoto(photo(i))
    const ids = st.listAlbum('h1').map((p) => p.id)
    expect(ids).toHaveLength(st.ALBUM_CAP)
    expect(ids).toContain('p0') // the pinned twist polaroid stays
    expect(ids).not.toContain('p1')
    expect(ids[0]).toBe(`p${st.ALBUM_CAP + 3}`)
  })

  it('deletes a photo', async () => {
    const st = await import('../storage')
    st.addAlbumPhoto(photo(1))
    st.addAlbumPhoto(photo(2))
    expect(st.deleteAlbumPhoto('h1', 'p1')).toBe(true)
    expect(st.listAlbum('h1').map((p) => p.id)).toEqual(['p2'])
    expect(st.deleteAlbumPhoto('h1', 'p2')).toBe(true)
    expect(st.listAlbum('h1')).toEqual([])
    expect(ls.getItem('apartment-designer:album:h1')).toBeNull()
  })

  it('makes room by dropping old photos when the quota is full', async () => {
    const st = await import('../storage')
    const one = JSON.stringify([photo(1)]).length
    ls.quota = one * 3
    for (let i = 1; i <= 6; i++) expect(st.addAlbumPhoto(photo(i))).toBe(true)
    const ids = st.listAlbum('h1').map((p) => p.id)
    expect(ids[0]).toBe('p6')
    expect(ids.length).toBeLessThan(6)
  })

  it('degrades gracefully when storage throws', async () => {
    const st = await import('../storage')
    st.addAlbumPhoto(photo(1))
    ls.setItem = () => {
      throw new Error('denied')
    }
    ls.getItem = () => {
      throw new Error('denied')
    }
    expect(() => st.addAlbumPhoto(photo(2))).not.toThrow()
    expect(st.addAlbumPhoto(photo(2))).toBe(false)
    expect(st.listAlbum('h1')).toEqual([])
    expect(st.deleteAlbumPhoto('h1', 'p1')).toBe(false)
  })

  it('works without any localStorage at all', async () => {
    vi.stubGlobal('localStorage', undefined)
    const st = await import('../storage')
    expect(st.addAlbumPhoto(photo(1))).toBe(false)
    expect(st.listAlbum('h1')).toEqual([])
  })

  it('ignores a broken album entry', async () => {
    ls.setItem('apartment-designer:album:h1', '{not json')
    const st = await import('../storage')
    expect(st.listAlbum('h1')).toEqual([])
    expect(st.addAlbumPhoto(photo(1))).toBe(true)
    expect(st.listAlbum('h1').map((p) => p.id)).toEqual(['p1'])
  })
})

describe('twist polaroids', () => {
  it("a remix's first Dream home drops a pinned polaroid tagged with the twist into the album", async () => {
    const { useStore } = await import('../store')
    const { sampleDoc } = await import('../model/sample')
    const album = await import('../game/album')
    // A tiny fake canvas: enough for the paper card the album draws when there is no 3D view.
    const ctx = new Proxy({}, { get: (_t, k) => (k === 'createLinearGradient' ? () => ({ addColorStop() {} }) : () => {}) })
    vi.stubGlobal('document', {
      createElement: () => ({ width: 0, height: 0, getContext: () => ctx, toDataURL: () => 'data:image/jpeg;base64,AAAA' }),
      querySelector: () => null,
    })
    album.watchRemixDreams()
    useStore.getState().replaceDoc(sampleDoc({ settled: true }))
    const src = useStore.getState().doc.layouts[0]
    useStore.getState().remixLayout(src.id, 'open-floor')
    const l = useStore.getState().doc.layouts.find((x) => x.id === useStore.getState().doc.activeLayoutId)!
    useStore.getState().markRemixDream(l.id)
    await new Promise((r) => setTimeout(r, 40))
    const photos = album.albumFor(useStore.getState().doc.apartment.id)
    expect(photos).toHaveLength(1)
    expect(photos[0]).toMatchObject({ layoutId: l.id, twist: 'open-floor', emoji: '🕺', pinned: true })
    // Only the first Dream home: marking again adds nothing.
    useStore.getState().markRemixDream(l.id)
    await new Promise((r) => setTimeout(r, 40))
    expect(album.albumFor(useStore.getState().doc.apartment.id)).toHaveLength(1)
    vi.unstubAllGlobals()
  })
})
