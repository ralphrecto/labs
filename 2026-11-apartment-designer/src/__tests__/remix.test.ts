import { beforeEach, describe, expect, it, vi } from 'vitest'

beforeEach(() => {
  const m = new Map<string, string>()
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => (m.has(k) ? m.get(k)! : null),
    setItem: (k: string, v: string) => void m.set(k, String(v)),
    removeItem: (k: string) => void m.delete(k),
    clear: () => m.clear(),
    key: (i: number) => [...m.keys()][i] ?? null,
    get length() {
      return m.size
    },
  })
  vi.resetModules()
})

describe('remix and copies in the store (round 8)', () => {
  it('remixes a design from its big pieces, as a fresh design with its own best of 0', async () => {
    const { useStore } = await import('../store')
    const { useGame } = await import('../game/gameStore')
    const { sampleDoc } = await import('../model/sample')
    const { bestLevelKey } = await import('../game/goals')
    useStore.getState().replaceDoc(sampleDoc({ settled: true }))
    const src = useStore.getState().doc.layouts[0]
    useGame.getState().noteLevel(bestLevelKey(useStore.getState().doc), 5)
    useStore.getState().remixLayout(src.id, 'talk-corner')
    const doc = useStore.getState().doc
    const l = doc.layouts.find((x) => x.id === doc.activeLayoutId)!
    expect(l.name).toBe('Talk corner')
    expect(l.remix).toMatchObject({ from: src.id, twist: 'talk-corner' })
    expect(l.placements.length).toBe(l.remix!.anchors)
    expect(l.placements.length).toBeLessThan(src.placements.length)
    expect(useGame.getState().bestLevel[bestLevelKey(doc)] ?? 0).toBe(0)
    expect(useStore.getState().past).toEqual([])

    useStore.getState().markRemixDream(l.id)
    const at = useStore.getState().doc.layouts.find((x) => x.id === l.id)!.remix!.dreamAt
    expect(at).toBeGreaterThan(0)
    useStore.getState().markRemixDream(l.id)
    expect(useStore.getState().doc.layouts.find((x) => x.id === l.id)!.remix!.dreamAt).toBe(at)
  })

  it('an exact copy inherits its source’s best level outright', async () => {
    const { useStore } = await import('../store')
    const { useGame } = await import('../game/gameStore')
    const { sampleDoc } = await import('../model/sample')
    const { bestLevelKey } = await import('../game/goals')
    useStore.getState().replaceDoc(sampleDoc())
    useGame.getState().noteLevel(bestLevelKey(useStore.getState().doc), 4)
    useStore.getState().duplicateLayout(useStore.getState().doc.activeLayoutId)
    expect(useGame.getState().bestLevel[bestLevelKey(useStore.getState().doc)]).toBe(4)
  })
})
