import { describe, expect, it } from 'vitest'
import { friendlyRoomName, isGenericScanName, notScanFileMessage } from '../scanUi'

describe('friendlyRoomName', () => {
  it('spots names that come from the scanning app', () => {
    for (const n of ['Scan 2026-11-20 14.32', '3DGS_8f2a91c0', 'IMG_0042', '2026-11-20', 'a8f9c03e2b1d', 'Untitled', '', '20.11.26 09:14'])
      expect(isGenericScanName(n), n).toBe(true)
    for (const n of ['My apartment', 'Living room', 'Kitchen', "Sam's bedroom"]) expect(isGenericScanName(n), n).toBe(false)
  })
  it('keeps a real name and guesses from furniture otherwise', () => {
    expect(friendlyRoomName('Kitchen.ply', [{ category: 'bed' }])).toBe('Kitchen')
    expect(friendlyRoomName('Scan 2026-11-20 14.32', [])).toBe('Living room')
    expect(friendlyRoomName('Scan 2026-11-20 14.32', [{ category: 'chair' }, { category: 'bed' }])).toBe('Bedroom')
    expect(friendlyRoomName('3DGS_abc123', [{ category: 'desk' }])).toBe('Office')
    expect(friendlyRoomName('IMG_0042', [{ category: 'dining-table' }])).toBe('Dining room')
  })
})

describe('notScanFileMessage', () => {
  it('names the unsupported format and how to export a working one', () => {
    expect(notScanFileMessage('Living room.usdz')).toMatch(/\.usdz and \.fbx aren’t supported yet/)
    expect(notScanFileMessage('Living room.usdz')).toMatch(/choose PLY \(or SPZ\)/)
    expect(notScanFileMessage('notes.txt')).toMatch(/^That isn’t a scan file/)
  })
})
