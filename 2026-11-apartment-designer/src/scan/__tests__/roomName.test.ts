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
    expect(friendlyRoomName('Scan 2026-11-20 14.32', [])).toBe('Your room')
    expect(friendlyRoomName("Sam's Bedroom.ply", [])).toBe("Sam's Bedroom")
    expect(friendlyRoomName('Scan 2026-11-20 14.32', [{ category: 'chair' }, { category: 'bed' }])).toBe('Bedroom')
    expect(friendlyRoomName('3DGS_abc123', [{ category: 'desk' }])).toBe('Office')
    expect(friendlyRoomName('IMG_0042', [{ category: 'dining-table' }])).toBe('Dining room')
  })
})

describe('friendlyRoomName for file slugs (round 18)', () => {
  it('names the room type a file slug mentions, never the slug itself', () => {
    expect(friendlyRoomName('my-living-room.ply', [{ category: 'bed' }])).toBe('Living room')
    expect(friendlyRoomName('master_bedroom.spz', [])).toBe('Bedroom')
    expect(friendlyRoomName('Sample living room.ply', [])).toBe('Living room')
  })
  it('guesses from the furniture, or says "Your room"', () => {
    expect(friendlyRoomName('apartment-scan.ply', [{ category: 'sofa' }])).toBe('Living room')
    expect(friendlyRoomName('My apartment.ply', [{ category: 'coffee-table' }])).toBe('Living room')
    expect(friendlyRoomName('flat_01.ply', [{ category: 'chair' }])).toBe('Your room')
  })
})

describe('notScanFileMessage', () => {
  it('names the unsupported format and how to export a working one', () => {
    expect(notScanFileMessage('Living room.usdz')).toMatch(/\.usdz and \.fbx aren’t supported yet/)
    expect(notScanFileMessage('Living room.usdz')).toMatch(/choose PLY \(or SPZ\)/)
    expect(notScanFileMessage('notes.txt')).toMatch(/^That isn’t a scan file/)
  })
})
