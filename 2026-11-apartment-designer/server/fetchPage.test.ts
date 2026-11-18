import { describe, expect, it } from 'vitest'
import { isPrivateAddress } from './fetchPage.ts'

describe('import server link checks', () => {
  it('rejects private and local addresses', () => {
    for (const ip of ['127.0.0.1', '10.1.2.3', '192.168.1.1', '172.20.0.1', '169.254.169.254', '::1', 'fd00::1', '::ffff:127.0.0.1']) expect(isPrivateAddress(ip), ip).toBe(true)
    for (const ip of ['8.8.8.8', '151.101.1.1', '2606:4700::1111']) expect(isPrivateAddress(ip), ip).toBe(false)
  })
})
