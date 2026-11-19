// Fetches a product page for the import server. Only public http(s) hosts are allowed, so a
// pasted link can't reach services on the machine or its private network.

import { lookup } from 'node:dns/promises'
import { isIP } from 'node:net'

const MAX_BYTES = 4 * 1024 * 1024
const TIMEOUT_MS = 15000
const MAX_REDIRECTS = 5

const BROWSER_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36',
  Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'Accept-Language': 'en-US,en;q=0.9',
}

export class FetchError extends Error {
  status?: number
  /** The link itself is the problem (not a web address, or a private one), so retrying elsewhere won't help. */
  badLink: boolean
  constructor(message: string, opts: { status?: number; badLink?: boolean } = {}) {
    super(message)
    this.status = opts.status
    this.badLink = !!opts.badLink
  }
}

export function isPrivateAddress(ip: string): boolean {
  if (isIP(ip) === 6) {
    const a = ip.toLowerCase()
    if (a === '::1' || a === '::') return true
    if (a.startsWith('fc') || a.startsWith('fd') || a.startsWith('fe80')) return true
    const mapped = a.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/)
    return mapped ? isPrivateAddress(mapped[1]) : false
  }
  const [a, b] = ip.split('.').map(Number)
  return a === 10 || a === 127 || a === 0 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127) || a >= 224
}

/** For local testing against pages served from this machine. Never set this on a shared server. */
const ALLOW_PRIVATE = process.env.IMPORT_ALLOW_PRIVATE === '1'

export async function assertPublicUrl(raw: string): Promise<URL> {
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    throw new FetchError("That doesn't look like a link. Paste the full address, starting with https://", { badLink: true })
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new FetchError('Only http and https links can be imported.', { badLink: true })
  if (url.username || url.password) throw new FetchError('Links with a username or password are not supported.', { badLink: true })
  if (ALLOW_PRIVATE) return url
  const host = url.hostname.replace(/^\[|\]$/g, '')
  if (host === 'localhost' || host.endsWith('.local') || host.endsWith('.internal')) throw new FetchError('Only public websites can be imported.', { badLink: true })
  const addrs = isIP(host) ? [{ address: host }] : await lookup(host, { all: true }).catch(() => [])
  if (!addrs.length) throw new FetchError(`Couldn't find the website ${host}.`)
  if (addrs.some((a) => isPrivateAddress(a.address))) throw new FetchError('Only public websites can be imported.', { badLink: true })
  return url
}

/** Fetch a page's HTML, following redirects and checking each hop. */
export async function fetchPage(raw: string): Promise<{ html: string; url: string }> {
  let url = await assertPublicUrl(raw)
  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    const res = await fetch(url, { headers: BROWSER_HEADERS, redirect: 'manual', signal: AbortSignal.timeout(TIMEOUT_MS) }).catch((e: Error) => {
      throw new FetchError(e.name === 'TimeoutError' ? 'The store took too long to answer.' : `Couldn't reach ${url.hostname}.`)
    })
    if (res.status >= 300 && res.status < 400 && res.headers.get('location')) {
      url = await assertPublicUrl(new URL(res.headers.get('location')!, url).toString())
      continue
    }
    if (!res.ok) throw new FetchError(`${url.hostname} answered with HTTP ${res.status}${res.status === 403 || res.status === 429 ? ' (many stores block automated requests)' : ''}.`, { status: res.status })
    const type = res.headers.get('content-type') ?? ''
    if (type && !/html|xml|text/i.test(type)) throw new FetchError(`That link is a ${type.split(';')[0]} file, not a product page.`, { badLink: true })
    const reader = res.body?.getReader()
    if (!reader) return { html: '', url: url.toString() }
    const chunks: Uint8Array[] = []
    let total = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.byteLength
      if (total > MAX_BYTES) {
        await reader.cancel()
        break
      }
      chunks.push(value)
    }
    return { html: Buffer.concat(chunks).toString('utf8'), url: url.toString() }
  }
  throw new FetchError('Too many redirects.')
}

const MAX_MODEL_BYTES = 60 * 1024 * 1024

/** Fetch a store's 3D model so the browser can load it (stores rarely allow cross-site loads). */
export async function fetchModel(raw: string): Promise<{ body: Buffer; type: string }> {
  let url = await assertPublicUrl(raw)
  if (!/\.(glb|gltf)$/i.test(url.pathname)) throw new FetchError('Only .glb models can be loaded.', { badLink: true })
  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    const res = await fetch(url, { headers: { 'User-Agent': BROWSER_HEADERS['User-Agent'] }, redirect: 'manual', signal: AbortSignal.timeout(30000) }).catch(() => {
      throw new FetchError(`Couldn't reach ${url.hostname}.`)
    })
    if (res.status >= 300 && res.status < 400 && res.headers.get('location')) {
      url = await assertPublicUrl(new URL(res.headers.get('location')!, url).toString())
      continue
    }
    if (!res.ok) throw new FetchError(`${url.hostname} answered with HTTP ${res.status}.`, { status: res.status })
    if (Number(res.headers.get('content-length') ?? 0) > MAX_MODEL_BYTES) throw new FetchError('That model is too large.', { badLink: true })
    const body = Buffer.from(await res.arrayBuffer())
    if (body.length > MAX_MODEL_BYTES) throw new FetchError('That model is too large.', { badLink: true })
    return { body, type: 'model/gltf-binary' }
  }
  throw new FetchError('Too many redirects.')
}
