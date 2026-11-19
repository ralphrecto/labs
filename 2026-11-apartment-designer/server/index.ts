// A small server for adding furniture from links and photos. The browser can't read store
// pages itself (stores don't allow cross-site requests), so this fetches the page, reads the
// product data out of it, and asks Claude to fill anything the parser missed.
//
//   npm run server        API on http://localhost:8787 (npm run dev proxies /api to it)
//   npm start             build the app, then serve it and the API together
//
// Set ANTHROPIC_API_KEY to enable photo import and the AI fallback for links.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import { extractFromHtml, isComplete, mergeInfo, pageDigest, type ProductInfo } from '../src/import/productPage.ts'
import { FetchError, fetchModel, fetchPage } from './fetchPage.ts'
import { fetchPageWithClaude, llmAvailable, LlmError, readPageWithClaude, readPhotoWithClaude } from './llm.ts'

const PORT = Number(process.env.PORT ?? 8787)
const DIST = fileURLToPath(new URL('../dist/', import.meta.url))
const MAX_BODY = 8 * 1024 * 1024

class HttpError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []
  let total = 0
  for await (const chunk of req) {
    total += (chunk as Buffer).length
    if (total > MAX_BODY) throw new HttpError(413, 'That file is too large. Try a photo under 8 MB.')
    chunks.push(chunk as Buffer)
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')
  } catch {
    throw new HttpError(400, 'Bad request.')
  }
}

function send(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' })
  res.end(JSON.stringify(body))
}

async function importLink(rawUrl: string): Promise<ProductInfo> {
  let parsed: ProductInfo | undefined
  let fetchProblem: string | undefined
  let html = ''
  let url = rawUrl
  try {
    const page = await fetchPage(rawUrl)
    html = page.html
    url = page.url
    parsed = extractFromHtml(html, url)
  } catch (e) {
    if (!(e instanceof FetchError)) throw e
    // A bad or private link is the user's to fix; a blocked or failed fetch can fall back to Claude.
    if (e.badLink) throw new HttpError(400, e.message)
    if (e.status === 404 || e.status === 410) throw new HttpError(400, "That page doesn't exist anymore (HTTP 404). Check the link.")
    fetchProblem = e.message
  }
  if (parsed && isComplete(parsed)) return parsed
  if (!llmAvailable()) {
    if (parsed) return parsed
    throw new HttpError(502, `${fetchProblem} Copy the size from the page and paste it in the Paste text tab instead.`)
  }
  try {
    if (parsed) return mergeInfo(parsed, await readPageWithClaude(url, pageDigest(html), parsed))
    const info = await fetchPageWithClaude(url)
    info.notes = [...(info.notes ?? []), `Read through Claude because ${fetchProblem?.replace(/\.$/, '').replace(/^./, (c) => c.toLowerCase())}.`]
    return info
  } catch (e) {
    if (parsed) return { ...parsed, notes: [...(parsed.notes ?? []), "The AI fallback didn't answer, so only what the parser found is shown."] }
    throw new HttpError(502, e instanceof LlmError ? e.message : "Couldn't read that page.")
  }
}

async function importPhoto(body: Record<string, unknown>): Promise<ProductInfo> {
  if (!llmAvailable()) throw new HttpError(501, 'Photo recognition needs the import server to have an Anthropic API key. You can still add the photo and type the size.')
  const dataUrl = String(body.image ?? '')
  const m = dataUrl.match(/^data:(image\/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)$/)
  if (!m) throw new HttpError(400, 'Send the photo as a JPEG, PNG, WebP or GIF.')
  const note = typeof body.note === 'string' ? body.note.slice(0, 500) : undefined
  try {
    return await readPhotoWithClaude({ mediaType: m[1] as 'image/jpeg', data: m[2] }, note)
  } catch (e) {
    throw new HttpError(502, e instanceof LlmError ? e.message : "Couldn't read that photo.")
  }
}

const TYPES: Record<string, string> = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png', '.json': 'application/json', '.glb': 'model/gltf-binary', '.ico': 'image/x-icon' }

async function serveStatic(req: IncomingMessage, res: ServerResponse) {
  const path = normalize(decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '')
  let file = join(DIST, path)
  if (!file.startsWith(DIST)) throw new HttpError(404, 'Not found')
  if (!(await stat(file).catch(() => null))?.isFile()) file = join(DIST, 'index.html')
  const data = await readFile(file).catch(() => null)
  if (!data) throw new HttpError(404, 'Not found. Run npm run build first, or use npm run dev.')
  res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' })
  res.end(data)
}

const server = createServer(async (req, res) => {
  const path = new URL(req.url ?? '/', 'http://x').pathname
  try {
    if (path === '/api/status' && req.method === 'GET') return send(res, 200, { ok: true, llm: llmAvailable() })
    if (path === '/api/import/link' && req.method === 'POST') {
      const body = await readJson(req)
      return send(res, 200, await importLink(String(body.url ?? '').trim()))
    }
    if (path === '/api/model' && req.method === 'GET') {
      const target = new URL(req.url ?? '', 'http://x').searchParams.get('url') ?? ''
      const model = await fetchModel(target).catch((e: Error) => {
        throw new HttpError(e instanceof FetchError && e.badLink ? 400 : 502, e.message)
      })
      res.writeHead(200, { 'Content-Type': model.type, 'Cache-Control': 'public, max-age=86400' })
      return res.end(model.body)
    }
    if (path === '/api/import/photo' && req.method === 'POST') return send(res, 200, await importPhoto(await readJson(req)))
    if (path.startsWith('/api/')) throw new HttpError(404, 'Not found')
    if (req.method === 'GET') return await serveStatic(req, res)
    throw new HttpError(405, 'Method not allowed')
  } catch (e) {
    const status = e instanceof HttpError ? e.status : 500
    if (status === 500) console.error(e)
    send(res, status, { error: e instanceof HttpError ? e.message : 'Something went wrong on the import server.' })
  }
})

server.listen(PORT, () => {
  console.log(`Import server on http://localhost:${PORT} (Claude ${llmAvailable() ? 'on' : 'off: set ANTHROPIC_API_KEY for photos and the link fallback'})`)
})
