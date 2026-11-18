// Claude calls for the import server: reading a product page that the plain parser couldn't,
// fetching a page the store blocked us from, and identifying furniture in a photo.

import Anthropic from '@anthropic-ai/sdk'
import { betaZodOutputFormat } from '@anthropic-ai/sdk/helpers/beta/zod'
import { z } from 'zod'
import { CATALOG } from '../src/model/catalog.ts'
import type { ProductInfo } from '../src/import/productPage.ts'

const MODEL = process.env.IMPORT_MODEL ?? 'claude-opus-5-5'
const CATEGORIES = CATALOG.map((c) => c.category) as [string, ...string[]]

/** True when the server has credentials for the Claude API. */
export const llmAvailable = () => !!(process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN)

let client: Anthropic | undefined
const anthropic = () => (client ??= new Anthropic())

const Extracted = z.object({
  name: z.string().describe('Short product name, without the store name'),
  category: z.enum(CATEGORIES).describe('Closest furniture category'),
  width_cm: z.number().nullable().describe('Overall width, side to side, in centimeters'),
  depth_cm: z.number().nullable().describe('Overall depth, front to back, in centimeters'),
  height_cm: z.number().nullable().describe('Overall height, floor to top, in centimeters'),
  size_source: z.enum(['listed', 'estimated']).describe('listed if the size is written on the page or in the image, estimated otherwise'),
  size_text: z.string().nullable().describe('The exact text the size was read from, when listed'),
  price: z.number().nullable(),
  currency: z.string().nullable().describe('ISO currency code such as USD'),
  color_hex: z.string().nullable().describe('Main visible color of the piece as #rrggbb'),
  image_url: z.string().nullable().describe('Main product image URL, if the page has one'),
  notes: z.string().nullable().describe('One short sentence the user should know, e.g. which variant was picked; null if nothing'),
})
type Extracted = z.infer<typeof Extracted>

const SIZE_RULES = `Sizes are the overall assembled dimensions of the piece in centimeters:
- width is side to side as you face the front, depth is front to back, height is floor to the highest point.
- For beds, width is across the mattress and depth is head to foot.
- Ignore seat, arm, leg, package, and shipping dimensions.
- If several sizes or variants are listed, use the one the link points to (or the first), and say which in notes.
- For sectionals or L shapes, give the bounding box of the whole piece and mention the shape in notes.
- Convert inches and feet to centimeters (1 in = 2.54 cm).`

class LlmError extends Error {}

async function run(params: { system: string; content: Anthropic.Beta.BetaContentBlockParam[]; tools?: Anthropic.Beta.BetaToolUnion[]; effort: 'low' | 'medium' }): Promise<Extracted> {
  const messages: Anthropic.Beta.BetaMessageParam[] = [{ role: 'user', content: params.content }]
  // Server tools (web fetch) can pause a long turn; continue it a few times.
  for (let i = 0; i < 4; i++) {
    const res = await anthropic().beta.messages.parse({
      model: MODEL,
      max_tokens: 16000,
      betas: ['server-side-fallback-2026-07-01'],
      fallbacks: 'default',
      system: params.system,
      messages,
      tools: params.tools,
      output_config: { effort: params.effort, format: betaZodOutputFormat(Extracted) },
    })
    if (res.stop_reason === 'pause_turn') {
      messages.push({ role: 'assistant', content: res.content })
      continue
    }
    if (res.stop_reason === 'refusal') throw new LlmError('The model declined to read this item.')
    if (!res.parsed_output) throw new LlmError("The model's answer couldn't be read.")
    return res.parsed_output
  }
  throw new LlmError('Reading the page took too long.')
}

function toInfo(x: Extracted, base: Partial<ProductInfo> = {}): ProductInfo {
  const m = (cm: number | null) => (cm != null && cm > 0 && cm < 800 ? cm / 100 : undefined)
  return {
    ...base,
    name: x.name || base.name,
    category: x.category,
    dims: { width: m(x.width_cm), depth: m(x.depth_cm), height: m(x.height_cm) },
    dimsSource: x.size_source,
    dimsText: x.size_text ?? undefined,
    price: x.price ?? undefined,
    currency: x.currency ?? undefined,
    color: x.color_hex && /^#[0-9a-f]{6}$/i.test(x.color_hex) ? x.color_hex : undefined,
    imageUrl: base.imageUrl ?? x.image_url ?? undefined,
    notes: x.notes ? [x.notes] : [],
  }
}

/** Read a page we already fetched, when the plain parser missed the name or part of the size. */
export async function readPageWithClaude(url: string, digest: string, found: ProductInfo): Promise<ProductInfo> {
  const x = await run({
    system: `You read furniture product pages and return the product's details as structured data.\n${SIZE_RULES}\nIf the page doesn't list a size, estimate it from typical sizes for this kind of item and set size_source to estimated.`,
    content: [
      {
        type: 'text',
        text: `Product page: ${url}\nAlready found by a parser (may be incomplete or wrong): ${JSON.stringify({ name: found.name, dims_m: found.dims, dimsText: found.dimsText, price: found.price })}\n\nPage text:\n<page>\n${digest}\n</page>`,
      },
    ],
    effort: 'low',
  })
  return toInfo(x, { url, imageUrl: found.imageUrl, glbUrl: found.glbUrl })
}

/** Have Claude fetch the page itself, for stores that block plain requests. */
export async function fetchPageWithClaude(url: string): Promise<ProductInfo> {
  const x = await run({
    system: `You look up furniture product pages and return the product's details as structured data. Use the web_fetch tool on the link you're given.\n${SIZE_RULES}\nIf you can't open the page, use what the link itself says (the product name is often in it) and estimate the size from typical sizes, with size_source set to estimated and the reason in notes.`,
    content: [{ type: 'text', text: `Product link: ${url}` }],
    tools: [{ type: 'web_fetch_20260209', name: 'web_fetch', max_uses: 2 }],
    effort: 'low',
  })
  return toInfo(x, { url })
}

/** Identify the furniture in a photo and estimate its size. */
export async function readPhotoWithClaude(image: { data: string; mediaType: 'image/jpeg' | 'image/png' | 'image/webp' | 'image/gif' }, note?: string): Promise<ProductInfo> {
  const x = await run({
    system: `You identify furniture in photos for an apartment planning app and return its details as structured data.\n${SIZE_RULES}\nIf the image shows written dimensions (a listing screenshot, a tag, a spec sheet), use them and set size_source to listed. Otherwise estimate the size from typical sizes for this kind of item and the proportions you see, and set size_source to estimated. If there are several pieces, describe the most prominent one.`,
    content: [
      { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } },
      { type: 'text', text: note ? `What the user said about it: ${note}` : 'What is this piece and how big is it?' },
    ],
    effort: 'medium',
  })
  return toInfo(x)
}

export { LlmError }
