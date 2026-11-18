// Photo helpers: shrink an uploaded photo for storage and upload, and pick its main color.

function loadImage(file: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      URL.revokeObjectURL(url)
      resolve(img)
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error("That file couldn't be opened as an image."))
    }
    img.src = url
  })
}

function draw(img: HTMLImageElement, max: number): HTMLCanvasElement {
  const s = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight))
  const c = document.createElement('canvas')
  c.width = Math.max(1, Math.round(img.naturalWidth * s))
  c.height = Math.max(1, Math.round(img.naturalHeight * s))
  const ctx = c.getContext('2d')!
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, c.width, c.height)
  ctx.drawImage(img, 0, 0, c.width, c.height)
  return c
}

/**
 * The most common strong color in the middle of the photo, where the item usually is.
 * Near-white and near-black pixels (backgrounds, shadows) count less.
 */
export function dominantColor(data: Uint8ClampedArray, width: number, height: number): string {
  const buckets = new Map<number, { n: number; r: number; g: number; b: number }>()
  for (let y = Math.floor(height * 0.15); y < height * 0.85; y++) {
    for (let x = Math.floor(width * 0.15); x < width * 0.85; x++) {
      const i = (y * width + x) * 4
      const r = data[i], g = data[i + 1], b = data[i + 2]
      const max = Math.max(r, g, b), min = Math.min(r, g, b)
      const light = (max + min) / 2
      let w = 1
      if (light > 235 || light < 20) w = 0.05
      else if (max - min < 12 && light > 200) w = 0.2
      const key = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
      const e = buckets.get(key) ?? { n: 0, r: 0, g: 0, b: 0 }
      e.n += w
      e.r += r * w
      e.g += g * w
      e.b += b * w
      buckets.set(key, e)
    }
  }
  let best = { n: 0, r: 154, g: 164, b: 173 }
  for (const e of buckets.values()) if (e.n > best.n) best = e
  const hex = (v: number) => Math.round(best.n ? v / best.n : v).toString(16).padStart(2, '0')
  return best.n ? `#${hex(best.r)}${hex(best.g)}${hex(best.b)}` : '#9aa4ad'
}

export interface PreparedPhoto {
  /** About 512 px, sent to the import server. */
  upload: string
  /** About 160 px, kept on the item (small enough for local storage). */
  thumbnail: string
  color: string
}

export async function preparePhoto(file: Blob): Promise<PreparedPhoto> {
  if (!file.type.startsWith('image/')) throw new Error('Choose an image file (JPEG, PNG, WebP or HEIC exported as JPEG).')
  const img = await loadImage(file)
  const big = draw(img, 768)
  const small = draw(img, 160)
  const ctx = small.getContext('2d')!
  const { data } = ctx.getImageData(0, 0, small.width, small.height)
  return { upload: big.toDataURL('image/jpeg', 0.85), thumbnail: small.toDataURL('image/jpeg', 0.7), color: dominantColor(data, small.width, small.height) }
}
