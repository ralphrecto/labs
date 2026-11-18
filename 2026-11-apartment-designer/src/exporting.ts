import { downloadBlob, slug } from './storage'

/** Render the plan SVG to a PNG at 2x and download it, with a title strip. */
export async function exportPlanPNG(svg: SVGSVGElement, title: string) {
  const w = svg.width.baseVal.value
  const h = svg.height.baseVal.value
  const clone = svg.cloneNode(true) as SVGSVGElement
  clone.setAttribute('width', String(w))
  clone.setAttribute('height', String(h))
  const xml = new XMLSerializer().serializeToString(clone)
  const img = new Image()
  const url = URL.createObjectURL(new Blob([xml], { type: 'image/svg+xml;charset=utf-8' }))
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error('Could not render the plan'))
    img.src = url
  })
  const scale = 2
  const header = 44
  const canvas = document.createElement('canvas')
  canvas.width = w * scale
  canvas.height = (h + header) * scale
  const ctx = canvas.getContext('2d')!
  ctx.scale(scale, scale)
  ctx.fillStyle = '#fbfaf7'
  ctx.fillRect(0, 0, w, h + header)
  ctx.fillStyle = '#3b3a36'
  ctx.font = '600 16px system-ui, sans-serif'
  ctx.fillText(title, 16, 28)
  ctx.drawImage(img, 0, header, w, h)
  URL.revokeObjectURL(url)
  const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, 'image/png'))
  if (blob) downloadBlob(blob, `${slug(title)}.png`)
}
