// Inline the built JS and CSS into one self-contained HTML file (dist-single/index.html),
// for hosts that serve a single page. Run after `VITE_EMBED=1 vite build`.
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

const dist = 'dist'
const html = readFileSync(join(dist, 'index.html'), 'utf8')
const assets = readdirSync(join(dist, 'assets'))
const js = assets.filter((f) => f.endsWith('.js')).map((f) => readFileSync(join(dist, 'assets', f), 'utf8'))
const css = assets.filter((f) => f.endsWith('.css')).map((f) => readFileSync(join(dist, 'assets', f), 'utf8'))
if (js.length !== 1) throw new Error(`expected one JS bundle, found ${js.length}`)
const title = html.match(/<title>.*?<\/title>/)?.[0] ?? '<title>Apartment designer</title>'
const out = `<meta charset="utf-8">
${title}
<style>${css.join('\n')}</style>
<div id="root"></div>
<script type="module">${js[0].replace(/<\/script/gi, '<\\/script')}</script>
`
mkdirSync('dist-single', { recursive: true })
writeFileSync('dist-single/index.html', out)
console.log(`dist-single/index.html: ${(out.length / 1024).toFixed(0)} KB`)
