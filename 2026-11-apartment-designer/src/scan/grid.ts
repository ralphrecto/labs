// Binary occupancy grids over the plan, with the few morphology operations detection needs.
// Square structuring elements keep axis-aligned rooms square.

export class Grid {
  readonly data: Uint8Array
  readonly w: number
  readonly h: number
  constructor(w: number, h: number, data?: Uint8Array) {
    this.w = w
    this.h = h
    this.data = data ?? new Uint8Array(w * h)
  }
  get(i: number, j: number) {
    return i >= 0 && j >= 0 && i < this.w && j < this.h ? this.data[j * this.w + i] : 0
  }
  set(i: number, j: number, v = 1) {
    this.data[j * this.w + i] = v
  }
  clone() {
    return new Grid(this.w, this.h, this.data.slice())
  }
  count() {
    let c = 0
    for (const v of this.data) c += v ? 1 : 0
    return c
  }
  and(o: Grid) {
    const g = this.clone()
    for (let k = 0; k < g.data.length; k++) g.data[k] = g.data[k] && o.data[k] ? 1 : 0
    return g
  }
  andNot(o: Grid) {
    const g = this.clone()
    for (let k = 0; k < g.data.length; k++) g.data[k] = g.data[k] && !o.data[k] ? 1 : 0
    return g
  }
  or(o: Grid) {
    const g = this.clone()
    for (let k = 0; k < g.data.length; k++) g.data[k] = g.data[k] || o.data[k] ? 1 : 0
    return g
  }
  /** Dilate with a (2r+1)² square, separably. */
  dilate(r: number) {
    return this.morph(r, true)
  }
  erode(r: number) {
    return this.morph(r, false)
  }
  close(r: number) {
    return this.dilate(r).erode(r)
  }
  open(r: number) {
    return this.erode(r).dilate(r)
  }
  private morph(r: number, dilate: boolean) {
    if (r <= 0) return this.clone()
    const { w, h } = this
    const tmp = new Uint8Array(w * h)
    const out = new Uint8Array(w * h)
    // Outside the grid counts as empty for both operations.
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        let v = dilate ? 0 : 1
        for (let d = -r; d <= r; d++) {
          const ii = i + d
          const x = ii >= 0 && ii < w ? this.data[j * w + ii] : 0
          if (dilate ? x : !x) {
            v = dilate ? 1 : 0
            break
          }
        }
        tmp[j * w + i] = v
      }
    }
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        let v = dilate ? 0 : 1
        for (let d = -r; d <= r; d++) {
          const jj = j + d
          const x = jj >= 0 && jj < h ? tmp[jj * w + i] : 0
          if (dilate ? x : !x) {
            v = dilate ? 1 : 0
            break
          }
        }
        out[j * w + i] = v
      }
    }
    return new Grid(w, h, out)
  }
  /** Fill enclosed holes: anything empty that the border can't reach becomes set. */
  fillHoles() {
    const { w, h } = this
    const seen = new Uint8Array(w * h)
    const stack: number[] = []
    const push = (i: number, j: number) => {
      if (i < 0 || j < 0 || i >= w || j >= h) return
      const k = j * w + i
      if (seen[k] || this.data[k]) return
      seen[k] = 1
      stack.push(k)
    }
    for (let i = 0; i < w; i++) {
      push(i, 0)
      push(i, h - 1)
    }
    for (let j = 0; j < h; j++) {
      push(0, j)
      push(w - 1, j)
    }
    while (stack.length) {
      const k = stack.pop()!
      const i = k % w
      const j = (k - i) / w
      push(i + 1, j)
      push(i - 1, j)
      push(i, j + 1)
      push(i, j - 1)
    }
    const g = this.clone()
    for (let k = 0; k < g.data.length; k++) if (!seen[k]) g.data[k] = 1
    return g
  }
  /** Label connected components (4- or 8-connected). Returns labels (0 = empty) and sizes by label. */
  components(eight = false): { labels: Int32Array; sizes: number[] } {
    const { w, h } = this
    const labels = new Int32Array(w * h)
    const sizes = [0]
    const stack: number[] = []
    for (let k0 = 0; k0 < w * h; k0++) {
      if (!this.data[k0] || labels[k0]) continue
      const label = sizes.length
      sizes.push(0)
      labels[k0] = label
      stack.push(k0)
      while (stack.length) {
        const k = stack.pop()!
        sizes[label]++
        const i = k % w
        const j = (k - i) / w
        for (let dj = -1; dj <= 1; dj++)
          for (let di = -1; di <= 1; di++) {
            if (!di && !dj) continue
            if (!eight && di && dj) continue
            const ii = i + di
            const jj = j + dj
            if (ii < 0 || jj < 0 || ii >= w || jj >= h) continue
            const kk = jj * w + ii
            if (this.data[kk] && !labels[kk]) {
              labels[kk] = label
              stack.push(kk)
            }
          }
      }
    }
    return { labels, sizes }
  }
  /** Keep only the largest 4-connected component. */
  largest() {
    const { labels, sizes } = this.components()
    let best = 0
    for (let l = 1; l < sizes.length; l++) if (sizes[l] > (sizes[best] ?? 0)) best = l
    const g = new Grid(this.w, this.h)
    if (best) for (let k = 0; k < labels.length; k++) g.data[k] = labels[k] === best ? 1 : 0
    return g
  }
}

/**
 * Outer boundary of a single (hole-free, 4-connected) region as a list of corner points in cell
 * units, going around with the region on the right in a y-down frame (clockwise on screen).
 */
export function traceOutline(g: Grid): { x: number; y: number }[] {
  const { w, h } = g
  // Directed boundary edges keyed by start corner. Corners are (i, j) in 0..w, 0..h.
  const key = (i: number, j: number) => j * (w + 1) + i
  const next = new Map<number, number[]>()
  const addEdge = (a: number, b: number) => {
    const l = next.get(a)
    if (l) l.push(b)
    else next.set(a, [b])
  }
  let start = -1
  for (let j = 0; j < h; j++)
    for (let i = 0; i < w; i++) {
      if (!g.get(i, j)) continue
      // Clockwise around the cell (y down): top edge left->right, right edge top->bottom, ...
      if (!g.get(i, j - 1)) addEdge(key(i, j), key(i + 1, j))
      if (!g.get(i + 1, j)) addEdge(key(i + 1, j), key(i + 1, j + 1))
      if (!g.get(i, j + 1)) addEdge(key(i + 1, j + 1), key(i, j + 1))
      if (!g.get(i - 1, j)) addEdge(key(i, j + 1), key(i, j))
      if (start < 0) start = key(i, j)
    }
  if (start < 0) return []
  const pts: { x: number; y: number }[] = []
  let cur = start
  let prevDir = { x: 1, y: 0 }
  for (let guard = 0; guard < 4 * w * h + 4; guard++) {
    const ci = cur % (w + 1)
    const cj = (cur - ci) / (w + 1)
    pts.push({ x: ci, y: cj })
    const outs = next.get(cur)
    if (!outs || !outs.length) break
    // At a pinch corner there are two ways out: take the one turning right (keeps the region on the right).
    let pick = 0
    if (outs.length > 1) {
      const turn = (b: number) => {
        const bi = b % (w + 1)
        const bj = (b - bi) / (w + 1)
        const d = { x: bi - ci, y: bj - cj }
        return prevDir.x * d.y - prevDir.y * d.x
      }
      pick = turn(outs[0]) >= turn(outs[1]) ? 0 : 1
    }
    const nb = outs.splice(pick, 1)[0]
    const bi = nb % (w + 1)
    const bj = (nb - bi) / (w + 1)
    prevDir = { x: bi - ci, y: bj - cj }
    cur = nb
    if (cur === start) break
  }
  // Keep corners only.
  const corners: { x: number; y: number }[] = []
  const n = pts.length
  for (let k = 0; k < n; k++) {
    const a = pts[(k - 1 + n) % n]
    const b = pts[k]
    const c = pts[(k + 1) % n]
    const collinear = (a.x === b.x && b.x === c.x) || (a.y === b.y && b.y === c.y)
    if (!collinear) corners.push(b)
  }
  return corners
}
