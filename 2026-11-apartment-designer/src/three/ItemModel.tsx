import { useGLTF } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { Component, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useGame } from '../game/gameStore'
import * as THREE from 'three'
import { LAMPS } from '../model/placement'
import { mountBottom } from '../model/wallDecor'
import type { Doc, Item } from '../model/types'
import { pointInPolygon } from '../model/geometry'
import { activeLayout, useStore } from '../store'
import { reduceMotion } from '../motion'
import { legs, PAL, PartMesh, rng, shade, type Part } from './furnitureKit'
import { restyle, useStyle } from './style'

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v))

// ------------------------------------------------------------------ the looks, one per category
//
// Procedural looks that echo the drawer thumbnails (ItemThumb.tsx). Local frame: width on x, depth
// on z, front is +z, floor at y = 0. The body of every piece fills the item's exact width, depth
// and height (collision is still that box); only a few props sit on top, as in the thumbnails:
// the book stack and mug on a coffee table, the laptop on a desk, the TV on a TV stand, and a bed's
// headboard, which stands taller than the mattress height the catalog lists.

function seating(item: Item): Part[] {
  const { width: w, depth: d, height: h } = item
  const legH = clamp(h * 0.1, 0.04, 0.09)
  const seatH = clamp(h * 0.52, 0.25, 0.47)
  const cushT = clamp(seatH * 0.3, 0.06, 0.14)
  const arm = clamp(w * 0.09, 0.08, Math.min(0.2, w * 0.18))
  const armH = Math.min(h - 0.02, seatH + 0.2)
  const backT = clamp(d * 0.2, 0.1, 0.2)
  const inner = w - 2 * arm
  const n = Math.max(1, Math.round(inner / 0.7))
  const gap = 0.012
  const cw = (inner - gap * (n + 1)) / n
  const seatD = d - backT - 0.02
  const bT = clamp(d * 0.17, 0.08, 0.17)
  const bH = Math.max(0.15, (h - seatH) * 0.88)
  const tilt = (8 * Math.PI) / 180
  const out: Part[] = [
    ...legs(w, d, legH, { t: 0.045, inset: 0.05, color: PAL.wood }),
    // base under the cushions, arms, back
    { kind: 'soft', size: [w, seatH - cushT - legH + 0.02, d], pos: [0, legH + (seatH - cushT - legH + 0.02) / 2, 0], r: 0.03, shade: 0.9, fabric: true },
    { kind: 'soft', size: [w, h - legH, backT], pos: [0, legH + (h - legH) / 2, -d / 2 + backT / 2], r: 0.05, shade: 0.92, fabric: true },
    { kind: 'soft', size: [arm, armH - legH, d], pos: [-w / 2 + arm / 2, legH + (armH - legH) / 2, 0], r: 0.045, fabric: true },
    { kind: 'soft', size: [arm, armH - legH, d], pos: [w / 2 - arm / 2, legH + (armH - legH) / 2, 0], r: 0.045, fabric: true },
  ]
  for (let i = 0; i < n; i++) {
    const x = -inner / 2 + gap + cw / 2 + i * (cw + gap)
    out.push({ kind: 'soft', size: [cw, cushT, seatD], pos: [x, seatH - cushT / 2, -d / 2 + backT + seatD / 2], r: 0.04, shade: 1.08, fabric: true })
    // back cushions lean back 8 degrees against the frame
    const y = seatH + bH / 2 - 0.01
    const z = -d / 2 + backT + bT / 2 - Math.sin(tilt) * (bH / 2) + 0.005
    out.push({ kind: 'soft', size: [cw, bH, bT], pos: [x, Math.min(y, h - bH / 2 - 0.005), z], rot: [-tilt, 0, 0], r: 0.05, shade: 1.03, fabric: true })
  }
  // a yellow throw pillow, like the thumbnail's
  const ps = Math.min(0.4, cw * 0.55, (h - seatH) * 0.95)
  if (ps > 0.18) {
    const px = -inner / 2 + gap + Math.min(cw * 0.32, ps * 0.75) + 0.02
    out.push({
      kind: 'soft',
      size: [ps, ps, 0.12],
      pos: [px, seatH + ps / 2 - 0.01, -d / 2 + backT + bT + 0.04],
      rot: [-0.32, 0.18, 0.2],
      r: 0.05,
      color: PAL.pillow,
      fabric: true,
    })
  }
  return out
}

function bed(item: Item): Part[] {
  const { width: w, depth: d, height: h, color } = item
  const legH = 0.07
  const frameTop = clamp(h * 0.45, 0.15, 0.3)
  const hbT = 0.07
  const hbH = Math.max(h, 1.0)
  const matTop = h - 0.15
  const len = d - hbT
  const out: Part[] = [
    ...legs(w, d, legH, { t: 0.06, inset: 0.04, color: PAL.wood }),
    { size: [w, frameTop - legH, len], pos: [0, legH + (frameTop - legH) / 2, hbT / 2], color: PAL.wood },
    // headboard with a soft upholstered panel
    { size: [w, hbH, hbT], pos: [0, hbH / 2, -d / 2 + hbT / 2], color: PAL.wood },
    {
      kind: 'soft',
      size: [w - 0.16, (hbH - matTop) * 0.62, 0.05],
      pos: [0, matTop + (hbH - matTop) * 0.52, -d / 2 + hbT + 0.02],
      r: 0.025,
      color: shade(color, 0.85),
      fabric: true,
    },
    // mattress
    { kind: 'soft', size: [w - 0.04, matTop - frameTop + 0.02, len - 0.03], pos: [0, frameTop - 0.02 + (matTop - frameTop + 0.02) / 2, hbT / 2 - 0.015], r: 0.04, color: PAL.white, fabric: true },
  ]
  // duvet over the front two thirds, draped a little over the sides, with the sheet folded back
  const duvD = len * 0.66
  const duvTop = matTop + 0.06
  const duvBot = matTop - 0.13
  out.push({ kind: 'soft', size: [w, duvTop - duvBot, duvD], pos: [0, (duvTop + duvBot) / 2, d / 2 - 0.005 - duvD / 2], r: 0.035, fabric: true })
  out.push({ kind: 'soft', size: [w + 0.0, 0.03, 0.2], pos: [0, duvTop + 0.005, d / 2 - 0.005 - duvD + 0.1], r: 0.012, color: PAL.linen, fabric: true })
  // pillows
  const two = w >= 1.05
  const pw = two ? Math.min(0.66, (w - 0.18) / 2) : Math.min(0.66, w - 0.16)
  const pz = -d / 2 + hbT + 0.25
  for (const px of two ? [-(pw / 2 + 0.03), pw / 2 + 0.03] : [0])
    out.push({ kind: 'soft', size: [pw, 0.12, 0.4], pos: [px, h - 0.075, pz], rot: [-0.14, 0, 0], r: 0.05, color: PAL.linen, fabric: true })
  return out
}

/** A round table: a "Round …" piece with a square footprint (the round dining and coffee tables). */
export function isRoundTable(item: Item): boolean {
  return (item.category === 'dining-table' || item.category === 'coffee-table' || item.category === 'side-table') && /\bround\b/i.test(item.name) && Math.abs(item.width - item.depth) < 0.03
}

/** A round top on a pedestal and a round foot, as in the drawer's Round thumbnail. */
function roundTable(item: Item): Part[] {
  const { width: w, depth: d, height: h, category } = item
  const topT = category === 'dining-table' ? 0.035 : 0.03
  const legH = h - topT
  const post = clamp(Math.min(w, d) * 0.12, 0.06, 0.12)
  const out: Part[] = [
    { kind: 'cyl', size: [w, topT, d], pos: [0, h - topT / 2, 0], seg: 40, shade: 1.06 },
    // the darker edge band, so it reads as a thick top
    { kind: 'cyl', size: [w + 0.006, topT * 0.55, d + 0.006], pos: [0, h - topT + (topT * 0.55) / 2, 0], seg: 40, shade: 0.62, noShadow: true },
    { kind: 'cyl', size: [post, legH - 0.03, post], pos: [0, 0.03 + (legH - 0.03) / 2, 0], seg: 16, shade: 0.8 },
    { kind: 'cyl', size: [w * 0.5, 0.03, d * 0.5], pos: [0, 0.015, 0], top: 0.8, seg: 28, shade: 0.75 },
  ]
  if (category === 'dining-table' && w > 0.7) {
    for (const a of [Math.PI * 0.25, Math.PI * 1.25]) {
      const px = Math.cos(a) * w * 0.28
      const pz = Math.sin(a) * d * 0.28
      out.push({ kind: 'cyl', size: [0.24, 0.014, 0.24], pos: [px, h + 0.007, pz], color: PAL.white, noShadow: true })
      out.push({ kind: 'cyl', size: [0.15, 0.016, 0.15], pos: [px, h + 0.01, pz], color: '#efe4d2', noShadow: true })
    }
    out.push({ kind: 'cyl', top: 0.7, bottom: 0.85, size: [0.09, 0.15, 0.09], pos: [0, h + 0.075, 0], color: '#7fa3b8', noShadow: true })
    out.push({ kind: 'ball', size: [0.13, 0.11, 0.13], pos: [0, h + 0.2, 0], color: '#6fae7c', noShadow: true })
  }
  return out
}

function table(item: Item): Part[] {
  if (isRoundTable(item)) return roundTable(item)
  const { width: w, depth: d, height: h, category } = item
  const dining = category === 'dining-table' || category === 'desk'
  const topT = category === 'bench' ? 0.05 : dining ? 0.035 : 0.03
  const lt = clamp(Math.min(w, d) * 0.08, 0.035, dining ? 0.06 : 0.05)
  const out: Part[] = [{ kind: 'soft', size: [w, topT, d], pos: [0, h - topT / 2, 0], r: 0.008, shade: 1.06 }]
  const legH = h - topT
  if (category === 'desk') {
    // drawer pedestal on the left, legs on the right
    const pw = Math.min(0.42, w * 0.3)
    const px = -w / 2 + pw / 2 + 0.02
    out.push({ size: [pw, legH, d - 0.06], pos: [px, legH / 2, -0.01], shade: 0.95 })
    const rows = 3
    const rh = (legH - 0.04) / rows
    for (let i = 0; i < rows; i++) {
      const y = 0.02 + rh * (i + 0.5)
      out.push({ size: [pw - 0.03, rh - 0.015, 0.012], pos: [px, y, d / 2 - 0.04 + 0.006 - 0.01], shade: 1.04 })
      out.push({ size: [0.08, 0.014, 0.014], pos: [px, y + rh * 0.2, d / 2 - 0.025], color: PAL.ink, noShadow: true })
    }
    for (const sz of [-1, 1]) out.push({ kind: 'cyl', seg: 4, bottom: 0.65, size: [lt, legH, lt], pos: [w / 2 - 0.03 - lt / 2, legH / 2, sz * (d / 2 - 0.03 - lt / 2)], shade: 0.8 })
    out.push({ size: [w - pw - 0.08, 0.07, 0.02], pos: [(pw + 0.02) / 2, legH - 0.035, -d / 2 + 0.06], shade: 0.85 })
    // a closed laptop and a mug
    if (w > 0.9 && d > 0.45) {
      out.push({ kind: 'soft', size: [0.33, 0.018, 0.23], pos: [w * 0.08, h + 0.009, 0.02], rot: [0, 0.12, 0], r: 0.006, color: PAL.metal, noShadow: true })
      out.push({ kind: 'cyl', size: [0.08, 0.095, 0.08], pos: [w * 0.33, h + 0.0475, -d * 0.18], color: PAL.books[0], noShadow: true })
    }
    return out
  }
  out.push(...legs(w, d, legH, { t: lt, inset: category === 'coffee-table' ? 0.05 : 0.03, shade: 0.8 }))
  if (dining || category === 'console-table') {
    // an apron under the top, set just inside the legs, on all four sides
    const a = category === 'dining-table' ? 0.09 : 0.07
    const ax = w / 2 - 0.03 - lt / 2
    const az = d / 2 - 0.03 - lt / 2
    out.push({ size: [2 * ax, a, 0.022], pos: [0, legH - a / 2, az], shade: 0.78 })
    out.push({ size: [2 * ax, a, 0.022], pos: [0, legH - a / 2, -az], shade: 0.78 })
    out.push({ size: [0.022, a, 2 * az], pos: [ax, legH - a / 2, 0], shade: 0.78 })
    out.push({ size: [0.022, a, 2 * az], pos: [-ax, legH - a / 2, 0], shade: 0.78 })
  }
  if (category === 'dining-table') {
    // a darker band round the edge of the top, so it reads as a thick tabletop and not a slab
    out.push({ size: [w + 0.006, topT * 0.55, d + 0.006], pos: [0, h - topT + (topT * 0.55) / 2, 0], shade: 0.62, noShadow: true })
    if (w > 0.7 && d > 0.6) {
      // two plates and a small vase with a sprig, as in the thumbnail
      const pz = d / 2 - 0.2
      for (const sx of [-1, 1]) {
        out.push({ kind: 'cyl', size: [0.24, 0.014, 0.24], pos: [sx * w * 0.24, h + 0.007, sx * pz * 0.9], color: PAL.white, noShadow: true })
        out.push({ kind: 'cyl', size: [0.15, 0.016, 0.15], pos: [sx * w * 0.24, h + 0.01, sx * pz * 0.9], color: '#efe4d2', noShadow: true })
      }
      out.push({ kind: 'cyl', top: 0.7, bottom: 0.85, size: [0.09, 0.15, 0.09], pos: [0, h + 0.075, 0], color: '#7fa3b8', noShadow: true })
      out.push({ kind: 'ball', size: [0.13, 0.11, 0.13], pos: [0, h + 0.2, 0], color: '#6fae7c', noShadow: true })
    }
  }
  if (category === 'side-table' || category === 'console-table') {
    out.push({ size: [w - 0.06 - 2 * lt, 0.022, d - 0.06], pos: [0, Math.min(0.16, h * 0.25), 0], shade: 0.95 })
  }
  if (category === 'bench') {
    out.push({ size: [w - 0.06 - 2 * lt, 0.03, 0.03], pos: [0, h * 0.35, 0], shade: 0.8 })
  }
  if (category === 'coffee-table' && w > 0.7 && d > 0.4) {
    // a book stack and a mug, as in the thumbnail
    const bx = -w * 0.2
    out.push({ size: [0.26, 0.035, 0.19], pos: [bx, h + 0.0175, -0.02], rot: [0, 0.08, 0], color: PAL.books[2], noShadow: true })
    out.push({ size: [0.21, 0.03, 0.15], pos: [bx + 0.01, h + 0.05, -0.02], rot: [0, -0.22, 0], color: PAL.books[0], noShadow: true })
    out.push({ kind: 'cyl', size: [0.085, 0.09, 0.085], pos: [w * 0.24, h + 0.045, 0.06], color: PAL.white, noShadow: true })
  }
  return out
}

function chair(item: Item): Part[] {
  const { width: w, depth: d, height: h, category } = item
  if (category === 'office-chair') {
    const out: Part[] = []
    const reach = Math.min(w, d) / 2 - 0.03
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2 + Math.PI / 2
      const cx = (Math.cos(a) * reach) / 2
      const cz = (Math.sin(a) * reach) / 2
      out.push({ size: [reach, 0.03, 0.04], pos: [cx, 0.07, cz], rot: [0, -a, 0], color: PAL.dark, noShadow: true })
      out.push({ kind: 'ball', size: [0.05, 0.05, 0.05], pos: [Math.cos(a) * (reach - 0.01), 0.025, Math.sin(a) * (reach - 0.01)], color: PAL.dark, noShadow: true })
    }
    const seatH = Math.min(0.48, h * 0.45)
    out.push({ kind: 'cyl', size: [0.05, seatH - 0.1, 0.05], pos: [0, 0.07 + (seatH - 0.1) / 2, 0], color: PAL.metal })
    const sw = w * 0.78
    const sd = d * 0.72
    out.push({ kind: 'soft', size: [sw, 0.08, sd], pos: [0, seatH, 0.02], r: 0.035, fabric: true })
    const backH = h - seatH - 0.08
    out.push({ size: [0.04, 0.16, 0.03], pos: [0, seatH + 0.06, -sd / 2 - 0.0], color: PAL.dark })
    out.push({ kind: 'soft', size: [sw * 0.88, backH, 0.07], pos: [0, h - backH / 2, -d / 2 + 0.06], rot: [-0.1, 0, 0], r: 0.04, fabric: true, shade: 0.95 })
    for (const sx of [-1, 1]) {
      out.push({ size: [0.03, 0.18, 0.03], pos: [sx * (sw / 2 + 0.01), seatH + 0.09, 0.02], color: PAL.dark, noShadow: true })
      out.push({ kind: 'soft', size: [0.05, 0.025, sd * 0.6], pos: [sx * (sw / 2 + 0.01), seatH + 0.19, 0.02], r: 0.01, color: PAL.dark, noShadow: true })
    }
    return out
  }
  const seatH = Math.min(0.46, h * 0.5)
  const seatT = 0.045
  const lt = clamp(w * 0.08, 0.03, 0.045)
  const out: Part[] = []
  // front legs; the back legs run up into the back posts
  for (const sx of [-1, 1]) {
    const x = sx * (w / 2 - 0.02 - lt / 2)
    out.push({ kind: 'cyl', seg: 4, bottom: 0.7, size: [lt, seatH - seatT, lt], pos: [x, (seatH - seatT) / 2, d / 2 - 0.02 - lt / 2], shade: 0.85 })
    out.push({ size: [lt, h, lt], pos: [x, h / 2, -d / 2 + lt / 2 + 0.01], shade: 0.85 })
  }
  out.push({ kind: 'soft', size: [w, seatT, d - 0.02], pos: [0, seatH - seatT / 2, 0.01], r: 0.012 })
  const rail = w - 2 * lt - 0.04
  out.push({ kind: 'soft', size: [rail, 0.09, 0.025], pos: [0, h - 0.07, -d / 2 + lt / 2 + 0.01], r: 0.01 })
  out.push({ size: [rail, 0.04, 0.02], pos: [0, seatH + (h - seatH) * 0.45, -d / 2 + lt / 2 + 0.01] })
  return out
}

function cabinet(item: Item): Part[] {
  const { width: w, depth: d, height: h, category } = item
  const wardrobe = category === 'wardrobe'
  const legH = wardrobe ? 0 : category === 'tv-stand' ? 0.07 : clamp(h * 0.1, 0.05, 0.1)
  const plinth = wardrobe ? 0.07 : 0
  const topT = wardrobe ? 0.04 : 0.025
  const bodyBot = legH + plinth
  const bodyTop = h - topT
  const bh = bodyTop - bodyBot
  const front = d / 2 - 0.03
  const out: Part[] = [
    { size: [w - 0.02, bh, d - 0.03], pos: [0, bodyBot + bh / 2, -0.015], shade: 0.94 },
    { size: [w, topT, d], pos: [0, h - topT / 2, 0], shade: 1.05 },
  ]
  if (legH) out.push(...legs(w, d, legH, { t: 0.04, inset: 0.04, shade: 0.7, taper: 0.7 }))
  if (plinth) out.push({ size: [w - 0.06, plinth, d - 0.06], pos: [0, plinth / 2, -0.01], shade: 0.6 })
  const handle = PAL.ink
  const gap = 0.012
  const panel = (x: number, y: number, pw: number, ph: number) =>
    out.push({ size: [pw - gap, ph - gap, 0.015], pos: [x, y, front + 0.0075], shade: 1.04 })
  const pull = (x: number, y: number, vertical = false, long = 0.1) =>
    out.push({ size: vertical ? [0.014, long, 0.014] : [long, 0.014, 0.014], pos: [x, y, front + 0.022], color: handle, noShadow: true })
  const inset = 0.02
  const fw = w - 0.02 - 2 * inset
  const fb = bodyBot + inset
  const fh = bh - 2 * inset
  if (category === 'nightstand' || category === 'dresser') {
    const rows = category === 'nightstand' ? 2 : clamp(Math.round(fh / 0.22), 2, 5)
    const cols = category === 'dresser' && w > 0.9 ? 2 : 1
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const x = -fw / 2 + (fw / cols) * (c + 0.5)
        const y = fb + (fh / rows) * (r + 0.5)
        panel(x, y, fw / cols, fh / rows)
        pull(x, y + fh / rows / 2 - 0.045, false, Math.min(0.12, fw / cols / 3))
      }
  } else if (category === 'tv-stand') {
    const cols = clamp(Math.round(w / 0.5), 2, 4)
    for (let c = 0; c < cols; c++) {
      const x = -fw / 2 + (fw / cols) * (c + 0.5)
      const y = fb + fh / 2
      panel(x, y, fw / cols, fh)
      out.push({ kind: 'ball', size: [0.035, 0.035, 0.03], pos: [x + (c < cols / 2 ? 1 : -1) * (fw / cols / 2 - 0.06), y + fh * 0.12, front + 0.02], color: handle, noShadow: true })
    }
    // the TV on a little foot, against the back
    const tw = Math.min(w * 0.78, 1.45)
    const th = tw * 0.57
    const tz = -d / 2 + 0.12
    out.push({ size: [Math.min(0.32, tw * 0.3), 0.02, 0.16], pos: [0, h + 0.01, tz], color: PAL.dark })
    out.push({ size: [0.06, 0.07, 0.03], pos: [0, h + 0.055, tz], color: PAL.dark })
    out.push({ size: [tw, th, 0.04], pos: [0, h + 0.09 + th / 2, tz], color: PAL.dark })
    out.push({ size: [tw - 0.04, th - 0.04, 0.005], pos: [0, h + 0.09 + th / 2, tz + 0.021], color: PAL.screen, noShadow: true })
  } else {
    // sideboard and wardrobe: doors with vertical pulls meeting in the middle of each pair
    const cols = clamp(Math.round(w / (wardrobe ? 0.5 : 0.42)), 1, 6)
    for (let c = 0; c < cols; c++) {
      const x = -fw / 2 + (fw / cols) * (c + 0.5)
      const y = fb + fh / 2
      panel(x, y, fw / cols, fh)
      const side = cols === 1 ? 1 : c % 2 === 0 ? 1 : -1
      const py = wardrobe ? Math.min(1.1, fb + fh * 0.55) : fb + fh * 0.7
      pull(x + side * (fw / cols / 2 - 0.045), py, true, wardrobe ? 0.22 : 0.1)
    }
  }
  return out
}

function bookshelf(item: Item): Part[] {
  const { width: w, depth: d, height: h } = item
  const t = 0.022
  const shelves = Math.max(2, Math.round(h / 0.36))
  const out: Part[] = [
    { size: [w - 2 * t, h - t, 0.012], pos: [0, (h - t) / 2, -d / 2 + 0.006], shade: 0.72 },
    { size: [t, h, d], pos: [-w / 2 + t / 2, h / 2, 0] },
    { size: [t, h, d], pos: [w / 2 - t / 2, h / 2, 0] },
  ]
  const span = (h - t) / shelves
  const rnd = rng(item.id)
  for (let i = 0; i <= shelves; i++) out.push({ size: [w - 2 * t, t, d - 0.01], pos: [0, t / 2 + i * span, 0.005], shade: 1.06 })
  const inner = w - 2 * t - 0.02
  for (let i = 0; i < shelves; i++) {
    const floor = t + i * span
    const room = span - t - 0.03
    if (i === shelves - 1 && inner > 0.3) {
      // a little plant on the top shelf
      const ps = Math.min(0.11, room * 0.45)
      out.push({ kind: 'cyl', bottom: 0.75, size: [ps, ps, ps], pos: [inner / 2 - ps, floor + ps / 2, 0], color: PAL.pot, noShadow: true })
      out.push({ kind: 'ball', size: [ps * 1.5, ps * 1.3, ps * 1.4], pos: [inner / 2 - ps, floor + ps * 1.45, 0], color: '#5f8f5a', noShadow: true })
    }
    let x = -inner / 2
    const stop = i === shelves - 1 ? inner * 0.15 : inner / 2 - (0.1 + rnd() * inner * 0.35)
    let k = 0
    while (x < stop) {
      const bw = 0.03 + rnd() * 0.025
      const bh = Math.min(room, 0.17 + rnd() * 0.09)
      const bd = Math.min(d - 0.06, 0.16 + rnd() * 0.06)
      const color = PAL.books[Math.floor(rnd() * PAL.books.length)]
      out.push({ size: [bw, bh, bd], pos: [x + bw / 2, floor + bh / 2, d / 2 - 0.03 - bd / 2], color, noShadow: true })
      x += bw + 0.003
      k++
      if (k > 9) break
    }
    // one book leaning on the row
    if (x < inner / 2 - 0.12) {
      const bh = Math.min(room, 0.22)
      const lean = 0.32
      out.push({ size: [0.03, bh, 0.17], pos: [x + Math.sin(lean) * bh * 0.5 + 0.017, floor + (Math.cos(lean) * bh) / 2 + 0.008, d / 2 - 0.03 - 0.085], rot: [0, 0, -lean], color: PAL.books[(i + 3) % 6], noShadow: true })
    }
  }
  return out
}

/** A lamp's shade height, shade diameter and bulb height: shared by the model and its evening glow. */
function shadeDims(item: Item) {
  const h = item.height
  const s = Math.min(item.width, item.depth)
  const shadeH = item.category === 'table-lamp' ? h * 0.38 : clamp(h * 0.2, 0.15, 0.36)
  return { shadeH, s, bulbY: h - shadeH * 0.6 }
}

/** How high a pendant hangs above its host table's top (matches Scene3D's 0.76 m lift). */
const PENDANT_DROP = 0.76
/** A dining table's usual height and a usual ceiling, when the pendant isn't told its own. */
const PENDANT_HOST_H = 0.75
const PENDANT_CEILING = 2.5

/**
 * A pendant: a dome shade over the table, a bulb inside, and a cord up to a small canopy on the
 * ceiling. The cord runs from the shade's top to the ceiling (clamped so it never vanishes or
 * dangles absurdly). The shade casts no shadow: by day its dome left a stain-like blob on the table.
 */
function pendant(item: Item, hostH = PENDANT_HOST_H, ceiling = PENDANT_CEILING): Part[] {
  const h = item.height
  const s = Math.min(item.width, item.depth)
  const cord = clamp(ceiling - (hostH + PENDANT_DROP + h), 0.25, 0.9)
  return [
    { kind: 'cyl', size: [s, h * 0.7, s], pos: [0, h * 0.35, 0], top: 0.32, seg: 24, color: item.color ?? PAL.shade, noShadow: true },
    { kind: 'cyl', size: [s * 0.3, h * 0.3, s * 0.3], pos: [0, h * 0.85, 0], top: 0.6, seg: 16, color: item.color ?? PAL.shade, noShadow: true },
    { kind: 'ball', size: [0.08, 0.09, 0.08], pos: [0, h * 0.25, 0], color: PAL.shade, noShadow: true },
    { kind: 'cyl', size: [0.008, cord, 0.008], pos: [0, h + cord / 2, 0], seg: 8, color: '#5a4e44', noShadow: true },
    { kind: 'cyl', size: [0.11, 0.025, 0.11], pos: [0, h + cord - 0.0125, 0], seg: 18, color: PAL.dark, noShadow: true },
  ]
}

/** A table lamp: a ceramic body, a short neck, a bulb and a fabric shade. */
function tableLamp(item: Item): Part[] {
  const h = item.height
  const { shadeH, s, bulbY } = shadeDims(item)
  const bodyH = h * 0.5
  const neck = Math.max(0.01, h - shadeH - bodyH)
  return [
    { kind: 'cyl', size: [s * 0.44, bodyH, s * 0.44], pos: [0, bodyH / 2, 0], top: 0.7, seg: 18, color: item.color ?? PAL.shade },
    { kind: 'cyl', size: [0.02, neck + 0.04, 0.02], pos: [0, bodyH + neck / 2, 0], seg: 10, color: PAL.metal },
    { kind: 'ball', size: [0.06, 0.07, 0.06], pos: [0, bulbY, 0], color: PAL.shade, noShadow: true },
    { kind: 'cyl', size: [s, shadeH, s], pos: [0, h - shadeH / 2, 0], top: 0.78, seg: 18, fabric: true },
  ]
}

function lamp(item: Item): Part[] {
  const { width: w, depth: d, height: h } = item
  const s = Math.min(w, d)
  const { shadeH } = shadeDims(item)
  return [
    { kind: 'cyl', size: [s * 0.65, 0.03, s * 0.65], pos: [0, 0.015, 0], top: 0.85, seg: 16, color: PAL.dark },
    { kind: 'cyl', size: [0.025, h - shadeH - 0.02, 0.025], pos: [0, (h - shadeH) / 2, 0], seg: 12, color: PAL.metal },
    { kind: 'ball', size: [0.07, 0.09, 0.07], pos: [0, h - shadeH * 0.6, 0], color: PAL.shade, noShadow: true },
    { kind: 'cyl', size: [s, shadeH, s], pos: [0, h - shadeH / 2, 0], top: 0.72, seg: 16, fabric: true },
  ]
}

function plant(item: Item): Part[] {
  const { width: w, depth: d, height: h } = item
  const potH = clamp(h * 0.3, 0.12, 0.38)
  const pw = Math.min(w, d) * 0.62
  const ch = h - potH
  const leaf = (sx: number, sy: number, sz: number, x: number, y: number, z: number, lit: boolean): Part => ({
    kind: 'ball',
    size: [w * sx, ch * sy, d * sz],
    pos: [w * x, potH + ch * y, d * z],
    shade: lit ? 1.18 : 1,
    fabric: true,
  })
  return [
    { kind: 'cyl', size: [pw, potH - 0.03, pw], pos: [0, (potH - 0.03) / 2, 0], bottom: 0.76, seg: 14, color: PAL.pot },
    { kind: 'cyl', size: [pw * 1.08, 0.05, pw * 1.08], pos: [0, potH - 0.025, 0], seg: 14, color: PAL.potRim },
    { kind: 'cyl', size: [0.03, ch * 0.3, 0.03], pos: [0, potH + ch * 0.15, 0], seg: 12, color: '#7a5a3c' },
    leaf(0.78, 0.5, 0.78, 0, 0.36, 0, false),
    leaf(0.5, 0.36, 0.5, -0.25, 0.25, 0.08, true),
    leaf(0.5, 0.38, 0.5, 0.25, 0.32, -0.06, false),
    leaf(0.46, 0.32, 0.46, 0.04, 0.2, 0.27, true),
    leaf(0.6, 0.42, 0.6, 0.02, 0.79, -0.02, true),
  ]
}

function rug(item: Item): Part[] {
  const { width: w, depth: d, color } = item
  const h = Math.max(0.006, item.height)
  const b = clamp(Math.min(w, d) * 0.08, 0.05, 0.16)
  const parts: Part[] = [
    { size: [w, h * 0.7, d], pos: [0, h * 0.35, 0], color: shade(color, 0.82), fabric: true },
    { size: [w - 2 * b, h * 0.85, d - 2 * b], pos: [0, h * 0.425, 0], fabric: true },
    { size: [w - 2 * b - 0.08, h * 0.92, d - 2 * b - 0.08], pos: [0, h * 0.46, 0], color: shade(color, 1.16), fabric: true },
    { size: [w - 2 * b - 0.12, h * 0.96, d - 2 * b - 0.12], pos: [0, h * 0.48, 0], fabric: true },
  ]
  const m = Math.min(w - 2 * b, d - 2 * b) * 0.38
  if (m > 0.12) parts.push({ size: [m, h, m], pos: [0, h / 2, 0], rot: [0, Math.PI / 4, 0], color: shade(color, 1.16), fabric: true })
  return parts
}

function ottoman(item: Item): Part[] {
  const { width: w, depth: d, height: h } = item
  const legH = clamp(h * 0.12, 0.03, 0.06)
  const topT = clamp((h - legH) * 0.32, 0.06, 0.14)
  const out: Part[] = [
    ...legs(w, d, legH, { t: 0.035, inset: 0.05, color: PAL.wood }),
    { kind: 'soft', size: [w, h - legH - topT + 0.02, d], pos: [0, legH + (h - legH - topT + 0.02) / 2, 0], r: 0.04, shade: 0.92, fabric: true },
    { kind: 'soft', size: [w, topT, d], pos: [0, h - topT / 2, 0], r: 0.05, shade: 1.06, fabric: true },
  ]
  // tufting buttons
  const nx = Math.max(1, Math.round(w / 0.25))
  const nz = Math.max(1, Math.round(d / 0.25))
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < nz; j++)
      out.push({ kind: 'ball', size: [0.025, 0.012, 0.025], pos: [-w / 2 + (w / nx) * (i + 0.5), h - 0.003, -d / 2 + (d / nz) * (j + 0.5)], shade: 0.55, noShadow: true })
  return out
}

/** Procedural looks for catalog categories. Local frame: width on x, depth on z, front is +z, floor at y=0. */
export function parts(item: Item, hang?: { hostH?: number; ceiling?: number }): Part[] {
  switch (item.category) {
    case 'sofa':
    case 'armchair':
      return seating(item)
    case 'bed':
      return bed(item)
    case 'coffee-table':
    case 'side-table':
    case 'console-table':
    case 'bench':
    case 'dining-table':
    case 'desk':
      return table(item)
    case 'nightstand':
    case 'sideboard':
    case 'dresser':
    case 'tv-stand':
    case 'wardrobe':
      return cabinet(item)
    case 'dining-chair':
    case 'office-chair':
      return chair(item)
    case 'bookshelf':
      return bookshelf(item)
    case 'ottoman':
      return ottoman(item)
    case 'rug':
      return rug(item)
    case 'floor-lamp':
      return lamp(item)
    case 'table-lamp':
      return tableLamp(item)
    case 'pendant':
      return pendant(item, hang?.hostH, hang?.ceiling)
    case 'plant':
      return plant(item)
    case 'wall-art':
    case 'mirror':
    case 'wall-shelf':
    case 'wall-clock':
    case 'curtains':
      return wallDecor(item)
    default: {
      if (item.category === 'box' && /box|carton/i.test(item.name)) return movingBox(item)
      const { width: w, depth: d, height: h } = item
      return [{ kind: 'soft', size: [w, h, d], pos: [0, h / 2, 0], r: Math.min(0.02, w / 4, d / 4, h / 4) } satisfies Part]
    }
  }
}

// ------------------------------------------------------------------ wall decor
// Hung on the wall: the back of the box (z = -d/2) is the wall face, and the height above the floor
// comes from the category (mountBottom), so these parts are lifted to it here.

/** Abstract canvas patterns, picked by the item's id so two prints differ. */
const CANVAS_INKS = ['#ffc24b', '#5fa67f', '#4f8be8', '#ff7a59', '#8a5cc7', '#fbf6ee']

function wallDecor(item: Item): Part[] {
  const { width: w, depth: d, height: h } = item
  const y0 = mountBottom(item)
  const cy = y0 + h / 2
  const back = -d / 2
  const face = [Math.PI / 2, 0, 0] as [number, number, number]
  switch (item.category) {
    case 'wall-art': {
      const r = rng(item.id)
      const f = Math.min(0.045, w * 0.08)
      const pick = () => CANVAS_INKS[Math.floor(r() * CANVAS_INKS.length)]
      const cw = w - 2 * f
      const ch = h - 2 * f
      const z = back + d * 0.75
      return [
        // Frame and a pale mat, then a canvas in the item's colour with a few soft blocks of colour.
        { kind: 'box', size: [w, h, d * 0.7], pos: [0, cy, back + d * 0.35], color: PAL.wood, shade: 1.15 },
        { kind: 'box', size: [cw, ch, 0.006], pos: [0, cy, back + d * 0.7 + 0.003], color: shade(item.color, 1.25) },
        { kind: 'ball', size: [cw * 0.34, cw * 0.34, 0.01], pos: [cw * (r() * 0.4 - 0.2), cy + ch * 0.2, z], color: pick(), noShadow: true },
        { kind: 'box', size: [cw * 0.9, ch * 0.32, 0.008], pos: [0, cy - ch * 0.28, z], color: item.color, noShadow: true },
        { kind: 'box', size: [cw * 0.45, ch * 0.16, 0.009], pos: [cw * (r() * 0.3 - 0.15), cy - ch * 0.08, z + 0.001], color: pick(), noShadow: true },
        { kind: 'ball', size: [cw * 0.2, cw * 0.2, 0.01], pos: [-cw * 0.28, cy + ch * 0.05, z + 0.001], color: pick(), noShadow: true },
      ]
    }
    case 'mirror': {
      const s = Math.min(w, h)
      return [
        { kind: 'cyl', size: [s, d * 0.7, s], pos: [0, cy, back + d * 0.35], rot: face, seg: 40, color: item.color },
        { kind: 'cyl', size: [s * 0.84, 0.006, s * 0.84], pos: [0, cy, back + d * 0.7 + 0.003], rot: face, seg: 40, color: '#e4f2f6' },
        // A light streak across the glass.
        { kind: 'box', size: [s * 0.08, s * 0.4, 0.004], pos: [-s * 0.12, cy + s * 0.06, back + d * 0.7 + 0.008], rot: [0, 0, -0.6], color: '#ffffff', noShadow: true },
      ]
    }
    case 'wall-shelf': {
      const t = 0.035
      const shelfY = y0 + t / 2
      const out: Part[] = [
        { kind: 'box', size: [w, t, d], pos: [0, shelfY, 0], color: item.color },
        { kind: 'box', size: [0.02, 0.06, d * 0.5], pos: [-w * 0.35, shelfY - 0.04, back + d * 0.25], color: PAL.dark },
        { kind: 'box', size: [0.02, 0.06, d * 0.5], pos: [w * 0.35, shelfY - 0.04, back + d * 0.25], color: PAL.dark },
      ]
      // Three or four books standing, the last one leaning; a little plant at the other end.
      const bh = Math.max(0.1, h - t - 0.03)
      const n = w >= 0.7 ? 4 : 3
      let x = -w / 2 + 0.06
      for (let i = 0; i < n; i++) {
        const hh = bh * [0.92, 0.78, 1, 0.7][i]
        const bw = [0.05, 0.04, 0.055, 0.045][i]
        const lean = i === n - 1 ? -0.3 : 0
        out.push({ kind: 'box', size: [bw, hh, d * 0.75], pos: [x + bw / 2 + (lean ? 0.03 : 0), y0 + t + hh / 2 - (lean ? 0.008 : 0), 0], rot: [0, 0, lean], color: PAL.books[i], noShadow: true })
        x += bw + 0.005
      }
      out.push({ kind: 'cyl', size: [0.09, 0.08, 0.09], pos: [w / 2 - 0.1, y0 + t + 0.04, 0], bottom: 0.8, seg: 12, color: PAL.pot, noShadow: true })
      out.push({ kind: 'ball', size: [0.14, 0.12, 0.12], pos: [w / 2 - 0.1, y0 + t + 0.13, 0], color: '#5f8f5a', fabric: true, noShadow: true })
      return out
    }
    case 'wall-clock': {
      const s = Math.min(w, h)
      const z = back + d * 0.8
      return [
        { kind: 'cyl', size: [s, d * 0.8, s], pos: [0, cy, back + d * 0.4], rot: face, seg: 32, color: PAL.wood },
        { kind: 'cyl', size: [s * 0.86, 0.006, s * 0.86], pos: [0, cy, z + 0.003], rot: face, seg: 32, color: item.color },
        { kind: 'box', size: [0.014, s * 0.28, 0.005], pos: [0, cy + s * 0.12, z + 0.008], color: PAL.ink, noShadow: true },
        { kind: 'box', size: [0.012, s * 0.2, 0.005], pos: [s * 0.07, cy - s * 0.04, z + 0.01], rot: [0, 0, -2.1], color: PAL.ink, noShadow: true },
        { kind: 'cyl', size: [0.025, 0.006, 0.025], pos: [0, cy, z + 0.013], rot: face, seg: 12, color: '#ff7a59', noShadow: true },
      ]
    }
    default: {
      // Curtains: a rod with two soft panels, each a few rounded folds, gathered toward the sides.
      const rodY = y0 + h - 0.02
      const out: Part[] = [{ kind: 'cyl', size: [0.025, w + 0.12, 0.025], pos: [0, rodY, 0], rot: [0, 0, Math.PI / 2], seg: 10, color: PAL.metal }]
      const panel = w * 0.26
      const folds = 4
      const fw = panel / folds
      for (const side of [-1, 1])
        for (let i = 0; i < folds; i++) {
          const x = side * (w / 2 - fw * (i + 0.5))
          out.push({ kind: 'soft', size: [fw * 1.08, h - 0.05, d * (i % 2 ? 0.55 : 0.8)], pos: [x, y0 + (h - 0.05) / 2, (i % 2 ? -0.1 : 0.1) * d], r: Math.min(fw * 0.5, 0.03), fabric: true, shade: i % 2 ? 0.92 : 1.04 })
        }
      return out
    }
  }
}

/** A cardboard moving box with a strip of tape across the top and down the front and back. */
function movingBox(item: Item): Part[] {
  const { width: w, depth: d, height: h } = item
  const card = '#c8a27a'
  const tape = '#e6cf9c'
  const tw = Math.min(0.06, w * 0.14)
  return [
    { kind: 'soft', size: [w, h, d], pos: [0, h / 2, 0], r: Math.min(0.012, w / 6, h / 6), color: card },
    { size: [tw, 0.004, d + 0.004], pos: [0, h + 0.001, 0], color: tape, noShadow: true },
    { size: [tw, h * 0.22, 0.004], pos: [0, h - h * 0.11, d / 2 + 0.002], color: tape, noShadow: true },
    { size: [tw, h * 0.22, 0.004], pos: [0, h - h * 0.11, -d / 2 - 0.002], color: tape, noShadow: true },
  ]
}

/** Plants and lamps are all curves: an edge outline on them is a scribble, so they show selection by the ring alone. */
const ORGANIC = new Set(['plant', 'floor-lamp', 'table-lamp', 'pendant'])

/** The height of the table a pendant hangs over (in the active design), if it hangs over one. */
function pendantHostHeight(doc: Doc, itemId: string): number | undefined {
  const l = activeLayout(doc)
  const p = l.placements.find((x) => x.itemId === itemId)
  const host = p?.on ? l.placements.find((x) => x.id === p.on) : undefined
  return host ? doc.items.find((i) => i.id === host.itemId)?.height : undefined
}

/** The ceiling height of the room a pendant hangs in (the lowest ceiling when it's in none). */
function pendantCeiling(doc: Doc, itemId: string): number | undefined {
  const p = activeLayout(doc).placements.find((x) => x.itemId === itemId)
  if (!doc.rooms.length) return undefined
  const room = p && doc.rooms.find((r) => pointInPolygon({ x: p.x, y: p.y }, r.polygon))
  return room?.ceilingHeight ?? Math.min(...doc.rooms.map((r) => r.ceilingHeight))
}

/** Wood pieces whose big flat fronts the Cartoon look would otherwise lift to a bright orange. */
const TOON_WOOD = new Set(['nightstand', 'dresser', 'sideboard', 'tv-stand', 'wardrobe', 'coffee-table', 'side-table', 'console-table', 'dining-table', 'desk', 'bench'])

function hsl(hex: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) return null
  const n = parseInt(m[1], 16)
  const r = ((n >> 16) & 255) / 255
  const g = ((n >> 8) & 255) / 255
  const b = (n & 255) / 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  if (max === min) return [0, 0, l]
  const dd = max - min
  const s = l > 0.5 ? dd / (2 - max - min) : dd / (max + min)
  let h = max === r ? (g - b) / dd + (g < b ? 6 : 0) : max === g ? (b - r) / dd + 2 : (r - g) / dd + 4
  h /= 6
  return [h, s, l]
}
function hslHex(h: number, s: number, l: number): string {
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  const f = (t: number) => {
    if (t < 0) t += 1
    if (t > 1) t -= 1
    if (t < 1 / 6) return p + (q - p) * 6 * t
    if (t < 1 / 2) return q
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
    return p
  }
  const c = (v: number) => Math.round(clamp(v, 0, 1) * 255).toString(16).padStart(2, '0')
  return '#' + (s === 0 ? c(l) + c(l) + c(l) : c(f(h + 1 / 3)) + c(f(h)) + c(f(h - 1 / 3)))
}

/**
 * The colour to hand the Cartoon look for a wood piece so it comes out like its drawer card: the
 * Cartoon look (toonHex: saturation x1.35, lightness lifted and snapped to sixths) would turn a
 * walnut dresser a flat orange, so this undoes the saturation boost and picks the lightness that
 * lands on the band at or just below the colour's own.
 */
export function toonWoodBase(hex: string): string {
  const c = hsl(hex)
  if (!c) return hex
  const [h, s, l] = c
  const band = Math.max(1 / 6, Math.floor(l * 6 + 0.25) / 6)
  const preL = clamp((band - 0.12) / 0.82, 0, 1)
  const preS = clamp((s - 0.04) / 1.35, 0, 1)
  return hslHex(h, preS, preL)
}

/**
 * Every other painted colour (fabric, upholstery, lacquer) for the Cartoon look: undo its
 * saturation boost and aim the lightness at the band nearest the colour's own, so a terracotta or
 * sage sofa comes out like its swatch instead of a louder cousin.
 */
export function toonSwatchBase(hex: string): string {
  const c = hsl(hex)
  if (!c) return hex
  const [h, s, l] = c
  const band = clamp(Math.round(l * 6) / 6, 1 / 6, 5 / 6)
  const preL = clamp((band - 0.12) / 0.82, 0, 1)
  const preS = clamp((s - 0.04) / 1.35, 0, 1)
  return hslHex(h, preS, preL)
}

/** The colour handed to the parts in the given look, so painted pieces match their swatches. */
export function styleBase(style: string, category: string, color: string): string {
  if (style !== 'toon') return color
  return TOON_WOOD.has(category) ? toonWoodBase(color) : toonSwatchBase(color)
}

export function ProceduralItem({ item, highlight }: { item: Item; highlight?: string }) {
  const { style } = useStyle()
  const base = styleBase(style, item.category, item.color)
  // A pendant's cord runs up to the real ceiling over its real table.
  const hostH = useStore((st) => (item.category === 'pendant' ? pendantHostHeight(st.doc, item.id) : undefined))
  const ceiling = useStore((st) => (item.category === 'pendant' ? pendantCeiling(st.doc, item.id) : undefined))
  const specs = useMemo(() => parts(item, { hostH, ceiling }), [item, hostH, ceiling])
  const lineStyle = style === 'blueprint' || style === 'scan' || style === 'classic'
  const hl = ORGANIC.has(item.category) && !lineStyle ? undefined : highlight
  return (
    <group>
      {specs.map((s, i) => (
        <PartMesh key={i} part={s} base={base} highlight={hl} />
      ))}
    </group>
  )
}

// ------------------------------------------------------------------ drop-and-settle on arrival

/** Item ids already shown. A remount later (camera change, style switch) doesn't replay the drop. */
const seen = new Set<string>()
const DROP_MS = 300
const PENDING = -1

/** A new item drops in and settles with a little squash. Items already there when the app opens don't. */
function Arrival({ id, children }: { id: string; children: ReactNode }) {
  const ref = useRef<THREE.Group>(null)
  /** 0: still; PENDING: start on the next frame; otherwise the start time. */
  const born = useRef(0)
  useLayoutEffect(() => {
    if (seen.has(id)) return
    seen.add(id)
    // With "reduce motion" on, new items simply appear in place.
    if (performance.now() > 2500 && !reduceMotion) born.current = PENDING
  }, [id])
  useFrame(() => {
    const g = ref.current
    if (!g || !born.current) return
    // The clock starts on the first frame, so a slow first render doesn't eat the animation.
    if (born.current === PENDING) born.current = performance.now()
    const t = (performance.now() - born.current) / DROP_MS
    if (t >= 1) {
      g.scale.set(1, 1, 1)
      g.position.y = 0
      born.current = 0
      return
    }
    if (t < 0.45) {
      const u = t / 0.45
      g.position.y = 0.28 * (1 - u * u)
      g.scale.set(0.96, 1.05, 0.96)
    } else {
      const u = (t - 0.45) / 0.55
      const squash = Math.sin(u * Math.PI * 2) * (1 - u) * 0.13
      g.position.y = 0
      g.scale.set(1 + squash * 0.5, 1 - squash, 1 + squash * 0.5)
    }
  })
  return <group ref={ref}>{children}</group>
}

function GLBItem({ item }: { item: Item }) {
  const { scene } = useGLTF(item.glbUrl!)
  const { style } = useStyle()
  const obj = useMemo(() => {
    // Scale the model so its bounding box matches the item's true size; listed dimensions win.
    const clone = scene.clone(true)
    const box = new THREE.Box3().setFromObject(clone)
    const size = box.getSize(new THREE.Vector3())
    const center = box.getCenter(new THREE.Vector3())
    clone.position.set(-center.x, -box.min.y, -center.z)
    const g = new THREE.Group()
    g.add(restyle(clone, style))
    g.scale.set(item.width / (size.x || 1), item.height / (size.y || 1), item.depth / (size.z || 1))
    return g
  }, [scene, item.width, item.height, item.depth, style])
  return <primitive object={obj} />
}

/** A model URL that fails to load (404, blocked by the store, not a GLB) falls back to the box look. */
class ModelBoundary extends Component<{ fallback: ReactNode; url: string; children: ReactNode }, { failed: string | null }> {
  state = { failed: null as string | null }
  static getDerivedStateFromError() {
    return { failed: 'yes' }
  }
  componentDidCatch(e: unknown) {
    console.warn(`Couldn't load 3D model ${this.props.url}; showing a box instead.`, e)
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

export function ItemModel({ item, highlight }: { item: Item; highlight?: string }) {
  return (
    <Arrival id={item.id}>
      <ItemLook item={item} highlight={highlight} />
      {LAMPS.has(item.category) && <LampGlow item={item} />}
    </Arrival>
  )
}

// ------------------------------------------------------------------ evening lamps

/** Point lights are costly (every lit material recompiles per light), so only this many lamps cast light. */
const MAX_LAMP_LIGHTS = 8
let lampLights = 0
const GLOW = '#ffcf8a'
const SHADE_GLOW = '#ffb25e'
const SHADE_EMISSIVE = '#ffa24a'

/** In the evening a floor lamp glows: a warm bulb and shade, and a small warm light at shade height. */
function LampGlow({ item }: { item: Item }) {
  const evening = useGame((s) => s.evening)
  const [lit, setLit] = useState(false)
  useEffect(() => {
    if (!evening || lampLights >= MAX_LAMP_LIGHTS) return
    lampLights++
    setLit(true)
    return () => {
      lampLights--
      setLit(false)
    }
  }, [evening])
  if (!evening) return null
  if (item.category === 'pendant') {
    // Matches pendant(): the dome glows and a soft cone of light falls on the table below.
    const h = item.height
    const s = Math.min(item.width, item.depth)
    return (
      <group>
        <mesh position={[0, h * 0.35, 0]} renderOrder={1}>
          <cylinderGeometry args={[(s / 2) * 0.32 * 1.02, (s / 2) * 1.02, h * 0.7 * 1.01, 24, 1, true]} />
          <meshStandardMaterial color={SHADE_GLOW} emissive={SHADE_EMISSIVE} emissiveIntensity={1.2} toneMapped={false} side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[0, h * 0.25, 0]}>
          <sphereGeometry args={[0.05, 16, 12]} />
          <meshStandardMaterial color="#fff3d6" emissive={GLOW} emissiveIntensity={2} toneMapped={false} />
        </mesh>
        {lit && <PendantSpot y={h * 0.2} />}
      </group>
    )
  }
  // Matches lamp() and tableLamp(): the shade's size and the bulb's height.
  const h = item.height
  const { shadeH, s, bulbY } = shadeDims(item)
  const table = item.category === 'table-lamp'
  const top = table ? 0.78 : 0.72
  return (
    <group>
      <mesh position={[0, h - shadeH / 2, 0]} renderOrder={1}>
        <cylinderGeometry args={[(s / 2) * top * 1.02, (s / 2) * 1.02, shadeH * 1.01, 20, 1, true]} />
        <meshStandardMaterial color={SHADE_GLOW} emissive={SHADE_EMISSIVE} emissiveIntensity={1.2} toneMapped={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, bulbY, 0]}>
        <sphereGeometry args={[0.055, 16, 12]} />
        <meshStandardMaterial color="#fff3d6" emissive={GLOW} emissiveIntensity={2} toneMapped={false} />
      </mesh>
      {lit &&
        (table ? (
          <pointLight position={[0, bulbY, 0]} color="#ffb070" intensity={2.2} distance={2.2} decay={2} castShadow={false} />
        ) : (
          // A little toward the room (the shade's open side), so the wall behind isn't blown out.
          <pointLight position={[0, bulbY, 0.15]} color="#ffb070" intensity={4} distance={3.2} decay={2} castShadow={false} />
        ))}
    </group>
  )
}

/** The pendant's light: a spot straight down onto the table (its target hangs just below it). */
function PendantSpot({ y }: { y: number }) {
  const target = useMemo(() => {
    const o = new THREE.Object3D()
    o.position.set(0, y - 1, 0)
    return o
  }, [y])
  return (
    <>
      <primitive object={target} />
      <spotLight position={[0, y, 0]} target={target} color="#ffb070" intensity={6} distance={2.2} angle={0.9} penumbra={0.6} decay={2} castShadow={false} />
    </>
  )
}

function ItemLook({ item, highlight }: { item: Item; highlight?: string }) {
  if (item.glbUrl) {
    const fallback = <ProceduralItem item={item} highlight={highlight} />
    return (
      <ModelBoundary key={item.glbUrl} fallback={fallback} url={item.glbUrl}>
        <Suspense fallback={fallback}>
          <GLBItem item={item} />
        </Suspense>
      </ModelBoundary>
    )
  }
  return <ProceduralItem item={item} highlight={highlight} />
}
