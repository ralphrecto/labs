// Little "product shots" for the buy drawer and the selection bar: one flat, softly outlined
// illustration per catalog category, drawn in the same oblique projection (depth runs up and to
// the right), with light from the top left: tops are lighter, right sides darker.
import { memo, type ReactNode } from 'react'
import type { ThumbLook } from './swatches'

// ---------------------------------------------------------------- color helpers

function parse(hex: string): [number, number, number] {
  let h = hex.trim().replace('#', '')
  if (h.length === 3) h = h.replace(/./g, (c) => c + c)
  const n = parseInt(h.slice(0, 6), 16)
  if (!/^[0-9a-f]{6}/i.test(h) || Number.isNaN(n)) return [154, 164, 173]
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** Mixes two colors: t = 0 is a, t = 1 is b. */
function mix(a: string, b: string, t: number): string {
  const ca = parse(a)
  const cb = parse(b)
  return '#' + ca.map((v, i) => Math.round(v + (cb[i] - v) * t).toString(16).padStart(2, '0')).join('')
}

interface Tone {
  f: string // front face
  t: string // top face (lit)
  s: string // right side (shade)
}
const tone = (c: string, lift = 0.3, drop = 0.18): Tone => ({ f: c, t: mix(c, '#ffffff', lift), s: mix(c, '#1e140c', drop) })

// ---------------------------------------------------------------- drawing kit

const DX = 0.55
const DY = -0.4
const GROUND = 82

type Pt = [number, number]
const pts = (p: Pt[]) => p.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')

interface BoxProps {
  /** Left edge and bottom of the front face (before the depth offset z). */
  x: number
  b: number
  w: number
  h: number
  /** Depth of the box, and how far its front face sits behind the picture plane. */
  d: number
  z?: number
  c: Tone
  line: string
}

/** An oblique box: top, right side, front. */
function Box({ x, b, w, h, d, z = 0, c, line }: BoxProps) {
  const fx = x + DX * z
  const fb = b + DY * z
  const ox = DX * d
  const oy = DY * d
  return (
    <g stroke={line}>
      <polygon points={pts([[fx, fb - h], [fx + w, fb - h], [fx + w + ox, fb - h + oy], [fx + ox, fb - h + oy]])} fill={c.t} />
      <polygon points={pts([[fx + w, fb], [fx + w, fb - h], [fx + w + ox, fb - h + oy], [fx + w + ox, fb + oy]])} fill={c.s} />
      <rect x={fx} y={fb - h} width={w} height={h} fill={c.f} />
    </g>
  )
}

/** Front-face point at height above the ground and depth z. */
const at = (x: number, b: number, z = 0): Pt => [x + DX * z, b + DY * z]

interface Pal {
  base: Tone
  line: string
  wood: Tone
  white: Tone
  metal: Tone
  ink: string
}

const BOOKS = ['#ff7a59', '#5fa67f', '#4f8be8', '#ffc24b', '#8a5cc7', '#e8603f']

/** Four legs under a slab: back legs first so the front ones overlap them. */
function Legs({ x, w, b, h, d, lw = 4, c, line }: { x: number; w: number; b: number; h: number; d: number; lw?: number; c: Tone; line: string }) {
  const inset = 2
  const z0 = inset
  const z1 = d - inset - lw * 0.8
  return (
    <>
      <Box x={x + inset} b={b} w={lw} h={h} d={lw * 0.8} z={z1} c={c} line={line} />
      <Box x={x + w - inset - lw} b={b} w={lw} h={h} d={lw * 0.8} z={z1} c={c} line={line} />
      <Box x={x + inset} b={b} w={lw} h={h} d={lw * 0.8} z={z0} c={c} line={line} />
      <Box x={x + w - inset - lw} b={b} w={lw} h={h} d={lw * 0.8} z={z0} c={c} line={line} />
    </>
  )
}

function Knob({ x, y, line }: { x: number; y: number; line: string }) {
  return <circle cx={x} cy={y} r={1.6} fill={line} stroke="none" />
}

// ---------------------------------------------------------------- the pieces

type Draw = (p: Pal) => ReactNode
interface Shot {
  draw: Draw
  /** Shadow: center x and half width. */
  sx?: number
  sw?: number
}

function seating(x: number, w: number, p: Pal, cushions: number): ReactNode {
  const { base, line } = p
  const seatB = GROUND - 5
  const seatD = 22
  const arm = 9
  const cw = (w - 2 * arm) / cushions
  return (
    <>
      {/* legs */}
      <Box x={x + 3} b={GROUND} w={3} h={6} d={2} z={seatD - 3} c={p.wood} line={line} />
      <Box x={x + w - 6} b={GROUND} w={3} h={6} d={2} z={seatD - 3} c={p.wood} line={line} />
      <Box x={x + 3} b={GROUND} w={3} h={6} d={2} z={1} c={p.wood} line={line} />
      <Box x={x + w - 6} b={GROUND} w={3} h={6} d={2} z={1} c={p.wood} line={line} />
      {/* back */}
      <Box x={x} b={seatB} w={w} h={36} d={8} z={seatD - 8} c={base} line={line} />
      {Array.from({ length: cushions }, (_, i) => {
        const [bx, by] = at(x + arm + i * cw + 1.5, seatB - 15, seatD - 9)
        return <rect key={i} x={bx} y={by - 20} width={cw - 3} height={20} rx={5} fill={base.t} stroke={line} />
      })}
      {/* left arm, seat, right arm */}
      <Box x={x} b={seatB} w={arm} h={24} d={seatD} c={base} line={line} />
      <Box x={x + arm} b={seatB} w={w - 2 * arm} h={14} d={seatD - 6} c={base} line={line} />
      {Array.from({ length: cushions - 1 }, (_, i) => {
        const lx = x + arm + (i + 1) * cw
        return <path key={i} d={`M${lx} ${seatB - 14}V${seatB - 3}`} stroke={line} strokeWidth={1.6} />
      })}
      <Box x={x + w - arm} b={seatB} w={arm} h={24} d={seatD} c={base} line={line} />
      {/* a throw pillow for charm */}
      <rect x={at(x + arm + 3, seatB - 14, 8)[0]} y={at(x + arm + 3, seatB - 14, 8)[1] - 13} width={13} height={13} rx={4} transform={`rotate(-12 ${at(x + arm + 9, seatB - 14, 8)[0]} ${at(x + arm + 9, seatB - 14, 8)[1] - 6})`} fill={mix(BOOKS[3], '#ffffff', 0.25)} stroke={line} />
    </>
  )
}

const SHOTS: Record<string, Shot> = {
  sofa: { draw: (p) => seating(10, 72, p, 3) },
  armchair: { draw: (p) => seating(26, 42, p, 1), sx: 52, sw: 28 },

  'coffee-table': {
    draw: (p) => (
      <>
        <Legs x={14} w={62} b={GROUND} h={20} d={26} c={p.base} line={p.line} />
        <Box x={14} b={GROUND - 20} w={62} h={6} d={26} c={p.base} line={p.line} />
        {/* books and a mug */}
        <Box x={24} b={GROUND - 26} w={20} h={4} d={12} z={8} c={tone(BOOKS[2])} line={p.line} />
        <Box x={26} b={GROUND - 30} w={16} h={4} d={10} z={9} c={tone(BOOKS[0])} line={p.line} />
        <rect x={at(58, GROUND - 26, 10)[0]} y={at(58, GROUND - 26, 10)[1] - 8} width={7} height={8} rx={2} fill={p.white.f} stroke={p.line} />
      </>
    ),
  },
  'side-table': {
    draw: (p) => (
      <>
        <Legs x={32} w={30} b={GROUND} h={40} d={22} lw={3.5} c={p.base} line={p.line} />
        <Box x={33} b={GROUND - 10} w={28} h={3} d={20} z={1} c={p.base} line={p.line} />
        <Box x={32} b={GROUND - 40} w={30} h={5} d={22} c={p.base} line={p.line} />
        {/* tiny vase */}
        <rect x={at(42, GROUND - 45, 10)[0]} y={at(42, GROUND - 45, 10)[1] - 10} width={7} height={10} rx={3} fill={p.white.f} stroke={p.line} />
        <path d={`M${at(45.5, GROUND - 55, 10)[0]} ${at(45.5, GROUND - 55, 10)[1]}c-2-6 1-10 4-12`} stroke={p.line} fill="none" />
      </>
    ),
    sx: 54,
    sw: 22,
  },
  'console-table': {
    draw: (p) => (
      <>
        <Legs x={12} w={70} b={GROUND} h={44} d={12} lw={3.5} c={p.base} line={p.line} />
        <Box x={13} b={GROUND - 10} w={68} h={3} d={10} z={1} c={p.base} line={p.line} />
        <Box x={12} b={GROUND - 44} w={70} h={5} d={12} c={p.base} line={p.line} />
        {/* lamp and books */}
        <Box x={56} b={GROUND - 49} w={14} h={3} d={6} z={3} c={tone(BOOKS[1])} line={p.line} />
        <path d={`M${at(24, GROUND - 49, 4)[0]} ${at(24, GROUND - 49, 4)[1]}v-12`} stroke={p.line} />
        <polygon points={pts([at(17, GROUND - 61, 4), at(31, GROUND - 61, 4), at(28, GROUND - 71, 4), at(20, GROUND - 71, 4)])} fill="#fff1c9" stroke={p.line} />
      </>
    ),
  },
  ottoman: {
    draw: (p) => (
      <>
        <Box x={28} b={GROUND} w={3} h={5} d={2} z={24} c={p.wood} line={p.line} />
        <Box x={65} b={GROUND} w={3} h={5} d={2} z={24} c={p.wood} line={p.line} />
        <Box x={28} b={GROUND} w={3} h={5} d={2} z={1} c={p.wood} line={p.line} />
        <Box x={65} b={GROUND} w={3} h={5} d={2} z={1} c={p.wood} line={p.line} />
        <Box x={26} b={GROUND - 4} w={44} h={28} d={28} c={p.base} line={p.line} />
        {[0, 1].map((r) =>
          [0, 1, 2].map((k) => {
            const [tx, ty] = at(34 + k * 14, GROUND - 32, 8 + r * 12)
            return <circle key={`${r}${k}`} cx={tx} cy={ty} r={1.3} fill={p.line} stroke="none" />
          }),
        )}
        <path d={`M28 ${GROUND - 11}h40`} stroke={p.line} strokeWidth={1.4} strokeDasharray="2 3" />
      </>
    ),
    sx: 54,
    sw: 28,
  },
  'tv-stand': {
    draw: (p) => (
      <>
        <Box x={14} b={GROUND} w={3} h={6} d={2} z={13} c={p.wood} line={p.line} />
        <Box x={77} b={GROUND} w={3} h={6} d={2} z={13} c={p.wood} line={p.line} />
        <Box x={14} b={GROUND} w={3} h={6} d={2} z={1} c={p.wood} line={p.line} />
        <Box x={77} b={GROUND} w={3} h={6} d={2} z={1} c={p.wood} line={p.line} />
        {/* TV */}
        <Box x={41} b={GROUND - 24} w={14} h={3} d={8} z={4} c={p.metal} line={p.line} />
        <Box x={22} b={GROUND - 27} w={52} h={32} d={3} z={7} c={tone('#2f2b28', 0.25)} line={p.line} />
        <polygon points={pts([at(26, GROUND - 31, 7), at(36, GROUND - 31, 7), at(52, GROUND - 55, 7), at(42, GROUND - 55, 7)])} fill="#ffffff" opacity={0.14} stroke="none" />
        <Box x={12} b={GROUND - 5} w={70} h={20} d={16} c={p.base} line={p.line} />
        <path d={`M35.3 ${GROUND - 25}v20M58.6 ${GROUND - 25}v20`} stroke={p.line} strokeWidth={1.6} />
        <Knob x={31} y={GROUND - 15} line={p.line} />
        <Knob x={39.6} y={GROUND - 15} line={p.line} />
        <Knob x={63} y={GROUND - 15} line={p.line} />
      </>
    ),
  },

  bed: {
    draw: (p) => {
      const frame = p.wood
      const x = 8
      const w = 60
      const d = 40
      return (
        <>
          {/* headboard with a soft panel */}
          <Box x={x} b={GROUND} w={w} h={46} d={4} z={d - 4} c={frame} line={p.line} />
          <rect x={at(x + 5, GROUND - 22, d - 4)[0]} y={at(x + 5, GROUND - 22, d - 4)[1] - 20} width={w - 10} height={20} rx={5} fill={frame.t} stroke={p.line} strokeWidth={1.6} />
          {/* frame and mattress */}
          <Box x={x} b={GROUND} w={w} h={10} d={d - 4} c={frame} line={p.line} />
          <Box x={x + 1.5} b={GROUND - 10} w={w - 3} h={7} d={d - 6} z={0.5} c={p.white} line={p.line} />
          {/* pillows */}
          <rect x={at(x + 6, GROUND - 17, d - 11)[0]} y={at(x + 6, GROUND - 17, d - 11)[1] - 8} width={21} height={9} rx={4} fill={p.white.t} stroke={p.line} />
          <rect x={at(x + 32, GROUND - 17, d - 11)[0]} y={at(x + 32, GROUND - 17, d - 11)[1] - 8} width={21} height={9} rx={4} fill={p.white.t} stroke={p.line} />
          {/* duvet draped over the front, with the sheet folded back over it */}
          <Box x={x - 1} b={GROUND - 4} w={w + 2} h={16} d={24} c={p.base} line={p.line} />
          <polygon points={pts([at(x - 1, GROUND - 20, 18), at(x + w + 1, GROUND - 20, 18), at(x + w + 1, GROUND - 20, 24), at(x - 1, GROUND - 20, 24)])} fill={p.white.f} stroke={p.line} />
          <path d={`M${x + 4} ${GROUND - 12}h${w - 8}`} stroke={p.base.s} strokeWidth={1.4} strokeDasharray="3 3" />
        </>
      )
    },
    sx: 50,
    sw: 40,
  },
  nightstand: {
    draw: (p) => (
      <>
        <Box x={32} b={GROUND} w={3} h={8} d={2} z={16} c={p.base} line={p.line} />
        <Box x={61} b={GROUND} w={3} h={8} d={2} z={16} c={p.base} line={p.line} />
        <Box x={32} b={GROUND} w={3} h={8} d={2} z={1} c={p.base} line={p.line} />
        <Box x={61} b={GROUND} w={3} h={8} d={2} z={1} c={p.base} line={p.line} />
        <Box x={30} b={GROUND - 7} w={36} h={30} d={20} c={p.base} line={p.line} />
        <path d={`M30 ${GROUND - 22}h36`} stroke={p.line} strokeWidth={1.6} />
        <Knob x={48} y={GROUND - 29.5} line={p.line} />
        <Knob x={48} y={GROUND - 14.5} line={p.line} />
        {/* lamp */}
        <path d={`M${at(44, GROUND - 37, 9)[0]} ${at(44, GROUND - 37, 9)[1]}v-10`} stroke={p.line} />
        <rect x={at(39, GROUND - 37, 9)[0]} y={at(39, GROUND - 37, 9)[1] - 3} width={10} height={3} rx={1.5} fill={p.metal.f} stroke={p.line} />
        <polygon points={pts([at(36, GROUND - 47, 9), at(52, GROUND - 47, 9), at(49, GROUND - 58, 9), at(39, GROUND - 58, 9)])} fill="#fff1c9" stroke={p.line} />
      </>
    ),
    sx: 54,
    sw: 26,
  },
  dresser: {
    draw: (p) => (
      <>
        <Box x={16} b={GROUND} w={3} h={6} d={2} z={16} c={p.base} line={p.line} />
        <Box x={73} b={GROUND} w={3} h={6} d={2} z={16} c={p.base} line={p.line} />
        <Box x={16} b={GROUND} w={3} h={6} d={2} z={1} c={p.base} line={p.line} />
        <Box x={73} b={GROUND} w={3} h={6} d={2} z={1} c={p.base} line={p.line} />
        <Box x={14} b={GROUND - 5} w={64} h={42} d={20} c={p.base} line={p.line} />
        <path d={`M14 ${GROUND - 19}h64M14 ${GROUND - 33}h64M46 ${GROUND - 19}v14M46 ${GROUND - 33}v14M46 ${GROUND - 47}v14`} stroke={p.line} strokeWidth={1.6} />
        {[0, 1, 2].map((r) => (
          <g key={r}>
            <rect x={26} y={GROUND - 13.5 - r * 14} width={8} height={2.6} rx={1.3} fill={p.line} stroke="none" />
            <rect x={58} y={GROUND - 13.5 - r * 14} width={8} height={2.6} rx={1.3} fill={p.line} stroke="none" />
          </g>
        ))}
      </>
    ),
  },
  sideboard: {
    draw: (p) => (
      <>
        <Box x={13} b={GROUND} w={3} h={12} d={2} z={14} c={p.wood} line={p.line} />
        <Box x={76} b={GROUND} w={3} h={12} d={2} z={14} c={p.wood} line={p.line} />
        <Box x={13} b={GROUND} w={3} h={12} d={2} z={1} c={p.wood} line={p.line} />
        <Box x={76} b={GROUND} w={3} h={12} d={2} z={1} c={p.wood} line={p.line} />
        <Box x={10} b={GROUND - 11} w={72} h={30} d={18} c={p.base} line={p.line} />
        <path d={`M28 ${GROUND - 41}v30M46 ${GROUND - 41}v30M64 ${GROUND - 41}v30`} stroke={p.line} strokeWidth={1.6} />
        {[24.5, 31.5, 60.5, 67.5].map((x) => (
          <rect key={x} x={x - 0.6} y={GROUND - 30} width={1.6} height={7} rx={0.8} fill={p.line} stroke="none" />
        ))}
        {/* vase with a twig */}
        <rect x={at(62, GROUND - 41, 8)[0]} y={at(62, GROUND - 41, 8)[1] - 11} width={8} height={11} rx={3.5} fill={tone(BOOKS[3]).f} stroke={p.line} />
        <path d={`M${at(66, GROUND - 52, 8)[0]} ${at(66, GROUND - 52, 8)[1]}c0-6 3-9 6-11M${at(66, GROUND - 52, 8)[0]} ${at(66, GROUND - 52, 8)[1] - 4}c-2-3-4-4-6-5`} stroke={p.line} fill="none" />
      </>
    ),
  },
  wardrobe: {
    draw: (p) => (
      <>
        <Box x={26} b={GROUND} w={42} h={70} d={18} c={p.base} line={p.line} />
        <Box x={24} b={GROUND - 70} w={46} h={4} d={20} z={-1} c={p.base} line={p.line} />
        <path d={`M47 ${GROUND - 68}v64M26 ${GROUND - 6}h42`} stroke={p.line} strokeWidth={1.6} />
        <rect x={43} y={GROUND - 42} width={1.8} height={10} rx={0.9} fill={p.line} stroke="none" />
        <rect x={49.2} y={GROUND - 42} width={1.8} height={10} rx={0.9} fill={p.line} stroke="none" />
        <rect x={29} y={GROUND - 64} width={15} height={8} rx={2} fill={p.base.t} stroke="none" opacity={0.6} />
      </>
    ),
    sx: 54,
    sw: 28,
  },

  'dining-table': {
    draw: (p) => (
      <>
        <Legs x={10} w={70} b={GROUND} h={38} d={30} c={p.base} line={p.line} />
        <Box x={10} b={GROUND - 38} w={70} h={5} d={30} c={p.base} line={p.line} />
        {/* plates */}
        {[24, 52].map((x) => (
          <ellipse key={x} cx={at(x, GROUND - 43, 12)[0]} cy={at(x, GROUND - 43, 12)[1]} rx={7} ry={2.6} fill={p.white.f} stroke={p.line} />
        ))}
        <rect x={at(39, GROUND - 43, 18)[0]} y={at(39, GROUND - 43, 18)[1] - 9} width={5} height={9} rx={2} fill={tone(BOOKS[1]).f} stroke={p.line} />
      </>
    ),
  },
  'dining-chair': {
    draw: (p) => (
      <>
        <Box x={33} b={GROUND} w={3.5} h={70} d={3} z={20} c={p.base} line={p.line} />
        <Box x={60} b={GROUND} w={3.5} h={70} d={3} z={20} c={p.base} line={p.line} />
        <Box x={36.5} b={GROUND - 62} w={23.5} h={8} d={2.5} z={20.2} c={p.base} line={p.line} />
        <Box x={36.5} b={GROUND - 46} w={23.5} h={4} d={2.5} z={20.2} c={p.base} line={p.line} />
        <Box x={33} b={GROUND} w={3.5} h={32} d={3} z={1} c={p.base} line={p.line} />
        <Box x={60} b={GROUND} w={3.5} h={32} d={3} z={1} c={p.base} line={p.line} />
        <Box x={32} b={GROUND - 32} w={32} h={5} d={24} c={p.base} line={p.line} />
      </>
    ),
    sx: 54,
    sw: 22,
  },
  bench: {
    draw: (p) => (
      <>
        <Legs x={12} w={70} b={GROUND} h={24} d={18} lw={4} c={p.base} line={p.line} />
        <Box x={12} b={GROUND - 24} w={70} h={6} d={18} c={p.base} line={p.line} />
        <rect x={at(20, GROUND - 30, 4)[0]} y={at(20, GROUND - 30, 4)[1] - 7} width={22} height={7} rx={3.5} fill={p.white.f} stroke={p.line} />
      </>
    ),
  },

  desk: {
    draw: (p) => (
      <>
        {/* monitor behind */}
        <Box x={52} b={GROUND - 41} w={10} h={3} d={8} z={18} c={p.metal} line={p.line} />
        <path d={`M${at(57, GROUND - 44, 22)[0]} ${at(57, GROUND - 44, 22)[1]}v-6`} stroke={p.line} strokeWidth={2.5} />
        <Box x={40} b={GROUND - 48} w={34} h={20} d={2} z={22} c={tone('#2f2b28', 0.25)} line={p.line} />
        <rect x={at(43, GROUND - 51, 22)[0]} y={at(43, GROUND - 51, 22)[1] - 14} width={28} height={14} rx={1.5} fill={mix('#4f8be8', '#ffffff', 0.35)} stroke="none" />
        {/* legs on the right, drawer pedestal on the left */}
        <Box x={74} b={GROUND} w={4} h={36} d={3} z={24} c={p.base} line={p.line} />
        <Box x={74} b={GROUND} w={4} h={36} d={3} z={2} c={p.base} line={p.line} />
        <Box x={12} b={GROUND} w={22} h={36} d={26} z={1} c={p.base} line={p.line} />
        <path d={`M12 ${GROUND - 24}h22M12 ${GROUND - 12}h22`} stroke={p.line} strokeWidth={1.6} />
        <Knob x={23} y={GROUND - 30} line={p.line} />
        <Knob x={23} y={GROUND - 18} line={p.line} />
        <Knob x={23} y={GROUND - 6} line={p.line} />
        <Box x={10} b={GROUND - 36} w={70} h={5} d={30} c={p.base} line={p.line} />
        {/* mug */}
        <rect x={at(24, GROUND - 41, 14)[0]} y={at(24, GROUND - 41, 14)[1] - 7} width={6} height={7} rx={2} fill={tone(BOOKS[0]).f} stroke={p.line} />
      </>
    ),
  },
  'office-chair': {
    draw: (p) => {
      const cx = 50
      return (
        <>
          {/* star base and wheels */}
          <path d={`M${cx} ${GROUND - 8}L${cx - 20} ${GROUND - 3}M${cx} ${GROUND - 8}L${cx + 20} ${GROUND - 3}M${cx} ${GROUND - 8}L${cx - 8} ${GROUND - 12}M${cx} ${GROUND - 8}L${cx + 10} ${GROUND - 12}M${cx} ${GROUND - 8}L${cx + 1} ${GROUND - 1}`} stroke={p.line} strokeWidth={4.5} />
          <path d={`M${cx} ${GROUND - 8}L${cx - 20} ${GROUND - 3}M${cx} ${GROUND - 8}L${cx + 20} ${GROUND - 3}M${cx} ${GROUND - 8}L${cx - 8} ${GROUND - 12}M${cx} ${GROUND - 8}L${cx + 10} ${GROUND - 12}M${cx} ${GROUND - 8}L${cx + 1} ${GROUND - 1}`} stroke={p.metal.t} strokeWidth={2} />
          {[
            [cx - 20, GROUND - 1],
            [cx + 20, GROUND - 1],
            [cx + 1, GROUND + 1],
          ].map(([x, y]) => (
            <circle key={x} cx={x} cy={y} r={2.6} fill={p.metal.s} stroke={p.line} />
          ))}
          <rect x={cx - 2} y={GROUND - 28} width={4} height={20} fill={p.metal.f} stroke={p.line} />
          {/* back with a stem */}
          <rect x={cx + 3} y={GROUND - 44} width={4} height={16} fill={p.metal.f} stroke={p.line} />
          <rect x={cx - 15} y={GROUND - 76} width={34} height={38} rx={11} fill={p.base.f} stroke={p.line} />
          <rect x={cx - 11} y={GROUND - 72} width={10} height={20} rx={5} fill={p.base.t} stroke="none" opacity={0.7} />
          {/* armrests and seat */}
          <path d={`M${cx - 20} ${GROUND - 30}v-10h6M${cx + 20} ${GROUND - 30}v-10h-6`} stroke={p.line} strokeWidth={4} fill="none" />
          <path d={`M${cx - 20} ${GROUND - 30}v-10h6M${cx + 20} ${GROUND - 30}v-10h-6`} stroke={p.metal.t} strokeWidth={1.6} fill="none" />
          <rect x={cx - 21} y={GROUND - 34} width={42} height={10} rx={5} fill={p.base.f} stroke={p.line} />
          <rect x={cx - 17} y={GROUND - 33} width={22} height={3} rx={1.5} fill={p.base.t} stroke="none" opacity={0.8} />
        </>
      )
    },
    sx: 50,
    sw: 26,
  },
  bookshelf: {
    draw: (p) => {
      const x = 26
      const w = 44
      const shelves = [GROUND - 6, GROUND - 24, GROUND - 42, GROUND - 60]
      return (
        <>
          <Box x={x} b={GROUND} w={w} h={72} d={16} c={p.base} line={p.line} />
          <rect x={x + 4} y={GROUND - 68} width={w - 8} height={62} fill={p.base.s} stroke={p.line} />
          {shelves.map((sb, i) => (
            <g key={sb}>
              <rect x={x + 4} y={sb - 1} width={w - 8} height={3} fill={p.base.t} stroke={p.line} strokeWidth={1.4} />
              {(i === 0 ? [7, 6, 8, 5] : i === 1 ? [5, 7, 6] : i === 2 ? [6, 8, 6, 7] : [8, 5]).map((bw, k, arr) => {
                const off = arr.slice(0, k).reduce((a, b) => a + b, 0)
                const bh = 11 + ((i * 3 + k * 5) % 5)
                const lean = i === 1 && k === arr.length - 1
                return (
                  <rect
                    key={k}
                    x={x + 6 + off}
                    y={sb - 1 - bh}
                    width={bw}
                    height={bh}
                    rx={1}
                    fill={BOOKS[(i * 2 + k) % BOOKS.length]}
                    stroke={p.line}
                    strokeWidth={1.4}
                    transform={lean ? `rotate(14 ${x + 6 + off} ${sb - 1})` : undefined}
                  />
                )
              })}
            </g>
          ))}
          {/* a little plant on the top shelf */}
          <rect x={x + w - 15} y={GROUND - 67} width={7} height={6} rx={1.5} fill="#d98a5f" stroke={p.line} strokeWidth={1.4} />
          <path d={`M${x + w - 11.5} ${GROUND - 67}c-3-3-4-5-3-7M${x + w - 11.5} ${GROUND - 67}c2-3 4-4 6-4`} stroke="#5f8f5a" strokeWidth={2} fill="none" />
        </>
      )
    },
    sx: 54,
    sw: 28,
  },

  rug: {
    draw: (p) => {
      const x = 5
      const w = 64
      const d = 50
      const b = GROUND - 4
      const q = (u: number, v: number): Pt => at(x + u, b, v)
      return (
        <>
          <polygon points={pts([q(0, 0), q(w, 0), q(w, d), q(w, d), [q(w, d)[0], q(w, d)[1] + 3], [q(w, 0)[0], q(w, 0)[1] + 3], [q(0, 0)[0], q(0, 0)[1] + 3]])} fill={p.base.s} stroke={p.line} />
          <polygon points={pts([q(0, 0), q(w, 0), q(w, d), q(0, d)])} fill={p.base.f} stroke={p.line} />
          <polygon points={pts([q(6, 5), q(w - 6, 5), q(w - 6, d - 5), q(6, d - 5)])} fill="none" stroke={p.base.t} strokeWidth={3} />
          <polygon points={pts([q(w / 2 - 14, d / 2), q(w / 2, d / 2 - 14), q(w / 2 + 14, d / 2), q(w / 2, d / 2 + 14)])} fill={p.base.t} stroke={p.line} strokeWidth={1.4} />
          <path d={Array.from({ length: 12 }, (_, i) => `M${x + 2.5 + i * 5.5} ${b + 3}v3.5`).join('')} stroke={p.line} strokeWidth={1.4} />
          <path d={Array.from({ length: 12 }, (_, i) => `M${at(x + 2.5 + i * 5.5, b, d)[0]} ${at(x + 2.5 + i * 5.5, b, d)[1]}v-3.5`).join('')} stroke={p.line} strokeWidth={1.4} />
        </>
      )
    },
    sx: 52,
    sw: 44,
  },
  'floor-lamp': {
    draw: (p) => (
      <>
        <circle cx={50} cy={22} r={22} fill="#ffe7a3" opacity={0.35} stroke="none" />
        <ellipse cx={50} cy={GROUND - 1} rx={12} ry={3.5} fill={p.metal.f} stroke={p.line} />
        <path d={`M50 ${GROUND - 2}V30`} stroke={p.line} strokeWidth={4.5} />
        <path d={`M50 ${GROUND - 2}V30`} stroke={p.metal.t} strokeWidth={2} />
        <polygon points={pts([[36, 36], [64, 36], [58, 12], [42, 12]])} fill={p.base.f} stroke={p.line} />
        <polygon points={pts([[40, 34], [46, 34], [46, 14], [43, 14]])} fill={p.base.t} stroke="none" opacity={0.8} />
        <ellipse cx={50} cy={36} rx={14} ry={2.6} fill={p.base.s} stroke={p.line} />
      </>
    ),
    sx: 50,
    sw: 16,
  },
  'table-lamp': {
    draw: (p) => (
      <>
        <circle cx={50} cy={46} r={20} fill="#ffe7a3" opacity={0.35} stroke="none" />
        <ellipse cx={50} cy={GROUND - 1} rx={9} ry={2.8} fill={p.base.s} stroke={p.line} />
        <path d={`M41 ${GROUND - 2}Q38 ${GROUND - 16} 46 ${GROUND - 26}H54Q62 ${GROUND - 16} 59 ${GROUND - 2}Z`} fill={p.base.f} stroke={p.line} />
        <path d={`M50 ${GROUND - 26}V54`} stroke={p.metal.t} strokeWidth={2} />
        <polygon points={pts([[30, 56], [70, 56], [62, 34], [38, 34]])} fill={p.base.t} stroke={p.line} />
        <ellipse cx={50} cy={56} rx={20} ry={3} fill={p.base.s} stroke={p.line} />
      </>
    ),
    sx: 50,
    sw: 20,
  },
  pendant: {
    draw: (p) => (
      <>
        <polygon points={pts([[30, 54], [70, 54], [80, GROUND - 2], [20, GROUND - 2]])} fill="#ffe7a3" opacity={0.22} stroke="none" />
        <ellipse cx={50} cy={GROUND - 2} rx={28} ry={5} fill="#ffe7a3" opacity={0.45} stroke="none" />
        <rect x={42} y={6} width={16} height={4} rx={2} fill={p.metal.t} stroke={p.line} />
        <path d="M50 10V29" stroke={p.line} strokeWidth={1.6} />
        <path d="M45 29h10l2 6H43z" fill={p.base.s} stroke={p.line} />
        <path d="M26 53Q28 35 50 34Q72 35 74 53Z" fill={p.base.f} stroke={p.line} />
        <path d="M33 48Q36 39 46 37" stroke={p.base.t} strokeWidth={3} fill="none" />
        <ellipse cx={50} cy={53} rx={24} ry={3.5} fill={p.base.s} stroke={p.line} />
        <circle cx={50} cy={55.5} r={4.5} fill="#fff3d6" stroke={p.line} strokeWidth={1.4} />
      </>
    ),
    sx: 50,
    sw: 14,
  },
  plant: {
    draw: (p) => {
      const leaf = (cx: number, cy: number, rot: number, s = 1, lit = false) => (
        <ellipse cx={cx} cy={cy} rx={6 * s} ry={15 * s} transform={`rotate(${rot} ${cx} ${cy})`} fill={lit ? p.base.t : p.base.f} stroke={p.line} />
      )
      return (
        <>
          {leaf(36, 40, -48, 1)}
          {leaf(64, 38, 50, 1)}
          {leaf(42, 26, -22, 1.1, true)}
          {leaf(58, 24, 24, 1.1)}
          {leaf(50, 18, 2, 1.2, true)}
          {leaf(32, 52, -70, 0.85)}
          {leaf(68, 52, 72, 0.85, true)}
          <path d="M50 52V30M50 52l-10-14M50 52l10-14" stroke={p.base.s} strokeWidth={1.6} />
          <polygon points={pts([[34, 52], [66, 52], [61, GROUND], [39, GROUND]])} fill="#d98a5f" stroke={p.line} />
          <rect x={32} y={50} width={36} height={7} rx={2} fill="#e8a27c" stroke={p.line} />
        </>
      )
    },
    sx: 50,
    sw: 22,
  },
}

// Wall decor: drawn face-on (it hangs flat on a wall), with a soft wall-shadow behind and no floor shadow.
const wallShadow = (x: number, y: number, w: number, h: number, r = 3) => <rect x={x + 2.5} y={y + 3} width={w} height={h} rx={r} fill="rgba(70, 45, 20, 0.14)" stroke="none" />
Object.assign(SHOTS, {
  'wall-art': {
    draw: (p: Pal) => (
      <>
        {wallShadow(26, 14, 48, 64)}
        <rect x={26} y={14} width={48} height={64} rx={2} fill={p.wood.t} stroke={p.line} />
        <rect x={32} y={20} width={36} height={52} fill={mix(p.base.f, '#ffffff', 0.55)} stroke={p.line} strokeWidth={1.6} />
        <circle cx={58} cy={33} r={6} fill="#ffc24b" stroke="none" />
        <path d="M32 72V58Q41 46 50 56T68 50V72Z" fill={p.base.f} stroke="none" />
        <path d="M32 72V64Q44 58 54 65T68 62V72Z" fill={p.base.s} stroke="none" />
        <rect x={32} y={20} width={36} height={52} fill="none" stroke={p.line} strokeWidth={1.6} />
      </>
    ),
    sx: 50,
    sw: 0,
  },
  mirror: {
    draw: (p: Pal) => (
      <>
        <circle cx={52.5} cy={49} r={30} fill="rgba(70, 45, 20, 0.14)" stroke="none" />
        <circle cx={50} cy={46} r={30} fill={p.base.t} stroke={p.line} />
        <circle cx={50} cy={46} r={24} fill="#dceef4" stroke={p.line} strokeWidth={1.6} />
        <path d="M36 40Q40 28 52 25" stroke="#ffffff" strokeWidth={4} fill="none" />
        <path d="M58 60L64 54" stroke="#ffffff" strokeWidth={2.4} fill="none" />
      </>
    ),
    sx: 50,
    sw: 0,
  },
  'wall-shelf': {
    draw: (p: Pal) => (
      <>
        {[0, 1, 2, 3].map((i) => {
          const h = [26, 22, 28, 18][i]
          const x = 24 + i * 9
          return i === 3 ? (
            <rect key={i} x={x + 2} y={60 - h} width={7} height={h} rx={1} transform={`rotate(14 ${x + 9} 60)`} fill={tone(BOOKS[i]).f} stroke={p.line} />
          ) : (
            <rect key={i} x={x} y={60 - h} width={8} height={h} rx={1} fill={tone(BOOKS[i]).f} stroke={p.line} />
          )
        })}
        <ellipse cx={68} cy={55} rx={6} ry={5} fill="#5f8f5a" stroke={p.line} />
        <rect x={64} y={54} width={8} height={6} rx={1} fill="#d98a5f" stroke={p.line} />
        <Box x={18} b={66} w={64} h={6} d={10} c={p.base} line={p.line} />
        <rect x={22} y={67} width={60} height={3} fill="rgba(70, 45, 20, 0.14)" stroke="none" />
      </>
    ),
    sx: 50,
    sw: 0,
  },
  'wall-clock': {
    draw: (p: Pal) => (
      <>
        <circle cx={52.5} cy={49} r={26} fill="rgba(70, 45, 20, 0.14)" stroke="none" />
        <circle cx={50} cy={46} r={26} fill={p.wood.f} stroke={p.line} />
        <circle cx={50} cy={46} r={21} fill={p.base.t} stroke={p.line} strokeWidth={1.6} />
        {[0, 1, 2, 3].map((i) => (
          <circle key={i} cx={50 + 16 * Math.sin((i * Math.PI) / 2)} cy={46 - 16 * Math.cos((i * Math.PI) / 2)} r={1.8} fill={p.ink} stroke="none" />
        ))}
        <path d="M50 46V32M50 46l9 5" stroke={p.ink} strokeWidth={2.4} />
        <circle cx={50} cy={46} r={2.2} fill="#ff7a59" stroke="none" />
      </>
    ),
    sx: 50,
    sw: 0,
  },
  curtains: {
    draw: (p: Pal) => (
      <>
        <rect x={34} y={18} width={32} height={42} fill="#dceef4" stroke={p.line} strokeWidth={1.6} />
        <path d="M50 18V60M34 39H66" stroke={p.line} strokeWidth={1.4} />
        <path d="M16 12H84" stroke={p.metal.s} strokeWidth={3} />
        {[-1, 1].map((k) => {
          const x0 = k < 0 ? 18 : 82
          const x1 = k < 0 ? 40 : 60
          return (
            <g key={k}>
              <path d={`M${x0} 13H${x1}Q${x1 - k * 2} 40 ${x1 + k * 4} 46Q${x1 - k * 6} 64 ${x1 - k * 2} ${GROUND}H${x0}Z`} fill={p.base.f} stroke={p.line} />
              <path d={`M${(x0 + x1) / 2} 15V${GROUND - 2}M${x0 + (x1 - x0) * 0.75} 16Q${x0 + (x1 - x0) * 0.7} 60 ${x0 + (x1 - x0) * 0.8} ${GROUND - 2}`} stroke={p.base.s} strokeWidth={1.8} fill="none" />
            </g>
          )
        })}
      </>
    ),
    sx: 50,
    sw: 0,
  },
} satisfies Record<string, Shot>)

/** Variant shots ("category:look"): a round dining or coffee table on a pedestal, a tall chest of drawers. */
const roundTable = (topY: number, rx: number, plates: boolean): Draw => (p) => {
  const cx = 50
  return (
    <>
      <ellipse cx={cx} cy={GROUND - 1} rx={rx * 0.42} ry={4} fill={p.base.s} stroke={p.line} />
      <Box x={cx - 4} b={GROUND - 2} w={8} h={GROUND - 2 - topY} d={4} z={-2} c={p.base} line={p.line} />
      <ellipse cx={cx} cy={topY + 2.5} rx={rx} ry={rx * 0.34} fill={p.base.s} stroke={p.line} />
      <ellipse cx={cx} cy={topY} rx={rx} ry={rx * 0.34} fill={p.base.t} stroke={p.line} />
      {plates &&
        [-1, 1].map((k) => <ellipse key={k} cx={cx + k * rx * 0.48} cy={topY} rx={6} ry={2.2} fill={p.white.f} stroke={p.line} />)}
      {plates && <rect x={cx - 2.5} y={topY - 9} width={5} height={9} rx={2} fill={tone(BOOKS[1]).f} stroke={p.line} />}
    </>
  )
}
/** A framed print drawn face-on in any box: frame, mat, a sun and two soft hills (tall and wide variants). */
const framedArt = (x: number, y: number, w: number, h: number): Draw => (p) => {
  const m = 6
  const [ix, iy, iw, ih] = [x + m, y + m, w - 2 * m, h - 2 * m]
  const b = iy + ih
  return (
    <>
      {wallShadow(x, y, w, h)}
      <rect x={x} y={y} width={w} height={h} rx={2} fill={p.wood.t} stroke={p.line} />
      <rect x={ix} y={iy} width={iw} height={ih} fill={mix(p.base.f, '#ffffff', 0.55)} stroke="none" />
      <circle cx={ix + iw * 0.72} cy={iy + ih * 0.25} r={Math.min(iw, ih) * 0.16} fill="#ffc24b" stroke="none" />
      <path d={`M${ix} ${b}V${b - ih * 0.3}Q${ix + iw * 0.25} ${b - ih * 0.55} ${ix + iw * 0.5} ${b - ih * 0.32}T${ix + iw} ${b - ih * 0.4}V${b}Z`} fill={p.base.f} stroke="none" />
      <path d={`M${ix} ${b}V${b - ih * 0.15}Q${ix + iw * 0.35} ${b - ih * 0.3} ${ix + iw * 0.6} ${b - ih * 0.14}T${ix + iw} ${b - ih * 0.18}V${b}Z`} fill={p.base.s} stroke="none" />
      <rect x={ix} y={iy} width={iw} height={ih} fill="none" stroke={p.line} strokeWidth={1.6} />
    </>
  )
}
const VARIANTS: Record<string, Shot> = {
  'wall-art:tall': { draw: framedArt(33, 6, 34, 80), sx: 50, sw: 0 },
  'wall-art:wide': { draw: framedArt(12, 24, 76, 52), sx: 50, sw: 0 },
  'dining-table:round': { draw: roundTable(GROUND - 44, 34, true), sx: 50, sw: 32 },
  'coffee-table:round': { draw: roundTable(GROUND - 26, 30, false), sx: 50, sw: 30 },
  'dresser:tall': {
    draw: (p) => (
      <>
        <Box x={30} b={GROUND} w={3} h={6} d={2} z={16} c={p.base} line={p.line} />
        <Box x={59} b={GROUND} w={3} h={6} d={2} z={16} c={p.base} line={p.line} />
        <Box x={30} b={GROUND} w={3} h={6} d={2} z={1} c={p.base} line={p.line} />
        <Box x={59} b={GROUND} w={3} h={6} d={2} z={1} c={p.base} line={p.line} />
        <Box x={28} b={GROUND - 5} w={36} h={66} d={20} c={p.base} line={p.line} />
        <path d={`M28 ${GROUND - 21.5}h36M28 ${GROUND - 38}h36M28 ${GROUND - 54.5}h36`} stroke={p.line} strokeWidth={1.6} />
        {[0, 1, 2, 3].map((r) => (
          <rect key={r} x={42} y={GROUND - 14.5 - r * 16.5} width={8} height={2.6} rx={1.3} fill={p.line} stroke="none" />
        ))}
      </>
    ),
    sx: 54,
    sw: 26,
  },
}

function boxShot(p: Pal) {
  return (
    <>
      <Box x={22} b={GROUND} w={46} h={42} d={26} c={p.base} line={p.line} />
      <polygon points={pts([at(41, GROUND - 42), at(49, GROUND - 42), at(49, GROUND - 42, 26), at(41, GROUND - 42, 26)])} fill={p.base.t} stroke="none" />
      <path d={`M${at(41, GROUND - 42)[0]} ${GROUND - 42}v12h8v-12`} fill={mix(p.base.f, '#ffffff', 0.2)} stroke={p.line} strokeWidth={1.4} />
      <path d={`M${at(45, GROUND - 42, 26)[0]} ${at(45, GROUND - 42, 26)[1]}L${at(45, GROUND - 42)[0]} ${GROUND - 42}`} stroke={p.line} strokeWidth={1.4} strokeDasharray="2 2" />
    </>
  )
}

/** Base color with a gentle floor on darkness so very dark items still read. */
function palette(color: string): Pal {
  const c = parse(color)
  const lum = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255
  const base = lum < 0.22 ? mix(color, '#ffffff', 0.12) : color
  return {
    base: tone(base),
    line: mix(base, '#2d2620', 0.72),
    wood: tone('#8a6b4f', 0.25),
    white: tone('#fbf6ee', 0.4, 0.1),
    metal: tone('#9aa0a6', 0.35),
    ink: '#2d2620',
  }
}

export const ItemThumb = memo(function ItemThumb({ category, color, size = 64, look }: { category: string; color: string; size?: number; look?: ThumbLook }) {
  const variant = look?.round ? VARIANTS[`${category}:round`] : look?.tall ? VARIANTS[`${category}:tall`] : look?.wide ? VARIANTS[`${category}:wide`] : undefined
  const shot = variant ?? SHOTS[category]
  const k = look?.scale && look.scale !== 1 ? look.scale : 1
  const p = palette(color || '#9aa4ad')
  const sx = shot?.sx ?? 52
  const sw = shot?.sw ?? 40
  return (
    <svg className="item-thumb" viewBox="0 0 100 100" width={size} height={size} aria-hidden strokeWidth={2.2} strokeLinejoin="round" strokeLinecap="round">
      <ellipse cx={sx} cy={GROUND + 2} rx={sw * k} ry={5.5} fill="rgba(70, 45, 20, 0.16)" />
      {k === 1 ? (
        shot ? shot.draw(p) : boxShot(p)
      ) : (
        // Smaller variants shrink toward the middle of their footprint, feet on the ground.
        <g transform={`translate(${sx} ${GROUND}) scale(${k}) translate(${-sx} ${-GROUND})`} strokeWidth={2.2 / k}>
          {shot ? shot.draw(p) : boxShot(p)}
        </g>
      )}
    </svg>
  )
})
