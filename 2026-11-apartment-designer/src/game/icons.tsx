// Tiny inline icon set (24px grid, stroke = currentColor) so the game needs no icon library.
import type { SVGProps } from 'react'

const P = (d: string) =>
  function Icon(props: SVGProps<SVGSVGElement>) {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" aria-hidden {...props}>
        <path d={d} />
      </svg>
    )
  }

export const IconUndo = P('M9 14 4 9l5-5M4 9h10.5a5.5 5.5 0 0 1 0 11H11')
export const IconRedo = P('m15 14 5-5-5-5M20 9H9.5a5.5 5.5 0 0 0 0 11H13')
export const IconGear = P('M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z')
export const IconRotate = P('M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5')
export const IconCopy = P('M8 8h12v12H8zM4 16V4h12')
export const IconTrash = P('M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14')
export const IconClose = P('M6 6l12 12M18 6 6 18')
export const IconMore = P('M5 12h.01M12 12h.01M19 12h.01')
export const IconSofa = P('M4 11V8a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v3M2 13a2 2 0 0 1 4 0v2h12v-2a2 2 0 0 1 4 0v5H2zM5 18v2M19 18v2')
export const IconHammer = P('m14 7 3-3 3 3-3 3M15.5 8.5 5 19a1.4 1.4 0 0 1-2-2L13.5 6.5M11 5l3 3')
export const IconEye = P('M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12ZM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z')
export const IconCube = P('M12 2 3 7v10l9 5 9-5V7zM3 7l9 5 9-5M12 12v10')
export const IconMap = P('M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3zM9 3v15M15 6v15')
export const IconStar = P('m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z')
export const IconRoom = P('M4 4h16v16H4zM4 12h7M11 12v8')
export const IconPen = P('M4 20l4-1L19 8l-3-3L5 16zM14 7l3 3')
export const IconDoor = P('M6 21V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v17M3 21h18M14 12h.01')
export const IconWindow = P('M4 4h16v16H4zM12 4v16M4 12h16')
export const IconScan = P('M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3M7 12h10')
export const IconRuler = P('M3 17 17 3l4 4L7 21zM7 13l2 2M10 10l2 2M13 7l2 2')
export const IconLink = P('M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1')
export const IconCamera = P('M3 8h3l2-3h8l2 3h3v12H3zM12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z')
export const IconPalette = P('M12 3a9 9 0 1 0 0 18c1 0 1.5-.8 1.5-1.6 0-1.2-1-1.4-1-2.4 0-.8.7-1.5 1.5-1.5H16a5 5 0 0 0 5-5c0-4.1-4-7.5-9-7.5ZM7.5 11h.01M10 7.5h.01M15 7.5h.01')
export const IconLayers = P('m12 3 9 5-9 5-9-5zM3 13l9 5 9-5')
export const IconCheck = P('m5 12 5 5L20 7')
export const IconPlus = P('M12 5v14M5 12h14')
export const IconSparkle = P('M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6')
export const IconHome = P('M3 11 12 3l9 8M5 9.5V21h14V9.5')
export const IconWalk = P('M13 4.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM9 21l2-6 3 3v3M7 12l3-4h4l3 4M11 15l-1-4')
/** Stacked cards: your homes and designs. */
export const IconDesigns = P('M7 7h12a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1ZM4 16V5a1 1 0 0 1 1-1h11M9.5 16.5l2.5-3 2 2 1.5-1.5 2 2.5')
