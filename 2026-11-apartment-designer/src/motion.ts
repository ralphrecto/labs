/**
 * The player's "reduce motion" system setting, read once at startup. Large 3D motion (the Look
 * mode spin, furniture dropping in) checks it; CSS animations are switched off in theme.css.
 */
export const reduceMotion: boolean =
  typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Pieces coming out of a moving box: each pops in (scale 0 → 1.1 → 1 with a small hop) over POP_MS, POP_STAGGER_MS apart. */
export const POP_MS = 350
export const POP_STAGGER_MS = 120
