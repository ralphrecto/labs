/**
 * The player's "reduce motion" system setting, read once at startup. Large 3D motion (the Look
 * mode spin, furniture dropping in) checks it; CSS animations are switched off in theme.css.
 */
export const reduceMotion: boolean =
  typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
