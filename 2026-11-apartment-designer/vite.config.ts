import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  // Relative asset paths so the build works from any sub-path (GitHub Pages, static hosts).
  base: './',
  plugins: [react()],
  // The single-file build (VITE_EMBED=1) needs everything in one script.
  build: process.env.VITE_EMBED === '1' ? { rolldownOptions: { output: { codeSplitting: false } } } : undefined,
})
