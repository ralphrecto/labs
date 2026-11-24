import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { Root } from './game/GameApp'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)

// Handles for browser tests and debugging in development builds only.
if (import.meta.env.DEV) {
  void Promise.all([import('./store'), import('./game/gameStore'), import('./scan/align'), import('./scan/sampleScan'), import('./scan/runtime')]).then(([store, game, align, sample, runtime]) => {
    ;(window as unknown as Record<string, unknown>).__apt = { useStore: store.useStore, useGame: game.useGame, align, sample, runtime }
  })
}
