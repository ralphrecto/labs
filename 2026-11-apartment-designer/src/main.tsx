import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Handles for browser tests and debugging in development builds only.
if (import.meta.env.DEV) {
  void Promise.all([import('./store'), import('./scan/align'), import('./scan/sampleScan'), import('./scan/runtime')]).then(([store, align, sample, runtime]) => {
    ;(window as unknown as Record<string, unknown>).__apt = { useStore: store.useStore, align, sample, runtime }
  })
}
