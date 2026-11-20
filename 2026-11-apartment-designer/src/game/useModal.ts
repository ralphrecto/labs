import { useEffect, useRef, type RefObject } from 'react'

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

const focusables = (root: HTMLElement) =>
  Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.getClientRects().length > 0 && !el.closest('[inert]'))

/** Small forms inside a sheet (add a room, set a length) use Escape themselves to cancel. */
const inForm = (el: EventTarget | null) => el instanceof HTMLElement && !!el.closest('form')

/**
 * Makes a game sheet behave as a modal dialog for keyboard and screen reader players: focus moves
 * to its first control on open, Tab and Shift+Tab stay inside, Escape closes it, everything else
 * in the game is `inert` while it is open, and focus goes back to what opened it on close.
 * `ref` is the dialog element (give it role="dialog" and aria-modal="true").
 */
export function useModal(ref: RefObject<HTMLElement | null>, onClose: () => void) {
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose
  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null

    // Everything beside the sheet's backdrop in the game root goes inert.
    const backdrop = dialog.closest<HTMLElement>('.g-scrim') ?? dialog
    const benched: HTMLElement[] = []
    for (const el of Array.from(backdrop?.parentElement?.children ?? [])) {
      if (el !== backdrop && el instanceof HTMLElement && !el.inert) {
        el.inert = true
        benched.push(el)
      }
    }

    if (!dialog.contains(document.activeElement)) (focusables(dialog)[0] ?? dialog).focus({ preventScroll: true })

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (inForm(e.target) || e.defaultPrevented) return
        e.preventDefault()
        onCloseRef.current()
      } else if (e.key === 'Tab') {
        const list = focusables(dialog)
        if (!list.length) return e.preventDefault()
        const first = list[0]
        const last = list[list.length - 1]
        const at = document.activeElement
        if (e.shiftKey && (at === first || !dialog.contains(at))) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && (at === last || !dialog.contains(at))) {
          e.preventDefault()
          first.focus()
        }
      }
    }
    dialog.addEventListener('keydown', onKey)
    return () => {
      dialog.removeEventListener('keydown', onKey)
      for (const el of benched) el.inert = false
      if (opener?.isConnected) opener.focus({ preventScroll: true })
    }
    // Runs once per opening; the latest onClose is read through onCloseRef.
  }, [ref])
}
