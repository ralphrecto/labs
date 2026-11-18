import type { ScanData } from './types'

// Point clouds are too big for localStorage, so they live in IndexedDB, keyed by scan id. The
// document only keeps a reference (Room.scan). Everything here fails soft: without IndexedDB
// (private mode, tests) scans still work for the session from the in-memory cache.

const DB = 'apartment-designer-scans'
const STORE = 'points'
const memory = new Map<string, ScanData>()

function open(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      if (typeof indexedDB === 'undefined') return resolve(null)
      const req = indexedDB.open(DB, 1)
      req.onupgradeneeded = () => req.result.createObjectStore(STORE)
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => resolve(null)
    } catch {
      resolve(null)
    }
  })
}

function run<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T | undefined> {
  return open().then(
    (db) =>
      new Promise((resolve) => {
        if (!db) return resolve(undefined)
        try {
          const tx = db.transaction(STORE, mode)
          const req = fn(tx.objectStore(STORE))
          req.onsuccess = () => resolve(req.result)
          req.onerror = () => resolve(undefined)
        } catch {
          resolve(undefined)
        }
      }),
  )
}

export async function saveScan(scan: ScanData) {
  memory.set(scan.id, scan)
  await run('readwrite', (s) => s.put(scan, scan.id))
}

export function cachedScan(id: string) {
  return memory.get(id)
}

export async function loadScan(id: string): Promise<ScanData | undefined> {
  const hit = memory.get(id)
  if (hit) return hit
  const scan = (await run<ScanData>('readonly', (s) => s.get(id))) ?? undefined
  if (scan) memory.set(id, scan)
  return scan
}

export async function listScans(): Promise<string[]> {
  const keys = (await run<IDBValidKey[]>('readonly', (s) => s.getAllKeys())) ?? []
  return [...new Set([...memory.keys(), ...keys.map(String)])]
}

export async function deleteScan(id: string) {
  memory.delete(id)
  await run('readwrite', (s) => s.delete(id))
}
