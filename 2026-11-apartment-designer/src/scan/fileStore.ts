// The original scan files, kept in IndexedDB (scans are tens to hundreds of MB, far past
// localStorage). Keyed by ScanRef id. The project JSON only references them.

const DB = 'apartment-designer-scan-files'
const STORE = 'files'

export interface StoredScanFile {
  name: string
  type: string
  blob: Blob
  savedAt?: number
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open()
  try {
    return await new Promise<T>((resolve, reject) => {
      const t = db.transaction(STORE, mode)
      const req = fn(t.objectStore(STORE))
      t.oncomplete = () => resolve(req.result)
      t.onerror = () => reject(t.error)
      t.onabort = () => reject(t.error)
    })
  } finally {
    db.close()
  }
}

export async function saveScanFile(id: string, file: StoredScanFile) {
  await tx('readwrite', (s) => s.put({ ...file, savedAt: Date.now() }, id))
}

export async function loadScanFile(id: string): Promise<StoredScanFile | null> {
  try {
    return ((await tx('readonly', (s) => s.get(id))) as StoredScanFile | undefined) ?? null
  } catch {
    return null
  }
}

export async function deleteScanFile(id: string) {
  try {
    await tx('readwrite', (s) => s.delete(id))
  } catch {
    // Nothing to clean up.
  }
}

/**
 * Delete stored files no scan in the document uses. Run at startup, when there's no undo history
 * to bring one back. Files from the last day are kept, in case another tab is using them.
 */
export async function pruneScanFiles(keep: string[]) {
  try {
    const keys = (await tx('readonly', (s) => s.getAllKeys())) as string[]
    const k = new Set(keep)
    const dayAgo = Date.now() - 24 * 3600 * 1000
    for (const key of keys) {
      if (k.has(key)) continue
      const f = await loadScanFile(key)
      if (f && (f.savedAt ?? 0) > dayAgo) continue
      await tx('readwrite', (s) => s.delete(key))
    }
  } catch {
    // IndexedDB unavailable: nothing stored anyway.
  }
}
