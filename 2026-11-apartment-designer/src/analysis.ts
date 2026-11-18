import { useMemo } from 'react'
import { checkLayout, pathIssues, walkingPaths, type Issue, type PathReport } from './model/checks'
import type { Doc, Layout } from './model/types'

export interface Analysis {
  issues: Issue[]
  paths: PathReport
  /** Placement ids that have an error or warning. */
  flagged: Map<string, 'error' | 'warning'>
}

export function analyze(doc: Doc, layout: Layout): Analysis {
  const paths = walkingPaths(doc, layout)
  const issues = [...checkLayout(doc, layout), ...pathIssues(paths, doc.apartment.units)]
  const flagged = new Map<string, 'error' | 'warning'>()
  for (const i of issues) {
    // Only the item that needs the clearance is flagged, not the one in its way.
    const ids = i.kind === 'clearance' ? i.placementIds.slice(0, 1) : i.placementIds
    for (const id of ids) {
      if (i.severity === 'error') flagged.set(id, 'error')
      else if (i.severity === 'warning' && !flagged.has(id)) flagged.set(id, 'warning')
    }
  }
  return { issues, paths, flagged }
}

export function useAnalysis(doc: Doc, layout: Layout) {
  // Rooms, items and the layout are the only inputs; units only affect messages.
  return useMemo(() => analyze(doc, layout), [doc.rooms, doc.items, layout, doc.apartment.units])
}
