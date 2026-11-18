# Apartment designer

A browser app for planning an apartment's furniture at true size, following the
[implementation plan](https://claude.ai/artifact/TWePZZMNEX1EkTwRPEe4Yf). Milestone 1 is a furniture
playground on a floor plan you enter by hand. Milestone 2 imports a phone scan (Scaniverse Gaussian
splat or mesh) as a backdrop to trace the floor plan from. Later milestones add furniture detection
and adding items from links and photos.

## Run it

```sh
npm install
npm run dev      # http://localhost:5173
npm test         # unit tests for geometry, units, snapping, checks and scan alignment
node scripts/check-scan-flow.cjs   # with `npm run dev` running: imports the sample scan, calibrates, traces
npm run build    # static build in dist/ (relative paths, host anywhere)
```

## What it does

- **Floor plan editor.** Add a rectangular room by size, or draw any shape corner by corner
  (snaps to 45° and the grid; type a length and press Enter for exact walls). Drag corners,
  add corners from wall midpoints, edit wall lengths in the side panel. Doors (hinge side,
  swing in/out, front door) and windows (sill height) sit on walls. Each room has its own
  ceiling height and wall thickness.
- **Two linked views.** A 2D plan (main editing view) and a 3D view with cut-away walls and an
  eye-level look-around. Drag furniture in either.
- **Furniture.** A catalog of common pieces with real default sizes (bed presets twin to king).
  Drag to move, R / Shift+R to rotate 15°, a rotation handle, arrow keys to nudge. Snaps backs
  against walls, edges to other items, and to a grid (hold Alt to skip snapping). Items are boxes
  for collision; an optional GLB URL replaces the look, scaled to the item's true size.
- **Clearance checks.** Every item carries clearance zones (front, back, left, right). The app
  flags overlapping items, items crossing walls, blocked clearance zones, and furniture in door
  swings, and lists them in the Checks panel.
- **Walking paths.** With a front door marked, a 5 cm occupancy grid finds the widest route from
  the entrance to every room and reports the narrowest gap that furniture creates (doorways and
  narrow halls don't count, since you can't move them).
- **Layouts.** A layout is a named set of placements over the apartment. Copy a layout to try a
  variant and compare two side by side. Items you don't own yet can carry a price and link, and
  the layout shows what you'd need to buy.
- **Undo/redo** for every change (Ctrl+Z, Ctrl+Shift+Z).
- **Export** the plan as a PNG, print or save it as a PDF, and download or open the whole
  project as JSON. Everything also autosaves in the browser.
- **Units.** Stored in meters; shown in cm/m or ft/in. Inputs accept `350`, `3.5m`, `11'6"`, `138in`.

## Importing a scan (Milestone 2)

Scan the room with [Scaniverse](https://scaniverse.com) (no LiDAR needed: it builds a Gaussian splat
from the camera alone), then export it and drop the file on the **Scans** panel.

- **Formats.** Splats: PLY (the standard 3DGS layout, also compressed PLY), SPZ, `.splat`, `.ksplat`,
  `.sog`. Meshes: GLB/glTF, OBJ, PLY with faces. Point clouds: PLY. Splats render with
  [Spark](https://sparkjs.dev); meshes and point clouds with three.js. **Try a sample** imports a
  synthetic splat of an L-shaped room (deliberately 4% too small and turned 17°) to try the tools.
- **Automatic first alignment.** On import the app finds which axis is up (the floor is the biggest
  flat surface; Y up gets a head start, since ARKit scans are Y up), the floor and ceiling heights (peaks
  in the height histogram), and turns the scan so its walls run along the plan axes (the rotation that
  makes the wall points' x/y histograms sharpest). Each can be redone or adjusted in the Scan panel:
  up axis, *Square up*, *Along a wall* (click two points on a wall), 90° turns, *Move*, floor nudge.
- **Scale from a tape measure.** Without LiDAR, the scan's scale is only approximate. Mark 2 or 3
  tape-measured lengths on the scan (whole walls, wall to wall, are best; ends snap to the scan's
  walls). The scale is the median of what each length implies, and each length shows how far it is
  from the others, with a warning above 2% so one bad measurement stands out instead of skewing
  the plan. References are stored in scan coordinates, so they survive re-scaling and re-aligning.
- **Wall tracing.** *Trace a room* is the normal room drawing tool, but corners snap to walls seen in
  the scan: along a wall to that wall, at a corner to both. A traced room takes the scan's ceiling
  height when one was found. On the sample, traced walls come out within about 1 cm of the truth.
- **Plan underlay.** *Wall outline* draws the points in a height band (default 1 to 2 m above the
  floor, above most furniture), which stack up into crisp lines where walls are. *Top view* is a
  colour image from above with everything over the cut height removed.
- **3D backdrop.** The 3D view shows the scan aligned with the model (*With model*, *Scan only*, or
  hidden). From above, the scan is cut at the cut height so you can see in; at eye level it's whole.
- **Storage.** Scan files live in IndexedDB (`apartment-designer-scan-files`); the project only holds
  each scan's alignment and references. Opening a project JSON on another computer asks for the scan
  file again.
- **For furniture detection.** `toScanData(ref, points)` in `src/scan/align.ts` returns the scan in
  the app's world frame (meters, Y up, floor at 0, plan (x, y) = world (x, z)), downsampled to under
  a million points, in the `ScanData` shape from `src/scan/types.ts`.

## Code map

| Path | What |
| --- | --- |
| `src/model/types.ts` | Data model: apartment, rooms (wall polygon, openings), items, layouts |
| `src/model/geometry.ts` | 2D vector math, oriented boxes, SAT overlap, door swings |
| `src/model/snapping.ts` | Wall, edge and grid snapping |
| `src/model/checks.ts` | Clearance checks, occupancy grid, walking paths |
| `src/model/catalog.ts` | Furniture catalog with default sizes and clearances |
| `src/store.ts` | Zustand store: document, undo/redo history, UI state, all edit actions |
| `src/storage.ts` | Local persistence and JSON import/export (swap for a server later) |
| `src/components/PlanView.tsx` | SVG plan editor |
| `src/three/` | react-three-fiber scene and procedural furniture models; `ScanBackdrop.tsx` draws scans |
| `src/scan/align.ts` | Scan math: up axis, floor and ceiling, straightening, calibration, underlays, wall snapping |
| `src/scan/readPly.ts` | Small PLY reader (binary and ASCII; splat, mesh and point PLYs) |
| `src/scan/importers.ts` | Loads scan files into points plus something to draw (Spark splats, three.js meshes) |
| `src/scan/runtime.ts` | Scans loaded in this session: import, IndexedDB reload, cached underlays |
| `src/scan/sampleScan.ts` | Synthetic splat room used by *Try a sample* and the tests |
| `src/components/ScanImport.tsx` | Scans panel and the scan inspector (align, scale, trace, display) |

Conventions: meters everywhere. Plan x is right and plan y is down; 3D world is Y up with plan
(x, y) at world (x, 0, y). An item's front is its local +y; rotation is clockwise on the plan.

## Assumptions

- Web-first, laptop browser as the main editing surface (the plan's default).
- No backend yet: the project lives in the browser's local storage, with JSON export as the backup.
- Metric by default; switch to feet and inches in the toolbar.
- Sofa front clearance defaults to 45 cm (the gap to a coffee table); walkways are covered by the
  walking path check instead. Every clearance is editable per item.
