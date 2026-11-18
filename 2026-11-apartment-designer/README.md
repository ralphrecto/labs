# Apartment designer

A browser app for planning an apartment's furniture at true size. Milestone 1 of the
[implementation plan](https://claude.ai/artifact/TWePZZMNEX1EkTwRPEe4Yf) is a furniture playground on
a floor plan you enter by hand; Milestone 3 adds rooms and existing furniture detected from a scan. Later milestones add video capture (Gaussian splats), furniture
detection, and adding items from links and photos.

## Run it

```sh
npm install
npm run dev      # http://localhost:5173
npm test         # unit tests for geometry, units, snapping and checks
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
- **Rooms and furniture from a scan** (Milestone 3). Open a point cloud or Gaussian splat export
  (`.ply`, in meters), or try the built-in sample scan. The app levels the scan, squares it up
  with the walls, and finds the room outline, ceiling height, doors and windows, and every piece of
  furniture as a sized box with a guessed type, color and facing (backs go against walls). A
  review panel lists what it found: uncheck clutter, relabel a wrong guess (it keeps the measured
  size), fix a size, and choose whether the scan becomes a new room or replaces one you drew.
  Added furniture is ordinary furniture marked as owned, so you can drag it around and the
  checks apply. The scan's points stay under the plan and in 3D (toggle "Scan") so you can line
  things up. If you run the scan through SpatialLM on a GPU, paste its layout text instead.
- **Undo/redo** for every change (Ctrl+Z, Ctrl+Shift+Z).
- **Export** the plan as a PNG, print or save it as a PDF, and download or open the whole
  project as JSON. Everything also autosaves in the browser.
- **Units.** Stored in meters; shown in cm/m or ft/in. Inputs accept `350`, `3.5m`, `11'6"`, `138in`.

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
| `src/three/` | react-three-fiber scene and procedural furniture models |
| `src/scan/types.ts` | `ScanData`: the scan input shared with the scan import code (Y up, meters) |
| `src/scan/estimate.ts` | `SceneEstimate`: what any detector outputs (room shell + furniture boxes, plan coordinates) |
| `src/scan/analyze.ts` | On-device detector: heading, room outline, doors and windows, furniture boxes |
| `src/scan/cloud.ts`, `grid.ts` | Point cloud leveling (RANSAC floor) and occupancy grid morphology |
| `src/scan/classify.ts` | Furniture type from size, and which side is the back |
| `src/scan/spatiallm.ts` | Reads SpatialLM's Wall/Door/Window/Bbox text into a `SceneEstimate` |
| `src/scan/apply.ts` | Turns a reviewed estimate into rooms, items and placements (one undo step) |
| `src/scan/scanUi.ts`, `components/ScanPanels.tsx` | Review state, import and review panels, plan overlay |
| `src/scan/scanStore.ts` | Leveled point clouds in IndexedDB, keyed by scan id (rooms keep `scan.scanId`) |
| `src/scan/ply.ts`, `synthetic.ts` | Small PLY reader; a synthetic studio scan for the demo and tests |

Conventions: meters everywhere. Plan x is right and plan y is down; 3D world is Y up with plan
(x, y) at world (x, 0, y). An item's front is its local +y; rotation is clockwise on the plan.

## How scan detection works, and its limits

Detection is geometric, not ML, so it runs in the browser on any scan. It squares the room to its
dominant wall direction, builds 5 cm occupancy grids, takes the largest floor region (cutting
openings narrower than about 90 cm, so a hallway seen through a door doesn't join), traces an
axis-aligned outline and snaps each wall to the wall surface in the points. Doors are gaps in the
wall from the floor up; windows are gaps above a solid sill. Furniture is everything standing
inside the walls, split where there's a gap or a big height change, boxed, and typed from its
size. Known limits: one room per scan; walls are square to each other (angled walls come out
stepped or straight); pieces that touch (a chair pushed under a desk) come out as one box; types
are a size guess (desk vs dresser is often wrong), and scaled-down or sparse scans do worse. A
server model like SpatialLM or SAM 3 can replace the detector: it only has to produce a
`SceneEstimate`, and the review step works the same.

## Assumptions

- Web-first, laptop browser as the main editing surface (the plan's default).
- No backend yet: the project lives in the browser's local storage, with JSON export as the backup.
- Metric by default; switch to feet and inches in the toolbar.
- Sofa front clearance defaults to 45 cm (the gap to a coffee table); walkways are covered by the
  walking path check instead. Every clearance is editable per item.
