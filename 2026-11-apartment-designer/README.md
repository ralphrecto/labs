# Apartment designer

A browser app for planning an apartment's furniture at true size. This is Milestone 1 of the
[implementation plan](https://claude.ai/artifact/TWePZZMNEX1EkTwRPEe4Yf): a furniture playground on a
floor plan you enter by hand. Later milestones add video capture (Gaussian splats), furniture
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

Conventions: meters everywhere. Plan x is right and plan y is down; 3D world is Y up with plan
(x, y) at world (x, 0, y). An item's front is its local +y; rotation is clockwise on the plan.

## Assumptions

- Web-first, laptop browser as the main editing surface (the plan's default).
- No backend yet: the project lives in the browser's local storage, with JSON export as the backup.
- Metric by default; switch to feet and inches in the toolbar.
- Sofa front clearance defaults to 45 cm (the gap to a coffee table); walkways are covered by the
  walking path check instead. Every clearance is editable per item.
