# Apartment designer

A browser app for planning an apartment's furniture at true size, following the
[implementation plan](https://claude.ai/artifact/TWePZZMNEX1EkTwRPEe4Yf). Milestone 1 is a furniture
playground on a floor plan you enter by hand. Milestone 4 adds a starter catalog and adding real
products from a link, a photo or pasted text. Milestones 2 and 3 (video capture, furniture
detection) come later.

## Run it

```sh
npm install
npm run dev      # http://localhost:5173
npm test         # unit tests for geometry, units, snapping and checks
npm run build    # static build in dist/ (relative paths, host anywhere)

npm run server   # import server for links and photos on :8787 (npm run dev proxies /api to it)
npm start        # build, then serve the app and the import server together on :8787
```

The app works without the server; link import then falls back to reading the product name from
the URL and the size from text you paste. Set `ANTHROPIC_API_KEY` before starting the server to
turn on photo recognition and the AI fallback for pages the parser can't read or stores that block
it (`IMPORT_MODEL` overrides the model, default `claude-opus-5-5`).

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
- **Starter catalog.** About 80 generic pieces at typical retail sizes (sofas, beds twin to
  California king, dining tables by seat count, rugs 5×8 to 9×12, desks, storage, appliances),
  searchable by name or size ("queen", "8x10", "bar stool").
- **Add from a link.** Paste a product page link. The import server reads schema.org product data
  and Open Graph tags, then the spec text ("Overall Width - Side to Side: 84''", `84"W x 38"D x
  34"H`, `213 x 96 x 85 cm`), and finds a published GLB model if there is one (loaded through the
  server, scaled to the listed size). When the parser misses the name or size, or the store blocks
  the request, Claude reads the page instead. Listed sizes are quoted back so you can check them.
- **Add from a photo.** Drop a photo or a listing screenshot. Claude names the piece, picks its
  type and estimates its size (or reads it exactly from a screenshot); the photo's main color
  tints the model and a thumbnail stays on the item. Estimated sizes are flagged until you edit them.
- **Paste text.** Paste a name and dimensions from anywhere; parsed in the browser, no server.
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
| `src/model/catalog.ts` | Base furniture categories with default sizes, clearances and looks |
| `src/model/starterCatalog.ts` | Starter catalog of generic products at typical sizes |
| `src/import/dimensions.ts` | Reads dimensions and a category out of free text (shared with the server) |
| `src/import/productPage.ts` | Reads a product page's HTML: JSON-LD, meta tags, spec text, GLB links |
| `src/import/draft.ts`, `src/components/AddDialogs.tsx` | Catalog browser and the link/photo/text import dialog |
| `server/` | Import server: safe page fetch (public hosts only), model proxy, Claude calls |
| `src/store.ts` | Zustand store: document, undo/redo history, UI state, all edit actions |
| `src/storage.ts` | Local persistence and JSON import/export (swap for a server later) |
| `src/components/PlanView.tsx` | SVG plan editor |
| `src/three/` | react-three-fiber scene and procedural furniture models |

Conventions: meters everywhere. Plan x is right and plan y is down; 3D world is Y up with plan
(x, y) at world (x, 0, y). An item's front is its local +y; rotation is clockwise on the plan.

## Assumptions

- Web-first, laptop browser as the main editing surface (the plan's default).
- The project lives in the browser's local storage, with JSON export as the backup. The import
  server is stateless; it only reads pages and photos.
- Photo import makes a sized, colored item with the photo attached, not a 3D model of the photo.
  Image-to-3D (TRELLIS.2 or SAM 3D on a GPU) is a later step from the plan.
- Metric by default; switch to feet and inches in the toolbar.
- Sofa front clearance defaults to 45 cm (the gap to a coffee table); walkways are covered by the
  walking path check instead. Every clearance is editable per item.
