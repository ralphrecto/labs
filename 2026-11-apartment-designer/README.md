# Apartment designer

A cozy design game about your real apartment: scan your place with your phone, then make it
beautiful. It follows the [implementation plan](https://claude.ai/artifact/TWePZZMNEX1EkTwRPEe4Yf):
a furniture playground at true size (M1), Scaniverse scan import with tape-measure scaling (M2),
room and furniture detection from the scan (M3), and adding real products from a link, a photo or
pasted text (M4), all wrapped in a game UI.

## The game

- **Move in** (first run, `src/game/Onboarding.tsx`): import a Scaniverse splat or mesh (or the
  sample scan), tap both ends of something you measured to set the true size, let it find your
  room and furniture, then start designing. "Try a sample home" and "Start with an empty room" skip
  the scan.
- **Three modes** on the bottom dock (`src/game/GameApp.tsx`): **Build** (rooms, walls, doors,
  windows, scans), **Furnish** (the furniture drawer, `src/game/BuyDrawer.tsx`, with illustrated
  items in `ItemThumb.tsx` and a "Shop from a link or photo" card), and **Look** (no chrome:
  dollhouse spin or walk in at eye level).
- **Contextual controls**: selecting something shows one bar with its actions (rotate, copy,
  color, details, remove) instead of toolbars. Exact sizes and the fine print live behind Details.
- **Goals and stars** (`src/game/goals.ts`, `Goals.tsx`): interior design rules of thumb (somewhere
  to sleep, clear walkways, a nightstand by the bed, coffee table 35 to 50 cm from the sofa, a rug
  under the seating...) earn stars and level the home up from "Empty shell" to "Dream home".
- **Settings** (gear): units, overlays, snapping, saved designs, files, and the **Classic editor**
  with every panel and tool from the earlier milestones.

Game state that isn't part of the document (phase, mode, camera, open sheet) is in
`src/game/gameStore.ts`. Shared look: `src/game/theme.css` tokens and `.g-*` primitives; each game
component has its own CSS file with a class prefix (`.hud-`, `.buy-`, `.goal-`, `.onb-`).

## Run it

```sh
npm install
npm run dev      # http://localhost:5173
npm test         # unit tests for geometry, units, snapping, checks and scan alignment
node scripts/check-scan-flow.cjs   # with `npm run dev` running: imports the sample scan, calibrates, traces
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
| `src/model/catalog.ts` | Base furniture categories with default sizes, clearances and looks |
| `src/model/starterCatalog.ts` | Starter catalog of generic products at typical sizes |
| `src/import/dimensions.ts` | Reads dimensions and a category out of free text (shared with the server) |
| `src/import/productPage.ts` | Reads a product page's HTML: JSON-LD, meta tags, spec text, GLB links |
| `src/import/draft.ts`, `src/components/AddDialogs.tsx` | Catalog browser and the link/photo/text import dialog |
| `server/` | Import server: safe page fetch (public hosts only), model proxy, Claude calls |
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
- The project lives in the browser's local storage, with JSON export as the backup. The import
  server is stateless; it only reads pages and photos.
- Photo import makes a sized, colored item with the photo attached, not a 3D model of the photo.
  Image-to-3D (TRELLIS.2 or SAM 3D on a GPU) is a later step from the plan.
- Metric by default; switch to feet and inches in the toolbar.
- Sofa front clearance defaults to 45 cm (the gap to a coffee table); walkways are covered by the
  walking path check instead. Every clearance is editable per item.
