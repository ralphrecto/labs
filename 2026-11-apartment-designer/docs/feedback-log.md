# Feedback loop log

Each round: `npm run dev`, then `node scripts/playthrough.cjs <dir>` plays the first-run scan flow
(desktop) and a sample-home session (iPhone 14), saving screenshots. Four reviewer agents play as
different players (cozy gamer, first-time player, art director, interior designer), each reporting
up to 8 ranked findings. The top findings are fixed in that round's branch; the rest roll over.

## Round 1 (2026-11-20)

Reviewers: cozy gamer (phone), first-time player, art director, interior designer. 30 findings.

**Fixed this round**
- New furniture landed across walls, in door swings and in the wrong rooms. Placement now looks
  for a legal spot in the right kind of room, backs wall pieces against a wall, and pairs
  nightstands with the bed.
- Stars rewarded broken layouts and buying furniture could drop your level. Only clean items
  count, levels sit on a fixed core goal set and never drop from adding furniture, the lamp
  goal skips bathrooms. The Goals sheet lost the extra Vibe bar and duplicate totals.
- Measure step read "5.2" as 5.2 cm and shrank the home 100x while saying "Sized!". It now
  infers the unit from the scan, shows a live readout, catches implausible lengths, and the
  detection failure screen can go back to re-measure.
- Copy named things that don't exist ("Tap Buy", "Ctrl+Z" on phone, "Layout 1"); scan step
  jargon (LiDAR, Splat, PLY/SPZ) replaced with plain language.
- Phone style picker opened under the furniture drawer.
- Walk-in started nose to the wall; toast hid the "Goal complete" banner; heavy Cartoon wall
  shadows; washed-out Lifelike and Clay; camera didn't re-frame around open panels; Look HUD
  inconsistencies.
- 3D furniture was grey boxes next to illustrated thumbnails; procedural models now match the
  thumbnail style (cushions, pillows, duvets, tapered legs, decor).

(Round 1's backlog is carried and updated in round 2 below.)

## Round 2 (2026-11-20)

Reviewers: accessibility and touch, casual mobile newcomer, returning player (day 3). A power-user
review was cut off by a usage limit and moves to round 3. 24 findings.

**Fixed this round**
- Round 1 follow-ups: the overview camera fills the visible area (about 10% margin, also with the
  drawer or selection bar open); windows have frames, sills and sky glass, plus a daylight patch;
  walk-in keeps 0.25 m from walls and passes through inner doorways; measuring lines only show
  while measuring.
- Load the sample home or Move into a new place silently overwrote your only save, scan
  included. Each home now has its own save slot (old saves migrate), and a home switcher sits at
  the top of Your designs. Designs get thumbnails, friendly names, a "…" menu and a delete
  confirm.
- No first goal: a "Next: … +N ★" chip under the star pill opens the right drawer tab and
  highlights the card. A level-up card with confetti, a "Welcome back" banner after 6 hours
  away, one level name everywhere, a grey Mine count, plant first in Decor, best level kept per
  design.
- Goals the plan can't finish (bed goals without a bedroom, dining without a dining space, front
  door) are dropped or inferred; locked goals fold into one line.
- A bed from the Bedroom tab landed in the living room hugging the coffee table. Beds, nightstands
  and wardrobes stay in the bedroom when there is one; a full bedroom says so instead.
- The sample scan measures itself ("We measured this one for you") and its card sits in the first
  screen; the drop-zone text is hidden on touch; Welcome home sets a first goal.
- Accessibility: coral buttons darkened to pass contrast (#c94a28), 44 px tap targets, Undo and
  Redo by the dock on phones, reduced motion stops the Look spin and the drop bounce, sheets trap
  focus, one visible focus ring, pinch no longer zooms the page, a landscape phone layout with
  the drawer on the right, 13 px minimum labels.

**Backlog (next rounds)**
- Power-user review (efficiency: taps per room, shortcuts, multi-place, rotate 90°).
- Photo or share moment from Look mode (polaroid with name, level and stars).
- Something to aim for after the top level: weekly challenges, badges; room focus ("Redo the
  bedroom" card with that room's stars).
- Keyboard selection of placed furniture; walk-in doesn't collide with furniture.
- The dresser's auto-placement can land in front of the bed (clearance warning).
- Placement sounds; catalog presets and more items; more design goals (rug size, TV distance).
- Landscape: the room centres on the full width, partly behind the side drawer.
- Playthrough harness: the Scan style stalls headless SwiftShader, so later desktop steps time out.

## Round 3 (2026-11-20)

Reviewers: power user (speedrun), art director re-check, interior designer re-check. 24 findings.

**Fixed this round**
- Furnishing speed: furnishing 13 pieces went from 29 clicks to 16. The drawer stays open after
  placing, with the piece's actions in a slim strip (on phones the drawer shrinks to one row of
  thumbnails so the room stays visible). Selecting no longer refits the camera, orbiting no
  longer deselects, near-miss clicks pick the closest piece. Rotate is 90° (`,`/`.` for 15°) and
  slides the piece to a clear spot if the turn collides. Arrows follow the camera; nudge runs are
  one undo step. Shortcuts B/F/L, P, Home, 1-6, Enter, and a "?" list on desktop.
- Size chips for sofas, rugs, dining tables, desks and lamps from the retail catalog; a table lamp.
- Placement: chairs and nightstands stay with their table, desk or bed (or are refused with a
  reason); dresser is bedroom-only; living-room pieces avoid the kitchen; full front clearance
  for beds, wardrobes, dressers and desks; lamps go to the seat without one; plants go to an
  empty room.
- Honest levels: Next points at layout problems first (overlaps, walkways, TV off-axis); the
  level is capped at Cozy while there's a snag; companions no longer cost stars. Fullness "just
  right" is 35% (38% bedrooms). New bonus goals: room to get into bed, rug big enough,
  comfortable TV distance.
- Looks: warm Lifelike and Clay stages with value separation; one softer coral (#e0603a) with
  terracotta only on fills; the toast sits below the Next chip; the level-up card gets a warm dim,
  sun header and badge; selected pieces keep their material with an outline and a pulsing floor
  ring and a small hop; baseboards and wall caps.

**Backlog (next rounds)**
- Phone: after the drawer shrinks the camera doesn't reframe, so the house stays small; consider
  refitting once when the drawer changes height by a lot.
- Photo/share moment from Look mode; weekly challenges and badges after the top level; room
  focus ("Redo the bedroom").
- Desktop layout: the room is small above the drawer; consider a side-panel drawer from ~1100 px.
- Your designs: 3D snapshot thumbnails, 2-up cards, "<level> living room" names.
- Non-companion pieces that can't keep walkways clear are still placed (console, small desk);
  desk can float mid-room; armchair can sit tight against the sofa.
- Double-click a room to frame it; Tab to cycle pieces; keyboard selection.
- Walk-in: furniture collision; Lifelike walk-in walls still neutral.
- Dev note: restart the dev server after big merges; stale HMR state caused bogus splat shader
  errors in the Scan style during round 3 testing.

## Round 4 (2026-11-20)

Reviewers: cozy phone gamer, first-time player (incl. the real-scan path), returning player.
A usage limit paused the round from 09:00 to 10:50 UTC. 24 findings.

**Fixed this round**
- Real-scan measuring: a corner magnet (about 25 px) snaps taps to wall corners and wall ends
  (two taps on the long wall now read 4.99 m, was 4.52 m and left the home 11% too big). A gentle
  ceiling check warns when the sized ceiling comes out outside 2.25-2.75 m. The length label no
  longer hides under the hint; no scale bar before sizing. Welcome home's "First up" matches the
  Next chip. "Back to <home>" from Move into a new place.
- The Next chip never dead-ends: it dry-runs placement and suggests a piece that fits (bookshelf
  when the bedroom is full), comes back after a refused placement, and skips goals nothing can
  finish. A living room is no longer treated as a studio, so the first goals are sofa, rug, lamp
  and plant. Pieces with a snag earn no stars or level-ups.
- The sample home is now move-in day (rooms, sofa, TV stand, bed, two boxes; starts Lived-in).
- Honest level: the pill shows the home's current level with a small "Best: …" note; only a new
  level on a clean layout celebrates. Welcome back says how long you were away.
- Calmer rewards: one-line toasts, level-up as a banner under the stars on phones, toast hidden
  while it shows, "½" feedback on the first half of two-step goals, no camera labels on phones.
- Forgiving drags: a green/red footprint while dragging; releasing in a bad spot slides to the
  nearest clean spot or back to the last clean pose with a wobble; one undo step per drag.
- Phones refit the camera to the selected piece's room (and back on deselect).
- Look mode fills the screen and has a photo moment: one shutter, a flash, a polaroid with the
  home name, level and stars, Share or Save. Walk-in stops short of furniture.

**Backlog (next rounds)**
- Your designs: show stars and level per design, a crown on the best, "Try a variation" as the
  primary action, copies named after their source, rename "Current".
- Home delete has no undo; a failed save (full storage) is silent.
- Levels climb fast (sample: Lived-in to Dream home in about 9 placements); retune thresholds.
- "A table to eat at" took 4 chairs before any counted as pulled up.
- Finger drags need a check on a real iPhone (headless touch can't hit pieces reliably).
- Measuring: refit the plan after sizing so the top wall isn't under the hint pill.
- Challenges after the top level; room focus; desktop side-panel drawer; cap pieces per design.
- Playthrough harness: on phones the tabs hide while a piece is selected, so the scripted
  tab switch after placing fails; deselect first.

## Round 5 (2026-11-20, 17:27-18:00 UTC)

Reviewers: interior designer re-check, art director re-check, returning player (two sessions on one profile).

**Fixed**
- Fairer scoring. Nothing solid can land between the sofa and the TV, and a desk in that spot is a snag. Wall pieces no longer fall back to the middle of the room.
- Levels are capped at Designer home while any warning shows. Dream home also needs a few bonus stars, and the level thresholds are stricter.
- A dresser fits at the foot of the sample bed (0.75 m hard minimum). Chairs only count at the table when they can be pulled out, and a table for 4 or more comes off the wall.
- The reading-corner Next chip now places the lamp beside the armchair, so it no longer loops. Armchairs get spots at the ends of the coffee table.
- Visuals:
  - Fixed the Blueprint lines staying on walls and windows after switching style.
  - The selection outline is coral and thinner.
  - The tab ring is no longer clipped, and size chips stay on one row.
  - Toasts are paper pills that replace the Next chip while they show.
  - Lifelike and Clay are darker.
  - The Scan style is hidden when the home has no scan.
  - Bolder photo shutter, instant flash, and a date on the polaroid.
  - The dining table has plates and a vase, the moving boxes are cardboard, and the Build tiles are terracotta.
- Saves: a failed save keeps retrying and shows a "storage full, download a backup" toast.
- Undo no longer switches designs.
- Your designs: "Try a variation" is the main button, each card shows its stars and level, the best has a crown, copies are named after their source, and the sample's design is "Move-in day".
- Deleting a home can be undone.
- Welcome back: shows the home, design and level, stays until you act, and leads with "Try a variation" at the top level. The design name shows in the phone top bar.
- Harness: the phone run deselects before switching tabs and skips the link card.

**Backlog (next rounds)**
- The sample house fills only about 45% of the phone screen width. Refit it between the Next pill and the drawer, and refit after sizing.
- Phone HUD has about 11 floating controls. Merge map and home into one view button, and show undo and redo only after the first edit.
- The chair-back rule in checks.ts (0.75 m behind the chair) disagrees with the table's 0.75 m pull-out, so the first sample chairs land against the wall and don't count.
- Lighting:
  - Table lamps are floor pieces; they should sit on side tables, nightstands and desks.
  - Add bedside and desk light goals.
  - No pendant lights yet.
- Conversation-seating goal. Rugs under the dining table and the bed.
- Catalog roles: benches, counter stools, shoe cabinet, chaise and L-sectional.
- After the top level: a mastery set and weekly room challenges.
- Trash is only restorable right after a delete; add a "Recently deleted" row.
- Lifelike is still light in the small dollhouse view.
- Real iPhone finger-drag check.
- Harness: the first style tap after opening the picker sometimes misses.

## Round 6 (2026-11-20 23:53 to 2026-11-21 00:20 UTC)

Reviewers: cozy gamer first session, phone clarity and clutter, lighting designer.

**Fixed**
- Dining chairs: a chair's room behind it is now measured from the table edge (`chairBackNeed` in checks.ts), so placement and checks agree. Chairs go to the open side. When none count, the eat goal asks you to move the table off the wall.
- The dresser warning now matches the 0.75 m placement rule.
- Level pacing: Dream home needs every core and Style goal plus 2 bonus goals. Designer home needs every flow goal. Following Next on the sample now reaches Dream home in 12 placements (enforced by a test, with no loops).
- The bedroom density limit is now 42%, and an office chair tucked under its desk doesn't count toward it.
- The sample bed's headboard sits on a solid wall.
- Light goals count real lamps only.
- Phone layout:
  - Compact drawer by default (tap the open tab to expand), and a one-row top bar.
  - The plan view fits around the HUD, and one View button replaces map and home.
  - Disabled undo and redo are hidden, tap targets are 44 px, and iPhone SE overlaps are fixed.
  - The house now fills 74% of the width on iPhone 14 (was 46%) and 24% of the screen (was 9%).
- Game feel:
  - The Next chip stays while goals remain at the top level.
  - A star loss shows a "−N★ goal · Undo" pill, and level drops wait 2 s.
  - A "Tap Plant → Bathroom" hint pill.
  - Dream home gets a confetti card with "Take its photo".
  - The polaroid is titled with the design and home.
- Evening mode in Look (unlocks at Cozy): dusk lighting, and lamps glow and cast warm light.

**Backlog (next rounds)**
- Table lamps are still floor pieces. Next: a `table-lamp` category hosted on nightstands, side tables and desks, plus bedside and desk light goals.
- Pendant lights over the table; a layered-light goal.
- Moving boxes should be tappable to unpack into owned decor, with an "Unpack your boxes" goal.
- The density goal's hint has no piece to suggest; offer "move X to <room>".
- The expanded drawer on iPhone SE clips the bottom of its cards.
- Turning off the right inset would let the house fill 89% of the width, but the camera column would overlap it.
- Evening lamp shades read cream rather than amber in Cartoon.
- Rugs under the dining table and the bed; a conversation-seating goal; catalog roles (benches, stools).
- Real iPhone finger-drag check.
- Harness: the first style tap after opening the picker sometimes misses.

## Round 7 (2026-11-21, 06:20 to 07:10 UTC)

Reviewers: art director on evening mode and the compact layout, lamp-hosting design spec, returning player (three sessions on one profile, plus a blank design and a scanned home).

**Fixed**
- Table lamps stand on furniture:
  - New `table-lamp` category with `Placement.on` hosting. Lamps go on nightstands, side tables, desks and dressers, and refuse politely when there's no surface.
  - A lamp follows its host when the host is moved or rotated, and comes off or moves to another surface when the host is deleted (`reconcileHosts` in store update).
  - Old saves are migrated, and copying a design keeps the lamps on their hosts.
- New bonus goals: a reading light on each open side of the bed, and a lamp on the desk.
- Evening:
  - Sunlight patches hidden; Lifelike's environment light is dimmed.
  - A cool blue dusk with a gradient sky and dusk windows with stars.
  - Amber lamp shades with brighter light.
  - "Take its photo" on the Dream home card turns evening on.
- Compact drawer: tab labels stay, card names fit, the link card is calmer, and the caret no longer overlaps. A stale hint pill now clears itself.
- Next-chip dead ends:
  - The coffee-table loop is fixed: a piece the layout already has becomes a "move it" goal, and a coffee table lands in front of the sofa.
  - A bedroom keeps room for a bed.
  - A "Make room for a bed" goal replaces the silent skip.
  - An empty design of the sample reaches Dream home in 19 steps, and the sample in 14, both checked by tests.
- Top level: the goals sheet shows how many bonus goals are left. The Dream home card ignores taps for 1.5 s and closes only from the backdrop or its close button.
- Removed the floor-lamp floor halo, which spilled past walls onto the backdrop.

**Backlog (next rounds)**
- Variations are exact copies with nothing to climb. Remix with anchor pieces only plus a twist challenge, and celebrate a variation's first Dream home.
- Tappable moving boxes that unpack into owned decor (plan in /tmp/claude-0/r7-plan-boxes.md, which may be gone; re-derive it from the round 7 thread if needed).
- Pendant light over the dining table; layered-light goal.
- Cartoon and Clay dusk still read warm brown rather than blue; the sky gradient is hidden by the ground plane in Dollhouse.
- The Designs sheet can't tell an untouched copy from its source.
- The Dream card's close button reuses the goals sheet style, so check how it looks.
- Reading-corner goal buys a new lamp even when spare lamps exist.
- Short climb on one-room scanned homes (8 steps).
- Harness:
  - The style picker closes after each pick, so later style taps miss.
  - Backdate the last visit with addInitScript before load, because pagehide overwrites it.

## Round 8 (2026-11-21, 13:00 to 13:30 UTC)

Reviewers: cozy first-timer (14 Next taps to Dream home, no dead ends), lighting designer, remix player (who also wrote the specs).

**Fixed / added**
- **Remix this home:**
  - Pick a twist, and the new design keeps only the big pieces (bed, sofa, TV stand, wardrobe, dining table).
  - Six twists: Talk corner, Open floor, Window seat, Fresh start, Bed swap, Less is more. Each is a core 3★ goal, so Dream home needs it.
  - The new design climbs the levels again (13 Next steps in the test) and gets its own "Remix complete!" Dream card.
  - The Designs sheet shows a twist badge, a level bar and a Dream ribbon. An untouched copy reads "Same as X". The header shows how many designs are Dream homes.
- **Moving boxes:**
  - Tap a box, then Unpack: it pops out a plant, lamps and a rug as owned decor, with confetti. Undo brings the box back.
  - "Unpack your boxes" is the first Next goal on move-in day, and packed boxes don't count as furniture.
- **Lighting:**
  - Dusk is lit by a cool moon instead of an orange sunset, so all three styles read blue with warm lamp pools.
  - Lamps light pools instead of whole rooms.
  - A lamp only moves to the same kind of surface in the same room; otherwise it waits in Mine. A desk holds one lamp.
  - The bedside goal only appears where a nightstand actually fits.
  - New: a pendant light over the dining table, hung 76 cm above it, with a "Pendant over the table" bonus goal.
- **Cozy-player fixes:**
  - Evening survives React's dev double-mount, so the Dream photo is taken at dusk.
  - The hint pill floats above the drawer, so the named card is readable.
  - Toasts never show a third line.
  - The Cozy level-up says evening is unlocked.
  - Tidy welcome links on narrow phones.

**Backlog (next rounds)**
- The guided climb has no choices; add a taste decision per step (size presets or colour swatches on the hinted card).
- Auto-placement crowds the living room (the dining table lands mid-room, the armchair jams against the sofa). Prefer a dining room, or keep the table at least 1.2 m off the sofa front.
- The reading-corner goal still buys a second floor lamp when one stands nearby.
- The Window seat twist is already met on the sample; skip twists the source already satisfies.
- The table lamp from box 1 has nowhere to stand on move-in day.
- The pendant has no thumbnail, and its cord is a fixed length rather than reaching the ceiling.
- The polaroid caption should lead with the level at the top; frame tighter.
- Harness: the first Lifelike tap after opening the picker still misses.
