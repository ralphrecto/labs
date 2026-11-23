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

## Round 9 (2026-11-21 19:30-20:15 UTC)

Reviewers: desktop player, cozy player on choices, art director (remix and unpack re-check).

Shipped:
- **Pick one:** each hinted step offers 2-3 variant cards (size, shape or colour), with "From your boxes" first when you own one. The game remembers your colour taste per family (wood, fabric, light) and puts that pick first.
- **Unpack all boxes:** the unpack goal stays urgent until both boxes are open (or 8 pieces). The next box is selected after one opens, and the camera holds on the unpacked spot. The confetti is no longer clipped.
- **Desktop:** the drawer is a right side column at ≥1100px, so the home fills about 40% of the screen (was 12%). Zoom to cursor, mouse rows in the shortcuts, key badges on mode buttons, Plan labels that avoid furniture, and Look framed between the bars.
- **Art:** a pendant thumbnail (was a box), a pendant cord sized to the ceiling with a canopy and no shadow blob, a kraft-coloured box, a softer evening sky, and "Pendant" naming.
- **Remix:** twist picker shows the hint and an emoji, the card reads "That's your Nth Dream home!" with a green twist pill, a Dream corner ribbon, copies left out of the Dream count, twists the home already meets are hidden, the twist goal points at the piece to move, and remixes land pieces in new spots.
- **Placement:** the dining table keeps away from the sofa front and leans to the kitchen.

Rolled over: "From your boxes" pick is untested in the browser; plan labels on a shared wall sit close; 700-1100px still uses the bottom sheet; toast copy for 3+ items.

209 tests pass.

## Round 10 (2026-11-22 02:10-02:50 UTC)

Reviewers: newcomer on iPhone (pick cards, boxes), tablet and interior designer (iPad, 1024x768, the Dream home result), long-session player (what happens after Dream home).

Shipped:
- **Remix depth:** the twist is the first goal of a remix and gets a honey pill and a pinned "Today's twist" section in the goals sheet. The twist picker puts unfinished twists first, rotates daily, never re-offers the one in progress, and counts "N of 6 twists done". Talk corner is now winnable with Dream home (3 m reach, a sofa-side lamp counts for the reading corner, a fit check before it's offered). Finished remixes get their own badge instead of competing for the crown. A remix toast says what was kept.
- **Seats for two:** a new comfort goal asks for a second dining chair, so Dream home no longer has a table for four with one chair.
- **Pick cards:** hidden while something is selected, headed with the goal title, wider two-line labels, visibly different variants (round table model, tall chest, smaller plant), "Carver" chairs, and walnut that stays brown in Cartoon. "From your boxes" matches anything the goal accepts. Remove sends bought pieces to Mine. The Mine badge counts only pieces waiting to be placed.
- **Toasts:** a clean two-line clamp and friendlier unpack copy ("Unpacked 4 things! 3 are waiting in Mine.").
- **Welcome back:** the stars moved into the badge, so the payoff isn't cut off.
- **Tablet:** the side column also applies to landscape screens from 1000px. iPad portrait gets one sideways row of cards. 44px touch targets on coarse pointers.
- **Plan:** upside-down labels fixed, wall dimensions never stack, labels hidden for hosted lamps and covered rugs, and the plan fit allows for the selection bar.
- **Placement:** the office chair centres on the desk with roll-back room, and plants keep off chair pull-out zones and hug walls.

Rolled over: a daily "visitor" request and a photo album (the long-session player's hooks); bathroom fixtures in the sample; the welcome-back card and the level-up toast still sit near the right-hand buttons on phones; the iPhone plan's leftmost wall dimension is clipped; the "Seats for two" goal is not yet tied to the pendant and table goals.

220 tests pass.

## Round 11 (2026-11-22 08:46-10:00 UTC)

This round built the two "reason to come back" hooks from round 10, alongside one fresh-eyes regression review (iPhone 14 and iPad landscape, sample home to Dream home plus a remix).

Shipped:
- **Visitor of the day** (`src/game/visits.ts`): 10 visitors (Grandma, book club, the little cousins, a friend with a dog...), one per local day, each with a wish checked against the real layout. It is a +2★ bonus goal that never caps the level. The welcome-back card names today's visitor, the goals sheet shows a visitor pill, a "Today's visitor" section and a sticker row ("4 of 10"), and finishing it says "<who> loved it!".
- **Photo album** (`src/game/album.ts`, `listAlbum`/`addAlbumPhoto` in `storage.ts`): every Look photo is kept (360px JPEG, 12 per home, the oldest unpinned photo goes first), with a corkboard row in the Designs sheet, a larger view, and delete with Undo. A remix's first Dream home pins a twist polaroid.
- **Dead end fixed:** "Make room for a desk" no longer tells you to move the bed out of the bedroom. It names a spare or movable living piece first.
- **Safe picks:** pick cards test-place each variant and drop any that would undo a goal or add a warning.
- **Dream card:** the level-up celebration is tracked separately from the banked level, so Dream home always gets its card.
- **Clearer blockers:** the goals sheet says `Finish "<goal>" to reach X` instead of a snag that isn't there.
- **Polish:** pick cards sit bottom-left on landscape tablets, no mid-word label breaks, "Talk corner" used everywhere, "From Mine" for pieces that didn't come from a box, office chair pick cards, and a "Remix this home" button in the goals sheet at Dream home.

Rolled over:
- Removing the dining table doesn't make Next ask for a table again.
- The welcome-back card is tall on iPhone with the visitor line.
- The album's larger view is soft on retina (360px source).
- The make-room goal doesn't suggest a smaller variant.

243 tests pass.

## Round 12 (2026-11-22 15:39-16:45 UTC)

Reviewers: a player over four simulated days (visitor and album), a desktop mouse-and-keyboard player, and a free decorator who ignores the Next chip.

Shipped:
- **Visitor:** today's visitor is picked after the move-in boxes, so the first one is met on purpose. The Cozy level-up shows both lines. The welcome-back card says who went home and who is coming, and adds "(once this design is Cozy)" when Next won't offer them yet. The visitor can lead when it's one pick away. The Designs sheet and album count the visitor's stars, so all star counts agree.
- **Welcome-back card:** sized to its content (up to 440px) instead of a 184px column.
- **Next and blockers:** when the stars are already in, Next goes to the core goal that holds the level. `levelBlocker` mentions a gate.
- **Goals sheet:** open goal rows are buttons that act like the Next chip.
- **Your palette:** a new +1★ style bonus for three painted pieces in a coherent palette, named from the colours (e.g. "Sage & Terracotta"). Painting also teaches your taste.
- **Gentler free decorating:** the coffee table counts at 35-60 cm (and no longer blocks a seat's front clearance), the living-room density cap is 40%, and loss pills wait 3 s and hide mid-gesture.
- **Desktop:**
  - Drag and drop lands where you let go, in 3D and Plan, with a footprint preview and a copy cursor.
  - Grab and grabbing cursors on pieces.
  - Undo no longer merges a copy or delete into a run of turns.
  - Esc leaves Look, and the focus ring uses the game's coral.
  - Pick cards sit beside the side drawer.
- **Album:** a visitor sticker and a 🌙 for evening on polaroids, and 480px photos.
- **Designs sheet:** the twist picker scrolls into view, and a new remix says "Just started".
- **Dream card:** stays visible and no longer leaves an invisible click-eating layer.
- **Colours:** painted colours in Cartoon now match their swatches.
- **Catalog:** size chips expose recliner, swivel, accent, round and small coffee tables, pouf, storage ottoman and C-shape side table.
- **Placement:** dining chairs never pull out through a wall, and the sofa and coffee table pair up better.
- **Phone:** drawer tabs stay while a piece is selected (a tap deselects).
- **Plan view:** lamps can move from one top to another.

Rolled over:
- The star number still changes live during a drag; holding the score until the gesture ends would need a change in `Goals.tsx`.
- Coffee tables aren't lamp surfaces.
- Wall art, mirrors and curtains are missing from the catalog.
- The sofa can sit off-centre from its coffee table in big rooms.

257 tests pass.

## Round 13 (2026-11-22 22:20-23:25 UTC)

The round started with two features built straight from the backlog (wall decor; steady stars while dragging). Two reviewers ran alongside them: one on the scan-your-own-home path, one on accessibility and performance. Two more implementers then fixed what those reviewers found, working on the merged code.

Shipped:
- **Wall decor** (`src/model/wallDecor.ts`):
  - Framed art (three sizes), a round mirror, a floating shelf with books, a clock, and curtains, in a new Walls tab.
  - Pieces snap flush to walls at sensible heights, keep off doors (and, except curtains, off windows), take no floor space, can hang over sofas and beds, and hide with cut-away walls.
  - Curtains snap only to windows.
  - Two bonus goals: "Something on the walls" and "Curtains in the bedroom".
- **Steady stars:** the meter holds its score while you drag and settles once at the end, with one net gain or one loss pill with Undo (`heldScore`, `settleStars`).
- **Coffee tables hold a lamp**, and a sofa centres on its coffee table (within 15 cm, 40-55 cm gap).
- **Scan-first onboarding:**
  - Rooms get friendly names (a "Scan 2026-…" file becomes "Living room", or "Bedroom" from a bed) and can be renamed inline.
  - "Add another room scan" places rooms side by side.
  - The export tip names PLY/SPZ, and .usdz gets a clear message.
  - The file button is above the fold on iPhone.
  - "Not sized yet" and "No door found → add it next" prompts.
  - The Scans panel becomes "Add a room" once you have a real scan, with "Find again" demoted.
  - A failed import no longer wipes the open home.
- **Accessibility:**
  - One always-present live region announces toasts, star changes, losses and level-ups.
  - The Dream card takes focus and stays while you read it.
  - The 3D room is a labelled image with a keyboard-focusable list of pieces (focus selects, so arrows/R/Delete work).
  - Q/E orbit, +/− zoom, `<main>` and a hidden `<h1>`.
  - The drawer tabs are one Tab stop with arrow keys.
- **Performance:**
  - On-demand rendering: idle draw calls fell from about 2,400/s to 0, and Furnish idles at 60 fps.
  - The canvas stays mounted across Home, walk-in and modes, cutting the worst Look long task from 503 ms to 190 ms.
  - No `preserveDrawingBuffer`; photos render and read in one frame.
  - Pixel density is capped at 1.5 on phones.
- **Fix found in the re-play:** the plain level-up card no longer stays up while the mouse rests on it (only the Dream card holds).

Rolled over:
- Font sizes in px (text doesn't scale with browser font settings).
- The 200% zoom layout on laptops.
- Every placed piece is a Tab stop.
- Wall decor doesn't re-snap when a room is reshaped.
- Curtains are a fixed width.
- The measure step can snap to a diagonal.

278 tests pass.

## Round 14 (2026-11-23 05:22-06:00 UTC)

Two implementers started on the round 13 leftovers right away. Alongside them, two reviewers ran: a gallery-wall decorator and a QA check of on-demand rendering. Two more implementers then fixed what those reviewers found, working on the merged code.

Shipped:
- **Wall decor follows the room:** `reconcileWallDecor` re-hangs each piece at the same spot along its wall after room, wall or opening edits. Curtains follow their window. Anything that no longer fits goes to Mine with a toast.
- **Curtains fit their window:** `curtainSpan` gives the window width plus 30 cm each side, clamped at wall ends and doors. The rod and folds scale with the width.
- **Seeing what you hung:**
  - Selecting or buying a wall piece on a cut-away wall turns the camera to face it.
  - A selected hidden piece shows as a ghost.
  - First load, Look and recenter pick the view that shows the most decor (`decoratedDir`).
  - A 3D drag no longer sends a frame to a hidden wall.
- **Gallery walls:** a frame dropped on another hangs above or below it (`Placement.mountY`), and frames only block frames at overlapping heights.
- **Art variety:** six motifs, landscape and tall prints, Large canvas keeps its blue, frame finishes (Black, White, Oak, Brass), tinted canvases in Clay, and an evening picture light.
- **Wall decor polish:** bigger plan badges on touch screens, drawn above furniture. "Placed a pair of curtains". On phones, Framed art opens a size picker.
- **Measuring:** taps prefer the wall over a corner, and a slanted line gets "That line isn't along a wall".
- **Text scales:** every font size is in rem, and the fixed 13px root that blocked the browser setting is gone. Pixel-identical at default; at 200% text the drawers grow instead of clipping.
- **Short screens:** laptops at 200% zoom (720x450) get the bottom-sheet drawer.
- **Rendering:**
  - The canvas redraws after a WebGL context restore and on pageshow.
  - Orbit damping gets its tail drawn.
  - A resize with no size change no longer reframes the camera.

Rolled over:
- 720x450 with both 200% zoom and 200% text still clips card names.
- An item used in two layouts shares one curtain width.
- The along-wall drag on a full wall barely moves.

307 tests pass.

## Round 15 (2026-11-23 11:58-12:55 UTC)

Reviewers:
- a fresh first-time player, start to finish
- a game-UX audit asking "does it still feel like a game, not a tool?"
- a returning player on small phones

The audit found the core loop lean, but about 17 systems visible from minute one. The fresh player reached Dream home after about 12 taps (~3 minutes), then found remixing felt like losing everything.

Shipped:
- **Declutter with unlocks:**
  - Desktop size chips appear on hover (Furnish 43 → 26 controls).
  - Settings is down to Units, plus "Scan in 3D" and Keyboard shortcuts (34 → 24 on phone, 47 → 28 on desktop).
  - The Classic editor is only reachable with `?classic`, and the Shop sheet loses its developer text.
  - Home management moved to "+ Add a home" in the Designs sheet.
  - Remix appears at Designer home, the Album after the first photo, the Mine tab once you own something, and Styles at Moving in. Blueprint left the game picker.
  - The locked Evening chip is gone.
  - Before Cozy, bonus goals fold into one row, and visitors and stickers show from Cozy, matching the Next chip.
  - The Details sheet is slimmer, and the "?" button moved into Settings.
  - Unlock lines appear on level-up cards.
- **Pacing:**
  - Dream home now also needs most of the bonus goals a pick can finish, with new small goals: art over seats and beds, living-room curtains, a bookshelf.
  - The sample takes about 20 Next steps (was 11) and the empty layout 24 (was 15), enforced by tests.
  - A walkway dead end found along the way is fixed.
- **Remix without the loss:** a remix shows its own progress ("Remix: 3 of 5", with the twist emoji) instead of a star drop. The picker says your pieces wait in Mine and the design keeps its stars, and your own pieces come first on pick cards.
- **Twists:** the Talk corner twist can be finished from the chip (`talkCornerSpot` in `pickSpot`). Rooms are matched for dining tables, beds and desks.
- **Teaching:** a one-time drag lesson on the 4th piece (a glowing target ring, then "Nice placement!"), and two-step nudges only for the goal the chip is showing.
- **After Dream home:** the chip says "Take its photo" until the design has a photo, then "Remix this home".
- **Phones:**
  - Landscape phones get the side drawer again (cards no longer fall under the dock).
  - Plan-view taps reach furniture before wall decor, and decor badges sit outside the walls.
  - A pinch on the HUD no longer zooms the page.
  - The design menu flips up on the SE, and rename inputs are 16px (no iOS zoom) and scroll into view.
  - The goals close button is sticky.
  - The welcome card clears the camera buttons.
  - Toasts use the full width and lead with the key word.
  - Card names fit on a short laptop at 200% text.

Rolled over:
- The Mine tab label is clipped ("Mi") on iPhone 14.
- Open-floor and fresh-spots remixes need a drag to finish.
- Nailing the drag lesson gives a sparkle but no star.
- Plan decor badges can overlap wall dimension labels.
- Older iOS can still pinch-zoom (no gesturestart guard).

314 tests pass.

## Round 16 (2026-11-23 18:56-20:20 UTC)

Playthrough: phone and desktop pass (only the known headless Scan-style flake). Reviewers: first-time phone player, cozy gamer (long desktop session with two remixes), interior designer and art director.

Shipped:
- **Round 15 leftovers:** phone buy tabs split the row evenly so "Mine" fits at 360-390 px; plan wall lengths move clear of decor badges (`wallLabelSpot`, `decorBadgeCenter`); iOS pinch guard (`src/gestureGuard.ts`); the Next chip can finish Open floor and Fresh start twists by moving a piece (`twistMove`); the drag lesson earns a one-time "Drag like a pro" star (`perks`, `awardDragStar`).
- **Next chip never loops or costs stars:** a pick goal that a placement doesn't advance is skipped for the design; nightstands only go beside the pillows (`bedsidePick`); `keepStars`/`starMove` avoid star-losing spots; the drag lesson's spot can't pinch a walkway and the loss pill hides during the lesson; a tapped chip comes back after 8 s or when the sheet closes.
- **Clear walkways is playable:** the goal points at the piece by the pinch, the chip selects it with paths and a pulsing ring (`PinchMarker`), and `goalMove` moves it to a clean spot when one exists.
- **Remixes feel different:** a goals-done tick on the remix pill, "✓ goal" toasts instead of "+1★", the twist leads, bonus order varies per remix, friendly twist copy without distances, and owned decor is reused instead of bought again.
- **Unlocks:** each unlock line shows once across homes; newly unlocked buttons glow with "New" until used; Remix copy points at the glowing designs button; Unpack keeps a label. After the Dream photo, the game returns to Furnish.
- **Room sense:** the sample's window moved so the TV sits on solid wall; a TV stand against a window is a warning and auto-placement avoids it; dining chairs face each other; art over a sofa or bed snaps to its centre, never behind the TV, with a Large canvas pick card; bathrooms are left out of "Every room has something".
- **Dream moments and photos:** the top-level celebration eases to the whole-home view; photos render through a camera fitted to the home.
- **Art pass:** round dusk stars, rugs toned to contrast with the floor with a darker border, smooth blurred scan floors in Cartoon and more contrast in Lifelike, dark TV screen in Clay, no lamp-shade blob, the sun patch fades under furniture.

Pacing: sample reaches Dream home in 19 Next steps, empty layout 23, open-floor remix 20. 342 tests pass.

Rolled over:
- Fresh start remix in the sample tops out at Designer home ("Make room for a desk" needs a move that costs a star).
- Remixes don't yet filter out goals the original already finished.
- Each chip move scores up to a dozen spots (0.3-2 s); time it on a real phone.
- Lifelike scan rug still faint (scan sampling mixes floor points under the rug).
- "Entrance" room label overlaps a wall length in the phone plan.
- Pacing still feels like a checklist: add a choice that matters every few goals (mood, style) or a hand-placed piece.
- Evening light unlock has no "New" glow.
