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
