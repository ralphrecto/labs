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

**Backlog (next rounds)**
- Overview camera now frames the home too small (lots of empty margin on desktop and phone after
  the panel-aware framing); tighten the fit. Windows are flat grey panels; walk-in has no wall
  collision.
- Measuring lines stay drawn on the room review plan.
- Placement juice: drop animation, puff, place/remove sounds.
- Sparser sample home so there's room to grow; "+stars" pop on each placement.
- Catalog: size presets for sofa, dining table, rug, desk; ~8 more items (loveseat, sectional,
  round table, bar stools, mirror, art, curtains, shelves, pendant) and a Kitchen & entry tab.
- Furnishing several pieces in a row: keep drawer open or "Place another"; Rotate 90° per tap;
  room kits (bedroom set).
- More design goals: headboard on a solid wall, not under a window; rug size vs sofa; TV
  viewing distance; pair of nightstands for queen and up.
- Walk-in "Snap a photo" moment.
