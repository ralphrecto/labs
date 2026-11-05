# Branchpoint

A prototype "choose your own adventure" simulator for alternate worlds.

- **Left:** the timeline tree. Every simulated moment is a node; untaken branches of the current moment appear as dashed rows.
- **Center:** the current moment: what happened, a divergence-from-our-history meter, and 3 choices (or type your own intervention).
- **Right:** "dispatches" from that world, rendered as fake front pages, news-wire headlines, social posts, TV chyrons, leaked memos and market boards.

## Run it

```bash
./build.sh
```

Then open `index.html` in a browser. A hand-written demo world (a cheap room-temperature superconductor in 2027) works offline. To simulate new branches or new worlds, click the engine button (top right) and paste an Anthropic API key. It is stored in `localStorage` and sent only to `api.anthropic.com`.

When published as a claude.ai Artifact, the page uses the viewer's own Claude account (the `sample` capability) instead of an API key.

## Files

- `app.html`: the app (a page fragment; this is also what gets published as the Artifact)
- `build.sh`: wraps `app.html` into a standalone `index.html`

## How generation works

Each branch is one model call. The prompt contains the premise, a compressed history of the path from the root, and the decision just taken. It asks for a single JSON object: title, date, divergence (0 to 100), summary paragraphs, 3 next choices, and 3 or 4 typed artifacts. All model output is rendered as text (never as HTML), and fictional outlets and people are required for quotes and posts.

## Ideas for later

- Stream artifacts in as they're written instead of waiting for the whole node
- Image generation for front-page photos
- Share or export a timeline; compare two branches side by side
- Persistent "world state" variables (population, GDP, tech level) that each step updates
