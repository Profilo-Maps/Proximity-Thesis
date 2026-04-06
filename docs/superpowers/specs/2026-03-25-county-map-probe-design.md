# County Map Probe Feature — Design Spec
**Date:** 2026-03-25
**Branch:** feat/intersection-analysis

---

## Overview

Add a click-to-probe widget to `_COUNTY_MAP_TEMPLATE` in `Implementations/test_maps.py`. When active, clicking the map shows two stacked panels: **Nearby Segments** (streets, sidewalks, bikeways) and **Nearby Points** (intersection nodes, curb ramps), each listing features within a configurable radius sorted by distance.

---

## Data Indices

### `segIndex` (existing — extend)
Map keyed by feature ID string → `{ layer, mid: [lat, lon], baseColor, baseWeight, label }`.

Current entries: streets (`street_grid_id`), sidewalks (`sidewalk_{side}_ID`).

**New entries — bikeways:** keyed as `bk_{side}_{n}_{street_grid_id}` (synthetic, no own ID column).
- Populated at tier ≥ 2 alongside the existing bikeway render block.
- `label`: `"Bikeway {side}-{n}: {type}"`.

**Add `label` field to existing street and sidewalk entries** for display in the probe list.

### `pointIndex` (new)
Map keyed by feature ID string → `{ layer, lat, lon, label, baseColor, baseRadius }`.

Entries:
- **Intersection nodes** — key `node_{nid}`, populated at tier ≥ 4. `baseColor = C.node`, `baseRadius = 4`.
- **Curb ramps** — key `ramp_{s}_{p}_{i}_{street_grid_id}`, populated at tier ≥ 5. `baseColor = C.ramp`, `baseRadius = 5`.

`pointIndex` is cleared in `clearAll()` alongside `segIndex` and `nodeIndex`.

> **Note:** `nodeIndex` (used by the existing search widget) keeps its current shape `{ mid }` and is untouched. `pointIndex` is independent.

---

## UI Components

### Toggle button
Fixed position, top-left (`top: 10px; left: 10px`), z-index 10001. Matches test_maps style.

- Default: white background, "🔍 Probe".
- Active: `#1a73e8` background, white text, "🔍 Probe ON — click map".

### Probe panel
Single panel (`id="px-probe"`) appears below the button when probe is active. Contains:

1. **Header bar** — dark blue (`#1a73e8`), title "Probe", close button (×) on right.
2. **Radius row** — `<input type="range">` 5–100 m, default 20 m, live label.
3. **Segments section** — heading "Nearby Segments", scrollable list (`max-height: 200px`).
4. **Divider** (`<hr>`).
5. **Points section** — heading "Nearby Points", scrollable list (`max-height: 160px`).

Panel is `display: none` until probe activates. `max-width: 280px; min-width: 200px`.

### List items (both panels)
Each result row:
```
[color swatch] [label]  [dist m]  [Flash] [Hide/Show]
```
- Color swatch: 10×10px inline block, `background: baseColor`.
- Clicking **Flash** calls `flashFeature(key, 'seg'|'pt')`.
- Clicking **Hide/Show** calls `toggleHide(key, 'seg'|'pt')`.
- "No features found" placeholder when list is empty.

---

## Behavior

### Probe activation
Clicking the toggle button flips `_probeActive`. Panel visibility follows. Close button (×) deactivates probe and hides panel.

### Click handler
Appended to the existing `map.on('click', ...)` handler (coordinates display is unaffected — both run on every click). When `_probeActive`:
1. Read radius slider value.
2. Run `probeAt(latlng, radius)` against `segIndex` → segment results.
3. Run `probeAt(latlng, radius)` against `pointIndex` (using `lat`/`lon`) → point results.
4. Both sorted by distance ascending.
5. Call `renderProbe(segResults, ptResults)` to populate the panel and make it visible.

### `haverDist(lat1, lon1, lat2, lon2)`
Same Haversine formula as `test_maps.py` `generate_map`. Reused for both indices.

### Flash
- **Segments:** `layer.setStyle({ color: '#ff4400', weight: baseWeight * 3 })` → restored after 2500 ms via per-key timer.
- **Points:** `layer.setStyle({ color: '#ff4400', fillColor: '#ff4400', radius: baseRadius * 2.5 })` → restored after 2500 ms.
- Per-key timers stored in `_probeTimerSeg` and `_probeTimerPt` objects (same pattern as `_probeTimer` in `generate_map`).

### Hide/Show
- **Segments:** hidden → `layer.setStyle({ opacity: 0, weight: 0 })`; shown → restore `baseColor`/`baseWeight`.
- **Points:** hidden → `layer.setStyle({ opacity: 0, fillOpacity: 0 })`; shown → restore `baseColor`/`baseRadius`.
- Hidden state tracked in `_hiddenSeg` and `_hiddenPt` objects.
- Button label toggles "Hide" ↔ "Show".

---

## Implementation Scope

All changes are within `_COUNTY_MAP_TEMPLATE` (the Python string) and the `renderRows()` JS function inside it:

1. **CSS** — add `#px-probe` and `#px-probe-toggle` styles inside `<style>`.
2. **HTML** — add probe button and panel `<div>`s in `<body>` before `<script>`.
3. **JS — index population** — in `renderRows()`:
   - Add bikeways to `segIndex`.
   - Add `label` to existing street/sidewalk `segIndex` entries.
   - Populate `pointIndex` for nodes (tier ≥ 4) and ramps (tier ≥ 5); store `layer` ref.
4. **JS — `clearAll()`** — add `pointIndex.clear()`.
5. **JS — probe logic** — add after legend toggle wiring:
   - `haverDist`, `probeAt`, `renderProbe`, `flashFeature`, `toggleHide`.
   - Toggle button click handler, close button handler.
   - Append probe call inside `map.on('click', ...)`.

No changes to Python outside `_COUNTY_MAP_TEMPLATE`. No new files. No changes to `generate_map`.
