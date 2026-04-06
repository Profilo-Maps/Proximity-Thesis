# County Map Probe Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-panel click-to-probe widget to `_COUNTY_MAP_TEMPLATE` that shows nearby segments (streets, sidewalks, bikeways) and nearby points (intersection nodes, curb ramps) when the user clicks the map.

**Architecture:** All changes are edits to the `_COUNTY_MAP_TEMPLATE` Python string in `Implementations/test_maps.py` (lines 1247–1888). No new files. No changes to `generate_map`. The probe uses two JS Maps (`segIndex`, extended; `pointIndex`, new) populated during `renderRows()`, queried on map click via Haversine distance, and rendered into two stacked scrollable lists with Flash and Hide/Show per item.

**Tech Stack:** Python string editing (Edit tool), Leaflet.js (circleMarker/GeoJSON layer refs), vanilla JS.

---

## File Map

| File | Change |
|------|--------|
| `Implementations/test_maps.py` | All edits — CSS, HTML, JS inside `_COUNTY_MAP_TEMPLATE` |

No test file (HTML/JS widget, no Python logic added).

---

### Task 1: Add CSS for probe button and panel

**Files:**
- Modify: `Implementations/test_maps.py` (inside `_COUNTY_MAP_TEMPLATE`, `<style>` block)

The `<style>` block ends with the `#px-search` rule. Append two new rules directly before `</style>`.

- [ ] **Step 1: Locate the closing style tag anchor**

The exact text to find (line ~1283):
```
#px-search {
  position: fixed; top: 70px; right: 10px; z-index: 9999;
  background: white; padding: 10px 14px; border: 2px solid #aaa;
  border-radius: 6px; font-size: 13px; font-family: sans-serif; line-height: 1.6;
}
</style>
```

- [ ] **Step 2: Replace with the above plus new probe CSS rules**

```
#px-search {
  position: fixed; top: 70px; right: 10px; z-index: 9999;
  background: white; padding: 10px 14px; border: 2px solid #aaa;
  border-radius: 6px; font-size: 13px; font-family: sans-serif; line-height: 1.6;
}
#px-probe-toggle {
  position: fixed; top: 10px; left: 10px; z-index: 10001;
  padding: 5px 12px; border: 2px solid #aaa; border-radius: 4px; cursor: pointer;
  background: white; color: #333; font: 12px sans-serif;
  box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}
#px-probe {
  position: fixed; top: 42px; left: 10px; z-index: 10002;
  background: white; border: 2px solid #aaa; border-radius: 6px;
  font: 12px/1.5 sans-serif; color: #333; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
  max-width: 280px; min-width: 200px; display: none;
}
</style>
```

- [ ] **Step 3: Verify the file still parses (no Python syntax errors)**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 2: Add probe button and panel HTML

**Files:**
- Modify: `Implementations/test_maps.py` (inside `_COUNTY_MAP_TEMPLATE`, `<body>` block)

Insert after the `#px-coords` div and before `<div id="legend">`.

- [ ] **Step 1: Locate the anchor — the opening of the legend div**

Exact text to find:
```
<div id="legend">
  <b style="font-size:14px">Legend</b><br>
```

- [ ] **Step 2: Replace with probe button + panel HTML inserted before legend**

```
<button id="px-probe-toggle">&#128269; Probe</button>

<div id="px-probe">
  <div style="background:#1a73e8;color:white;padding:5px 10px;border-radius:4px 4px 0 0;
              display:flex;justify-content:space-between;align-items:center;">
    <b>Probe</b>
    <span id="px-probe-close" style="cursor:pointer;font-size:14px;line-height:1;">&times;</span>
  </div>
  <div style="padding:6px 10px;">
    <label style="font-size:11px;color:#666;">
      Radius:
      <input id="px-probe-radius" type="range" min="5" max="100" value="20"
             style="width:90px;vertical-align:middle;">
      <span id="px-probe-radius-val">20</span> m
    </label>
  </div>
  <div style="padding:0 8px 4px;">
    <b style="font-size:11px;color:#555;">Nearby Segments</b>
    <div id="px-probe-segs" style="max-height:200px;overflow-y:auto;margin-top:2px;"></div>
  </div>
  <hr style="margin:4px 8px;border:none;border-top:1px solid #ddd;">
  <div style="padding:0 8px 8px;">
    <b style="font-size:11px;color:#555;">Nearby Points</b>
    <div id="px-probe-pts" style="max-height:160px;overflow-y:auto;margin-top:2px;"></div>
  </div>
</div>

<div id="legend">
  <b style="font-size:14px">Legend</b><br>
```

- [ ] **Step 3: Verify no Python syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 3: Declare `pointIndex` Map and extend `clearAll()`

**Files:**
- Modify: `Implementations/test_maps.py` (inside `_COUNTY_MAP_TEMPLATE` `<script>` block)

- [ ] **Step 1: Locate the index declarations**

Exact text to find:
```
const segIndex  = new Map();   // street_grid_id / sidewalk_*_ID → { layer, mid, baseColor, baseWeight }
const nodeIndex = new Map();   // node_id (string) → { mid }
let   _pxMode   = 'seg';
```

- [ ] **Step 2: Add `pointIndex` declaration**

```
const segIndex   = new Map();   // street_grid_id / sidewalk_*_ID / bk_* → { layer, mid, baseColor, baseWeight, label }
const nodeIndex  = new Map();   // node_id (string) → { mid }
const pointIndex = new Map();   // node_* / ramp_* → { layer, lat, lon, label, baseColor, baseRadius }
let   _pxMode    = 'seg';
```

- [ ] **Step 3: Locate `clearAll()` and add `pointIndex.clear()`**

Exact text to find:
```
function clearAll() {
  ['streets','bk_sep','bk_off','sw_sep','sw_off','nodes','ramps','calm'].forEach(k => lg[k].clearLayers());
  segIndex.clear();
  nodeIndex.clear();
}
```

Replace with:
```
function clearAll() {
  ['streets','bk_sep','bk_off','sw_sep','sw_off','nodes','ramps','calm'].forEach(k => lg[k].clearLayers());
  segIndex.clear();
  nodeIndex.clear();
  pointIndex.clear();
}
```

- [ ] **Step 4: Verify no Python syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 4: Add `label` and `segIndex` entry for streets and sidewalks

**Files:**
- Modify: `Implementations/test_maps.py` — `renderRows()` street and sidewalk blocks

- [ ] **Step 1: Locate the street segIndex entry**

Exact text to find:
```
        const mid = midLatLon(g);
        segIndex.set(String(r.street_grid_id), { layer, mid, baseColor: col, baseWeight: 2 });
```

Replace with:
```
        const mid = midLatLon(g);
        segIndex.set(String(r.street_grid_id), { layer, mid, baseColor: col, baseWeight: 2, label: String(r.name || r.highway || r.street_grid_id) });
```

- [ ] **Step 2: Locate the sidewalk segIndex entry**

Exact text to find:
```
        const swId = String(r[`sidewalk_${side}_ID`] || '');
        if (swId) segIndex.set(swId, { layer, mid, baseColor: col, baseWeight: 2 });
```

Replace with:
```
        const swId = String(r[`sidewalk_${side}_ID`] || '');
        if (swId) segIndex.set(swId, { layer, mid, baseColor: col, baseWeight: 2, label: `Sidewalk ${side}: ${r.name || swId}` });
```

- [ ] **Step 3: Verify no Python syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 5: Add bikeways to `segIndex`

**Files:**
- Modify: `Implementations/test_maps.py` — bikeway render block inside `renderRows()`

- [ ] **Step 1: Locate the bikeway render block**

Exact text to find:
```
        layer.addTo(isOff ? lg.bk_off : lg.bk_sep);
      }
    }

    if (tier >= 3) {
```

- [ ] **Step 2: Replace — capture layer var and add segIndex entry before `addTo`**

Exact text to find (the full bikeway layer creation lines):
```
        const layer = L.geoJSON(g, { style: () => ({ color: col, weight: 2, opacity: 0.85 }) })
          .bindTooltip(`Bikeway ${side}-${idx}: ${r[`bikeway_${side}_${idx}_type`] || ''}`)
          .bindPopup(makePopup(`Bikeway (${side} ${idx})`, [
            ['Type',      r[`bikeway_${side}_${idx}_type`]],
            ['Offset',    r[`bikeway_${side}_${idx}_offset`]],
            ['Width (m)', r[`bikeway_${side}_${idx}_width`]],
            ['Surface',   r[`bikeway_${side}_${idx}_surface`]],
            ['Incline',   r[`bikeway_${side}_${idx}_incline`]],
            ['Permitted', r[`bikeway_${side}_${idx}_permitted`]],
            ['Separator', r[`bikeway_${side}_${idx}_seperator`]],
            ['Street',    r.name],
          ]));
        layer.addTo(isOff ? lg.bk_off : lg.bk_sep);
```

Replace with:
```
        const bkMid = midLatLon(g);
        const bkKey = `bk_${side}_${idx}_${r.street_grid_id}`;
        const layer = L.geoJSON(g, { style: () => ({ color: col, weight: 2, opacity: 0.85 }) })
          .bindTooltip(`Bikeway ${side}-${idx}: ${r[`bikeway_${side}_${idx}_type`] || ''}`)
          .bindPopup(makePopup(`Bikeway (${side} ${idx})`, [
            ['Type',      r[`bikeway_${side}_${idx}_type`]],
            ['Offset',    r[`bikeway_${side}_${idx}_offset`]],
            ['Width (m)', r[`bikeway_${side}_${idx}_width`]],
            ['Surface',   r[`bikeway_${side}_${idx}_surface`]],
            ['Incline',   r[`bikeway_${side}_${idx}_incline`]],
            ['Permitted', r[`bikeway_${side}_${idx}_permitted`]],
            ['Separator', r[`bikeway_${side}_${idx}_seperator`]],
            ['Street',    r.name],
          ]));
        segIndex.set(bkKey, { layer, mid: bkMid, baseColor: col, baseWeight: 2, label: `Bikeway ${side}-${idx}: ${r[`bikeway_${side}_${idx}_type`] || ''} (${r.name || ''})` });
        layer.addTo(isOff ? lg.bk_off : lg.bk_sep);
```

- [ ] **Step 3: Verify no Python syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 6: Capture node layer refs and populate `pointIndex`

**Files:**
- Modify: `Implementations/test_maps.py` — node render block inside `renderRows()` (tier ≥ 4)

- [ ] **Step 1: Locate the node render block**

Exact text to find:
```
        if (!nodeIndex.has(nid)) {
          L.circleMarker(mid, {
            radius: 4, color: C.node, fillColor: C.node, fillOpacity: 0.9, weight: 1,
          })
            .bindTooltip(`Node ${nid}`)
            .addTo(lg.nodes);
          nodeIndex.set(nid, { mid });
        }
```

- [ ] **Step 2: Replace — capture marker ref, add to both nodeIndex and pointIndex**

```
        if (!nodeIndex.has(nid)) {
          const nodeMk = L.circleMarker(mid, {
            radius: 4, color: C.node, fillColor: C.node, fillOpacity: 0.9, weight: 1,
          })
            .bindTooltip(`Node ${nid}`)
            .addTo(lg.nodes);
          nodeIndex.set(nid, { mid });
          pointIndex.set(`node_${nid}`, { layer: nodeMk, lat: mid[0], lon: mid[1], label: `Node ${nid}`, baseColor: C.node, baseRadius: 4 });
        }
```

- [ ] **Step 3: Verify no Python syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 7: Capture curb ramp layer refs and populate `pointIndex`

**Files:**
- Modify: `Implementations/test_maps.py` — curb ramp render block inside `renderRows()` (tier ≥ 5)

- [ ] **Step 1: Locate the curb ramp render block**

Exact text to find:
```
        L.circleMarker(mid, {
          radius: 5, color: C.ramp, fillColor: C.ramp, fillOpacity: 0.9, weight: 1,
        })
          .bindTooltip(`Curb ramp ${s}-${p}-${i}: loc=${rloc || ''}, pos=${rpos || ''}`)
          .bindPopup(makePopup(`Curb Ramp (${s} ${p} #${i})`, [
            ['ID',               r[`${base}_ID`]],
            ['Side',             s],
            ['Position',         p],
            ['Index',            i],
            ['Return direction', rloc],
            ['Return position',  rpos],
            ['Condition score',  r[`${base}_condition_score`]],
            ['Street',           r.name],
          ]))
          .addTo(lg.ramps);
```

- [ ] **Step 2: Replace — capture marker ref and add to pointIndex**

```
        const rampKey = `ramp_${s}_${p}_${i}_${r.street_grid_id}`;
        const rampMk = L.circleMarker(mid, {
          radius: 5, color: C.ramp, fillColor: C.ramp, fillOpacity: 0.9, weight: 1,
        })
          .bindTooltip(`Curb ramp ${s}-${p}-${i}: loc=${rloc || ''}, pos=${rpos || ''}`)
          .bindPopup(makePopup(`Curb Ramp (${s} ${p} #${i})`, [
            ['ID',               r[`${base}_ID`]],
            ['Side',             s],
            ['Position',         p],
            ['Index',            i],
            ['Return direction', rloc],
            ['Return position',  rpos],
            ['Condition score',  r[`${base}_condition_score`]],
            ['Street',           r.name],
          ]))
          .addTo(lg.ramps);
        pointIndex.set(rampKey, { layer: rampMk, lat: mid[0], lon: mid[1], label: `Ramp ${s}-${p}-${i} (${r.name || ''})`, baseColor: C.ramp, baseRadius: 5 });
```

- [ ] **Step 3: Verify no Python syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 8: Add probe logic JS block

**Files:**
- Modify: `Implementations/test_maps.py` — after legend toggle wiring, before search functions

- [ ] **Step 1: Locate the insertion point — between legend toggles and search**

Exact text to find:
```
document.getElementById('px-mode-seg').onclick  = () => pxSetMode('seg');
document.getElementById('px-mode-node').onclick = () => pxSetMode('node');
```

- [ ] **Step 2: Replace — insert probe logic block before the existing search wiring**

```
document.getElementById('px-mode-seg').onclick  = () => pxSetMode('seg');
document.getElementById('px-mode-node').onclick = () => pxSetMode('node');

// ── Probe ─────────────────────────────────────────────────────────────────────
let _probeActive = false;
const _probeTimerSeg = {};
const _probeTimerPt  = {};
const _hiddenSeg     = {};
const _hiddenPt      = {};

function haverDist(lat1, lon1, lat2, lon2) {
  const R = 6371000, r = Math.PI / 180;
  const dLat = (lat2 - lat1) * r, dLon = (lon2 - lon1) * r;
  const a = Math.sin(dLat/2)**2 + Math.cos(lat1*r)*Math.cos(lat2*r)*Math.sin(dLon/2)**2;
  return R * 2 * Math.asin(Math.sqrt(a));
}

function probeSegments(latlng, radiusM) {
  const found = [];
  segIndex.forEach((e, key) => {
    const d = haverDist(latlng.lat, latlng.lng, e.mid[0], e.mid[1]);
    if (d <= radiusM) found.push({ key, dist: d, entry: e });
  });
  found.sort((a, b) => a.dist - b.dist);
  return found;
}

function probePoints(latlng, radiusM) {
  const found = [];
  pointIndex.forEach((e, key) => {
    const d = haverDist(latlng.lat, latlng.lng, e.lat, e.lon);
    if (d <= radiusM) found.push({ key, dist: d, entry: e });
  });
  found.sort((a, b) => a.dist - b.dist);
  return found;
}

function flashFeature(key, kind) {
  if (kind === 'seg') {
    const e = segIndex.get(key); if (!e || !e.layer) return;
    if (_probeTimerSeg[key]) clearTimeout(_probeTimerSeg[key]);
    try { e.layer.setStyle({ color: '#ff4400', weight: e.baseWeight * 3 }); } catch(_) {}
    _probeTimerSeg[key] = setTimeout(() => {
      try { e.layer.setStyle({ color: e.baseColor, weight: e.baseWeight }); } catch(_) {}
    }, 2500);
  } else {
    const e = pointIndex.get(key); if (!e || !e.layer) return;
    if (_probeTimerPt[key]) clearTimeout(_probeTimerPt[key]);
    try { e.layer.setStyle({ color: '#ff4400', fillColor: '#ff4400', radius: e.baseRadius * 2.5 }); } catch(_) {}
    _probeTimerPt[key] = setTimeout(() => {
      try { e.layer.setStyle({ color: e.baseColor, fillColor: e.baseColor, radius: e.baseRadius }); } catch(_) {}
    }, 2500);
  }
}

function toggleHide(key, kind) {
  const btn = document.getElementById('px-hide-' + kind + '-' + key.replace(/[^a-zA-Z0-9_]/g, '_'));
  if (kind === 'seg') {
    const e = segIndex.get(key); if (!e || !e.layer) return;
    if (_hiddenSeg[key]) {
      try { e.layer.setStyle({ color: e.baseColor, weight: e.baseWeight, opacity: 0.8 }); } catch(_) {}
      delete _hiddenSeg[key];
      if (btn) btn.textContent = 'Hide';
    } else {
      try { e.layer.setStyle({ opacity: 0, weight: 0 }); } catch(_) {}
      _hiddenSeg[key] = true;
      if (btn) btn.textContent = 'Show';
    }
  } else {
    const e = pointIndex.get(key); if (!e || !e.layer) return;
    if (_hiddenPt[key]) {
      try { e.layer.setStyle({ color: e.baseColor, fillColor: e.baseColor, fillOpacity: 0.9, opacity: 1, radius: e.baseRadius }); } catch(_) {}
      delete _hiddenPt[key];
      if (btn) btn.textContent = 'Hide';
    } else {
      try { e.layer.setStyle({ opacity: 0, fillOpacity: 0 }); } catch(_) {}
      _hiddenPt[key] = true;
      if (btn) btn.textContent = 'Show';
    }
  }
}

function _probeRow(key, dist, entry, kind) {
  const safeKey = key.replace(/[^a-zA-Z0-9_]/g, '_');
  const isHidden = kind === 'seg' ? !!_hiddenSeg[key] : !!_hiddenPt[key];
  return `<div style="display:flex;align-items:center;gap:4px;padding:3px 2px;border-bottom:1px solid #f0f0f0;font-size:11px;">` +
    `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;flex-shrink:0;background:${entry.baseColor || entry.color || '#888'};"></span>` +
    `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${entry.label || key}">${entry.label || key}</span>` +
    `<span style="color:#888;flex-shrink:0;">${Math.round(dist)}m</span>` +
    `<button onclick="flashFeature('${key}','${kind}')" style="padding:1px 5px;font-size:10px;cursor:pointer;border:1px solid #ccc;border-radius:3px;">Flash</button>` +
    `<button id="px-hide-${kind}-${safeKey}" onclick="toggleHide('${key}','${kind}')" style="padding:1px 5px;font-size:10px;cursor:pointer;border:1px solid #ccc;border-radius:3px;">${isHidden ? 'Show' : 'Hide'}</button>` +
    `</div>`;
}

function renderProbe(segResults, ptResults) {
  const panel    = document.getElementById('px-probe');
  const segList  = document.getElementById('px-probe-segs');
  const ptList   = document.getElementById('px-probe-pts');
  segList.innerHTML = segResults.length
    ? segResults.map(f => _probeRow(f.key, f.dist, f.entry, 'seg')).join('')
    : '<div style="color:#999;font-size:11px;padding:4px 2px;">No segments found</div>';
  ptList.innerHTML  = ptResults.length
    ? ptResults.map(f => _probeRow(f.key, f.dist, f.entry, 'pt')).join('')
    : '<div style="color:#999;font-size:11px;padding:4px 2px;">No points found</div>';
  panel.style.display = 'block';
}

function _setProbeActive(active) {
  _probeActive = active;
  const btn = document.getElementById('px-probe-toggle');
  btn.style.background  = active ? '#1a73e8' : 'white';
  btn.style.color       = active ? 'white'   : '#333';
  btn.style.borderColor = active ? '#1a73e8' : '#aaa';
  btn.textContent       = active ? '\u{1F50D} Probe ON \u2014 click map' : '\u{1F50D} Probe';
  if (!active) document.getElementById('px-probe').style.display = 'none';
}

document.getElementById('px-probe-toggle').addEventListener('click', e => {
  e.stopPropagation();
  _setProbeActive(!_probeActive);
});

document.getElementById('px-probe-close').addEventListener('click', () => _setProbeActive(false));

const _probeRadiusInput = document.getElementById('px-probe-radius');
const _probeRadiusVal   = document.getElementById('px-probe-radius-val');
_probeRadiusInput.addEventListener('input', () => { _probeRadiusVal.textContent = _probeRadiusInput.value; });
```

- [ ] **Step 3: Verify no Python syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 9: Wire probe into the map click handler

**Files:**
- Modify: `Implementations/test_maps.py` — existing `map.on('click', ...)` handler

- [ ] **Step 1: Locate the existing click handler**

Exact text to find:
```
const _coordBox = document.getElementById('px-coords');
map.on('click', e => {
  _coordBox.style.display = 'block';
  _coordBox.textContent = e.latlng.lat.toFixed(7) + ', ' + e.latlng.lng.toFixed(7);
});
```

- [ ] **Step 2: Replace — add probe branch inside the same handler**

```
const _coordBox = document.getElementById('px-coords');
map.on('click', e => {
  _coordBox.style.display = 'block';
  _coordBox.textContent = e.latlng.lat.toFixed(7) + ', ' + e.latlng.lng.toFixed(7);
  if (_probeActive) {
    const r = parseInt(_probeRadiusInput.value, 10);
    renderProbe(probeSegments(e.latlng, r), probePoints(e.latlng, r));
  }
});
```

> **Note:** `_probeActive`, `_probeRadiusInput`, `probeSegments`, `probePoints`, and `renderProbe` are all declared in the probe logic block added in Task 8. Because the probe logic block is inserted *before* the click handler in the JS, these names are in scope. ✓

- [ ] **Step 3: Verify no Python syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('Implementations/test_maps.py').read()); print('OK')`
Expected: `OK`

---

### Task 10: Regenerate county maps and smoke-test

**Files:**
- Read-only verify

- [ ] **Step 1: Run the map generator**

Run: `uv run python Implementations/test_maps.py`
Expected: No Python errors. Output includes lines like:
```
County map saved -> Output/test_maps/San_Francisco_County_California_USA_county_map.html
County map saved -> Output/test_maps/Alameda_County_California_USA_county_map.html
```

- [ ] **Step 2: Serve and open the SF county map**

Run in a separate terminal: `cd Output && python -m http.server 8080 --bind 127.0.0.1`
Open: `http://localhost:8080/test_maps/San_Francisco_County_California_USA_county_map.html`

- [ ] **Step 3: Verify probe button appears**

Expected: "🔍 Probe" button visible at top-left. Clicking it turns it blue and changes text to "🔍 Probe ON — click map".

- [ ] **Step 4: Verify probe panel at zoom 15+ (tier 3)**

Zoom to 15+, wait for DuckDB to load data. Click the map. Expected: probe panel appears with "Nearby Segments" list (streets, sidewalks) and "Nearby Points" list (may be empty below z17).

- [ ] **Step 5: Verify probe panel at zoom 18+ (tier 5)**

Zoom to 18+. Click a street intersection. Expected: both lists populate. Flash button highlights the feature briefly. Hide button removes it from view.

- [ ] **Step 6: Verify bikeways appear in segments list**

In an area with bikeways (e.g. SF Market St), zoom to 15+, click near a bikeway. Expected: a "Bikeway left-1:..." or similar entry in the segments list.

- [ ] **Step 7: Commit**

```bash
git add Implementations/test_maps.py docs/superpowers/specs/2026-03-25-county-map-probe-design.md docs/superpowers/plans/2026-03-25-county-map-probe.md
git commit -m "feat: add two-panel click-to-probe widget to county map"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Probe toggle button (top-left, blue when active) | Tasks 1, 2, 8 |
| Single panel, two stacked sections (Segments / Points) | Tasks 2, 8 |
| Shared radius slider | Tasks 2, 8 |
| `segIndex` extended with bikeways | Task 5 |
| `label` added to street/sidewalk segIndex entries | Task 4 |
| `pointIndex` declared + cleared | Task 3 |
| Nodes → `pointIndex` with layer ref | Task 6 |
| Curb ramps → `pointIndex` with layer ref | Task 7 |
| Haversine probe for both indices | Task 8 |
| Flash: segments (color+weight) | Task 8 |
| Flash: points (color+radius) | Task 8 |
| Hide/Show: segments | Task 8 |
| Hide/Show: points | Task 8 |
| Probe fires on map click (coords unaffected) | Task 9 |
| Close button deactivates probe | Task 8 |
| `nodeIndex` untouched | Tasks 3, 6 ✓ |

**Placeholder scan:** No TBDs. All code blocks complete.

**Type consistency:**
- `segIndex` value shape `{ layer, mid, baseColor, baseWeight, label }` — used consistently in Tasks 4, 5, 8.
- `pointIndex` value shape `{ layer, lat, lon, label, baseColor, baseRadius }` — used consistently in Tasks 6, 7, 8.
- `flashFeature(key, 'seg'|'pt')` — defined Task 8, called in `_probeRow` Task 8. ✓
- `toggleHide(key, 'seg'|'pt')` — defined Task 8, called in `_probeRow` Task 8. ✓
- `_probeActive`, `_probeRadiusInput`, `probeSegments`, `probePoints`, `renderProbe` — all defined Task 8, referenced Task 9. Task 8 runs before Task 9 in execution order. ✓

**One ordering note:** Task 9 wires the click handler which references names defined in Task 8's JS block. In the template, the probe logic block (Task 8) is inserted *before* the `map.on('click', ...)` line, so JS hoisting/declaration order is correct.
