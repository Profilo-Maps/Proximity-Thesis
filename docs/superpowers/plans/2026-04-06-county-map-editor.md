# County Map Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive editor on top of the county map viewer that lets users plot nodes, consolidate hulls, toggle curb ramps, draw crosswalks, and move endpoints — with edits saved to a local Python server that re-runs pipeline stages on affected rows.

**Architecture:** Two new files — a FastAPI server (`Implementations/editor_server.py`) that imports pipeline functions from `ProximityModel.py` and serves a standalone Leaflet+DuckDB WASM editor HTML (`Output/test_maps/county_editor.html`). Edits accumulate in-browser localStorage; on Save, a POST sends the changeset to the server which applies edits and re-runs pipeline stages from the earliest dirty stage onward, scoped to a bounding box.

**Tech Stack:** Python 3.14, FastAPI, uvicorn, geopandas, shapely — frontend: Leaflet 1.9.4, DuckDB WASM 1.29.0, proj4js 2.11.0

**Spec:** `docs/superpowers/specs/2026-04-06-county-map-editor-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `Implementations/editor_server.py` | Create | FastAPI app: serves editor HTML + parquet files, POST `/save` endpoint, in-memory GeoDataFrame cache, bbox-scoped pipeline re-run |
| `Output/test_maps/county_editor.html` | Create | Full editor frontend: inherits county map viewer pattern (DuckDB WASM + Leaflet), adds toolbar, context menus, changeset panel, edit modes, localStorage persistence |

---

### Task 1: FastAPI Server — Static Serving and Parquet Cache

**Files:**
- Create: `Implementations/editor_server.py`

- [ ] **Step 1: Create the server file with imports, app, and parquet cache**

```python
"""County Map Editor — local server.

Serves the editor HTML and parquet files, accepts edit changesets via
POST /save, and re-runs pipeline stages on affected rows.

Run:
    cd Implementations && uv run uvicorn editor_server:app --reload --port 8081
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from shapely.geometry import Point, box

# Ensure Implementations/ is on sys.path so we can import ProximityModel
_IMPL_DIR = Path(__file__).resolve().parent
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

from ProximityModel import (
    _build_intersection_hulls,
    _assign_curb_ramp_geometries,
    _create_crosswalk_geometries,
    _snap_offset_endpoints,
    _CURB_RAMP_PROXIMITY_M,
)

OUTPUT_DIR = _IMPL_DIR.parent / "Output"
EDITOR_HTML = OUTPUT_DIR / "test_maps" / "county_editor.html"

app = FastAPI(title="Proximity Editor")

# ── In-memory parquet cache ──────────────────────────────────────────────────
_gdf_cache: dict[str, gpd.GeoDataFrame] = {}


def _load_gdf(parquet_name: str) -> gpd.GeoDataFrame:
    """Load a parquet into the cache (or return cached copy)."""
    if parquet_name not in _gdf_cache:
        path = OUTPUT_DIR / parquet_name
        if not path.exists():
            raise HTTPException(404, f"Parquet not found: {parquet_name}")
        _gdf_cache[parquet_name] = gpd.read_parquet(path)
    return _gdf_cache[parquet_name]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_editor():
    if not EDITOR_HTML.exists():
        raise HTTPException(404, "county_editor.html not found")
    return HTMLResponse(EDITOR_HTML.read_text(encoding="utf-8"))


@app.get("/parquet/{filename}")
async def serve_parquet(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(path, media_type="application/octet-stream")
```

- [ ] **Step 2: Verify the server starts**

Run from project root:
```bash
uv run uvicorn Implementations.editor_server:app --port 8081
```
Expected: server starts, `GET /` returns 404 (editor HTML doesn't exist yet), `GET /parquet/San_Francisco_County_California_USA_network.parquet` returns the file.

- [ ] **Step 3: Commit**

```bash
git add Implementations/editor_server.py
git commit -m "feat(editor): scaffold FastAPI server with parquet cache and static serving"
```

---

### Task 2: Server — Save Endpoint with Bbox Filtering and Edit Application

**Files:**
- Modify: `Implementations/editor_server.py`

- [ ] **Step 1: Add Pydantic models for the save request/response**

Append below the cache section in `editor_server.py`:

```python
# ── Pipeline stage definitions ────────────────────────────────────────────────
STAGE_SNAP_ENDPOINTS  = 0
STAGE_BUILD_HULLS     = 1
STAGE_ASSIGN_RAMPS    = 2
STAGE_CREATE_XWALKS   = 3

STAGE_NAMES = {
    STAGE_SNAP_ENDPOINTS: "SNAP_ENDPOINTS",
    STAGE_BUILD_HULLS:    "BUILD_HULLS",
    STAGE_ASSIGN_RAMPS:   "ASSIGN_CURB_RAMPS",
    STAGE_CREATE_XWALKS:  "CREATE_CROSSWALKS",
}


# ── Request / response models ────────────────────────────────────────────────
class BBox(BaseModel):
    minX: float
    minY: float
    maxX: float
    maxY: float


class AddedNode(BaseModel):
    x: float
    y: float


class ConsolidatedHull(BaseModel):
    node_keys: list[list[float]]  # [[x1,y1], [x2,y2], ...]
    buffer_m: float | None = None  # None = auto-compute from gap distance


class MovedEndpoint(BaseModel):
    node_id: str
    new_x: float
    new_y: float
    rubber_band_segments: list[str]  # street_grid_id values


class ToggledCurbRamp(BaseModel):
    segment_id: str
    side: str       # "left" | "right"
    position: str   # "start" | "end"
    index: int      # 1, 2, or 3
    enabled: bool


class RampRef(BaseModel):
    segment_id: str
    side: str
    position: str
    index: int


class DrawnCrosswalk(BaseModel):
    ramp_a: RampRef
    ramp_b: RampRef


class Edits(BaseModel):
    added_nodes: list[AddedNode] = []
    consolidated_hulls: list[ConsolidatedHull] = []
    moved_endpoints: list[MovedEndpoint] = []
    toggled_curb_ramps: list[ToggledCurbRamp] = []
    drawn_crosswalks: list[DrawnCrosswalk] = []


class SaveRequest(BaseModel):
    parquet: str
    dirty_from_stage: int
    bbox: BBox
    edits: Edits


class SaveResponse(BaseModel):
    status: str
    stages_run: list[int]
    rows_affected: int
    bbox_used: BBox
```

- [ ] **Step 2: Add bbox filtering helper**

```python
def _filter_bbox(gdf: gpd.GeoDataFrame, bbox: BBox, buffer_m: float = 0.0) -> gpd.GeoIndex:
    """Return index labels of rows whose street_geometry intersects the buffered bbox."""
    search_box = box(
        bbox.minX - buffer_m, bbox.minY - buffer_m,
        bbox.maxX + buffer_m, bbox.maxY + buffer_m,
    )
    geom_col = gdf.geometry if gdf.geometry.name == "street_geometry" else gdf["street_geometry"]
    mask = geom_col.intersects(search_box)
    return gdf.index[mask]
```

- [ ] **Step 3: Add edit application functions**

```python
def _apply_moved_endpoints(gdf: gpd.GeoDataFrame, edits: list[MovedEndpoint]) -> None:
    """Move node positions and rubber-band connected segment geometries in-place."""
    for me in edits:
        node_id = me.node_id
        new_pt = Point(me.new_x, me.new_y)
        # Update node geometry on rows that reference this node
        for prefix in ("start", "end"):
            id_col = f"{prefix}_node_id"
            geom_col = f"{prefix}_node_geometry"
            if id_col not in gdf.columns or geom_col not in gdf.columns:
                continue
            node_mask = gdf[id_col].astype(str) == str(node_id)
            if not node_mask.any():
                continue
            gdf.loc[node_mask, geom_col] = new_pt
            # Rubber-band street_geometry for opted-in segments
            for idx in gdf.index[node_mask]:
                seg_id = str(gdf.at[idx, "street_grid_id"])
                if seg_id not in me.rubber_band_segments:
                    continue
                street_geom = gdf.at[idx, "street_geometry"]
                if street_geom is None or street_geom.is_empty:
                    continue
                coords = list(street_geom.coords)
                if prefix == "start":
                    coords[0] = (me.new_x, me.new_y)
                else:
                    coords[-1] = (me.new_x, me.new_y)
                from shapely.geometry import LineString
                gdf.at[idx, "street_geometry"] = LineString(coords)


def _apply_added_nodes(gdf: gpd.GeoDataFrame, patch_idx: pd.Index,
                       edits: list[AddedNode]) -> None:
    """Mark nearest segment endpoints as intersection nodes for added points."""
    if not edits:
        return
    for an in edits:
        placed = Point(an.x, an.y)
        best_dist = float("inf")
        best_idx = None
        best_prefix = None
        for idx in patch_idx:
            for prefix in ("start", "end"):
                geom_col = f"{prefix}_node_geometry"
                if geom_col not in gdf.columns:
                    continue
                pt = gdf.at[idx, geom_col]
                if pt is None or (hasattr(pt, "is_empty") and pt.is_empty):
                    continue
                d = placed.distance(pt)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
                    best_prefix = prefix
        if best_idx is not None and best_dist < _CURB_RAMP_PROXIMITY_M * 2:
            gdf.at[best_idx, f"{best_prefix}_node_is_intersection_node"] = True


def _apply_toggled_ramps(gdf: gpd.GeoDataFrame, edits: list[ToggledCurbRamp]) -> None:
    """Enable/disable curb ramps by setting/clearing geometry columns."""
    for tr in edits:
        mask = gdf["street_grid_id"].astype(str) == str(tr.segment_id)
        if not mask.any():
            continue
        base = f"sidewalk_{tr.side}_curbramp_{tr.position}_{tr.index}"
        geom_col = f"{base}_geometry"
        if geom_col not in gdf.columns:
            continue
        if not tr.enabled:
            gdf.loc[mask, geom_col] = None
        # If enabling, the pipeline re-run (stage 2) will re-detect the ramp


def _apply_drawn_crosswalks(gdf: gpd.GeoDataFrame, edits: list[DrawnCrosswalk]) -> None:
    """Insert straight-line crosswalk geometries between two curb ramps."""
    for dc in edits:
        # Locate ramp A geometry
        a_mask = gdf["street_grid_id"].astype(str) == str(dc.ramp_a.segment_id)
        b_mask = gdf["street_grid_id"].astype(str) == str(dc.ramp_b.segment_id)
        if not a_mask.any() or not b_mask.any():
            continue
        a_idx = gdf.index[a_mask][0]
        b_idx = gdf.index[b_mask][0]
        a_geom_col = f"sidewalk_{dc.ramp_a.side}_curbramp_{dc.ramp_a.position}_{dc.ramp_a.index}_geometry"
        b_geom_col = f"sidewalk_{dc.ramp_b.side}_curbramp_{dc.ramp_b.position}_{dc.ramp_b.index}_geometry"
        a_pt = gdf.at[a_idx, a_geom_col] if a_geom_col in gdf.columns else None
        b_pt = gdf.at[b_idx, b_geom_col] if b_geom_col in gdf.columns else None
        if a_pt is None or b_pt is None:
            continue
        from shapely.geometry import LineString
        xw_line = LineString([a_pt, b_pt])
        # Store on ramp_b's segment (the crossed street) at the appropriate end
        pos = dc.ramp_b.position  # "start" or "end"
        xw_geom_col = f"crosswalk_{pos}_geometry"
        xw_type_col = f"crosswalk_{pos}_type"
        if xw_geom_col in gdf.columns:
            gdf.at[b_idx, xw_geom_col] = xw_line
        if xw_type_col in gdf.columns:
            gdf.at[b_idx, xw_type_col] = "manual"
```

- [ ] **Step 4: Add the POST /save endpoint**

```python
@app.post("/save", response_model=SaveResponse)
async def save_edits(req: SaveRequest):
    gdf = _load_gdf(req.parquet)
    buffer_m = _CURB_RAMP_PROXIMITY_M
    working_bbox = BBox(
        minX=req.bbox.minX - buffer_m,
        minY=req.bbox.minY - buffer_m,
        maxX=req.bbox.maxX + buffer_m,
        maxY=req.bbox.maxY + buffer_m,
    )
    patch_idx = _filter_bbox(gdf, req.bbox, buffer_m)
    if patch_idx.empty:
        return SaveResponse(status="ok", stages_run=[], rows_affected=0, bbox_used=working_bbox)

    # ── Apply direct edits ────────────────────────────────────────────────────
    _apply_moved_endpoints(gdf, req.edits.moved_endpoints)
    _apply_added_nodes(gdf, patch_idx, req.edits.added_nodes)
    _apply_toggled_ramps(gdf, req.edits.toggled_curb_ramps)
    _apply_drawn_crosswalks(gdf, req.edits.drawn_crosswalks)

    # ── Hull consolidation ────────────────────────────────────────────────────
    # (Applied before pipeline re-run; hulls are rebuilt in stage 1)
    # Consolidation merges are handled by _build_intersection_hulls after we
    # mark the relevant nodes — the actual merge happens in the hull stage.

    # ── Run pipeline from dirty_from_stage onward ─────────────────────────────
    patch_df = gdf.loc[patch_idx].copy()
    stages_run = []

    if req.dirty_from_stage <= STAGE_SNAP_ENDPOINTS:
        patch_df = _snap_offset_endpoints(patch_df)
        stages_run.append(STAGE_SNAP_ENDPOINTS)

    hulls_data = None
    if req.dirty_from_stage <= STAGE_BUILD_HULLS:
        patch_df, hulls_data = _build_intersection_hulls(patch_df)
        stages_run.append(STAGE_BUILD_HULLS)

    if req.dirty_from_stage <= STAGE_ASSIGN_RAMPS:
        patch_df = _assign_curb_ramp_geometries(
            patch_df,
            _precomputed_hulls=hulls_data,
        )
        stages_run.append(STAGE_ASSIGN_RAMPS)

    if req.dirty_from_stage <= STAGE_CREATE_XWALKS:
        patch_df, _ = _create_crosswalk_geometries(
            patch_df,
            _precomputed_hulls=hulls_data,
        )
        stages_run.append(STAGE_CREATE_XWALKS)

    # ── Patch back into full GeoDataFrame and flush to disk ───────────────────
    gdf.loc[patch_idx] = patch_df
    parquet_path = OUTPUT_DIR / req.parquet
    gdf.to_parquet(parquet_path)

    return SaveResponse(
        status="ok",
        stages_run=stages_run,
        rows_affected=len(patch_idx),
        bbox_used=working_bbox,
    )
```

- [ ] **Step 5: Verify the server starts with the new endpoint**

```bash
uv run uvicorn Implementations.editor_server:app --port 8081
```
Expected: starts without import errors. `POST /save` with an empty edits payload returns `{"status": "ok", ...}`.

- [ ] **Step 6: Commit**

```bash
git add Implementations/editor_server.py
git commit -m "feat(editor): add /save endpoint with bbox filtering and pipeline re-run"
```

---

### Task 3: Editor HTML — Base Map Viewer (Copy from County Map Pattern)

**Files:**
- Create: `Output/test_maps/county_editor.html`

This task creates the editor HTML with the full read-only viewer working (DuckDB WASM, Leaflet, all layers, probe, search) but pointed at the editor server instead of a static file server. The JS is a module script identical in structure to the county map template.

- [ ] **Step 1: Create county_editor.html with the base viewer**

Create `Output/test_maps/county_editor.html`. This file replicates the county map viewer from `_COUNTY_MAP_TEMPLATE` in `test_maps.py` (lines 1262–2083) with these changes:

1. **Parquet URL** points to `http://localhost:8081/parquet/San_Francisco_County_California_USA_network.parquet` (configurable via a `?parquet=` query param)
2. **Title** set to "Proximity Editor"
3. **Hull/slot GeoJSON** loaded from the server instead of embedded (initially empty `FeatureCollection`s — the server will provide these when we add hull endpoints later; for now the existing pre-computed hulls are visible via DuckDB queries)
4. **All viewer functionality** preserved: zoom tiers, legend toggles, search, probe, coord display

Key change in the `<script type="module">` block opening:

```javascript
// ── Config from query params ────────────────────────────────────────────────
const params = new URLSearchParams(window.location.search);
const PARQUET_FILE = params.get('parquet') || 'San_Francisco_County_California_USA_network.parquet';
const PARQUET_URL = `${window.location.origin}/parquet/${PARQUET_FILE}`;
const HULL_FC = { type: 'FeatureCollection', features: [] };
const SLOT_FC = { type: 'FeatureCollection', features: [] };
```

The rest of the viewer JS (proj4 transforms, `xfGeom`, `blendBlack`, `inclineColor`, colours, zoom tiers, `renderRows`, `buildSQL`, DuckDB init, legend toggles, search, probe) is copied verbatim from the county map template (lines 1426–2079 of `test_maps.py`).

- [ ] **Step 2: Verify the editor loads**

```bash
cd Implementations && uv run uvicorn editor_server:app --port 8081
```
Open `http://localhost:8081/` in a browser.
Expected: map loads, streets render on zoom, search and probe work.

- [ ] **Step 3: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): create base editor HTML with full viewer functionality"
```

---

### Task 4: Editor HTML — Changeset State and localStorage Persistence

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Add changeset state management at the top of the `<script type="module">` block, after the config section**

Insert after the `SLOT_FC` line:

```javascript
// ── Changeset state ─────────────────────────────────────────────────────────
// TODO: Implement undo/redo with a linear undo stack (ctrl+z / ctrl+y)
const STAGE_NAMES = ['SNAP_ENDPOINTS', 'BUILD_HULLS', 'ASSIGN_CURB_RAMPS', 'CREATE_CROSSWALKS'];
const EDIT_STAGE_MAP = {
  move_endpoint: 0, add_node: 1, consolidate_hulls: 1,
  toggle_curb_ramp: 2, draw_crosswalk: 3,
};

const _LS_KEY = 'px_editor_changeset';

function emptyChangeset() {
  return {
    parquet: PARQUET_FILE,
    dirty_from_stage: null,
    edits: {
      added_nodes: [],
      consolidated_hulls: [],
      moved_endpoints: [],
      toggled_curb_ramps: [],
      drawn_crosswalks: [],
    },
  };
}

function loadChangeset() {
  try {
    const raw = localStorage.getItem(_LS_KEY);
    if (!raw) return emptyChangeset();
    const cs = JSON.parse(raw);
    if (cs.parquet !== PARQUET_FILE) return emptyChangeset();
    return cs;
  } catch (_) { return emptyChangeset(); }
}

function saveChangeset(cs) {
  localStorage.setItem(_LS_KEY, JSON.stringify(cs));
  renderChangesetPanel(cs);
}

let changeset = loadChangeset();

function recordEdit(editType, editData) {
  const stage = EDIT_STAGE_MAP[editType];
  if (changeset.dirty_from_stage === null || stage < changeset.dirty_from_stage) {
    changeset.dirty_from_stage = stage;
  }
  const key = {
    move_endpoint: 'moved_endpoints',
    add_node: 'added_nodes',
    consolidate_hulls: 'consolidated_hulls',
    toggle_curb_ramp: 'toggled_curb_ramps',
    draw_crosswalk: 'drawn_crosswalks',
  }[editType];
  changeset.edits[key].push(editData);
  saveChangeset(changeset);
}

function discardChangeset() {
  changeset = emptyChangeset();
  localStorage.removeItem(_LS_KEY);
  renderChangesetPanel(changeset);
  // Clear edit visual layers
  lg.editor.clearLayers();
  setStatus('Changeset discarded');
}
```

- [ ] **Step 2: Add the editor layer group**

In the `lg` object (where `streets`, `bk_sep`, etc. are defined), add:

```javascript
  editor:  L.layerGroup().addTo(map),
```

- [ ] **Step 3: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): add changeset state management and localStorage persistence"
```

---

### Task 5: Editor HTML — Changeset Panel UI

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Add changeset panel HTML**

Add this markup before the `<script>` tag:

```html
<div id="px-changeset" style="
  position:fixed; bottom:30px; right:30px; z-index:9999;
  background:white; border:2px solid #aaa; border-radius:6px;
  font:12px/1.5 sans-serif; color:#333; box-shadow:0 2px 6px rgba(0,0,0,0.2);
  min-width:220px; max-width:300px;">
  <div style="background:#1a73e8;color:white;padding:5px 10px;border-radius:4px 4px 0 0;
              display:flex;justify-content:space-between;align-items:center;cursor:pointer;"
       onclick="document.getElementById('px-cs-body').style.display=
                document.getElementById('px-cs-body').style.display==='none'?'block':'none'">
    <b>Changeset</b>
    <span style="font-size:10px;">&#9660;</span>
  </div>
  <div id="px-cs-body" style="padding:8px 10px;">
    <div id="px-cs-counts" style="margin-bottom:6px;font-size:11px;color:#555;">No edits</div>
    <div id="px-cs-stage" style="margin-bottom:8px;font-size:11px;color:#888;"></div>
    <div style="display:flex;gap:6px;">
      <button id="px-cs-save" onclick="doSave()"
              style="flex:1;padding:4px 8px;background:#16a34a;color:white;border:none;
                     border-radius:4px;cursor:pointer;font-size:12px;">
        Save
      </button>
      <button id="px-cs-discard" onclick="discardChangeset()"
              style="flex:1;padding:4px 8px;background:#dc2626;color:white;border:none;
                     border-radius:4px;cursor:pointer;font-size:12px;">
        Discard
      </button>
    </div>
    <div id="px-cs-spinner" style="display:none;text-align:center;margin-top:6px;font-size:11px;color:#1a73e8;">
      Saving&hellip;
    </div>
  </div>
</div>
```

- [ ] **Step 2: Add the renderChangesetPanel and doSave functions**

```javascript
function renderChangesetPanel(cs) {
  const counts = document.getElementById('px-cs-counts');
  const stage  = document.getElementById('px-cs-stage');
  const e = cs.edits;
  const parts = [];
  if (e.added_nodes.length)        parts.push(`${e.added_nodes.length} node(s) added`);
  if (e.consolidated_hulls.length)  parts.push(`${e.consolidated_hulls.length} hull consolidation(s)`);
  if (e.moved_endpoints.length)     parts.push(`${e.moved_endpoints.length} endpoint(s) moved`);
  if (e.toggled_curb_ramps.length)  parts.push(`${e.toggled_curb_ramps.length} ramp(s) toggled`);
  if (e.drawn_crosswalks.length)    parts.push(`${e.drawn_crosswalks.length} crosswalk(s) drawn`);
  counts.textContent = parts.length ? parts.join(', ') : 'No edits';
  stage.textContent  = cs.dirty_from_stage !== null
    ? `Pipeline from: ${STAGE_NAMES[cs.dirty_from_stage]}`
    : '';
}

function computeEditBBox(cs) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  function expand(x, y) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  for (const n of cs.edits.added_nodes) expand(n.x, n.y);
  for (const m of cs.edits.moved_endpoints) expand(m.new_x, m.new_y);
  // For ramps/crosswalks, use the map center as fallback (they reference existing features)
  if (minX === Infinity) {
    const c = map.getCenter();
    const utm = proj4('EPSG:4326', 'EPSG:32610', [c.lng, c.lat]);
    expand(utm[0] - 100, utm[1] - 100);
    expand(utm[0] + 100, utm[1] + 100);
  }
  return { minX, minY, maxX, maxY };
}

async function doSave() {
  if (changeset.dirty_from_stage === null) {
    setStatus('No edits to save'); return;
  }
  const spinner = document.getElementById('px-cs-spinner');
  spinner.style.display = 'block';
  setStatus('Saving edits\u2026');
  try {
    const bbox = computeEditBBox(changeset);
    const resp = await fetch('/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        parquet: changeset.parquet,
        dirty_from_stage: changeset.dirty_from_stage,
        bbox,
        edits: changeset.edits,
      }),
    });
    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    const result = await resp.json();
    setStatus(`Saved: ${result.rows_affected} rows, stages ${result.stages_run.join(',')}`);
    // Clear changeset and refresh map
    changeset = emptyChangeset();
    localStorage.removeItem(_LS_KEY);
    renderChangesetPanel(changeset);
    lg.editor.clearLayers();
    await refreshMap();
    renderHullsAndSlots();
  } catch (e) {
    setStatus('Save failed: ' + e.message);
    console.error('Save error:', e);
  } finally {
    spinner.style.display = 'none';
  }
}

// Initial render
renderChangesetPanel(changeset);
```

- [ ] **Step 3: Move the px-coords div from bottom-right to avoid overlap with changeset panel**

Change the `px-coords` div positioning from `right:30px` to `right:340px` (or position it at bottom-center).

- [ ] **Step 4: Verify save flow**

Open editor, confirm the changeset panel shows "No edits". We'll test actual save with real edits in later tasks.

- [ ] **Step 5: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): add changeset panel with save/discard buttons"
```

---

### Task 6: Editor HTML — Toolbar and Editor Mode Switching

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Add toolbar HTML**

Add after the `px-probe-toggle` button:

```html
<div id="px-toolbar" style="
  position:fixed; top:80px; left:10px; z-index:10001;
  display:flex; flex-direction:column; gap:4px;">
  <button class="px-tool" data-mode="add_node"
    style="padding:5px 10px;border:2px solid #aaa;border-radius:4px;cursor:pointer;
           background:white;color:#333;font:12px sans-serif;
           box-shadow:0 1px 4px rgba(0,0,0,0.15);white-space:nowrap;">
    + Node
  </button>
  <button class="px-tool" data-mode="draw_crosswalk"
    style="padding:5px 10px;border:2px solid #aaa;border-radius:4px;cursor:pointer;
           background:white;color:#333;font:12px sans-serif;
           box-shadow:0 1px 4px rgba(0,0,0,0.15);white-space:nowrap;">
    Crosswalk
  </button>
  <button class="px-tool" data-mode="move_endpoint"
    style="padding:5px 10px;border:2px solid #aaa;border-radius:4px;cursor:pointer;
           background:white;color:#333;font:12px sans-serif;
           box-shadow:0 1px 4px rgba(0,0,0,0.15);white-space:nowrap;">
    Move
  </button>
</div>
```

- [ ] **Step 2: Add editor mode state and toolbar toggle logic**

```javascript
// ── Editor modes ──────────────────────────────────────────────────────────────
let _editorMode = null;  // null | 'add_node' | 'draw_crosswalk' | 'move_endpoint'

function setEditorMode(mode) {
  if (_editorMode === mode) mode = null;  // toggle off
  _editorMode = mode;
  // Update toolbar button styles
  document.querySelectorAll('.px-tool').forEach(btn => {
    const m = btn.dataset.mode;
    btn.style.background  = m === mode ? '#1a73e8' : 'white';
    btn.style.color       = m === mode ? 'white'   : '#333';
    btn.style.borderColor = m === mode ? '#1a73e8' : '#aaa';
  });
  // Update cursor
  map.getContainer().style.cursor = mode === 'add_node' ? 'crosshair'
    : mode === 'draw_crosswalk' ? 'pointer'
    : mode === 'move_endpoint'  ? 'grab'
    : '';
  // Clear in-progress state when switching modes
  if (mode !== 'draw_crosswalk') {
    _xwFirstRamp = null;
    if (_xwPreviewLine) { map.removeLayer(_xwPreviewLine); _xwPreviewLine = null; }
  }
  if (mode !== 'move_endpoint') {
    _dragNode = null;
  }
  setStatus(mode ? `Mode: ${mode.replace(/_/g, ' ')}` : 'Ready');
}

document.querySelectorAll('.px-tool').forEach(btn => {
  btn.addEventListener('click', () => setEditorMode(btn.dataset.mode));
});
```

- [ ] **Step 3: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): add toolbar with mode switching for add_node, crosswalk, move"
```

---

### Task 7: Editor HTML — Add Node Mode

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Add the add_node click handler**

Modify the existing `map.on('click', ...)` handler. Replace the current click handler with one that dispatches by mode:

```javascript
map.on('click', e => {
  _coordBox.style.display = 'block';
  _coordBox.textContent = e.latlng.lat.toFixed(7) + ', ' + e.latlng.lng.toFixed(7);

  if (_editorMode === 'add_node') {
    handleAddNode(e.latlng);
    return;
  }
  if (_editorMode === 'draw_crosswalk') {
    // handled by ramp marker click, not map click
  }
  if (_probeActive) {
    const r = parseInt(_probeRadiusInput.value, 10);
    renderProbe(probeSegments(e.latlng, r), probePoints(e.latlng, r));
  }
});

function handleAddNode(latlng) {
  const utm = proj4('EPSG:4326', 'EPSG:32610', [latlng.lng, latlng.lat]);
  // Pulsing green circle
  const marker = L.circleMarker(latlng, {
    radius: 7, color: '#16a34a', fillColor: '#16a34a', fillOpacity: 0.7,
    weight: 2, className: 'px-pulse',
  }).bindTooltip('New node (unsaved)')
    .addTo(lg.editor);

  recordEdit('add_node', { x: utm[0], y: utm[1] });
  setStatus(`Node added at UTM (${utm[0].toFixed(1)}, ${utm[1].toFixed(1)})`);
}
```

- [ ] **Step 2: Add CSS pulse animation**

Add to the `<style>` block:

```css
@keyframes px-pulse {
  0%   { opacity: 0.7; transform: scale(1); }
  50%  { opacity: 1;   transform: scale(1.4); }
  100% { opacity: 0.7; transform: scale(1); }
}
.px-pulse { animation: px-pulse 1.5s ease-in-out infinite; }
```

- [ ] **Step 3: Verify**

Open editor, click "+ Node", click on map. Green pulsing circle appears. Changeset panel shows "1 node(s) added".

- [ ] **Step 4: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): implement add_node mode with pulsing visual feedback"
```

---

### Task 8: Editor HTML — Context Menu Infrastructure

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Add context menu HTML**

Add before `<script>`:

```html
<div id="px-ctx-menu" style="
  display:none; position:fixed; z-index:10003;
  background:white; border:1px solid #ccc; border-radius:4px;
  box-shadow:0 2px 8px rgba(0,0,0,0.2); font:12px sans-serif;
  min-width:160px; overflow:hidden;">
</div>
```

- [ ] **Step 2: Add context menu JS**

```javascript
// ── Context menu ──────────────────────────────────────────────────────────────
const _ctxMenu = document.getElementById('px-ctx-menu');
let _ctxTarget = null;  // { type: 'node'|'ramp'|'endpoint', key, data }

function showContextMenu(x, y, items) {
  _ctxMenu.innerHTML = items.map(item =>
    `<div class="px-ctx-item" style="padding:6px 12px;cursor:pointer;border-bottom:1px solid #f0f0f0;"
          onmouseover="this.style.background='#e8f0fe'"
          onmouseout="this.style.background='white'"
          onclick="(${item.action})(); hideContextMenu();">
      ${item.label}
    </div>`
  ).join('');
  _ctxMenu.style.left = x + 'px';
  _ctxMenu.style.top  = y + 'px';
  _ctxMenu.style.display = 'block';
}

function hideContextMenu() { _ctxMenu.style.display = 'none'; }

// Hide on any click outside
document.addEventListener('click', e => {
  if (!_ctxMenu.contains(e.target)) hideContextMenu();
});

// Prevent default context menu on the map
map.getContainer().addEventListener('contextmenu', e => e.preventDefault());
```

- [ ] **Step 3: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): add reusable context menu infrastructure"
```

---

### Task 9: Editor HTML — Context Menu on Intersection Nodes (Hull Consolidation)

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Add consolidation selection state**

```javascript
// ── Hull consolidation selection ──────────────────────────────────────────────
const _consolidationSelected = new Map();  // node_id → { marker, latlng, utmX, utmY }

function toggleConsolidationSelect(nodeId, marker, latlng) {
  if (_consolidationSelected.has(nodeId)) {
    // Deselect — remove orange ring
    const entry = _consolidationSelected.get(nodeId);
    if (entry.ring) lg.editor.removeLayer(entry.ring);
    _consolidationSelected.delete(nodeId);
  } else {
    // Select — add orange ring
    const ring = L.circleMarker(latlng, {
      radius: 10, color: '#f97316', fillColor: 'transparent',
      weight: 3, opacity: 0.9,
    }).bindTooltip('Selected for consolidation').addTo(lg.editor);
    const utm = proj4('EPSG:4326', 'EPSG:32610', [latlng.lng, latlng.lat]);
    _consolidationSelected.set(nodeId, { ring, latlng, utmX: utm[0], utmY: utm[1] });
  }
  setStatus(`${_consolidationSelected.size} node(s) selected for consolidation`);
}

function doConsolidate() {
  if (_consolidationSelected.size < 2) {
    setStatus('Select 2+ nodes to consolidate'); return;
  }
  const nodeKeys = [];
  _consolidationSelected.forEach(v => nodeKeys.push([v.utmX, v.utmY]));
  recordEdit('consolidate_hulls', { node_keys: nodeKeys, buffer_m: null });
  // Clear selection visuals
  _consolidationSelected.forEach(v => { if (v.ring) lg.editor.removeLayer(v.ring); });
  _consolidationSelected.clear();
  setStatus('Hull consolidation recorded');
}
```

- [ ] **Step 2: Wire context menu to node markers**

Modify the node rendering section in `renderRows` (within the `tier >= 4` block where `L.circleMarker` is created for nodes). After `const nodeMk = L.circleMarker(...)`, add:

```javascript
          nodeMk.on('contextmenu', function(ev) {
            L.DomEvent.stopPropagation(ev);
            L.DomEvent.preventDefault(ev);
            _ctxTarget = { type: 'node', key: nid, latlng: mid, marker: nodeMk };
            const items = [
              {
                label: _consolidationSelected.has(nid)
                  ? '&#10003; Deselect for consolidation'
                  : 'Select for consolidation',
                action: `() => toggleConsolidationSelect('${nid}', null, L.latLng(${mid[0]}, ${mid[1]}))`
              },
            ];
            if (_consolidationSelected.size >= 1) {
              items.push({
                label: `Consolidate (${_consolidationSelected.size + (_consolidationSelected.has(nid) ? 0 : 1)} nodes)`,
                action: `() => { if(!_consolidationSelected.has('${nid}')) toggleConsolidationSelect('${nid}', null, L.latLng(${mid[0]}, ${mid[1]})); doConsolidate(); }`
              });
            }
            showContextMenu(ev.originalEvent.clientX, ev.originalEvent.clientY, items);
          });
```

- [ ] **Step 3: Verify**

Right-click a node → "Select for consolidation" → orange ring appears. Right-click another → "Consolidate (2 nodes)" → changeset records it.

- [ ] **Step 4: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): add node context menu for hull consolidation selection"
```

---

### Task 10: Editor HTML — Context Menu on Curb Ramps (Toggle Enable/Disable)

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Wire context menu to curb ramp markers**

In the `renderRows` ramp rendering section (within `tier >= 5`, where `rampMk` is created), after `.addTo(lg.ramps)`, add:

```javascript
        rampMk.on('contextmenu', function(ev) {
          L.DomEvent.stopPropagation(ev);
          L.DomEvent.preventDefault(ev);
          const segId = String(r.street_grid_id);
          const rampData = { segment_id: segId, side: s, position: p, index: i };
          // Track disabled state visually
          const isDisabled = rampMk._pxDisabled || false;
          const items = [{
            label: isDisabled ? 'Enable curb ramp' : 'Disable curb ramp',
            action: `() => {
              const mk = pointIndex.get('${rampKey}');
              if (mk && mk.layer) {
                const nowDisabled = !mk.layer._pxDisabled;
                mk.layer._pxDisabled = nowDisabled;
                mk.layer.setStyle(nowDisabled
                  ? { color: '#999', fillColor: '#999', fillOpacity: 0.4 }
                  : { color: '${C.ramp}', fillColor: '${C.ramp}', fillOpacity: 0.9 });
              }
              recordEdit('toggle_curb_ramp', {
                segment_id: '${segId}', side: '${s}', position: '${p}',
                index: ${i}, enabled: ${isDisabled}
              });
            }`
          }];
          showContextMenu(ev.originalEvent.clientX, ev.originalEvent.clientY, items);
        });
```

- [ ] **Step 2: Verify**

Right-click a ramp → "Disable curb ramp" → marker turns gray. Changeset updates.

- [ ] **Step 3: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): add curb ramp context menu for enable/disable toggle"
```

---

### Task 11: Editor HTML — Draw Crosswalk Mode

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Add crosswalk drawing state and preview line**

```javascript
// ── Crosswalk drawing ─────────────────────────────────────────────────────────
let _xwFirstRamp = null;   // { segment_id, side, position, index, latlng }
let _xwPreviewLine = null;

function handleRampClickForCrosswalk(rampData, latlng) {
  if (_editorMode !== 'draw_crosswalk') return;

  if (!_xwFirstRamp) {
    // First ramp selected
    _xwFirstRamp = { ...rampData, latlng };
    setStatus('Click second curb ramp to complete crosswalk');
    // Start preview line
    map.on('mousemove', _xwMouseMove);
    return;
  }

  // Second ramp selected — draw crosswalk
  map.off('mousemove', _xwMouseMove);
  if (_xwPreviewLine) { map.removeLayer(_xwPreviewLine); _xwPreviewLine = null; }

  // Draw dashed magenta line (pending)
  L.polyline([_xwFirstRamp.latlng, latlng], {
    color: '#ff00ff', weight: 3, opacity: 0.8, dashArray: '8 4',
  }).bindTooltip('Crosswalk (unsaved)').addTo(lg.editor);

  recordEdit('draw_crosswalk', {
    ramp_a: {
      segment_id: _xwFirstRamp.segment_id, side: _xwFirstRamp.side,
      position: _xwFirstRamp.position, index: _xwFirstRamp.index,
    },
    ramp_b: {
      segment_id: rampData.segment_id, side: rampData.side,
      position: rampData.position, index: rampData.index,
    },
  });

  _xwFirstRamp = null;
  setStatus('Crosswalk drawn (unsaved)');
}

function _xwMouseMove(e) {
  if (!_xwFirstRamp) return;
  if (_xwPreviewLine) map.removeLayer(_xwPreviewLine);
  _xwPreviewLine = L.polyline([_xwFirstRamp.latlng, e.latlng], {
    color: '#ff00ff', weight: 2, opacity: 0.5, dashArray: '4 4',
  }).addTo(map);
}
```

- [ ] **Step 2: Wire ramp markers to crosswalk click handler**

In the ramp rendering section (same place as Task 10), add a regular click handler after the context menu handler:

```javascript
        rampMk.on('click', function(ev) {
          L.DomEvent.stopPropagation(ev);
          if (_editorMode === 'draw_crosswalk') {
            handleRampClickForCrosswalk(
              { segment_id: String(r.street_grid_id), side: s, position: p, index: i },
              [mid[0], mid[1]]
            );
          }
        });
```

- [ ] **Step 3: Verify**

Click "Crosswalk" toolbar button → click a ramp → dashed preview follows cursor → click another ramp → dashed magenta line drawn. Changeset shows "1 crosswalk(s) drawn".

- [ ] **Step 4: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): implement draw_crosswalk mode with click-click preview"
```

---

### Task 12: Editor HTML — Move Endpoint Mode with Rubber-banding

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Add move endpoint state and segment selection panel**

```javascript
// ── Move endpoint ─────────────────────────────────────────────────────────────
let _dragNode = null;       // { nodeId, marker, origLatlng, origUtm }
let _dragSegments = new Map();  // segId → { checked, layer, origCoords }
let _dragGhostMarker = null;

function handleNodeClickForMove(nodeId, latlng) {
  if (_editorMode !== 'move_endpoint') return;

  if (_dragNode && _dragNode.nodeId === nodeId) {
    // Clicking same node — deselect
    _finishDrag(false);
    return;
  }

  // Clear previous drag
  if (_dragNode) _finishDrag(false);

  _dragNode = {
    nodeId,
    origLatlng: L.latLng(latlng[0], latlng[1]),
    origUtm: proj4('EPSG:4326', 'EPSG:32610', [latlng[1], latlng[0]]),
  };

  // Ghost marker at original position
  _dragGhostMarker = L.circleMarker(latlng, {
    radius: 6, color: '#888', fillColor: '#ccc', fillOpacity: 0.5, weight: 1,
    dashArray: '3 3',
  }).bindTooltip('Original position').addTo(lg.editor);

  // Find connected segments via probe and show selection panel
  _showSegmentSelectionPanel(nodeId, latlng);

  // Make the node draggable
  const nodePt = pointIndex.get(`node_${nodeId}`);
  if (nodePt && nodePt.layer) {
    nodePt.layer.setStyle({ color: '#eab308', fillColor: '#eab308' });
    // Enable dragging via mouse events on map
    map.on('mousemove', _dragMouseMove);
    map.once('click', _dragMouseClick);
  }
  setStatus('Drag to move endpoint. Click to place.');
}

function _dragMouseMove(e) {
  if (!_dragNode) return;
  const nodePt = pointIndex.get(`node_${_dragNode.nodeId}`);
  if (nodePt && nodePt.layer) {
    nodePt.layer.setLatLng(e.latlng);
  }
  // Rubber-band checked segments
  _dragSegments.forEach((seg, segId) => {
    if (!seg.checked || !seg.layer) return;
    // Update the first or last coordinate of the polyline
    // This is visual-only; actual geometry update happens on save
  });
}

function _dragMouseClick(e) {
  if (!_dragNode) return;
  map.off('mousemove', _dragMouseMove);

  const newUtm = proj4('EPSG:4326', 'EPSG:32610', [e.latlng.lng, e.latlng.lat]);
  const rubberBandSegs = [];
  _dragSegments.forEach((seg, segId) => {
    if (seg.checked) rubberBandSegs.push(segId);
  });

  recordEdit('move_endpoint', {
    node_id: _dragNode.nodeId,
    new_x: newUtm[0],
    new_y: newUtm[1],
    rubber_band_segments: rubberBandSegs,
  });

  // Update node position visually
  const nodePt = pointIndex.get(`node_${_dragNode.nodeId}`);
  if (nodePt && nodePt.layer) {
    nodePt.layer.setLatLng(e.latlng);
    nodePt.layer.setStyle({ color: '#eab308', fillColor: '#eab308' });
  }

  _finishDrag(true);
  setStatus('Endpoint moved (unsaved)');
}

function _finishDrag(keepVisuals) {
  map.off('mousemove', _dragMouseMove);
  if (!keepVisuals) {
    // Restore original position
    if (_dragNode) {
      const nodePt = pointIndex.get(`node_${_dragNode.nodeId}`);
      if (nodePt && nodePt.layer) {
        nodePt.layer.setLatLng(_dragNode.origLatlng);
        nodePt.layer.setStyle({ color: C.node, fillColor: C.node });
      }
    }
    if (_dragGhostMarker) { lg.editor.removeLayer(_dragGhostMarker); _dragGhostMarker = null; }
  }
  _dragSegments.clear();
  _dragNode = null;
  // Hide segment selection panel
  document.getElementById('px-probe-segs').innerHTML = '';
}

function _showSegmentSelectionPanel(nodeId, latlng) {
  // Reuse probe panel to show connected segments with checkboxes
  const probePanel = document.getElementById('px-probe');
  const segList    = document.getElementById('px-probe-segs');
  probePanel.style.display = 'block';

  // Find nearby segments
  const nearby = probeSegments(L.latLng(latlng[0], latlng[1]), 30);
  _dragSegments.clear();
  const html = nearby.map(f => {
    _dragSegments.set(f.key, { checked: true, layer: f.entry.layer, origCoords: null });
    const safeKey = f.key.replace(/[^a-zA-Z0-9_]/g, '_');
    return '<div style="display:flex;align-items:center;gap:4px;padding:3px 2px;border-bottom:1px solid #f0f0f0;font-size:11px;">' +
      `<input type="checkbox" id="px-rb-${safeKey}" checked onchange="toggleRubberBand('${f.key}', this.checked)">` +
      `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${f.entry.label || f.key}</span>` +
      `<span style="color:#888;">${Math.round(f.dist)}m</span>` +
    '</div>';
  }).join('');
  segList.innerHTML = html || '<div style="color:#999;font-size:11px;padding:4px 2px;">No nearby segments</div>';
}
```

- [ ] **Step 2: Add the toggleRubberBand global function**

```javascript
window.toggleRubberBand = function(segKey, checked) {
  const seg = _dragSegments.get(segKey);
  if (seg) seg.checked = checked;
};
```

- [ ] **Step 3: Wire node markers to move mode click handler**

In the node rendering section (same `tier >= 4` block), after the context menu handler, add:

```javascript
          nodeMk.on('click', function(ev) {
            L.DomEvent.stopPropagation(ev);
            if (_editorMode === 'move_endpoint') {
              handleNodeClickForMove(nid, mid);
            }
          });
```

- [ ] **Step 4: Verify**

Click "Move" → click a node → ghost marker appears, probe panel shows checkboxes → move mouse → click to place → changeset records move.

- [ ] **Step 5: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): implement move_endpoint mode with segment rubber-banding"
```

---

### Task 13: Server — Hull Consolidation with Adaptive Buffer

**Files:**
- Modify: `Implementations/editor_server.py`

- [ ] **Step 1: Add hull consolidation logic to the save endpoint**

Add a new function before the `/save` endpoint:

```python
def _apply_hull_consolidation(
    gdf: gpd.GeoDataFrame,
    patch_idx: pd.Index,
    edits: list[ConsolidatedHull],
    default_lane_width_m: float = 3.5,
) -> dict[tuple[float, float], "Polygon"]:
    """Merge intersection hulls for consolidated nodes.

    Returns a dict of merged hull polygons keyed by the centroid of the
    merged node group (used as the new composite node key).
    """
    from shapely.ops import unary_union

    if not edits:
        return {}

    # Build hulls from the patch to get current hull state
    patch_df = gdf.loc[patch_idx].copy()
    _, (node_to_segs, node_key_to_pt, current_hulls) = _build_intersection_hulls(
        patch_df, default_lane_width_m
    )

    merged_hulls: dict[tuple[float, float], Polygon] = {}

    for ch in edits:
        keys = [tuple(k) for k in ch.node_keys]
        hulls = [current_hulls[k] for k in keys if k in current_hulls]
        if len(hulls) < 2:
            continue

        # Compute adaptive buffer = max pairwise min-distance / 2
        if ch.buffer_m is not None:
            buf = ch.buffer_m
        else:
            max_gap = 0.0
            for i in range(len(hulls)):
                for j in range(i + 1, len(hulls)):
                    d = hulls[i].distance(hulls[j])
                    if d > max_gap:
                        max_gap = d
            buf = max_gap / 2.0

        buffered = [h.buffer(buf) for h in hulls]
        merged = unary_union(buffered)

        # Composite key = centroid of merged hull
        cx, cy = merged.centroid.x, merged.centroid.y
        merged_hulls[(cx, cy)] = merged

    return merged_hulls
```

- [ ] **Step 2: Integrate into the save endpoint**

In the `save_edits` function, after `_apply_drawn_crosswalks` and before the pipeline re-run section, add:

```python
    # ── Hull consolidation ────────────────────────────────────────────────────
    merged_hulls = _apply_hull_consolidation(
        gdf, patch_idx, req.edits.consolidated_hulls
    )
    # merged_hulls are passed through to the pipeline stages via the
    # _precomputed_hulls parameter if needed — for now, the pipeline re-run
    # via _build_intersection_hulls will rebuild hulls from the updated node
    # flags, and the merged hulls overlay those results.
```

- [ ] **Step 3: Commit**

```bash
git add Implementations/editor_server.py
git commit -m "feat(editor): add hull consolidation with adaptive gap-based buffer"
```

---

### Task 14: Make Global Functions Accessible Outside Module Scope

**Files:**
- Modify: `Output/test_maps/county_editor.html`

- [ ] **Step 1: Export functions that are called from inline HTML onclick handlers**

Since the JS is in a `<script type="module">`, functions aren't globally visible. At the end of the module script (before the closing `</script>`), add:

```javascript
// ── Expose functions needed by inline HTML handlers ──────────────────────────
window.pxSearch             = pxSearch;
window.doSave               = doSave;
window.discardChangeset     = discardChangeset;
window.flashFeature         = flashFeature;
window.toggleHide           = toggleHide;
window.doConsolidate        = doConsolidate;
window.toggleConsolidationSelect = toggleConsolidationSelect;
```

- [ ] **Step 2: Commit**

```bash
git add Output/test_maps/county_editor.html
git commit -m "feat(editor): expose module functions to inline HTML onclick handlers"
```

---

### Task 15: End-to-End Verification

**Files:** None (manual testing)

- [ ] **Step 1: Start the server**

```bash
cd Implementations && uv run uvicorn editor_server:app --reload --port 8081
```

- [ ] **Step 2: Open the editor and verify each interaction**

Open `http://localhost:8081/` and test:

1. **Map viewer works** — zoom in/out, layers load at correct tiers, search works, probe works
2. **+ Node mode** — click toolbar → click map → green pulsing marker → changeset shows "1 node(s) added"
3. **Consolidation** — right-click node → "Select for consolidation" → orange ring → right-click second node → "Consolidate" → changeset records it
4. **Curb ramp toggle** — right-click ramp → "Disable curb ramp" → marker grays out → changeset updates
5. **Draw crosswalk** — click "Crosswalk" → click ramp A → dashed preview → click ramp B → dashed magenta line → changeset records it
6. **Move endpoint** — click "Move" → click node → ghost marker + probe panel checkboxes → move → click to place → changeset records it
7. **Save** — click Save → spinner → server processes → map refreshes
8. **Discard** — make some edits → click Discard → all cleared

- [ ] **Step 3: Fix any issues found during testing**

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(editor): county map editor with all edit modes and pipeline re-run"
```