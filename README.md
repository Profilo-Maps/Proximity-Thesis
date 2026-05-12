# Proximity Model — Active Transport Infrastructure Graph

Processes OpenStreetMap data to generate per-city street network GeoParquet files with detailed sidewalk, bikeway, crosswalk, and curb ramp attributes. Implements the 15-step pipeline defined in [`specs/ProximityPipelineOutline.md`](specs/ProximityPipelineOutline.md).

---

## Files

| File | Role |
|---|---|
| `Implementations/ProximityModel.py` | Main pipeline (15 steps, 3 phases) |
| `editor/server/editor_server.py` | FastAPI server for the county map editor |
| `editor/web/` | React + Vite + MapLibre GL frontend |
| `editor/shared/` | `@proximity/shared` — TypeScript types and utilities shared with RollTracks |
| `run_pipeline.bat` | Pipeline runner with selectable mode (Full / Pipeline only / Editor only) |

### Data

- `Output/` — Generated GeoParquet files and map HTML
- `Implementations/.osm_cache/` — Cached OSMnx graphs and phase checkpoints

---

## Quick Start

```bat
run_pipeline.bat
```

Interactive menu — pick one:

| Key | Mode | What it does |
|---|---|---|
| `1` | Full | Pipeline + Editor |
| `2` | Pipeline only | Run pipeline |
| `3` | Editor only | Launch editor (skip pipeline) |

The editor opens automatically at `http://localhost:5173` with the API server on port 8000.

### Manual launch (CLI)

```bash
# Pipeline
uv run python -c "
from Implementations.ProximityModel import PipelineConfig, run_pipeline
cfg = PipelineConfig(
    place_name='San Francisco County, California, USA',
    output_path='Output/sf_network.parquet',
)
run_pipeline(cfg)
"

# Editor (two terminals)
cd editor/server && uv run uvicorn editor_server:app --reload --port 8000
cd editor/web && npm run dev
```

---

## County Map Editor

A browser-based GIS editor for inspecting and correcting pipeline output against a live parquet file. Built with React, MapLibre GL JS (WebGL), Zustand for state management, and a FastAPI backend.

### Architecture

```
editor/
  shared/                     @proximity/shared — types + utils
    src/types/
      NetworkSegment.ts       Full 256-column schema as TypeScript interfaces
      changeset.ts            SaveRequest, Edits, PipelineStage types
    src/tools/
      types.ts                MapAdapter, ToolCallbacks, ToolSubtype, SourceMutation
      toolLogic.ts            Platform-agnostic tool state machines (12 tools)
      geometry.ts             Pure math: point projection, nearest-on-line
      index.ts                Barrel export
    src/utils/
      SegmentSpatialGrid.ts   Degree-based spatial grid with haversine queries
      wkbToGeoJSON.ts         WKB hex → GeoJSON decoder
      utmToWgs84.ts           UTM Zone 10N → WGS84 projection

  web/                        React + Vite frontend
    src/
      api/editorApi.ts        Typed API client for FastAPI endpoints
      store/
        editorStore.ts        App state (tools, subtypes, selection, layers, viewport)
        changesetStore.ts     Edit accumulation with localStorage persistence
      components/
        MapView.tsx           MapLibre GL map with 15 layers + edit overlays
        ToolBar.tsx           12 edit tools grouped by Point/Line/Poly
        SubtypeSelector.tsx   Floating subtype popover for creation tools
        panels/
          ProbePanel.tsx      Radius-based feature inspection
          HistoryPanel.tsx    Chronological edit log with undo + save
          AttributeTable.tsx  Bottom strip with inline cell editing
          LayersPanel.tsx     Layer visibility toggles
          SearchPanel.tsx     Segment/node ID search
          ConfigPanel.tsx     Pipeline config editor (global/city)
      hooks/
        useMapLayers.ts       Layer definitions + colors
        useToolHandler.ts     Tool → map interaction wiring (passes subtype)
        useVertexDrag.ts      Click-drag vertex editing with real-time GeoJSON mutation
        mapLibreAdapter.ts    Web-specific MapAdapter implementation

  server/
    editor_server.py          FastAPI backend (parquet I/O, pipeline re-runs)
```

### Shared Package

`editor/shared/` (`@proximity/shared`) provides TypeScript types and utilities shared between the web editor and the [RollTracks](https://github.com/your-org/rolltracks) React Native mobile app. The mobile app's DataRanger service consumes the same `NetworkSegment` types and spatial utilities.

Install via file reference: `"@proximity/shared": "file:../shared"`.

### UI Layout

```
┌──────────────────────────────────────────────────────┐
│ Toolbar │  Map (MapLibre GL)              │ Probe     │
│  (left) │                                │ History   │
│         │         [Layers panel]          │  (right)  │
├─────────┴────────────────────────────────┴───────────┤
│ Attribute Table strip  [pop-out ↗]                   │
└──────────────────────────────────────────────────────┘
```

- **Left toolbar** — vertical icon bar with 14 edit tools organized by type (Point, Segment, Polygon)
- **Probe panel** (top-right) — radius-based feature inspection with visibility toggles
- **History panel** (bottom-right) — pending changeset entries with save/discard
- **Layers panel** (bottom-left, dockable) — per-layer visibility toggles
- **Attribute table** (bottom strip) — expandable table with inline cell editing
- **Config panel** (slide-out) — Global and City tabs for `PipelineConfig` values
- **Search panel** (slide-out) — fly to segments or nodes by ID

### Edit Tools

| Group | Tool | Action |
|---|---|---|
| **Point** | Move Point | Click-drag any point or line vertex (nodes, ramps, calming, street/bikeway/sidewalk vertices); endpoints highlighted |
| | Snap Point | Snap node to nearest segment |
| | Add Node | Place new point (subtype selector: node / ramp / calm) |
| | Delete Point | Remove point (works on all point types: nodes, curb ramps, calming points) |
| **Segment** | Merge Segments | Join two connected segments (earlier grid ID survives) |
| | Split Segment | Split segment at click point |
| | Draw Segment | Draw new line (subtype selector: street / bikeway / sidewalk / crosswalk / curb return) |
| | Draw Crosswalk | Connect two curb ramps |
| **Polygon** | Add Polygon | Draw new polygon (subtype selector: hull / slot) |
| | Delete Polygon | Remove hull polygon (marks node as non-intersection) |
| | Edit Polygon Face | Drag hull vertices |

### Feature Subtypes

Tools that create new features (Add Node, Draw Segment, Add Polygon) show a floating subtype selector when active. This determines what type of feature is placed:

| Tool | Subtypes | Notes |
|---|---|---|
| **Add Node** | `node` (intersection), `ramp` (curb ramp), `calm` (traffic calming) | Ramp subtype: tap a sidewalk first, then click to place the ramp at that position |
| **Draw Segment** | `street`, `bikeway`, `sidewalk`, `crosswalk`, `curb_return` | Sets the feature `_t` property and layer color |
| **Add Polygon** | `hull` (intersection hull), `slot` (crosswalk slot) | |

### Point Feature Types

The network contains three point feature types:

| Type | `_t` value | Description | Layer |
|---|---|---|---|
| Intersection node | `node` | Street network intersection point | `px-nodes` (zoom ≥ 17) |
| Curb ramp | `ramp` | Accessible ramp at sidewalk endpoint | `px-ramps` (zoom ≥ 18) |
| Traffic calming | `calm` | Speed bump, chicane, or other calming feature | `px-calming` (zoom ≥ 15) |

Feature highlighting (selection) is done via probe panel click or direct map click — no separate toolbar tool needed.

### Real-time Editing

All edits mutate the GeoJSON source in-place for immediate visual feedback:
- **Vertex drag** — line vertices (streets, bikeways, sidewalks) are extracted into a separate draggable points layer with endpoint highlighting
- **Add/delete** — features are added to or removed from the main source instantly
- **Undo** — Ctrl+Z reverts both the changeset and the map state via JSON snapshots
- **Confirm** — Enter key confirms multi-step tools (draw segment, add polygon, edit polygon face)

### Server Endpoints

| Endpoint | Description |
|---|---|
| `GET /features/{parquet}` | GeoJSON features filtered by bbox + zoom tier |
| `GET /rows/{parquet}` | Attribute table rows (no geometry) |
| `GET /hulls/{parquet}` | Hull polygons + crosswalk slots for bbox |
| `GET /config/{parquet}` | Pipeline config field definitions + overrides |
| `POST /config/{parquet}` | Update global or city config overrides |
| `POST /save` | Apply changeset, re-run pipeline stages, write parquet |

### Save & Pipeline Re-run

Edits accumulate in the changeset store (persisted to `localStorage`). On save, the server:

1. Filters rows within the edit bounding box (± 20m buffer)
2. Applies direct edits to the in-memory GeoDataFrame
3. Re-runs affected pipeline stages on filtered rows
4. Writes the updated parquet file

**Stage → Step mapping:**

| Stage | Index | Steps re-run | Triggered by |
|---|---|---|---|
| `SNAP_ENDPOINTS` | 0 | `step_10` | Move endpoint, split/merge/draw segment |
| `BUILD_HULLS` | 1 | `step_12`, `step_13` | Add/delete node, add/delete polygon |
| `ASSIGN_RAMPS` | 2 | `step_12` ramp phase | Toggle curb ramp, edit hull vertices |
| `CREATE_XWALKS` | 3 | `step_14`, `step_15` | Draw crosswalk |
| *(attr-only)* | 99 | none | Attribute table cell edits |

---

## Pipeline Overview

### Phase 1 — Street Network Foundation (~65s for SF County)

| Step | Description |
|---|---|
| 1 | Create empty GeoParquet with all 256 schema columns |
| 2 | Download OSM graph via OSMnx; populate street attributes; cache crosswalk nodes |
| 3 | Split segments with inter-vertex deflection > 45°; assign synthetic node IDs per parent segment |
| 4 | Compute normalized bearings; assign 500m grid IDs; flag intersection nodes |
| 5 | Populate traffic calming features from OSM nodes |
| 6 | Enrich start/end node elevations and segment incline from USGS 3DEP (optional) |

### Phase 2 — Sidewalks and Bikeways (~500s for SF County)

| Step | Description |
|---|---|
| 7 | Coalesce sidewalk and bikeway OSM tags into left/right schema slots |
| 8 | Swap left/right facility columns on reverse-bearing edges |
| 9 | Match separately-mapped cycleways and footways; generate offset geometries |
| 10 | Snap sidewalk endpoints at intersections (angular snap → ray-ray corner → trim) |
| 11 | Derive grid IDs for all sidewalk and bikeway slots |

### Phase 3 — Intersection Analysis (~230s for SF County)

| Step | Description |
|---|---|
| 12 | Place curb ramps at intersection corners; build convex hull zones from ramp points |
| 13 | Merge nearby hulls (< `hull_merge_buffer_m` apart) via buffer-cascade |
| 14 | Create crosswalk slot rectangles by buffering each hull face centerline |
| 15 | Detect/create crosswalk geometries within each slot (Cases A–E priority) |

**Case priority (Step 15):**
- **A** — Separately-mapped `highway=footway` crossing chain inside hull
- **B** — Sidewalk segment crossing a named street within the hull
- **C** — Curb ramp pair on opposing arms (cross-arm inference)
- **D** — Curb ramp pair on same arm (own-arm inference)
- **E** — Sidewalk endpoint pair fallback

---

## Curb Ramp Detection

Curb ramps are **inferred geometrically** from sidewalk endpoint positions — no OSM `kerb=*` or `barrier=kerb` tags are used.

### Algorithm (Step 12)

1. **Identify intersection nodes** — segments whose `start_node_is_intersection_node` or `end_node_is_intersection_node` flag is set are collected.

2. **Collect candidate ramp points** — for every segment arm at an intersection node, the endpoint of each sidewalk geometry (left and right) is extracted.

3. **Pre-snap close pairs** — candidate points within `curb_ramp_snap_m` (default 1.0m) are merged to their midpoint.

4. **Write ramp geometry** — snapped points fill schema slots (`sidewalk_{L|R}_curbramp_{start|end}_{1|2|3}_geometry`). Manual overrides from the editor are preserved on re-runs.

5. **Build intersection hull** — unique ramp points form a convex hull used by Steps 13–15.

### Limitations

- Ramps are only inferred where sidewalk geometry terminates at an intersection.
- Cannot distinguish a ramp from a street-level corner.
- Ramp attributes (surface, slope, tactile paving) require manual enrichment via the editor.

---

## PipelineConfig Options

| Field | Default | Description |
|---|---|---|
| `place_name` | required | OSM geocodable place name |
| `output_path` | required | Path for output `.parquet` file |
| `default_maxspeed` | `25` | mph fallback when OSM speed tag is missing |
| `default_lane_width` | `3.0` | Lane width fallback (metres) |
| `default_lanes` | `2` | Lane count fallback |
| `deflection_angle_threshold` | `45.0` | Degrees above which a segment is split (step 3) |
| `default_bikeway_width` | `1.5` | Bikeway width fallback (metres) |
| `default_sidewalk_width` | `1.5` | Sidewalk width fallback (metres) |
| `hull_merge_buffer_m` | `10.0` | Max gap between hulls to merge them (step 13) |
| `crosswalk_buffer_m` | `3.0` | Half-width of crosswalk slot rectangles (step 14) |
| `curb_ramp_snap_m` | `1.0` | Distance threshold for snapping close ramp points (step 12) |
| `enable_elevation` | `True` | Enable USGS 3DEP elevation enrichment (step 6) |
| `grid_cell_size` | `500.0` | Grid cell size in metres (step 4) |
| `network_type` | `"all"` | OSMnx network type filter |
| `custom_filter` | `None` | Custom OSMnx way filter string |

---

## Output

### `{city}_network.parquet` — 256 columns, EPSG:4326

| Category | Description |
|---|---|
| Street Centerline | ID, grid ID, nodes, bearing, name, highway, speed, lanes, surface, condition, incline |
| Street Features | Traffic calming multipoints with types, geometries, projected geometries, and attributes |
| Sidewalk (left/right) | ID, grid ID, presence, geometry, curb ramps (up to 3 slots × start/end), surface, width, condition |
| Crosswalk (start/end) | ID, geometry, controlled, marked, markings, signals |
| Bikeway (left/right × 1/2) | ID, grid ID, type, surface, permitted, width, incline, features |
| Main Geometries | Street geometry, start/end node geometries, sidewalk geometries |

All geometry stored as GeoParquet native geometry in **EPSG:4326**. Metric calculations use UTM EPSG:32610 internally.

---

## Performance (SF County — 184k raw edges → 203k rows)

| Phase | Time |
|---|---|
| Phase 1 (Steps 1–6) | ~65s |
| Phase 2 (Steps 7–11) | ~500s |
| Phase 3 (Steps 12–15) | ~230s |
| **Full pipeline** | **~790s (~13 min)** |

---

## Dependencies

**Python (pipeline + server):**
```
osmnx  geopandas  shapely  pyarrow  pandas  numpy  scipy  pyproj  requests  tqdm  fastapi  uvicorn
```

**Node (editor frontend):**
```
react  react-dom  maplibre-gl  zustand  @turf/*  vite  typescript
```

Run pipeline with `uv run python` from project root. Run editor with `npm run dev` from `editor/web/`.
