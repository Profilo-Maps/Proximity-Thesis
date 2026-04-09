# Proximity Model — Active Transport Infrastructure Graph

Processes OpenStreetMap data to generate per-city street network GeoParquet files with detailed sidewalk, bikeway, crosswalk, and curb ramp attributes. Implements the 15-step pipeline defined in [`specs/ProximityPipelineOutline.md`](specs/ProximityPipelineOutline.md).

---

## Files

| File | Role |
|---|---|
| `Implementations/ProximityModel.py` | Main pipeline (15 steps, 3 phases) |
| `Implementations/editor_server.py` | FastAPI server for the county map editor |
| `Implementations/county_editor.html` | Interactive county map editor frontend |
| `Implementations/test_maps.py` | Diagnostic map generation (read-only folium maps) |
| `run_pipeline.bat` | Full pipeline runner — builds parquet, generates maps, launches editor |
| `specs/ProximityPipelineOutline.md` | Step-by-step pipeline specification |
| `specs/ProximitySchema.md` | Full 256-column output schema |

### Data
- `Output/` — Generated GeoParquet files and map HTML
- `Implementations/.osm_cache/` — Cached OSMnx graphs and phase checkpoints

---

## Quick Start

### Option A — Full pipeline + editor (Windows)

```bat
run_pipeline.bat
```

Runs `ProximityModel.py` then `test_maps.py`, then launches the county editor at `http://localhost:8081` automatically.

### Option B — Pipeline only

```bash
uv run python -c "
from Implementations.ProximityModel import PipelineConfig, run_pipeline
cfg = PipelineConfig(
    place_name='San Francisco County, California, USA',
    output_path='Output/sf_network.parquet',
)
run_pipeline(cfg)
"
```

### Option C — Editor only (parquet already built)

```bash
uv run uvicorn Implementations.editor_server:app --reload --port 8081
```

Then open `http://localhost:8081`.

---

## County Map Editor

The editor lets you inspect and correct the pipeline output interactively against a live parquet file.

### Edit Modes (toolbar)

| Mode | Action |
|---|---|
| **+ Node** | Click to place a new intersection node |
| **Crosswalk** | Click two curb ramps to draw a crosswalk between them |
| **Move** | Drag a segment endpoint; select which segments rubber-band with it |

### Context Menus

- **Right-click intersection node** — Consolidate hulls: merges selected nodes into a single hull zone, using an adaptive buffer based on the gap between hull edges
- **Right-click curb ramp** — Enable / disable the ramp

### Save & Pipeline Re-run

Edits are staged in the changeset panel (persisted in `localStorage`). Clicking **Save** sends the changeset to the server, which:

1. Applies direct edits to the in-memory GeoDataFrame
2. Determines the earliest affected pipeline stage
3. Re-runs only the affected stages on the rows within the edit bounding box
4. Writes the result back to the parquet file

**Stage → Step mapping:**

| Stage | Steps re-run |
|---|---|
| `SNAP_ENDPOINTS` | `step_10` |
| `BUILD_HULLS` + `ASSIGN_RAMPS` | `step_12`, `step_13` |
| `CREATE_CROSSWALKS` | `step_14`, `step_15` |

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

1. **Identify intersection nodes** — segments whose `start_node_is_intersection_node` or `end_node_is_intersection_node` flag is set are collected. A lookup table maps each intersection node ID → list of `(segment_index, "start"|"end")` pairs.

2. **Collect candidate ramp points** — for every segment arm attached to an intersection node, the endpoint of each populated sidewalk geometry (left and right sides) is extracted:
   - `which_end == "start"` → first coordinate of the sidewalk LineString
   - `which_end == "end"` → last coordinate of the sidewalk LineString
   - Nodes with fewer than 2 candidate points are skipped.

3. **Pre-snap close pairs** — all candidate points are projected to UTM. Any two points within `curb_ramp_snap_m` (default 1.0 m) are replaced by their midpoint (back-projected to WGS-84). This merges ramp positions where left/right sidewalk endpoints nearly coincide at a tight corner.

4. **Write ramp geometry** — each snapped point is stored in the first available slot in the schema columns:
   ```
   sidewalk_{left|right}_curbramp_{start|end}_{1|2|3}_geometry
   sidewalk_{left|right}_curbramp_{start|end}_{1|2|3}_ID
   ```
   Up to 3 ramp slots exist per side per endpoint. The slot is filled only if the column is currently null, so manual overrides (from the editor) are preserved on pipeline re-runs.

5. **Build intersection hull** — after snapping, the unique ramp points at the node form a convex hull (or a thin buffered line if only 2 unique points remain). This hull becomes the intersection zone used by Steps 13–15.

### Limitations

- Ramps are placed at sidewalk **endpoints**, so a ramp is only inferred where a sidewalk geometry actually terminates at an intersection. Segments with no mapped sidewalk produce no ramp.
- The approach cannot distinguish a ramp from a street-level corner where the sidewalk happens to end — it models the *location* of the transition, not whether a physical ramp structure is present.
- Ramp attributes (surface, slope, tactile paving) are not populated automatically; they require manual enrichment via the county editor.

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

```
osmnx  geopandas  shapely  pyarrow  pandas  numpy  scipy  pyproj  requests  tqdm  fastapi  uvicorn
```

Run with `uv run python` from the project root. Python path: `C:/Dev/Proximity/.venv/Scripts/python.exe`.
