# County Map Editor — Design Spec

**Date:** 2026-04-06
**Status:** Approved

## Overview

Transform the read-only county map viewer into an interactive editor. Users can plot intersection nodes, consolidate hulls, toggle curb ramps, draw crosswalks, and move segment endpoints. Edits accumulate in-browser and are saved to a local Python server that re-runs the pipeline on affected rows.

## Files

| File | Role |
|---|---|
| `Implementations/editor_server.py` | FastAPI server — serves editor, static parquet, REST `/save` endpoint. Imports pipeline functions from `ProximityModel.py` |
| `Output/test_maps/county_editor.html` | Editor frontend — Leaflet + DuckDB WASM, editor modes, changeset tracking |

Existing `test_maps.py` is not modified.

## Pipeline Stage Model

Ordered stages, each depending on all prior:

| Stage | Index | Function (from ProximityModel) | Triggered by |
|---|---|---|---|
| `SNAP_ENDPOINTS` | 0 | `_snap_offset_endpoints` | Moving segment endpoints |
| `BUILD_HULLS` | 1 | `_build_intersection_hulls` | Adding/moving nodes, consolidating hulls |
| `ASSIGN_CURB_RAMPS` | 2 | `_assign_curb_ramp_geometries` | Toggling curb ramp status |
| `CREATE_CROSSWALKS` | 3 | `_create_crosswalk_geometries` | Drawing/deleting crosswalks |

### High-water-mark tracking

The browser changeset tracks `dirty_from_stage` (integer, initially `null`). Each edit type maps to a stage index:

- `move_endpoint` → 0
- `add_node` → 1
- `consolidate_hulls` → 1
- `toggle_curb_ramp` → 2
- `draw_crosswalk` → 3

On edit: `dirty_from_stage = min(dirty_from_stage, edit_stage_index)`.

On save, the server runs stages `dirty_from_stage` through 3 on bbox-filtered rows.

## Server API

### `GET /`
Serves `county_editor.html`.

### `GET /parquet/{filename}`
Static file serving for DuckDB WASM.

### `POST /save`

**Request:**
```json
{
  "parquet": "San_Francisco_County_California_USA_network.parquet",
  "dirty_from_stage": 1,
  "bbox": { "minX": 551000, "minY": 4178000, "maxX": 551500, "maxY": 4178500 },
  "edits": {
    "added_nodes": [
      { "x": 551234.5, "y": 4178123.4 }
    ],
    "consolidated_hulls": [
      { "node_keys": [[551100, 4178200], [551105, 4178210]], "buffer_m": null }
    ],
    "moved_endpoints": [
      { "node_id": "123456", "new_x": 551200, "new_y": 4178300,
        "rubber_band_segments": ["12_34_0", "12_35_0"] }
    ],
    "toggled_curb_ramps": [
      { "segment_id": "12_34_0", "side": "left", "position": "start", "index": 1,
        "enabled": false }
    ],
    "drawn_crosswalks": [
      { "ramp_a": { "segment_id": "12_34_0", "side": "left", "position": "start", "index": 1 },
        "ramp_b": { "segment_id": "12_35_0", "side": "right", "position": "end", "index": 1 } }
    ]
  }
}
```

**Response:**
```json
{
  "status": "ok",
  "stages_run": [1, 2, 3],
  "rows_affected": 42,
  "bbox_used": { "minX": 550980, "minY": 4177980, "maxX": 551520, "maxY": 4178520 }
}
```

### Data flow on save

1. Load full parquet into GeoDataFrame (cached in memory after first load)
2. Compute working bbox = edit bbox + 20m buffer (using `_CURB_RAMP_PROXIMITY_M`)
3. Filter rows where `street_geometry` intersects working bbox → `patch_df`
4. Apply direct edits to `patch_df` (move endpoints, add nodes, merge hulls, toggle ramps, insert crosswalk geometries)
5. Run pipeline stages `dirty_from_stage` → 3 on `patch_df`
6. Write `patch_df` rows back into the full GeoDataFrame by index
7. Export updated parquet, overwriting the original
8. Return summary; browser triggers DuckDB re-query to refresh map

## Adding Intersection Nodes

When the user places a new node on the map, the server finds the nearest segment(s) within a threshold and marks the corresponding `start_node_is_intersection_node` or `end_node_is_intersection_node` as `True` on rows whose endpoint is closest to the placed point. If the click is not near any existing segment endpoint, the node is stored as a standalone point — hull building will pick it up via proximity to nearby segments. The `add_node` edit payload stores the UTM coordinate; the server resolves it to affected rows.

## Hull Consolidation Geometry

When merging selected nodes' hulls:

1. Retrieve existing hull polygons for each selected node key
2. Compute gap distance = minimum distance between closest edges of any two hulls in the set
3. Buffer each hull by `gap_distance / 2`
4. Union all buffered hulls into a single polygon
5. Store the merged hull under a new composite node key; remove the individual entries

## Editor Frontend

### Toolbar (left side, below probe button)

| Button | Mode | Behavior |
|---|---|---|
| **+ Node** | `add_node` | Click map → place intersection node at click position |
| **Crosswalk** | `draw_crosswalk` | Click ramp A → dashed preview follows cursor → click ramp B → crosswalk drawn |
| **Move** | `move_endpoint` | Click node/endpoint → draggable. Connected segments rubber-band. Segment selection via probe panel checkboxes |

Only one mode active at a time. Clicking active mode deactivates it.

### Context Menu (right-click features)

| Feature | Menu items |
|---|---|
| Intersection node | "Select for consolidation" (toggle highlight; 2+ selected → "Consolidate" action appears) |
| Curb ramp | "Disable curb ramp" / "Enable curb ramp" |
| Segment endpoint | "Toggle curb ramp" |
| Node (in move mode) | Opens segment selection in probe panel — checkboxes per connected segment |

### Visual Feedback

| Edit type | Indicator |
|---|---|
| Added node | Pulsing green circle |
| Selected-for-consolidation | Orange ring highlight |
| Disabled curb ramp | Grayed/crossed-out marker |
| Drawn crosswalk (pending) | Dashed magenta line |
| Drawn crosswalk (saved) | Solid magenta line |
| Moved endpoint | Yellow marker at new pos, ghost at original |
| Crosswalk preview | Dashed line following cursor from ramp A |

### Changeset Panel (bottom-right, collapsible)

- Edit counts by type
- Current `dirty_from_stage` displayed as stage name
- **Save** button → POST `/save`, spinner, refresh on success
- **Discard** button → clear localStorage, reload map

### localStorage Schema

```json
{
  "px_editor_changeset": {
    "parquet": "San_Francisco_County_California_USA_network.parquet",
    "dirty_from_stage": 1,
    "edits": {
      "added_nodes": [],
      "consolidated_hulls": [],
      "moved_endpoints": [],
      "toggled_curb_ramps": [],
      "drawn_crosswalks": []
    }
  }
}
```

Bounding box is computed at save time from the union of all edit geometries.

## Future Work (TODO in code)

- Undo/redo: linear undo stack over the changeset (ctrl+z / ctrl+y)
