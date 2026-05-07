# Government Curb Ramp Enrichment — Design Spec
_Date: 2026-05-06_

## Overview

Enrich the Proximity parquet with city-provided curb ramp inventory data. Existing
topology-inferred ramp slots (placed by step 12) are distinguished from government-confirmed
ramps via the `quality` field. New ramp locations from the dataset that have no matching
topology slot are inserted into the nearest available road-row slot.

## Config

Two new `PipelineConfig` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `curb_ramp_data_path` | `str` | `""` | Absolute or relative path to CSV. Empty = skip. |
| `curb_ramp_match_radius_m` | `float` | `5.0` | UTM distance (metres) to match a CSV point to an existing slot. |

A module-level constant `_CURB_RAMP_DATA_PATH: str = ""` provides the compile-time default.

## Data Source

**File:** `Data/Curb_Ramps_20260217.csv`
**Filter:** `crExist == "1"` (physically existing ramps only)

### Coordinate resolution (per row, in order)

1. Use `Latitude` / `Longitude` columns if both are non-null and numeric → already WGS84.
2. Fallback: reproject `xLoc` / `yLoc` from **EPSG:2227** (California State Plane Zone III,
   US survey feet) to **EPSG:4326** via `pyproj.Transformer`.
3. If neither source yields a valid point, skip the row and increment a warning counter.

### CSV → slot field mapping

| CSV column | Slot field |
|---|---|
| `locID` | `public_data_id_sidewalk_{side}_curbramp_{pos}_{n}` |
| `curbReturnLoc` | `sidewalk_{side}_curbramp_{pos}_{n}_returnloc` |
| `positionOnReturn` | `sidewalk_{side}_curbramp_{pos}_{n}_returnposition` |
| `conditionScore` | `sidewalk_{side}_curbramp_{pos}_{n}_condition_score` |

## Quality Values

`sidewalk_{side}_curbramp_{pos}_{n}_quality` takes two values:

- **`"topology"`** — set by step 12 at the moment a ramp point is written (default for all
  topology-placed ramps; replaces the current `None`).
- **`"government"`** — set by step 17 when a slot is matched to or created from the CSV.

Step 17 does **not** perform a retroactive backfill; step 12 is the single source of truth
for topology quality assignment.

## Algorithm — `step_17_enrich_curb_ramps`

### Graceful skip conditions

Return `gdf` unchanged (log info) if:
- `config.curb_ramp_data_path` is empty
- File not found
- Zero rows survive the `crExist == "1"` filter

### Spatial index

Collect every non-null `sidewalk_{side}_curbramp_{pos}_{n}_geometry` Point across all road
rows into a flat list of `(road_idx, side, pos, n, utm_point)` tuples. Build one `STRtree`
over the UTM-projected versions of those points.

### Per-CSV-ramp matching

For each valid CSV point (projected to UTM EPSG:32610):

**Match found** (nearest existing slot within `curb_ramp_match_radius_m`):
- Enrich that slot with the CSV field mapping above.
- Set `_quality = "government"`.

**No match** (new slot):
1. Find the nearest road row via an STRtree on road geometries.
2. **`pos`**: compare UTM distance to `start_node_geometry` vs. `end_node_geometry`; take
   the nearer end.
3. **`side`**: project the CSV point onto the road geometry; cross-product of the nearest
   road-segment vector against the offset vector from the projection point to the CSV point.
   Positive cross = `"left"`, negative = `"right"`.
4. **`n`**: first of `(1, 2, 3)` whose `_geometry` slot is null for that
   `(road_idx, side, pos)` triple. If all three are occupied, skip and log a warning.
5. Write `_geometry` (WGS84 Point), `_quality = "government"`, and all CSV field mappings.
6. Assign `_ID` using the same incrementing counter pattern as step 12.

### Step 12 change

At the point where step 12 writes a ramp geometry to a slot, also set
`sidewalk_{side}_curbramp_{pos}_{n}_quality = "topology"`.

## Pipeline placement

```
step_15_crosswalk_geometries
step_16_consolidate_orphaned_facilities
step_11_facility_grid_ids   ← re-run
step_17_enrich_curb_ramps   ← new
# pre-write normalization
```
