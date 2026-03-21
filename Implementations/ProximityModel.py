import math
import pickle
import re
import numpy as np
import osmnx as ox
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry

# Pre-compiled regex for extracting a leading numeric value from OSM tag strings
# (e.g. "25 mph", "3.5 m", "-3.2%").  Compiled once at import time.
_NUMERIC_PREFIX_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)")

# --- Global Config ---
OUTPUT_DIR = Path("Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TODO: REMOVE BEFORE PRODUCTION - temporary OSM graph cache for faster testing
CACHE_DIR = Path("Implementations") / ".osm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Maximum distance (metres) for a separate facility edge to be considered
# coincident with the street centerline.  Edges within this threshold for ≥95%
# of their length are treated as centerline data (buffered offset replaces the
# original geometry).  Increase to catch more OSM tagging errors; decrease to
# preserve close-but-genuinely-separate facilities.
CENTERLINE_COINCIDENCE_THRESHOLD_M = 2.0

# Maximum distance (metres) from a street centerline to a separate sidewalk's
# midpoint before buffering is suppressed on that side.  A parallel separate
# sidewalk within this radius indicates the street already has sidewalk geometry
# and does not need a buffered offset.
NEARBY_SEPARATE_SIDEWALK_THRESHOLD_M = 20.0

# When True, query USGS 3DEP to compute street_incline (slope %) for each
# segment.  Requires network access.  Set to False to skip API calls.
ENRICH_TOPOGRAPHY = True

# Resolution of the USGS 3DEP DEM tile in metres.  Segments shorter than this
# threshold produce unreliable slope values (nearest-neighbour sampling of
# adjacent DEM pixels can fabricate large elevation jumps), so they are
# excluded from direct slope computation and filled via neighbor propagation.
DEM_RESOLUTION_M = 10.0

# Hard cap on computed slope values (%).  Slopes beyond ±SLOPE_CAP_PCT are
# almost certainly DEM artifacts (no urban street exceeds ~40% grade) and are
# set to None so they remain available for crowdsourced correction.
SLOPE_CAP_PCT = 40.0

# --- Cities Config ---
CITIES_CONFIG = [
    {"place": "San Francisco County, California, USA", "default_lane_width_m": 3.5, "default_maxspeed": 25},
    {"place": "Alameda County, California, USA",       "default_lane_width_m": 3.5, "default_maxspeed": 25},
]


# --- Grid Pipeline ---

# Highway types that are not driveable roads and should be excluded from
# intersection-degree counting and bearing normalisation.
_NON_ROAD_HIGHWAY = frozenset({
    "cycleway", "footway", "pedestrian", "path",
    "steps", "corridor", "bridleway",
})


@dataclass
class GridCache:
    origin_x: float      # SW corner x (EPSG:32610 metres)
    origin_y: float      # SW corner y
    cell_size: float     # 500.0
    n_cols: int
    n_rows: int


@dataclass
class GridResult:
    edge_bearings: dict[int, float]
    # row_index → normalized_bearing (0-359°)
    edge_grid_ids: dict[int, str]
    # row_index → "col_row_seq" grid ID
    intersection_node_ids: set[int]
    # nodes with degree >= 3 in the road graph
    grid: GridCache


def _flatten_coords(geom: BaseGeometry) -> list[tuple[float, float]]:
    """Flatten a LineString or MultiLineString to a single coordinate list."""
    if isinstance(geom, MultiLineString):
        return [(c[0], c[1]) for ls in geom.geoms for c in ls.coords]
    if hasattr(geom, "coords"):
        return [(c[0], c[1]) for c in geom.coords]
    return []


def _find_deflection_split_points(
    coords: list[tuple[float, float]], threshold_rad: float
) -> list[int]:
    """Return coordinate indices of ALL internal vertices where the deflection
    angle exceeds *threshold_rad*.

    Deflection at vertex i depends only on coords[i-1], coords[i], coords[i+1],
    so splitting elsewhere in the segment does not change the result.  This
    allows a single-pass approach instead of recursive max-only splitting.
    """
    if len(coords) < 3:
        return []

    coords_arr = np.array(coords, dtype=np.float64)

    incoming = coords_arr[1:-1] - coords_arr[:-2]
    outgoing = coords_arr[2:] - coords_arr[1:-1]

    incoming_len = np.linalg.norm(incoming, axis=1)
    outgoing_len = np.linalg.norm(outgoing, axis=1)

    valid = (incoming_len >= 1e-9) & (outgoing_len >= 1e-9)
    if not np.any(valid):
        return []

    dot = np.where(valid,
                   np.sum(incoming * outgoing, axis=1) / (incoming_len * outgoing_len),
                   1.0)
    dot = np.clip(dot, -1.0, 1.0)
    deflections = np.arccos(dot)

    # Internal vertex index 0 → coord index 1, etc.
    exceed_mask = deflections > threshold_rad
    return (np.flatnonzero(exceed_mask) + 1).tolist()


def split_deflected_segments(
    edges_reset: gpd.GeoDataFrame,
    deflection_threshold_deg: float = 45.0,
) -> gpd.GeoDataFrame:
    """Split segments at ALL internal vertices where the deflection angle
    exceeds the threshold in a single pass (no recursion).

    Split node IDs are negative sequential integers: -1, -2, ...
    All pieces share the parent's attributes except u/v/geometry.
    """
    from shapely.geometry import LineString as _LS

    threshold_rad = math.radians(deflection_threshold_deg)
    split_counter = 0
    rows_to_drop: list[int] = []
    new_rows: list[dict[str, Any]] = []

    for idx in edges_reset.index:
        geom = edges_reset.at[idx, "geometry"]
        if geom is None:
            continue
        coords = _flatten_coords(cast(BaseGeometry, geom))
        if len(coords) < 3:
            continue

        split_indices = _find_deflection_split_points(coords, threshold_rad)
        if not split_indices:
            continue

        row_data = edges_reset.loc[idx].to_dict()
        original_u = row_data.get("u")
        original_v = row_data.get("v")

        # Build split boundaries: [0, split_idx_1, split_idx_2, ..., len-1]
        boundaries = [0] + split_indices + [len(coords) - 1]
        rows_to_drop.append(idx)

        for i in range(len(boundaries) - 1):
            piece = dict(row_data)
            start_b = boundaries[i]
            end_b = boundaries[i + 1]
            piece["geometry"] = _LS(coords[start_b: end_b + 1])

            # Assign u/v: first piece keeps original u, last keeps original v,
            # intermediate boundaries get synthetic node IDs.
            if i == 0:
                piece["u"] = original_u
            else:
                piece["u"] = split_counter

            if i == len(boundaries) - 2:
                piece["v"] = original_v
            else:
                split_counter -= 1
                piece["v"] = split_counter

            new_rows.append(piece)

    if not new_rows:
        print(f"Deflection splitting: 0 segments split (threshold={deflection_threshold_deg}°)")
        return edges_reset

    n_segments_split = len(rows_to_drop)
    result = edges_reset.drop(index=rows_to_drop)
    new_gdf = gpd.GeoDataFrame(new_rows, crs=edges_reset.crs)
    result = gpd.GeoDataFrame(
        pd.concat([result, new_gdf], ignore_index=True),
        crs=edges_reset.crs,
    )

    print(f"Deflection splitting: {n_segments_split} segments → "
          f"{len(new_rows)} pieces (threshold={deflection_threshold_deg}°)")

    return result


def compute_grid_assignments(
    edges_reset: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    G: Any,
) -> GridResult:
    """Compute grid IDs, normalized bearings, and intersection nodes.

    Replaces the old detect_blocks() with a simpler pipeline:
      1. Build 500m grid from node bounding box
      2. Compute raw bearings per segment
      3. Normalize bearings (one-way: direction of travel; two-way: Strategy C
         majority vote by street name, fallback northward preference)
      4. Assign grid IDs (col_row_seq) by segment midpoint
      5. Detect intersection nodes (degree >= 3)
    """
    with tqdm(total=5, desc="Grid assignments", unit="step") as pbar:
        # --- 1. Grid setup ---
        xs = np.array(nodes.geometry.x.values, dtype=np.float64)
        ys = np.array(nodes.geometry.y.values, dtype=np.float64)
        cell = 500.0
        ox_g = float(xs.min())
        oy_g = float(ys.min())
        n_cols = int(math.ceil((float(xs.max()) - ox_g) / cell)) + 1
        n_rows = int(math.ceil((float(ys.max()) - oy_g) / cell)) + 1
        grid = GridCache(origin_x=ox_g, origin_y=oy_g, cell_size=cell,
                         n_cols=n_cols, n_rows=n_rows)
        pbar.update(1)

        # --- 2. Raw bearings ---
        raw_bearings: dict[int, float] = {}
        pbar.set_postfix_str("computing raw bearings")
        for idx in edges_reset.index:
            geom = edges_reset.at[idx, "geometry"]
            if geom is not None:
                b = _linestring_bearing(cast(BaseGeometry, geom))
                if b is not None:
                    raw_bearings[idx] = b
        pbar.update(1)

        # --- 3. Bearing normalization ---
        pbar.set_postfix_str("normalizing bearings")
        # Determine one-way status per row
        oneway_col = edges_reset["oneway"] if "oneway" in edges_reset.columns else pd.Series(
            False, index=edges_reset.index
        )

        edge_bearings: dict[int, float] = {}

        # One-way segments: use direction of travel
        for idx in edges_reset.index:
            if idx not in raw_bearings:
                continue
            ow = oneway_col.at[idx]
            if ow in (True, "yes", "1", 1):
                # Forward one-way: raw bearing is direction of travel
                edge_bearings[idx] = raw_bearings[idx]
            elif ow in ("-1", "reverse"):
                # Reverse one-way: flip by 180°
                edge_bearings[idx] = (raw_bearings[idx] + 180.0) % 360

        # Two-way segments: Strategy C — majority vote by name
        name_col = edges_reset["name"] if "name" in edges_reset.columns else pd.Series(
            None, index=edges_reset.index
        )

        # Collect two-way segment indices (not yet assigned a bearing)
        twoway_indices = [idx for idx in edges_reset.index
                          if idx in raw_bearings and idx not in edge_bearings]

        # Group by name for majority vote
        name_groups: dict[str, list[int]] = {}
        unnamed_indices: list[int] = []
        for idx in twoway_indices:
            name = name_col.at[idx]
            if name is not None and not (isinstance(name, float) and math.isnan(name)):
                name_str = str(name).strip()
                if name_str:
                    name_groups.setdefault(name_str, []).append(idx)
                    continue
            unnamed_indices.append(idx)

        # Majority vote per named group
        for name, indices in name_groups.items():
            north_count = sum(1 for i in indices if raw_bearings[i] < 180.0)
            south_count = len(indices) - north_count
            if north_count >= south_count:
                # Majority is [0, 180): flip any in [180, 360)
                for i in indices:
                    b = raw_bearings[i]
                    edge_bearings[i] = (b + 180.0) % 360 if b >= 180.0 else b
            else:
                # Majority is [180, 360): flip any in [0, 180)
                for i in indices:
                    b = raw_bearings[i]
                    edge_bearings[i] = (b + 180.0) % 360 if b < 180.0 else b

        # Unnamed / unassigned: infer from adjacent named segments, then northward fallback
        # Build node → [edge_index, …] lookup so we can find neighbors efficiently
        node_to_edge_indices: dict[Any, list[int]] = {}
        if "u" in edges_reset.columns and "v" in edges_reset.columns:
            for node_col in ("u", "v"):
                for node_id, group_idxs in edges_reset.groupby(node_col).groups.items():
                    lst = node_to_edge_indices.setdefault(node_id, [])
                    lst.extend(group_idxs.tolist())

        northward_fallback_indices: list[int] = []
        for idx in unnamed_indices:
            b = raw_bearings[idx]
            inferred = _infer_bearing_from_named_neighbors(
                idx, b, name_col, node_to_edge_indices, edge_bearings, edges_reset
            )
            if inferred is not None:
                edge_bearings[idx] = inferred
            else:
                edge_bearings[idx] = (b + 180.0) % 360 if b >= 180.0 else b
                northward_fallback_indices.append(idx)
        pbar.update(1)

        # --- 4. Grid ID assignment ---
        pbar.set_postfix_str("assigning grid IDs")
        # Batch-extract centroids for all edges at once, then compute col/row
        # with vectorized integer arithmetic instead of per-row .at[] access.
        centroids = edges_reset.geometry.centroid
        cx_vals = np.asarray(centroids.x, dtype=np.float64)
        cy_vals = np.asarray(centroids.y, dtype=np.float64)
        col_arr = np.floor((cx_vals - ox_g) / cell).astype(int)
        row_arr = np.floor((cy_vals - oy_g) / cell).astype(int)
        # Sequential counter per cell still needs a small Python loop since
        # each row's sequence number depends on how many prior rows share its cell.
        cell_seq: dict[tuple[int, int], int] = {}
        seqs: list[int] = []
        for c, r in zip(col_arr.tolist(), row_arr.tolist()):
            key = (c, r)
            seq = cell_seq.get(key, 0)
            cell_seq[key] = seq + 1
            seqs.append(seq)
        seqs_arr = np.array(seqs, dtype=int)
        grid_id_strs = (
            pd.Series(col_arr, index=edges_reset.index).astype(str)
            + "_"
            + pd.Series(row_arr, index=edges_reset.index).astype(str)
            + "_"
            + pd.Series(seqs_arr, index=edges_reset.index).astype(str)
        )
        edge_grid_ids: dict[int, str] = cast(dict[int, str], grid_id_strs.to_dict())
        pbar.update(1)

        # --- 5. Intersection detection ---
        pbar.set_postfix_str("detecting intersections")
        def _is_non_road(data: dict[str, Any]) -> bool:
            hw = data.get("highway", "")
            hw_vals = hw if isinstance(hw, list) else [hw]
            return bool(frozenset(str(h) for h in hw_vals) & _NON_ROAD_HIGHWAY)

        node_neighbors: dict[int, set[int]] = {}
        for u, v, data in G.edges(data=True):
            if _is_non_road(data):
                continue
            node_neighbors.setdefault(u, set()).add(v)
            node_neighbors.setdefault(v, set()).add(u)
        intersection_node_ids = {nid for nid, nbrs in node_neighbors.items() if len(nbrs) >= 3}
        pbar.update(1)

    print(f"Grid assignments: {len(edge_grid_ids)} segments across "
          f"{len(cell_seq)} grid cells ({n_cols}×{n_rows} @ {cell:.0f}m) "
          f"| {len(intersection_node_ids)} intersection nodes")
    n_oneway = sum(1 for idx in edges_reset.index
                   if idx in edge_bearings and oneway_col.at[idx] in (True, "yes", "1", 1, "-1", "reverse"))
    n_named = sum(len(v) for v in name_groups.values())
    n_inferred = len(unnamed_indices) - len(northward_fallback_indices)
    print(f"  Bearings: {n_oneway} one-way, {n_named} named two-way (majority vote), "
          f"{n_inferred} unnamed (neighbor-inferred), "
          f"{len(northward_fallback_indices)} unnamed (northward fallback)")

    return GridResult(
        edge_bearings=edge_bearings,
        edge_grid_ids=edge_grid_ids,
        intersection_node_ids=intersection_node_ids,
        grid=grid,
    )


def _populate_grid_columns(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
    grid_result: GridResult,
) -> None:
    """Write grid-derived columns into the schema GeoDataFrame (in-place)."""
    idx = edges_reset.index
    intersection_ids = grid_result.intersection_node_ids

    populated["street_grid_id"] = pd.Series(grid_result.edge_grid_ids, dtype=object).reindex(idx)
    populated["normalized_bearing"] = pd.Series(grid_result.edge_bearings, dtype=float).reindex(idx)

    # Synthetic split nodes have negative IDs and are never in intersection_ids
    # (which contains only positive OSM node IDs), so .isin() handles them correctly.
    populated["start_node_is_intersection_node"] = edges_reset["u"].isin(intersection_ids)
    populated["end_node_is_intersection_node"] = edges_reset["v"].isin(intersection_ids)


def swap_facilities_by_bearing(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Swap left/right facility columns for edges whose OSM geometry is stored
    in reverse order relative to the graph direction u→v.

    When normalized_bearing (u→v from graph nodes) and the raw geometry bearing
    (coords[0]→coords[-1]) differ by ~180° (±30°), the OSM sidewalk/cycleway
    tags were authored relative to the reversed geometry and need to be re-mapped.
    """
    # Identify left↔right swap pairs from schema columns
    cols = set(populated.columns)
    swap_pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for col in populated.columns:
        if "_left_" in col:
            right_col = col.replace("_left_", "_right_", 1)
        elif col.endswith("_left"):
            right_col = col[: -len("_left")] + "_right"
        else:
            continue
        if right_col in cols and right_col not in seen:
            swap_pairs.append((col, right_col))
            seen.add(right_col)

    # Compute bearings as numeric Series
    norm_bear = pd.to_numeric(populated["normalized_bearing"], errors="coerce")
    raw_bear = pd.to_numeric(
        edges_reset["geometry"].apply(_linestring_bearing), errors="coerce"
    )

    both_valid = norm_bear.notna() & raw_bear.notna()
    diff = (norm_bear - raw_bear).abs() % 360
    diff_sym = diff.where(diff <= 180, 360 - diff)
    # ~180° difference (within 30° tolerance) → geometry is reversed
    is_reversed = (diff_sym >= 150) & both_valid

    n_swapped = int(is_reversed.sum())
    if n_swapped > 0:
        for left_col, right_col in swap_pairs:
            tmp = populated.loc[is_reversed, left_col].copy()
            populated.loc[is_reversed, left_col] = populated.loc[is_reversed, right_col].values
            populated.loc[is_reversed, right_col] = tmp.values

    # Diagnostics
    n_both = int(both_valid.sum())
    n_norm_only = int(norm_bear.notna().sum() - n_both)
    n_raw_only = int(raw_bear.notna().sum() - n_both)
    n_neither = int((~norm_bear.notna() & ~raw_bear.notna()).sum())
    if n_both > 0:
        diffs_valid = diff_sym[both_valid]
        mean_diff = float(diffs_valid.mean())
        max_diff = float(diffs_valid.max())
        n_exact = int((diffs_valid < 0.1).sum())
        n_small = int(((diffs_valid >= 0.1) & (diffs_valid < 30)).sum())
        n_mid = int(((diffs_valid >= 30) & (diffs_valid < 150)).sum())
        n_rev = int((diffs_valid >= 150).sum())
        print(f"Facility bearing swap: {n_swapped} edges swapped left↔right "
              f"(pairs={len(swap_pairs)}, compared={n_both}, "
              f"norm_only={n_norm_only}, raw_only={n_raw_only}, neither={n_neither})")
        print(f"  Bearing diff distribution: exact(<0.1°)={n_exact}, "
              f"small(0.1-30°)={n_small}, mid(30-150°)={n_mid}, "
              f"reversed(>=150°)={n_rev} | mean={mean_diff:.1f}°, max={max_diff:.1f}°")
    else:
        print(f"Facility bearing swap: no edges with both bearings available "
              f"(norm_only={n_norm_only}, raw_only={n_raw_only}, neither={n_neither})")
    return populated


# ---------------------------------------------------------------------------
# Topography enrichment  (pipeline step 5)
# ---------------------------------------------------------------------------

def _parse_incline(val: object) -> float | None:
    """Convert an OSM incline string (e.g. '5%', '-3.2%', 'up', 'down') to a
    float percentage.  Returns None for unparseable values."""
    if val is None:
        return None
    try:
        if pd.isna(val):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip().lower()
    if not s:
        return None
    s = s.replace("°", "").replace("%", "").strip()
    if s in ("up",):
        return None  # direction-only, no magnitude
    if s in ("down",):
        return None
    try:
        return round(float(s), 4)
    except ValueError:
        return None


def _parse_maxspeed(val: object) -> int | None:
    """Parse an OSM maxspeed value to int, discarding any unit suffix."""
    if val is None:
        return None
    try:
        if pd.isna(val):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s:
        return None
    # Strip common unit suffixes: "25 mph", "40 km/h", "30 knots"
    m = _NUMERIC_PREFIX_RE.match(s)
    if m:
        return int(float(m.group(1)))
    return None


def _parse_int_tag(val: object) -> int | None:
    """Parse an OSM tag value to int (e.g. lanes='2')."""
    if val is None:
        return None
    try:
        if pd.isna(val):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(val))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_float_tag(val: object) -> float | None:
    """Parse an OSM tag value to float (e.g. width='3.5')."""
    if val is None:
        return None
    try:
        if pd.isna(val):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    # Strip unit suffixes like "3.5 m", "12 ft"
    m = _NUMERIC_PREFIX_RE.match(s)
    if m:
        return float(m.group(1))
    return None


# Bounding box for contiguous US (USGS 3DEP coverage)
_US_BBOX = {"minx": -125.0, "miny": 24.0, "maxx": -66.9, "maxy": 49.4}


def _point_in_us(lon: float, lat: float) -> bool:
    return (_US_BBOX["minx"] <= lon <= _US_BBOX["maxx"]
            and _US_BBOX["miny"] <= lat <= _US_BBOX["maxy"])


def _query_dem_tile(
    us_pts: list[tuple[float, float]],
) -> dict[tuple[float, float], float | None]:
    """Fetch a USGS 3DEP DEM tile covering all points and sample elevations.

    Downloads a single raster tile (10 m resolution) covering the bounding box
    of *us_pts*, then samples each point via nearest-neighbour interpolation.
    Vastly faster than per-point EPQS queries for geographically dense inputs.

    NOTE: py3dep always returns the DEM in EPSG:5070 (NAD83 / Conus Albers),
    so query points must be transformed to EPSG:5070 before interpolation.
    """
    import py3dep
    import xarray as xr
    from pyproj import Transformer

    if not us_pts:
        return {}

    lons = np.array([p[0] for p in us_pts], dtype=np.float64)
    lats = np.array([p[1] for p in us_pts], dtype=np.float64)

    # Small buffer so edge points aren't clipped by the tile boundary (~100 m)
    buf = 0.001
    bbox = (float(lons.min()) - buf, float(lats.min()) - buf,
            float(lons.max()) + buf, float(lats.max()) + buf)

    # --- Fetch DEM tile (single network call) ---
    bbox_str = f"({bbox[0]:.4f},{bbox[1]:.4f}) → ({bbox[2]:.4f},{bbox[3]:.4f})"
    with tqdm(total=1, desc="Topography DEM fetch", unit="tile",
              postfix={"bbox": bbox_str}) as pbar:
        dem: xr.DataArray = py3dep.get_dem(bbox, crs="EPSG:4326", resolution=10)
        pbar.update(1)

    # py3dep returns DEM in EPSG:5070 — transform query points to match
    to_5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs_5070, ys_5070 = to_5070.transform(lons, lats)

    # --- Sample elevations in batches ---
    BATCH_SIZE = 50_000
    n_pts = len(us_pts)
    sampled = np.empty(n_pts, dtype=np.float64)

    with tqdm(total=n_pts, desc="Topography elevation sampling",
              unit="pt", unit_scale=True) as pbar:
        for start in range(0, n_pts, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_pts)
            xs = xr.DataArray(xs_5070[start:end], dims="points")
            ys = xr.DataArray(ys_5070[start:end], dims="points")
            sampled[start:end] = dem.interp(x=xs, y=ys, method="nearest").values
            pbar.update(end - start)

    result: dict[tuple[float, float], float | None] = {}
    for pt, val in zip(us_pts, sampled):
        result[pt] = None if (np.isnan(val) or float(val) < -1000) else round(float(val), 3)

    valid_elevs = [v for v in result.values() if v is not None]
    n_missing = len(result) - len(valid_elevs)
    if valid_elevs:
        print(f"  Topography DEM: {len(valid_elevs)} elevations sampled "
              f"({n_missing} NaN/clipped), "
              f"range {min(valid_elevs):.1f}–{max(valid_elevs):.1f} m")
    else:
        print(f"  Topography DEM: no valid elevations returned — "
              f"all {len(result)} points were NaN or clipped")

    return result


def _query_dem_by_grid(
    us_pts: list[tuple[float, float]],
    pt_to_cell: dict[tuple[float, float], tuple[int, int]],
    grid: "GridCache",
    src_crs: Any,
) -> dict[tuple[float, float], float | None]:
    """Fetch USGS 3DEP DEM tiles per 500m grid cell and sample elevations.

    One tile is fetched per occupied grid cell, giving reproducible elevation
    values independent of the total query bounding box.  Tiles are cached in
    memory for the duration of the call.
    """
    import gc
    import py3dep
    import xarray as xr
    from collections import defaultdict
    from pyproj import Transformer

    if not us_pts:
        return {}

    BUF_M = 100.0  # metre buffer added to each cell edge before fetching

    to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    to_5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    # Group unique US points by their grid cell
    cell_to_pts: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for pt in us_pts:
        cell = pt_to_cell.get(pt)
        if cell is not None:
            cell_to_pts[cell].append(pt)

    result: dict[tuple[float, float], float | None] = {}
    n_valid = n_nan = 0
    n_cells = len(cell_to_pts)

    with tqdm(total=n_cells, desc="Topography DEM (grid cells)", unit="cell") as pbar:
        for (col, row), pts in cell_to_pts.items():
            # Compute cell bbox in native CRS with buffer
            x_min = grid.origin_x + col * grid.cell_size - BUF_M
            x_max = grid.origin_x + (col + 1) * grid.cell_size + BUF_M
            y_min = grid.origin_y + row * grid.cell_size - BUF_M
            y_max = grid.origin_y + (row + 1) * grid.cell_size + BUF_M

            # Transform all four corners to WGS-84 and take the envelope
            corn_x = [x_min, x_max, x_min, x_max]
            corn_y = [y_min, y_min, y_max, y_max]
            lons_c, lats_c = to_wgs84.transform(corn_x, corn_y)
            bbox_wgs = (
                float(min(lons_c)), float(min(lats_c)),
                float(max(lons_c)), float(max(lats_c)),
            )

            try:
                dem: xr.DataArray = py3dep.get_dem(bbox_wgs, crs="EPSG:4326", resolution=10)
                dem = dem.astype(np.float32)  # halve in-memory footprint
            except Exception as exc:
                for pt in pts:
                    result[pt] = None
                pbar.update(1)
                print(f"\n  DEM fetch failed for cell ({col},{row}): {exc}")
                continue

            # Vectorized sampling for all points in this cell
            lons_pts = np.array([p[0] for p in pts], dtype=np.float64)
            lats_pts = np.array([p[1] for p in pts], dtype=np.float64)
            xs_5070, ys_5070 = to_5070.transform(lons_pts, lats_pts)
            xs_da = xr.DataArray(xs_5070, dims="points")
            ys_da = xr.DataArray(ys_5070, dims="points")
            sampled = dem.interp(x=xs_da, y=ys_da, method="nearest").values
            del dem, xs_da, ys_da

            for pt, val in zip(pts, sampled):
                if np.isnan(val) or float(val) < -1000:
                    result[pt] = None
                    n_nan += 1
                else:
                    result[pt] = round(float(val), 3)
                    n_valid += 1
            pbar.update(1)
            if n_valid % 10_000 < len(pts):
                gc.collect()

    print(f"  Topography DEM: {n_valid} elevations sampled ({n_nan} NaN/clipped) "
          f"across {n_cells} grid cells")
    return result


def _query_dem_by_native_tile(
    us_pts: list[tuple[float, float]],
) -> dict[tuple[float, float], float | None]:
    """Fetch USGS 3DEP DEM tiles grouped by 0.1° fixed grid cells (~10 km × ~10 km).

    Groups all query points into 0.1° × 0.1° cells and fetches one DEM per
    occupied cell.  A small border buffer is added to each cell bbox so points
    near cell edges are never clipped by the returned raster.
    """
    import gc
    import math
    import py3dep
    import xarray as xr
    from collections import defaultdict
    from pyproj import Transformer

    if not us_pts:
        return {}

    CELL_DEG   = 0.1    # ~10 km per cell side at mid-latitudes
    BUFFER_DEG = 0.01   # ~1 km border so boundary points aren't clipped

    to_5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    lons = np.array([p[0] for p in us_pts], dtype=np.float64)
    lats = np.array([p[1] for p in us_pts], dtype=np.float64)
    xs_5070, ys_5070 = to_5070.transform(lons, lats)

    # ── Group endpoints by 0.1° cell ─────────────────────────────────────────
    tile_to_idxs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (lon, lat) in enumerate(us_pts):
        col = math.floor(lon / CELL_DEG)
        row = math.floor(lat / CELL_DEG)
        tile_to_idxs[(col, row)].append(i)

    result: dict[tuple[float, float], float | None] = {}
    n_valid = n_nan = 0
    n_cells = len(tile_to_idxs)

    print(f"  Topography DEM: {len(us_pts)} endpoints → {n_cells} × {CELL_DEG}° cells")

    with tqdm(total=n_cells, desc="Topography DEM (grid cells)", unit="cell") as pbar:
        for (tc, tr), idxs in tile_to_idxs.items():
            bbox_wgs = (
                tc * CELL_DEG - BUFFER_DEG,
                tr * CELL_DEG - BUFFER_DEG,
                (tc + 1) * CELL_DEG + BUFFER_DEG,
                (tr + 1) * CELL_DEG + BUFFER_DEG,
            )

            try:
                dem = py3dep.get_dem(bbox_wgs, crs="EPSG:4326", resolution=10)
                dem = dem.astype(np.float32)  # halve in-memory footprint
            except Exception as exc:
                for i in idxs:
                    result[us_pts[i]] = None
                    n_nan += 1
                pbar.update(1)
                print(f"\n  DEM fetch failed for cell ({tc},{tr}): {exc}")
                continue

            idx_arr = np.array(idxs, dtype=np.intp)
            xs_da = xr.DataArray(xs_5070[idx_arr], dims="points")
            ys_da = xr.DataArray(ys_5070[idx_arr], dims="points")
            vals = dem.interp(x=xs_da, y=ys_da, method="nearest").values
            del dem, xs_da, ys_da
            gc.collect()

            for i, val in zip(idxs, vals):
                if np.isnan(val) or float(val) < -1000:
                    result[us_pts[i]] = None
                    n_nan += 1
                else:
                    result[us_pts[i]] = round(float(val), 3)
                    n_valid += 1
            pbar.update(1)

    print(f"  Topography DEM: {n_valid} elevations sampled ({n_nan} NaN/clipped) "
          f"across {n_cells} cells")
    return result


def _bfs_nearest_slope(
    pos: int,
    slopes: "np.ndarray[Any, Any]",
    start_ids: list[Any],
    end_ids: list[Any],
    node_to_positions: dict[Any, list[int]],
    node_to_lonlat: dict[Any, tuple[float, float]],
    cx: float,
    cy: float,
    dists: "np.ndarray[Any, Any]",
    max_dist_m: float,
) -> float | None:
    """Dijkstra through the street network from the segment at *pos*.

    Returns the slope of the nearest connected segment (by network path
    distance) that already has a non-NaN slope, provided the shared node
    is within *max_dist_m* straight-line distance from (cx, cy) — the WGS-84
    centroid of the query segment.  Returns None if no such segment exists.
    """
    import heapq

    u0, v0 = start_ids[pos], end_ids[pos]
    heap: list[tuple[float, Any]] = []
    visited: set[Any] = set()

    for node in (u0, v0):
        if pd.notna(node):
            heapq.heappush(heap, (0.0, node))

    while heap:
        net_dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        # Straight-line cutoff: don't expand from nodes beyond max_dist_m
        if node in node_to_lonlat:
            nlon, nlat = node_to_lonlat[node]
            if _haversine_m(cx, cy, nlon, nlat) > max_dist_m:
                continue

        for nbr_pos in node_to_positions.get(node, []):
            if nbr_pos == pos:
                continue
            slope = slopes[nbr_pos]
            if not np.isnan(slope):
                return float(slope)
            # No slope yet — traverse through this segment to its far node
            seg_len = float(dists[nbr_pos])
            nu, nv = start_ids[nbr_pos], end_ids[nbr_pos]
            far = nv if nu == node else nu
            if pd.notna(far) and far not in visited:
                heapq.heappush(heap, (net_dist + seg_len, far))

    return None


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Horizontal distance between two WGS-84 points in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _haversine_m_vectorized(
    lon1: np.ndarray, lat1: np.ndarray,
    lon2: np.ndarray, lat2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance between arrays of WGS-84 points in metres."""
    R = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def enrich_topography(
    populated: gpd.GeoDataFrame,
    place: str,
) -> gpd.GeoDataFrame:
    """Compute street_incline (slope %) for every segment using USGS 3DEP.

    Works in WGS-84: reprojects the active geometry to EPSG:4326, extracts
    start/end points, deduplicates, fetches one DEM per native 3DEP tile
    (reproducible across runs), then computes rise / run × 100.  Non-US
    points are skipped (left as NA).

    Short segments (<DEM_RESOLUTION_M) receive their slope via a Dijkstra
    BFS through the street network that accepts the nearest connected segment
    (up to MAX_PROP_DIST_M straight-line distance) that already has a slope.
    """
    from pyproj import Transformer

    MAX_PROP_DIST_M = 250.0

    src_crs = populated.crs
    geom_col = populated.geometry.name

    # Build a WGS-84 transformer if needed
    if src_crs is not None and src_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    else:
        transformer = None

    # Vectorized extraction of start/end coordinates from all geometries
    geom_series = populated[geom_col]
    start_x = np.array([g.coords[0][0] for g in geom_series], dtype=np.float64)
    start_y = np.array([g.coords[0][1] for g in geom_series], dtype=np.float64)
    end_x = np.array([g.coords[-1][0] for g in geom_series], dtype=np.float64)
    end_y = np.array([g.coords[-1][1] for g in geom_series], dtype=np.float64)

    # Batch-transform all coordinates at once (pyproj supports arrays)
    if transformer is not None:
        start_x, start_y = transformer.transform(start_x, start_y)
        end_x, end_y = transformer.transform(end_x, end_y)

    # Build tuple lists for deduplication and elevation lookup
    start_lonlat = list(zip(start_x.tolist(), start_y.tolist()))
    end_lonlat = list(zip(end_x.tolist(), end_y.tolist()))

    # Deduplicate points
    all_pts = start_lonlat + end_lonlat
    unique_pts = list(set(all_pts))

    # Filter to US-only points (USGS 3DEP coverage)
    us_pts = [p for p in unique_pts if _point_in_us(p[0], p[1])]
    non_us = len(unique_pts) - len(us_pts)
    if non_us > 0:
        print(f"  Topography: skipping {non_us} non-US points (no 3DEP coverage)")

    print(f"  Topography: {len(populated)} segments → {len(us_pts)} unique US endpoints to query")

    # Fetch elevations grouped by native 3DEP tile for reproducibility.
    # Cache the result to disk so re-runs skip the 3DEP API entirely.
    place_slug = place.replace(", ", "_").replace(" ", "_")
    elev_cache_path = CACHE_DIR / f"{place_slug}_elevation.pkl"
    if elev_cache_path.exists():
        with open(elev_cache_path, "rb") as _f:
            elev_map: dict[tuple[float, float], float | None] = pickle.load(_f)
        print(f"  Topography: loaded elevation cache ({len(elev_map)} points) from {elev_cache_path}")
    else:
        elev_map = _query_dem_by_native_tile(us_pts)
        with open(elev_cache_path, "wb") as _f:
            pickle.dump(elev_map, _f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Topography: elevation cache saved to {elev_cache_path}")

    # Vectorized slope computation
    elev_start = np.array([elev_map.get(p) for p in start_lonlat], dtype=np.float64)
    elev_end = np.array([elev_map.get(p) for p in end_lonlat], dtype=np.float64)
    dists = _haversine_m_vectorized(start_x, start_y, end_x, end_y)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_slopes = (elev_end - elev_start) / dists * 100

    # Mask invalid results: missing elevation or segments shorter than DEM
    # resolution (nearest-neighbour sampling is unreliable below this distance)
    has_both_elev = np.isfinite(elev_start) & np.isfinite(elev_end)
    valid_mask = (
        has_both_elev
        & (dists >= DEM_RESOLUTION_M) & np.isfinite(raw_slopes)
    )
    raw_slopes = np.round(raw_slopes, 4)

    # Keep slopes as a numpy float64 array; NaN represents "no slope yet".
    slopes_arr = np.where(valid_mask, raw_slopes, np.nan)

    # Propagate slopes to short segments (<DEM_RESOLUTION_M) via Dijkstra BFS
    # through the street network.  Accepts the nearest connected segment (by
    # network path distance) within MAX_PROP_DIST_M straight-line radius that
    # already has a non-NaN slope.
    too_short_positions = np.flatnonzero(np.isnan(slopes_arr) & has_both_elev).tolist()
    n_propagated = 0
    if too_short_positions and "start_node_id" in populated.columns and "end_node_id" in populated.columns:
        start_ids = populated["start_node_id"].tolist()
        end_ids = populated["end_node_id"].tolist()
        node_to_positions: dict[Any, list[int]] = {}
        for pos, (u, v) in enumerate(zip(start_ids, end_ids)):
            if pd.notna(u):
                node_to_positions.setdefault(u, []).append(pos)
            if pd.notna(v):
                node_to_positions.setdefault(v, []).append(pos)

        # Build node → WGS-84 coordinate lookup for straight-line distance check
        node_to_lonlat: dict[Any, tuple[float, float]] = {}
        for i, (u, v) in enumerate(zip(start_ids, end_ids)):
            if pd.notna(u):
                node_to_lonlat.setdefault(u, start_lonlat[i])
            if pd.notna(v):
                node_to_lonlat.setdefault(v, end_lonlat[i])

        for pos in too_short_positions:
            cx = (start_lonlat[pos][0] + end_lonlat[pos][0]) / 2.0
            cy = (start_lonlat[pos][1] + end_lonlat[pos][1]) / 2.0
            slope = _bfs_nearest_slope(
                pos, slopes_arr, start_ids, end_ids,
                node_to_positions, node_to_lonlat,
                cx, cy, dists, MAX_PROP_DIST_M,
            )
            if slope is not None:
                slopes_arr[pos] = slope
                n_propagated += 1

    # Apply hard cap: slopes beyond ±SLOPE_CAP_PCT are DEM artifacts; null them
    # so they remain flagged for crowdsourced correction.
    finite_mask = ~np.isnan(slopes_arr)
    cap_mask = finite_mask & (np.abs(slopes_arr) > SLOPE_CAP_PCT)
    n_capped = int(cap_mask.sum())
    slopes_arr[cap_mask] = np.nan

    populated["street_incline"] = slopes_arr

    n_ok = int(finite_mask.sum()) - n_capped
    n_missing_elev = int((~has_both_elev).sum())
    n_too_short_total = len(too_short_positions)
    n_too_short_unfilled = n_too_short_total - n_propagated
    print(f"  Topography: slope resolved for {n_ok}/{len(populated)} segments "
          f"({n_ok / len(populated) * 100:.1f}%) | "
          f"{n_missing_elev} missing elevation, "
          f"{n_too_short_total} short (<{DEM_RESOLUTION_M:.0f} m): "
          f"{n_propagated} filled from neighbors, {n_too_short_unfilled} unfilled | "
          f"{n_capped} nulled (|slope| > {SLOPE_CAP_PCT:.0f}%)")

    valid_slopes_arr = slopes_arr[~np.isnan(slopes_arr)]
    if len(valid_slopes_arr) > 0:
        print(f"  Topography slopes: min={valid_slopes_arr.min():.2f}% max={valid_slopes_arr.max():.2f}% "
              f"mean={valid_slopes_arr.mean():.2f}% median={float(np.median(valid_slopes_arr)):.2f}% "
              f"| uphill={int((valid_slopes_arr > 0).sum())} downhill={int((valid_slopes_arr < 0).sum())} "
              f"flat={int((valid_slopes_arr == 0).sum())}")

    return populated


# ---------------------------------------------------------------------------
# Street Centerline Features – Traffic Calming
# ---------------------------------------------------------------------------

# Maximum snap distance (metres, projected CRS) for matching a traffic-calming
# node to a street segment node (u, v, or intermediate vertex).
_TC_COINCIDENCE_THRESHOLD_M = 5.0


def _query_traffic_calming_nodes(place: str) -> gpd.GeoDataFrame:
    """Query OSM for traffic_calming=* nodes inside *place* via osmnx.

    Returns a point GeoDataFrame in EPSG:4326 with columns
    ``traffic_calming`` (the raw OSM value) and ``osmid``.
    """
    tags: dict[str, bool | str | list[str]] = {"traffic_calming": True}
    gdf = ox.features_from_place(place, tags=tags)
    # features_from_place can return polygons/relations — keep only points
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    if gdf.empty:
        return gdf
    # Flatten the index (osm type / id multi-index) and keep all tag columns
    gdf = gdf.reset_index()
    for c in ("osmid", "traffic_calming"):
        if c not in gdf.columns:
            gdf[c] = pd.NA
    return gdf.copy()


def _find_coincident_segment(
    tc_point: Point,
    edges_reset: gpd.GeoDataFrame,
    edge_sindex: Any,
) -> int | None:
    """Return the edge index whose node (u, v, or intermediate vertex) is
    coincident with *tc_point*, or ``None`` if no match within threshold.

    Checks actual linestring vertices — not just nearest-line distance — so
    that only segments whose geometry passes through the traffic-calming node
    are matched.
    """
    # Query candidates within threshold bbox
    buf = tc_point.buffer(_TC_COINCIDENCE_THRESHOLD_M)
    candidate_idxs: list[int] = list(edge_sindex.query(buf, predicate="intersects"))
    if not candidate_idxs:
        return None

    best_idx: int | None = None
    best_dist = float("inf")
    for idx in candidate_idxs:
        geom: BaseGeometry = edges_reset.geometry.iloc[idx]  # type: ignore[assignment]
        for coord in geom.coords:
            vertex = Point(coord)
            d = tc_point.distance(vertex)
            if d <= _TC_COINCIDENCE_THRESHOLD_M and d < best_dist:
                best_dist = d
                best_idx = idx
    return best_idx


def populate_street_features(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
    place: str,
) -> gpd.GeoDataFrame:
    """Populate street_feature_* columns with OSM traffic calming data.

    1. Query traffic_calming nodes from OSM for *place*.
    2. Reproject to the working CRS.
    3. For each node, find the coincident street segment by vertex matching.
    4. Accumulate feature types and geometries per segment.
    5. Compute projected (snapped-to-line) geometries.
    """
    print("  Street features: querying traffic calming nodes …")
    tc_gdf = _query_traffic_calming_nodes(place)
    if tc_gdf.empty:
        print("  Street features: no traffic calming nodes found.")
        return populated

    if populated.crs is None:
        raise ValueError("populated GeoDataFrame must have a valid CRS")
    tc_gdf = tc_gdf.set_crs("EPSG:4326").to_crs(populated.crs)
    print(f"  Street features: {len(tc_gdf)} traffic calming nodes found.")

    edge_sindex = edges_reset.geometry.sindex

    # Accumulate per-segment: list of (type_str, point_geom, attrs_dict)
    seg_features: dict[int, list[tuple[str, Point, dict[str, Any]]]] = {}

    # Columns to exclude from the attributes dict
    _SKIP_COLS = frozenset({"geometry", "osmid", "element_type"})

    matched = 0
    for _, row in tqdm(tc_gdf.iterrows(), total=len(tc_gdf),
                       desc="Matching traffic calming", unit="node"):
        pt: Point = row.geometry  # type: ignore[assignment]
        tc_type: str = str(row["traffic_calming"])
        seg_idx = _find_coincident_segment(pt, edges_reset, edge_sindex)
        if seg_idx is None:
            continue
        matched += 1
        attrs: dict[str, Any] = {
            str(k): v for k, v in row.items()
            if k not in _SKIP_COLS and pd.notna(v)
        }
        seg_features.setdefault(seg_idx, []).append(
            (f"traffic_calming:{tc_type}", pt, attrs)
        )

    print(f"  Street features: matched {matched}/{len(tc_gdf)} nodes to segments "
          f"across {len(seg_features)} segments.")

    # Write into populated dataframe
    for seg_idx, features in seg_features.items():
        types  = [f[0] for f in features]
        points = [f[1] for f in features]
        attrs_list = [f[2] for f in features]
        pub_ids: list[None] = [None] * len(features)

        # Project each point onto the street linestring
        street_geom: LineString = populated.at[seg_idx, "street_geometry"]  # type: ignore[assignment]
        projected_points = [
            street_geom.interpolate(street_geom.project(p))
            for p in points
        ]

        populated.at[seg_idx, "street_feature_types"] = types  # type: ignore[index]
        populated.at[seg_idx, "public_data_id_street_feature"] = pub_ids  # type: ignore[index]
        populated.at[seg_idx, "street_feature_geometry"] = MultiPoint(points)  # type: ignore[index]
        populated.at[seg_idx, "street_feature_geometry_projected"] = MultiPoint(  # type: ignore[index]
            projected_points
        )
        populated.at[seg_idx, "street_feature_attributes"] = attrs_list  # type: ignore[index]

    return populated


#---Data Pipeline---
def populate_schema(place: str, *, default_lane_width_m: float = 3.5, default_maxspeed: int | float = 25) -> gpd.GeoDataFrame:
    """Load OSM street network for a place, map edge/node data into the proximity
    schema columns, export the result as a parquet to OUTPUT_DIR, and return it."""
    # --- Load OSM graph ---
    bike_tags = [
        "lanes", "lane_width", "maxspeed", "surface",
        # Primary bikeway tags
        "cycleway", "cycleway:left", "cycleway:right", "cycleway:both",
        "cycleway:left:surface", "cycleway:right:surface", "cycleway:surface",
        "cycleway:left:width", "cycleway:right:width", "cycleway:width",
        "cycleway:left:buffer", "cycleway:right:buffer", "cycleway:buffer",
        "cycleway:left:lane", "cycleway:right:lane",
        # Secondary bikeway tags (parallel/second bikeway on same edge)
        "cycleway:left:2", "cycleway:right:2", "cycleway:both:2",
        "cycleway:left:2:surface", "cycleway:right:2:surface",
        "cycleway:left:2:width", "cycleway:right:2:width",
        "cycleway:left:2:buffer", "cycleway:right:2:buffer",
        "cycleway:left:2:smoothness", "cycleway:right:2:smoothness",
        "bicycle", "incline",
    ]
    sidewalk_tags = [
        # Presence / side
        "sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both",
        # Surface
        "sidewalk:left:surface", "sidewalk:right:surface",
        # Width
        "sidewalk:left:width", "sidewalk:right:width",
        # Incline
        "sidewalk:left:incline", "sidewalk:right:incline",
        # Smoothness / quality
        "sidewalk:left:smoothness", "sidewalk:right:smoothness",
        # Separator strip (physical buffer between sidewalk and road)
        "sidewalk:left:buffer", "sidewalk:right:buffer",
        # Foot access permission
        "foot",
    ]
    all_extra_tags = bike_tags + sidewalk_tags
    ox.settings.useful_tags_way = ox.settings.useful_tags_way + [
        tag for tag in all_extra_tags if tag not in ox.settings.useful_tags_way
    ]
    with tqdm(total=4, desc=f"Loading street network ({place})", unit="step") as pbar:
        pbar.set_postfix_str("downloading OSM graph")
        # TODO: REMOVE BEFORE PRODUCTION - temporary caching for testing
        cache_file = CACHE_DIR / f"{place.replace('/', '_')}.pkl"
        if cache_file.exists():
            pbar.set_postfix_str("loading OSM graph from cache")
            with open(cache_file, "rb") as f:
                G = pickle.load(f)
        else:
            G = ox.graph_from_place(place, network_type="all")
            with open(cache_file, "wb") as f:
                pickle.dump(G, f)
        pbar.update(1)

        pbar.set_postfix_str("converting to GeoDataFrames")
        nodes, edges = ox.graph_to_gdfs(G)
        pbar.update(1)

        pbar.set_postfix_str("reprojecting to EPSG:32610")
        edges = edges.to_crs("EPSG:32610")
        nodes = nodes.to_crs("EPSG:32610")
        pbar.update(1)

        pbar.set_postfix_str("resetting edge index")
        edges_reset = edges.reset_index()
        pbar.update(1)

    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")

    # --- Vertex deflection splitting ---
    edges_reset = split_deflected_segments(edges_reset)

    # --- Grid assignments (bearings, grid IDs, intersection nodes) ---
    grid_result = compute_grid_assignments(edges_reset, nodes, G)

    # --- Map OSM edge/node data into schema columns ---
    schema = _create_schema_dataframe()

    def _get(col):
        return edges_reset[col] if col in edges_reset.columns else None

    def _coalesce(*cols):
        result = pd.Series(pd.NA, index=edges_reset.index, dtype=object)
        for col in cols:
            if col in edges_reset.columns:
                result = result.where(result.notna(), edges_reset[col])
        return result

    init_data: dict[str, Any] = {col: pd.NA for col in schema.columns}
    init_data["street_geometry"] = edges_reset["geometry"].values
    populated = gpd.GeoDataFrame(
        init_data,
        index=edges_reset.index,
        geometry="street_geometry",
        crs="EPSG:32610",
    )

    # Street identifiers & topology
    populated["street_id"]     = _get("osmid")
    populated["start_node_id"] = _get("u")
    populated["end_node_id"]   = _get("v")

    # OSM tags that map directly (string tags)
    for tag in ("name", "highway", "oneway", "surface"):
        populated[tag] = _get(tag)
    # Numeric tags: parse to typed columns
    raw_maxspeed = _get("maxspeed")
    if raw_maxspeed is not None:
        populated["maxspeed"] = raw_maxspeed.apply(_parse_maxspeed)
    populated["maxspeed"] = populated["maxspeed"].fillna(default_maxspeed).astype("Int64")
    raw_lanes = _get("lanes")
    if raw_lanes is not None:
        populated["lanes"] = raw_lanes.apply(_parse_int_tag)
    populated["lanes"] = populated["lanes"].astype("Int64")
    raw_lane_width = _get("lane_width")
    if raw_lane_width is not None:
        populated["lane_width"] = raw_lane_width.apply(_parse_float_tag)
    populated["lane_width"] = populated["lane_width"].astype("Float64")

    # Main street geometry (the edge LineString)
    populated["street_geometry"] = edges_reset["geometry"]

    # Start/end node point geometries looked up from the nodes GDF
    node_geom = nodes["geometry"]
    populated["start_node_geometry"] = edges_reset["u"].map(node_geom)
    populated["end_node_geometry"]   = edges_reset["v"].map(node_geom)

    # Grid columns: street_grid_id, normalized_bearing, intersection node flags
    _populate_grid_columns(populated, edges_reset, grid_result)

    # --- Street centerline features (pipeline step 3) ---
    populated = populate_street_features(populated, edges_reset, place)

    # --- Topography enrichment (pipeline step 5) ---
    if ENRICH_TOPOGRAPHY:
        populated = enrich_topography(populated, place)

    populated = populate_base_bikelanes(populated, edges_reset)
    populated = populate_base_footlanes(populated, edges_reset)
    # Swap left/right facility columns on edges with reversed geometry (deferred
    # until after tag population so all sidewalk/bikeway columns exist)
    populated = swap_facilities_by_bearing(populated, edges_reset)
    populated = _populate_separate_facilities(populated, edges_reset, default_lane_width_m)
    populated = _assign_facility_grid_ids(populated)
    populated = _assign_curb_ramp_geometries(populated)
    # --- Export ---
    # Geometry columns other than the active one must be serialized to WKB so
    # they round-trip correctly through parquet (GeoParquet only encodes the
    # active geometry column; raw Shapely objects in object columns do not survive).
    # Use:
#     from shapely import wkb
#     gdf["bikeway_left_1_geometry"] = gdf["bikeway_left_1_geometry"].apply(
#     lambda h: wkb.loads(h, hex=True) if h else None)

    SECONDARY_GEOM_COLS = [
        "start_node_geometry", "end_node_geometry",
        "sidewalk_left_geometry", "sidewalk_right_geometry", "curb_return_geometry",
        "bikeway_left_1_geometry", "bikeway_left_2_geometry",
        "bikeway_right_1_geometry", "bikeway_right_2_geometry",
        "street_feature_geometry", "street_feature_geometry_projected",
        "sidewalk_left_feature_geometry", "sidewalk_left_feature_geometry_projected",
        "sidewalk_right_feature_geometry", "sidewalk_right_feature_geometry_projected",
        "bikeway_left_1_feature_geometry", "bikeway_left_1_feature_geometry_projected",
        "bikeway_left_2_feature_geometry", "bikeway_left_2_feature_geometry_projected",
        "bikeway_right_1_feature_geometry", "bikeway_right_1_feature_geometry_projected",
        "bikeway_right_2_feature_geometry", "bikeway_right_2_feature_geometry_projected",
        "crosswalk_start_geometry", "crosswalk_start_island_geometry",
        "crosswalk_end_geometry", "crosswalk_end_island_geometry",
        "sidewalk_left_curbramp_start_1_geometry", "sidewalk_left_curbramp_start_2_geometry",
        "sidewalk_left_curbramp_start_3_geometry", "sidewalk_left_curbramp_end_1_geometry",
        "sidewalk_left_curbramp_end_2_geometry", "sidewalk_left_curbramp_end_3_geometry",
        "sidewalk_right_curbramp_start_1_geometry", "sidewalk_right_curbramp_start_2_geometry",
        "sidewalk_right_curbramp_start_3_geometry", "sidewalk_right_curbramp_end_1_geometry",
        "sidewalk_right_curbramp_end_2_geometry", "sidewalk_right_curbramp_end_3_geometry",
    ]
    present_geom_cols = [col for col in SECONDARY_GEOM_COLS if col in populated.columns]
    with tqdm(total=len(present_geom_cols) + 1, desc="Exporting parquet", unit="col") as pbar:
        # Convert geometry columns to WKB hex in-place (avoids full DataFrame copy)
        for col in present_geom_cols:
            pbar.set_postfix_str(col)
            populated[col] = populated[col].apply(
                lambda g: g.wkb_hex if hasattr(g, 'wkb_hex') else None
            )
            pbar.update(1)

        pbar.set_postfix_str("normalizing buffered columns")
        # Normalize _buffered columns: pipeline writes True/False booleans while OSM
        # tags provide strings like 'yes'/'no'. Mixed types cause PyArrow serialization
        # failures, so coerce everything to consistent strings before export.
        buffered_cols = [c for c in populated.columns if c.endswith("_buffered")]
        for col in buffered_cols:
            mask_true = populated[col] == True  # noqa: E712
            mask_false = populated[col] == False  # noqa: E712
            populated.loc[mask_true, col] = "yes"
            populated.loc[mask_false, col] = "no"

        pbar.set_postfix_str("normalizing object columns")
        # OSMnx edge simplification can produce list values in ANY tag column
        # (merged edges store multiple original values). These appear sparsely,
        # so sampling misses them. Scan all object columns exhaustively: any
        # column with at least one non-string, non-null value gets cast to str.
        geom_skip = set(present_geom_cols) | {populated.geometry.name}
        for col in populated.columns:
            if col in geom_skip:
                continue
            if populated[col].dtype != object:
                continue
            non_null = populated[col].dropna()
            if len(non_null) == 0:
                continue
            # Vectorized type check: if all non-null values are str, no action needed
            has_non_str = not non_null.apply(isinstance, args=(str,)).all()
            if has_non_str:
                mask_na = populated[col].isna()
                populated[col] = populated[col].astype(str)
                populated.loc[mask_na, col] = pd.NA

        place_slug = place.replace(", ", "_").replace(" ", "_")
        output_path = OUTPUT_DIR / f"{place_slug}_network.parquet"
        pbar.set_postfix_str("writing parquet")
        populated.to_parquet(output_path)
        pbar.update(1)

    print(f"Populated schema parquet exported to {output_path}")

    return populated

def _create_schema_dataframe():
    """Create an empty parquet file with columns from ProximitySchema.md"""
    columns = [
        # Street Centerlines
        "street_id", "street_grid_id", "public_data_id_street",
        "start_node_id", "start_node_is_intersection_node",
        "end_node_id", "end_node_is_intersection_node",
        "public_data_id_start_end_nodes", "normalized_bearing", "name", "highway",
        "maxspeed", "oneway", "lanes", "lane_width", "surface", "street_incline",
        # Street Centerline Features
        "street_feature_types", "public_data_id_street_feature",
        "street_feature_geometry", "street_feature_geometry_projected",
        "street_feature_attributes",
        # Sidewalk Left
        "sidewalk_left_ID", "sidewalk_left_grid_ID", "sidewalk_left_presence",
        "public_data_id_sidewalk_left", "sidewalk_left_surface", "sidewalk_left_quality",
        "sidewalk_left_width", "sidewalk_left_incline", "sidewalk_left_seperator", "sidewalk_left_buffered",
        # Curb Ramps Left
        "sidewalk_left_curbramp_start_1_ID", "public_data_id_sidewalk_left_curbramp_start_1",
        "sidewalk_left_curbramp_start_1_returnloc", "sidewalk_left_curbramp_start_1_returnposition",
        "sidewalk_left_curbramp_start_1_condition_score", "sidewalk_left_curbramp_start_1_geometry",
        "sidewalk_left_curbramp_start_2_ID", "public_data_id_sidewalk_left_curbramp_start_2",
        "sidewalk_left_curbramp_start_2_returnloc", "sidewalk_left_curbramp_start_2_returnposition",
        "sidewalk_left_curbramp_start_2_condition_score", "sidewalk_left_curbramp_start_2_geometry",
        "sidewalk_left_curbramp_start_3_ID", "public_data_id_sidewalk_left_curbramp_start_3",
        "sidewalk_left_curbramp_start_3_returnloc", "sidewalk_left_curbramp_start_3_returnposition",
        "sidewalk_left_curbramp_start_3_condition_score", "sidewalk_left_curbramp_start_3_geometry",
        "sidewalk_left_curbramp_end_1_ID", "public_data_id_sidewalk_left_curbramp_end_1",
        "sidewalk_left_curbramp_end_1_returnloc", "sidewalk_left_curbramp_end_1_returnposition",
        "sidewalk_left_curbramp_end_1_condition_score", "sidewalk_left_curbramp_end_1_geometry",
        "sidewalk_left_curbramp_end_2_ID", "public_data_id_sidewalk_left_curbramp_end_2",
        "sidewalk_left_curbramp_end_2_returnloc", "sidewalk_left_curbramp_end_2_returnposition",
        "sidewalk_left_curbramp_end_2_condition_score", "sidewalk_left_curbramp_end_2_geometry",
        "sidewalk_left_curbramp_end_3_ID", "public_data_id_sidewalk_left_curbramp_end_3",
        "sidewalk_left_curbramp_end_3_returnloc", "sidewalk_left_curbramp_end_3_returnposition",
        "sidewalk_left_curbramp_end_3_condition_score", "sidewalk_left_curbramp_end_3_geometry",
        # Sidewalk Left Features
        "sidewalk_left_feature_ids", "sidewalk_left_feature_types",
        "public_data_id_sidewalk_left_feature", "sidewalk_left_feature_geometry",
        "sidewalk_left_feature_geometry_projected",
        # Sidewalk Right
        "sidewalk_right_ID", "sidewalk_right_grid_ID", "sidewalk_right_presence",
        "public_data_id_sidewalk_right", "sidewalk_right_surface", "sidewalk_right_quality",
        "sidewalk_right_width", "sidewalk_right_incline", "sidewalk_right_seperator", "sidewalk_right_buffered",
        # Curb Ramps Right
        "sidewalk_right_curbramp_start_1_ID", "public_data_id_sidewalk_right_curbramp_start_1",
        "sidewalk_right_curbramp_start_1_returnloc", "sidewalk_right_curbramp_start_1_returnposition",
        "sidewalk_right_curbramp_start_1_condition_score", "sidewalk_right_curbramp_start_1_geometry",
        "sidewalk_right_curbramp_start_2_ID", "public_data_id_sidewalk_right_curbramp_start_2",
        "sidewalk_right_curbramp_start_2_returnloc", "sidewalk_right_curbramp_start_2_returnposition",
        "sidewalk_right_curbramp_start_2_condition_score", "sidewalk_right_curbramp_start_2_geometry",
        "sidewalk_right_curbramp_start_3_ID", "public_data_id_sidewalk_right_curbramp_start_3",
        "sidewalk_right_curbramp_start_3_returnloc", "sidewalk_right_curbramp_start_3_returnposition",
        "sidewalk_right_curbramp_start_3_condition_score", "sidewalk_right_curbramp_start_3_geometry",
        "sidewalk_right_curbramp_end_1_ID", "public_data_id_sidewalk_right_curbramp_end_1",
        "sidewalk_right_curbramp_end_1_returnloc", "sidewalk_right_curbramp_end_1_returnposition",
        "sidewalk_right_curbramp_end_1_condition_score", "sidewalk_right_curbramp_end_1_geometry",
        "sidewalk_right_curbramp_end_2_ID", "public_data_id_sidewalk_right_curbramp_end_2",
        "sidewalk_right_curbramp_end_2_returnloc", "sidewalk_right_curbramp_end_2_returnposition",
        "sidewalk_right_curbramp_end_2_condition_score", "sidewalk_right_curbramp_end_2_geometry",
        "sidewalk_right_curbramp_end_3_ID", "public_data_id_sidewalk_right_curbramp_end_3",
        "sidewalk_right_curbramp_end_3_returnloc", "sidewalk_right_curbramp_end_3_returnposition",
        "sidewalk_right_curbramp_end_3_condition_score", "sidewalk_right_curbramp_end_3_geometry",
        # Sidewalk Right Features
        "sidewalk_right_feature_ids", "sidewalk_right_feature_types",
        "public_data_id_sidewalk_right_feature", "sidewalk_right_feature_geometry",
        "sidewalk_right_feature_geometry_projected",
        # Crosswalks
        "crosswalk_start_id", "crosswalk_start_grid_ids", "crosswalk_start_type",
        "public_data_id_crosswalk_start", "crosswalk_start_controlled", "crosswalk_start_marked",
        "crosswalk_start_markings", "crosswalk_start_signals", "crosswalk_start_island",
        "crosswalk_start_kerb", "crosswalk_start_tactile_paving", "crosswalk_start_traffic_calming",
        "crosswalk_start_continuous", "crosswalk_start_condition", "crosswalk_start_geometry",
        "crosswalk_start_island_geometry",
        "crosswalk_end_id", "crosswalk_end_grid_ids", "crosswalk_end_type",
        "public_data_id_crosswalk_end", "crosswalk_end_controlled", "crosswalk_end_marked",
        "crosswalk_end_markings", "crosswalk_end_signals", "crosswalk_end_island",
        "crosswalk_end_kerb", "crosswalk_end_tactile_paving", "crosswalk_end_traffic_calming",
        "crosswalk_end_continuous", "crosswalk_end_condition", "crosswalk_end_geometry",
        "crosswalk_end_island_geometry",
        # Bikeways Left
        "bikeway_left_1_id", "bikeway_left_1_grid_id", "public_data_id_bikeway_left_1",
        "bikeway_left_1_type", "bikeway_left_1_surface", "bikeway_left_1_quality",
        "bikeway_left_1_permitted", "bikeway_left_1_width", "bikeway_left_1_incline",
        "bikeway_left_1_seperator", "bikeway_left_1_buffered",
        "bikeway_left_2_id", "public_data_id_bikeway_left_2",
        "bikeway_left_2_type", "bikeway_left_2_surface", "bikeway_left_2_quality",
        "bikeway_left_2_permitted", "bikeway_left_2_width", "bikeway_left_2_incline",
        "bikeway_left_2_seperator", "bikeway_left_2_buffered",
        # Bikeway Features Left
        "bikeway_left_1_feature_ids", "bikeway_left_1_feature_types",
        "public_data_id_bikeway_left_1_features", "bikeway_left_1_feature_geometry",
        "bikeway_left_1_feature_geometry_projected", "bikeway_left_2_feature_types",
        "public_data_id_bikeway_left_2_features", "bikeway_left_2_feature_geometry",
        "bikeway_left_2_feature_geometry_projected",
        # Bikeways Right
        "bikeway_right_1_id", "bikeway_right_1_grid_id", "public_data_id_bikeway_right_1",
        "bikeway_right_1_type", "bikeway_right_1_surface", "bikeway_right_1_quality",
        "bikeway_right_1_permitted", "bikeway_right_1_width", "bikeway_right_1_incline",
        "bikeway_right_1_seperator", "bikeway_right_1_buffered",
        "bikeway_right_2_id", "public_data_id_bikeway_right_2",
        "bikeway_right_2_type", "bikeway_right_2_surface", "bikeway_right_2_quality",
        "bikeway_right_2_permitted", "bikeway_right_2_width", "bikeway_right_2_incline",
        "bikeway_right_2_seperator", "bikeway_right_2_buffered",
        # Bikeway Features Right
        "bikeway_right_1_feature_ids", "bikeway_right_1_feature_types",
        "public_data_id_bikeway_right_1_features", "bikeway_right_1_feature_geometry",
        "bikeway_right_1_feature_geometry_projected", "bikeway_right_2_feature_types",
        "public_data_id_bikeway_right_2_features", "bikeway_right_2_feature_geometry",
        "bikeway_right_2_feature_geometry_projected",
        # Main Geometries
        "street_geometry", "start_node_geometry", "end_node_geometry",
        "sidewalk_left_geometry", "sidewalk_right_geometry", "curb_return_geometry",
        "bikeway_left_1_geometry", "bikeway_left_2_geometry",
        "bikeway_right_1_geometry", "bikeway_right_2_geometry",
    ]


    # Create empty GeoDataFrame with specified columns
    gdf = gpd.GeoDataFrame(columns=columns)

    
    return gdf

def populate_base_bikelanes(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Populate centerline-derived bikeway columns from OSM cycleway tags, then
    spatially match any independently mapped cycleway edges."""

    def _get(col):
        return edges_reset[col] if col in edges_reset.columns else None

    def _coalesce(*cols):
        result = pd.Series(pd.NA, index=edges_reset.index, dtype=object)
        for col in cols:
            if col in edges_reset.columns:
                result = result.where(result.notna(), edges_reset[col])
        return result

    with tqdm(total=5, desc="Loading bikeway data", unit="step") as pbar:
        # Type: prefer side-specific tag, fall back to cycleway:both, then bare cycleway
        pbar.set_postfix_str("type")
        populated["bikeway_left_1_type"]  = _coalesce("cycleway:left",  "cycleway:both", "cycleway")
        populated["bikeway_right_1_type"] = _coalesce("cycleway:right", "cycleway:both", "cycleway")
        populated["bikeway_left_2_type"]  = _coalesce("cycleway:left:2",  "cycleway:both:2")
        populated["bikeway_right_2_type"] = _coalesce("cycleway:right:2", "cycleway:both:2")
        pbar.update(1)

        # Sub-type, surface, width, separator
        pbar.set_postfix_str("quality / surface / width / separator")
        populated["bikeway_left_1_quality"]  = _coalesce("cycleway:left:smoothness",  "cycleway:smoothness")
        populated["bikeway_right_1_quality"] = _coalesce("cycleway:right:smoothness", "cycleway:smoothness")
        populated["bikeway_left_1_surface"]  = _coalesce("cycleway:left:surface",  "cycleway:surface")
        populated["bikeway_right_1_surface"] = _coalesce("cycleway:right:surface", "cycleway:surface")
        populated["bikeway_left_1_width"]    = _coalesce("cycleway:left:width",  "cycleway:width").apply(_parse_float_tag)
        populated["bikeway_right_1_width"]   = _coalesce("cycleway:right:width", "cycleway:width").apply(_parse_float_tag)
        populated["bikeway_left_1_seperator"]   = _coalesce("cycleway:left:buffer",  "cycleway:buffer")
        populated["bikeway_right_1_seperator"] = _coalesce("cycleway:right:buffer", "cycleway:buffer")
        pbar.update(1)

        # Permitted / incline
        pbar.set_postfix_str("permitted / incline")
        populated["bikeway_left_1_permitted"]  = _get("bicycle")
        populated["bikeway_right_1_permitted"] = _get("bicycle")
        _raw_incline = _get("incline")
        _parsed_incline = _raw_incline.apply(_parse_incline) if _raw_incline is not None else None
        if _parsed_incline is not None:
            populated["bikeway_left_1_incline"]  = _parsed_incline
            populated["bikeway_right_1_incline"] = _parsed_incline
        else:
            populated["bikeway_left_1_incline"]  = pd.NA
            populated["bikeway_right_1_incline"] = pd.NA
        pbar.update(1)

        # Secondary bikeway slots (_2)
        pbar.set_postfix_str("secondary bikeway slots")
        populated["bikeway_left_2_quality"]   = _get("cycleway:left:2:smoothness")
        populated["bikeway_right_2_quality"]  = _get("cycleway:right:2:smoothness")
        populated["bikeway_left_2_surface"]   = _get("cycleway:left:2:surface")
        populated["bikeway_right_2_surface"]  = _get("cycleway:right:2:surface")
        _raw_bl2w = _get("cycleway:left:2:width")
        populated["bikeway_left_2_width"]     = _raw_bl2w.apply(_parse_float_tag) if _raw_bl2w is not None else pd.NA
        _raw_br2w = _get("cycleway:right:2:width")
        populated["bikeway_right_2_width"]    = _raw_br2w.apply(_parse_float_tag) if _raw_br2w is not None else pd.NA
        populated["bikeway_left_2_permitted"] = _get("bicycle")
        populated["bikeway_right_2_permitted"]= _get("bicycle")
        if _parsed_incline is not None:
            populated["bikeway_left_2_incline"]  = _parsed_incline
            populated["bikeway_right_2_incline"] = _parsed_incline
        else:
            populated["bikeway_left_2_incline"]  = pd.NA
            populated["bikeway_right_2_incline"] = pd.NA
        populated["bikeway_left_2_seperator"]  = _get("cycleway:left:2:buffer")
        populated["bikeway_right_2_seperator"] = _get("cycleway:right:2:buffer")
        pbar.update(1)

        pbar.update(1)

    return populated



def populate_base_footlanes(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Populate centerline-derived sidewalk columns from OSM sidewalk tags.

    Does not set buffered or geometry — the separate facilities pass writes
    geometry where OSM footway edges exist, and the buffering pass generates
    perpendicular offsets for any side that has presence data but no geometry.
    """

    def _get(col):
        return edges_reset[col] if col in edges_reset.columns else None

    def _coalesce(*cols):
        result = pd.Series(pd.NA, index=edges_reset.index, dtype=object)
        for col in cols:
            if col in edges_reset.columns:
                result = result.where(result.notna(), edges_reset[col])
        return result

    with tqdm(total=3, desc="Loading sidewalk data", unit="step") as pbar:
        # Presence: prefer side-specific, fall back to sidewalk:both, then bare sidewalk
        pbar.set_postfix_str("presence")
        populated["sidewalk_left_presence"]  = _coalesce("sidewalk:left",  "sidewalk:both", "sidewalk")
        populated["sidewalk_right_presence"] = _coalesce("sidewalk:right", "sidewalk:both", "sidewalk")
        pbar.update(1)

        # Surface, width, incline, quality, separator
        pbar.set_postfix_str("surface / width / incline / quality / separator")
        populated["sidewalk_left_surface"]    = _get("sidewalk:left:surface")
        populated["sidewalk_right_surface"]   = _get("sidewalk:right:surface")
        _raw_slw = _get("sidewalk:left:width")
        populated["sidewalk_left_width"]      = _raw_slw.apply(_parse_float_tag) if _raw_slw is not None else pd.NA
        _raw_srw = _get("sidewalk:right:width")
        populated["sidewalk_right_width"]     = _raw_srw.apply(_parse_float_tag) if _raw_srw is not None else pd.NA
        _raw_sw_l = _get("sidewalk:left:incline")
        populated["sidewalk_left_incline"]  = _raw_sw_l.apply(_parse_incline) if _raw_sw_l is not None else pd.NA
        _raw_sw_r = _get("sidewalk:right:incline")
        populated["sidewalk_right_incline"] = _raw_sw_r.apply(_parse_incline) if _raw_sw_r is not None else pd.NA
        populated["sidewalk_left_quality"]    = _get("sidewalk:left:smoothness")
        populated["sidewalk_right_quality"]   = _get("sidewalk:right:smoothness")
        populated["sidewalk_left_seperator"]  = _get("sidewalk:left:buffer")
        populated["sidewalk_right_seperator"] = _get("sidewalk:right:buffer")
        pbar.update(1)

        # Centerline-tagged sidewalks: mark buffered=True and clear geometry
        pbar.update(1)

    return populated


def _road_side(road_geom: BaseGeometry, point: Point) -> str:
    """Return 'left' or 'right' based on cross product of road direction × road→point."""
    coords = list(road_geom.coords)
    ax, ay = coords[0]
    bx, by = coords[-1]
    px, py = point.x, point.y
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    return "left" if cross > 0 else "right"


def _sindex_nearest_idx(sindex, geom, df) -> int:
    """Return the index label of the nearest row in *df* to *geom*.

    Handles all geopandas sindex.nearest() return formats:
    - tuple (input_indices, tree_indices): geopandas >= 0.12 with PyGEOS
    - 2D ndarray shape (2, n) [[input_idx...], [tree_idx...]]: geopandas PyGEOS backend
    - 1D ndarray or scalar: older geopandas / rtree backend
    """
    result = sindex.nearest(geom)
    if isinstance(result, tuple):
        best_pos = int(result[1].flat[0])
    else:
        arr = np.asarray(result)
        if arr.ndim == 2:
            # Row 0 = input indices (always 0 for single-geometry query),
            # Row 1 = tree indices (the actual nearest positions).
            best_pos = int(arr[1][0])
        else:
            best_pos = int(arr.flat[0])
    return df.index[best_pos]


def _populate_separate_facilities(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
    default_lane_width_m: float = 3.5,
) -> gpd.GeoDataFrame:
    """Match independently mapped cycleway and footway edges to their parent road
    segments and write attributes + geometry into the appropriate schema slots.

    Processes bikelanes first, then sidewalks. For ambiguous edges (e.g. a
    ``path`` with no bicycle/foot qualifier), bikelane classification takes
    priority. Separate geometry always wins over buffered offsets — if a
    matched edge is written, buffered is set to False.
    - collision check during proximity matching is row-scoped per spec step 4
    """
    hw      = edges_reset.get("highway", pd.Series(dtype=object))
    bicycle = edges_reset.get("bicycle", pd.Series(dtype=object))
    foot    = edges_reset.get("foot",    pd.Series(dtype=object))

    # ── Classification ──────────────────────────────────────────────────────
    BIKEWAY_HW = {"cycleway", "path", "bridleway"}
    FOOTWAY_HW = {"footway", "pedestrian", "path", "steps", "corridor"}

    is_bikeway = (
        hw.isin(BIKEWAY_HW) |
        (hw.isin({"path", "footway"}) & bicycle.isin({"designated", "yes"}))
    )
    is_footway = (
        hw.isin(FOOTWAY_HW) |
        (hw.isin({"path"}) & foot.isin({"designated", "yes"}))
    ) & ~bicycle.isin({"designated"}).astype(bool)
    is_footway = is_footway & ~is_bikeway   # bikeway takes priority for ambiguous edges

    is_separate = is_bikeway | is_footway
    roads = populated[~is_separate.to_numpy(dtype=bool)].copy()
    road_sindex = roads.geometry.sindex

    # ── Occupancy tracking (independent of geometry/type columns) ────────────
    # Both bikeways and sidewalks start empty — separate OSM edges take
    # priority over any centerline-derived data already written, so slots
    # are never pre-occupied.
    bike_slots_used: set = set()   # {(road_idx, side, slot_str)}
    foot_slots_used: set = set()   # {(road_idx, side_str)}

    # ── Discard counters ────────────────────────────────────────────────────
    n_foot_suspect:  dict[str, int] = {"left": 0, "right": 0}
    n_foot_replaced: dict[str, int] = {"left": 0, "right": 0}   # inferior dupe discarded

    # ── Slot helpers ────────────────────────────────────────────────────────
    # Merge threshold: if a new cycleway edge is within this distance of an
    # existing slot's geometry, it belongs to the same facility (merge).
    # Beyond this, it's a distinct parallel facility (new slot).
    _BIKE_MERGE_THRESHOLD_M = 5.0
    # Adjacent sidewalk segments within this distance are merged rather than
    # deduplicated, so corner connectors and block-spanning footways combine.
    _FOOT_MERGE_THRESHOLD_M = 3.0

    def _bikeway_slot_or_merge(road_idx, side, cy_geom) -> tuple[str | None, bool]:
        """Determine whether a cycleway edge merges into an existing slot or gets a new one.

        Returns (slot_str, is_merge):
        - ("1", False) / ("2", False)  — new slot assignment
        - ("1", True)  / ("2", True)   — merge into existing slot
        - (None, False)                — no room (both slots occupied, neither mergeable)
        """
        for slot in ("1", "2"):
            if (road_idx, side, slot) not in bike_slots_used:
                return slot, False
            # Slot occupied — check if this edge belongs to the same facility
            existing = populated.at[road_idx, f"bikeway_{side}_{slot}_geometry"]  # type: ignore[index]
            if existing is not None and isinstance(existing, BaseGeometry):
                if existing.distance(cy_geom) <= _BIKE_MERGE_THRESHOLD_M:
                    return slot, True
        return None, False

    def _sidewalk_slot(road_idx, side) -> str | None:
        """Return '' when the slot is free, None when it is occupied.

        Uses the in-memory ``foot_slots_used`` set to track occupied slots.
        """
        return None if (road_idx, side) in foot_slots_used else ""

    # Maximum search radius (metres) for name-aware facility matching.
    _NAME_MATCH_RADIUS_M = 30.0

    def _normalize_name(val) -> str | None:
        """Return a lowered, stripped name string, or None if missing."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip().lower()
        return s if s else None

    def _match_road_by_name(sindex, fac_geom, fac_mid, df, fac_name: str | None):
        """Find the best road segment for a facility edge using name + proximity.

        Strategy:
        1. Query all roads within ``_NAME_MATCH_RADIUS_M`` of the facility.
        2. Among roads whose name matches ``fac_name``, pick the closest.
        3. If no name match (or facility has no name), fall back to the
           spatially nearest road (``_sindex_nearest_idx``).

        Returns (road_idx, road_geom, side).
        """
        best_idx = None
        best_geom = None
        best_side = None
        best_dist = float("inf")

        norm_fac = _normalize_name(fac_name)

        if norm_fac is not None:
            search_area = fac_geom.buffer(_NAME_MATCH_RADIUS_M)
            hit_positions = sindex.query(search_area)
            for pos in hit_positions:
                ridx = df.index[pos]
                rname = _normalize_name(populated.at[ridx, "name"] if "name" in populated.columns else None)  # type: ignore[index]
                if rname != norm_fac:
                    continue
                rgeom = populated.at[ridx, "street_geometry"]  # type: ignore[index]
                if not isinstance(rgeom, BaseGeometry):
                    continue
                d = rgeom.distance(fac_geom)
                if d < best_dist:
                    best_dist = d
                    best_idx = ridx
                    best_geom = rgeom
                    best_side = _road_side(rgeom, fac_mid)

        if best_idx is not None:
            return best_idx, best_geom, best_side

        # Fallback: spatially nearest road (no name filter)
        nearest_idx = _sindex_nearest_idx(sindex, fac_geom, df)
        nearest_geom = populated.at[nearest_idx, "street_geometry"]  # type: ignore[index]
        if not isinstance(nearest_geom, BaseGeometry):
            return None, None, None
        return nearest_idx, nearest_geom, _road_side(nearest_geom, fac_mid)

    # ── Bikelane pass ───────────────────────────────────────────────────────
    cycleways = edges_reset[is_bikeway].copy()
    print(f"Separate facility classification: {len(cycleways)} bikeway edges, "
          f"{is_footway.sum()} footway edges from {len(edges_reset)} total edges.")
    if len(cycleways):
        hw_counts = cycleways["highway"].value_counts() if "highway" in cycleways.columns else {}
        print(f"  Bikeway highway types: {dict(hw_counts)}")
    n_bike_matched = 0
    n_bike_merged = 0
    n_bike_slot_full = 0

    # Pre-compute geometry properties for bikeways (positional lists avoid duplicate-index ambiguity)
    cy_mids_list  = [g.interpolate(0.5, normalized=True) if g is not None else None for g in cycleways.geometry]
    cy_names_list = list(cycleways["name"]) if "name" in cycleways.columns else [None] * len(cycleways)

    for cy_pos in tqdm(range(len(cycleways)), total=len(cycleways), desc="Matching bikeways", unit="edge"):
        cy_row     = cycleways.iloc[cy_pos]
        cy_geom    = cast(BaseGeometry, cy_row["geometry"])
        cy_mid     = cy_mids_list[cy_pos]
        cy_name_raw = cy_names_list[cy_pos]
        # Coerce list/array to first str element (OSM names can be multi-valued)
        if isinstance(cy_name_raw, (list, np.ndarray)):
            cy_name_raw = next((x for x in cy_name_raw if isinstance(x, str)), None)
        # Convert scalar to str | None
        if isinstance(cy_name_raw, bytes):
            cy_name: str | None = cy_name_raw.decode('utf-8')
        elif cy_name_raw is None or (isinstance(cy_name_raw, float) and pd.isna(cy_name_raw)):
            cy_name = None
        else:
            cy_name = cy_name_raw if isinstance(cy_name_raw, str) else None

        road_idx, road_geom, side = _match_road_by_name(
            road_sindex, cy_geom, cy_mid, roads, cy_name)
        if road_idx is None:
            continue
        slot, is_merge = _bikeway_slot_or_merge(road_idx, side, cy_geom)
        if slot is None:
            n_bike_slot_full += 1
            continue

        prefix = f"bikeway_{side}_{slot}"
        if is_merge:
            # Same facility — merge geometry into existing slot
            existing = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
            if existing is not None and isinstance(existing, BaseGeometry):
                if isinstance(existing, MultiLineString):
                    parts = list(existing.geoms) + [cy_geom]
                else:
                    parts = [existing, cy_geom]
                populated.at[road_idx, f"{prefix}_geometry"] = MultiLineString(parts)  # type: ignore[index]
            else:
                populated.at[road_idx, f"{prefix}_geometry"] = cy_geom  # type: ignore[index]
            n_bike_merged += 1
        else:
            # New facility — write all attributes
            populated.at[road_idx, f"{prefix}_type"]      = cy_row.get("highway",  pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_surface"]   = cy_row.get("surface",  pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_width"]     = _parse_float_tag(cy_row.get("width",    pd.NA))  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_permitted"] = cy_row.get("bicycle",  pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_incline"]   = _parse_incline(cy_row.get("incline",  pd.NA))  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_geometry"]  = cy_geom  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_buffered"] = False  # type: ignore[index]
            bike_slots_used.add((road_idx, side, slot))
            _is_centerline(cy_geom, cast(BaseGeometry, road_geom), road_idx, prefix, populated)
            n_bike_matched += 1

    print(f"Matched {n_bike_matched} separate cycleway edges to road segments "
          f"({n_bike_merged} merged, {n_bike_slot_full} skipped — both slots full).")

    # ── Sidewalk pass ───────────────────────────────────────────────────────
    footways = edges_reset[is_footway].copy()
    if len(footways):
        hw_counts = footways["highway"].value_counts() if "highway" in footways.columns else {}
        print(f"  Footway highway types: {dict(hw_counts)}")
    n_foot_matched = 0
    n_foot_merged = 0
    n_foot_name_matched = 0

    # Pre-compute geometry properties for footways (positional lists avoid duplicate-index ambiguity)
    fw_mids_list  = [g.interpolate(0.5, normalized=True) if g is not None else None for g in footways.geometry]
    fw_names_list = list(footways["name"]) if "name" in footways.columns else [None] * len(footways)

    for fw_pos in tqdm(range(len(footways)), total=len(footways), desc="Matching footways", unit="edge"):
        fw_row  = footways.iloc[fw_pos]
        fw_geom = cast(BaseGeometry, fw_row["geometry"])
        fw_mid  = fw_mids_list[fw_pos]
        fw_name_raw = fw_names_list[fw_pos]
        # Coerce list/array to first str element (OSM names can be multi-valued)
        if isinstance(fw_name_raw, (list, np.ndarray)):
            fw_name_raw = next((x for x in fw_name_raw if isinstance(x, str)), None)
        # Convert scalar to str | None
        if isinstance(fw_name_raw, bytes):
            fw_name: str | None = fw_name_raw.decode('utf-8')
        elif fw_name_raw is None or (isinstance(fw_name_raw, float) and pd.isna(fw_name_raw)):
            fw_name = None
        else:
            fw_name = fw_name_raw if isinstance(fw_name_raw, str) else None

        road_idx, road_geom, side = _match_road_by_name(
            road_sindex, fw_geom, fw_mid, roads, fw_name)
        if road_idx is None:
            continue
        side = cast(str, side)  # type guard: side is non-None when road_idx is non-None
        # Track whether this was a name-based match
        if _normalize_name(fw_name) is not None and _normalize_name(fw_name) == _normalize_name(
                populated.at[road_idx, "name"] if "name" in populated.columns else None):  # type: ignore[index]
            n_foot_name_matched += 1

        prefix = f"sidewalk_{side}"
        slot_key = (road_idx, side)

        if slot_key in foot_slots_used:
            # Deduplication: merge adjacent segments; otherwise keep the longer geometry.
            existing = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
            if existing is not None and isinstance(existing, BaseGeometry):
                if existing.distance(fw_geom) <= _FOOT_MERGE_THRESHOLD_M:
                    # Segments are adjacent — merge so corner connectors and
                    # block-spanning footways form a continuous path together.
                    parts = list(existing.geoms) if isinstance(existing, MultiLineString) else [existing]
                    parts.append(cast(LineString, fw_geom))  # type: ignore[arg-type]
                    populated.at[road_idx, f"{prefix}_geometry"] = MultiLineString(parts)  # type: ignore[index]
                elif fw_geom.length > existing.length:
                    # Not adjacent and new edge is longer — replace and re-validate.
                    populated.at[road_idx, f"{prefix}_geometry"] = fw_geom  # type: ignore[index]
                    winner = fw_geom
                    n_foot_replaced[side] += 1
                    _is_centerline(winner, cast(BaseGeometry, road_geom), road_idx, prefix, populated)
                    winner_after = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
                    if winner_after is not None and isinstance(winner_after, BaseGeometry):
                        if _is_suspect_geometry(winner_after):
                            populated.at[road_idx, f"{prefix}_geometry"] = None  # type: ignore[index]
                            populated.at[road_idx, f"{prefix}_buffered"] = True  # type: ignore[index]
                            n_foot_suspect[side] += 1
                else:
                    # Existing is longer and not adjacent — discard new edge.
                    n_foot_replaced[side] += 1
            else:
                populated.at[road_idx, f"{prefix}_geometry"] = fw_geom  # type: ignore[index]
            n_foot_merged += 1
        else:
            populated.at[road_idx, f"{prefix}_presence"] = "separate"  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_surface"]  = fw_row.get("surface",    pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_width"]    = _parse_float_tag(fw_row.get("width",      pd.NA))  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_incline"]  = _parse_incline(fw_row.get("incline",    pd.NA))  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_quality"]  = fw_row.get("smoothness", pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_geometry"] = fw_geom  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_buffered"] = False  # type: ignore[index]
            foot_slots_used.add(slot_key)
            _is_centerline(fw_geom, cast(BaseGeometry, road_geom), road_idx, prefix, populated)
            # Check if geometry is suspect after centerline check
            geom_after = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
            if geom_after is not None and isinstance(geom_after, BaseGeometry):
                if _is_suspect_geometry(geom_after):
                    populated.at[road_idx, f"{prefix}_geometry"] = None  # type: ignore[index]
                    populated.at[road_idx, f"{prefix}_buffered"] = True  # type: ignore[index]
                    n_foot_suspect[side] += 1
            n_foot_matched += 1

    print(f"Matched {n_foot_matched} separate footway edges to road segments "
          f"({n_foot_name_matched} by name, {n_foot_merged} merged into existing slots).")
    print(
        f"[sidewalk] Suspect geometries discarded → flagged for buffering: "
        f"left={n_foot_suspect['left']}, right={n_foot_suspect['right']}"
    )
    print(
        f"[sidewalk] Inferior duplicate footways discarded (fewer vertices): "
        f"left={n_foot_replaced['left']}, right={n_foot_replaced['right']}"
    )

    # ── Buffering pass (spec steps 4 & 5) ────────────────────────────────
    # For road segments with presence/type data but no geometry, generate a
    # perpendicular-offset geometry from the street centerline (buffered=True).
    #
    # Sidewalk suppression: before buffering a sidewalk slot, check whether
    # any separate sidewalk geometry (left OR right, from any road) with a
    # parallel bearing already exists within NEARBY_SEPARATE_SIDEWALK_THRESHOLD_M
    # of the street centerline.  This prevents duplicate buffered sidewalks on
    # inner service roads that run parallel to a primary road whose real
    # outer sidewalk is already mapped.  The unified (left+right) tree is used
    # so that cross-slot duplicates (service road left ↔ primary road right) are
    # detected correctly.
    _NEGATIVE_VALUES = {"no", "none"}
    # Presence values indicating the sidewalk is mapped as a separate OSM way.
    # These should never trigger buffered-offset geometry.
    # Separate footway edges always write "separate" to the presence column;
    # OSM centerline tags (sidewalk:left/right=separate) also produce "separate".
    _SEPARATE_PRESENCE_VALUES = {"separate"}

    # Build unified tree of ALL separate sidewalk geometries (both sides).
    all_sep_sw_geoms: list[BaseGeometry] = []
    all_sep_sw_bearings: list[float] = []
    all_sep_sw_row_indices: list = []  # row index to prevent self-suppression
    for sw_side in ("left", "right"):
        gcol = f"sidewalk_{sw_side}_geometry"
        bcol = f"sidewalk_{sw_side}_buffered"
        if gcol not in populated.columns:
            continue
        geom_series = populated[gcol]

        # Vectorized pre-filter: rows that have real geometry and are not buffered
        has_geom = geom_series.apply(lambda g: g is not None and hasattr(g, "geom_type"))
        if bcol in populated.columns:
            bvals = populated[bcol]
            is_buffered = bvals.eq(True) | bvals.astype(str).str.lower().eq("yes")
        else:
            is_buffered = pd.Series(False, index=populated.index)
        valid_idx = populated.index[has_geom & ~is_buffered]

        for idx in valid_idx:
            g = cast(BaseGeometry, geom_series.at[idx])
            bearing = _linestring_bearing(g)
            if bearing is None:
                continue
            all_sep_sw_geoms.append(g)
            all_sep_sw_bearings.append(bearing)
            all_sep_sw_row_indices.append(idx)

    sep_sw_tree = STRtree(all_sep_sw_geoms) if all_sep_sw_geoms else None
    total_buffered = 0
    total_skipped = 0

    # Debug counters for _buffer_segment failure modes
    _buffer_debug = {"n_empty_offset": 0, "collision_counts": {}}

    for kind, side, slot in _FACILITY_SLOTS:
        sub_id   = f"{kind}_{side}_{slot}" if slot else f"{kind}_{side}"
        geom_col = f"{sub_id}_geometry"
        buff_col = f"{sub_id}_buffered"
        data_col = f"{sub_id}_type" if kind == "bikeway" else f"{sub_id}_presence"

        if geom_col not in populated.columns or data_col not in populated.columns:
            continue

        skip_values = _NEGATIVE_VALUES | _SEPARATE_PRESENCE_VALUES if kind == "sidewalk" else _NEGATIVE_VALUES
        has_data = populated[data_col].notna() & ~populated[data_col].astype(str).str.lower().isin(skip_values)
        no_geom  = populated[geom_col].apply(lambda g: g is None or not hasattr(g, "geom_type"))
        candidates = has_data & no_geom
        print(f"  [{sub_id}] candidates: {candidates.sum()}, has_data: {has_data.sum()}, no_geom: {no_geom.sum()}")
        if not candidates.any():
            continue

        n_no_street_geom = 0
        n_suppressed_here = 0
        n_buffer_called = 0
        n_buffer_wrote = 0
        for idx in populated.index[candidates]:
            street_geom = populated.at[idx, "street_geometry"]
            if street_geom is None or not hasattr(street_geom, "geom_type"):
                n_no_street_geom += 1
                continue
            street_geom = cast(BaseGeometry, street_geom)

            # Sidewalk suppression: skip if a nearby parallel separate sidewalk
            # exists on the same side.  Only guard is same-row (a row's own
            # separate geometry can't cause false suppression since it already
            # has geometry and therefore isn't a buffering candidate).  This
            # allows separate sidewalks from adjacent segments of the SAME
            # street to suppress buffering, preventing mixed separate/buffered
            # output on streets with inconsistent OSM footway coverage.
            if kind == "sidewalk" and sep_sw_tree is not None:
                street_bearing = _linestring_bearing(street_geom)
                if street_bearing is not None:
                    search_area = street_geom.buffer(NEARBY_SEPARATE_SIDEWALK_THRESHOLD_M)
                    hit_indices = sep_sw_tree.query(search_area)
                    skip = False
                    for hi in hit_indices:
                        # Same row → can't be a real suppression source
                        if all_sep_sw_row_indices[hi] == idx:
                            continue
                        if not _bearings_parallel(street_bearing, all_sep_sw_bearings[hi]):
                            continue
                        fac_mid = all_sep_sw_geoms[hi].interpolate(0.5, normalized=True)
                        if (street_geom.distance(fac_mid) <= NEARBY_SEPARATE_SIDEWALK_THRESHOLD_M
                                and _road_side(street_geom, fac_mid) == side):
                            skip = True
                            break
                    if skip:
                        total_skipped += 1
                        n_suppressed_here += 1
                        continue

            n_buffer_called += 1
            populated = _buffer_segment(idx, sub_id, street_geom, populated, side, _buffer_debug, default_lane_width_m)
            geom_after = populated.at[idx, geom_col]
            if geom_after is not None and hasattr(geom_after, "geom_type"):
                n_buffer_wrote += 1
                total_buffered += 1

        print(f"    no_street_geom={n_no_street_geom}, suppressed={n_suppressed_here}, "
              f"buffer_called={n_buffer_called}, buffer_wrote={n_buffer_wrote}")

    print(f"Buffering pass: {total_buffered} facility segments offset from centerline"
          f" ({total_skipped} sidewalk slots skipped — nearby separate sidewalk exists).")
    print(f"  _buffer_segment failures: empty_offset={_buffer_debug['n_empty_offset']}, "
          f"collisions={_buffer_debug['collision_counts']}")
    if "empty_offset_detail" in _buffer_debug:
        print(f"  empty_offset breakdown: {_buffer_debug['empty_offset_detail']}")
    if "nan_source_counts" in _buffer_debug:
        print(f"  NaN source breakdown: {_buffer_debug['nan_source_counts']}")
    return populated


def run_multi_city():
    results = {}
    for cfg in CITIES_CONFIG:
        place = cfg["place"]
        print(f"\n=== Processing: {place} ===")
        results[place] = populate_schema(
            place,
            default_lane_width_m=cfg.get("default_lane_width_m", _DEFAULT_LANE_WIDTH_M),
            default_maxspeed=cfg.get("default_maxspeed", _DEFAULT_MAXSPEED),
        )
    return results

_FACILITY_SLOTS = [
    ("bikeway",  "left",  "1"),
    ("bikeway",  "left",  "2"),
    ("bikeway",  "right", "1"),
    ("bikeway",  "right", "2"),
    ("sidewalk", "left",  None),
    ("sidewalk", "right", None),
]


# Maximum angular difference (degrees) to consider two bearings parallel.
_PARALLEL_BEARING_TOLERANCE_DEG = 30.0


def _linestring_bearing(geom: BaseGeometry) -> float | None:
    """Return the bearing (0-360 deg) of a LineString from its first to last coordinate."""
    coords = []
    if isinstance(geom, MultiLineString):
        # For MultiLineString, use first coord of first part and last coord of last part
        if len(geom.geoms) > 0:
            first_line = geom.geoms[0]
            last_line = geom.geoms[-1]
            coords = [list(first_line.coords)[0], list(last_line.coords)[-1]]
    elif hasattr(geom, "coords"):
        try:
            coords = list(geom.coords)
        except NotImplementedError:
            return None

    if len(coords) < 2:
        return None
    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return None
    return np.degrees(np.arctan2(dx, dy)) % 360


def _bearings_parallel(a: float, b: float, tolerance: float = _PARALLEL_BEARING_TOLERANCE_DEG) -> bool:
    """Return True if bearings *a* and *b* are within *tolerance* degrees.

    Accounts for 180 deg equivalence (a road bearing 10 and 190 are the same axis).
    """
    diff = abs(a - b) % 360
    if diff > 180:
        diff = 360 - diff
    if diff > 90:
        diff = 180 - diff
    return diff <= tolerance


def _infer_bearing_from_named_neighbors(
    idx: int,
    raw_bearing: float,
    name_col: pd.Series,  # type: ignore[type-arg]
    node_to_edge_indices: dict[Any, list[int]],
    edge_bearings: dict[int, float],
    edges_reset: gpd.GeoDataFrame,
    parallel_threshold_deg: float = 45.0,
) -> float | None:
    """Infer canonical direction for an unnamed two-way segment from adjacent named segments.

    For each named edge that shares an endpoint node and is roughly parallel to this
    segment, cast a vote for the raw bearing or its 180° flip.  Returns the winning
    bearing, or None if no parallel named neighbors exist (caller should fall back to
    northward normalization).
    """
    u = edges_reset.at[idx, "u"]
    v = edges_reset.at[idx, "v"]

    flipped = (raw_bearing + 180.0) % 360.0
    raw_axis = raw_bearing % 180.0

    votes_raw = 0
    votes_flip = 0

    for node in (u, v):
        for neighbor_idx in node_to_edge_indices.get(node, []):
            if neighbor_idx == idx:
                continue
            name = name_col.at[neighbor_idx]
            is_named = (
                name is not None
                and not (isinstance(name, float) and math.isnan(name))
                and str(name).strip()
            )
            if not is_named or neighbor_idx not in edge_bearings:
                continue

            nb = edge_bearings[neighbor_idx]
            nb_axis = nb % 180.0

            # Only consider roughly parallel neighbors
            axis_diff = abs(raw_axis - nb_axis)
            if axis_diff > 90.0:
                axis_diff = 180.0 - axis_diff
            if axis_diff > parallel_threshold_deg:
                continue

            diff_raw = abs(raw_bearing - nb) % 360.0
            if diff_raw > 180.0:
                diff_raw = 360.0 - diff_raw
            diff_flip = abs(flipped - nb) % 360.0
            if diff_flip > 180.0:
                diff_flip = 360.0 - diff_flip

            if diff_raw <= diff_flip:
                votes_raw += 1
            else:
                votes_flip += 1

    if votes_raw == 0 and votes_flip == 0:
        return None
    return raw_bearing if votes_raw >= votes_flip else flipped


# ── Geometry quality thresholds ────────────────────────────────────────────────
SINUOSITY_THRESHOLD = 3.0   # length / crow-fly; above this = suspect
MIN_CROW_FLY_M      = 5.0   # ignore tiny stubs when judging sinuosity


def _vertex_count(geom: BaseGeometry) -> int:
    """Count vertices in a LineString or MultiLineString."""
    if isinstance(geom, LineString):
        return len(list(geom.coords))
    if isinstance(geom, MultiLineString):
        return sum(len(list(ls.coords)) for ls in geom.geoms)
    return 0


def _is_suspect_geometry(geom: BaseGeometry) -> bool:
    """Return True if geometry is self-intersecting or has sinuosity > threshold."""
    if not geom.is_simple:
        return True
    if isinstance(geom, (LineString, MultiLineString)):
        coords = list(geom.coords) if isinstance(geom, LineString) else \
                 [c for ls in geom.geoms for c in ls.coords]
        if len(coords) >= 2:
            crow = math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
            if crow > MIN_CROW_FLY_M and geom.length / crow > SINUOSITY_THRESHOLD:
                return True
    return False


def _is_centerline(
    facility_geom: BaseGeometry,
    road_geom: BaseGeometry,
    road_idx,
    sub_facility_id: str,
    populated: gpd.GeoDataFrame,
) -> bool:
    """Check whether a separate facility edge actually coincides with the
    street centerline (a common OSM tagging error).

    Uses ``CENTERLINE_COINCIDENCE_THRESHOLD_M`` (default 1 m) and requires
    ≥95% of the facility length to fall within that corridor.  When confirmed,
    clears the geometry and marks ``buffered=True`` so the buffering pass knows
    to generate a perpendicular offset for this row.

    Returns True when the facility was flagged as centerline-coincident.
    """
    threshold = CENTERLINE_COINCIDENCE_THRESHOLD_M

    if facility_geom.distance(road_geom) > threshold:
        return False

    corridor = road_geom.buffer(threshold)
    covered = facility_geom.intersection(corridor).length
    if covered / facility_geom.length < 0.95:
        return False

    # Confirmed centerline-coincident: clear geometry and mark row for buffering
    geom_col = f"{sub_facility_id}_geometry"
    if geom_col in populated.columns:
        populated.at[road_idx, geom_col] = None

    buffered_col = f"{sub_facility_id}_buffered"
    if buffered_col in populated.columns:
        populated.at[road_idx, buffered_col] = True

    return True


_DEFAULT_LANE_WIDTH_M  = 3.5   # fallback when lane_width is missing (overridden by per-city config)
_DEFAULT_BIKE_WIDTH_M  = 1.5   # fallback when a bikeway width cell is missing
_DEFAULT_MAXSPEED      = 25    # fallback maxspeed (mph) when OSM tag is absent (overridden by per-city config)


def _parse_numeric(val, default: float) -> float:
    """Coerce *val* to float, returning *default* on failure or NaN."""
    try:
        result = float(val)
        if result != result:  # NaN check
            return default
        return result
    except (TypeError, ValueError):
        return default


def _buffer_segment(
    street_id,
    sub_facility_id: str,
    facility_geom,
    populated: gpd.GeoDataFrame,
    side: str,
    debug: dict[str, Any],
    default_lane_width_m: float = _DEFAULT_LANE_WIDTH_M,
) -> gpd.GeoDataFrame:
    """Offset a bikelane or sidewalk geometry away from its street centerline.

    1. Computes a perpendicular offset distance based on facility type:
       - **Bikelane**: ``(lanes × lane_width) / 2``
       - **Sidewalk**: ``(lanes × lane_width) / 2 + sum(bikeway widths on same side)``
    2. Generates a parallel-offset LineString in the direction of *side*.
       Falls back to progressively smaller offsets if the geometry is degenerate.
    3. Row-scoped collision check against same-row facility geometries (spec step 4).
    4. Writes the geometry and sets the ``*_buffered`` flag.

    Parameters
    ----------
    street_id :
        Index label of the parent street-centerline row in *populated*.
    sub_facility_id : str
        Schema column prefix identifying the facility slot, e.g.
        ``"bikeway_left_1"``, ``"bikeway_right_2"``, or ``"sidewalk_left"``.
    facility_geom : shapely geometry
        Street centerline geometry in the projected CRS (metres).
    populated : GeoDataFrame
        The proximity-schema GeoDataFrame, modified in place.
    side : str
        ``"left"`` or ``"right"`` — the side of the street the facility occupies.

    Returns
    -------
    GeoDataFrame
        *populated* with the facility slot updated.
    """
    row = populated.loc[street_id]

    # --- 1. Parse facility kind and slot number from the prefix ---------------
    # Expected forms: "bikeway_left_1", "bikeway_right_2", "sidewalk_left", "sidewalk_right"
    parts = sub_facility_id.split("_")          # e.g. ["bikeway","left","1"]
    facility_kind = parts[0]                    # "bikeway" | "sidewalk"

    # --- 2. Compute perpendicular offset distance (metres) --------------------
    raw_lanes = row.get("lanes")
    raw_lane_width = row.get("lane_width")
    lanes      = _parse_numeric(raw_lanes,      2.0)
    lane_width = _parse_numeric(raw_lane_width, default_lane_width_m)
    half_road  = (lanes * lane_width) / 2.0   # distance from centreline to kerb edge

    # Track NaN sources
    if _is_na(raw_lanes) or _is_na(raw_lane_width):
        debug.setdefault("nan_source_counts", {})
        key = f"lanes={'NaN' if _is_na(raw_lanes) else 'ok'}|width={'NaN' if _is_na(raw_lane_width) else 'ok'}"
        debug["nan_source_counts"][key] = debug["nan_source_counts"].get(key, 0) + 1

    if facility_kind == "bikeway":
        offset_m = half_road

    elif facility_kind == "sidewalk":
        # Add widths of all bikeway slots on the same side
        bike_width = 0.0
        for slot in ("1", "2"):
            w = row.get(f"bikeway_{side}_{slot}_width")
            bike_width += _parse_numeric(w, 0.0) if not _is_na(w) else _DEFAULT_BIKE_WIDTH_M \
                if not _is_na(row.get(f"bikeway_{side}_{slot}_type")) else 0.0
        offset_m = half_road + bike_width

    else:
        return populated  # unknown facility kind — nothing to do

    # --- 4. Generate the parallel-offset geometry -----------------------------
    # Use offset_curve (Shapely ≥ 2.0): positive = left, negative = right.
    # If the full offset fails (empty/degenerate), try progressively smaller
    # offsets down to 25% of the original distance.
    sign = 1 if side == "left" else -1
    buffered_geom: BaseGeometry | None = None
    for fraction in (1.0, 0.75, 0.5, 0.25):
        try:
            cur_offset = sign * offset_m * fraction
            if hasattr(facility_geom, "offset_curve"):
                candidate = facility_geom.offset_curve(cur_offset)
            else:
                candidate = facility_geom.parallel_offset(
                    abs(cur_offset), side=side, resolution=16, join_style=2,
                )
            if not candidate.is_empty:
                buffered_geom = candidate
                break
        except Exception:
            continue
    if buffered_geom is None:
        debug["n_empty_offset"] += 1
        # Track why offsets are empty
        gt = facility_geom.geom_type if hasattr(facility_geom, "geom_type") else "unknown"
        fl = facility_geom.length if hasattr(facility_geom, "length") else -1
        key = f"{gt}|len<{1 if fl < 1 else 5 if fl < 5 else 10 if fl < 10 else 50 if fl < 50 else 'big'}"
        debug.setdefault("empty_offset_detail", {})
        debug["empty_offset_detail"][key] = debug["empty_offset_detail"].get(key, 0) + 1
        if debug["n_empty_offset"] <= 3:
            print(f"    [EMPTY_OFFSET] id={street_id}, sub={sub_facility_id}, "
                  f"geom_type={gt}, length={fl:.4f}, offset_m={offset_m:.2f}")
        return populated

    # --- 5. Collision checks -----------------------------------------------
    own_geom_col    = f"{sub_facility_id}_geometry"
    endpoint_buffer = buffered_geom.boundary.buffer(1e-6)

    def _mid_intersection_clear(a, b) -> bool:
        """Return True if a∩b is empty or confined to endpoints only."""
        if not a.intersects(b):
            return True
        return a.intersection(b).within(endpoint_buffer)

    # Row-level: check same-row facility geometries (spec step 4)
    # Bikeways are closer to the centerline than sidewalks, so same-side
    # sidewalk intersections are geometry-precision artifacts (the separate
    # sidewalk follows its own mapped path, not a perfect parallel offset).
    # Skip those checks to avoid false-positive rejections.
    same_side_sidewalk_col = f"sidewalk_{side}_geometry"

    _FACILITY_GEOM_COLS = [
        "street_geometry",
        "bikeway_left_1_geometry",  "bikeway_left_2_geometry",
        "bikeway_right_1_geometry", "bikeway_right_2_geometry",
        "sidewalk_left_geometry",   "sidewalk_right_geometry",
    ]
    for col in _FACILITY_GEOM_COLS:
        if col == own_geom_col or col not in populated.columns:
            continue
        if facility_kind == "bikeway" and col == same_side_sidewalk_col:
            continue
        other_geom = populated.at[street_id, col]
        if other_geom is None or not hasattr(other_geom, "intersects"):
            continue
        if not _mid_intersection_clear(buffered_geom, other_geom):
            debug["collision_counts"][col] = debug["collision_counts"].get(col, 0) + 1
            return populated

    # Network-level check removed: offset facilities naturally cross
    # perpendicular streets at intersections in grid networks.
    # Collision checking is row-scoped only (per spec step 4).

    geom_col = own_geom_col
    if geom_col in populated.columns:
        populated.at[street_id, geom_col] = buffered_geom  # type: ignore[index]

    # --- 6. Mark the slot as buffered -----------------------------------------
    buffered_col = f"{sub_facility_id}_buffered"
    if buffered_col in populated.columns:
        populated.at[street_id, buffered_col] = True

    return populated


def _is_na(val) -> bool:
    """Return True if *val* is pandas/numpy NA or None."""
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


_CURB_RAMP_PROXIMITY_M = 20.0
# Sidewalk endpoints within this distance of any intersection node get a curb
# ramp even when their parent road segment does not directly touch that node.
# Catches separate sidewalk segments that terminate near (but not at) an
# intersection — e.g. octagon-corner footways ~12 m from the centreline node.


def _assign_curb_ramp_geometries(populated: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Place curb ramp points at sidewalk endpoints adjacent to intersections.

    Curb ramps belong to sidewalk segments only.  A ramp is placed at each
    end of a sidewalk geometry whose endpoint falls within
    ``_CURB_RAMP_PROXIMITY_M`` of any intersection node.  The ramp coordinate
    is the sidewalk endpoint itself — never a street centerline node.

    Slots 2 and 3 are reserved for crowdsourced / government data.
    """
    def _endpoint(geom: BaseGeometry | None, which: str) -> Point | None:
        if not isinstance(geom, BaseGeometry) or geom.is_empty:
            return None
        coords = _flatten_coords(geom)
        if not coords:
            return None
        return Point(coords[0] if which == "start" else coords[-1])

    # ── Build STRtree of all intersection node positions ─────────────────────
    node_pts: list[Point] = []
    seen_xy: set[tuple[float, float]] = set()
    for node_col, flag_col in (
        ("start_node_geometry", "start_node_is_intersection_node"),
        ("end_node_geometry",   "end_node_is_intersection_node"),
    ):
        if node_col not in populated.columns or flag_col not in populated.columns:
            continue
        mask = populated[flag_col] == True  # noqa: E712
        for geom in populated.loc[mask, node_col]:
            if not isinstance(geom, BaseGeometry) or geom.is_empty:
                continue
            point = cast(Point, geom)
            xy = (round(point.x, 2), round(point.y, 2))
            if xy not in seen_xy:
                seen_xy.add(xy)
                node_pts.append(point)
    intersect_tree: STRtree | None = STRtree(node_pts) if node_pts else None

    n_assigned = 0

    # ── Proximity-based: sidewalk endpoint → nearest intersection node ────────
    if intersect_tree is not None:
        for side in ("left", "right"):
            geom_col = f"sidewalk_{side}_geometry"
            if geom_col not in populated.columns:
                continue
            for position in ("start", "end"):
                ramp_col = f"sidewalk_{side}_curbramp_{position}_1_geometry"
                if ramp_col not in populated.columns:
                    continue
                has_geom = populated[geom_col].apply(
                    lambda g: isinstance(g, BaseGeometry) and not g.is_empty
                )
                ramp_empty = populated[ramp_col].isna()
                for idx in populated.index[has_geom & ramp_empty]:
                    geom = cast(BaseGeometry | None, populated.at[idx, geom_col])
                    pt = _endpoint(geom, position)
                    if pt is None:
                        continue
                    nearby = intersect_tree.query(pt.buffer(_CURB_RAMP_PROXIMITY_M))
                    if len(nearby) > 0 and any(
                        node_pts[i].distance(pt) <= _CURB_RAMP_PROXIMITY_M
                        for i in nearby
                    ):
                        populated.at[idx, ramp_col] = cast(BaseGeometry, pt)  # type: ignore[index]
                        n_assigned += 1

    print(f"Curb ramp geometries assigned: {n_assigned} ramps")
    return populated


def _assign_facility_grid_ids(populated: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign grid IDs and sequential IDs to sidewalk and bikeway slots.

    IDs are derived from the parent street's grid ID with a side/slot suffix:
      sidewalk_left_ID  / sidewalk_left_grid_ID   → "{street_grid_id}L"
      sidewalk_right_ID / sidewalk_right_grid_ID  → "{street_grid_id}R"
      bikeway_left_1_id / bikeway_left_1_grid_id  → "{street_grid_id}L1"
      bikeway_left_2_id                           → "{street_grid_id}L2"
      bikeway_right_1_id / bikeway_right_1_grid_id → "{street_grid_id}R1"
      bikeway_right_2_id                          → "{street_grid_id}R2"

    bikeway_left_1_grid_id / bikeway_right_1_grid_id are assigned whenever
    *either* slot 1 or slot 2 is eligible (per schema note that slot 1 grid_id
    serves both slots).  Slot 2 has no separate grid_id column.

    Eligibility:
      Sidewalk: presence NOT in {'no', 'none'} — NaN/unknown IS eligible
      Bikeway:  type    NOT in {'no', 'none'} — NaN/unknown IS eligible
    """
    _ABSENT = {"no", "none"}
    sgid = populated["street_grid_id"] if "street_grid_id" in populated.columns else None

    def _absent_mask(col: str) -> "pd.Series[bool]":
        """True where the column value is a known-absent value ('no'/'none').
        NaN/None values are NOT absent (they are eligible/unknown)."""
        if col not in populated.columns:
            return pd.Series(True, index=populated.index)
        s = populated[col]
        na_mask = s.isna()
        # NaN rows → False (not absent); only 'no'/'none' strings → True
        return (~na_mask) & s.astype(str).str.strip().str.lower().isin(_ABSENT)

    def _write_ids(id_col: str, grid_id_col: str | None, suffix: str,
                   eligible_mask: "pd.Series[bool]") -> None:
        """Write id and grid_id columns where eligible and street_grid_id is not null."""
        if sgid is None:
            return
        active = eligible_mask & sgid.notna()
        val_series = sgid[active].astype(str) + suffix
        if id_col in populated.columns:
            populated[id_col] = pd.NA
            populated.loc[active, id_col] = val_series
        if grid_id_col is not None and grid_id_col in populated.columns:
            populated[grid_id_col] = pd.NA
            populated.loc[active, grid_id_col] = val_series

    # ── Sidewalks ──────────────────────────────────────────────────────────────
    for side, suffix in (("left", "L"), ("right", "R")):
        eligible = ~_absent_mask(f"sidewalk_{side}_presence")
        _write_ids(
            f"sidewalk_{side}_ID",
            f"sidewalk_{side}_grid_ID",
            suffix,
            eligible,
        )

    # ── Bikeways ───────────────────────────────────────────────────────────────
    for side, suffix in (("left", "L"), ("right", "R")):
        absent1 = _absent_mask(f"bikeway_{side}_1_type")
        absent2 = _absent_mask(f"bikeway_{side}_2_type")
        eligible1 = ~absent1
        eligible2 = ~absent2
        either_eligible = eligible1 | eligible2

        # Slot 1 id and the shared grid_id (covers both slots)
        _write_ids(
            f"bikeway_{side}_1_id",
            f"bikeway_{side}_1_grid_id",
            f"{suffix}1",
            either_eligible,   # grid_id assigned if any slot present
        )
        # Slot 1 id should only be set when slot 1 itself is eligible; fix it
        if f"bikeway_{side}_1_id" in populated.columns and sgid is not None:
            only1 = eligible1 & sgid.notna()
            populated[f"bikeway_{side}_1_id"] = pd.NA
            populated.loc[only1, f"bikeway_{side}_1_id"] = sgid[only1].astype(str) + f"{suffix}1"

        # Slot 2 id (no separate grid_id column per schema)
        if f"bikeway_{side}_2_id" in populated.columns and sgid is not None:
            only2 = eligible2 & sgid.notna()
            populated[f"bikeway_{side}_2_id"] = pd.NA
            populated.loc[only2, f"bikeway_{side}_2_id"] = sgid[only2].astype(str) + f"{suffix}2"

    n_sw_l = populated["sidewalk_left_ID"].notna().sum() if "sidewalk_left_ID" in populated.columns else 0
    n_sw_r = populated["sidewalk_right_ID"].notna().sum() if "sidewalk_right_ID" in populated.columns else 0
    n_bk_l = populated["bikeway_left_1_id"].notna().sum() if "bikeway_left_1_id" in populated.columns else 0
    n_bk_r = populated["bikeway_right_1_id"].notna().sum() if "bikeway_right_1_id" in populated.columns else 0
    print(f"Facility grid IDs assigned: sidewalk_left={n_sw_l}, sidewalk_right={n_sw_r}, "
          f"bikeway_left={n_bk_l}, bikeway_right={n_bk_r}")

    return populated

# def create_boundary_grid():
#     return

# def block_assignment():
#     return

# def intersection_analysis():
#     return

if __name__ == "__main__":
    run_multi_city()

