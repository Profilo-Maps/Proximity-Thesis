import math
import pickle
import re
import time
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import osmnx as ox
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, cast
from tqdm import tqdm
import shapely
from shapely import STRtree
from shapely.ops import nearest_points, substring as _sw_substring, linemerge as _sw_linemerge
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.geometry.collection import GeometryCollection

# Pre-compiled regex for extracting a leading numeric value from OSM tag strings
# (e.g. "25 mph", "3.5 m", "-3.2%").  Compiled once at import time.
_NUMERIC_PREFIX_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)")


def _add_cols(gdf: gpd.GeoDataFrame, new_cols: "dict[str, Any]") -> gpd.GeoDataFrame:
    """Add new columns to a GeoDataFrame without fragmenting the BlockManager.

    ``GeoDataFrame.assign(**kwargs)`` calls ``copy()`` then ``__setitem__``
    once per key, inserting one new block per column.  With large dicts this
    exceeds pandas' fragmentation threshold and triggers PerformanceWarning.

    This function instead builds a single supplementary DataFrame and joins it
    via ``pd.concat(axis=1)``, which produces at most one new block per dtype
    group regardless of how many columns are added.
    """
    if not new_cols:
        return gdf
    extra = pd.DataFrame(new_cols, index=gdf.index)
    return gpd.GeoDataFrame(
        pd.concat([gdf, extra], axis=1),
        geometry=gdf.geometry.name,
        crs=gdf.crs,
    )

# --- Performance Tracking ---
_SLOW_STEP_THRESHOLD_S = 300.0  # 5 minutes

_slow_steps: list[tuple[str, float]] = []


@contextmanager
def _track_step(label: str) -> Generator[None, None, None]:
    """Time a pipeline step; log and record it if it exceeds the threshold."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if elapsed >= _SLOW_STEP_THRESHOLD_S:
        mins = elapsed / 60.0
        print(f"  SLOW STEP [{mins:.1f} min]: {label}")
        _slow_steps.append((label, elapsed))


def _print_slow_step_summary() -> None:
    """Print a summary of all slow steps at the end of a pipeline run."""
    if not _slow_steps:
        print("Performance: no steps exceeded 5 min.")
        return
    print(f"\n{'='*60}")
    print(f"SLOW STEPS (>{_SLOW_STEP_THRESHOLD_S / 60:.0f} min threshold):")
    print(f"{'='*60}")
    for label, elapsed in sorted(_slow_steps, key=lambda x: -x[1]):
        print(f"  {elapsed / 60:.1f} min - {label}")
    print(f"{'='*60}\n")


# --- Global Config ---
OUTPUT_DIR = Path("Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TODO: REMOVE BEFORE PRODUCTION - temporary OSM graph cache for faster testing
CACHE_DIR = Path("Implementations") / ".osm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Maximum distance (metres) for a separate facility edge to be considered
# coincident with the street centerline.  Edges within this threshold for ≥95%
# of their length are treated as centerline data (offset replaces the
# original geometry).  Increase to catch more OSM tagging errors; decrease to
# preserve close-but-genuinely-separate facilities.
CENTERLINE_COINCIDENCE_THRESHOLD_M = 2.0

# Maximum distance (metres) from a street centerline to a separate sidewalk's
# midpoint before offset is suppressed on that side.  A parallel separate
# sidewalk within this radius indicates the street already has sidewalk geometry
# and does not need an offset.
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

    normalized = np.sum(incoming * outgoing, axis=1) / (incoming_len * outgoing_len)
    dot = np.where(valid, normalized, 1.0)
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

    # Pre-extract column arrays to avoid expensive .loc[].to_dict() per row
    _sds_cols = edges_reset.columns.tolist()
    _sds_arrays = {c: edges_reset[c].values for c in _sds_cols}

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

        row_data = {c: _sds_arrays[c][idx] for c in _sds_cols}
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
        print(f"Deflection splitting: 0 segments split (threshold={deflection_threshold_deg} deg)")
        return edges_reset

    n_segments_split = len(rows_to_drop)
    result = edges_reset.drop(index=rows_to_drop)
    new_gdf = gpd.GeoDataFrame(new_rows, geometry="geometry", crs=edges_reset.crs)
    result = gpd.GeoDataFrame(
        pd.concat([result, new_gdf], ignore_index=True),
        geometry="geometry",
        crs=edges_reset.crs,
    )

    print(f"Deflection splitting: {n_segments_split} segments -> "
          f"{len(new_rows)} pieces (threshold={deflection_threshold_deg} deg)")

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

        # --- 2. Raw bearings (vectorized via .apply) ---
        pbar.set_postfix_str("computing raw bearings")
        _raw_bear_series = _linestring_bearings_vectorized(edges_reset["geometry"])
        raw_bearings: dict[int, float] = cast(dict[int, float], _raw_bear_series.dropna().to_dict())
        pbar.update(1)

        # --- 3. Bearing normalization ---
        pbar.set_postfix_str("normalizing bearings")
        # Determine one-way status per row
        oneway_col = edges_reset["oneway"] if "oneway" in edges_reset.columns else pd.Series(
            False, index=edges_reset.index
        )

        edge_bearings: dict[int, float] = {}

        # One-way segments: use direction of travel (vectorized)
        raw_bear_s = pd.Series(raw_bearings, dtype=np.float64)
        ow_at_raw = oneway_col.reindex(raw_bear_s.index)
        fwd_mask = ow_at_raw.isin([True, "yes", "1", 1])
        rev_mask = ow_at_raw.isin(["-1", "reverse"])
        edge_bearings.update(cast(dict[int, float], raw_bear_s[fwd_mask].to_dict()))
        edge_bearings.update(cast(dict[int, float], ((raw_bear_s[rev_mask] + 180.0) % 360).to_dict()))

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

        # Majority vote per named group (vectorized per group)
        for name, indices in name_groups.items():
            bearings_arr = np.array([raw_bearings[i] for i in indices], dtype=np.float64)
            north_count = int((bearings_arr < 180.0).sum())
            if north_count >= len(indices) - north_count:
                # Majority is [0, 180): flip any in [180, 360)
                flipped = np.where(bearings_arr >= 180.0, (bearings_arr + 180.0) % 360, bearings_arr)
            else:
                # Majority is [180, 360): flip any in [0, 180)
                flipped = np.where(bearings_arr < 180.0, (bearings_arr + 180.0) % 360, bearings_arr)
            for i, b in zip(indices, flipped):
                edge_bearings[i] = float(b)

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
    _eb_index = pd.Index(edge_bearings.keys())
    n_oneway = int(oneway_col.reindex(_eb_index).isin([True, "yes", "1", 1, "-1", "reverse"]).sum())
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
) -> gpd.GeoDataFrame:
    """Return a new GeoDataFrame with grid-derived columns added."""
    idx = edges_reset.index
    intersection_ids = grid_result.intersection_node_ids

    # Synthetic split nodes have negative IDs and are never in intersection_ids
    # (which contains only positive OSM node IDs), so .isin() handles them correctly.
    return _add_cols(populated, {
        "street_grid_id": pd.Series(grid_result.edge_grid_ids, dtype=object).reindex(idx),
        "normalized_bearing": pd.Series(grid_result.edge_bearings, dtype=float).reindex(idx),
        "start_node_is_intersection_node": edges_reset["u"].isin(intersection_ids),
        "end_node_is_intersection_node": edges_reset["v"].isin(intersection_ids),
    })


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
    raw_bear = _linestring_bearings_vectorized(edges_reset["geometry"])

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
        print(f"Facility bearing swap: {n_swapped} edges swapped left<->right "
              f"(pairs={len(swap_pairs)}, compared={n_both}, "
              f"norm_only={n_norm_only}, raw_only={n_raw_only}, neither={n_neither})")
        print(f"  Bearing diff distribution: exact(<0.1 deg)={n_exact}, "
              f"small(0.1-30 deg)={n_small}, mid(30-150 deg)={n_mid}, "
              f"reversed(>=150 deg)={n_rev} | mean={mean_diff:.1f} deg, max={max_diff:.1f} deg")
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


# --- Vectorized parse helpers (replace per-row .apply() calls) ---

def _parse_maxspeed_series(s: Any) -> pd.Series:  # type: ignore[type-arg]
    """Vectorized _parse_maxspeed: extract leading integer from OSM maxspeed strings."""
    ser = pd.Series(s) if not isinstance(s, pd.Series) else s
    str_vals = ser.astype(str).str.strip()
    extracted = str_vals.str.extract(r'([+-]?\d+(?:\.\d+)?)', expand=False)
    numeric = pd.to_numeric(extracted, errors='coerce')
    # Floor toward zero to match int(float(x)) behavior; wrap for Pylance
    return pd.Series(np.trunc(numeric.values), index=numeric.index).astype('Int64')


def _parse_int_tag_series(s: Any) -> pd.Series:  # type: ignore[type-arg]
    """Vectorized _parse_int_tag: convert OSM tag values to nullable int."""
    ser = pd.Series(s) if not isinstance(s, pd.Series) else s
    numeric = pd.to_numeric(ser, errors='coerce')
    return pd.Series(np.trunc(numeric.values), index=numeric.index).astype('Int64')


def _parse_float_tag_series(s: Any) -> pd.Series:  # type: ignore[type-arg]
    """Vectorized _parse_float_tag: extract leading float from OSM tag strings."""
    ser = pd.Series(s) if not isinstance(s, pd.Series) else s
    str_vals = ser.astype(str).str.strip()
    extracted = str_vals.str.extract(r'([+-]?\d+(?:\.\d+)?)', expand=False)
    return pd.to_numeric(extracted, errors='coerce')


def _parse_incline_series(s: Any) -> pd.Series:  # type: ignore[type-arg]
    """Vectorized _parse_incline: convert OSM incline strings to float %."""
    ser = pd.Series(s) if not isinstance(s, pd.Series) else s
    str_vals = ser.astype(str).str.strip().str.lower()
    str_vals = str_vals.str.replace('°', '', regex=False).str.replace('%', '', regex=False).str.strip()
    str_vals = str_vals.where(~str_vals.isin(('up', 'down')))
    return pd.to_numeric(str_vals, errors='coerce').round(4)


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
    bbox_str = f"({bbox[0]:.4f},{bbox[1]:.4f}) -> ({bbox[2]:.4f},{bbox[3]:.4f})"
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

    _invalid = np.isnan(sampled) | (sampled < -1000)
    _rounded = np.round(sampled, 3)
    result: dict[tuple[float, float], float | None] = {
        pt: (None if inv else float(v))
        for pt, v, inv in zip(us_pts, _rounded, _invalid)
    }

    valid_elevs = [v for v in result.values() if v is not None]
    n_missing = len(result) - len(valid_elevs)
    if valid_elevs:
        print(f"  Topography DEM: {len(valid_elevs)} elevations sampled "
              f"({n_missing} NaN/clipped), "
              f"range {min(valid_elevs):.1f}-{max(valid_elevs):.1f} m")
    else:
        print(f"  Topography DEM: no valid elevations returned - "
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

            _inv_mask = np.isnan(sampled) | (sampled < -1000)
            _rnd_vals = np.round(sampled, 3)
            for pt, val, inv in zip(pts, _rnd_vals, _inv_mask):
                if inv:
                    result[pt] = None
                    n_nan += 1
                else:
                    result[pt] = float(val)
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
    import py3dep
    import xarray as xr
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

    print(f"  Topography DEM: {len(us_pts)} endpoints -> {n_cells} x {CELL_DEG} deg cells")

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

            _inv_mask = np.isnan(vals) | (vals < -1000)
            _rnd_vals = np.round(vals, 3)
            for i, val, inv in zip(idxs, _rnd_vals, _inv_mask):
                if inv:
                    result[us_pts[i]] = None
                    n_nan += 1
                else:
                    result[us_pts[i]] = float(val)
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

    print(f"  Topography: {len(populated)} segments -> {len(us_pts)} unique US endpoints to query")

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

    populated = _add_cols(populated, {"street_incline": slopes_arr})

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
    tc_x, tc_y = tc_point.x, tc_point.y
    for idx in candidate_idxs:
        geom: BaseGeometry = edges_reset.geometry.iloc[idx]  # type: ignore[assignment]
        coords_arr = shapely.get_coordinates(geom)
        dists = np.sqrt((coords_arr[:, 0] - tc_x) ** 2 + (coords_arr[:, 1] - tc_y) ** 2)
        min_d = float(dists.min())
        if min_d <= _TC_COINCIDENCE_THRESHOLD_M and min_d < best_dist:
            best_dist = min_d
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
    print("  Street features: querying traffic calming nodes...")
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
    # Use pre-extracted arrays instead of iterrows() for speed
    _tc_geoms = tc_gdf.geometry.values
    _tc_types = tc_gdf["traffic_calming"].values
    _tc_attr_cols = [c for c in tc_gdf.columns if c not in _SKIP_COLS]
    _tc_col_arrays = {c: tc_gdf[c].values for c in _tc_attr_cols}

    for _tc_pos in tqdm(range(len(tc_gdf)), total=len(tc_gdf),
                        desc="Matching traffic calming", unit="node"):
        pt: Point = _tc_geoms[_tc_pos]  # type: ignore[assignment]
        tc_type: str = str(_tc_types[_tc_pos])
        seg_idx = _find_coincident_segment(pt, edges_reset, edge_sindex)
        if seg_idx is None:
            continue
        matched += 1
        attrs: dict[str, Any] = {
            c: _tc_col_arrays[c][_tc_pos] for c in _tc_attr_cols
            if pd.notna(_tc_col_arrays[c][_tc_pos])
        }
        seg_features.setdefault(seg_idx, []).append(
            (f"traffic_calming:{tc_type}", pt, attrs)
        )

    print(f"  Street features: matched {matched}/{len(tc_gdf)} nodes to segments "
          f"across {len(seg_features)} segments.")

    # Ensure feature columns exist before cell-level .at[] writes (which
    # fail on nonexistent columns when the value is a list/iterable).
    _feature_cols = ("street_feature_types", "public_data_id_street_feature",
                     "street_feature_geometry", "street_feature_geometry_projected",
                     "street_feature_attributes")
    _missing_fc = {fc: pd.array([pd.NA] * len(populated), dtype=object) for fc in _feature_cols if fc not in populated.columns}
    if _missing_fc:
        populated = _add_cols(populated, _missing_fc)

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

    _slow_steps.clear()
    pipeline_t0 = time.perf_counter()

    # --- Vertex deflection splitting ---
    with _track_step("split_deflected_segments"):
        edges_reset = split_deflected_segments(edges_reset)

    # --- Grid assignments (bearings, grid IDs, intersection nodes) ---
    with _track_step("compute_grid_assignments"):
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

    # Build the GeoDataFrame with only the geometry column to avoid a huge
    # up-front allocation (225 object columns × 600k+ rows triggers an OOM
    # during pandas block consolidation).  Missing schema columns are added
    # lazily as each pipeline step writes to them, and any still-absent
    # columns are back-filled with pd.NA before export.
    schema_columns = list(schema.columns)
    populated = gpd.GeoDataFrame(
        {"street_geometry": edges_reset["geometry"].values},
        index=edges_reset.index,
        geometry="street_geometry",
        crs="EPSG:32610",
    )

    # Build all initial columns at once to avoid DataFrame fragmentation
    n = len(populated)
    raw_maxspeed = _get("maxspeed")
    maxspeed_s = (_parse_maxspeed_series(raw_maxspeed) if raw_maxspeed is not None
                  else pd.array([pd.NA] * n, dtype="Int64"))
    raw_lanes = _get("lanes")
    lanes_s = (_parse_int_tag_series(raw_lanes) if raw_lanes is not None
               else pd.array([pd.NA] * n, dtype="Int64"))
    raw_lane_width = _get("lane_width")
    lw_s = (_parse_float_tag_series(raw_lane_width) if raw_lane_width is not None
            else pd.array([pd.NA] * n, dtype="Float64"))
    node_geom = nodes["geometry"]
    init_cols: dict[str, Any] = {
        "street_id":             _get("osmid"),
        "start_node_id":         _get("u"),
        "end_node_id":           _get("v"),
        "name":                  _get("name"),
        "highway":               _get("highway"),
        "junction":              _get("junction"),
        "oneway":                _get("oneway"),
        "surface":               _get("surface"),
        "maxspeed":              pd.Series(maxspeed_s, index=populated.index).fillna(default_maxspeed).astype("Int64"),
        "lanes":                 pd.Series(lanes_s, index=populated.index).astype("Int64"),
        "lane_width":            pd.Series(lw_s, index=populated.index).astype("Float64"),
        "start_node_geometry":   edges_reset["u"].map(node_geom),
        "end_node_geometry":     edges_reset["v"].map(node_geom),
    }
    populated = _add_cols(populated, init_cols)

    # Grid columns: street_grid_id, normalized_bearing, intersection node flags
    populated = _populate_grid_columns(populated, edges_reset, grid_result)

    # --- Street centerline features (pipeline step 3) ---
    with _track_step("populate_street_features"):
        populated = populate_street_features(populated, edges_reset, place)

    # --- Topography enrichment (pipeline step 5) ---
    if ENRICH_TOPOGRAPHY:
        with _track_step("enrich_topography"):
            populated = enrich_topography(populated, place)

    with _track_step("populate_base_bikelanes"):
        populated = populate_base_bikelanes(populated, edges_reset)
    with _track_step("populate_base_footlanes"):
        populated = populate_base_footlanes(populated, edges_reset)
    # Swap left/right facility columns on edges with reversed geometry (deferred
    # until after tag population so all sidewalk/bikeway columns exist)
    with _track_step("swap_facilities_by_bearing"):
        populated = swap_facilities_by_bearing(populated, edges_reset)
    # Defragment before the expensive separate-facility matching loops.
    # Column insertions in earlier steps leave the DataFrame with many small
    # memory blocks; copying consolidates them into a single contiguous layout.
    populated = populated.copy()
    with _track_step("_populate_separate_facilities"):
        populated = _populate_separate_facilities(populated, edges_reset, default_lane_width_m)
    with _track_step("_assign_facility_grid_ids"):
        populated = _assign_facility_grid_ids(populated)
    with _track_step("_snap_offset_endpoints"):
        populated = _snap_offset_endpoints(populated, default_lane_width_m)
    with _track_step("_assign_curb_ramp_geometries"):
        populated = _assign_curb_ramp_geometries(populated, default_lane_width_m)
    # --- Ensure all schema columns exist and are ordered correctly ---
    _missing_schema = {col: pd.array([pd.NA] * len(populated), dtype=object) for col in schema_columns if col not in populated.columns}
    if _missing_schema:
        populated = _add_cols(populated, _missing_schema)
    populated = populated[
        [c for c in schema_columns if c in populated.columns]
        + [c for c in populated.columns if c not in schema_columns]
    ].copy()

    # --- Export ---
    _export_t0 = time.perf_counter()
    # Secondary geometry columns are serialized to WKB hex so they survive
    # parquet round-tripping (GeoParquet only encodes the active geometry column).

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
        # Convert geometry columns to WKB hex using vectorized shapely.to_wkb
        for col in present_geom_cols:
            pbar.set_postfix_str(col)
            col_values = populated[col].values
            valid_mask = np.asarray(shapely.is_geometry(col_values), dtype=bool)
            wkb_result = np.full(len(col_values), None, dtype=object)
            if valid_mask.any():
                wkb_result[valid_mask] = shapely.to_wkb(np.asarray(col_values[valid_mask]), hex=True)
            populated[col] = wkb_result
            pbar.update(1)

        pbar.set_postfix_str("normalizing offset columns")
        # Normalize _offset columns: pipeline writes True/False booleans while OSM
        # tags provide strings like 'yes'/'no'. Mixed types cause PyArrow serialization
        # failures, so coerce everything to consistent strings before export.
        offset_cols = [c for c in populated.columns if c.endswith("_offset")]
        for col in offset_cols:
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
            has_non_str = pd.api.types.infer_dtype(non_null, skipna=True) != "string"
            if has_non_str:
                mask_na = populated[col].isna()
                populated[col] = populated[col].astype(str)
                populated.loc[mask_na, col] = pd.NA

        place_slug = place.replace(", ", "_").replace(" ", "_")
        output_path = OUTPUT_DIR / f"{place_slug}_network.parquet"
        pbar.set_postfix_str("writing parquet")
        populated.to_parquet(output_path)
        pbar.update(1)

    _export_elapsed = time.perf_counter() - _export_t0
    if _export_elapsed >= _SLOW_STEP_THRESHOLD_S:
        print(f"  SLOW STEP [{_export_elapsed / 60:.1f} min]: export_parquet")
        _slow_steps.append(("export_parquet", _export_elapsed))

    print(f"Populated schema parquet exported to {output_path}")

    pipeline_elapsed = time.perf_counter() - pipeline_t0
    print(f"Total pipeline time: {pipeline_elapsed / 60:.1f} min")
    _print_slow_step_summary()

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
        "sidewalk_left_width", "sidewalk_left_incline", "sidewalk_left_seperator", "sidewalk_left_offset",
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
        "sidewalk_right_width", "sidewalk_right_incline", "sidewalk_right_seperator", "sidewalk_right_offset",
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
        "bikeway_left_1_seperator", "bikeway_left_1_offset",
        "bikeway_left_2_id", "public_data_id_bikeway_left_2",
        "bikeway_left_2_type", "bikeway_left_2_surface", "bikeway_left_2_quality",
        "bikeway_left_2_permitted", "bikeway_left_2_width", "bikeway_left_2_incline",
        "bikeway_left_2_seperator", "bikeway_left_2_offset",
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
        "bikeway_right_1_seperator", "bikeway_right_1_offset",
        "bikeway_right_2_id", "public_data_id_bikeway_right_2",
        "bikeway_right_2_type", "bikeway_right_2_surface", "bikeway_right_2_quality",
        "bikeway_right_2_permitted", "bikeway_right_2_width", "bikeway_right_2_incline",
        "bikeway_right_2_seperator", "bikeway_right_2_offset",
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

    new_cols: dict[str, Any] = {}

    with tqdm(total=5, desc="Loading bikeway data", unit="step") as pbar:
        # Type: prefer side-specific tag, fall back to cycleway:both, then bare cycleway
        pbar.set_postfix_str("type")
        new_cols["bikeway_left_1_type"]  = _coalesce("cycleway:left",  "cycleway:both", "cycleway")
        new_cols["bikeway_right_1_type"] = _coalesce("cycleway:right", "cycleway:both", "cycleway")
        new_cols["bikeway_left_2_type"]  = _coalesce("cycleway:left:2",  "cycleway:both:2")
        new_cols["bikeway_right_2_type"] = _coalesce("cycleway:right:2", "cycleway:both:2")
        pbar.update(1)

        # Sub-type, surface, width, separator
        pbar.set_postfix_str("quality / surface / width / separator")
        new_cols["bikeway_left_1_quality"]  = _coalesce("cycleway:left:smoothness",  "cycleway:smoothness")
        new_cols["bikeway_right_1_quality"] = _coalesce("cycleway:right:smoothness", "cycleway:smoothness")
        new_cols["bikeway_left_1_surface"]  = _coalesce("cycleway:left:surface",  "cycleway:surface")
        new_cols["bikeway_right_1_surface"] = _coalesce("cycleway:right:surface", "cycleway:surface")
        new_cols["bikeway_left_1_width"]    = _parse_float_tag_series(_coalesce("cycleway:left:width",  "cycleway:width"))
        new_cols["bikeway_right_1_width"]   = _parse_float_tag_series(_coalesce("cycleway:right:width", "cycleway:width"))
        new_cols["bikeway_left_1_seperator"]   = _coalesce("cycleway:left:buffer",  "cycleway:buffer")
        new_cols["bikeway_right_1_seperator"] = _coalesce("cycleway:right:buffer", "cycleway:buffer")
        pbar.update(1)

        # Permitted / incline
        pbar.set_postfix_str("permitted / incline")
        new_cols["bikeway_left_1_permitted"]  = _get("bicycle")
        new_cols["bikeway_right_1_permitted"] = _get("bicycle")
        _raw_incline = _get("incline")
        _parsed_incline = _parse_incline_series(_raw_incline) if _raw_incline is not None else None
        if _parsed_incline is not None:
            new_cols["bikeway_left_1_incline"]  = _parsed_incline
            new_cols["bikeway_right_1_incline"] = _parsed_incline
        else:
            new_cols["bikeway_left_1_incline"]  = pd.NA
            new_cols["bikeway_right_1_incline"] = pd.NA
        pbar.update(1)

        # Secondary bikeway slots (_2)
        pbar.set_postfix_str("secondary bikeway slots")
        new_cols["bikeway_left_2_quality"]   = _get("cycleway:left:2:smoothness")
        new_cols["bikeway_right_2_quality"]  = _get("cycleway:right:2:smoothness")
        new_cols["bikeway_left_2_surface"]   = _get("cycleway:left:2:surface")
        new_cols["bikeway_right_2_surface"]  = _get("cycleway:right:2:surface")
        _raw_bl2w = _get("cycleway:left:2:width")
        new_cols["bikeway_left_2_width"]     = _parse_float_tag_series(_raw_bl2w) if _raw_bl2w is not None else pd.NA
        _raw_br2w = _get("cycleway:right:2:width")
        new_cols["bikeway_right_2_width"]    = _parse_float_tag_series(_raw_br2w) if _raw_br2w is not None else pd.NA
        new_cols["bikeway_left_2_permitted"] = _get("bicycle")
        new_cols["bikeway_right_2_permitted"]= _get("bicycle")
        if _parsed_incline is not None:
            new_cols["bikeway_left_2_incline"]  = _parsed_incline
            new_cols["bikeway_right_2_incline"] = _parsed_incline
        else:
            new_cols["bikeway_left_2_incline"]  = pd.NA
            new_cols["bikeway_right_2_incline"] = pd.NA
        new_cols["bikeway_left_2_seperator"]  = _get("cycleway:left:2:buffer")
        new_cols["bikeway_right_2_seperator"] = _get("cycleway:right:2:buffer")
        pbar.update(1)

        # Pre-create geometry and offset columns so the separate-facility
        # matching loop never triggers __setitem__ column creation (fragmentation).
        for _side in ("left", "right"):
            for _slot in ("1", "2"):
                new_cols[f"bikeway_{_side}_{_slot}_geometry"] = None
                new_cols[f"bikeway_{_side}_{_slot}_offset"] = pd.NA
        pbar.update(1)

    return _add_cols(populated, new_cols)



def populate_base_footlanes(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Populate centerline-derived sidewalk columns from OSM sidewalk tags.

    Does not set offset or geometry — the separate facilities pass writes
    geometry where OSM footway edges exist, and the offset pass generates
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

    new_cols: dict[str, Any] = {}

    with tqdm(total=3, desc="Loading sidewalk data", unit="step") as pbar:
        # Presence: prefer side-specific, fall back to sidewalk:both, then bare sidewalk
        pbar.set_postfix_str("presence")
        new_cols["sidewalk_left_presence"]  = _coalesce("sidewalk:left",  "sidewalk:both", "sidewalk")
        new_cols["sidewalk_right_presence"] = _coalesce("sidewalk:right", "sidewalk:both", "sidewalk")
        pbar.update(1)

        # Surface, width, incline, quality, separator
        pbar.set_postfix_str("surface / width / incline / quality / separator")
        new_cols["sidewalk_left_surface"]    = _get("sidewalk:left:surface")
        new_cols["sidewalk_right_surface"]   = _get("sidewalk:right:surface")
        _raw_slw = _get("sidewalk:left:width")
        new_cols["sidewalk_left_width"]      = _parse_float_tag_series(_raw_slw) if _raw_slw is not None else pd.NA
        _raw_srw = _get("sidewalk:right:width")
        new_cols["sidewalk_right_width"]     = _parse_float_tag_series(_raw_srw) if _raw_srw is not None else pd.NA
        _raw_sw_l = _get("sidewalk:left:incline")
        new_cols["sidewalk_left_incline"]  = _parse_incline_series(_raw_sw_l) if _raw_sw_l is not None else pd.NA
        _raw_sw_r = _get("sidewalk:right:incline")
        new_cols["sidewalk_right_incline"] = _parse_incline_series(_raw_sw_r) if _raw_sw_r is not None else pd.NA
        new_cols["sidewalk_left_quality"]    = _get("sidewalk:left:smoothness")
        new_cols["sidewalk_right_quality"]   = _get("sidewalk:right:smoothness")
        new_cols["sidewalk_left_seperator"]  = _get("sidewalk:left:buffer")
        new_cols["sidewalk_right_seperator"] = _get("sidewalk:right:buffer")
        pbar.update(1)

        # Pre-create geometry and offset columns so the separate-facility
        # matching loop never triggers __setitem__ column creation (fragmentation).
        for _side in ("left", "right"):
            new_cols[f"sidewalk_{_side}_geometry"] = None
            new_cols[f"sidewalk_{_side}_offset"] = pd.NA
        pbar.update(1)

    return _add_cols(populated, new_cols)


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
    priority. Separate geometry always wins over offset geometry — if a
    matched edge is written, offset is set to False.
    - collision check during proximity matching is row-scoped per spec step 4
    """
    def _set_cell(idx: int, col: str, value: object) -> None:
        """Set a single cell, handling list/tuple values that ``pd.DataFrame.at``
        rejects with 'Must have equal len keys and value when setting with an
        iterable'.  Falls back to direct numpy array assignment."""
        # Arrow-backed columns don't support element assignment —
        # convert to object on first encounter.
        if "arrow" in str(populated[col].dtype).lower():
            populated[col] = pd.array(populated[col], dtype=object)
        try:
            populated.at[idx, col] = value  # type: ignore[index]
        except ValueError:
            populated[col].values[populated.index.get_loc(idx)] = value  # type: ignore[index]

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

    # Pre-compute road bearings once — vectorized to avoid per-road Python calls.
    _road_bearing_series = _linestring_bearings_vectorized(roads.geometry)
    _road_bearing_cache: dict[int, float | None] = cast(
        dict[int, float | None], _road_bearing_series.dropna().to_dict()
    )

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
    # Cumulative length guard: reject a footway match when the total
    # accumulated sidewalk geometry would exceed the road length by this
    # factor.  Prevents unrelated plazas / trails from piling onto short
    # service roads (e.g. Stow Plaza → Frank Schlessinger Way).
    # The ratio scales smoothly from _CUM_RATIO_MAX for very short road
    # segments (≤ _CUM_RATIO_SHORT_M) down to _CUM_RATIO_MIN for longer
    # segments (≥ _CUM_RATIO_LONG_M).  Short intersection-adjacent segments
    # need a generous ratio because block-spanning footways legitimately
    # exceed the stub segment's length.
    _CUM_RATIO_MIN = 4.0       # ratio for long road segments
    _CUM_RATIO_MAX = 12.0      # ratio for very short road segments
    _CUM_RATIO_SHORT_M = 15.0  # road length at or below which max ratio applies
    _CUM_RATIO_LONG_M = 50.0   # road length at or above which min ratio applies

    def _cumulative_sw_ratio(road_len: float) -> float:
        """Smoothly interpolate the max cumulative sidewalk/road ratio.

        Short intersection stubs get a generous ratio (block-spanning
        footways are legitimately much longer); longer mid-block segments
        use a tighter ratio to reject plazas and trails.
        """
        if road_len <= _CUM_RATIO_SHORT_M:
            return _CUM_RATIO_MAX
        if road_len >= _CUM_RATIO_LONG_M:
            return _CUM_RATIO_MIN
        # Linear interpolation between short and long thresholds
        t = (road_len - _CUM_RATIO_SHORT_M) / (_CUM_RATIO_LONG_M - _CUM_RATIO_SHORT_M)
        return _CUM_RATIO_MAX + t * (_CUM_RATIO_MIN - _CUM_RATIO_MAX)
    # Overlap deduplication: before adding a footway to a road's sidewalk slot,
    # check if an already-assigned sidewalk covers the same corridor.
    # An existing geometry is considered "covering" if ≥ this fraction of its
    # length falls within _SIDEWALK_DEDUP_BUFFER_M of the candidate geometry,
    # OR the two geometries share both start and end points.
    _SIDEWALK_DEDUP_BUFFER_M  = 3.0   # metres
    _SIDEWALK_DEDUP_THRESHOLD = 0.75  # fraction of existing segment's length

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

    def _coerce_name_raw(val) -> str | None:
        """Coerce a raw OSM name value (str, bytes, list, ndarray, NA) to str | None."""
        if isinstance(val, (list, np.ndarray)):
            val = next((x for x in val if isinstance(x, str)), None)
        if isinstance(val, bytes):
            return val.decode("utf-8")
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return val if isinstance(val, str) else None

    # Pre-cache normalized road names and street geometries for O(1) lookup
    # inside _match_road_by_name — replaces hot-path populated.at[] reads.
    _road_norm_names_cache: dict[int, str | None] = cast(
        dict[int, str | None],
        {idx: _normalize_name(v) for idx, v in roads["name"].items()}
        if "name" in roads.columns else {}
    )
    _road_street_geoms_cache: dict[int, BaseGeometry] = cast(
        dict[int, BaseGeometry],
        {idx: g for idx, g in roads["street_geometry"].items() if isinstance(g, BaseGeometry)}
        if "street_geometry" in roads.columns else {}
    )

    def _match_road_by_name(sindex, fac_geom, fac_mid, df, fac_name: str | None,
                            fac_bearing: float | None = None):
        """Find the best road segment for a facility edge using name + proximity.

        Strategy:
        1. Query all roads within ``_NAME_MATCH_RADIUS_M`` of the facility.
        2. Among roads whose name matches ``fac_name``, pick the closest.
        3. If the facility has no name, fall back to the spatially nearest road.
        4. If the facility has a name but no road matches it, return None —
           named footways/cycleways that don't match any road by name are
           independent paths (plazas, trails, etc.) and should not be
           adopted as a sidewalk of the nearest road.

        ``fac_bearing`` may be supplied as a pre-computed value to avoid a
        redundant scalar bearing computation inside the unnamed-facility branch.

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
                rname = _road_norm_names_cache.get(ridx)
                if rname != norm_fac:
                    continue
                rgeom = _road_street_geoms_cache.get(ridx)
                if rgeom is None:
                    continue
                d = rgeom.distance(fac_geom)
                if d < best_dist:
                    best_dist = d
                    best_idx = ridx
                    best_geom = rgeom
                    best_side = _road_side(rgeom, fac_mid)

            # Named facility with no road name match → independent path, skip.
            return best_idx, best_geom, best_side

        # Unnamed facility: fall back to nearest road whose bearing is
        # roughly parallel.  A sidewalk always runs alongside its road; a
        # perpendicular footway (e.g. one sharing only a node with a service
        # driveway) must not be adopted by that road.
        if fac_bearing is None:
            fac_bearing = _linestring_bearing(fac_geom)
        fac_len = fac_geom.length
        search_area = fac_geom.buffer(_NAME_MATCH_RADIUS_M)
        hit_positions = sindex.query(search_area)
        # Find the nearest parallel road.  When multiple roads are at
        # effectively the same distance (within epsilon), prefer the one
        # whose length is closest to the footway — short connector footways
        # naturally match intersection stubs, long block-spanning footways
        # match mid-block segments.
        _DIST_TIE_EPSILON_M = 2.0
        _par_candidates: list[tuple[float, int, BaseGeometry]] = []
        for pos in hit_positions:
            ridx = df.index[pos]
            rgeom = _road_street_geoms_cache.get(ridx)
            if rgeom is None:
                continue
            if fac_bearing is not None:
                road_bearing = _road_bearing_cache.get(ridx)
                if road_bearing is not None and not _bearings_parallel(fac_bearing, road_bearing):
                    continue
            d = rgeom.distance(fac_geom)
            _par_candidates.append((d, ridx, rgeom))
        if not _par_candidates:
            return None, None, None
        # Primary: distance.  Among ties (within epsilon of min), pick
        # the road whose length is closest to the footway's length.
        _min_d = min(c[0] for c in _par_candidates)
        _tied = [(d, ridx, rgeom) for d, ridx, rgeom in _par_candidates
                 if d <= _min_d + _DIST_TIE_EPSILON_M]
        _best = min(_tied, key=lambda c: abs(c[2].length - fac_len))
        return _best[1], _best[2], _road_side(_best[2], fac_mid)

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

    # Pre-compute geometry and tag properties for bikeways as positional lists.
    cy_mids_list  = shapely.line_interpolate_point(np.asarray(cycleways.geometry), 0.5, normalized=True).tolist()
    cy_names_list = list(cycleways["name"]) if "name" in cycleways.columns else [None] * len(cycleways)
    cy_geoms_list: list[BaseGeometry] = list(cycleways.geometry)
    cy_highway_list = list(cycleways["highway"]) if "highway" in cycleways.columns else [pd.NA] * len(cycleways)
    cy_surface_list = list(cycleways["surface"]) if "surface" in cycleways.columns else [pd.NA] * len(cycleways)
    cy_width_list   = list(cycleways["width"])   if "width"   in cycleways.columns else [pd.NA] * len(cycleways)
    cy_bicycle_list = list(cycleways["bicycle"]) if "bicycle" in cycleways.columns else [pd.NA] * len(cycleways)
    cy_incline_list = list(cycleways["incline"]) if "incline" in cycleways.columns else [pd.NA] * len(cycleways)
    cy_coerced_names_list = [_coerce_name_raw(n) for n in cy_names_list]
    # Pre-batch bearing computation for all cycleways (avoids per-edge scalar call inside _match_road_by_name)
    _cy_bear_arr = _linestring_bearings_vectorized(cycleways.geometry).values
    cy_bearings_list: list[float | None] = [
        None if np.isnan(b) else float(b) for b in _cy_bear_arr
    ]

    for cy_pos in tqdm(range(len(cycleways)), total=len(cycleways), desc="Matching bikeways", unit="edge"):
        cy_geom    = cy_geoms_list[cy_pos]
        cy_mid     = cy_mids_list[cy_pos]
        cy_name = cy_coerced_names_list[cy_pos]

        road_idx, road_geom, side = _match_road_by_name(
            road_sindex, cy_geom, cy_mid, roads, cy_name,
            fac_bearing=cy_bearings_list[cy_pos])
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
            _set_cell(road_idx, f"{prefix}_type",      cy_highway_list[cy_pos])
            _set_cell(road_idx, f"{prefix}_surface",   cy_surface_list[cy_pos])
            _set_cell(road_idx, f"{prefix}_width",     _parse_float_tag(cy_width_list[cy_pos]))
            _set_cell(road_idx, f"{prefix}_permitted", cy_bicycle_list[cy_pos])
            _set_cell(road_idx, f"{prefix}_incline",   _parse_incline(cy_incline_list[cy_pos]))
            _set_cell(road_idx, f"{prefix}_geometry",  cy_geom)
            _set_cell(road_idx, f"{prefix}_offset",    False)
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
    n_foot_deduped = 0
    # Debug counters for rejection reasons
    _dbg_no_match = 0
    _dbg_cum_length = 0
    _dbg_centerline = 0
    _dbg_suspect = 0

    # Track assigned sidewalk geometries for cross-road overlap deduplication.
    # Uses an STRtree rebuilt periodically for O(log n) spatial queries instead
    # of an O(n) linear scan.
    assigned_sw_geoms: list[BaseGeometry] = []
    _sw_dedup_tree: STRtree | None = None
    _sw_dedup_tree_size: int = 0  # size when tree was last built
    _SW_DEDUP_TREE_REBUILD_INTERVAL = 500  # rebuild every N new geometries

    def _rebuild_dedup_tree_if_needed() -> None:
        nonlocal _sw_dedup_tree, _sw_dedup_tree_size
        new_count = len(assigned_sw_geoms) - _sw_dedup_tree_size
        if new_count >= _SW_DEDUP_TREE_REBUILD_INTERVAL or _sw_dedup_tree is None:
            _sw_dedup_tree = STRtree(assigned_sw_geoms)
            _sw_dedup_tree_size = len(assigned_sw_geoms)

    def _is_sidewalk_covered(new_geom: BaseGeometry) -> bool:
        """True if *new_geom* substantially duplicates an already-assigned sidewalk."""
        if not assigned_sw_geoms:
            return False
        _rebuild_dedup_tree_if_needed()
        assert _sw_dedup_tree is not None
        search_buf = new_geom.buffer(_SIDEWALK_DEDUP_BUFFER_M)
        hit_positions = _sw_dedup_tree.query(search_buf)
        if len(hit_positions) == 0:
            return False
        new_pts = _flatten_coords(new_geom)
        for pos in hit_positions:
            if pos >= len(assigned_sw_geoms):
                continue  # stale tree entry; will be caught after next rebuild
            eg = assigned_sw_geoms[pos]
            # Length-coverage check: ≥ threshold of existing geometry within buffer
            covered = eg.intersection(search_buf).length
            if covered / max(eg.length, 1e-6) >= _SIDEWALK_DEDUP_THRESHOLD:
                return True
            # Shared-endpoints check (raw coord arithmetic avoids Point construction)
            eg_pts = _flatten_coords(eg)
            if new_pts and eg_pts:
                ns, ne = new_pts[0], new_pts[-1]
                es, ee = eg_pts[0], eg_pts[-1]
                if ((np.hypot(ns[0]-es[0], ns[1]-es[1]) < 0.5 and
                     np.hypot(ne[0]-ee[0], ne[1]-ee[1]) < 0.5) or
                    (np.hypot(ns[0]-ee[0], ns[1]-ee[1]) < 0.5 and
                     np.hypot(ne[0]-es[0], ne[1]-es[1]) < 0.5)):
                    return True
        return False

    # Pre-compute geometry and tag properties for footways as positional lists.
    # Avoids expensive DataFrame.iloc[] lookups (~107k times) in the hot loop.
    fw_mids_list  = shapely.line_interpolate_point(np.asarray(footways.geometry), 0.5, normalized=True).tolist()
    fw_names_list = list(footways["name"]) if "name" in footways.columns else [None] * len(footways)
    fw_geoms_list: list[BaseGeometry] = list(footways.geometry)
    fw_surface_list = list(footways["surface"]) if "surface" in footways.columns else [pd.NA] * len(footways)
    fw_width_list   = list(footways["width"])   if "width"   in footways.columns else [pd.NA] * len(footways)
    fw_incline_list = list(footways["incline"]) if "incline" in footways.columns else [pd.NA] * len(footways)
    fw_smooth_list  = list(footways["smoothness"]) if "smoothness" in footways.columns else [pd.NA] * len(footways)

    # ── Name inheritance for unnamed footways ────────────────────────────
    # Unnamed footway segments adjacent to named footways (e.g. unnamed
    # parts of "Stow Plaza") should inherit the name so that name-gating
    # in _match_road_by_name correctly rejects them instead of matching
    # them to unrelated nearby roads.
    _NAME_INHERIT_DIST_M = 3.0
    coerced_fw_names: list[str | None] = [_coerce_name_raw(n) for n in fw_names_list]
    named_fw_indices = [i for i, n in enumerate(coerced_fw_names) if n is not None]
    if named_fw_indices:
        named_fw_geoms_for_tree = [fw_geoms_list[i] for i in named_fw_indices]
        named_fw_tree = STRtree(named_fw_geoms_for_tree)
        n_name_inherited = 0
        for fw_pos in range(len(footways)):
            if coerced_fw_names[fw_pos] is not None:
                continue  # already named
            buf = fw_geoms_list[fw_pos].buffer(_NAME_INHERIT_DIST_M)
            hits = named_fw_tree.query(buf)
            for h in hits:
                if named_fw_geoms_for_tree[h].distance(fw_geoms_list[fw_pos]) <= _NAME_INHERIT_DIST_M:
                    inherited_name = coerced_fw_names[named_fw_indices[h]]
                    coerced_fw_names[fw_pos] = inherited_name
                    # Also update the raw list so the matching loop sees it
                    fw_names_list[fw_pos] = inherited_name
                    n_name_inherited += 1
                    break
        if n_name_inherited:
            print(f"  Name inheritance: {n_name_inherited} unnamed footways inherited names from adjacent named footways")

    # Pre-normalize footway names once — avoids double _normalize_name call per iteration.
    fw_norm_names_list = [_normalize_name(n) for n in coerced_fw_names]
    # Pre-batch bearing computation for all footways (avoids per-edge scalar call inside _match_road_by_name)
    _fw_bear_arr = _linestring_bearings_vectorized(footways.geometry).values
    fw_bearings_list: list[float | None] = [
        None if np.isnan(b) else float(b) for b in _fw_bear_arr
    ]

    for fw_pos in tqdm(range(len(footways)), total=len(footways), desc="Matching footways", unit="edge"):
        fw_geom = fw_geoms_list[fw_pos]
        fw_mid  = fw_mids_list[fw_pos]
        fw_name = coerced_fw_names[fw_pos]

        road_idx, road_geom, side = _match_road_by_name(
            road_sindex, fw_geom, fw_mid, roads, fw_name,
            fac_bearing=fw_bearings_list[fw_pos])
        if road_idx is None:
            _dbg_no_match += 1
            continue
        side = cast(str, side)  # type guard: side is non-None when road_idx is non-None
        # Track whether this was a name-based match
        _norm_fw = fw_norm_names_list[fw_pos]
        if _norm_fw is not None and _norm_fw == _road_norm_names_cache.get(road_idx):
            n_foot_name_matched += 1

        prefix = f"sidewalk_{side}"
        slot_key = (road_idx, side)

        # Overlap deduplication: skip if this corridor is already covered
        # by a previously assigned sidewalk geometry.  However, bypass dedup
        # when the target road slot is empty and tagged as separate — short
        # intersection stubs legitimately overlap the corridor of a block-
        # spanning footway already assigned to a neighbouring mid-block segment,
        # and starving them of geometry prevents curb-ramp detection.
        _slot_empty = slot_key not in foot_slots_used
        _pres_col = f"{prefix}_presence"
        _slot_starved = (
            _slot_empty
            and _pres_col in populated.columns
            and str(populated.at[road_idx, _pres_col]).lower() == "separate"  # type: ignore[index]
        )
        if not _slot_starved and _is_sidewalk_covered(fw_geom):
            n_foot_deduped += 1
            continue

        # Cumulative length guard: reject when accumulated sidewalk geometry
        # would far exceed the road segment length — indicates a mismatched
        # plaza / trail rather than a legitimate sidewalk.
        geom_col = f"{prefix}_geometry"
        if road_geom is not None and isinstance(road_geom, BaseGeometry):
            existing_sw = populated.at[road_idx, geom_col] if geom_col in populated.columns else None  # type: ignore[index]
            existing_len = existing_sw.length if existing_sw is not None and isinstance(existing_sw, BaseGeometry) else 0.0
            ratio = _cumulative_sw_ratio(road_geom.length)
            if (existing_len + fw_geom.length) > road_geom.length * ratio:
                _dbg_cum_length += 1
                continue

        if slot_key in foot_slots_used:
            # Deduplication: merge adjacent segments; otherwise keep the longer geometry.
            existing = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
            if existing is not None and isinstance(existing, BaseGeometry):
                # Reject reverse duplicates: OSM graphs often have both u->v
                # and v->u edges for the same footway.  Merging a reverse edge
                # creates a hairpin MultiLineString whose endpoints are far
                # from the intersection, breaking curb-ramp detection.
                _new_fc = _flatten_coords(fw_geom)
                _ex_parts = list(existing.geoms) if isinstance(existing, MultiLineString) else [existing]
                _is_rev_dup = False
                _new_arr = np.array(_new_fc, dtype=np.float64) if _new_fc else None
                for _ep in _ex_parts:
                    _ep_fc = _flatten_coords(_ep)
                    if _new_arr is not None and len(_new_fc) == len(_ep_fc) and len(_new_fc) >= 2:
                        _ep_arr = np.array(_ep_fc, dtype=np.float64)
                        if np.all(np.abs(_new_arr[::-1] - _ep_arr) < 0.05):
                            _is_rev_dup = True
                            break
                if _is_rev_dup:
                    n_foot_deduped += 1
                elif existing.distance(fw_geom) <= _FOOT_MERGE_THRESHOLD_M:
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
                        if not winner_after.is_simple:
                            populated.at[road_idx, f"{prefix}_geometry"] = None  # type: ignore[index]
                            populated.at[road_idx, f"{prefix}_offset"] = True  # type: ignore[index]
                            n_foot_suspect[side] += 1
                else:
                    # Existing is longer and not adjacent — discard new edge.
                    n_foot_replaced[side] += 1
            else:
                populated.at[road_idx, f"{prefix}_geometry"] = fw_geom  # type: ignore[index]
            n_foot_merged += 1
        else:
            # New slot — write all attributes
            _set_cell(road_idx, f"{prefix}_presence", "separate")
            _set_cell(road_idx, f"{prefix}_surface",  fw_surface_list[fw_pos])
            _set_cell(road_idx, f"{prefix}_width",    _parse_float_tag(fw_width_list[fw_pos]))
            _set_cell(road_idx, f"{prefix}_incline",  _parse_incline(fw_incline_list[fw_pos]))
            _set_cell(road_idx, f"{prefix}_quality",  fw_smooth_list[fw_pos])
            _set_cell(road_idx, f"{prefix}_geometry", fw_geom)
            _set_cell(road_idx, f"{prefix}_offset",   False)
            foot_slots_used.add(slot_key)
            if _is_centerline(fw_geom, cast(BaseGeometry, road_geom), road_idx, prefix, populated):
                _dbg_centerline += 1
            # For separate facilities, only reject self-intersecting geometry.
            # Sinuosity-based rejection is too aggressive for legitimate curved
            # sidewalks (cul-de-sac perimeters, U-turns) — prioritize separate
            # OSM geometry over algorithmic offsets.
            geom_after = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
            if geom_after is not None and isinstance(geom_after, BaseGeometry):
                if not geom_after.is_simple:
                    populated.at[road_idx, f"{prefix}_geometry"] = None  # type: ignore[index]
                    populated.at[road_idx, f"{prefix}_offset"] = True  # type: ignore[index]
                    n_foot_suspect[side] += 1
                    _dbg_suspect += 1
            n_foot_matched += 1
            # Track for cross-road overlap deduplication
            final_geom = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
            if final_geom is not None and isinstance(final_geom, BaseGeometry):
                assigned_sw_geoms.append(final_geom)

    print(f"Matched {n_foot_matched} separate footway edges to road segments "
          f"({n_foot_name_matched} by name, {n_foot_merged} merged, "
          f"{n_foot_deduped} skipped — overlap dedup).")
    print(f"  [DEBUG] Rejection breakdown: no_match={_dbg_no_match}, "
          f"cum_length={_dbg_cum_length}, dedup={n_foot_deduped}, "
          f"centerline={_dbg_centerline}, suspect={_dbg_suspect}")
    print(
        f"[sidewalk] Suspect geometries discarded -> flagged for offset: "
        f"left={n_foot_suspect['left']}, right={n_foot_suspect['right']}"
    )
    print(
        f"[sidewalk] Inferior duplicate footways discarded (fewer vertices): "
        f"left={n_foot_replaced['left']}, right={n_foot_replaced['right']}"
    )

    # ── Offset pass (spec steps 4 & 5) ───────────────────────────────────
    # For road segments with presence/type data but no geometry, generate a
    # perpendicular-offset geometry from the street centerline (offset=True).
    #
    # Sidewalk suppression: before offsetting a sidewalk slot, check whether
    # any separate sidewalk geometry (left OR right, from any road) with a
    # parallel bearing already exists within NEARBY_SEPARATE_SIDEWALK_THRESHOLD_M
    # of the street centerline.  This prevents duplicate offset sidewalks on
    # inner service roads that run parallel to a primary road whose real
    # outer sidewalk is already mapped.  The unified (left+right) tree is used
    # so that cross-slot duplicates (service road left ↔ primary road right) are
    # detected correctly.
    _NEGATIVE_VALUES = {"no", "none"}
    # Presence values indicating the sidewalk is mapped as a separate OSM way.
    # These should never trigger offset geometry.
    # Separate footway edges always write "separate" to the presence column;
    # OSM centerline tags (sidewalk:left/right=separate) also produce "separate".
    _SEPARATE_PRESENCE_VALUES = {"separate"}

    # Build unified tree of ALL separate sidewalk geometries (both sides).
    all_sep_sw_geoms: list[BaseGeometry] = []
    all_sep_sw_bearings: list[float] = []
    all_sep_sw_row_indices: list = []  # row index to prevent self-suppression
    for sw_side in ("left", "right"):
        gcol = f"sidewalk_{sw_side}_geometry"
        bcol = f"sidewalk_{sw_side}_offset"
        if gcol not in populated.columns:
            continue
        geom_series = populated[gcol]

        # Vectorized pre-filter: rows that have real geometry and are not offset
        has_geom = pd.Series(np.asarray(shapely.is_geometry(geom_series.values), dtype=bool),
                              index=populated.index)
        if bcol in populated.columns:
            bvals = populated[bcol]
            is_offset = bvals.eq(True) | bvals.astype(str).str.lower().eq("yes")
        else:
            is_offset = pd.Series(False, index=populated.index)
        valid_idx = populated.index[has_geom & ~is_offset]

        # Batch-compute bearings for all valid geometries at once
        _valid_geoms    = geom_series.loc[valid_idx]
        _bearing_series = _linestring_bearings_vectorized(_valid_geoms)
        _has_bearing    = _bearing_series.notna()
        all_sep_sw_geoms.extend(list(_valid_geoms[_has_bearing]))
        all_sep_sw_bearings.extend(_bearing_series[_has_bearing].tolist())
        all_sep_sw_row_indices.extend(_valid_geoms[_has_bearing].index.tolist())

    sep_sw_tree = STRtree(all_sep_sw_geoms) if all_sep_sw_geoms else None
    total_offset = 0
    total_skipped = 0

    # ── Antiparallel deduplication ────────────────────────────────────────────
    # For two-way streets osmnx creates both the (u→v) and (v→u) directed edges.
    # Both inherit the same sidewalk tags, so both become offset candidates.
    # Offsetting both produces duplicate offset geometries (A.left = B.right,
    # A.right = B.left).  Pre-mark the later-indexed reversed edge for
    # suppression so only the first-seen direction is offset.
    _antiparallel_suppressed: set = set()
    if ("start_node_id" in populated.columns
            and "end_node_id" in populated.columns
            and "street_id" in populated.columns):
        _seen_dir: dict[tuple, Any] = {}
        _ap_u_arr   = populated["start_node_id"].values
        _ap_v_arr   = populated["end_node_id"].values
        _ap_sid_arr = populated["street_id"].values
        for _ap_i, _ap_idx in enumerate(populated.index):
            _u   = _ap_u_arr[_ap_i]
            _v   = _ap_v_arr[_ap_i]
            _sid = _ap_sid_arr[_ap_i]
            if pd.isna(_u) or pd.isna(_v):
                continue
            _u = cast(int, int(_u))
            _v = cast(int, int(_v))
            _sid_key = tuple(_sid) if isinstance(_sid, list) else _sid
            _fwd = (_sid_key, _u, _v)
            _rev = (_sid_key, _v, _u)
            if _rev in _seen_dir:
                _antiparallel_suppressed.add(_ap_idx)
            else:
                _seen_dir[_fwd] = _ap_idx
    print(f"Antiparallel suppression: {len(_antiparallel_suppressed)} reverse-direction edges marked.")

    # Pre-create all facility geometry and offset columns at once
    _pre_facility_cols = {}
    for _kind, _side, _slot in _FACILITY_SLOTS:
        _sub = f"{_kind}_{_side}_{_slot}" if _slot else f"{_kind}_{_side}"
        for _col in (f"{_sub}_geometry", f"{_sub}_offset"):
            if _col not in populated.columns:
                _pre_facility_cols[_col] = pd.NA
    if _pre_facility_cols:
        populated = _add_cols(populated, _pre_facility_cols)

    # Debug counters for _offset_segment failure modes
    _offset_debug = {"n_empty_offset": 0, "collision_counts": {}}

    for kind, side, slot in _FACILITY_SLOTS:
        sub_id   = f"{kind}_{side}_{slot}" if slot else f"{kind}_{side}"
        geom_col = f"{sub_id}_geometry"
        offset_col = f"{sub_id}_offset"
        data_col = f"{sub_id}_type" if kind == "bikeway" else f"{sub_id}_presence"

        if data_col not in populated.columns:
            continue

        if kind == "sidewalk":
            # "right" stored in the left slot (or "left" in the right slot) means the
            # bare OSM tag `sidewalk=right/left` propagated via coalesce fallback — the
            # opposite-side value indicates no sidewalk on this side.
            opposite_side = "right" if side == "left" else "left"
            skip_values = _NEGATIVE_VALUES | _SEPARATE_PRESENCE_VALUES | {opposite_side}
        else:
            skip_values = _NEGATIVE_VALUES
        has_data = populated[data_col].notna() & ~populated[data_col].astype(str).str.lower().isin(skip_values)
        no_geom  = ~pd.Series(np.asarray(shapely.is_geometry(populated[geom_col].values), dtype=bool),
                               index=populated.index)
        candidates = has_data & no_geom
        print(f"  [{sub_id}] candidates: {candidates.sum()}, has_data: {has_data.sum()}, no_geom: {no_geom.sum()}")
        if not candidates.any():
            continue

        n_no_street_geom = 0
        n_suppressed_here = 0
        n_suppressed_intersection = 0
        n_offset_called = 0
        n_offset_wrote = 0
        for idx in populated.index[candidates]:
            street_geom = populated.at[idx, "street_geometry"]
            if street_geom is None or not hasattr(street_geom, "geom_type"):
                n_no_street_geom += 1
                continue
            street_geom = cast(BaseGeometry, street_geom)

            # Intersection approach suppression: segments with at least one
            # intersection endpoint that are shorter than the road width
            # (lanes × lane_width) represent road surface within the
            # intersection itself — there is no physical sidewalk alongside
            # them.  Skip offset entirely.
            start_is_inter = bool(populated.at[idx, "start_node_is_intersection_node"])
            end_is_inter   = bool(populated.at[idx, "end_node_is_intersection_node"])
            if start_is_inter or end_is_inter:
                raw_lanes = populated.at[idx, "lanes"]
                raw_lw    = populated.at[idx, "lane_width"]
                seg_lanes = _parse_numeric(raw_lanes, 2.0)
                seg_lw    = _parse_numeric(raw_lw, default_lane_width_m)
                road_width = seg_lanes * seg_lw
                if street_geom.length < road_width:
                    n_suppressed_intersection += 1
                    continue

            # Roundabout suppression: roundabout segments get no offset
            # geometry at all (neither side).  The real sidewalks are on the
            # approaching streets, which snap to each other at each corner.
            if "junction" in populated.columns:
                junction_val = populated.at[idx, "junction"]
                if isinstance(junction_val, str) and junction_val.lower() == "roundabout":
                    n_suppressed_intersection += 1
                    continue

            # Antiparallel suppression: skip the reverse-direction duplicate
            # of a two-way street edge (A.left = B.right, A.right = B.left).
            if idx in _antiparallel_suppressed:
                n_suppressed_intersection += 1
                continue

            # Sidewalk suppression: skip if a nearby parallel separate sidewalk
            # exists on the same side.  Only guard is same-row (a row's own
            # separate geometry can't cause false suppression since it already
            # has geometry and therefore isn't an offset candidate).  This
            # allows separate sidewalks from adjacent segments of the SAME
            # street to suppress offset, preventing mixed separate/offset
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
                        # Use nearest-point distance instead of midpoint so
                        # that short approach segments near the end of a long
                        # separate sidewalk are still correctly suppressed.
                        sep_geom = all_sep_sw_geoms[hi]
                        if street_geom.distance(sep_geom) <= NEARBY_SEPARATE_SIDEWALK_THRESHOLD_M:
                            _, near_pt = nearest_points(street_geom, sep_geom)
                            if _road_side(street_geom, near_pt) == side:
                                skip = True
                                break
                    if skip:
                        total_skipped += 1
                        n_suppressed_here += 1
                        continue

            n_offset_called += 1
            populated = _offset_segment(idx, sub_id, street_geom, populated, side, _offset_debug, default_lane_width_m)
            geom_after = populated.at[idx, geom_col]
            if geom_after is not None and hasattr(geom_after, "geom_type"):
                n_offset_wrote += 1
                total_offset += 1

        print(f"    no_street_geom={n_no_street_geom}, suppressed={n_suppressed_here}, "
              f"intersection_skip={n_suppressed_intersection}, "
              f"offset_called={n_offset_called}, offset_wrote={n_offset_wrote}")

    print(f"Offset pass: {total_offset} facility segments offset from centerline"
          f" ({total_skipped} sidewalk slots skipped — nearby separate sidewalk exists).")
    print(f"  _offset_segment failures: empty_offset={_offset_debug['n_empty_offset']}, "
          f"collisions={_offset_debug['collision_counts']}")
    if "empty_offset_detail" in _offset_debug:
        print(f"  empty_offset breakdown: {_offset_debug['empty_offset_detail']}")
    if "nan_source_counts" in _offset_debug:
        print(f"  NaN source breakdown: {_offset_debug['nan_source_counts']}")
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


def _linestring_bearings_vectorized(geom_series: Any) -> pd.Series:  # type: ignore[type-arg]
    """Vectorized bearing computation for a GeoSeries of LineStrings.

    Uses shapely C-level array ops for the common LineString case, with a
    Python fallback for MultiLineStrings and edge cases.
    """
    geom_arr = np.asarray(geom_series)
    n = len(geom_arr)
    bearings = np.full(n, np.nan, dtype=np.float64)

    # Fast path: shapely vectorized ops extract first/last points of LineStrings.
    # Returns None (→ NaN coords) for MultiLineString and missing geometries.
    first_pts = shapely.get_point(geom_arr, 0)
    last_pts = shapely.get_point(geom_arr, -1)

    x0 = shapely.get_x(first_pts)
    y0 = shapely.get_y(first_pts)
    x1 = shapely.get_x(last_pts)
    y1 = shapely.get_y(last_pts)

    dx = x1 - x0
    dy = y1 - y0
    valid = np.isfinite(x0) & np.isfinite(x1) & ((dx != 0) | (dy != 0))
    bearings[valid] = np.degrees(np.arctan2(dx[valid], dy[valid])) % 360

    # Slow fallback for MultiLineStrings and other edge cases
    for i in np.flatnonzero(np.isnan(bearings)):
        g = geom_arr[i]
        if g is not None and isinstance(g, BaseGeometry):
            b = _linestring_bearing(g)
            if b is not None:
                bearings[i] = b

    return pd.Series(bearings, index=geom_series.index)


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
            crow = np.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
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
    clears the geometry and marks ``offset=True`` so the offset pass knows
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

    # Confirmed centerline-coincident: clear geometry and mark row for offset
    geom_col = f"{sub_facility_id}_geometry"
    populated.at[road_idx, geom_col] = None

    offset_col = f"{sub_facility_id}_offset"
    populated.at[road_idx, offset_col] = True

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


def _offset_segment(
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
    4. Writes the geometry and sets the ``*_offset`` flag.

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
            if not _is_na(w):
                bike_width += _parse_numeric(w, 0.0)
            elif not _is_na(row.get(f"bikeway_{side}_{slot}_type")):
                bike_width += _DEFAULT_BIKE_WIDTH_M
        offset_m = half_road + bike_width

    else:
        return populated  # unknown facility kind — nothing to do

    # --- 4. Generate the parallel-offset geometry -----------------------------
    # Use offset_curve (Shapely ≥ 2.0): positive = left, negative = right.
    # If the full offset fails (empty/degenerate), try progressively smaller
    # offsets down to 25% of the original distance.
    sign = 1 if side == "left" else -1
    offset_geom: BaseGeometry | None = None
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
                offset_geom = candidate
                break
        except Exception:
            continue
    if offset_geom is None:
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
    endpoint_buffer = offset_geom.boundary.buffer(1e-6)

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
        if not _mid_intersection_clear(offset_geom, other_geom):
            debug["collision_counts"][col] = debug["collision_counts"].get(col, 0) + 1
            return populated

    # Network-level check removed: offset facilities naturally cross
    # perpendicular streets at intersections in grid networks.
    # Collision checking is row-scoped only (per spec step 4).

    geom_col = own_geom_col
    populated.at[street_id, geom_col] = offset_geom  # type: ignore[index]

    # --- 6. Mark the slot as offset -------------------------------------------
    offset_col = f"{sub_facility_id}_offset"
    populated.at[street_id, offset_col] = True

    return populated


def _is_na(val) -> bool:
    """Return True if *val* is pandas/numpy NA or None."""
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


_ENDPOINT_SNAP_APPROACH_M = 5.0    # use last N metres of line to compute approach bearing
_ENDPOINT_SNAP_MAX_EXTEND_M = 12.0 # max extension length for line-intersection snapping
_ENDPOINT_SNAP_CORNER_ANGLE = 60.0 # max angular spread (degrees) for endpoints in one corner


def _snap_offset_endpoints(
    populated: gpd.GeoDataFrame,
    default_lane_width_m: float = _DEFAULT_LANE_WIDTH_M,
) -> gpd.GeoDataFrame:
    """Snap offset sidewalk endpoints at intersection nodes.

    Corner-grouping algorithm:

    1. At each intersection node, collect all sidewalk endpoints with their
       positions, tangent bearings, and angle from the node center.
    2. Group endpoints into **corners** by angular proximity (sorted by
       angle, consecutive endpoints within ``_ENDPOINT_SNAP_CORNER_ANGLE``
       degrees).
    3. For each corner with 2+ endpoints from different segments:
       a. Find all non-parallel ray-ray intersections among the endpoints.
       b. Average the valid intersection points to get the corner point.
       c. If no valid intersections (all parallel), use the centroid.
       d. Move every endpoint in the corner to the corner point.
    """
    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tangent_bearing(coords: list[tuple[float, float]], end: str) -> float:
        """Bearing the sidewalk line is *heading* at the given endpoint.

        For ``end="start"``: bearing from interior toward coords[0].
        For ``end="end"``: bearing from interior toward coords[-1].
        Uses the last ``_ENDPOINT_SNAP_APPROACH_M`` metres of the line.
        """
        if end == "start":
            total = 0.0
            prev = coords[0]
            interior = coords[min(1, len(coords) - 1)]
            for i in range(1, len(coords)):
                cx, cy = coords[i]
                dx, dy = cx - prev[0], cy - prev[1]
                total += math.sqrt(dx * dx + dy * dy)
                interior = coords[i]
                prev = coords[i]
                if total >= _ENDPOINT_SNAP_APPROACH_M:
                    break
            ddx = coords[0][0] - interior[0]
            ddy = coords[0][1] - interior[1]
        else:
            total = 0.0
            prev = coords[-1]
            interior = coords[max(-2, -len(coords))]
            for i in range(len(coords) - 2, -1, -1):
                cx, cy = coords[i]
                dx, dy = cx - prev[0], cy - prev[1]
                total += math.sqrt(dx * dx + dy * dy)
                interior = coords[i]
                prev = coords[i]
                if total >= _ENDPOINT_SNAP_APPROACH_M:
                    break
            ddx = coords[-1][0] - interior[0]
            ddy = coords[-1][1] - interior[1]

        if ddx == 0 and ddy == 0:
            return 0.0
        return math.degrees(math.atan2(ddx, ddy)) % 360

    def _ray_intersect(
        p1: tuple[float, float], bearing1: float,
        p2: tuple[float, float], bearing2: float,
        max_dist: float,
    ) -> tuple[float, float] | None:
        """Intersect two forward rays.  Returns None if parallel, behind, or too far."""
        r1, r2 = math.radians(bearing1), math.radians(bearing2)
        dx1, dy1 = math.sin(r1), math.cos(r1)
        dx2, dy2 = math.sin(r2), math.cos(r2)

        det = dx1 * dy2 - dy1 * dx2
        if abs(det) < 1e-10:
            return None  # parallel / antiparallel

        dpx, dpy = p2[0] - p1[0], p2[1] - p1[1]
        t1 = (dpx * dy2 - dpy * dx2) / det
        t2 = (dpx * dy1 - dpy * dx1) / det

        if t1 < -0.5 or t2 < -0.5:
            return None  # meeting point behind one of the endpoints

        ix = p1[0] + t1 * dx1
        iy = p1[1] + t1 * dy1

        d1_sq = (ix - p1[0]) ** 2 + (iy - p1[1]) ** 2
        d2_sq = (ix - p2[0]) ** 2 + (iy - p2[1]) ** 2
        if d1_sq > max_dist * max_dist or d2_sq > max_dist * max_dist:
            return None

        return (ix, iy)

    def _line_intersect(
        p1: tuple[float, float], bearing1: float,
        p2: tuple[float, float], bearing2: float,
        max_dist: float,
    ) -> tuple[float, float] | None:
        """Intersect two infinite lines (no forward check).

        Used in Stage 2 where cross-product already ensures correct pairing.
        """
        r1, r2 = math.radians(bearing1), math.radians(bearing2)
        dx1, dy1 = math.sin(r1), math.cos(r1)
        dx2, dy2 = math.sin(r2), math.cos(r2)

        det = dx1 * dy2 - dy1 * dx2
        if abs(det) < 1e-10:
            return None

        dpx, dpy = p2[0] - p1[0], p2[1] - p1[1]
        t1 = (dpx * dy2 - dpy * dx2) / det

        ix = p1[0] + t1 * dx1
        iy = p1[1] + t1 * dy1

        d1_sq = (ix - p1[0]) ** 2 + (iy - p1[1]) ** 2
        d2_sq = (ix - p2[0]) ** 2 + (iy - p2[1]) ** 2
        if d1_sq > max_dist * max_dist or d2_sq > max_dist * max_dist:
            return None

        return (ix, iy)

    def _move_endpoint(
        geom: BaseGeometry, end: str, new_xy: tuple[float, float],
    ) -> BaseGeometry:
        """Return a copy of *geom* with the start or end coordinate replaced."""
        if isinstance(geom, MultiLineString):
            parts = [list(ls.coords) for ls in geom.geoms]
            if end == "start":
                parts[0][0] = new_xy
            else:
                parts[-1][-1] = new_xy
            return MultiLineString([LineString(p) for p in parts])
        coords = list(geom.coords)  # type: ignore[union-attr]
        if end == "start":
            coords[0] = new_xy
        else:
            coords[-1] = new_xy
        return LineString(coords)

    def _to_linestring(geom: BaseGeometry) -> "LineString | None":
        """Return a LineString equivalent, merging MultiLineString if possible."""
        if isinstance(geom, LineString):
            return geom
        if isinstance(geom, MultiLineString):
            merged = _sw_linemerge(geom)
            return merged if isinstance(merged, LineString) else None
        return None

    # ── Build intersection-node -> segment mapping ────────────────────────
    # Scan ALL start/end nodes (no is_intersection flag filter).  Only nodes
    # shared by 2+ distinct segment IDs are kept — this captures true
    # intersections AND T-junction bases where OSM omits the flag.

    node_to_segs: dict[tuple[float, float], list[tuple[int, str]]] = defaultdict(list)
    node_key_to_pt: dict[tuple[float, float], Point] = {}

    # Identify roundabout segments early so they can be excluded from node
    # clustering.  Roundabout arcs chain adjacent entry nodes together, and
    # including them in the proximity clustering causes ALL roundabout entries
    # to merge into one super-node, producing wrong centroid snaps.
    _roundabout_idxs: set[int] = set()
    if "junction" in populated.columns:
        _junc = populated["junction"]
        _is_ra = _junc.notna() & _junc.astype(str).str.lower().eq("roundabout")
        _roundabout_idxs = set(int(i) for i in populated.index[_is_ra])

    # _all_candidates: node_key -> [(seg_idx, position)]
    _all_candidates: dict[tuple[float, float], list[tuple[int, str]]] = defaultdict(list)

    for node_col, position in (
        ("start_node_geometry", "start"),
        ("end_node_geometry",   "end"),
    ):
        if node_col not in populated.columns:
            continue
        idx_arr  = populated.index.to_numpy()
        geom_arr = populated[node_col].to_numpy(dtype=object)

        # Vectorised coordinate extraction (Shapely 2.x)
        valid_mask = np.asarray(
            shapely.is_geometry(geom_arr) & ~shapely.is_empty(geom_arr), dtype=bool
        )
        valid_geoms = geom_arr[valid_mask]
        valid_idx   = idx_arr[valid_mask]
        if len(valid_geoms) == 0:
            continue

        xs = np.round(shapely.get_x(valid_geoms), 1)
        ys = np.round(shapely.get_y(valid_geoms), 1)

        for seg_idx, x, y, geom in zip(valid_idx, xs, ys, valid_geoms):
            if int(seg_idx) in _roundabout_idxs:
                continue
            key: tuple[float, float] = (float(x), float(y))
            _all_candidates[key].append((int(seg_idx), position))
            if key not in node_key_to_pt:
                node_key_to_pt[key] = cast(Point, geom)

    for key, entries in _all_candidates.items():
        distinct = {idx for idx, _ in entries}
        if len(distinct) >= 2:
            node_to_segs[key] = entries

    # ── Proximity node clustering ──────────────────────────────────────────
    # Cluster ALL nodes (including degree-1) by proximity so that nearby OSM
    # nodes that collectively serve 2+ distinct segments are merged into a
    # single virtual node.  Degree-1 nodes filtered out of node_to_segs are
    # intentionally included here so that split T-junction bases (like the
    # Delaware/Bonita case where Node B has only one segment) are merged with
    # their neighbour.  The "2+ distinct segments" filter is applied after
    # merging.  The actual node geometry data in `populated` is NOT modified.

    _SNAP_NODE_CLUSTER_M = 7.5

    # Collect rounded-key positions of ALL roundabout segment nodes so we can
    # prevent adjacent roundabout entry nodes from clustering together.
    _ra_node_keys: set[tuple[float, float]] = set()
    for _ra_idx in _roundabout_idxs:
        for _nc in ("start_node_geometry", "end_node_geometry"):
            if _nc not in populated.columns:
                continue
            _rg = populated.at[_ra_idx, _nc]
            if isinstance(_rg, BaseGeometry) and not _rg.is_empty and hasattr(_rg, "x"):
                _ra_node_keys.add((round(_rg.x, 1), round(_rg.y, 1)))  # type: ignore[union-attr]

    # Use all nodes (node_key_to_pt), not just multi-segment nodes (node_to_segs)
    _all_nk_list = list(node_key_to_pt.keys())
    _stage_nodes: list[tuple[Point, list[tuple[int, str]]]] = []

    if _all_nk_list:
        _nd_geoms = np.array([node_key_to_pt[k] for k in _all_nk_list], dtype=object)
        _nd_tree  = STRtree(_nd_geoms)

        # Vectorised: find all (i, j) pairs with i < j and distance ≤ threshold
        _q_arr, _t_arr = _nd_tree.query(_nd_geoms, predicate="dwithin",
                                         distance=_SNAP_NODE_CLUSTER_M)
        _pair_mask = _q_arr < _t_arr
        _close_pairs = list(zip(_q_arr[_pair_mask].tolist(),
                                _t_arr[_pair_mask].tolist()))

        # Union-find clustering
        _uf: list[int] = list(range(len(_all_nk_list)))

        def _uf_find(i: int) -> int:
            while _uf[i] != i:
                _uf[i] = _uf[_uf[i]]
                i = _uf[i]
            return i

        for _i, _j in _close_pairs:
            # Don't merge two roundabout-entry nodes: each entry should stay
            # separate so its approaching street is snapped independently.
            if (_all_nk_list[_i] in _ra_node_keys
                    and _all_nk_list[_j] in _ra_node_keys):
                continue
            _ri, _rj = _uf_find(_i), _uf_find(_j)
            if _ri != _rj:
                _uf[_ri] = _rj

        _clusters_d: dict[int, list[int]] = defaultdict(list)
        for _i in range(len(_all_nk_list)):
            _clusters_d[_uf_find(_i)].append(_i)

        n_virtual = 0
        for members in _clusters_d.values():
            # Gather segments from _all_candidates (includes degree-1 nodes)
            _all_segs: list[tuple[int, str]] = []
            _seen_segs: set[tuple[int, str]] = set()
            _cx = _cy = 0.0
            for _m in members:
                _k = _all_nk_list[_m]
                _cx += _k[0]; _cy += _k[1]
                for _entry in _all_candidates[_k]:
                    if _entry not in _seen_segs:
                        _all_segs.append(_entry)
                        _seen_segs.add(_entry)
            # Only process virtual nodes with 2+ distinct segment IDs
            if len({idx for idx, _ in _all_segs}) < 2:
                continue
            _cx /= len(members); _cy /= len(members)
            if len(members) == 1:
                _k = _all_nk_list[members[0]]
                _stage_nodes.append((node_key_to_pt[_k], _all_segs))
            else:
                _stage_nodes.append((Point(_cx, _cy), _all_segs))
                n_virtual += 1

        print(f"  Node clustering: {n_virtual} virtual merged nodes "
              f"(threshold={_SNAP_NODE_CLUSTER_M} m, total={len(_stage_nodes)})")
    # end proximity clustering

    # ── Pre-extract sidewalk geometry coords for fast access ──────────────

    sw_coords_cache: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for side in ("left", "right"):
        geom_col = f"sidewalk_{side}_geometry"
        if geom_col not in populated.columns:
            continue
        geom_arr = populated[geom_col].to_numpy(dtype=object)
        for pos, g in enumerate(geom_arr):
            if isinstance(g, BaseGeometry) and not g.is_empty:
                sw_coords_cache[(populated.index[pos], side)] = _flatten_coords(g)

    n_stage1 = 0
    n_stage2 = 0
    n_stage2a = 0
    n_endpoints_moved = 0

    # Track Stage 1 group membership: endpoint_key -> group_id
    # and group_id -> [endpoint_keys]
    ep_to_group: dict[tuple[int, str, str], int] = {}
    group_members: dict[int, list[tuple[int, str, str]]] = {}
    next_group_id = 0

    # ── Stage 1: Angular centroid snap (same-street continuations) ────────

    for node_pt, seg_entries in _stage_nodes:
        if len(seg_entries) < 2:
            continue
        nx, ny = node_pt.x, node_pt.y

        eps: list[tuple[int, str, str, tuple[float, float], float]] = []
        for seg_idx, seg_end in seg_entries:
            for side in ("left", "right"):
                cache_key = (seg_idx, side)
                if cache_key not in sw_coords_cache:
                    continue
                coords = sw_coords_cache[cache_key]
                if len(coords) < 2:
                    continue
                ep = coords[0] if seg_end == "start" else coords[-1]
                angle = math.degrees(math.atan2(ep[0] - nx, ep[1] - ny)) % 360
                eps.append((seg_idx, seg_end, side, ep, angle))

        if len(eps) < 2:
            continue

        eps.sort(key=lambda e: e[4])

        # Group into angular clusters
        corners: list[list[int]] = []
        current: list[int] = [0]
        for k in range(1, len(eps)):
            if eps[k][4] - eps[current[0]][4] <= _ENDPOINT_SNAP_CORNER_ANGLE:
                current.append(k)
            else:
                corners.append(current)
                current = [k]
        corners.append(current)

        if len(corners) > 1:
            first_ang = eps[corners[0][0]][4]
            last_ang = eps[corners[-1][-1]][4]
            if (360 - last_ang) + first_ang <= _ENDPOINT_SNAP_CORNER_ANGLE:
                corners[-1].extend(corners[0])
                corners.pop(0)

        for corner_indices in corners:
            if len(corner_indices) < 2:
                continue
            corner_eps = [eps[k] for k in corner_indices]
            seg_ids = {e[0] for e in corner_eps}
            if len(seg_ids) < 2:
                continue

            cx = sum(e[3][0] for e in corner_eps) / len(corner_eps)
            cy = sum(e[3][1] for e in corner_eps) / len(corner_eps)
            centroid = (cx, cy)

            # Check all within max extend
            _snap_max_sq = _ENDPOINT_SNAP_MAX_EXTEND_M * _ENDPOINT_SNAP_MAX_EXTEND_M
            if any((centroid[0] - e[3][0]) ** 2 + (centroid[1] - e[3][1]) ** 2
                   > _snap_max_sq for e in corner_eps):
                continue

            gid = next_group_id
            next_group_id += 1
            group_members[gid] = []

            for seg_idx, seg_end, side, ep_xy, _ in corner_eps:
                ep_key = (seg_idx, side, seg_end)
                prev_gid = ep_to_group.get(ep_key)
                if prev_gid is not None and prev_gid != gid:
                    # Already assigned by an earlier node — don't overwrite,
                    # otherwise Stage 2 group-member propagation breaks.
                    continue
                ep_to_group[ep_key] = gid
                group_members[gid].append(ep_key)

                if seg_idx in _roundabout_idxs:
                    continue
                _dsq = (centroid[0] - ep_xy[0]) ** 2 + (centroid[1] - ep_xy[1]) ** 2
                if _dsq < 0.0001:  # 0.01² = 0.0001
                    continue
                geom_col = f"sidewalk_{side}_geometry"
                geom = populated.at[seg_idx, geom_col]
                if not isinstance(geom, BaseGeometry) or geom.is_empty:
                    continue
                populated.at[seg_idx, geom_col] = _move_endpoint(geom, seg_end, centroid)  # type: ignore[index]
                cache_key = (seg_idx, side)
                if cache_key in sw_coords_cache:
                    c = list(sw_coords_cache[cache_key])
                    if seg_end == "start":
                        c[0] = centroid
                    else:
                        c[-1] = centroid
                    sw_coords_cache[cache_key] = c
                n_endpoints_moved += 1

            n_stage1 += 1

    # ── Stage 2: Cross-street corner extension ────────────────────────────
    # When a primary endpoint is moved, also move its Stage 1 group members.

    for node_pt, seg_entries in _stage_nodes:
        if len(seg_entries) < 2:
            continue
        nx, ny = node_pt.x, node_pt.y

        seg_dirs: list[tuple[int, str, float, float, float]] = []
        for seg_idx, seg_end in seg_entries:
            if seg_idx in _roundabout_idxs:
                continue
            street_geom = populated.at[seg_idx, "street_geometry"]
            if not isinstance(street_geom, BaseGeometry) or street_geom.is_empty:
                continue
            coords = _flatten_coords(street_geom)
            if len(coords) < 2:
                continue
            if seg_end == "start":
                odx = coords[min(1, len(coords) - 1)][0] - coords[0][0]
                ody = coords[min(1, len(coords) - 1)][1] - coords[0][1]
            else:
                odx = coords[max(-2, -len(coords))][0] - coords[-1][0]
                ody = coords[max(-2, -len(coords))][1] - coords[-1][1]
            length = math.sqrt(odx * odx + ody * ody)
            if length < 0.001:
                continue
            odx /= length
            ody /= length
            bearing = math.degrees(math.atan2(odx, ody)) % 360
            seg_dirs.append((seg_idx, seg_end, odx, ody, bearing))

        # Deduplicate antiparallel edges (parallel or ~180 apart).
        # Track ALL antiparallel pairs for Stage 2a by scanning all pairs of
        # seg_dirs directly (O(n²) but n is small per node).
        unique_dirs: list[tuple[int, str, float, float, float]] = []
        antiparallel_pairs: list[
            tuple[
                tuple[int, str, float, float, float],
                tuple[int, str, float, float, float],
            ]
        ] = []
        seen_ap_pairs: set[tuple[int, int]] = set()
        for entry in seg_dirs:
            is_dup = False
            for existing in unique_dirs:
                diff = abs(entry[4] - existing[4]) % 360
                if diff > 180:
                    diff = 360 - diff
                if diff < 20:          # parallel — deduplicate, no snap needed
                    is_dup = True
                    break
                if (180 - diff) < 20:  # antiparallel — deduplicate
                    is_dup = True
                    break
            if not is_dup:
                unique_dirs.append(entry)
        # Find all antiparallel pairs across seg_dirs (avoids missed pairs when
        # 3+ segments share a direction cluster, e.g. two eastbound + one westbound)
        for _pi in range(len(seg_dirs)):
            for _pj in range(_pi + 1, len(seg_dirs)):
                _ei, _ej = seg_dirs[_pi], seg_dirs[_pj]
                if _ei[0] == _ej[0]:  # same segment index
                    continue
                _pair_key = (min(_ei[0], _ej[0]), max(_ei[0], _ej[0]))
                if _pair_key in seen_ap_pairs:
                    continue
                _diff = abs(_ei[4] - _ej[4]) % 360
                if _diff > 180:
                    _diff = 360 - _diff
                if (180 - _diff) < 20:
                    antiparallel_pairs.append((_ei, _ej))
                    seen_ap_pairs.add(_pair_key)

        if len(unique_dirs) < 2:
            continue

        unique_dirs.sort(key=lambda x: x[4])

        n_streets = len(unique_dirs)
        for k in range(n_streets):
            idx_a, end_a, dx_a, dy_a, _ = unique_dirs[k]
            idx_b, end_b, dx_b, dy_b, _ = unique_dirs[(k + 1) % n_streets]
            if idx_a == idx_b:
                continue

            # Find physical-right of A and physical-left of B via cross product
            right_a: tuple[str, tuple[float, float], float] | None = None
            for side in ("left", "right"):
                ck = (idx_a, side)
                if ck not in sw_coords_cache:
                    continue
                cds = sw_coords_cache[ck]
                if len(cds) < 2:
                    continue
                ep = cds[0] if end_a == "start" else cds[-1]
                cross_val = dx_a * (ep[1] - ny) - dy_a * (ep[0] - nx)
                if cross_val < 0:
                    right_a = (side, ep, _tangent_bearing(cds, end_a))

            left_b: tuple[str, tuple[float, float], float] | None = None
            for side in ("left", "right"):
                ck = (idx_b, side)
                if ck not in sw_coords_cache:
                    continue
                cds = sw_coords_cache[ck]
                if len(cds) < 2:
                    continue
                ep = cds[0] if end_b == "start" else cds[-1]
                cross_val = dx_b * (ep[1] - ny) - dy_b * (ep[0] - nx)
                if cross_val > 0:
                    left_b = (side, ep, _tangent_bearing(cds, end_b))

            if right_a is None or left_b is None:
                continue

            side_a, ep_a, brg_a = right_a
            side_b, ep_b, brg_b = left_b

            dd = math.sqrt((ep_a[0] - ep_b[0]) ** 2 + (ep_a[1] - ep_b[1]) ** 2)
            if dd < 0.01 or dd > _ENDPOINT_SNAP_MAX_EXTEND_M:
                continue

            meet = _line_intersect(
                ep_a, brg_a, ep_b, brg_b,
                max_dist=_ENDPOINT_SNAP_MAX_EXTEND_M,
            )
            if meet is None:
                # Fallback: midpoint
                meet = ((ep_a[0] + ep_b[0]) / 2, (ep_a[1] + ep_b[1]) / 2)

            nd = math.sqrt((meet[0] - nx) ** 2 + (meet[1] - ny) ** 2)
            if nd > _ENDPOINT_SNAP_MAX_EXTEND_M * 2:
                continue

            # Collect all endpoints to move: the two primaries + their group members
            to_move: list[tuple[int, str, str]] = []
            for s_idx, s_end, s_side in (
                (idx_a, end_a, side_a),
                (idx_b, end_b, side_b),
            ):
                ep_key = (s_idx, s_side, s_end)
                to_move.append(ep_key)
                # Also add Stage 1 group members
                gid = ep_to_group.get(ep_key)
                if gid is not None:
                    for member in group_members[gid]:
                        if member not in to_move:
                            to_move.append(member)

            for s_idx, s_side, s_end in to_move:
                ck = (s_idx, s_side)
                if ck not in sw_coords_cache:
                    continue
                cds = sw_coords_cache[ck]
                cur_ep = cds[0] if s_end == "start" else cds[-1]
                d_move = math.sqrt((meet[0] - cur_ep[0]) ** 2 + (meet[1] - cur_ep[1]) ** 2)
                if d_move < 0.01:
                    continue
                if d_move > _ENDPOINT_SNAP_MAX_EXTEND_M:
                    continue
                geom_col = f"sidewalk_{s_side}_geometry"
                geom = populated.at[s_idx, geom_col]
                if not isinstance(geom, BaseGeometry) or geom.is_empty:
                    continue
                populated.at[s_idx, geom_col] = _move_endpoint(geom, s_end, meet)  # type: ignore[index]
                if ck in sw_coords_cache:
                    c = list(sw_coords_cache[ck])
                    if s_end == "start":
                        c[0] = meet
                    else:
                        c[-1] = meet
                    sw_coords_cache[ck] = c
                n_endpoints_moved += 1

            n_stage2 += 1

        # ── Stage 2a: Antiparallel outer-side snap ────────────────────────
        # For each antiparallel pair (canonical A, duplicate B), snap the two
        # same-physical-side endpoint pairs:
        #   • right_of_A  ↔  left_of_B  (outer side, e.g. north at a T-top)
        #   • left_of_A   ↔  right_of_B (inner side — usually already at 0 m)
        # The dd < 0.01 guard skips pairs already snapped by Stage 1/2.

        for _canon, _dup in antiparallel_pairs:
            _idx_a, _end_a, _dx_a, _dy_a, _ = _canon
            _idx_b, _end_b, _dx_b, _dy_b, _ = _dup

            if _idx_a == _idx_b:
                continue

            # Iterate over both cross-sign combinations:
            #   (a_sign=-1, b_sign=+1) → right_of_A + left_of_B
            #   (a_sign=+1, b_sign=-1) → left_of_A  + right_of_B
            for _a_sign, _b_sign in ((-1, +1), (+1, -1)):
                _ep_a_info: tuple[str, tuple[float, float], float] | None = None
                _ep_b_info: tuple[str, tuple[float, float], float] | None = None

                for _side in ("left", "right"):
                    _ck = (_idx_a, _side)
                    if _ck not in sw_coords_cache:
                        continue
                    _cds = sw_coords_cache[_ck]
                    if len(_cds) < 2:
                        continue
                    _ep = _cds[0] if _end_a == "start" else _cds[-1]
                    _cv = _dx_a * (_ep[1] - ny) - _dy_a * (_ep[0] - nx)
                    if (_a_sign < 0 and _cv < 0) or (_a_sign > 0 and _cv > 0):
                        _ep_a_info = (_side, _ep, _tangent_bearing(_cds, _end_a))

                for _side in ("left", "right"):
                    _ck = (_idx_b, _side)
                    if _ck not in sw_coords_cache:
                        continue
                    _cds = sw_coords_cache[_ck]
                    if len(_cds) < 2:
                        continue
                    _ep = _cds[0] if _end_b == "start" else _cds[-1]
                    _cv = _dx_b * (_ep[1] - ny) - _dy_b * (_ep[0] - nx)
                    if (_b_sign < 0 and _cv < 0) or (_b_sign > 0 and _cv > 0):
                        _ep_b_info = (_side, _ep, _tangent_bearing(_cds, _end_b))

                if _ep_a_info is None or _ep_b_info is None:
                    continue

                _side_a, _ep_a, _brg_a = _ep_a_info
                _side_b, _ep_b, _brg_b = _ep_b_info

                _dd = math.sqrt(
                    (_ep_a[0] - _ep_b[0]) ** 2 + (_ep_a[1] - _ep_b[1]) ** 2
                )
                if _dd < 0.01 or _dd > _ENDPOINT_SNAP_MAX_EXTEND_M:
                    continue

                _meet = _line_intersect(
                    _ep_a, _brg_a, _ep_b, _brg_b,
                    max_dist=_ENDPOINT_SNAP_MAX_EXTEND_M,
                )
                if _meet is None:
                    _meet = (
                        (_ep_a[0] + _ep_b[0]) / 2,
                        (_ep_a[1] + _ep_b[1]) / 2,
                    )

                _nd = math.sqrt((_meet[0] - nx) ** 2 + (_meet[1] - ny) ** 2)
                if _nd > _ENDPOINT_SNAP_MAX_EXTEND_M * 2:
                    continue

                _to_move: list[tuple[int, str, str]] = []
                for _s_idx, _s_end, _s_side in (
                    (_idx_a, _end_a, _side_a),
                    (_idx_b, _end_b, _side_b),
                ):
                    _ep_key = (_s_idx, _s_side, _s_end)
                    _to_move.append(_ep_key)
                    _gid = ep_to_group.get(_ep_key)
                    if _gid is not None:
                        for _member in group_members[_gid]:
                            if _member not in _to_move:
                                _to_move.append(_member)

                for _s_idx, _s_side, _s_end in _to_move:
                    _ck = (_s_idx, _s_side)
                    if _ck not in sw_coords_cache:
                        continue
                    _cds2 = sw_coords_cache[_ck]
                    _cur = _cds2[0] if _s_end == "start" else _cds2[-1]
                    _dm = math.sqrt(
                        (_meet[0] - _cur[0]) ** 2 + (_meet[1] - _cur[1]) ** 2
                    )
                    if _dm < 0.01 or _dm > _ENDPOINT_SNAP_MAX_EXTEND_M:
                        continue
                    _gcol = f"sidewalk_{_s_side}_geometry"
                    _geom = populated.at[_s_idx, _gcol]
                    if not isinstance(_geom, BaseGeometry) or _geom.is_empty:
                        continue
                    populated.at[_s_idx, _gcol] = _move_endpoint(  # type: ignore[index]
                        _geom, _s_end, _meet
                    )
                    _cl = list(sw_coords_cache[_ck])
                    if _s_end == "start":
                        _cl[0] = _meet
                    else:
                        _cl[-1] = _meet
                    sw_coords_cache[_ck] = _cl
                    n_endpoints_moved += 1

                n_stage2a += 1

    # ── Stage 2c: Singleton corner pairing ────────────────────────────────
    # Stage 1 angular grouping uses a tight threshold (_ENDPOINT_SNAP_CORNER_ANGLE).
    # At intersections where two perpendicular streets are both offset sidewalks,
    # the two endpoints that should meet at a corner can land ~90° apart from the
    # node (one due east, one due north), exceeding the threshold.  Stage 2 misses
    # them if neither segment appears as a unique_dirs representative.
    #
    # This stage collects, per node, all endpoints NOT yet placed into a Stage 1
    # group (ep_to_group), sorts them by angle, and pairs adjacent singletons from
    # different segments via the same line-intersection logic as Stage 2.

    n_stage2c = 0
    _s2c_processed: set[frozenset[tuple[int, str, str]]] = set()

    for node_pt, seg_entries in _stage_nodes:
        nx, ny = node_pt.x, node_pt.y

        singletons: list[tuple[int, str, str, tuple[float, float], float, float]] = []
        for seg_idx, seg_end in seg_entries:
            if seg_idx in _roundabout_idxs:
                continue
            for side in ("left", "right"):
                ck = (seg_idx, side)
                if ck not in sw_coords_cache:
                    continue
                coords = sw_coords_cache[ck]
                if len(coords) < 2:
                    continue
                ep_key = (seg_idx, side, seg_end)
                if ep_key in ep_to_group:
                    continue  # already handled by Stage 1/2
                ep = coords[0] if seg_end == "start" else coords[-1]
                angle = math.degrees(math.atan2(ep[0] - nx, ep[1] - ny)) % 360
                brg = _tangent_bearing(coords, seg_end)
                singletons.append((seg_idx, seg_end, side, ep, angle, brg))

        if len(singletons) < 2:
            continue

        singletons.sort(key=lambda e: e[4])
        n_s = len(singletons)

        for k in range(n_s):
            idx_a, end_a, side_a, pt_a, angle_a, brg_a = singletons[k]
            idx_b, end_b, side_b, pt_b, angle_b, brg_b = singletons[(k + 1) % n_s]

            if idx_a == idx_b:
                continue

            pair_key: frozenset[tuple[int, str, str]] = frozenset(
                {(idx_a, side_a, end_a), (idx_b, side_b, end_b)}
            )
            if pair_key in _s2c_processed:
                continue
            _s2c_processed.add(pair_key)

            dd = math.sqrt((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2)
            if dd < 0.01 or dd > _ENDPOINT_SNAP_MAX_EXTEND_M:
                continue

            meet = _line_intersect(
                pt_a, brg_a, pt_b, brg_b,
                max_dist=_ENDPOINT_SNAP_MAX_EXTEND_M,
            )
            if meet is None:
                meet = ((pt_a[0] + pt_b[0]) / 2, (pt_a[1] + pt_b[1]) / 2)

            nd = math.sqrt((meet[0] - nx) ** 2 + (meet[1] - ny) ** 2)
            if nd > _ENDPOINT_SNAP_MAX_EXTEND_M * 2:
                continue

            for s_idx, s_side, s_end in ((idx_a, side_a, end_a), (idx_b, side_b, end_b)):
                ck = (s_idx, s_side)
                if ck not in sw_coords_cache:
                    continue
                cds = sw_coords_cache[ck]
                cur_ep = cds[0] if s_end == "start" else cds[-1]
                d_move = math.sqrt(
                    (meet[0] - cur_ep[0]) ** 2 + (meet[1] - cur_ep[1]) ** 2
                )
                if d_move < 0.01 or d_move > _ENDPOINT_SNAP_MAX_EXTEND_M:
                    continue
                geom_col = f"sidewalk_{s_side}_geometry"
                geom = populated.at[s_idx, geom_col]
                if not isinstance(geom, BaseGeometry) or geom.is_empty:
                    continue
                populated.at[s_idx, geom_col] = _move_endpoint(  # type: ignore[index]
                    geom, s_end, meet
                )
                c = list(sw_coords_cache[ck])
                if s_end == "start":
                    c[0] = meet
                else:
                    c[-1] = meet
                sw_coords_cache[ck] = c
                n_endpoints_moved += 1

            n_stage2c += 1

    # ── Stage 3: Shapely-crosses trimming ─────────────────────────────────
    # Vectorised: build one STRtree of all intersection-endpoint sidewalk
    # geometries, bulk-query for crossing pairs, then trim each segment at
    # the crossing point.  Only pairs that share an intersection node are
    # processed.

    # Build flat list: one entry per (node_idx, seg_idx, seg_end, side)
    # node_idx is the index into _stage_nodes (unique per virtual merged node).
    s3_geoms:    list[BaseGeometry] = []  # raw geometry (may be Multi)
    s3_ls:       list[LineString | None] = []  # merged LineString (or None)
    s3_node_idx: list[int] = []           # index into _stage_nodes
    s3_seg_idx:  list[int] = []
    s3_seg_end:  list[str] = []
    s3_side:     list[str] = []

    for _ni, (_node_pt_s3, seg_entries) in enumerate(_stage_nodes):
        for seg_idx, seg_end in seg_entries:
            if seg_idx in _roundabout_idxs:
                continue
            for side in ("left", "right"):
                geom_col = f"sidewalk_{side}_geometry"
                if geom_col not in populated.columns:
                    continue
                if (seg_idx, side) not in sw_coords_cache:
                    continue
                geom = populated.at[seg_idx, geom_col]
                if not isinstance(geom, (LineString, MultiLineString)) or geom.is_empty:
                    continue
                s3_geoms.append(geom)
                s3_ls.append(_to_linestring(geom))
                s3_node_idx.append(_ni)
                s3_seg_idx.append(seg_idx)
                s3_seg_end.append(seg_end)
                s3_side.append(side)

    n_stage3 = 0
    trimmed_keys: set[tuple[int, str, str]] = set()  # (seg_idx, side, end)

    if s3_geoms:
        s3_tree = STRtree(s3_geoms)
        # Bulk query: returns (query_indices, tree_indices) for all crossing pairs
        q_idx, t_idx = s3_tree.query(s3_geoms, predicate="crosses")

        for qi, ti in zip(q_idx.tolist(), t_idx.tolist()):
            if ti <= qi:
                continue  # process each unordered pair once
            if s3_node_idx[qi] != s3_node_idx[ti]:
                continue  # different intersection nodes (or virtual merged nodes)
            if s3_seg_idx[qi] == s3_seg_idx[ti]:
                continue  # same street segment

            # Confirm crossing and get intersection point (bulk query may use
            # bbox overlap under the hood for some predicates)
            geom_q = s3_geoms[qi]
            geom_t = s3_geoms[ti]
            crossing_pt = geom_q.intersection(geom_t)
            if not isinstance(crossing_pt, Point):
                continue  # degenerate (overlap/multipoint)

            for arr_i in (qi, ti):
                ls = s3_ls[arr_i]
                if ls is None:
                    continue  # MultiLineString that couldn't be merged — skip
                seg_idx = s3_seg_idx[arr_i]
                seg_end = s3_seg_end[arr_i]
                side    = s3_side[arr_i]

                trim_key = (seg_idx, side, seg_end)
                if trim_key in trimmed_keys:
                    continue

                total_len = ls.length
                if total_len < 0.01:
                    continue

                dist_along = ls.project(crossing_pt)
                dist_from_ep = (
                    total_len - dist_along if seg_end == "end" else dist_along
                )
                if dist_from_ep > _ENDPOINT_SNAP_MAX_EXTEND_M:
                    continue  # crossing too far from the relevant endpoint

                if seg_end == "end":
                    if dist_along < 0.01:
                        continue
                    new_geom: BaseGeometry = _sw_substring(ls, 0.0, dist_along)
                else:
                    if dist_along > total_len - 0.01:
                        continue
                    new_geom = _sw_substring(ls, dist_along, total_len)

                if new_geom is None or new_geom.is_empty:
                    continue

                geom_col = f"sidewalk_{side}_geometry"
                populated.at[seg_idx, geom_col] = new_geom  # type: ignore[index]
                trimmed_keys.add(trim_key)
                sw_coords_cache[(seg_idx, side)] = _flatten_coords(new_geom)
                # Keep s3_ls in sync so a subsequent pair sees the trimmed geom
                s3_ls[arr_i] = new_geom if isinstance(new_geom, LineString) else None
                n_endpoints_moved += 1

            n_stage3 += 1

    # ── Stage 2b: Backward-probe crosses trimming ─────────────────────────
    # Runs AFTER Stage 3 so that endpoints moved by crosses-trimming are in
    # their final positions when probes are built.  This lets Stage 2b detect
    # cases where a moved endpoint now lies on another segment's body.
    #
    # For each endpoint P on segment A (side s, end e):
    #   1. Build a short probe extending OUTWARD from P (backward, away from
    #      the segment interior) by _ENDPOINT_SNAP_MAX_EXTEND_M, plus a
    #      small extra buffer so the probe endpoint is past any nearby segment.
    #   2. Use STRtree + predicate="crosses" to find sidewalk segments B that
    #      the probe crosses.
    #   3. For each crossing: find the intersection point C.
    #   4. If C is within _ENDPOINT_SNAP_MAX_EXTEND_M of one end of B, trim
    #      that end of B to C (using shapely substring) and move P to C.

    n_stage2b = 0
    _PROBE_EXTRA_M = 1.0  # extend probe slightly past P so crossing is interior

    # Build probes for every endpoint currently in sw_coords_cache
    _s2b_probe_geoms: list[LineString] = []
    # Each key stores (seg_idx, side, seg_end, ep_at_build_time) so that the
    # cross_dist_from_P guard can compare against the *original* probe endpoint
    # even if sw_coords_cache is updated by earlier Stage 2b iterations.
    _s2b_probe_keys: list[tuple[int, str, str, tuple[float, float]]] = []

    for (seg_idx, side), cds in sw_coords_cache.items():
        if seg_idx in _roundabout_idxs:
            continue
        if len(cds) < 2:
            continue
        for seg_end in ("start", "end"):
            ep = cds[0] if seg_end == "start" else cds[-1]
            # Tangent direction pointing INTO the segment from this endpoint
            if seg_end == "start":
                t_dx = cds[1][0] - cds[0][0]
                t_dy = cds[1][1] - cds[0][1]
            else:
                t_dx = cds[-2][0] - cds[-1][0]
                t_dy = cds[-2][1] - cds[-1][1]
            t_len = math.sqrt(t_dx * t_dx + t_dy * t_dy)
            if t_len < 0.001:
                continue
            t_dx /= t_len
            t_dy /= t_len
            # Probe: from MAX_EXTEND_M behind P to PROBE_EXTRA_M past P
            probe_back = (ep[0] - t_dx * _ENDPOINT_SNAP_MAX_EXTEND_M,
                          ep[1] - t_dy * _ENDPOINT_SNAP_MAX_EXTEND_M)
            probe_fwd  = (ep[0] + t_dx * _PROBE_EXTRA_M,
                          ep[1] + t_dy * _PROBE_EXTRA_M)
            _s2b_probe_geoms.append(LineString([probe_back, probe_fwd]))
            _s2b_probe_keys.append((seg_idx, side, seg_end, ep))

    # Build STRtree of all current sidewalk geometries
    _s2b_sw_geoms: list[LineString] = []
    _s2b_sw_keys: list[tuple[int, str]] = []
    for (s_idx, s_side), cds in sw_coords_cache.items():
        if len(cds) >= 2:
            _s2b_sw_geoms.append(LineString(cds))
            _s2b_sw_keys.append((s_idx, s_side))

    if _s2b_probe_geoms and _s2b_sw_geoms:
        _s2b_tree = STRtree(np.array(_s2b_sw_geoms, dtype=object))
        _s2b_qi, _s2b_ti = _s2b_tree.query(
            np.array(_s2b_probe_geoms, dtype=object), predicate="crosses"
        )

        for qi, ti in zip(_s2b_qi.tolist(), _s2b_ti.tolist()):
            probe_seg_idx, probe_side, probe_end, probe_ep = _s2b_probe_keys[qi]
            target_seg_idx, target_side = _s2b_sw_keys[ti]

            if probe_seg_idx == target_seg_idx:
                continue
            if target_seg_idx in _roundabout_idxs:
                continue

            probe_geom = _s2b_probe_geoms[qi]
            target_cds = sw_coords_cache.get((target_seg_idx, target_side))
            if not target_cds or len(target_cds) < 2:
                continue

            target_ls = LineString(target_cds)
            ix = probe_geom.intersection(target_ls)
            if ix.is_empty:
                continue
            # Resolve to a single point
            if hasattr(ix, "geoms"):
                pts = [g for g in ix.geoms if hasattr(g, "x")]
                if not pts:
                    continue
                cross_pt: tuple[float, float] = (pts[0].x, pts[0].y)
            elif hasattr(ix, "x"):
                cross_pt = (ix.x, ix.y)
            else:
                continue

            # Guard: crossing must be very close to P (the probe's source endpoint
            # at build time).  A 12 m backward probe will geometrically cross many
            # perpendicular sidewalks far from P — those are false positives.
            # We use probe_ep (captured at build time) rather than the current cache
            # value, because sw_coords_cache may have been updated by earlier
            # iterations in this same Stage 2b loop.
            _d_from_P = math.sqrt(
                (cross_pt[0] - probe_ep[0]) ** 2 + (cross_pt[1] - probe_ep[1]) ** 2
            )
            if _d_from_P > _PROBE_EXTRA_M + 0.5:
                continue

            # Determine which end of the target to trim
            t_start = target_cds[0]
            t_end   = target_cds[-1]
            d_start = math.sqrt((cross_pt[0]-t_start[0])**2 + (cross_pt[1]-t_start[1])**2)
            d_end   = math.sqrt((cross_pt[0]-t_end[0])**2   + (cross_pt[1]-t_end[1])**2)

            if d_start < d_end and d_start <= _ENDPOINT_SNAP_MAX_EXTEND_M and d_start > 0.01:
                trim_end = "start"
            elif d_end <= d_start and d_end <= _ENDPOINT_SNAP_MAX_EXTEND_M and d_end > 0.01:
                trim_end = "end"
            else:
                continue

            # Don't re-trim an endpoint already placed by Stage 3 crosses-trimming
            tgt_trim_key = (target_seg_idx, target_side, trim_end)
            if tgt_trim_key in trimmed_keys:
                continue

            # Trim the target segment with substring
            tgt_geom_col = f"sidewalk_{target_side}_geometry"
            tgt_geom = populated.at[target_seg_idx, tgt_geom_col]
            if not isinstance(tgt_geom, BaseGeometry) or tgt_geom.is_empty:
                continue
            tgt_ls_full = _to_linestring(tgt_geom)
            if tgt_ls_full is None or tgt_ls_full.length < 0.1:
                continue
            t_proj = tgt_ls_full.project(Point(cross_pt))
            if trim_end == "start":
                trimmed = _sw_substring(tgt_ls_full, t_proj, tgt_ls_full.length)
            else:
                trimmed = _sw_substring(tgt_ls_full, 0, t_proj)
            if trimmed is None or trimmed.is_empty or trimmed.length < 0.1:
                continue

            populated.at[target_seg_idx, tgt_geom_col] = trimmed  # type: ignore[index]
            new_cds = [(c[0], c[1]) for c in trimmed.coords]
            sw_coords_cache[(target_seg_idx, target_side)] = new_cds
            n_endpoints_moved += 1

            # Snap the probe's own endpoint to the crossing (may be sub-centimetre).
            # Skip if Stage 3 already placed this endpoint correctly.
            probe_geom_col = f"sidewalk_{probe_side}_geometry"
            probe_obj = populated.at[probe_seg_idx, probe_geom_col]
            if isinstance(probe_obj, BaseGeometry) and not probe_obj.is_empty:
                probe_trim_key = (probe_seg_idx, probe_side, probe_end)
                if probe_trim_key not in trimmed_keys:
                    pr_cds = sw_coords_cache.get((probe_seg_idx, probe_side))
                    if pr_cds:
                        cur_ep = pr_cds[0] if probe_end == "start" else pr_cds[-1]
                        d_ep = math.sqrt((cross_pt[0]-cur_ep[0])**2 + (cross_pt[1]-cur_ep[1])**2)
                        if 0.01 < d_ep <= _ENDPOINT_SNAP_MAX_EXTEND_M:
                            populated.at[probe_seg_idx, probe_geom_col] = _move_endpoint(  # type: ignore[index]
                                probe_obj, probe_end, cross_pt
                            )
                            cl = list(pr_cds)
                            if probe_end == "start":
                                cl[0] = cross_pt
                            else:
                                cl[-1] = cross_pt
                            sw_coords_cache[(probe_seg_idx, probe_side)] = cl
                            n_endpoints_moved += 1

            n_stage2b += 1

    # ── Stage RA: Roundabout corner snap ──────────────────────────────────
    # At each roundabout corner (between two adjacent entry streets), snap
    # the two closest sidewalk endpoints to their midpoint so they share a
    # single curb-ramp position.
    n_stage_ra = 0

    if _roundabout_idxs:
        # Build graph of roundabout segment connectivity: node_id -> {neighbour_ids}
        _ra_graph: dict[object, set[object]] = defaultdict(set)
        _ra_node_ids: set[object] = set()
        for _ra_idx in _roundabout_idxs:
            _sn = populated.at[_ra_idx, "start_node_id"]
            _en = populated.at[_ra_idx, "end_node_id"]
            _ra_graph[_sn].add(_en)
            _ra_graph[_en].add(_sn)
            _ra_node_ids.add(_sn)
            _ra_node_ids.add(_en)

        # Find entry nodes: roundabout nodes also touched by non-roundabout segments
        _ra_entry_segs: dict[object, list[tuple[int, str]]] = defaultdict(list)
        _non_ra_mask = ~populated.index.isin(list(_roundabout_idxs))
        for _node_col, _pos in (("start_node_id", "start"), ("end_node_id", "end")):
            if _node_col not in populated.columns:
                continue
            _in_ra = populated[_node_col].isin(_ra_node_ids) & _non_ra_mask
            for _idx in populated.index[_in_ra]:
                _nid = populated.at[_idx, _node_col]
                _ra_entry_segs[_nid].append((int(_idx), _pos))

        # BFS from each entry node through roundabout-only nodes to find
        # adjacent entry node pairs.
        _ra_processed_pairs: set[frozenset[object]] = set()

        for _entry_a in list(_ra_entry_segs.keys()):
            _visited: set[object] = {_entry_a}
            _queue = list(_ra_graph.get(_entry_a, set()))
            while _queue:
                _cur = _queue.pop(0)
                if _cur in _visited:
                    continue
                _visited.add(_cur)
                if _cur in _ra_entry_segs:
                    # Found adjacent entry node — deduplicate the pair
                    _pk2 = frozenset((_entry_a, _cur))
                    if _pk2 in _ra_processed_pairs:
                        continue
                    _ra_processed_pairs.add(_pk2)

                    # Collect sidewalk endpoints from approaching streets
                    _eps_a: list[tuple[int, str, str, tuple[float, float]]] = []
                    _eps_b: list[tuple[int, str, str, tuple[float, float]]] = []

                    for _seg_idx, _seg_end in _ra_entry_segs[_entry_a]:
                        for _sw_side in ("left", "right"):
                            _cds = sw_coords_cache.get((_seg_idx, _sw_side))
                            if not _cds:
                                continue
                            _ep = _cds[0] if _seg_end == "start" else _cds[-1]
                            _eps_a.append((_seg_idx, _sw_side, _seg_end, _ep))

                    for _seg_idx, _seg_end in _ra_entry_segs[_cur]:
                        for _sw_side in ("left", "right"):
                            _cds = sw_coords_cache.get((_seg_idx, _sw_side))
                            if not _cds:
                                continue
                            _ep = _cds[0] if _seg_end == "start" else _cds[-1]
                            _eps_b.append((_seg_idx, _sw_side, _seg_end, _ep))

                    if not _eps_a or not _eps_b:
                        continue

                    # Find closest pair across the two entry nodes
                    _best_d = float("inf")
                    _best_a: tuple[int, str, str, tuple[float, float]] | None = None
                    _best_b: tuple[int, str, str, tuple[float, float]] | None = None
                    for _a in _eps_a:
                        for _b in _eps_b:
                            _d = math.sqrt((_a[3][0] - _b[3][0]) ** 2
                                           + (_a[3][1] - _b[3][1]) ** 2)
                            if _d < _best_d:
                                _best_d = _d
                                _best_a = _a
                                _best_b = _b

                    if (_best_a is None or _best_b is None
                            or _best_d > _ENDPOINT_SNAP_MAX_EXTEND_M):
                        continue

                    # Snap both to midpoint
                    _mid = ((_best_a[3][0] + _best_b[3][0]) / 2,
                            (_best_a[3][1] + _best_b[3][1]) / 2)

                    for _ep_info in (_best_a, _best_b):
                        _seg_idx, _sw_side, _seg_end, _ep_xy = _ep_info
                        _geom_col = f"sidewalk_{_sw_side}_geometry"
                        _g = populated.at[_seg_idx, _geom_col]
                        if not isinstance(_g, BaseGeometry) or _g.is_empty:
                            continue
                        _dsq = ((_mid[0] - _ep_xy[0]) ** 2
                                + (_mid[1] - _ep_xy[1]) ** 2)
                        if _dsq < 0.0001:
                            continue
                        populated.at[_seg_idx, _geom_col] = _move_endpoint(  # type: ignore[index]
                            _g, _seg_end, _mid
                        )
                        _cached = sw_coords_cache.get((_seg_idx, _sw_side))
                        if _cached:
                            _cl = list(_cached)
                            if _seg_end == "start":
                                _cl[0] = _mid
                            else:
                                _cl[-1] = _mid
                            sw_coords_cache[(_seg_idx, _sw_side)] = _cl
                        n_endpoints_moved += 1

                    n_stage_ra += 1
                else:
                    # Intermediate roundabout-only node: keep searching
                    for _nb in _ra_graph[_cur]:
                        if _nb not in _visited:
                            _queue.append(_nb)

    print(f"Endpoint snapping: {n_stage1} same-street, "
          f"{n_stage2} cross-street, "
          f"{n_stage2a} antiparallel-outer, "
          f"{n_stage2c} singleton-pairs, "
          f"{n_stage2b} probe-crosses, "
          f"{n_stage3} crosses-trimmed, "
          f"{n_stage_ra} roundabout-corners, "
          f"{n_endpoints_moved} endpoints moved.")
    return populated


_CURB_RAMP_PROXIMITY_M   = 20.0   # fallback: sidewalk endpoint → intersection node distance
_CURB_RAMP_SNAP_M        = 1.0    # snap close-but-not-touching sidewalk endpoints together
_CURB_RAMP_HULL_FALLBACK_R = 15.0 # hull radius when < 3 non-collinear projected points exist


_DEFAULT_SIDEWALK_WIDTH_M = 1.5  # typical US sidewalk width for hull margin

def _sidewalk_max_offset_m(row: "pd.Series[Any]", default_lane_width_m: float) -> float:
    """Max perpendicular offset (m) from road centreline to the outer sidewalk edge.

    Includes half the road width, any bikeway widths on the wider side,
    and the sidewalk width (from data or a default).  This gives the full
    distance from the centreline to where sidewalks can physically cross
    at intersection corners.
    """
    lanes      = _parse_numeric(row.get("lanes"),      2.0)
    lane_width = _parse_numeric(row.get("lane_width"), default_lane_width_m)
    half_road  = (lanes * lane_width) / 2.0
    max_offset = 0.0
    for s in ("left", "right"):
        bike_w = 0.0
        for slot in ("1", "2"):
            w = row.get(f"bikeway_{s}_{slot}_width")
            bike_w += (
                _parse_numeric(w, 0.0) if not _is_na(w)
                else _DEFAULT_BIKE_WIDTH_M if not _is_na(row.get(f"bikeway_{s}_{slot}_type"))
                else 0.0
            )
        sw_w_raw = row.get(f"sidewalk_{s}_width")
        sw_w = _parse_numeric(sw_w_raw, _DEFAULT_SIDEWALK_WIDTH_M) if not _is_na(sw_w_raw) else _DEFAULT_SIDEWALK_WIDTH_M
        max_offset = max(max_offset, half_road + bike_w + sw_w)
    return max_offset if max_offset > 0 else _CURB_RAMP_HULL_FALLBACK_R


def _assign_curb_ramp_geometries(
    populated: gpd.GeoDataFrame,
    default_lane_width_m: float = _DEFAULT_LANE_WIDTH_M,
) -> gpd.GeoDataFrame:
    """Place curb ramp points where sidewalk segments meet at intersection corners.

    Definition
    ----------
    A curb ramp is the geometric intersection of two sidewalk segment
    geometries that falls within the intersection hull for a given node.
    Ramps belong to sidewalk segments only — street centerline nodes are
    never used as ramp positions.

    Hull construction
    -----------------
    For each intersection node N, project outward from N along each connected
    road segment's approach direction by that segment's max sidewalk offset.
    The convex hull of those projected points defines the intersection zone.
    If fewer than 3 non-collinear projected points exist, fall back to a
    circle of radius ``_CURB_RAMP_HULL_FALLBACK_R`` centred on N.

    Snapping
    --------
    Before checking for intersections, sidewalk endpoints within
    ``_CURB_RAMP_SNAP_M`` of each other (but not already coincident) are
    snapped to their midpoint.  The geometry in *populated* is updated in
    place so the snap persists in the final output.

    Redundancy
    ----------
    Both road segment rows that share a curb ramp store the same Point in
    their respective ``sidewalk_{side}_curbramp_{pos}_1_geometry`` column.
    This is intentional: graph builders deduplicate nodes by coordinate, so
    the redundancy is harmless for pathfinding performance.

    Fallback
    --------
    Sidewalk endpoints that are within ``_CURB_RAMP_PROXIMITY_M`` of any
    intersection node but where no pairwise sidewalk intersection was found
    (e.g. a T-junction with only one sidewalk on a corner) receive a ramp
    via direct proximity assignment.

    Slots 2 and 3 are reserved for crowdsourced / government data.
    """
    # ── Inner helpers ─────────────────────────────────────────────────────────

    def _endpoint(geom: BaseGeometry | None, which: str, coords: list[tuple[float, float]] | None = None) -> Point | None:
        """Start (coords[0]) or end (coords[-1]) Point of a geometry.

        If coords is provided, use it instead of re-computing from geom.
        """
        if not isinstance(geom, BaseGeometry) or geom.is_empty:
            return None
        if coords is None:
            coords = _flatten_coords(geom)
        return Point(coords[0] if which == "start" else coords[-1]) if coords else None

    def _closer_position(geom: BaseGeometry, ref: Point, coords: list[tuple[float, float]] | None = None) -> str:
        """Return 'start' or 'end' — whichever endpoint of *geom* is closer to *ref*.

        If coords is provided, use it instead of re-computing from geom.
        """
        if coords is None:
            coords = _flatten_coords(geom)
        if not coords:
            return "start"
        d_start = np.hypot(coords[0][0] - ref.x, coords[0][1] - ref.y)
        d_end = np.hypot(coords[-1][0] - ref.x, coords[-1][1] - ref.y)
        return "start" if d_start <= d_end else "end"

    def _snap_endpoint(geom: BaseGeometry, which: str, new_pt: Point) -> BaseGeometry:
        """Return a copy of *geom* with the start or end coordinate moved to *new_pt*."""
        if isinstance(geom, MultiLineString):
            parts = [list(ls.coords) for ls in geom.geoms]
            if which == "start":
                parts[0][0] = (new_pt.x, new_pt.y)
            else:
                parts[-1][-1] = (new_pt.x, new_pt.y)
            return MultiLineString([LineString(p) for p in parts])
        coords = list(geom.coords)  # type: ignore[union-attr]
        if which == "start":
            coords[0] = (new_pt.x, new_pt.y)
        else:
            coords[-1] = (new_pt.x, new_pt.y)
        return LineString(coords)

    # Pre-create all curb ramp geometry columns at once to avoid fragmentation
    _ramp_cols = {
        f"sidewalk_{side}_curbramp_{pos}_1_geometry": pd.array([pd.NA] * len(populated), dtype=object)
        for side in ("left", "right") for pos in ("start", "end")
        if f"sidewalk_{side}_curbramp_{pos}_1_geometry" not in populated.columns
    }
    if _ramp_cols:
        populated = _add_cols(populated, _ramp_cols)

    # ── Step 1: Group road segments by intersection node ──────────────────────
    # node_key (rounded x, y) → [(row_idx, "start"|"end"), ...]
    node_to_segs: dict[tuple[float, float], list[tuple[int, str]]] = defaultdict(list)
    node_key_to_pt: dict[tuple[float, float], Point] = {}
    for node_col, flag_col, position in (
        ("start_node_geometry", "start_node_is_intersection_node", "start"),
        ("end_node_geometry",   "end_node_is_intersection_node",   "end"),
    ):
        if node_col not in populated.columns or flag_col not in populated.columns:
            continue
        mask = populated[flag_col] == True  # noqa: E712
        masked_idxs = populated.index[mask]
        masked_geoms = populated.loc[mask, node_col].to_numpy(dtype=object)
        _valid_geom_mask = np.asarray(
            shapely.is_geometry(masked_geoms) & ~shapely.is_empty(masked_geoms), dtype=bool
        )
        masked_idxs  = masked_idxs[_valid_geom_mask]
        masked_geoms = masked_geoms[_valid_geom_mask]
        for idx, geom in zip(masked_idxs, masked_geoms):
            pt = cast(Point, geom)
            key = (round(pt.x, 1), round(pt.y, 1))
            node_to_segs[key].append((idx, position))
            if key not in node_key_to_pt:
                node_key_to_pt[key] = pt

    # ── Step 1.5: Retroactive footway slot filling ────────────────────────────
    # Footway-type road edges dropped during the initial sidewalk-matching pass
    # (slot occupied, dedup, or distance rejection) may still have endpoints
    # within _CURB_RAMP_PROXIMITY_M of an intersection node.
    #
    # Strategy
    # --------
    # a) If a parallel road at that node has an *empty* sidewalk slot on the
    #    correct side, populate it so the pairwise/fallback logic fires naturally.
    # b) Otherwise write the curb ramp directly to the footway edge's own
    #    sidewalk_left_curbramp_{pos}_1_geometry column (provided the footway is
    #    parallel to at least one road at the intersection — excludes connectors
    #    and crosswalks whose endpoints happen to be near a node).
    #
    # Steps are excluded: they lack curb ramps by definition.
    _FW_HW_RETRO = {"footway", "pedestrian", "path", "corridor"}

    import math as _math_cr  # noqa: PLC0415 — module-level import is cached

    def _retro_bearing(geom: BaseGeometry) -> float | None:
        coords = _flatten_coords(geom)
        if len(coords) < 2:
            return None
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        return _math_cr.degrees(_math_cr.atan2(dx, dy)) % 180

    def _retro_parallel(b1: float | None, b2: float | None, tol: float = 30.0) -> bool:
        if b1 is None or b2 is None:
            return True
        diff = abs(b1 - b2) % 180
        return diff < tol or diff > (180.0 - tol)

    def _retro_side(road_geom: BaseGeometry, query_pt: Point) -> str:
        """Left or right of road direction vector using cross product."""
        coords = _flatten_coords(road_geom)
        if len(coords) < 2:
            return "left"
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        px = query_pt.x - coords[0][0]
        py = query_pt.y - coords[0][1]
        return "left" if (dx * py - dy * px) > 0 else "right"

    _retro_int_pts: list[Point] = list(node_key_to_pt.values())
    _retro_int_keys: list[tuple[float, float]] = list(node_key_to_pt.keys())
    _retro_int_tree: STRtree | None = STRtree(_retro_int_pts) if _retro_int_pts else None

    # ── Fix 1: geometry index dict — O(1) row lookup without pandas overhead ──
    _geom_by_idx: dict[int, Any] = dict(
        zip(populated.index, populated.geometry.to_numpy(dtype=object))
    )
    # ── Fix 2: bearing cache — pre-compute once per connection segment ─────────
    _bearing_cache: dict[int, float | None] = {}
    for _bc_segs in node_to_segs.values():
        for _bc_idx, _ in _bc_segs:
            if _bc_idx not in _bearing_cache:
                _bc_g = _geom_by_idx.get(_bc_idx)
                _bearing_cache[_bc_idx] = (
                    _retro_bearing(_bc_g) if isinstance(_bc_g, BaseGeometry) else None
                )

    n_fw_slot_filled = 0
    n_fw_ramp_direct = 0

    if _retro_int_tree is not None and "highway" in populated.columns:
        # Pre-filter to footway-type rows to avoid iterating all segments.
        _fw_retro_mask = populated["highway"].isin(_FW_HW_RETRO)
        _fw_retro_idxs = populated.index[_fw_retro_mask]
        _fw_retro_geoms = populated.geometry[_fw_retro_mask].to_numpy(dtype=object)
        _fw_left_sw_arr = (
            populated["sidewalk_left_geometry"][_fw_retro_mask].to_numpy(dtype=object)
            if "sidewalk_left_geometry" in populated.columns
            else np.full(len(_fw_retro_idxs), None)
        )
        _fw_right_sw_arr = (
            populated["sidewalk_right_geometry"][_fw_retro_mask].to_numpy(dtype=object)
            if "sidewalk_right_geometry" in populated.columns
            else np.full(len(_fw_retro_idxs), None)
        )
        for _fi, fw_idx in enumerate(tqdm(_fw_retro_idxs, total=len(_fw_retro_idxs),
                                          desc="Footway retroactive", unit="seg")):
            fw_geom = _fw_retro_geoms[_fi]
            if not isinstance(fw_geom, BaseGeometry) or fw_geom.is_empty:
                continue
            # Skip footways that already received a sidewalk geometry match
            # during the initial pass — the standard logic handles those.
            _left_sw  = _fw_left_sw_arr[_fi]
            _right_sw = _fw_right_sw_arr[_fi]
            if isinstance(_left_sw, BaseGeometry) or isinstance(_right_sw, BaseGeometry):
                continue
            fw_bearing = _retro_bearing(fw_geom)
            fw_coords  = _flatten_coords(fw_geom)
            if len(fw_coords) < 2:
                continue
            for fw_coord, fw_pos in [(fw_coords[0], "start"), (fw_coords[-1], "end")]:
                fw_ep = Point(fw_coord)
                near_idxs = _retro_int_tree.query(fw_ep.buffer(_CURB_RAMP_PROXIMITY_M))
                if not len(near_idxs):
                    continue
                # Find nearest intersection node within threshold
                nearest_key: tuple[float, float] | None = None
                nearest_dist = float("inf")
                for ni in near_idxs:
                    d = _retro_int_pts[ni].distance(fw_ep)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_key = _retro_int_keys[ni]
                if nearest_key is None or nearest_dist > _CURB_RAMP_PROXIMITY_M:
                    continue

                connections_at_node = node_to_segs.get(nearest_key, [])

                # a) Try to fill an empty parallel sidewalk slot at the node
                best_slot: tuple[int, str] | None = None
                best_slot_dist = float("inf")
                for road_idx_r, _ in connections_at_node:
                    if road_idx_r == fw_idx:
                        continue
                    road_g = _geom_by_idx.get(road_idx_r)
                    if not isinstance(road_g, BaseGeometry):
                        continue
                    if road_idx_r not in _bearing_cache:
                        _bearing_cache[road_idx_r] = _retro_bearing(road_g)
                    if not _retro_parallel(fw_bearing, _bearing_cache[road_idx_r]):
                        continue
                    side_r = _retro_side(road_g, fw_ep)
                    geom_col_r = f"sidewalk_{side_r}_geometry"
                    if geom_col_r not in populated.columns:
                        continue
                    existing_r = populated.at[road_idx_r, geom_col_r]
                    if isinstance(existing_r, BaseGeometry) and not existing_r.is_empty:
                        continue  # slot occupied
                    d_r = road_g.distance(fw_geom)
                    if d_r < best_slot_dist:
                        best_slot_dist = d_r
                        best_slot = (road_idx_r, side_r)

                if best_slot is not None:
                    road_idx_b, side_b = best_slot
                    populated.at[road_idx_b, f"sidewalk_{side_b}_geometry"] = fw_geom  # type: ignore[index]
                    n_fw_slot_filled += 1
                    continue  # slot filled; pairwise/fallback will assign the ramp

                # b) No empty slot — write curb ramp directly if footway is
                #    parallel to at least one road at this intersection node
                #    (guards against connectors / crosswalk endpoints).
                is_parallel_to_road = any(
                    _retro_parallel(fw_bearing, _bearing_cache.get(r_idx))
                    for r_idx, _ in connections_at_node
                    if r_idx != fw_idx
                    and isinstance(_geom_by_idx.get(r_idx), BaseGeometry)
                )
                if not is_parallel_to_road:
                    continue
                ramp_col_direct = f"sidewalk_left_curbramp_{fw_pos}_1_geometry"
                if ramp_col_direct in populated.columns and pd.isna(
                    populated.at[fw_idx, ramp_col_direct]
                ):
                    populated.at[fw_idx, ramp_col_direct] = fw_ep  # type: ignore[index]
                    n_fw_ramp_direct += 1

    print(f"  Footway retroactive: {n_fw_slot_filled} slots filled, {n_fw_ramp_direct} direct ramps")

    # ── Step 2: Pre-build STRtree over all sidewalk geometries ────────────────
    # Stored as (row_idx, "left"|"right") parallel to the tree's geometry list.
    all_sw_keys: list[tuple[int, str]] = []
    all_sw_geoms: list[BaseGeometry] = []
    for side in ("left", "right"):
        col = f"sidewalk_{side}_geometry"
        if col not in populated.columns:
            continue
        col_vals = populated[col].to_numpy(dtype=object)
        for idx, g in zip(populated.index, col_vals):
            if isinstance(g, BaseGeometry) and not g.is_empty:
                all_sw_keys.append((idx, side))
                all_sw_geoms.append(g)

    sw_tree: STRtree | None = STRtree(all_sw_geoms) if all_sw_geoms else None

    # ── Fix 5: Pre-fetch hull columns to avoid populated.loc[row_idx] per node ─
    _start_node_geom_map: dict[int, Any] = (
        dict(zip(populated.index, populated["start_node_geometry"].to_numpy(dtype=object)))
        if "start_node_geometry" in populated.columns else {}
    )
    _end_node_geom_map: dict[int, Any] = (
        dict(zip(populated.index, populated["end_node_geometry"].to_numpy(dtype=object)))
        if "end_node_geometry" in populated.columns else {}
    )
    _hull_offset_cols = [c for c in (
        "lanes", "lane_width",
        "bikeway_left_1_width", "bikeway_left_1_type",
        "bikeway_left_2_width", "bikeway_left_2_type",
        "bikeway_right_1_width", "bikeway_right_1_type",
        "bikeway_right_2_width", "bikeway_right_2_type",
    ) if c in populated.columns]
    _conn_idxs = list({row_idx for segs in node_to_segs.values() for row_idx, _ in segs})
    _offset_sub = (
        populated.loc[_conn_idxs, _hull_offset_cols]
        if _hull_offset_cols else pd.DataFrame(index=_conn_idxs)
    )
    _offset_by_idx: dict[int, float] = {
        int(idx): _sidewalk_max_offset_m(_offset_sub.loc[idx], default_lane_width_m)
        for idx in _offset_sub.index
    }

    n_assigned = 0
    n_snapped  = 0

    # ── Step 3: Process each intersection node ────────────────────────────────
    for node_key, connections in tqdm(node_to_segs.items(), total=len(node_to_segs),
                                      desc="Curb ramp geometries", unit="node"):
        node_pt = node_key_to_pt.get(node_key)
        if node_pt is None:
            continue

        # ── Build convex hull of projected sidewalk offsets ───────────────────
        # Phase 1: project one point per *unique* road approach direction along
        #   its centreline.  Duplicate / antiparallel approaches (common at 4-way
        #   intersections where osmnx creates both u→v and v→u edges) are merged
        #   to prevent spurious corner-fill points.
        # Phase 1b: include actual sidewalk endpoints near the node so the hull
        #   covers all relevant geometry (offset-only projections can fall short
        #   of where sidewalk bodies actually cross).
        # Phase 2: add corner points at bisector directions between adjacent
        #   approaches so the hull fills intersection corners rather than
        #   forming a diamond.  The corner distance is max(o1,o2)/cos(half_gap),
        #   capped at 3× the larger offset to avoid runaway hulls on near-
        #   parallel approaches.
        import math as _hm
        hull_pts_list: list[Point] = []
        _hull_approaches_raw: list[tuple[float, float]] = []
        for row_idx, position in connections:
            far_geom = (
                _end_node_geom_map if position == "start" else _start_node_geom_map
            ).get(row_idx)
            if not isinstance(far_geom, BaseGeometry) or far_geom.is_empty:
                continue
            far_pt = cast(Point, far_geom)
            dx = far_pt.x - node_pt.x
            dy = far_pt.y - node_pt.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1e-6:
                continue
            offset = _offset_by_idx.get(row_idx, _CURB_RAMP_HULL_FALLBACK_R)
            hull_pts_list.append(
                Point(node_pt.x + dx / dist * offset, node_pt.y + dy / dist * offset)
            )
            _hull_approaches_raw.append((_hm.degrees(_hm.atan2(dx, dy)) % 360.0, offset))

        # Deduplicate approaches: merge entries within 20° (parallel or
        # antiparallel) keeping the maximum offset.
        _hull_approaches: list[tuple[float, float]] = []
        _HULL_DEDUP_TOL = 20.0
        for _brg, _off in sorted(_hull_approaches_raw, key=lambda x: x[0]):
            _merged = False
            for _ei in range(len(_hull_approaches)):
                _ebrg, _eoff = _hull_approaches[_ei]
                _bdiff = abs(_brg - _ebrg) % 360.0
                if _bdiff > 180.0:
                    _bdiff = 360.0 - _bdiff
                if _bdiff < _HULL_DEDUP_TOL or (180.0 - _bdiff) < _HULL_DEDUP_TOL:
                    _hull_approaches[_ei] = (_ebrg, max(_eoff, _off))
                    _merged = True
                    break
            if not _merged:
                _hull_approaches.append((_brg, _off))

        # Phase 1b: include actual sidewalk endpoints near the node.
        # This ensures the hull covers sidewalk body crossings that extend
        # beyond the road-width-based projections.
        _hull_sw_max_d = _CURB_RAMP_HULL_FALLBACK_R
        for row_idx, position in connections:
            for _sw_side in ("left", "right"):
                _sw_col = f"sidewalk_{_sw_side}_geometry"
                if _sw_col not in populated.columns:
                    continue
                _sw_g = populated.at[row_idx, _sw_col]
                if not isinstance(_sw_g, BaseGeometry) or _sw_g.is_empty:
                    continue
                _sw_cds = _flatten_coords(_sw_g)
                if not _sw_cds:
                    continue
                _sw_ep = _sw_cds[0] if position == "start" else _sw_cds[-1]
                _sw_d = _hm.sqrt((_sw_ep[0] - node_pt.x) ** 2 + (_sw_ep[1] - node_pt.y) ** 2)
                if _sw_d <= _hull_sw_max_d:
                    hull_pts_list.append(Point(_sw_ep[0], _sw_ep[1]))

        # Phase 2: corner fill
        if len(_hull_approaches) >= 2:
            _hull_approaches.sort(key=lambda x: x[0])
            _n_app = len(_hull_approaches)
            for _i in range(_n_app):
                _b1, _o1 = _hull_approaches[_i]
                _b2, _o2 = _hull_approaches[(_i + 1) % _n_app]
                if _b2 <= _b1:          # wrap-around (e.g. 350° → 10°)
                    _b2 += 360.0
                _gap      = _b2 - _b1
                _bisector = (_b1 + _gap / 2.0) % 360.0
                _half_rad = _hm.radians(_gap / 2.0)
                _max_o    = max(_o1, _o2)
                _sum_o    = _o1 + _o2
                if _half_rad >= _hm.radians(85.0):
                    _c_dist = _max_o * 1.5   # cap for very wide gaps
                else:
                    # Use the larger of cosine-based and sum-based estimates.
                    # Cosine works well for acute angles; sum-of-offsets
                    # correctly models where offset sidewalks cross at ~90°
                    # intersections (diagonal ≈ o1 + o2).
                    _c_dist = min(max(_max_o / _hm.cos(_half_rad), _sum_o),
                                  _max_o * 3.0)
                _bR = _hm.radians(_bisector)
                hull_pts_list.append(Point(
                    node_pt.x + _hm.sin(_bR) * _c_dist,
                    node_pt.y + _hm.cos(_bR) * _c_dist,
                ))

        if len(hull_pts_list) >= 3:
            candidate = MultiPoint(hull_pts_list).convex_hull
            hull: BaseGeometry = (
                candidate if candidate.geom_type in ("Polygon", "MultiPolygon")
                else node_pt.buffer(_CURB_RAMP_HULL_FALLBACK_R)
            )
        elif len(hull_pts_list) == 2:
            p0, p1 = hull_pts_list
            r = max(p0.distance(p1) / 2.0, _CURB_RAMP_HULL_FALLBACK_R)
            hull = Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2).buffer(r)
        else:
            hull = node_pt.buffer(_CURB_RAMP_HULL_FALLBACK_R)

        # ── Collect sidewalk geometries within this hull ──────────────────────
        if sw_tree is None:
            continue
        nearby_tree_idxs = sw_tree.query(hull, predicate="intersects")
        local_sw: list[tuple[int, str]] = [all_sw_keys[i] for i in nearby_tree_idxs]
        if len(local_sw) < 2:
            continue

        # ── Snap close-but-not-touching endpoints ─────────────────────────────
        # Pre-compute flattened coordinates for all local_sw geometries to avoid redundant traversals
        _coords_cache: dict[int, list[tuple[float, float]]] = {}
        for row_idx, side_idx in local_sw:
            geom = cast(BaseGeometry | None, populated.at[row_idx, f"sidewalk_{side_idx}_geometry"])
            if isinstance(geom, BaseGeometry) and not geom.is_empty:
                _coords_cache[id(geom)] = _flatten_coords(geom)

        for i in range(len(local_sw)):
            ri, si = local_sw[i]
            gi = cast(BaseGeometry | None, populated.at[ri, f"sidewalk_{si}_geometry"])
            if not isinstance(gi, BaseGeometry) or gi.is_empty:
                continue
            gi_coords = _coords_cache.get(id(gi))
            pos_i = _closer_position(gi, node_pt, gi_coords)
            end_i = _endpoint(gi, pos_i, gi_coords)
            if end_i is None:
                continue
            for j in range(i + 1, len(local_sw)):
                rj, sj = local_sw[j]
                gj = cast(BaseGeometry | None, populated.at[rj, f"sidewalk_{sj}_geometry"])
                if not isinstance(gj, BaseGeometry) or gj.is_empty:
                    continue
                gj_coords = _coords_cache.get(id(gj))
                pos_j = _closer_position(gj, node_pt, gj_coords)
                end_j = _endpoint(gj, pos_j, gj_coords)
                if end_j is None:
                    continue
                d = np.hypot(end_i.x - end_j.x, end_i.y - end_j.y)
                if 0 < d <= _CURB_RAMP_SNAP_M:
                    mid = Point((end_i.x + end_j.x) / 2, (end_i.y + end_j.y) / 2)
                    populated.at[ri, f"sidewalk_{si}_geometry"] = _snap_endpoint(gi, pos_i, mid)  # type: ignore[index]
                    populated.at[rj, f"sidewalk_{sj}_geometry"] = _snap_endpoint(gj, pos_j, mid)  # type: ignore[index]
                    n_snapped += 1
                    # Update gi/end_i for subsequent inner-loop iterations
                    gi = cast(BaseGeometry, populated.at[ri, f"sidewalk_{si}_geometry"])
                    end_i = mid
                    # Invalidate cache for updated geometry
                    if id(gi) in _coords_cache:
                        del _coords_cache[id(gi)]

        # ── Find pairwise sidewalk intersections within hull ──────────────────
        # Accumulate updates for batch writing instead of individual .at[] calls
        _updates: defaultdict[tuple[int, str], Any] = defaultdict(lambda: None)
        _snap_updates: defaultdict[tuple[int, str], Any] = defaultdict(lambda: None)

        for i, (ri, si) in enumerate(local_sw):
            gi = cast(BaseGeometry | None, populated.at[ri, f"sidewalk_{si}_geometry"])
            if not isinstance(gi, BaseGeometry) or gi.is_empty:
                continue
            gi_coords = _coords_cache.get(id(gi))
            for rj, sj in local_sw[i + 1:]:
                gj = cast(BaseGeometry | None, populated.at[rj, f"sidewalk_{sj}_geometry"])
                if not isinstance(gj, BaseGeometry) or gj.is_empty:
                    continue
                gj_coords = _coords_cache.get(id(gj))
                inter = gi.intersection(gj)
                if inter.is_empty:
                    continue
                # Collect candidate Points only (ignore collinear overlaps)
                if isinstance(inter, Point):
                    ramp_candidates: list[Point] = [inter]
                elif isinstance(inter, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
                    ramp_candidates = [p for p in inter.geoms if isinstance(p, Point)]
                else:
                    continue
                for ramp_pt in ramp_candidates:
                    if not hull.covers(ramp_pt):
                        continue
                    for row_x, side_x, geom_x, geom_coords in ((ri, si, gi, gi_coords), (rj, sj, gj, gj_coords)):
                        pos_x   = _closer_position(geom_x, ramp_pt, geom_coords)
                        ramp_col = f"sidewalk_{side_x}_curbramp_{pos_x}_1_geometry"
                        if pd.isna(populated.at[row_x, ramp_col]):
                            _updates[(row_x, ramp_col)] = ramp_pt
                            n_assigned += 1
                        # Snap the endpoint to the ramp point so the geometry
                        # terminates at the corner, not at the overshooting tip.
                        ep_coord = _endpoint(geom_x, pos_x, geom_coords)
                        if ep_coord is not None:
                            d_ep = np.hypot(ep_coord.x - ramp_pt.x,
                                            ep_coord.y - ramp_pt.y)
                            if d_ep > 1e-3:
                                geom_col_x = f"sidewalk_{side_x}_geometry"
                                _snap_updates[(row_x, geom_col_x)] = _snap_endpoint(
                                    geom_x, pos_x, ramp_pt
                                )
                                n_snapped += 1

        # Apply batched updates
        for (row_x, col), val in _updates.items():
            populated.at[row_x, col] = val  # type: ignore[index]
        for (row_x, col), val in _snap_updates.items():
            populated.at[row_x, col] = val  # type: ignore[index]

    # ── Fallback: proximity-based for slots still empty ───────────────────────
    all_node_pts = list(node_key_to_pt.values())
    if all_node_pts:
        # Pre-extract node coordinates as a numpy array for batch distance checks
        _node_coords = np.array([(p.x, p.y) for p in all_node_pts], dtype=np.float64)
        intersect_tree = STRtree(all_node_pts)
        for side in ("left", "right"):
            geom_col = f"sidewalk_{side}_geometry"
            if geom_col not in populated.columns:
                continue
            # Build has_geom and coords cache once per side (shared by start + end).
            _geom_arr = populated[geom_col].to_numpy(dtype=object)
            has_geom = pd.Series(
                shapely.is_geometry(_geom_arr) & ~shapely.is_empty(_geom_arr),
                index=populated.index, dtype=bool,
            )
            _geom_mask = has_geom.to_numpy()
            _geom_pos = np.where(_geom_mask)[0]
            _coords_map: dict = {
                idx: _flatten_coords(g)
                for idx, g in zip(populated.index[_geom_pos], _geom_arr[_geom_pos])
            }
            for position in ("start", "end"):
                ramp_col = f"sidewalk_{side}_curbramp_{position}_1_geometry"
                ramp_empty = populated[ramp_col].isna()
                _cand_mask = (has_geom & ramp_empty).to_numpy()
                candidate_idxs = populated.index[np.where(_cand_mask)[0]]
                if len(candidate_idxs) == 0:
                    continue

                # Batch-build endpoint points and buffers
                _ep_coords_list = []
                _valid_indices = []
                for idx in candidate_idxs:
                    coords = _coords_map.get(idx)
                    if not coords:
                        continue
                    _ep_coords_list.append(coords[0] if position == "start" else coords[-1])
                    _valid_indices.append(idx)

                if not _ep_coords_list:
                    continue

                _ep_xy = np.array(_ep_coords_list, dtype=np.float64)
                _ep_pts = shapely.points(_ep_xy[:, 0], _ep_xy[:, 1])

                # Batch query: dwithin returns (input_idx, tree_idx) pairs
                # where each point is within _CURB_RAMP_PROXIMITY_M of a tree geometry
                _tree_hits_l, _tree_hits_r = intersect_tree.query(
                    _ep_pts, predicate="dwithin", distance=_CURB_RAMP_PROXIMITY_M
                )

                # Unique input indices that had at least one hit — batch write
                _hit_set: set[int] = set(_tree_hits_l.tolist())
                if _hit_set:
                    _fb_idxs = [_valid_indices[_qi] for _qi in _hit_set]
                    _fb_pts  = [Point(_ep_coords_list[_qi]) for _qi in _hit_set]
                    populated.loc[_fb_idxs, ramp_col] = _fb_pts  # type: ignore[index]
                    n_assigned += len(_hit_set)

    print(f"Curb ramp geometries assigned: {n_assigned} ramps, {n_snapped} endpoint snaps, "
          f"{n_fw_slot_filled} footway slots filled, {n_fw_ramp_direct} footway direct ramps")
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

    # Pre-create all id/grid_id columns at once to avoid per-write fragmentation
    _id_cols_init = {}
    for _side in ("left", "right"):
        for _c in (f"sidewalk_{_side}_ID", f"sidewalk_{_side}_grid_ID",
                   f"bikeway_{_side}_1_id", f"bikeway_{_side}_1_grid_id",
                   f"bikeway_{_side}_2_id"):
            if _c not in populated.columns:
                _id_cols_init[_c] = pd.NA
    if _id_cols_init:
        populated = _add_cols(populated, _id_cols_init)

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
        populated[id_col] = pd.NA
        populated.loc[active, id_col] = val_series
        if grid_id_col is not None:
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
        if sgid is not None:
            only1 = eligible1 & sgid.notna()
            populated[f"bikeway_{side}_1_id"] = pd.NA
            populated.loc[only1, f"bikeway_{side}_1_id"] = sgid[only1].astype(str) + f"{suffix}1"

        # Slot 2 id (no separate grid_id column per schema)
        if sgid is not None:
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

if __name__ == "__main__":
    run_multi_city()

