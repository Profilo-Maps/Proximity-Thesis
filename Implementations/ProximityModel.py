"""
Proximity Pipeline – builds a pedestrian/cycling network parquet from OSM data.

Each step is a function that mutates a GeoDataFrame in-place.
run_pipeline() orchestrates them in order.
The GDF stores geometries in EPSG:4326 throughout; UTM projection is done
ad-hoc for metric calculations (bearings, grid cells, distances).
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import time
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, split, snap, linemerge, transform as _shapely_transform
from shapely import STRtree, intersection as _shapely_intersection, length as _shapely_length
from scipy.spatial import cKDTree  # type: ignore[attr-defined]
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE_DIR = Path("Implementations/.osm_cache")

USGS_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
USGS_MAX_RETRIES = 3
USGS_RETRY_DELAY = 2  # seconds
USGS_WORKERS = 10
USGS_TIMEOUT = 15

_NON_ROAD_HIGHWAY = frozenset({
    "cycleway", "footway", "pedestrian", "path",
    "steps", "corridor", "bridleway",
})

_TC_COINCIDENCE_THRESHOLD_M = 5.0  # snap distance for traffic-calming matching
_CURB_RAMP_DATA_PATH: str = ""

# ── OSM tags to request ──────────────────────────────────────────────────────
_EXTRA_WAY_TAGS = [
    # Street
    "lanes", "lane_width", "maxspeed", "surface", "smoothness", "incline",
    # Cycleway
    "cycleway", "cycleway:left", "cycleway:right", "cycleway:both",
    "cycleway:left:surface", "cycleway:right:surface", "cycleway:surface",
    "cycleway:left:width", "cycleway:right:width", "cycleway:width",
    "cycleway:left:buffer", "cycleway:right:buffer", "cycleway:buffer",
    "cycleway:left:lane", "cycleway:right:lane",
    "cycleway:left:2", "cycleway:right:2", "cycleway:both:2",
    "cycleway:left:2:surface", "cycleway:right:2:surface",
    "cycleway:left:2:width", "cycleway:right:2:width",
    "cycleway:left:2:buffer", "cycleway:right:2:buffer",
    "cycleway:left:2:smoothness", "cycleway:right:2:smoothness",
    "bicycle",
    # Sidewalk
    "sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both",
    "sidewalk:left:surface", "sidewalk:right:surface",
    "sidewalk:left:width", "sidewalk:right:width",
    "sidewalk:left:incline", "sidewalk:right:incline",
    "sidewalk:left:smoothness", "sidewalk:right:smoothness",
    "sidewalk:left:buffer", "sidewalk:right:buffer",
    "foot",
]

_CROSSING_NODE_TAGS = [
    "crossing", "crossing:markings", "crossing:signals",
    "crossing:island", "crossing:continuous",
    "kerb", "tactile_paving", "traffic_calming",
]

# ── Coordinate helpers ────────────────────────────────────────────────────────
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
_to_wgs = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)


def _project_to_utm(geom):
    """Transform a Shapely geometry from EPSG:4326 → EPSG:32610.
    Uses shapely.ops.transform with the module-level Transformer directly
    (no GeoSeries round-trip) — ~100–1000x faster in tight loops.
    """
    return _shapely_transform(_to_utm.transform, geom)


def _project_to_wgs(geom):
    """Transform a Shapely geometry from EPSG:32610 → EPSG:4326.
    Uses shapely.ops.transform with the module-level Transformer directly
    (no GeoSeries round-trip) — ~100–1000x faster in tight loops.
    """
    return _shapely_transform(_to_wgs.transform, geom)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Per-city configuration for the Proximity pipeline."""

    # ── Location ──────────────────────────────────────────────────────────
    place_name: str                          # OSMnx geocodable name
    output_path: str                         # e.g. "Output/…_network.parquet"

    # ── Street defaults ───────────────────────────────────────────────────
    default_maxspeed: int = 25               # mph fallback
    default_lane_width: float = 3.0          # metres
    default_lanes: int = 2

    # ── Deflection splitting (step 3) ─────────────────────────────────────
    deflection_angle_threshold: float = 45.0 # degrees

    # ── Sidewalk offset (step 9) ──────────────────────────────────────────
    default_bikeway_width: float = 1.5       # metres
    default_sidewalk_width: float = 1.5      # metres

    # ── Intersection analysis (steps 12-15) ───────────────────────────────
    hull_merge_buffer_m: float = 10.0
    crosswalk_buffer_m: float = 3.0
    curb_ramp_snap_m: float = 1.0

    # ── USGS elevation (step 6) ───────────────────────────────────────────
    enable_elevation: bool = True
    usgs_resolution: int = 10
    dem_resolution_m: float = 10.0           # segments shorter than this use propagation
    slope_cap_pct: float = 40.0              # hard cap on computed slopes
    max_propagation_dist_m: float = 250.0    # BFS radius for short-segment slope fill

    # ── Grid (step 4) ────────────────────────────────────────────────────
    grid_cell_size: float = 500.0            # metres

    # ── Network filter ────────────────────────────────────────────────────
    network_type: str = "all"
    custom_filter: str | None = None

    # ── Curb ramp enrichment (step 17) ────────────────────────────────────
    curb_ramp_data_path: str = "Data/Curb_Ramps_20260217.csv"
    curb_ramp_match_radius_m: float = 5.0

    @classmethod
    def from_json(cls, path: str) -> PipelineConfig:
        import json as _json
        with open(path) as f:
            return cls(**_json.load(f))


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═════════════════════════════════════════════════════════════════════════════

GEOMETRY_COLUMNS: list[str] = [
    "street_geometry",
    "start_node_geometry",
    "end_node_geometry",
    "sidewalk_left_geometry",
    "sidewalk_right_geometry",
    "curb_return_geometry",
    "bikeway_left_1_geometry",
    "bikeway_left_2_geometry",
    "bikeway_right_1_geometry",
    "bikeway_right_2_geometry",
    "street_feature_geometry",
    "street_feature_geometry_projected",
    "sidewalk_left_feature_geometry",
    "sidewalk_left_feature_geometry_projected",
    "sidewalk_right_feature_geometry",
    "sidewalk_right_feature_geometry_projected",
    "bikeway_left_1_feature_geometry",
    "bikeway_left_1_feature_geometry_projected",
    "bikeway_left_2_feature_geometry",
    "bikeway_left_2_feature_geometry_projected",
    "bikeway_right_1_feature_geometry",
    "bikeway_right_1_feature_geometry_projected",
    "bikeway_right_2_feature_geometry",
    "bikeway_right_2_feature_geometry_projected",
    "crosswalk_start_geometry",
    "crosswalk_start_island_geometry",
    "crosswalk_end_geometry",
    "crosswalk_end_island_geometry",
]
for _side in ("left", "right"):
    for _pos in ("start", "end"):
        for _n in (1, 2, 3):
            GEOMETRY_COLUMNS.append(f"sidewalk_{_side}_curbramp_{_pos}_{_n}_geometry")


def _build_empty_schema() -> dict[str, Any]:
    """Return {column_name: default_value} for every column in the output schema."""
    cols: dict[str, Any] = {}

    # ── Street centerline ─────────────────────────────────────────────────
    cols["intersection_review_flag"] = False
    cols["street_id"] = ""
    cols["street_grid_id"] = ""
    cols["public_data_id_street"] = None
    cols["start_node_id"] = 0
    cols["start_node_is_intersection_node"] = False
    cols["end_node_id"] = 0
    cols["end_node_is_intersection_node"] = False
    cols["public_data_id_start_end_nodes"] = None
    cols["normalized_bearing"] = 0.0
    cols["name"] = None
    cols["highway"] = None
    cols["maxspeed"] = 0
    cols["oneway"] = None
    cols["lanes"] = None
    cols["lane_width"] = None
    cols["surface"] = None
    cols["street_condition"] = None
    cols["street_incline"] = None
    cols["elev_start_m"] = None
    cols["elev_end_m"] = None

    # ── Street features ───────────────────────────────────────────────────
    cols["street_feature_types"] = None
    cols["public_data_id_street_feature"] = None
    cols["street_feature_attributes"] = None

    # ── Sidewalks (left + right) ──────────────────────────────────────────
    for side in ("left", "right"):
        cols[f"sidewalk_{side}_ID"] = None
        cols[f"sidewalk_{side}_grid_ID"] = None
        cols[f"sidewalk_{side}_presence"] = None
        cols[f"public_data_id_sidewalk_{side}"] = None
        cols[f"sidewalk_{side}_surface"] = None
        cols[f"sidewalk_{side}_condition"] = None
        cols[f"sidewalk_{side}_quality"] = None
        cols[f"sidewalk_{side}_width"] = None
        cols[f"sidewalk_{side}_incline"] = None
        cols[f"sidewalk_{side}_seperator"] = None
        cols[f"sidewalk_{side}_offset"] = "no"
        for pos in ("start", "end"):
            for n in (1, 2, 3):
                pfx = f"sidewalk_{side}_curbramp_{pos}_{n}"
                cols[f"{pfx}_ID"] = None
                cols[f"public_data_id_{pfx}"] = None
                cols[f"{pfx}_returnloc"] = None
                cols[f"{pfx}_returnposition"] = None
                cols[f"{pfx}_condition_score"] = None
                cols[f"{pfx}_quality"] = None
        cols[f"sidewalk_{side}_feature_ids"] = None
        cols[f"sidewalk_{side}_feature_types"] = None
        cols[f"public_data_id_sidewalk_{side}_feature"] = None

    # ── Crosswalks (start + end) ──────────────────────────────────────────
    for pos in ("start", "end"):
        cols[f"crosswalk_{pos}_id"] = None
        cols[f"crosswalk_{pos}_grid_ids"] = None
        cols[f"crosswalk_{pos}_type"] = None
        cols[f"public_data_id_crosswalk_{pos}"] = None
        cols[f"crosswalk_{pos}_controlled"] = None
        cols[f"crosswalk_{pos}_marked"] = None
        cols[f"crosswalk_{pos}_markings"] = None
        cols[f"crosswalk_{pos}_signals"] = None
        cols[f"crosswalk_{pos}_island"] = None
        cols[f"crosswalk_{pos}_kerb"] = None
        cols[f"crosswalk_{pos}_tactile_paving"] = None
        cols[f"crosswalk_{pos}_traffic_calming"] = None
        cols[f"crosswalk_{pos}_continuous"] = None
        cols[f"crosswalk_{pos}_condition"] = None
        cols[f"crosswalk_{pos}_source"] = None
        cols[f"crosswalk_{pos}_quality"] = None

    # ── Bikeways (left/right × 1/2) ──────────────────────────────────────
    for side in ("left", "right"):
        for n in (1, 2):
            pfx = f"bikeway_{side}_{n}"
            cols[f"{pfx}_id"] = None
            cols[f"{pfx}_grid_id"] = None
            cols[f"public_data_id_{pfx}"] = None
            cols[f"{pfx}_type"] = None
            cols[f"{pfx}_surface"] = None
            cols[f"{pfx}_condition"] = None
            cols[f"{pfx}_quality"] = None
            cols[f"{pfx}_permitted"] = None
            cols[f"{pfx}_width"] = None
            cols[f"{pfx}_incline"] = None
            cols[f"{pfx}_seperator"] = None
            cols[f"{pfx}_offset"] = "no"
            cols[f"{pfx}_feature_ids"] = None
            cols[f"{pfx}_feature_types"] = None
            cols[f"public_data_id_{pfx}_features"] = None

    # ── Neighbor adjacency lists ─────────────────────────────────────────
    # Each column stores a list of grid IDs of adjacent segments (sidewalks,
    # crosswalks, bikeways) connecting at the start (from) or end (to) of
    # this facility. Used for pedestrian/cycling graph traversal.
    for side in ("left", "right"):
        cols[f"sidewalk_{side}_from_segments"] = None
        cols[f"sidewalk_{side}_to_segments"] = None
    for pos in ("start", "end"):
        cols[f"crosswalk_{pos}_from_segments"] = None
        cols[f"crosswalk_{pos}_to_segments"] = None
    for side in ("left", "right"):
        for n in (1, 2):
            cols[f"bikeway_{side}_{n}_from_segments"] = None
            cols[f"bikeway_{side}_{n}_to_segments"] = None

    # ── Geometry columns default to None ──────────────────────────────────
    for gc in GEOMETRY_COLUMNS:
        cols[gc] = None

    return cols


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — parsing
# ═════════════════════════════════════════════════════════════════════════════

def _parse_maxspeed(val: Any, default: int) -> int:
    if pd.isna(val) if not isinstance(val, list) else False:
        return default
    s = str(val[0] if isinstance(val, list) else val).strip()
    for suffix in (" mph", " km/h", " kph"):
        s = s.replace(suffix, "")
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def _parse_int_or_none(val: Any) -> int | None:
    if isinstance(val, list):
        val = val[0]
    if pd.isna(val) if not isinstance(val, str) else False:
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def _parse_float_or_none(val: Any) -> float | None:
    if isinstance(val, list):
        val = val[0]
    if pd.isna(val) if not isinstance(val, str) else False:
        return None
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return column if it exists, else a Series of NA."""
    if name in df.columns:
        return df[name]
    return pd.Series(pd.NA, index=df.index, dtype=object)


def _parse_float_tag_series(s: Any) -> pd.Series:
    """Vectorized: extract leading float from OSM tag strings like '1.5 m'."""
    ser = pd.Series(s) if not isinstance(s, pd.Series) else s
    extracted = ser.astype(str).str.strip().str.extract(r"([+-]?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _parse_incline_series(s: Any) -> pd.Series:
    """Vectorized: convert OSM incline strings ('5%', '-3.2', 'up') to float %."""
    ser = pd.Series(s) if not isinstance(s, pd.Series) else s
    vals = ser.astype(str).str.strip().str.lower()
    vals = vals.str.replace("°", "", regex=False).str.replace("%", "", regex=False).str.strip()
    vals = vals.where(~vals.isin(("up", "down", "nan", "none", "<na>")))
    return pd.to_numeric(vals, errors="coerce").round(4)


def _first_if_list(val: Any) -> Any:
    """Return the first element of a list, or the value itself if scalar/None."""
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _scalar_col(series: pd.Series) -> pd.Series:
    """Coerce a Series that may contain lists to scalar values (take first element)."""
    if any(type(v) is list for v in series):
        return series.apply(_first_if_list)
    return series


def _coalesce_tags(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Return first non-NA value across columns, left to right."""
    result = pd.Series(pd.NA, index=df.index, dtype=object)
    for col in cols:
        if col in df.columns:
            result = result.where(result.notna(), df[col])
    return result


def _ensure_cols(gdf: gpd.GeoDataFrame, schema: dict[str, Any]) -> gpd.GeoDataFrame:
    """Add any missing schema columns with their default values (chunked to bound peak memory)."""
    missing = {k: v for k, v in schema.items() if k not in gdf.columns}
    if not missing:
        return gdf
    n = len(gdf)
    # Build missing columns in chunks of 50 so no single np.vstack blows up
    # peak RAM for large regions (e.g. Alameda ~527k rows x 235 cols = ~945 MB).
    chunk_size = 50
    items = list(missing.items())
    frames = [gdf]
    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        frames.append(pd.DataFrame(
            {col: pd.array([default] * n, dtype=object) for col, default in chunk},
            index=gdf.index,
        ))
    return gpd.GeoDataFrame(
        pd.concat(frames, axis=1),
        geometry="street_geometry",
        crs=gdf.crs,
    )


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — geometry
# ═════════════════════════════════════════════════════════════════════════════

def _flatten_coords(geom: BaseGeometry) -> list[tuple[float, float]]:
    """Flatten a LineString or MultiLineString to a single coordinate list."""
    if isinstance(geom, MultiLineString):
        return [(c[0], c[1]) for ls in geom.geoms for c in ls.coords]
    if hasattr(geom, "coords"):
        return [(c[0], c[1]) for c in geom.coords]
    return []


def _consolidate_line_geom(geom: BaseGeometry) -> BaseGeometry:
    """Merge a MultiLineString to a single LineString where possible.

    Borrowed from the old _to_linestring / _sw_linemerge pattern.
    - Try shapely linemerge (works when parts share endpoints).
    - If parts are genuinely disconnected, keep only the longest.
    This cleans up MultiLineStrings produced by offset_curve() on curved
    roads and by OSMnx edge simplification of separately-mapped footways.
    """
    if not isinstance(geom, MultiLineString):
        return geom
    merged = linemerge(geom)
    if isinstance(merged, LineString):
        return merged
    # Still fragmented — take the longest contiguous part
    parts: list[BaseGeometry] = list(merged.geoms) if isinstance(merged, MultiLineString) else [merged]
    return max(parts, key=lambda g: g.length)


def _find_deflection_split_points(
    coords: list[tuple[float, float]], threshold_rad: float,
) -> list[int]:
    """Return coordinate indices where inter-vertex deflection exceeds threshold."""
    if len(coords) < 3:
        return []
    arr = np.array(coords, dtype=np.float64)
    incoming = arr[1:-1] - arr[:-2]
    outgoing = arr[2:] - arr[1:-1]
    in_len = np.linalg.norm(incoming, axis=1)
    out_len = np.linalg.norm(outgoing, axis=1)
    valid = (in_len >= 1e-9) & (out_len >= 1e-9)
    if not np.any(valid):
        return []
    dot = np.where(valid, np.sum(incoming * outgoing, axis=1) / (in_len * out_len), 1.0)
    dot = np.clip(dot, -1.0, 1.0)
    deflections = np.arccos(dot)
    return (np.flatnonzero(deflections > threshold_rad) + 1).tolist()


def _linestring_bearings_vectorized(geom_series: gpd.GeoSeries) -> pd.Series:
    """Vectorized bearing computation (0-360°) for a UTM GeoSeries of LineStrings."""
    geom_arr = np.asarray(geom_series)
    n = len(geom_arr)
    bearings = np.full(n, np.nan, dtype=np.float64)

    first_pts = shapely.get_point(geom_arr, 0)
    last_pts = shapely.get_point(geom_arr, -1)
    x0, y0 = shapely.get_x(first_pts), shapely.get_y(first_pts)
    x1, y1 = shapely.get_x(last_pts), shapely.get_y(last_pts)
    dx, dy = x1 - x0, y1 - y0
    valid = np.isfinite(x0) & np.isfinite(x1) & ((dx != 0) | (dy != 0))
    bearings[valid] = np.degrees(np.arctan2(dx[valid], dy[valid])) % 360

    # Fallback for MultiLineString
    for i in np.flatnonzero(np.isnan(bearings)):
        g = geom_arr[i]
        if g is None or not isinstance(g, BaseGeometry):
            continue
        coords = _flatten_coords(g)
        if len(coords) < 2:
            continue
        ddx, ddy = coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]
        if ddx == 0 and ddy == 0:
            continue
        bearings[i] = math.degrees(math.atan2(ddx, ddy)) % 360

    return pd.Series(bearings, index=geom_series.index)


def _infer_bearing_from_named_neighbors(
    idx: int,
    raw_bearing: float,
    name_col: pd.Series,
    node_to_edges: dict[Any, list[int]],
    edge_bearings: dict[int, float],
    edges: gpd.GeoDataFrame,
    parallel_threshold_deg: float = 45.0,
) -> float | None:
    """Infer canonical direction for an unnamed two-way segment from adjacent named segments."""
    u = edges.at[idx, "start_node_id"]
    v = edges.at[idx, "end_node_id"]
    flipped = (raw_bearing + 180.0) % 360.0
    raw_axis = raw_bearing % 180.0
    votes_raw = votes_flip = 0

    for node in (u, v):
        for nb_idx in node_to_edges.get(node, []):
            if nb_idx == idx or nb_idx not in edge_bearings:
                continue
            nm = name_col.at[nb_idx]
            if nm is None or (isinstance(nm, float) and math.isnan(nm)) or not str(nm).strip():
                continue
            nb = edge_bearings[nb_idx]
            nb_axis = nb % 180.0
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


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — crosswalk cache
# ═════════════════════════════════════════════════════════════════════════════

_MARKED_TYPES = frozenset({"marked", "zebra", "pelican", "toucan", "puffin", "crosswalk"})
_SIGNAL_TYPES = frozenset({"traffic_signals", "traffic_light", "pelican", "puffin", "toucan"})


def _extract_crossing_node_cache(nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of crossing nodes with schema-aligned attributes."""
    crossing_col = _safe_col(nodes_gdf, "crossing")
    highway_col = _safe_col(nodes_gdf, "highway")
    markings_col = _safe_col(nodes_gdf, "crossing:markings")
    signals_col = _safe_col(nodes_gdf, "crossing:signals")
    island_col = _safe_col(nodes_gdf, "crossing:island")
    continuous_col = _safe_col(nodes_gdf, "crossing:continuous")
    kerb_col = _safe_col(nodes_gdf, "kerb")
    tp_col = _safe_col(nodes_gdf, "tactile_paving")
    tc_col = _safe_col(nodes_gdf, "traffic_calming")

    relevant = (
        crossing_col.notna()
        | (highway_col == "crossing")
        | kerb_col.notna()
        | tp_col.notna()
    )
    if not relevant.any():
        return gpd.GeoDataFrame(columns=["geometry", "type", "controlled", "marked",
                                          "markings", "signals", "island", "kerb",
                                          "tactile_paving", "traffic_calming", "continuous"],
                                 geometry="geometry", crs=nodes_gdf.crs)

    sub = nodes_gdf.loc[relevant].copy()
    idx = sub.index

    def _str_col(col: pd.Series) -> pd.Series:
        """Convert a series to stripped str values; NA → None."""
        s = col.loc[idx]
        mask = s.notna()
        result = pd.Series(None, index=idx, dtype=object)
        result[mask] = s[mask].astype(str).str.strip()
        return result

    c_vals   = _str_col(crossing_col)
    hw_vals  = _str_col(highway_col)
    sig_vals = _str_col(signals_col)
    mark_vals = _str_col(markings_col)
    isl_vals  = _str_col(island_col)
    kerb_vals = _str_col(kerb_col)
    tp_vals   = _str_col(tp_col)
    tc_vals   = _str_col(tc_col)
    cont_vals = _str_col(continuous_col)

    # type: c_val or (hw_val if hw_val == "crossing" else None)
    type_vals = c_vals.where(c_vals.notna(),
                             hw_vals.where(hw_vals == "crossing", other=None))

    # controlled: "yes" if signal type, else None
    controlled_bool = c_vals.isin(_SIGNAL_TYPES) | sig_vals.isin(("yes", "traffic_signals"))
    controlled_vals = controlled_bool.map({True: "yes", False: None})

    # marked: "yes" if marked type or markings tag matches, else None
    marked_bool = c_vals.isin(_MARKED_TYPES) | mark_vals.isin(("yes", "lines", "dots", "dashes"))
    marked_vals = marked_bool.map({True: "yes", False: None})

    result_df = pd.DataFrame({
        "osmid":         idx.astype(int),
        "geometry":      sub["geometry"].values,
        "type":          type_vals.values,
        "controlled":    controlled_vals.values,
        "marked":        marked_vals.values,
        "markings":      mark_vals.values,
        "signals":       sig_vals.values,
        "island":        isl_vals.values,
        "kerb":          kerb_vals.values,
        "tactile_paving": tp_vals.values,
        "traffic_calming": tc_vals.values,
        "continuous":    cont_vals.values,
    })
    return gpd.GeoDataFrame(result_df, geometry="geometry", crs=nodes_gdf.crs)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — traffic calming (step 5)
# ═════════════════════════════════════════════════════════════════════════════

def _query_traffic_calming_nodes(place: str) -> gpd.GeoDataFrame:
    """Query OSM for traffic_calming=* point nodes inside place."""
    gdf = ox.features_from_place(place, tags={"traffic_calming": True})
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    if gdf.empty:
        return gdf
    gdf = gdf.reset_index()
    for c in ("osmid", "traffic_calming"):
        if c not in gdf.columns:
            gdf[c] = pd.NA
    return gdf


def _find_coincident_segment(
    tc_point_utm: Point,
    edges_utm: gpd.GeoDataFrame,
    sindex: Any,
) -> int | None:
    """Return edge index whose vertex is within _TC_COINCIDENCE_THRESHOLD_M of tc_point."""
    buf = tc_point_utm.buffer(_TC_COINCIDENCE_THRESHOLD_M)
    candidates: list[int] = list(sindex.query(buf, predicate="intersects"))
    if not candidates:
        return None
    best_idx, best_dist = None, float("inf")
    tc_x, tc_y = tc_point_utm.x, tc_point_utm.y
    for idx in candidates:
        geom = edges_utm.geometry.iloc[idx]
        coords_arr = shapely.get_coordinates(cast(BaseGeometry, geom))
        dists = np.sqrt((coords_arr[:, 0] - tc_x) ** 2 + (coords_arr[:, 1] - tc_y) ** 2)
        min_d = float(dists.min())
        if min_d <= _TC_COINCIDENCE_THRESHOLD_M and min_d < best_dist:
            best_dist = min_d
            best_idx = idx
    return best_idx


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — elevation (step 6)
# ═════════════════════════════════════════════════════════════════════════════

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in metres between two WGS-84 points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _fetch_usgs_elevation_point(lon: float, lat: float) -> float | None:
    """Query USGS EPQS for elevation at a single WGS-84 point. Returns metres or None."""
    params = urllib.parse.urlencode({"x": lon, "y": lat, "wkid": 4326, "includeDate": "false"})
    url = f"{USGS_EPQS_URL}?{params}"
    for attempt in range(USGS_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=USGS_TIMEOUT) as resp:
                data = json.loads(resp.read())
                elev = data.get("value", -1_000_000)
                if elev is None or float(elev) == -1_000_000:
                    return None
                return round(float(elev), 3)
        except Exception:
            if attempt < USGS_MAX_RETRIES - 1:
                time.sleep(USGS_RETRY_DELAY * (attempt + 1))
    return None


def _fetch_elevations_batch(
    points: dict[int, tuple[float, float]],
) -> dict[int, float | None]:
    """Fetch elevations for {node_id: (lon, lat)} concurrently."""
    results: dict[int, float | None] = {}
    with ThreadPoolExecutor(max_workers=USGS_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_usgs_elevation_point, lon, lat): nid
            for nid, (lon, lat) in points.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Fetching USGS elevations", unit="pt"):
            nid = futures[future]
            try:
                results[nid] = future.result()
            except Exception:
                results[nid] = None
    return results


def _propagate_slopes_short_segments(
    gdf: gpd.GeoDataFrame,
    dem_resolution_m: float,
    max_prop_dist_m: float,
) -> gpd.GeoDataFrame:
    """Fill missing slopes on short segments from directly-connected neighbors with a valid slope."""
    if "_seg_length_m" not in gdf.columns:
        first_pts = shapely.get_point(np.asarray(gdf["street_geometry"]), 0)
        last_pts = shapely.get_point(np.asarray(gdf["street_geometry"]), -1)
        x0, y0 = shapely.get_x(first_pts), shapely.get_y(first_pts)
        x1, y1 = shapely.get_x(last_pts), shapely.get_y(last_pts)
        phi1, phi2 = np.radians(y0), np.radians(y1)
        dphi, dlam = np.radians(y1 - y0), np.radians(x1 - x0)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        gdf["_seg_length_m"] = 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    short_mask = (gdf["_seg_length_m"] < dem_resolution_m) & gdf["street_incline"].isna()
    if not short_mask.any():
        return gdf

    # Build node → slope lookup from all segments that already have a valid slope.
    # Each valid segment contributes its slope to both its start and end node.
    valid = gdf["street_incline"].notna()
    node_slope = pd.concat([
        gdf.loc[valid, ["start_node_id", "street_incline"]].rename(columns={"start_node_id": "node_id"}),
        gdf.loc[valid, ["end_node_id",   "street_incline"]].rename(columns={"end_node_id":   "node_id"}),
    ]).groupby("node_id")["street_incline"].first()

    # For each short-missing segment, inherit slope from start node, then end node.
    short_df = gdf.loc[short_mask, ["start_node_id", "end_node_id"]]
    filled_slope = (
        short_df["start_node_id"].map(node_slope)
        .fillna(short_df["end_node_id"].map(node_slope))
    )

    gdf.loc[short_mask, "street_incline"] = filled_slope.values

    filled = int(filled_slope.notna().sum())
    if filled:
        log.info("  Propagated slopes to %d short segments", filled)
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Step 1 — Create output parquet (empty GDF with full schema)
# ---------------------------------------------------------------------------
def step_01_create_parquet(config: PipelineConfig) -> gpd.GeoDataFrame:
    """Create an empty GeoDataFrame with the full output schema."""
    schema = _build_empty_schema()
    gdf = gpd.GeoDataFrame(
        {col: pd.Series(dtype=object) for col in schema},
        geometry="street_geometry",
        crs="EPSG:4326",
    )
    log.info("Step 1: Created empty parquet schema with %d columns", len(gdf.columns))
    return gdf


# ---------------------------------------------------------------------------
# Step 2 — Load OSM streets into parquet, cache crosswalk data
# ---------------------------------------------------------------------------
def _load_or_cache_graph(place: str, network_type: str, suffix: str = "") -> Any:
    """Download an OSMnx graph or load from pickle cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = place.replace("/", "_").replace(" ", "_")
    cache_file = CACHE_DIR / f"{slug}{suffix}.pkl"
    if cache_file.exists():
        log.info("  Loading cached %s graph from %s", network_type, cache_file)
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    log.info("  Downloading %s graph from OSM…", network_type)
    G = ox.graph_from_place(place, network_type=network_type)
    with open(cache_file, "wb") as f:
        pickle.dump(G, f)
    log.info("  Cached %s graph to %s", network_type, cache_file)
    return G


def step_02_load_streets(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Any, gpd.GeoDataFrame,
           gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Pull street, walk, and bike networks from OSM via OSMnx, populate street columns.
    Returns (gdf, crosswalk_cache, G, nodes_gdf, edges_reset, walk_edges, bike_edges).
    """
    place = config.place_name
    log.info("Step 2: Loading OSM networks for %s", place)

    # ── Configure OSMnx tags ──────────────────────────────────────────────
    ox.settings.useful_tags_way = list(set(
        ox.settings.useful_tags_way + _EXTRA_WAY_TAGS
    ))
    ox.settings.useful_tags_node = list(set(
        ox.settings.useful_tags_node + _CROSSING_NODE_TAGS
    ))

    # ── Download / cache graphs ───────────────────────────────────────────
    G = _load_or_cache_graph(place, config.network_type)
    G_walk = _load_or_cache_graph(place, "walk", suffix="_walk")
    G_bike = _load_or_cache_graph(place, "bike", suffix="_bike")

    # ── Convert to GeoDataFrames ──────────────────────────────────────────
    nodes, edges = ox.graph_to_gdfs(G)
    edges_reset = edges.reset_index()
    log.info("  Street network: %d edges, %d nodes", len(edges_reset), len(nodes))

    _, walk_edges_raw = ox.graph_to_gdfs(G_walk)
    walk_edges = walk_edges_raw.reset_index()
    log.info("  Walk network: %d edges", len(walk_edges))

    _, bike_edges_raw = ox.graph_to_gdfs(G_bike)
    bike_edges = bike_edges_raw.reset_index()
    log.info("  Bike network: %d edges", len(bike_edges))

    # ── Build crosswalk cache from nodes ──────────────────────────────────
    crosswalk_cache = _extract_crossing_node_cache(nodes)
    log.info("  Crosswalk cache: %d crossing nodes", len(crosswalk_cache))

    # ── Build node geometry lookup ────────────────────────────────────────
    node_geom: dict[int, Point] = cast(dict[int, Point], nodes.geometry.to_dict())

    # ── Create populated GDF ──────────────────────────────────────────────
    n = len(edges_reset)

    # Parse maxspeed vectorized — flatten lists, strip unit suffixes, bulk-convert
    maxspeed_raw = _safe_col(edges_reset, "maxspeed")
    ms_series = pd.Series(maxspeed_raw)
    ms_flat = ms_series.apply(lambda v: v[0] if isinstance(v, list) else v)
    ms_str = ms_flat.astype(str).str.strip()
    for _sfx in (" mph", r" km\h", " kph"):
        ms_str = ms_str.str.replace(_sfx, "", regex=False)
    maxspeed_vals = pd.to_numeric(ms_str, errors="coerce").fillna(config.default_maxspeed).astype(np.int64).values

    # Parse lanes — flatten lists, convert to nullable Int64 array (avoids Python list comprehension)
    lanes_raw = _safe_col(edges_reset, "lanes")
    _lanes_flat = lanes_raw.apply(lambda v: v[0] if isinstance(v, list) else v)
    _lanes_numeric = pd.to_numeric(_lanes_flat.astype(str).str.strip(), errors="coerce")
    lanes_vals = pd.array(_lanes_numeric, dtype="Int64")  # NA-safe nullable int

    # Parse lane_width — flatten lists, convert to nullable Float64 array
    lw_raw = _safe_col(edges_reset, "lane_width")
    _lw_flat = lw_raw.apply(lambda v: v[0] if isinstance(v, list) else v)
    _lw_numeric = pd.to_numeric(_lw_flat.astype(str).str.strip(), errors="coerce")
    lw_vals = pd.array(_lw_numeric, dtype="Float64")

    # Stringify osmid (may be int or list of ints from simplification)
    street_ids = edges_reset["osmid"].astype(str).values

    # Scalar columns: OSMnx may return lists for multi-value tags; take first element.
    # surface is intentionally kept as list (schema: Str | List[Str] | null).
    data = {
        "street_id": street_ids,
        "start_node_id": edges_reset["u"].values,
        "end_node_id": edges_reset["v"].values,
        "name":             _scalar_col(_safe_col(edges_reset, "name")).values,
        "highway":          _scalar_col(_safe_col(edges_reset, "highway")).values,
        "oneway":           _scalar_col(_safe_col(edges_reset, "oneway")).values,
        "surface":          _safe_col(edges_reset, "surface").values,        # list OK
        "street_condition": _scalar_col(_safe_col(edges_reset, "smoothness")).values,
        "maxspeed": maxspeed_vals,
        "lanes": lanes_vals,
        "lane_width": lw_vals,
        "street_geometry": edges_reset["geometry"].values,
        "start_node_geometry": edges_reset["u"].map(node_geom).values,
        "end_node_geometry": edges_reset["v"].map(node_geom).values,
    }

    populated = gpd.GeoDataFrame(data, geometry="street_geometry", crs="EPSG:4326")

    # Add all missing schema columns with defaults
    populated = _ensure_cols(populated, _build_empty_schema())

    log.info("Step 2: Populated %d street segments, %d schema columns",
             len(populated), len(populated.columns))

    return populated, crosswalk_cache, G, nodes, edges_reset, walk_edges, bike_edges


# ---------------------------------------------------------------------------
# Step 3 — Split segments at deflection vertices (>45°)
# ---------------------------------------------------------------------------
def step_03_split_deflections(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Split street segments where inter-vertex deflection exceeds threshold.
    Synthetic node IDs are negative integers, reset per segment.

    Optimization: vectorized screening pass filters 184K segments down to ~14K
    candidates (those with >=3 vertices) before any Python-level coord work.
    """
    threshold_rad = math.radians(config.deflection_angle_threshold)

    # Project to UTM for accurate angle measurement
    utm_geom = gdf["street_geometry"].to_crs("EPSG:32610")

    # ── Vectorized screening: skip simple 2-vertex segments ───────────────
    geom_arr = np.asarray(utm_geom)
    n_coords = shapely.get_num_coordinates(geom_arr)
    candidates = np.flatnonzero(n_coords >= 3)
    log.info("  Step 3: %d/%d segments have >=3 vertices, checking deflections",
             len(candidates), len(gdf))

    # ── Deflection detection on candidates only ───────────────────────────
    wgs_geom_arr = np.asarray(gdf["street_geometry"])
    cols = gdf.columns.tolist()
    col_arrays = {c: gdf[c].values for c in cols}

    rows_to_drop: list[int] = []
    new_rows: list[dict[str, Any]] = []

    for idx in candidates:
        g_utm = cast(BaseGeometry, geom_arr[idx])
        if g_utm is None:
            continue
        coords_utm = _flatten_coords(g_utm)
        if len(coords_utm) < 3:
            continue

        split_indices = _find_deflection_split_points(coords_utm, threshold_rad)
        if not split_indices:
            continue

        g_wgs = cast(BaseGeometry, wgs_geom_arr[idx])
        coords_wgs = _flatten_coords(g_wgs)
        if len(coords_wgs) != len(coords_utm):
            continue

        row_data = {c: col_arrays[c][idx] for c in cols}
        original_u = row_data.get("start_node_id")
        original_v = row_data.get("end_node_id")

        boundaries = [0] + split_indices + [len(coords_wgs) - 1]
        rows_to_drop.append(cast(int, gdf.index[idx]))

        seg_synth_counter = 0
        for i in range(len(boundaries) - 1):
            piece = dict(row_data)
            start_b, end_b = boundaries[i], boundaries[i + 1]
            piece["street_geometry"] = LineString(coords_wgs[start_b: end_b + 1])

            if i == 0:
                piece["start_node_id"] = original_u
            else:
                piece["start_node_id"] = seg_synth_counter

            if i == len(boundaries) - 2:
                piece["end_node_id"] = original_v
            else:
                seg_synth_counter -= 1
                piece["end_node_id"] = seg_synth_counter

            new_rows.append(piece)

    if not new_rows:
        log.info("Step 3: 0 segments split (threshold=%.0f deg)", config.deflection_angle_threshold)
        return gdf

    n_split = len(rows_to_drop)
    result = gdf.drop(index=rows_to_drop)
    new_gdf = gpd.GeoDataFrame(new_rows, geometry="street_geometry", crs="EPSG:4326")
    result = gpd.GeoDataFrame(
        pd.concat([result, new_gdf], ignore_index=True),
        geometry="street_geometry",
        crs="EPSG:4326",
    )
    log.info("Step 3: Split %d segments -> %d pieces (threshold=%.0f deg)",
             n_split, len(new_rows), config.deflection_angle_threshold)
    return result


# ---------------------------------------------------------------------------
# Step 4 — Compute normalized bearings, assign grid IDs, flag intersections
# ---------------------------------------------------------------------------
def step_04_bearings_and_grid_ids(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
    G: Any = None,
    nodes: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Compute normalized bearings, assign grid IDs, flag intersection nodes."""
    log.info("Step 4: Computing bearings and grid IDs")

    # ── 1. Raw bearings in UTM ────────────────────────────────────────────
    utm_geom = gdf["street_geometry"].to_crs("EPSG:32610")
    raw_bearings = _linestring_bearings_vectorized(utm_geom)

    # ── 2. Bearing normalization (fully vectorized) ─────────────────────
    oneway_col = _safe_col(gdf, "oneway")
    name_col = _safe_col(gdf, "name")

    has_bearing = raw_bearings.notna()
    norm = raw_bearings.copy()  # will hold final normalized bearings

    # One-way forward: keep as-is
    fwd_mask = oneway_col.isin([True, "yes", "1", 1]) & has_bearing
    # One-way reverse: flip 180
    rev_mask = oneway_col.isin(["-1", "reverse"]) & has_bearing
    norm[rev_mask] = (raw_bearings[rev_mask] + 180.0) % 360
    assigned = fwd_mask | rev_mask

    # Two-way segments: majority vote by street name
    twoway_mask = has_bearing & ~assigned
    name_s = name_col.where(twoway_mask, other=None)
    # Clean names — convert to string, strip, replace empty with NA
    name_clean = name_s.astype(str).str.strip().replace(["", "nan", "None", "<NA>"], pd.NA)  # type: ignore[arg-type]
    has_name = twoway_mask & name_clean.notna()
    unnamed_mask = twoway_mask & ~has_name

    # Vectorized majority vote per name group
    if has_name.any():
        named_df = pd.DataFrame({
            "name": name_clean[has_name],
            "bearing": raw_bearings[has_name],
        })
        # Compute north_majority per name group
        named_df["is_north"] = named_df["bearing"] < 180.0
        group_north = named_df.groupby("name")["is_north"].transform("mean")
        # If majority northward (>=50%), flip southward; else flip northward
        flip_south = (group_north >= 0.5) & (~named_df["is_north"])
        flip_north = (group_north < 0.5) & named_df["is_north"]
        named_df["norm"] = named_df["bearing"]
        named_df.loc[flip_south | flip_north, "norm"] = (
            named_df.loc[flip_south | flip_north, "bearing"] + 180.0
        ) % 360
        norm[named_df.index] = named_df["norm"]
        assigned = assigned | has_name

    # Unnamed two-way: northward fallback (flip bearings >= 180)
    if unnamed_mask.any():
        raw_unnamed = raw_bearings[unnamed_mask]
        norm[unnamed_mask] = np.where(
            raw_unnamed >= 180.0, (raw_unnamed + 180.0) % 360, raw_unnamed,
        )
        assigned = assigned | unnamed_mask

    gdf["normalized_bearing"] = norm.fillna(0.0)

    # ── 3. Grid ID assignment (UTM) ───────────────────────────────────────
    if nodes is not None:
        nodes_utm = nodes.to_crs("EPSG:32610")
        xs = np.asarray(nodes_utm.geometry.x, dtype=np.float64)
        ys = np.asarray(nodes_utm.geometry.y, dtype=np.float64)
    else:
        centroids_utm = gdf["street_geometry"].to_crs("EPSG:32610").centroid
        xs = np.asarray(centroids_utm.x, dtype=np.float64)
        ys = np.asarray(centroids_utm.y, dtype=np.float64)

    cell = config.grid_cell_size
    origin_x, origin_y = float(xs.min()), float(ys.min())

    # Store grid origin for parquet metadata (used by RollTracks for viewport-based loading)
    gdf.attrs["grid_origin_x"] = origin_x
    gdf.attrs["grid_origin_y"] = origin_y
    gdf.attrs["grid_cell_size"] = cell

    # Compute centroid grid cells for each segment
    seg_centroids_utm = utm_geom.centroid
    cx = np.asarray(seg_centroids_utm.x, dtype=np.float64)
    cy = np.asarray(seg_centroids_utm.y, dtype=np.float64)
    col_arr = np.floor((cx - origin_x) / cell).astype(int)
    row_arr = np.floor((cy - origin_y) / cell).astype(int)

    # Per-cell sequence numbers via sort
    n = len(col_arr)
    sort_idx = np.lexsort((row_arr, col_arr))
    sc, sr = col_arr[sort_idx], row_arr[sort_idx]
    is_new = np.ones(n, dtype=bool)
    is_new[1:] = (sc[1:] != sc[:-1]) | (sr[1:] != sr[:-1])
    group_starts = np.where(is_new)[0]
    gid = np.cumsum(is_new, dtype=np.int32) - 1
    seqs_sorted = np.arange(n, dtype=np.int32) - group_starts[gid].astype(np.int32)
    seqs = np.empty(n, dtype=np.int32)
    seqs[sort_idx] = seqs_sorted

    grid_ids = np.char.add(
        np.char.add(col_arr.astype(str), "_"),
        np.char.add(row_arr.astype(str), np.char.add("_", seqs.astype(str))),
    )
    gdf["street_grid_id"] = grid_ids.tolist()

    # ── 4. Intersection node detection ────────────────────────────────────
    if G is not None:
        node_neighbors: dict[int, set[int]] = defaultdict(set)
        for u, v, data in G.edges(data=True):
            hw = data.get("highway", "")
            hw_vals = hw if isinstance(hw, list) else [hw]
            if frozenset(str(h) for h in hw_vals) & _NON_ROAD_HIGHWAY:
                continue
            node_neighbors[u].add(v)
            node_neighbors[v].add(u)
        intersection_nodes = {nid for nid, nbrs in node_neighbors.items() if len(nbrs) >= 3}
    else:
        # Fallback: count from GDF
        all_nodes = pd.concat([gdf["start_node_id"], gdf["end_node_id"]])
        counts = all_nodes.value_counts()
        intersection_nodes = set(counts[counts >= 3].index)

    gdf["start_node_is_intersection_node"] = gdf["start_node_id"].isin(intersection_nodes)
    gdf["end_node_is_intersection_node"] = gdf["end_node_id"].isin(intersection_nodes)

    n_isect = len(intersection_nodes)
    log.info("Step 4: %d segments, %d grid cells, %d intersection nodes",
             len(gdf), len(set(grid_ids.tolist())), n_isect)
    return gdf


# ---------------------------------------------------------------------------
# Step 5 — Populate traffic calming features
# ---------------------------------------------------------------------------
def step_05_traffic_calming(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Populate traffic calming features into street features."""
    log.info("Step 5: Querying traffic calming nodes…")
    tc_gdf = _query_traffic_calming_nodes(config.place_name)
    if tc_gdf.empty:
        log.info("  No traffic calming nodes found")
        return gdf

    # Project both to UTM for distance matching
    tc_utm = tc_gdf.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:32610")
    edges_utm = gpd.GeoDataFrame(
        geometry=gdf["street_geometry"].to_crs("EPSG:32610"),
    )
    sindex = edges_utm.geometry.sindex

    log.info("  %d traffic calming nodes found", len(tc_utm))

    # Accumulate per-segment features
    seg_features: dict[int, list[tuple[str, Point, dict[str, Any]]]] = {}
    _skip = frozenset({"geometry", "osmid", "element_type"})
    _attr_cols = [c for c in tc_gdf.columns if c not in _skip]

    matched = 0
    for pos in range(len(tc_utm)):
        pt_utm = cast(Point, tc_utm.geometry.iloc[pos])
        tc_type = str(tc_gdf["traffic_calming"].iloc[pos])
        seg_idx = _find_coincident_segment(pt_utm, edges_utm, sindex)
        if seg_idx is None:
            continue
        matched += 1
        attrs = {c: tc_gdf[c].iloc[pos] for c in _attr_cols if pd.notna(tc_gdf[c].iloc[pos])}
        # Store the WGS84 point, not UTM
        pt_wgs = cast(Point, tc_gdf.geometry.iloc[pos])
        seg_features.setdefault(seg_idx, []).append(
            (f"traffic_calming:{tc_type}", pt_wgs, attrs)
        )

    log.info("  Matched %d/%d nodes to %d segments", matched, len(tc_utm), len(seg_features))

    # Ensure feature columns exist
    for fc in ("street_feature_types", "public_data_id_street_feature",
               "street_feature_geometry", "street_feature_geometry_projected",
               "street_feature_attributes"):
        if fc not in gdf.columns:
            gdf[fc] = None

    # Write into GDF
    for seg_idx, features in seg_features.items():
        types = [f[0] for f in features]
        points = [f[1] for f in features]
        attrs_list = [f[2] for f in features]

        # Project each point onto the street LineString
        street_geom = cast(LineString, gdf.at[seg_idx, "street_geometry"])
        pts_arr = np.asarray(points)
        projected = list(shapely.line_interpolate_point(
            street_geom, shapely.line_locate_point(street_geom, pts_arr),
        ))

        gdf.at[seg_idx, "street_feature_types"] = types  # type: ignore[call-overload]
        gdf.at[seg_idx, "public_data_id_street_feature"] = [None] * len(features)  # type: ignore[call-overload]
        gdf.at[seg_idx, "street_feature_geometry"] = MultiPoint(points)  # type: ignore[call-overload]
        gdf.at[seg_idx, "street_feature_geometry_projected"] = MultiPoint(projected)  # type: ignore[call-overload]
        gdf.at[seg_idx, "street_feature_attributes"] = attrs_list  # type: ignore[call-overload]

    log.info("Step 5: Traffic calming complete")
    return gdf


# ---------------------------------------------------------------------------
# Step 6 — Enrich with USGS elevation / incline
# ---------------------------------------------------------------------------
def step_06_usgs_elevation(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Enrich street geometries with topography data from USGS 3DEP."""
    if not config.enable_elevation:
        log.info("Step 6: Elevation enrichment disabled")
        return gdf
    log.info("Step 6: Enriching with USGS elevation data")

    # ── Extract unique endpoint coordinates (WGS84) ───────────────────────
    geom_arr = np.asarray(gdf["street_geometry"])
    first_pts = shapely.get_point(geom_arr, 0)
    last_pts = shapely.get_point(geom_arr, -1)

    # Collect unique REAL node → (lon, lat); skip synthetic split nodes (negative IDs).
    # Synthetic nodes inherit elevation by interpolation below.
    _start_ids = gdf["start_node_id"].to_numpy()
    _end_ids   = gdf["end_node_id"].to_numpy()
    _xs_start  = shapely.get_x(first_pts).astype(np.float64)
    _ys_start  = shapely.get_y(first_pts).astype(np.float64)
    _xs_end    = shapely.get_x(last_pts).astype(np.float64)
    _ys_end    = shapely.get_y(last_pts).astype(np.float64)
    _sr = _start_ids > 0
    _er = _end_ids > 0
    _node_df = pd.concat([
        pd.DataFrame({"node_id": _start_ids[_sr], "x": _xs_start[_sr], "y": _ys_start[_sr]}),
        pd.DataFrame({"node_id": _end_ids[_er],   "x": _xs_end[_er],   "y": _ys_end[_er]}),
    ]).dropna(subset=["x", "y"]).drop_duplicates("node_id")
    unique_points: dict[int, tuple[float, float]] = {
        int(nid): (float(x), float(y))
        for nid, x, y in zip(_node_df["node_id"], _node_df["x"], _node_df["y"])
    }

    log.info("  %d unique nodes to query", len(unique_points))

    # ── Check for cached elevations ───────────────────────────────────────
    place_slug = config.place_name.replace(" ", "_").replace(",", "").replace("/", "_")
    elev_cache = CACHE_DIR / f"{place_slug}_elevation.pkl"
    if elev_cache.exists():
        log.info("  Loading cached elevations from %s", elev_cache)
        with open(elev_cache, "rb") as f:
            elevations = pickle.load(f)
        # Query any missing nodes
        missing = {k: v for k, v in unique_points.items() if k not in elevations}
        if missing:
            log.info("  Fetching %d new elevation points", len(missing))
            new_elevs = _fetch_elevations_batch(missing)
            elevations.update(new_elevs)
            with open(elev_cache, "wb") as f:
                pickle.dump(elevations, f)
    else:
        elevations = _fetch_elevations_batch(unique_points)
        with open(elev_cache, "wb") as f:
            pickle.dump(elevations, f)
        log.info("  Cached elevations to %s", elev_cache)

    # ── Map elevations to segments ────────────────────────────────────────
    # Synthetic split nodes (negative IDs) are not in the USGS cache.
    # Approximate their elevation as the midpoint between their two OSM parent endpoints.
    # We do a two-pass map: first real nodes, then fill synthetics from their neighbours.
    elev_start = gdf["start_node_id"].map(elevations).astype("Float64")
    elev_end   = gdf["end_node_id"].map(elevations).astype("Float64")

    # For rows whose start is synthetic, the previous segment ends at the same point
    # (same coordinate), so its elev_end should match.  Average the two known neighbours.
    syn_start = gdf["start_node_id"] < 0
    syn_end   = gdf["end_node_id"]   < 0
    if syn_start.any() or syn_end.any():
        # Build a coord→elevation lookup from all real-node rows
        xs = shapely.get_x(first_pts).astype(np.float64)
        ys = shapely.get_y(first_pts).astype(np.float64)
        xe = shapely.get_x(last_pts).astype(np.float64)
        ye = shapely.get_y(last_pts).astype(np.float64)
        # Build coord→elevation from real nodes (vectorized mask + zip)
        _syn_s = syn_start.values
        _syn_e = syn_end.values
        _es = elev_start.values
        _ee = elev_end.values
        _real_s = ~_syn_s & ~np.isnan(_es.astype(np.float64))
        _real_e = ~_syn_e & ~np.isnan(_ee.astype(np.float64))
        coord_elev: dict[tuple[float, float], float] = {}
        coord_elev.update(zip(
            zip(np.round(xs[_real_s], 7), np.round(ys[_real_s], 7)),
            _es[_real_s].astype(np.float64),
        ))
        coord_elev.update(zip(
            zip(np.round(xe[_real_e], 7), np.round(ye[_real_e], 7)),
            _ee[_real_e].astype(np.float64),
        ))
        # Fill synthetic nodes from coord lookup (vectorized assignment)
        _fill_s = _syn_s & np.isnan(_es.astype(np.float64))
        if _fill_s.any():
            _keys_s = list(zip(np.round(xs[_fill_s], 7), np.round(ys[_fill_s], 7)))
            _vals_s = pd.array([coord_elev.get(k) for k in _keys_s], dtype="Float64")
            _idxs_s = gdf.index[_fill_s]
            _valid_s = pd.notna(_vals_s)
            if _valid_s.any():
                elev_start.loc[_idxs_s[_valid_s]] = _vals_s[_valid_s]
        _fill_e = _syn_e & np.isnan(_ee.astype(np.float64))
        if _fill_e.any():
            _keys_e = list(zip(np.round(xe[_fill_e], 7), np.round(ye[_fill_e], 7)))
            _vals_e = pd.array([coord_elev.get(k) for k in _keys_e], dtype="Float64")
            _idxs_e = gdf.index[_fill_e]
            _valid_e = pd.notna(_vals_e)
            if _valid_e.any():
                elev_end.loc[_idxs_e[_valid_e]] = _vals_e[_valid_e]

    gdf["elev_start_m"] = elev_start
    gdf["elev_end_m"] = elev_end

    # ── Compute haversine segment lengths ─────────────────────────────────
    x0 = shapely.get_x(first_pts).astype(np.float64)
    y0 = shapely.get_y(first_pts).astype(np.float64)
    x1 = shapely.get_x(last_pts).astype(np.float64)
    y1 = shapely.get_y(last_pts).astype(np.float64)

    phi1, phi2 = np.radians(y0), np.radians(y1)
    dphi = np.radians(y1 - y0)
    dlam = np.radians(x1 - x0)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    seg_lengths = 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    gdf["_seg_length_m"] = seg_lengths

    # ── Compute raw slopes ────────────────────────────────────────────────
    has_both = elev_start.notna() & elev_end.notna()
    long_enough = seg_lengths >= config.dem_resolution_m
    valid = has_both & long_enough & np.isfinite(seg_lengths)

    raw_slope = pd.Series(np.nan, index=gdf.index, dtype=np.float64)
    raw_slope[valid] = (
        (elev_end[valid].to_numpy() - elev_start[valid].to_numpy())
        / seg_lengths[valid] * 100
    )

    # Cap at ±slope_cap_pct
    cap = config.slope_cap_pct
    raw_slope = raw_slope.where(raw_slope.abs() <= cap, other=np.nan)  # type: ignore[operator]

    gdf["street_incline"] = raw_slope

    n_valid = int(raw_slope.notna().sum())
    log.info("  Computed slopes for %d/%d segments", n_valid, len(gdf))

    # ── Propagate to short segments ───────────────────────────────────────
    gdf = _propagate_slopes_short_segments(gdf, config.dem_resolution_m, config.max_propagation_dist_m)

    # Drop internal helper column
    gdf.drop(columns=["_seg_length_m"], inplace=True, errors="ignore")

    n_final = int(gdf["street_incline"].notna().sum())
    log.info("Step 6: Elevation complete — %d/%d segments have incline", n_final, len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Step 7 — Populate bikeway and sidewalk tags from OSM
# ---------------------------------------------------------------------------
def step_07_populate_facilities(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
    edges_reset: gpd.GeoDataFrame | None = None,
    walk_edges: gpd.GeoDataFrame | None = None,
    bike_edges: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Populate bikeway and sidewalk presence/attribute columns from inline OSM tags.

    Uses cascade precedence: side-specific tag > :both tag > bare tag.
    Walk/bike edge GeoDataFrames carry the pedestrian and cycling network topology
    from OSM; stored on the GDF as attrs for use by downstream neighbor computation.
    """
    if edges_reset is None:
        log.warning("Step 7: No edges_reset provided, skipping facility population")
        return gdf
    log.info("Step 7: Populating sidewalk and bikeway tags")

    # Store walk/bike topology for downstream neighbor computation (step 11b)
    if walk_edges is not None:
        gdf.attrs["_walk_edges"] = walk_edges
        log.info("  Walk topology stored: %d edges", len(walk_edges))
    if bike_edges is not None:
        gdf.attrs["_bike_edges"] = bike_edges
        log.info("  Bike topology stored: %d edges", len(bike_edges))

    er = edges_reset  # alias

    # ── Sidewalks ─────────────────────────────────────────────────────────
    gdf["sidewalk_left_presence"] = _coalesce_tags(er, "sidewalk:left", "sidewalk:both", "sidewalk")
    gdf["sidewalk_right_presence"] = _coalesce_tags(er, "sidewalk:right", "sidewalk:both", "sidewalk")
    gdf["sidewalk_left_surface"] = _safe_col(er, "sidewalk:left:surface")
    gdf["sidewalk_right_surface"] = _safe_col(er, "sidewalk:right:surface")
    gdf["sidewalk_left_condition"] = _safe_col(er, "sidewalk:left:smoothness")
    gdf["sidewalk_right_condition"] = _safe_col(er, "sidewalk:right:smoothness")
    gdf["sidewalk_left_seperator"] = _safe_col(er, "sidewalk:left:buffer")
    gdf["sidewalk_right_seperator"] = _safe_col(er, "sidewalk:right:buffer")

    raw_slw = _safe_col(er, "sidewalk:left:width")
    raw_srw = _safe_col(er, "sidewalk:right:width")
    gdf["sidewalk_left_width"] = _parse_float_tag_series(raw_slw) if raw_slw.notna().any() else pd.NA
    gdf["sidewalk_right_width"] = _parse_float_tag_series(raw_srw) if raw_srw.notna().any() else pd.NA
    raw_sli = _safe_col(er, "sidewalk:left:incline")
    raw_sri = _safe_col(er, "sidewalk:right:incline")
    gdf["sidewalk_left_incline"] = _parse_incline_series(raw_sli) if raw_sli.notna().any() else pd.NA
    gdf["sidewalk_right_incline"] = _parse_incline_series(raw_sri) if raw_sri.notna().any() else pd.NA

    # ── Bikeways slot 1 ──────────────────────────────────────────────────
    gdf["bikeway_left_1_type"] = _coalesce_tags(er, "cycleway:left", "cycleway:both", "cycleway")
    gdf["bikeway_right_1_type"] = _coalesce_tags(er, "cycleway:right", "cycleway:both", "cycleway")
    gdf["bikeway_left_1_surface"] = _coalesce_tags(er, "cycleway:left:surface", "cycleway:surface")
    gdf["bikeway_right_1_surface"] = _coalesce_tags(er, "cycleway:right:surface", "cycleway:surface")
    gdf["bikeway_left_1_condition"] = _coalesce_tags(er, "cycleway:left:smoothness", "cycleway:smoothness")
    gdf["bikeway_right_1_condition"] = _coalesce_tags(er, "cycleway:right:smoothness", "cycleway:smoothness")
    gdf["bikeway_left_1_width"] = _parse_float_tag_series(
        _coalesce_tags(er, "cycleway:left:width", "cycleway:width"))
    gdf["bikeway_right_1_width"] = _parse_float_tag_series(
        _coalesce_tags(er, "cycleway:right:width", "cycleway:width"))
    gdf["bikeway_left_1_seperator"] = _coalesce_tags(er, "cycleway:left:buffer", "cycleway:buffer")
    gdf["bikeway_right_1_seperator"] = _coalesce_tags(er, "cycleway:right:buffer", "cycleway:buffer")
    gdf["bikeway_left_1_permitted"] = _safe_col(er, "bicycle")
    gdf["bikeway_right_1_permitted"] = _safe_col(er, "bicycle")

    raw_incline = _safe_col(er, "incline")
    parsed_incline = _parse_incline_series(raw_incline) if raw_incline.notna().any() else None
    if parsed_incline is not None:
        gdf["bikeway_left_1_incline"] = parsed_incline
        gdf["bikeway_right_1_incline"] = parsed_incline

    # ── Bikeways slot 2 ──────────────────────────────────────────────────
    gdf["bikeway_left_2_type"] = _coalesce_tags(er, "cycleway:left:2", "cycleway:both:2")
    gdf["bikeway_right_2_type"] = _coalesce_tags(er, "cycleway:right:2", "cycleway:both:2")
    gdf["bikeway_left_2_surface"] = _safe_col(er, "cycleway:left:2:surface")
    gdf["bikeway_right_2_surface"] = _safe_col(er, "cycleway:right:2:surface")
    gdf["bikeway_left_2_condition"] = _safe_col(er, "cycleway:left:2:smoothness")
    gdf["bikeway_right_2_condition"] = _safe_col(er, "cycleway:right:2:smoothness")
    raw_bl2w = _safe_col(er, "cycleway:left:2:width")
    raw_br2w = _safe_col(er, "cycleway:right:2:width")
    gdf["bikeway_left_2_width"] = _parse_float_tag_series(raw_bl2w) if raw_bl2w.notna().any() else pd.NA
    gdf["bikeway_right_2_width"] = _parse_float_tag_series(raw_br2w) if raw_br2w.notna().any() else pd.NA
    gdf["bikeway_left_2_seperator"] = _safe_col(er, "cycleway:left:2:buffer")
    gdf["bikeway_right_2_seperator"] = _safe_col(er, "cycleway:right:2:buffer")
    gdf["bikeway_left_2_permitted"] = _safe_col(er, "bicycle")
    gdf["bikeway_right_2_permitted"] = _safe_col(er, "bicycle")
    if parsed_incline is not None:
        gdf["bikeway_left_2_incline"] = parsed_incline
        gdf["bikeway_right_2_incline"] = parsed_incline

    # Stats
    sw_l = gdf["sidewalk_left_presence"].notna().sum()
    sw_r = gdf["sidewalk_right_presence"].notna().sum()
    bk_l = gdf["bikeway_left_1_type"].notna().sum()
    bk_r = gdf["bikeway_right_1_type"].notna().sum()
    log.info("Step 7: sidewalk L=%d R=%d, bikeway L=%d R=%d", sw_l, sw_r, bk_l, bk_r)
    return gdf


# ---------------------------------------------------------------------------
# Step 8 — Swap left/right on reverse-bearing edges
# ---------------------------------------------------------------------------
def step_08_swap_left_right(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Swap left/right facility columns on edges whose geometry is stored
    in reverse order relative to the normalized bearing.

    When normalized_bearing and raw geometry bearing differ by ~180 deg
    (within 30 deg tolerance), OSM tags were authored relative to the
    reversed geometry and need left<->right remapping.
    """
    log.info("Step 8: Checking left/right swap")

    # Recompute raw bearings from current geometry
    utm_geom = gdf["street_geometry"].to_crs("EPSG:32610")
    raw_bear = _linestring_bearings_vectorized(utm_geom)
    norm_bear = pd.to_numeric(gdf["normalized_bearing"], errors="coerce")

    both_valid = norm_bear.notna() & raw_bear.notna()
    diff = (norm_bear - raw_bear).abs() % 360
    diff_sym = diff.where(diff <= 180, 360 - diff)
    is_reversed = (diff_sym >= 150) & both_valid

    n_swapped = int(is_reversed.sum())
    if n_swapped == 0:
        log.info("  No edges need left/right swap")
        return gdf

    # Auto-detect all left<->right column pairs
    cols = set(gdf.columns)
    swap_pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for col in gdf.columns:
        if "_left_" in col:
            right_col = col.replace("_left_", "_right_", 1)
        elif col.endswith("_left"):
            right_col = col[:-len("_left")] + "_right"
        else:
            continue
        if right_col in cols and right_col not in seen:
            swap_pairs.append((col, right_col))
            seen.add(right_col)

    # Vectorized swap
    for left_col, right_col in swap_pairs:
        tmp = gdf.loc[is_reversed, left_col].copy()
        gdf.loc[is_reversed, left_col] = gdf.loc[is_reversed, right_col].values
        gdf.loc[is_reversed, right_col] = tmp.values

    log.info("Step 8: Swapped %d edges across %d column pairs", n_swapped, len(swap_pairs))
    return gdf


# ---------------------------------------------------------------------------
# Step 9 — Match separately-mapped cycleways and footways
# ---------------------------------------------------------------------------
_BIKEWAY_HW = frozenset({"cycleway", "path", "bridleway"})
_FOOTWAY_HW = frozenset({"footway", "pedestrian", "path", "steps", "corridor"})
_ROAD_HW = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "service", "unclassified", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
})
_MATCH_RADIUS_M = 30.0
_PARALLEL_THRESHOLD_DEG = 45.0
_DEDUP_BUFFER_M = 1.0        # tight buffer for coincidence check
_DEDUP_COVERAGE_THRESHOLD = 0.90  # fraction of road length that must be covered


def step_09_match_separate_facilities(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """
    Match separately-mapped cycleway and footway geometries to parent road slots.
    Then generate offset geometries for any facility with presence/type data but
    no separate geometry.
    """
    log.info("Step 9: Matching separate facilities and generating offsets")

    hw = gdf["highway"].astype(str).str.lower().str.strip()
    bicycle_col = _safe_col(gdf, "bikeway_left_1_permitted").astype(str).str.lower()

    # Classify rows
    is_road = hw.isin(_ROAD_HW)
    is_cycleway = hw.isin(_BIKEWAY_HW) | (
        hw.isin({"path", "footway"}) & bicycle_col.isin({"designated", "yes"})
    )
    is_footway = hw.isin(_FOOTWAY_HW) & ~is_cycleway

    n_roads = int(is_road.sum())
    n_cycle = int(is_cycleway.sum())
    n_foot = int(is_footway.sum())
    log.info("  Roads=%d, cycleways=%d, footways=%d", n_roads, n_cycle, n_foot)

    if n_roads == 0 or (n_cycle + n_foot) == 0:
        log.info("  Nothing to match, proceeding to offset generation")
        return _generate_offset_geometries(gdf, config)

    # Project to UTM for spatial matching
    utm_street = gdf["street_geometry"].to_crs("EPSG:32610")
    road_idx = gdf.index[is_road].tolist()
    road_geoms_utm = utm_street[is_road]
    road_sindex = road_geoms_utm.sindex

    # Precompute road bearings for parallelism check
    road_bearings = gdf.loc[is_road, "normalized_bearing"].to_dict()
    road_names = gdf.loc[is_road, "name"].to_dict()

    # Track which slots are occupied
    foot_used: set[tuple[int, str]] = set()  # (road_idx, side)
    bike_used: set[tuple[int, str, str]] = set()  # (road_idx, side, slot)
    matched_cycle = matched_foot = 0
    duplicate_to_centerline: dict[int, int] = {}  # dup_road_idx → true_centerline_idx
    coincident_no_road: set[int] = set()  # dup_road_idx with no true centerline to map to

    fac_indices = gdf.index[is_cycleway | is_footway]
    for fac_idx in tqdm(fac_indices, desc="Matching facilities", unit="seg"):
        fac_geom = utm_street.at[fac_idx]
        if fac_geom is None or fac_geom.is_empty:
            continue
        fac_type = "bike" if is_cycleway.at[fac_idx] else "foot"

        # Find candidate roads within radius
        buf = fac_geom.buffer(_MATCH_RADIUS_M)
        candidates = list(road_sindex.query(buf, predicate="intersects"))
        if not candidates:
            continue

        # Compute facility bearing in UTM
        fc = _flatten_coords(fac_geom)
        if len(fc) < 2:
            continue
        fac_bear = math.degrees(math.atan2(fc[-1][0] - fc[0][0], fc[-1][1] - fc[0][1])) % 360

        # For footways, partition candidates into coincident (mistagged road rows whose
        # geometry nearly traces the footway) and true road candidates. Scoring runs only
        # on true candidates so a mistagged duplicate is never selected as the parent road.
        if fac_type == "foot":
            tight_buf = fac_geom.buffer(_DEDUP_BUFFER_M)
            cand_geoms = road_geoms_utm.iloc[candidates].values
            road_lens = _shapely_length(cand_geoms)
            int_lens = _shapely_length(_shapely_intersection(cand_geoms, tight_buf))
            coverages = np.where(road_lens > 0, int_lens / road_lens, 0.0)
            coincident_mask = coverages > _DEDUP_COVERAGE_THRESHOLD
            coincident_positions = [candidates[i] for i, m in enumerate(coincident_mask) if m]
            score_positions = [candidates[i] for i, m in enumerate(coincident_mask) if not m]
        else:
            coincident_positions = []
            score_positions = candidates

        def _compute_side(road_geom: Any, fac_geom: Any) -> str:
            fac_mid = fac_geom.interpolate(0.5, normalized=True)
            proj_dist = road_geom.project(fac_mid)
            proj_pt = road_geom.interpolate(proj_dist)
            rc = list(road_geom.coords)
            seg_start_i = max(0, min(int(proj_dist / road_geom.length * (len(rc) - 1)), len(rc) - 2))
            rx = rc[seg_start_i + 1][0] - rc[seg_start_i][0]
            ry = rc[seg_start_i + 1][1] - rc[seg_start_i][1]
            fx, fy = fac_mid.x - proj_pt.x, fac_mid.y - proj_pt.y
            return "left" if (rx * fy - ry * fx) > 0 else "right"

        # Name match takes priority: if any candidate shares the footway name,
        # pick the closest among them immediately without bearing/distance scoring.
        best_road, best_side = None, "left"
        fac_name = str(gdf.at[fac_idx, "name"]).strip() if gdf.at[fac_idx, "name"] else ""
        if fac_name:
            name_matches = [
                (road_idx[p], road_geoms_utm.iloc[p])
                for p in score_positions
                if str(road_names.get(road_idx[p], "") or "").strip() == fac_name
            ]
            if name_matches:
                best_cand_idx, best_geom = min(name_matches, key=lambda t: fac_geom.distance(t[1]))
                best_road = best_cand_idx
                best_side = _compute_side(best_geom, fac_geom)

        # Fallback: score by bearing parallelism + proximity
        if best_road is None:
            best_score = float("inf")
            for cand_pos in score_positions:
                cand_idx = road_idx[cand_pos]
                road_geom = road_geoms_utm.iloc[cand_pos]

                rb = road_bearings.get(cand_idx, 0.0)
                axis_diff = abs((fac_bear % 180) - (rb % 180))
                if axis_diff > 90:
                    axis_diff = 180 - axis_diff
                if axis_diff > _PARALLEL_THRESHOLD_DEG:
                    continue

                dist = fac_geom.distance(road_geom)
                if dist < best_score:
                    best_score = dist
                    best_road = cand_idx
                    best_side = _compute_side(road_geom, fac_geom)

        if best_road is None:
            # No true road candidate — coincident rows are still dropped (footway is
            # the correct geometry; no centerline exists to consolidate attributes into)
            for cand_pos in coincident_positions:
                coincident_no_road.add(road_idx[cand_pos])
            continue

        # Record coincident road rows as duplicates of the true centerline
        for cand_pos in coincident_positions:
            dup_idx = road_idx[cand_pos]
            if dup_idx not in duplicate_to_centerline:
                duplicate_to_centerline[dup_idx] = best_road

        # If best_road was already flagged as a duplicate by a prior footway, redirect
        # the slot assignment to its paired centerline so the geometry isn't lost on drop.
        assign_road = duplicate_to_centerline.get(best_road, best_road)

        # Assign to slot
        fac_wgs_geom = gdf.at[fac_idx, "street_geometry"]
        if fac_type == "foot":
            if (assign_road, best_side) not in foot_used:
                gdf.at[assign_road, f"sidewalk_{best_side}_geometry"] = fac_wgs_geom  # type: ignore[call-overload]
                gdf.at[assign_road, f"sidewalk_{best_side}_quality"] = "separate"
                gdf.at[assign_road, f"sidewalk_{best_side}_offset"] = "no"
                foot_used.add((assign_road, best_side))
                matched_foot += 1
        else:
            for slot in ("1", "2"):
                if (assign_road, best_side, slot) not in bike_used:
                    gdf.at[assign_road, f"bikeway_{best_side}_{slot}_geometry"] = fac_wgs_geom  # type: ignore[call-overload]
                    gdf.at[assign_road, f"bikeway_{best_side}_{slot}_quality"] = "separate"
                    gdf.at[assign_road, f"bikeway_{best_side}_{slot}_offset"] = "no"
                    bike_used.add((assign_road, best_side, slot))
                    matched_cycle += 1
                    break

    log.info("  Matched %d cycleways, %d footways to road slots", matched_cycle, matched_foot)

    # Consolidate sidewalk attributes from mistagged duplicate road rows into their
    # true centerlines (fill-null only), then drop the duplicates.
    if duplicate_to_centerline:
        _sw_consolidate_cols = [
            f"sidewalk_{side}_{attr}"
            for side in ("left", "right")
            for attr in ("presence", "surface", "condition", "width", "incline", "seperator", "geometry")
        ] + [
            f"bikeway_{side}_{slot}_{attr}"
            for side in ("left", "right")
            for slot in ("1", "2")
            for attr in ("type", "surface", "condition", "quality", "permitted",
                         "width", "incline", "seperator", "offset", "geometry")
        ]
        existing_cols = [c for c in _sw_consolidate_cols if c in gdf.columns]
        dup_indices = list(duplicate_to_centerline.keys())
        dup_df = gdf.loc[dup_indices, existing_cols].copy()
        dup_df["_center"] = [duplicate_to_centerline[i] for i in dup_indices]
        # For centers with multiple duplicates, take the first non-null value per column
        fill_vals = dup_df.groupby("_center")[existing_cols].first()
        for col in existing_cols:
            col_fills = fill_vals[col].dropna()
            if col_fills.empty:
                continue
            center_null_mask = pd.isna(gdf.loc[col_fills.index, col])
            to_update = col_fills.index[center_null_mask]
            if not to_update.empty:
                gdf.loc[to_update, col] = col_fills[to_update]  # type: ignore[call-overload]
        gdf = gdf.drop(index=dup_indices)
        log.info("  Removed %d mistagged street rows coinciding with sidewalk segments",
                 len(duplicate_to_centerline))

    if coincident_no_road:
        # Drop coincident road rows that had no true parallel centerline
        orphan_drop = list(coincident_no_road - set(duplicate_to_centerline.keys()))
        if orphan_drop:
            gdf = gdf.drop(index=orphan_drop)
            log.info("  Removed %d coincident road rows with no true centerline", len(orphan_drop))

    # Generate offset geometries for remaining facilities
    gdf = _generate_offset_geometries(gdf, config)
    return gdf


def _generate_offset_geometries(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Generate perpendicular offset geometries for facilities with presence data
    but no separate geometry. Offset = (lanes * lane_width) / 2, plus bikeway_width
    for sidewalks. Uses batch UTM->WGS84 conversion for performance.
    """
    _ABSENT = {"no", "none", "nan", ""}
    utm_geom = gdf["street_geometry"].to_crs("EPSG:32610")
    generated = 0

    # Precompute base offsets vectorized
    lanes_arr = pd.to_numeric(gdf["lanes"], errors="coerce").fillna(config.default_lanes)
    lw_arr = pd.to_numeric(gdf["lane_width"], errors="coerce").fillna(config.default_lane_width)
    base_offsets = (lanes_arr * lw_arr) / 2.0

    for side in ("left", "right"):
        sign = 1.0 if side == "left" else -1.0

        # Sidewalk offset mask
        sw_pres = gdf[f"sidewalk_{side}_presence"].astype(str).str.lower().str.strip()
        sw_has_geom = gdf[f"sidewalk_{side}_geometry"].notna()
        sw_needs = ~sw_pres.isin(_ABSENT) & sw_pres.notna() & ~sw_has_geom

        # Bikeway offset mask
        bk_type = gdf[f"bikeway_{side}_1_type"].astype(str).str.lower().str.strip()
        bk_has_geom = gdf[f"bikeway_{side}_1_geometry"].notna()
        bk_needs = ~bk_type.isin(_ABSENT) & bk_type.notna() & ~bk_has_geom

        any_needs = sw_needs | bk_needs
        if not any_needs.any():
            continue

        # Batch: compute offsets in UTM, collect for batch WGS84 conversion
        bk_results: list[tuple[int, BaseGeometry]] = []
        sw_results: list[tuple[int, BaseGeometry]] = []

        for idx in gdf.index[any_needs]:
            geom_u = utm_geom.at[idx]
            if geom_u is None or geom_u.is_empty or geom_u.length < 1.0:
                continue
            bo = cast(float, base_offsets.at[idx])

            if bk_needs.at[idx]:
                try:
                    bk_results.append((idx, _consolidate_line_geom(geom_u.offset_curve(sign * bo))))
                except Exception:
                    pass

            if sw_needs.at[idx]:
                bk_w = config.default_bikeway_width if bk_needs.at[idx] else 0.0
                try:
                    sw_results.append((idx, _consolidate_line_geom(geom_u.offset_curve(sign * (bo + bk_w)))))
                except Exception:
                    pass

        # Batch convert UTM -> WGS84 and assign via .loc[]
        if bk_results:
            bk_idxs, bk_geoms = zip(*bk_results)
            bk_wgs = gpd.GeoSeries(bk_geoms, crs="EPSG:32610").to_crs("EPSG:4326")
            _bk_idx_list = list(bk_idxs)
            gdf.loc[_bk_idx_list, f"bikeway_{side}_1_geometry"] = bk_wgs.values
            gdf.loc[_bk_idx_list, f"bikeway_{side}_1_quality"] = "buffered"
            gdf.loc[_bk_idx_list, f"bikeway_{side}_1_offset"] = "yes"
            generated += len(bk_results)

        if sw_results:
            sw_idxs, sw_geoms = zip(*sw_results)
            sw_wgs = gpd.GeoSeries(sw_geoms, crs="EPSG:32610").to_crs("EPSG:4326")
            _sw_idx_list = list(sw_idxs)
            gdf.loc[_sw_idx_list, f"sidewalk_{side}_geometry"] = sw_wgs.values
            gdf.loc[_sw_idx_list, f"sidewalk_{side}_quality"] = "buffered"
            gdf.loc[_sw_idx_list, f"sidewalk_{side}_offset"] = "yes"
            generated += len(sw_results)

    log.info("  Generated %d offset geometries", generated)
    return gdf


# ---------------------------------------------------------------------------
# Step 12 — Snap sidewalk endpoints at intersections
# ---------------------------------------------------------------------------
def step_12_snap_endpoints(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Snap offset sidewalk endpoints at intersections.

    Fresh geometric approach: for each intersection node, collect converging
    sidewalk endpoints, sort by angle, and pair adjacent endpoints. Then
    extend/trim each pair to meet at a corner point via ray-ray intersection.

    Stages:
      (a) Build intersection endpoint registry
      (b) Same-street continuations — extend to midpoint
      (c) Corner connections — ray-ray intersection for adjacent bearings
      (d) Trim sidewalks that cross a street centerline
      (e) Roundabout entry corners — pair closest endpoints
    """
    log.info("Step 12: Snapping sidewalk endpoints")

    # Consolidate any MultiLineString sidewalk geometries produced by
    # offset_curve (on curved roads) or OSMnx footway edge simplification.
    # Must run before endpoint collection so coords[0]/coords[-1] are reliable.
    for side in ("left", "right"):
        col = f"sidewalk_{side}_geometry"
        vals = gdf[col].values
        multi_mask = np.array([isinstance(v, MultiLineString) for v in vals], dtype=bool)
        if multi_mask.any():
            fixed = np.where(
                multi_mask,
                [_consolidate_line_geom(v) if isinstance(v, MultiLineString) else v for v in vals],
                vals,
            )
            gdf[col] = fixed
            n_fixed = int(multi_mask.sum())
            log.info("  Consolidated %d MultiLineString %s sidewalk geometries", n_fixed, side)

    # Only process intersection nodes
    isect_start = gdf.loc[gdf["start_node_is_intersection_node"], "start_node_id"]
    isect_end = gdf.loc[gdf["end_node_is_intersection_node"], "end_node_id"]
    all_isect_nodes = set(isect_start.values) | set(isect_end.values)
    log.info("  %d intersection nodes to process", len(all_isect_nodes))

    if not all_isect_nodes:
        return gdf

    # Project to UTM for geometric operations — batched (single vectorized conversion per side)
    utm_street = gdf["street_geometry"].to_crs("EPSG:32610")
    utm_sw_left = gpd.GeoSeries(
        list(gdf["sidewalk_left_geometry"]), crs="EPSG:4326", index=gdf.index
    ).to_crs("EPSG:32610")
    utm_sw_right = gpd.GeoSeries(
        list(gdf["sidewalk_right_geometry"]), crs="EPSG:4326", index=gdf.index
    ).to_crs("EPSG:32610")

    # ── (a) Build endpoint registry per intersection ──────────────────────
    # For each intersection node: list of (seg_idx, side, which_end, endpoint_utm, bearing_at_node)
    node_endpoints: dict[Any, list[tuple]] = defaultdict(list)

    for idx in gdf.index:
        for side, utm_col in (("left", utm_sw_left), ("right", utm_sw_right)):
            sw_geom = utm_col.at[idx]
            if sw_geom is None or not isinstance(sw_geom, BaseGeometry) or sw_geom.is_empty:
                continue
            coords = _flatten_coords(sw_geom)
            if len(coords) < 2:
                continue

            bearing = gdf.at[idx, "normalized_bearing"]

            # Check start end
            u = gdf.at[idx, "start_node_id"]
            if u in all_isect_nodes:
                node_endpoints[u].append((idx, side, "start", sw_geom, bearing, coords[0]))

            # Check end end
            v = gdf.at[idx, "end_node_id"]
            if v in all_isect_nodes:
                node_endpoints[v].append((idx, side, "end", sw_geom, bearing, coords[-1]))

    n_endpoints = sum(len(v) for v in node_endpoints.values())
    log.info("  Collected %d sidewalk endpoints at %d intersection nodes",
             n_endpoints, len(node_endpoints))

    # ── (b) & (c) Pair and connect endpoints at each intersection ─────────
    snapped = 0
    trimmed = 0

    for node_id, endpoints in node_endpoints.items():
        if len(endpoints) < 2:
            continue

        # Compute angle of each endpoint relative to intersection center
        # Use the street geometry endpoint as the center
        node_geom_idx = None
        for ep in endpoints:
            seg_idx = ep[0]
            if gdf.at[seg_idx, "start_node_id"] == node_id:
                node_geom_idx = seg_idx
                break
            elif gdf.at[seg_idx, "end_node_id"] == node_id:
                node_geom_idx = seg_idx
                break
        if node_geom_idx is None:
            continue

        # Get intersection center in UTM
        street_utm = utm_street.at[node_geom_idx]
        if street_utm is None:
            continue
        st_coords = _flatten_coords(street_utm)
        if gdf.at[node_geom_idx, "start_node_id"] == node_id:
            center = st_coords[0]
        else:
            center = st_coords[-1]

        # Sort endpoints by angle from center
        def _angle_from_center(ep: tuple) -> float:
            px, py = ep[5]
            dx, dy = px - center[0], py - center[1]
            return math.atan2(dx, dy) % (2 * math.pi)

        sorted_eps = sorted(endpoints, key=_angle_from_center)

        # Pair adjacent endpoints (clockwise neighbors)
        for i in range(len(sorted_eps)):
            ep_a = sorted_eps[i]
            ep_b = sorted_eps[(i + 1) % len(sorted_eps)]

            # Don't pair endpoints from the same segment+side
            if ep_a[0] == ep_b[0] and ep_a[1] == ep_b[1]:
                continue

            pt_a = ep_a[5]
            pt_b = ep_b[5]
            dist = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])

            # Skip pairs that are too far apart (likely not adjacent)
            if dist > 50.0:
                continue

            # (b) Same-street continuation: extend to midpoint
            seg_a_name = gdf.at[ep_a[0], "name"]
            seg_b_name = gdf.at[ep_b[0], "name"]
            if (seg_a_name and seg_b_name
                    and str(seg_a_name).strip() == str(seg_b_name).strip()
                    and ep_a[1] == ep_b[1]):  # same side
                mid = ((pt_a[0] + pt_b[0]) / 2, (pt_a[1] + pt_b[1]) / 2)
                _extend_sidewalk_to_point(gdf, ep_a, mid, utm_sw_left, utm_sw_right)
                _extend_sidewalk_to_point(gdf, ep_b, mid, utm_sw_left, utm_sw_right)
                snapped += 2
                continue

            # (c) Corner connection: ray-ray intersection
            corner = _ray_ray_intersection(ep_a, ep_b, max_dist=30.0)
            if corner is not None:
                _extend_sidewalk_to_point(gdf, ep_a, corner, utm_sw_left, utm_sw_right)
                _extend_sidewalk_to_point(gdf, ep_b, corner, utm_sw_left, utm_sw_right)
                snapped += 2

    # ── (d) Trim sidewalks that cross street centerlines ──────────────────
    for idx in gdf.index:
        for side, utm_col in (("left", utm_sw_left), ("right", utm_sw_right)):
            sw_geom = gdf.at[idx, f"sidewalk_{side}_geometry"]
            if sw_geom is None or not isinstance(sw_geom, BaseGeometry):
                continue
            street_geom = cast(BaseGeometry, gdf.at[idx, "street_geometry"])
            if street_geom is None:
                continue
            # Check if sidewalk crosses its own street
            if sw_geom.crosses(street_geom):
                try:
                    intersection_pt = sw_geom.intersection(street_geom)
                    if not intersection_pt.is_empty:
                        cut_dist = sw_geom.project(intersection_pt)  # type: ignore[call-overload]
                        if 0 < cut_dist < sw_geom.length:
                            trimmed_geom = _substring(sw_geom, 0, cut_dist)
                            if trimmed_geom is not None and trimmed_geom.length > 1.0:
                                gdf.at[idx, f"sidewalk_{side}_geometry"] = trimmed_geom  # type: ignore[call-overload]
                                trimmed += 1
                except Exception:
                    pass

    # ── (e) Cross-sidewalk trimming at intersections ──────────────────────
    # When two sidewalks from different segments converging at the same
    # intersection node cross each other, trim back the portion that extends
    # past the crossing point. Prevents sidewalks from penetrating into
    # intersection hulls through junction points.
    # Ported from Old/ProximityModelOLD2.py Stage 3 (STRtree crossing detection).
    s3_geoms: list[BaseGeometry] = []
    s3_meta: list[tuple[int, str, str, Any]] = []  # (seg_idx, side, which_end, node_id)

    for nid, endpoints in node_endpoints.items():
        for (seg_idx, side, which_end, sw_geom_utm, _bearing, _ep) in endpoints:
            s3_geoms.append(sw_geom_utm)
            s3_meta.append((seg_idx, side, which_end, nid))

    cross_trimmed = 0
    if s3_geoms:
        tree = STRtree(s3_geoms)
        left_idx, right_idx = tree.query(s3_geoms, predicate="crosses")
        trimmed_keys: set[tuple[int, str]] = set()

        for qi, ti in zip(left_idx.tolist(), right_idx.tolist()):
            if ti <= qi:
                continue
            if s3_meta[qi][3] != s3_meta[ti][3]:
                continue  # different intersection nodes
            if s3_meta[qi][0] == s3_meta[ti][0]:
                continue  # same segment

            ix = s3_geoms[qi].intersection(s3_geoms[ti])
            if not isinstance(ix, Point):
                continue

            for arr_i in (qi, ti):
                seg_idx, side, which_end, _ = s3_meta[arr_i]
                trim_key = (seg_idx, side)
                if trim_key in trimmed_keys:
                    continue

                ls = s3_geoms[arr_i]
                if not isinstance(ls, LineString) or ls.length < 0.01:
                    continue

                d = ls.project(ix)
                dist_from_ep = (ls.length - d) if which_end == "end" else d
                if dist_from_ep > 15.0:
                    continue

                if which_end == "end":
                    new_geom = _substring(ls, 0, d) if d > 0.01 else None
                else:
                    new_geom = _substring(ls, d, ls.length) if d < ls.length - 0.01 else None

                if new_geom is None or new_geom.length < 1.0:
                    continue

                gdf.at[seg_idx, f"sidewalk_{side}_geometry"] = _project_to_wgs(new_geom)
                s3_geoms[arr_i] = new_geom
                trimmed_keys.add(trim_key)
                cross_trimmed += 1

    trimmed += cross_trimmed
    log.info("Step 12: Snapped %d endpoints, trimmed %d crossings (%d cross-sidewalk)",
             snapped, trimmed, cross_trimmed)
    return gdf


def _ray_ray_intersection(
    ep_a: tuple, ep_b: tuple, max_dist: float = 30.0,
) -> tuple[float, float] | None:
    """Find intersection of rays extending from two sidewalk endpoints.
    Returns the corner point (x, y) in UTM or None if rays don't converge.
    """
    # Get direction vectors from last two coords of each sidewalk
    geom_a = ep_a[3]
    geom_b = ep_b[3]
    coords_a = _flatten_coords(geom_a)
    coords_b = _flatten_coords(geom_b)
    if len(coords_a) < 2 or len(coords_b) < 2:
        return None

    # Direction at endpoint
    if ep_a[2] == "end":
        da = (coords_a[-1][0] - coords_a[-2][0], coords_a[-1][1] - coords_a[-2][1])
        pa = coords_a[-1]
    else:
        da = (coords_a[0][0] - coords_a[1][0], coords_a[0][1] - coords_a[1][1])
        pa = coords_a[0]

    if ep_b[2] == "end":
        db = (coords_b[-1][0] - coords_b[-2][0], coords_b[-1][1] - coords_b[-2][1])
        pb = coords_b[-1]
    else:
        db = (coords_b[0][0] - coords_b[1][0], coords_b[0][1] - coords_b[1][1])
        pb = coords_b[0]

    # Solve pa + t*da = pb + s*db
    det = da[0] * (-db[1]) - da[1] * (-db[0])
    if abs(det) < 1e-10:
        return None  # parallel

    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    t = (dx * (-db[1]) - dy * (-db[0])) / det

    if t < 0:
        return None  # intersection behind ray A

    ix = pa[0] + t * da[0]
    iy = pa[1] + t * da[1]

    # Check distance from both endpoints to intersection
    d_a = math.hypot(ix - pa[0], iy - pa[1])
    d_b = math.hypot(ix - pb[0], iy - pb[1])
    if d_a > max_dist or d_b > max_dist:
        return None

    return (ix, iy)


def _extend_sidewalk_to_point(
    gdf: gpd.GeoDataFrame,
    ep: tuple,
    target: tuple[float, float],
    utm_sw_left: pd.Series,
    utm_sw_right: pd.Series,
) -> None:
    """Extend a sidewalk geometry to reach target point, then write back to GDF in WGS84."""
    seg_idx, side, end = ep[0], ep[1], ep[2]
    col = f"sidewalk_{side}_geometry"
    current = gdf.at[seg_idx, col]
    if current is None or not isinstance(current, BaseGeometry):
        return

    # Work in UTM
    utm_geom = cast(BaseGeometry, utm_sw_left.at[seg_idx] if side == "left" else utm_sw_right.at[seg_idx])
    if utm_geom is None:
        return

    coords = _flatten_coords(utm_geom)
    if len(coords) < 2:
        return

    if end == "start":
        coords.insert(0, target)
    else:
        coords.append(target)

    new_utm = LineString(coords)
    new_wgs = _project_to_wgs(new_utm)
    gdf.at[seg_idx, col] = new_wgs  # type: ignore[call-overload]


def _substring(geom: BaseGeometry, start_dist: float, end_dist: float) -> LineString | None:
    """Extract a substring of a LineString between two distances along it."""
    if not hasattr(geom, "coords"):
        return None
    coords = list(geom.coords)
    if len(coords) < 2:
        return None

    result_coords = []
    cumulative = 0.0

    for i in range(len(coords)):
        if i > 0:
            seg_len = math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
            prev_cum = cumulative
            cumulative += seg_len

            if prev_cum < start_dist <= cumulative:
                frac = (start_dist - prev_cum) / seg_len if seg_len > 0 else 0
                x = coords[i-1][0] + frac * (coords[i][0] - coords[i-1][0])
                y = coords[i-1][1] + frac * (coords[i][1] - coords[i-1][1])
                result_coords.append((x, y))

            if prev_cum >= start_dist and cumulative <= end_dist:
                result_coords.append(coords[i])
            elif cumulative > end_dist and prev_cum < end_dist:
                frac = (end_dist - prev_cum) / seg_len if seg_len > 0 else 0
                x = coords[i-1][0] + frac * (coords[i][0] - coords[i-1][0])
                y = coords[i-1][1] + frac * (coords[i][1] - coords[i-1][1])
                result_coords.append((x, y))
                break
        elif cumulative >= start_dist:
            result_coords.append(coords[i])

    if len(result_coords) < 2:
        return None
    return LineString(result_coords)


# ---------------------------------------------------------------------------
# Step 11 — Assign grid IDs to sidewalk and bikeway slots
# ---------------------------------------------------------------------------
def step_11_facility_grid_ids(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Assign grid IDs to sidewalk and bikeway slots from parent street grid ID.

    Suffix mapping: L/R for sidewalks, L1/L2/R1/R2 for bikeways.
    Eligibility: presence/type NOT in {'no','none'} — NaN IS eligible.
    """
    log.info("Step 11: Assigning facility grid IDs")
    _ABSENT = {"no", "none"}
    sgid = gdf["street_grid_id"] if "street_grid_id" in gdf.columns else None

    if sgid is None or sgid.isna().all():
        log.warning("  No street_grid_id available, skipping")
        return gdf

    def _absent_mask(col: str) -> pd.Series:
        if col not in gdf.columns:
            return pd.Series(True, index=gdf.index)
        s = gdf[col]
        na_mask = s.isna()
        return (~na_mask) & s.astype(str).str.strip().str.lower().isin(_ABSENT)

    def _write_ids(id_col: str, grid_id_col: str | None, suffix: str,
                   eligible: pd.Series) -> None:
        active = eligible & sgid.notna()
        vals = sgid[active].astype(str) + suffix
        gdf[id_col] = pd.NA
        gdf.loc[active, id_col] = vals
        if grid_id_col is not None:
            gdf[grid_id_col] = pd.NA
            gdf.loc[active, grid_id_col] = vals

    # Sidewalks
    for side, suffix in (("left", "L"), ("right", "R")):
        eligible = ~_absent_mask(f"sidewalk_{side}_presence")
        _write_ids(f"sidewalk_{side}_ID", f"sidewalk_{side}_grid_ID", suffix, eligible)

    # Bikeways
    for side, suffix in (("left", "L"), ("right", "R")):
        absent1 = _absent_mask(f"bikeway_{side}_1_type")
        absent2 = _absent_mask(f"bikeway_{side}_2_type")
        either_eligible = ~absent1 | ~absent2

        # Shared grid_id for both slots
        _write_ids(f"bikeway_{side}_1_id", f"bikeway_{side}_1_grid_id",
                   f"{suffix}1", either_eligible)
        # Fix slot 1 id to only where slot 1 is eligible
        only1 = ~absent1 & sgid.notna()
        gdf[f"bikeway_{side}_1_id"] = pd.NA
        gdf.loc[only1, f"bikeway_{side}_1_id"] = sgid[only1].astype(str) + f"{suffix}1"

        # Slot 2 grid_id (shared with slot 1) and id
        gdf[f"bikeway_{side}_2_grid_id"] = pd.NA
        gdf.loc[either_eligible & sgid.notna(), f"bikeway_{side}_2_grid_id"] = (
            sgid[either_eligible & sgid.notna()].astype(str) + f"{suffix}2"
        )
        only2 = ~absent2 & sgid.notna()
        gdf[f"bikeway_{side}_2_id"] = pd.NA
        gdf.loc[only2, f"bikeway_{side}_2_id"] = sgid[only2].astype(str) + f"{suffix}2"

    # Stats
    sw_l = gdf["sidewalk_left_ID"].notna().sum() if "sidewalk_left_ID" in gdf.columns else 0
    sw_r = gdf["sidewalk_right_ID"].notna().sum() if "sidewalk_right_ID" in gdf.columns else 0
    bk_l = gdf["bikeway_left_1_id"].notna().sum() if "bikeway_left_1_id" in gdf.columns else 0
    bk_r = gdf["bikeway_right_1_id"].notna().sum() if "bikeway_right_1_id" in gdf.columns else 0
    log.info("Step 11: sidewalk IDs L=%d R=%d, bikeway IDs L=%d R=%d", sw_l, sw_r, bk_l, bk_r)
    return gdf


# ---------------------------------------------------------------------------
# Step 11b — Compute sidewalk / bikeway / crosswalk neighbor adjacency
# ---------------------------------------------------------------------------

def step_11b_compute_neighbors(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Populate from_segments / to_segments columns on each facility.

    For each sidewalk, bikeway, and crosswalk, determine which other facility
    segments connect at each end (start-node end = from, end-node end = to).
    Sources of adjacency:
      1. Road-node topology: roads sharing intersection nodes → their facilities
         are neighbors at that node.
      2. Walk/bike network topology (if stored in gdf.attrs): explicitly-connected
         OSM footway/cycleway edges → their matched road-row facilities.
      3. Mid-block splits: consecutive segments from the same OSM way that were
         split by step 03 share an intermediate node.
    """
    log.info("Step 11b: Computing facility neighbor adjacency")

    # ── Build node → facility mapping ─────────────────────────────────────
    # For each intersection node, collect all facility IDs whose parent road
    # meets that node (at start or end).
    node_to_fac_start: dict[Any, list[str]] = defaultdict(list)  # node → IDs at their start
    node_to_fac_end: dict[Any, list[str]] = defaultdict(list)    # node → IDs at their end

    for idx in gdf.index:
        start_nid = gdf.at[idx, "start_node_id"]
        end_nid = gdf.at[idx, "end_node_id"]

        for side in ("left", "right"):
            sw_id = gdf.at[idx, f"sidewalk_{side}_grid_ID"]
            if sw_id is not None and not (isinstance(sw_id, float) and pd.isna(sw_id)):
                node_to_fac_start[start_nid].append(str(sw_id))
                node_to_fac_end[end_nid].append(str(sw_id))

        for side in ("left", "right"):
            for n in (1, 2):
                bk_id = gdf.at[idx, f"bikeway_{side}_{n}_grid_id"]
                if bk_id is not None and not (isinstance(bk_id, float) and pd.isna(bk_id)):
                    node_to_fac_start[start_nid].append(str(bk_id))
                    node_to_fac_end[end_nid].append(str(bk_id))

        for pos in ("start", "end"):
            xw_id = gdf.at[idx, f"crosswalk_{pos}_id"]
            if xw_id is not None and not (isinstance(xw_id, float) and pd.isna(xw_id)):
                # Crosswalk connects at the node corresponding to its pos
                nid = start_nid if pos == "start" else end_nid
                node_to_fac_start[nid].append(str(xw_id))
                node_to_fac_end[nid].append(str(xw_id))

    # ── Build adjacency: for each facility, neighbors at each end ─────────
    # A facility's "from" neighbors = other facilities at the same start node
    # A facility's "to" neighbors = other facilities at the same end node

    total_links = 0
    for idx in gdf.index:
        start_nid = gdf.at[idx, "start_node_id"]
        end_nid = gdf.at[idx, "end_node_id"]

        # All facilities meeting at start_nid and end_nid
        at_start = set(node_to_fac_start.get(start_nid, []) +
                       node_to_fac_end.get(start_nid, []))
        at_end = set(node_to_fac_start.get(end_nid, []) +
                     node_to_fac_end.get(end_nid, []))

        for side in ("left", "right"):
            sw_id = gdf.at[idx, f"sidewalk_{side}_grid_ID"]
            if sw_id is None or (isinstance(sw_id, float) and pd.isna(sw_id)):
                continue
            sw_id_str = str(sw_id)
            from_nbrs = sorted(at_start - {sw_id_str})
            to_nbrs = sorted(at_end - {sw_id_str})
            gdf.at[idx, f"sidewalk_{side}_from_segments"] = from_nbrs  # type: ignore[call-overload]
            gdf.at[idx, f"sidewalk_{side}_to_segments"] = to_nbrs  # type: ignore[call-overload]
            total_links += len(from_nbrs) + len(to_nbrs)

        for side in ("left", "right"):
            for n in (1, 2):
                pfx = f"bikeway_{side}_{n}"
                bk_id = gdf.at[idx, f"{pfx}_grid_id"]
                if bk_id is None or (isinstance(bk_id, float) and pd.isna(bk_id)):
                    continue
                bk_id_str = str(bk_id)
                from_nbrs = sorted(at_start - {bk_id_str})
                to_nbrs = sorted(at_end - {bk_id_str})
                gdf.at[idx, f"{pfx}_from_segments"] = from_nbrs  # type: ignore[call-overload]
                gdf.at[idx, f"{pfx}_to_segments"] = to_nbrs  # type: ignore[call-overload]
                total_links += len(from_nbrs) + len(to_nbrs)

        for pos in ("start", "end"):
            xw_id = gdf.at[idx, f"crosswalk_{pos}_id"]
            if xw_id is None or (isinstance(xw_id, float) and pd.isna(xw_id)):
                continue
            xw_id_str = str(xw_id)
            nid = start_nid if pos == "start" else end_nid
            at_xw = set(node_to_fac_start.get(nid, []) +
                        node_to_fac_end.get(nid, []))
            xw_nbrs = sorted(at_xw - {xw_id_str})
            gdf.at[idx, f"crosswalk_{pos}_from_segments"] = xw_nbrs  # type: ignore[call-overload]
            gdf.at[idx, f"crosswalk_{pos}_to_segments"] = xw_nbrs  # type: ignore[call-overload]
            total_links += len(xw_nbrs) * 2

    log.info("Step 11b: %d adjacency links computed", total_links)
    return gdf


# ---------------------------------------------------------------------------
# Step 13 — Place curb ramps and build intersection hulls
# ---------------------------------------------------------------------------
def step_13_curb_ramps_and_hulls(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Place curb ramp points at sidewalk endpoints near intersection nodes.
    Build convex hull intersection zones from those ramp points.

    Returns (gdf, hulls_gdf).
    """
    log.info("Step 13: Placing curb ramps and building intersection hulls")

    # Identify intersection nodes
    isect_start = gdf.loc[gdf["start_node_is_intersection_node"]]
    isect_end = gdf.loc[gdf["end_node_is_intersection_node"]]
    all_isect_nodes = set(isect_start["start_node_id"].values) | set(isect_end["end_node_id"].values)

    # Build node -> segment lookup — vectorized via boolean masks + zip (no .at[] in a loop)
    node_to_segs: dict[Any, list[tuple[int, str]]] = defaultdict(list)
    start_ids = gdf["start_node_id"].values
    end_ids = gdf["end_node_id"].values
    idx_arr = gdf.index.values
    start_mask = gdf["start_node_is_intersection_node"].values
    end_mask = gdf["end_node_is_intersection_node"].values
    for idx, nid in zip(idx_arr[np.asarray(start_mask, dtype=bool)], start_ids[np.asarray(start_mask, dtype=bool)]):
        node_to_segs[nid].append((idx, "start"))
    for idx, nid in zip(idx_arr[np.asarray(end_mask, dtype=bool)], end_ids[np.asarray(end_mask, dtype=bool)]):
        node_to_segs[nid].append((idx, "end"))

    # Precompute UTM coordinates for every intersection node so the hull
    # builder can filter out ramp points that are too far from the node.
    # Ramps > _HULL_RAMP_MAX_DIST_M away belong to a different intersection
    # zone and should not stretch the hull toward it.
    _HULL_RAMP_MAX_DIST_M = 15.0
    node_utm_coords: dict[Any, tuple[float, float]] = {}
    _new_nids: list[Any] = []
    _new_xs: list[float] = []
    _new_ys: list[float] = []
    for geom_col, id_col in [
        ("start_node_geometry", "start_node_id"),
        ("end_node_geometry",   "end_node_id"),
    ]:
        ids = gdf[id_col].values
        geoms = gdf[geom_col].values
        for nid, g in zip(ids, geoms):
            if nid in node_utm_coords:
                continue
            if isinstance(g, Point) and not g.is_empty:
                node_utm_coords[nid] = None  # placeholder to avoid duplicates
                _new_nids.append(nid)
                _new_xs.append(g.x)
                _new_ys.append(g.y)
    if _new_nids:
        _uxs, _uys = _to_utm.transform(
            np.array(_new_xs, dtype=np.float64),
            np.array(_new_ys, dtype=np.float64),
        )
        for nid, ux, uy in zip(_new_nids, _uxs, _uys):
            node_utm_coords[nid] = (float(ux), float(uy))

    hull_rows = []
    ramps_placed = 0
    ramp_id_counter = 0

    for node_id, seg_ends in node_to_segs.items():
        ramp_points: list[Point] = []
        ramp_info: list[tuple[int, str, str, Point]] = []  # (seg_idx, side, pos, point)

        # ── Topology 1: Sidewalk endpoint nearest to intersection node ───
        for seg_idx, which_end in seg_ends:
            # Fetch the street-node geometry to resolve endpoint direction
            node_geom_col = "start_node_geometry" if which_end == "start" else "end_node_geometry"
            node_pt = gdf.at[seg_idx, node_geom_col]
            if not isinstance(node_pt, BaseGeometry):
                node_pt = None

            for side in ("left", "right"):
                geom = gdf.at[seg_idx, f"sidewalk_{side}_geometry"]
                if geom is None or not isinstance(geom, BaseGeometry) or geom.is_empty:
                    continue
                coords = _flatten_coords(geom)
                if len(coords) < 2:
                    continue
                # Pick whichever endpoint is closer to the intersection node.
                # Separately-mapped footways may be stored in reverse direction
                # relative to the parent street; nearest-endpoint selection is
                # robust to this without requiring geometry realignment.
                if node_pt is not None:
                    d0 = Point(coords[0]).distance(node_pt)
                    d1 = Point(coords[-1]).distance(node_pt)
                    pt_coords = coords[0] if d0 <= d1 else coords[-1]
                else:
                    pt_coords = coords[0] if which_end == "start" else coords[-1]
                pt = Point(pt_coords)
                ramp_points.append(pt)
                ramp_info.append((seg_idx, side, which_end, pt))

        # ── Topology 2: Sidewalk-sidewalk geometric crossings ────────────
        # When two sidewalk geometries from arms of this intersection cross
        # each other (e.g. a footway mapped through the full intersection
        # crossing orthogonal sidewalks), the crossing points are curb ramps.
        # Each crossing point is assigned to both sidewalks involved.
        node_utm_coord = node_utm_coords.get(node_id)
        if node_utm_coord is not None and len(ramp_info) >= 2:
            # Collect all (seg_idx, side, pos, wgs_geom) for this node
            sw_line_geoms: list[tuple[int, str, str, Any]] = []
            for seg_idx, which_end in seg_ends:
                for side in ("left", "right"):
                    sw_g = gdf.at[seg_idx, f"sidewalk_{side}_geometry"]
                    if sw_g is None or not isinstance(sw_g, BaseGeometry) or sw_g.is_empty:
                        continue
                    sw_line_geoms.append((seg_idx, side, which_end, sw_g))
            for i in range(len(sw_line_geoms)):
                for j in range(i + 1, len(sw_line_geoms)):
                    si, ss, sp, sg = sw_line_geoms[i]
                    ti, ts, tp, tg = sw_line_geoms[j]
                    if si == ti and ss == ts:
                        continue  # same geometry
                    xsect = sg.intersection(tg)
                    if xsect is None or xsect.is_empty:
                        continue
                    pts = list(xsect.geoms) if hasattr(xsect, "geoms") else [xsect]
                    for cp in pts:
                        if not isinstance(cp, Point):
                            continue
                        cpx, cpy = _to_utm.transform(
                            np.array([cp.x], dtype=np.float64),
                            np.array([cp.y], dtype=np.float64),
                        )
                        dist = math.hypot(float(cpx[0]) - node_utm_coord[0],
                                          float(cpy[0]) - node_utm_coord[1])
                        if dist > _HULL_RAMP_MAX_DIST_M:
                            continue
                        # Assign to both sidewalks involved in the crossing
                        ramp_points.append(cp)
                        ramp_info.append((si, ss, sp, cp))
                        ramp_points.append(cp)
                        ramp_info.append((ti, ts, tp, cp))

        # ── Topology 3: Footway-footway shared intermediate nodes ────────
        # When two footway arms share an intermediate node (physical footway
        # crossing), include that shared node as a hull vertex and ramp point.
        footway_arms = [
            (si, we) for si, we in seg_ends
            if str(gdf.at[si, "highway"]).lower()
            in ("footway", "path", "pedestrian", "crossing", "steps")
        ]
        if len(footway_arms) >= 2:
            seen_shared: set[Any] = set()
            for i, (fi, wi) in enumerate(footway_arms):
                fi_nodes = {gdf.at[fi, "start_node_id"], gdf.at[fi, "end_node_id"]}
                for j in range(i + 1, len(footway_arms)):
                    fj, wj = footway_arms[j]
                    fj_nodes = {gdf.at[fj, "start_node_id"], gdf.at[fj, "end_node_id"]}
                    shared = (fi_nodes & fj_nodes) - {node_id}
                    for sn in shared:
                        if sn in seen_shared:
                            continue
                        seen_shared.add(sn)
                        spt = None
                        if gdf.at[fi, "start_node_id"] == sn:
                            spt = gdf.at[fi, "start_node_geometry"]
                        elif gdf.at[fi, "end_node_id"] == sn:
                            spt = gdf.at[fi, "end_node_geometry"]
                        if isinstance(spt, Point) and not spt.is_empty:
                            ramp_points.append(spt)
                            ramp_info.append((fi, "left", "start", spt))

        if len(ramp_points) < 2:
            continue

        # Pre-snap close pairs (<1m) — batch projection via transformer (no GeoSeries round-trip)
        snap_threshold = config.curb_ramp_snap_m
        xs = np.fromiter((p.x for p in ramp_points), dtype=np.float64, count=len(ramp_points))
        ys = np.fromiter((p.y for p in ramp_points), dtype=np.float64, count=len(ramp_points))
        ux, uy = _to_utm.transform(xs, ys)
        ux = np.asarray(ux, dtype=np.float64)
        uy = np.asarray(uy, dtype=np.float64)
        snapped = list(ramp_points)
        k = len(ux)
        if k > 1:
            _coords = np.column_stack([ux, uy])
            _diff   = _coords[:, None, :] - _coords[None, :, :]
            _dists  = np.linalg.norm(_diff, axis=2)
            _mask   = np.triu((_dists > 0) & (_dists < snap_threshold), k=1)
            for i, j in zip(*np.where(_mask)):
                mx = (ux[i] + ux[j]) / 2
                my = (uy[i] + uy[j]) / 2
                lon, lat = _to_wgs.transform(mx, my)
                mid_wgs = Point(lon, lat)
                snapped[i] = mid_wgs
                snapped[j] = mid_wgs

        # Write curb ramp geometries to GDF
        for k, (seg_idx, side, pos, _) in enumerate(ramp_info):
            for n in (1, 2, 3):
                col = f"sidewalk_{side}_curbramp_{pos}_{n}_geometry"
                if col in gdf.columns and pd.isna(gdf.at[seg_idx, col]):
                    gdf.at[seg_idx, col] = snapped[k]  # type: ignore[call-overload]
                    quality_col = f"sidewalk_{side}_curbramp_{pos}_{n}_quality"
                    if quality_col in gdf.columns:
                        gdf.at[seg_idx, quality_col] = "topology"
                    id_col = f"sidewalk_{side}_curbramp_{pos}_{n}_ID"
                    if id_col in gdf.columns:
                        sgid = gdf.at[seg_idx, "street_grid_id"]
                        gdf.at[seg_idx, id_col] = f"{sgid}_CR_{side}_{pos}_{n}"
                    ramps_placed += 1
                    break

        # Build convex hull only from ramp points close to the intersection node.
        # Far-away ramp points (e.g. sidewalks that don't reach the intersection,
        # or reversed geometries) would produce elongated hulls that trigger
        # spurious hull merges in step_13.
        node_utm = node_utm_coords.get(node_id)
        if node_utm is not None and snapped:
            nux, nuy = node_utm
            s_xs = np.fromiter((q.x for q in snapped), dtype=np.float64, count=len(snapped))
            s_ys = np.fromiter((q.y for q in snapped), dtype=np.float64, count=len(snapped))
            s_ux, s_uy = _to_utm.transform(s_xs, s_ys)
            s_ux = np.asarray(s_ux, dtype=np.float64)
            s_uy = np.asarray(s_uy, dtype=np.float64)
            dists = np.hypot(s_ux - nux, s_uy - nuy)
            hull_snapped = [p for p, d in zip(snapped, dists) if d <= _HULL_RAMP_MAX_DIST_M]
        else:
            hull_snapped = snapped

        unique_pts = list({(p.x, p.y) for p in hull_snapped})
        if len(unique_pts) >= 3:
            hull_geom = MultiPoint([Point(c) for c in unique_pts]).convex_hull
        elif len(unique_pts) == 2:
            hull_geom = LineString([Point(c) for c in unique_pts]).buffer(0.00001)
        else:
            continue

        hull_rows.append({
            "node_id": node_id,
            "geometry": hull_geom,
            "n_ramps": len(ramp_points),
            "arm_indices": [se[0] for se in seg_ends],
        })

    hulls = gpd.GeoDataFrame(hull_rows, geometry="geometry", crs="EPSG:4326") if hull_rows else gpd.GeoDataFrame(
        columns=["node_id", "geometry", "n_ramps", "arm_indices"], geometry="geometry", crs="EPSG:4326"
    )

    log.info("Step 13: %d curb ramps placed, %d intersection hulls built", ramps_placed, len(hulls))
    return gdf, hulls


# ---------------------------------------------------------------------------
# Step 12b — Clear hull-captive sidewalk slots
# ---------------------------------------------------------------------------
def step_13b_clear_hull_artifact_sidewalks(
    gdf: gpd.GeoDataFrame,
    hulls: gpd.GeoDataFrame,
    hull_sw_threshold: float = 0.80,
) -> gpd.GeoDataFrame:
    """Clear separately-matched sidewalk slots whose geometry falls mostly inside
    an intersection hull.

    Corner connectors and crosswalk approach footways often get mis-assigned as
    the sidewalk for the road segment they happen to touch. Because they lie
    inside the intersection zone, they produce short stub geometries that look
    fragmented around the hull. Clearing them restores the buffered offset as
    the fallback, which gives a cleaner appearance.

    Adapted from _filter_intersection_hull_artifacts (secondary check) in
    Old/ProximityModelOLD2.py.
    """
    # node_id → hull polygon
    node_hull: dict[Any, BaseGeometry] = dict(zip(hulls["node_id"], hulls.geometry))
    if not node_hull:
        return gdf

    start_ids = gdf["start_node_id"].values
    end_ids   = gdf["end_node_id"].values
    idx_arr   = gdf.index.values
    n_cleared = 0

    for side in ("left", "right"):
        pres_col = f"sidewalk_{side}_presence"
        geom_col = f"sidewalk_{side}_geometry"

        pres_vals = gdf[pres_col].values
        geom_vals = gdf[geom_col].values

        sep_mask  = np.array([str(p).lower() == "separate" for p in pres_vals], dtype=bool)
        geom_ok   = np.array([isinstance(g, BaseGeometry) and not g.is_empty for g in geom_vals], dtype=bool)
        candidates = np.where(sep_mask & geom_ok)[0]

        clear_positions: list[int] = []
        for pos in candidates:
            sw_g = geom_vals[pos]
            sw_len = sw_g.length
            if sw_len <= 0:
                continue
            max_frac = 0.0
            for hull in (node_hull.get(start_ids[pos]), node_hull.get(end_ids[pos])):
                if hull is None:
                    continue
                try:
                    frac = sw_g.intersection(hull).length / sw_len
                    if frac > max_frac:
                        max_frac = frac
                except Exception:
                    pass
            if max_frac >= hull_sw_threshold:
                clear_positions.append(pos)

        if clear_positions:
            clear_idx = idx_arr[clear_positions]
            gdf.loc[clear_idx, geom_col] = None
            gdf.loc[clear_idx, pres_col] = pd.NA
            n_cleared += len(clear_positions)

    log.info("Step 13b: Cleared %d hull-captive sidewalk slots (threshold=%.0f%%)",
             n_cleared, hull_sw_threshold * 100)
    return gdf


# ---------------------------------------------------------------------------
# Step 14 — Merge nearby intersection hulls
# ---------------------------------------------------------------------------
def step_14_merge_hulls(
    hulls: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Merge intersection hulls whose outer edges are within hull_merge_buffer_m.

    Uses buffer-cascade: buffer each hull, find overlaps via spatial index,
    union overlapping groups.
    """
    if hulls.empty:
        log.info("Step 14: No hulls to merge")
        return hulls
    log.info("Step 14: Merging nearby intersection hulls")

    # Project to UTM for metric buffering
    hulls_utm = hulls.to_crs("EPSG:32610").copy()
    buf_dist = config.hull_merge_buffer_m
    buffered = hulls_utm.geometry.buffer(buf_dist)

    # Find connected components via spatial overlap
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import lil_matrix

    n = len(hulls_utm)
    adj = lil_matrix((n, n), dtype=bool)
    sindex = buffered.sindex

    left_idx, right_idx = sindex.query(buffered, predicate="intersects")
    for i, j in zip(left_idx.tolist(), right_idx.tolist()):
        if i != j:
            adj[i, j] = True
            adj[j, i] = True

    n_components, labels = connected_components(adj.tocsr(), directed=False)

    # Merge each component
    from shapely.ops import unary_union as _unary_union
    merged_rows = []
    for comp in range(n_components):
        members = np.flatnonzero(labels == comp)
        if len(members) == 1:
            idx = members[0]
            r = hulls.iloc[idx]
            merged_rows.append({
                "node_id": r["node_id"],
                "geometry": r["geometry"],
                "n_ramps": r["n_ramps"],
                "arm_indices": r["arm_indices"],
            })
        else:
            sub = hulls.iloc[members]
            sub_utm = hulls_utm.iloc[members]
            union_utm = _unary_union(sub_utm.geometry.values).convex_hull
            merged_wgs = _project_to_wgs(union_utm)
            all_arms: list[int] = [arm for arms in sub["arm_indices"] for arm in arms]
            merged_rows.append({
                "node_id": sub.iloc[0]["node_id"],
                "geometry": merged_wgs,
                "n_ramps": int(sub["n_ramps"].sum()),
                "arm_indices": list(set(all_arms)),
            })

    merged = gpd.GeoDataFrame(merged_rows, geometry="geometry", crs="EPSG:4326")
    n_merged = n - len(merged)
    log.info("Step 14: %d hulls -> %d after merging (%d merged)", n, len(merged), n_merged)
    return merged


# ---------------------------------------------------------------------------
# Step 15 — Calculate crosswalk slots from hull faces
# ---------------------------------------------------------------------------
def step_15_crosswalk_slots(
    gdf: gpd.GeoDataFrame,
    hulls: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Create crosswalk slots by buffering each hull face centerline.

    Each hull face (edge of the convex hull polygon) defines a potential
    crosswalk location. The slot is a rectangle created by buffering the
    face line by crosswalk_buffer_m.

    Stores slots as a 'crosswalk_slots' column on the hulls GDF.
    """
    if hulls.empty:
        log.info("Step 15: No hulls, no crosswalk slots")
        return gdf
    log.info("Step 15: Computing crosswalk slots from hull faces")

    # Project hulls to UTM for metric buffering
    hulls_utm = hulls.copy()
    hulls_utm["geometry"] = hulls["geometry"].to_crs("EPSG:32610")

    all_slots: list[dict[str, Any]] = []

    for hull_idx in hulls_utm.index:
        hull_geom = cast(Polygon, hulls_utm.at[hull_idx, "geometry"])
        arm_indices = hulls_utm.at[hull_idx, "arm_indices"]
        if hull_geom is None or hull_geom.is_empty:
            continue

        # Extract hull faces (edges of the polygon boundary)
        if hasattr(hull_geom, "exterior"):
            boundary_coords = list(hull_geom.exterior.coords)
        else:
            continue

        for i in range(len(boundary_coords) - 1):
            face_line = LineString([boundary_coords[i], boundary_coords[i + 1]])
            if face_line.length < 0.5:  # skip degenerate faces
                continue
            slot_geom_utm = face_line.buffer(config.crosswalk_buffer_m, cap_style="flat")
            slot_geom_wgs = _project_to_wgs(slot_geom_utm)
            face_wgs = _project_to_wgs(face_line)
            all_slots.append({
                "hull_idx": hull_idx,
                "face_line": face_wgs,
                "slot_geom": slot_geom_wgs,
                "arm_indices": arm_indices,
                "node_id": hulls.at[hull_idx, "node_id"],
            })

    # Store slots on hulls GDF for step 15
    hulls.attrs["crosswalk_slots"] = all_slots
    log.info("Step 15: %d crosswalk slots created from %d hulls", len(all_slots), len(hulls))
    return gdf


# ---------------------------------------------------------------------------
# Step 16 — Detect/create crosswalk geometries (Cases A-E) and populate attrs
# ---------------------------------------------------------------------------
def step_16_crosswalk_geometries(
    gdf: gpd.GeoDataFrame,
    hulls: gpd.GeoDataFrame,
    crosswalk_cache: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Detect or create crosswalk geometries. Priority order:

    OSM A:    OSM crossing point + ≥2 curb ramps → polyline ramp → point → ramp
    OSM B:    OSM crossing point + 1 curb ramp → line extended equally past crossing point
    OSM C:    OSM crossing point, no ramps → orthogonal line to sidewalks; new ramp slots written
    OSM D:    Separately-mapped footway crossing chain
    Topo A:   Sidewalk crosses a street segment
    Topo B:   Curb ramp pair (any arm in slot)
    Topo C:   Sidewalk endpoint pair (fallback)
    """
    slots = hulls.attrs.get("crosswalk_slots", [])
    if not slots:
        log.info("Step 16: No crosswalk slots to process")
        return gdf
    log.info("Step 16: Processing %d crosswalk slots", len(slots))

    hw_col = gdf["highway"].astype(str).str.lower()
    is_footway_row = hw_col.isin({"footway", "path", "pedestrian", "crossing", "steps"})

    # Build spatial index for road segments
    road_mask = hw_col.isin({
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "service", "unclassified", "living_street",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    })

    # Build spatial index for footway-type segments — avoids 21K x 106K nested scan for Case D.
    footway_gdf = gdf.loc[is_footway_row, ["street_geometry"]].copy()
    footway_gs = gpd.GeoSeries(
        list(footway_gdf["street_geometry"]), crs="EPSG:4326", index=footway_gdf.index
    )
    footway_sindex = footway_gs.sindex if len(footway_gs) > 0 else None

    case_counts = {"OSM_A": 0, "OSM_B": 0, "OSM_C": 0, "OSM_D": 0, "Topo_A": 0, "Topo_B": 0, "Topo_C": 0}
    xwalk_id = 0
    absorbed_footway_rows: set[int] = set()  # footway rows consumed by Case D

    # Build crossing-node spatial index for OSM point cases
    xwalk_sindex = crosswalk_cache.sindex if (
        not crosswalk_cache.empty and len(crosswalk_cache) > 0
    ) else None

    for slot in slots:
        slot_geom = slot["slot_geom"]
        arm_indices = slot["arm_indices"]
        if slot_geom is None:
            continue

        # Find the street segment being crossed (the one whose centerline
        # intersects this hull face most perpendicularly)
        crossed_idx = None
        for aidx in arm_indices:
            if aidx in gdf.index and road_mask.at[aidx]:
                street_geom = gdf.at[aidx, "street_geometry"]
                if street_geom is not None and slot_geom.intersects(street_geom):
                    crossed_idx = aidx
                    break

        if crossed_idx is None:
            # Try any arm
            for aidx in arm_indices:
                if aidx in gdf.index:
                    street_geom = gdf.at[aidx, "street_geometry"]
                    if street_geom is not None and slot_geom.intersects(street_geom):
                        crossed_idx = aidx
                        break

        if crossed_idx is None:
            continue

        # Determine which end of the crossed segment (start or end)
        node_id = slot["node_id"]
        if gdf.at[crossed_idx, "start_node_id"] == node_id:
            pos = "start"
        elif gdf.at[crossed_idx, "end_node_id"] == node_id:
            pos = "end"
        else:
            pos = "start"

        # Skip if already filled
        if gdf.at[crossed_idx, f"crosswalk_{pos}_geometry"] is not None:
            if isinstance(gdf.at[crossed_idx, f"crosswalk_{pos}_geometry"], BaseGeometry):
                continue

        xwalk_geom = None
        source = None

        # ── OSM A/B/C: OSM crossing-point cases ──────────────────────────────
        # Find OSM crossing nodes whose point geometry falls inside this slot.
        osm_crossing_pts: list[Any] = []
        if xwalk_sindex is not None:
            cn_hits = list(xwalk_sindex.query(slot_geom, predicate="intersects"))
            for hi in cn_hits:
                cp = crosswalk_cache.geometry.iloc[hi]
                if isinstance(cp, Point) and not cp.is_empty and slot_geom.contains(cp):
                    osm_crossing_pts.append(cp)

        if osm_crossing_pts:
            # Use the crossing node nearest to the slot centroid
            slot_centroid = slot_geom.centroid
            osm_pt = min(osm_crossing_pts, key=lambda p: p.distance(slot_centroid))

            # Collect all curb ramps in slot from every arm
            osm_ramp_pts: list[Any] = []
            for aidx in arm_indices:
                if aidx not in gdf.index:
                    continue
                for sw_side in ("left", "right"):
                    for sw_p in ("start", "end"):
                        for sw_n in (1, 2, 3):
                            rcol = f"sidewalk_{sw_side}_curbramp_{sw_p}_{sw_n}_geometry"
                            if rcol in gdf.columns:
                                rg = gdf.at[aidx, rcol]
                                if rg is not None and isinstance(rg, BaseGeometry) and not rg.is_empty:
                                    if slot_geom.contains(rg):
                                        osm_ramp_pts.append(rg)

            if len(osm_ramp_pts) >= 2:
                # OSM A: polyline ramp1 → crossing_point → ramp2
                # Pick the two most distant ramps to span the crosswalk
                best_pair_osm = None
                best_dist_osm = 0.0
                for i in range(len(osm_ramp_pts)):
                    for j in range(i + 1, len(osm_ramp_pts)):
                        d = osm_ramp_pts[i].distance(osm_ramp_pts[j])
                        if d > best_dist_osm:
                            best_dist_osm = d
                            best_pair_osm = (osm_ramp_pts[i], osm_ramp_pts[j])
                if best_pair_osm:
                    xwalk_geom = LineString([
                        (best_pair_osm[0].x, best_pair_osm[0].y),
                        (osm_pt.x, osm_pt.y),
                        (best_pair_osm[1].x, best_pair_osm[1].y),
                    ])
                    source = "osm_a"
                    case_counts["OSM_A"] += 1

            elif len(osm_ramp_pts) == 1:
                # OSM B: extend line equally past crossing point from single ramp
                ramp_pt = osm_ramp_pts[0]
                dx = osm_pt.x - ramp_pt.x
                dy = osm_pt.y - ramp_pt.y
                xwalk_geom = LineString([
                    (ramp_pt.x, ramp_pt.y),
                    (osm_pt.x, osm_pt.y),
                    (osm_pt.x + dx, osm_pt.y + dy),
                ])
                source = "osm_b"
                case_counts["OSM_B"] += 1

            else:
                # OSM C: no ramps — extend orthogonal line to sidewalks (or 2× road width)
                bearing_deg = float(gdf.at[crossed_idx, "normalized_bearing"] or 0.0)
                orth_rad = math.radians((bearing_deg + 90.0) % 360.0)
                dx_unit = math.sin(orth_rad)
                dy_unit = math.cos(orth_rad)

                raw_lanes = gdf.at[crossed_idx, "lanes"]
                raw_lw = gdf.at[crossed_idx, "lane_width"]
                lanes_val = float(raw_lanes) if not pd.isna(raw_lanes) else config.default_lanes
                lw_val = float(raw_lw) if not pd.isna(raw_lw) else config.default_lane_width
                max_half_ext_m = lanes_val * lw_val  # total line = 2× road width

                # Project crossing point to UTM
                cp_arr_x = np.array([osm_pt.x], dtype=np.float64)
                cp_arr_y = np.array([osm_pt.y], dtype=np.float64)
                cp_ux_arr, cp_uy_arr = _to_utm.transform(cp_arr_x, cp_arr_y)
                cp_ux, cp_uy = float(cp_ux_arr[0]), float(cp_uy_arr[0])

                # Collect sidewalk geometries from arm segments projected to UTM
                sw_utm_geoms: list[tuple[int, str, Any]] = []
                for aidx in arm_indices:
                    if aidx not in gdf.index:
                        continue
                    for sw_side in ("left", "right"):
                        sw = gdf.at[aidx, f"sidewalk_{sw_side}_geometry"]
                        if sw is None or not isinstance(sw, BaseGeometry) or sw.is_empty:
                            continue
                        sw_utm = gpd.GeoSeries([sw], crs="EPSG:4326").to_crs("EPSG:32610").iloc[0]
                        sw_utm_geoms.append((aidx, sw_side, sw_utm))

                def _extend_to_sw(direction: float) -> tuple[Any, int | None, str | None]:
                    """Extend from crossing point; return (endpoint_wgs, aidx, side)."""
                    end_ux = cp_ux + dx_unit * direction * max_half_ext_m
                    end_uy = cp_uy + dy_unit * direction * max_half_ext_m
                    ray = LineString([(cp_ux, cp_uy), (end_ux, end_uy)])
                    best_d, best_pt_utm, best_aidx, best_side = float("inf"), None, None, None
                    for aidx, sw_side, sw_utm in sw_utm_geoms:
                        if not ray.intersects(sw_utm):
                            continue
                        ix = ray.intersection(sw_utm)
                        if ix.is_empty:
                            continue
                        ix_pts = list(ix.geoms) if hasattr(ix, "geoms") else [ix]
                        for ip in ix_pts:
                            if not isinstance(ip, Point):
                                continue
                            d = math.hypot(ip.x - cp_ux, ip.y - cp_uy)
                            if d < best_d:
                                best_d, best_pt_utm, best_aidx, best_side = d, ip, aidx, sw_side
                    if best_pt_utm is not None:
                        lons, lats = _to_wgs.transform(
                            np.array([best_pt_utm.x], dtype=np.float64),
                            np.array([best_pt_utm.y], dtype=np.float64),
                        )
                        return Point(float(lons[0]), float(lats[0])), best_aidx, best_side
                    # No sidewalk hit — truncate at max extension
                    lons, lats = _to_wgs.transform(
                        np.array([end_ux], dtype=np.float64),
                        np.array([end_uy], dtype=np.float64),
                    )
                    return Point(float(lons[0]), float(lats[0])), None, None

                pt_pos, pos_aidx, pos_side = _extend_to_sw(1.0)
                pt_neg, neg_aidx, neg_side = _extend_to_sw(-1.0)

                xwalk_geom = LineString([
                    (pt_neg.x, pt_neg.y),
                    (osm_pt.x, osm_pt.y),
                    (pt_pos.x, pt_pos.y),
                ])
                source = "osm_c"
                case_counts["OSM_C"] += 1

                # Write new curb ramp slots where the line hit sidewalks
                for hit_pt, hit_aidx, hit_side in [
                    (pt_pos, pos_aidx, pos_side),
                    (pt_neg, neg_aidx, neg_side),
                ]:
                    if hit_aidx is None or hit_side is None:
                        continue
                    hit_pos = "start" if gdf.at[hit_aidx, "start_node_id"] == node_id else "end"
                    for n in (1, 2, 3):
                        gcol = f"sidewalk_{hit_side}_curbramp_{hit_pos}_{n}_geometry"
                        if gcol not in gdf.columns:
                            gdf[gcol] = None
                        val = gdf.at[hit_aidx, gcol]
                        if val is None or not isinstance(val, BaseGeometry):
                            gdf.at[hit_aidx, gcol] = hit_pt  # type: ignore[call-overload]
                            qcol = f"sidewalk_{hit_side}_curbramp_{hit_pos}_{n}_quality"
                            if qcol not in gdf.columns:
                                gdf[qcol] = None
                            gdf.at[hit_aidx, qcol] = "osmpoint"
                            id_col = f"sidewalk_{hit_side}_curbramp_{hit_pos}_{n}_ID"
                            if id_col in gdf.columns:
                                sgid = gdf.at[hit_aidx, "street_grid_id"]
                                gdf.at[hit_aidx, id_col] = f"{sgid}_CR_{hit_side}_{hit_pos}_{n}"
                            break

        # ── OSM D: Footway crossing chain (spatial-index candidate query) ───
        footway_in_slot = []
        if footway_sindex is not None:
            cand_pos = list(footway_sindex.query(slot_geom, predicate="intersects"))
            for cp in cand_pos:
                fidx = footway_gs.index[cp]
                fg = footway_gs.iloc[cp]
                if fg is None or not isinstance(fg, BaseGeometry) or fg.is_empty:
                    continue
                if fg.length < 0.0003 and slot_geom.contains(fg):
                    footway_in_slot.append(fidx)
                elif slot_geom.intersects(fg):
                    try:
                        inter = fg.intersection(slot_geom)
                        if fg.length > 0 and inter.length / fg.length >= 0.8:
                            footway_in_slot.append(fidx)
                    except Exception:
                        pass

        if len(footway_in_slot) >= 1:
            # Chain: merge footway geometries
            chain_coords = []
            for fidx in footway_in_slot:
                chain_coords.extend(_flatten_coords(cast(BaseGeometry, gdf.at[fidx, "street_geometry"])))
            if len(chain_coords) >= 2:
                xwalk_geom = LineString(chain_coords)
                source = "osm_d"
                case_counts["OSM_D"] += 1
                absorbed_footway_rows.update(footway_in_slot)

        # ── Topo A: Sidewalk crosses street ───────────────────────────
        if xwalk_geom is None:
            street_geom = cast(BaseGeometry, gdf.at[crossed_idx, "street_geometry"])
            for side in ("left", "right"):
                for aidx in arm_indices:
                    if aidx not in gdf.index:
                        continue
                    sw = gdf.at[aidx, f"sidewalk_{side}_geometry"]
                    if sw is None or not isinstance(sw, BaseGeometry):
                        continue
                    if sw.crosses(street_geom):
                        try:
                            ix = sw.intersection(street_geom)
                            if not ix.is_empty:
                                xwalk_geom = LineString([
                                    _flatten_coords(sw)[0],
                                    _flatten_coords(sw)[-1],
                                ]) if sw.length < 0.0003 else sw
                                source = "topo_a"
                                case_counts["Topo_A"] += 1
                                break
                        except Exception:
                            pass
                if xwalk_geom is not None:
                    break

        # ── Topo B: Curb ramp pair (any arm in slot) ─────────────────
        if xwalk_geom is None:
            ramp_pts_in_slot = []
            for aidx in arm_indices:
                if aidx not in gdf.index:
                    continue
                for side in ("left", "right"):
                    for p in ("start", "end"):
                        for n in (1, 2, 3):
                            col = f"sidewalk_{side}_curbramp_{p}_{n}_geometry"
                            if col in gdf.columns:
                                rg = gdf.at[aidx, col]
                                if rg is not None and isinstance(rg, BaseGeometry):
                                    if slot_geom.contains(rg):
                                        ramp_pts_in_slot.append(rg)

            if len(ramp_pts_in_slot) >= 2:
                xwalk_geom = LineString([
                    (ramp_pts_in_slot[0].x, ramp_pts_in_slot[0].y),
                    (ramp_pts_in_slot[1].x, ramp_pts_in_slot[1].y),
                ])
                source = "topo_b"
                case_counts["Topo_B"] += 1

        # ── Topo C: Sidewalk endpoint pair (fallback) ────────────────
        if xwalk_geom is None:
            sw_endpoints = []
            for aidx in arm_indices:
                if aidx not in gdf.index:
                    continue
                for side in ("left", "right"):
                    sw = gdf.at[aidx, f"sidewalk_{side}_geometry"]
                    if sw is None or not isinstance(sw, BaseGeometry):
                        continue
                    coords = _flatten_coords(sw)
                    if not coords:
                        continue
                    for pt_c in (coords[0], coords[-1]):
                        pt = Point(pt_c)
                        if slot_geom.contains(pt):
                            sw_endpoints.append(pt)

            if len(sw_endpoints) >= 2:
                # Pick the two most distant endpoints (likely on opposite sides)
                best_pair = None
                best_dist = 0
                for i in range(len(sw_endpoints)):
                    for j in range(i + 1, len(sw_endpoints)):
                        d = sw_endpoints[i].distance(sw_endpoints[j])
                        if d > best_dist:
                            best_dist = d
                            best_pair = (sw_endpoints[i], sw_endpoints[j])
                if best_pair:
                    xwalk_geom = LineString([
                        (best_pair[0].x, best_pair[0].y),
                        (best_pair[1].x, best_pair[1].y),
                    ])
                    source = "topo_c"
                    case_counts["Topo_C"] += 1

        # ── Write crosswalk to GDF ────────────────────────────────────
        if xwalk_geom is not None:
            xwalk_id += 1
            gdf.at[crossed_idx, f"crosswalk_{pos}_geometry"] = xwalk_geom  # type: ignore[call-overload]
            xwalk_sgid = gdf.at[crossed_idx, "street_grid_id"]
            gdf.at[crossed_idx, f"crosswalk_{pos}_id"] = f"{xwalk_sgid}_XW_{pos}"
            gdf.at[crossed_idx, f"crosswalk_{pos}_source"] = source
            # Grid IDs: tuple of the two sidewalk grid IDs the crosswalk connects
            sw_l_gid = gdf.at[crossed_idx, "sidewalk_left_grid_ID"]
            sw_r_gid = gdf.at[crossed_idx, "sidewalk_right_grid_ID"]
            grid_pair = (
                str(sw_l_gid) if pd.notna(sw_l_gid) else None,
                str(sw_r_gid) if pd.notna(sw_r_gid) else None,
            )
            gdf.at[crossed_idx, f"crosswalk_{pos}_grid_ids"] = grid_pair

    # ── Populate crosswalk attributes from cache ──────────────────────
    if not crosswalk_cache.empty and len(crosswalk_cache) > 0:
        xwalk_sindex = crosswalk_cache.sindex
        for pos in ("start", "end"):
            geom_col = f"crosswalk_{pos}_geometry"
            geom_arr = np.asarray(gdf[geom_col])
            active_mask = np.array([isinstance(v, BaseGeometry) for v in geom_arr], dtype=bool)
            active_index = gdf.index[active_mask]
            for idx in active_index:
                xg = gdf.at[idx, geom_col]
                # Find nearest crossing node by true distance, not index order
                hits = list(xwalk_sindex.query(xg.buffer(0.0002), predicate="intersects"))
                if not hits:
                    continue
                xg_centroid = xg.centroid
                nearest = min(hits, key=lambda i: crosswalk_cache.geometry.iloc[i].distance(xg_centroid))
                cache_row = crosswalk_cache.iloc[nearest]
                for attr in ("type", "controlled", "marked", "markings", "signals",
                              "island", "kerb", "tactile_paving", "traffic_calming", "continuous"):
                    val = cache_row.get(attr)
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        col = f"crosswalk_{pos}_{attr}"
                        if col in gdf.columns:
                            gdf.at[idx, col] = val

    # ── Fix 1: Null sidewalk/curbramp geometry on absorbed footway rows ──
    # Footway rows consumed by Case D had their geometry merged into a
    # crosswalk.  Clear their sidewalk and curb ramp slots so the geometry
    # doesn't persist as a duplicate alongside the crosswalk.
    n_absorbed_cleared = 0
    _existing_cols = set(gdf.columns)
    for ar_idx in absorbed_footway_rows:
        if ar_idx not in gdf.index:
            continue
        for side in ("left", "right"):
            sw_col = f"sidewalk_{side}_geometry"
            if sw_col in _existing_cols and not pd.isna(gdf.at[ar_idx, sw_col]):
                gdf.at[ar_idx, sw_col] = None  # type: ignore[call-overload]
                n_absorbed_cleared += 1
            for rp in ("start", "end"):
                for sl in ("1", "2", "3"):
                    rc = f"sidewalk_{side}_curbramp_{rp}_{sl}_geometry"
                    if rc in _existing_cols and not pd.isna(gdf.at[ar_idx, rc]):
                        gdf.at[ar_idx, rc] = None  # type: ignore[call-overload]

    # ── Fix 2: Post-crosswalk stub cleanup per intersection hull ──────
    # After crosswalk slots are filled, sidewalk stubs >90% inside a hull
    # are artifacts (approach footways, corner connectors).  Clear them and
    # also null any curb ramp points sitting inside the hull on sidewalks
    # that extend beyond it.
    _SW_STUB_HULL_FRAC = 0.90
    n_stubs_cleared = 0
    n_ramps_inside_cleared = 0
    node_hull_map: dict[Any, BaseGeometry] = dict(zip(hulls["node_id"], hulls.geometry))
    start_ids = gdf["start_node_id"].values
    end_ids = gdf["end_node_id"].values
    idx_arr = gdf.index.values

    for side in ("left", "right"):
        sw_geom_col = f"sidewalk_{side}_geometry"
        if sw_geom_col not in _existing_cols:
            continue
        sw_geom_vals = gdf[sw_geom_col].values
        ramp_cols = [
            f"sidewalk_{side}_curbramp_{rp}_{sl}_geometry"
            for rp in ("start", "end") for sl in ("1", "2", "3")
        ]
        ramp_cols = [c for c in ramp_cols if c in _existing_cols]

        for pos_i in range(len(idx_arr)):
            sw_g = sw_geom_vals[pos_i]
            if not isinstance(sw_g, BaseGeometry) or sw_g.is_empty or sw_g.length < 1e-6:
                continue
            # Collect adjacent hull(s)
            adj_hulls = []
            for nid in (start_ids[pos_i], end_ids[pos_i]):
                h = node_hull_map.get(nid)
                if h is not None:
                    adj_hulls.append(h)
            if not adj_hulls:
                continue

            max_frac = 0.0
            best_hull = None
            for h in adj_hulls:
                try:
                    frac = sw_g.intersection(h).length / sw_g.length
                except Exception:
                    continue
                if frac > max_frac:
                    max_frac = frac
                    best_hull = h

            row_idx = idx_arr[pos_i]
            if max_frac >= _SW_STUB_HULL_FRAC:
                # Sidewalk is a stub — clear geometry and all curb ramps
                gdf.at[row_idx, sw_geom_col] = None  # type: ignore[call-overload]
                for rc in ramp_cols:
                    if not pd.isna(gdf.at[row_idx, rc]):
                        gdf.at[row_idx, rc] = None  # type: ignore[call-overload]
                n_stubs_cleared += 1
            elif best_hull is not None:
                # Sidewalk extends beyond — clear only curb ramps inside hull
                for rc in ramp_cols:
                    rv = gdf.at[row_idx, rc]
                    if isinstance(rv, Point) and best_hull.covers(rv):
                        gdf.at[row_idx, rc] = None  # type: ignore[call-overload]
                        n_ramps_inside_cleared += 1

    total = sum(case_counts.values())
    log.info(
        "Step 16: %d crosswalks created "
        "(OSM_A=%d OSM_B=%d OSM_C=%d OSM_D=%d Topo_A=%d Topo_B=%d Topo_C=%d), "
        "%d absorbed footway slots cleared, %d post-xwalk stubs cleared, "
        "%d hull-interior ramps cleared",
        total,
        case_counts["OSM_A"], case_counts["OSM_B"], case_counts["OSM_C"], case_counts["OSM_D"],
        case_counts["Topo_A"], case_counts["Topo_B"], case_counts["Topo_C"],
        n_absorbed_cleared, n_stubs_cleared, n_ramps_inside_cleared,
    )
    return gdf


# ---------------------------------------------------------------------------
# Step 10 — Consolidate orphaned facility rows into road slots
# ---------------------------------------------------------------------------
def step_10_consolidate_orphaned_facilities(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Final pass: match every remaining standalone footway / cycleway row into
    its nearest parallel road row's facility slot, then remove all such rows
    so every block shares a single road row.

    Rows that cannot be matched (no parallel road within _MATCH_RADIUS_M) are
    also removed; their count is logged as a warning.
    """
    log.info("Step 10: Consolidating orphaned facility rows into road slots")

    hw = gdf["highway"].astype(str).str.lower().str.strip()
    bicycle_col = _safe_col(gdf, "bikeway_left_1_permitted").astype(str).str.lower()
    is_road = hw.isin(_ROAD_HW)
    is_cycleway = hw.isin(_BIKEWAY_HW) | (
        hw.isin({"path", "footway"}) & bicycle_col.isin({"designated", "yes"})
    )
    # Exclude crossing-type ways — these are crosswalk geometries, not parallel
    # sidewalk facilities, and must survive until step 16's Case D can use them.
    footway_tag = _safe_col(gdf, "footway").astype(str).str.lower()
    is_crossing = hw.isin({"crossing"}) | footway_tag.isin({"crossing"})
    is_footway = hw.isin(_FOOTWAY_HW) & ~is_cycleway & ~is_crossing
    is_facility = is_footway | is_cycleway

    fac_indices = gdf.index[is_facility]
    n_fac = len(fac_indices)
    if n_fac == 0 or not is_road.any():
        log.info("  No facility rows present — nothing to consolidate")
        return gdf

    utm_street = gdf["street_geometry"].to_crs("EPSG:32610")
    road_idx = gdf.index[is_road].tolist()
    road_geoms_utm = utm_street[is_road]
    road_sindex = road_geoms_utm.sindex
    road_bearings = gdf.loc[is_road, "normalized_bearing"].to_dict()
    road_names = gdf.loc[is_road, "name"].to_dict()

    # Pre-populate used sets from slots already occupied by step 09
    foot_used: set[tuple[int, str]] = set()
    bike_used: set[tuple[int, str, str]] = set()
    road_df = gdf.loc[road_idx]
    for side in ("left", "right"):
        gcol = f"sidewalk_{side}_geometry"
        if gcol in gdf.columns:
            geom_arr = road_df[gcol].values
            has_geom = np.array(
                [isinstance(g, BaseGeometry) and not g.is_empty for g in geom_arr], dtype=bool
            )
            for ri in road_df.index[has_geom]:
                foot_used.add((ri, side))
        for slot in ("1", "2"):
            bcol = f"bikeway_{side}_{slot}_geometry"
            if bcol in gdf.columns:
                geom_arr = road_df[bcol].values
                has_geom = np.array(
                    [isinstance(g, BaseGeometry) and not g.is_empty for g in geom_arr], dtype=bool
                )
                for ri in road_df.index[has_geom]:
                    bike_used.add((ri, side, slot))

    def _slot_is_empty(row_idx: int, col: str) -> bool:
        """Return True if the slot value is absent/null (handles list-valued cells)."""
        if col not in gdf.columns:
            return False
        val = _first_if_list(gdf.at[row_idx, col])
        return val is None or pd.isna(val)

    matched_foot = matched_bike = 0
    matched_indices: set[int] = set()

    for fac_idx in tqdm(fac_indices, desc="Consolidating facilities", unit="seg"):
        fac_geom = utm_street.at[fac_idx]
        if fac_geom is None or fac_geom.is_empty:
            continue

        fac_type = "bike" if is_cycleway.at[fac_idx] else "foot"
        fc = _flatten_coords(fac_geom)
        if len(fc) < 2:
            continue

        fac_bear = math.degrees(math.atan2(fc[-1][0] - fc[0][0], fc[-1][1] - fc[0][1])) % 360
        buf = fac_geom.buffer(_MATCH_RADIUS_M)
        candidates = list(road_sindex.query(buf, predicate="intersects"))
        if not candidates:
            continue

        best_road, best_score, best_side = None, float("inf"), "left"
        for cand_pos in candidates:
            cand_idx = road_idx[cand_pos]
            road_geom = road_geoms_utm.iloc[cand_pos]
            dist = fac_geom.distance(road_geom)
            rb = road_bearings.get(cand_idx, 0.0)
            axis_diff = abs((fac_bear % 180) - (rb % 180))
            if axis_diff > 90:
                axis_diff = 180 - axis_diff
            if axis_diff > _PARALLEL_THRESHOLD_DEG:
                continue
            fac_name = gdf.at[fac_idx, "name"]
            road_name = road_names.get(cand_idx)
            name_bonus = -10.0 if (
                fac_name and road_name
                and str(fac_name).strip() == str(road_name).strip()
            ) else 0.0
            score = dist + name_bonus
            fac_mid = fac_geom.interpolate(0.5, normalized=True)
            proj_dist = road_geom.project(fac_mid)
            proj_pt = road_geom.interpolate(proj_dist)
            rc = list(road_geom.coords)
            seg_i = max(0, min(int(proj_dist / road_geom.length * (len(rc) - 1)), len(rc) - 2))
            rx = rc[seg_i + 1][0] - rc[seg_i][0]
            ry = rc[seg_i + 1][1] - rc[seg_i][1]
            fx, fy = fac_mid.x - proj_pt.x, fac_mid.y - proj_pt.y
            side = "left" if (rx * fy - ry * fx) > 0 else "right"
            if score < best_score:
                best_score = score
                best_road = cand_idx
                best_side = side

        if best_road is None:
            continue

        fac_wgs = gdf.at[fac_idx, "street_geometry"]
        _raw_surface = _first_if_list(gdf.at[fac_idx, "surface"]) if "surface" in gdf.columns else None
        fac_surface = None if (_raw_surface is None or pd.isna(_raw_surface)) else _raw_surface

        if fac_type == "foot":
            if (best_road, best_side) not in foot_used:
                gcol = f"sidewalk_{best_side}_geometry"
                if gcol in gdf.columns:
                    gdf.at[best_road, gcol] = fac_wgs  # type: ignore[call-overload]
                gdf.at[best_road, f"sidewalk_{best_side}_quality"] = "separate"
                gdf.at[best_road, f"sidewalk_{best_side}_offset"] = "no"
                pcol = f"sidewalk_{best_side}_presence"
                if _slot_is_empty(best_road, pcol):
                    gdf.at[best_road, pcol] = "separate"
                if fac_surface is not None:
                    sc = f"sidewalk_{best_side}_surface"
                    if _slot_is_empty(best_road, sc):
                        gdf.at[best_road, sc] = fac_surface  # type: ignore[call-overload]
                foot_used.add((best_road, best_side))
                matched_indices.add(fac_idx)
                matched_foot += 1
        else:
            for slot in ("1", "2"):
                if (best_road, best_side, slot) not in bike_used:
                    gcol = f"bikeway_{best_side}_{slot}_geometry"
                    if gcol in gdf.columns:
                        gdf.at[best_road, gcol] = fac_wgs  # type: ignore[call-overload]
                    gdf.at[best_road, f"bikeway_{best_side}_{slot}_quality"] = "separate"
                    gdf.at[best_road, f"bikeway_{best_side}_{slot}_offset"] = "no"
                    if fac_surface is not None:
                        sc = f"bikeway_{best_side}_{slot}_surface"
                        if _slot_is_empty(best_road, sc):
                            gdf.at[best_road, sc] = fac_surface  # type: ignore[call-overload]
                    bike_used.add((best_road, best_side, slot))
                    matched_indices.add(fac_idx)
                    matched_bike += 1
                    break

    unmatched_count = n_fac - len(matched_indices)
    if unmatched_count:
        log.info(
            "  Step 10: %d facility rows had no matching road within %.0f m — preserved",
            unmatched_count, _MATCH_RADIUS_M,
        )

    gdf = gdf.drop(index=list(matched_indices))
    log.info(
        "  Step 10: Consolidated %d footways, %d bikeways; removed %d matched facility rows",
        matched_foot, matched_bike, len(matched_indices),
    )
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 17 — Enrich curb ramps with government data
# ─────────────────────────────────────────────────────────────────────────────

def step_17_enrich_curb_ramps(
    gdf: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> gpd.GeoDataFrame:
    """Enrich existing curb-ramp slots (or create new ones) from government CSV data."""

    # ── Graceful skip checks ────────────────────────────────────────────────
    if not config.curb_ramp_data_path:
        log.info("Step 17: skipped — curb_ramp_data_path is empty")
        return gdf

    import os
    if not os.path.isfile(config.curb_ramp_data_path):
        log.info("Step 17: skipped — file not found: %s", config.curb_ramp_data_path)
        return gdf

    df = pd.read_csv(config.curb_ramp_data_path)
    df = df[df["crExist"].astype(str) == "1"]

    if len(df) == 0:
        log.info("Step 17: skipped — zero rows with crExist==1")
        return gdf

    # ── Coordinate resolution ───────────────────────────────────────────────
    has_latlon = ("Latitude" in df.columns and "Longitude" in df.columns)
    has_xy = ("xLoc" in df.columns and "yLoc" in df.columns)

    # Build reprojector for State Plane fallback
    sp_transformer = None
    if has_xy:
        from pyproj import Transformer as _Tf
        sp_transformer = _Tf.from_crs(2227, 4326, always_xy=True)

    csv_points: list[tuple[float, float, Any]] = []  # (lon, lat, row)

    # Resolve coordinates vectorized: Priority 1 = Lat/Lon, Priority 2 = State Plane xLoc/yLoc
    resolved_lons = pd.Series(pd.NA, index=df.index, dtype="Float64")
    resolved_lats = pd.Series(pd.NA, index=df.index, dtype="Float64")

    if has_latlon:
        lat_num = pd.to_numeric(df["Latitude"], errors="coerce")
        lon_num = pd.to_numeric(df["Longitude"], errors="coerce")
        valid_ll = lat_num.notna() & lon_num.notna()
        resolved_lats[valid_ll] = lat_num[valid_ll]
        resolved_lons[valid_ll] = lon_num[valid_ll]

    if has_xy and sp_transformer is not None:
        still_missing = resolved_lons.isna()
        if still_missing.any():
            x_num = pd.to_numeric(df.loc[still_missing, "xLoc"], errors="coerce")
            y_num = pd.to_numeric(df.loc[still_missing, "yLoc"], errors="coerce")
            xy_valid = x_num.notna() & y_num.notna()
            if xy_valid.any():
                proj_lons, proj_lats = sp_transformer.transform(
                    x_num[xy_valid].values.astype(np.float64),
                    y_num[xy_valid].values.astype(np.float64),
                )
                resolved_lons.iloc[np.where(still_missing)[0][xy_valid.values]] = proj_lons
                resolved_lats.iloc[np.where(still_missing)[0][xy_valid.values]] = proj_lats

    valid_mask = resolved_lons.notna() & resolved_lats.notna()
    skipped_no_coords = int((~valid_mask).sum())
    for i in np.where(valid_mask.values)[0]:
        csv_points.append((float(resolved_lons.iloc[i]), float(resolved_lats.iloc[i]), df.iloc[i]))

    if not csv_points:
        log.info("Step 17: skipped — no CSV rows with valid coordinates")
        return gdf

    # ── Build spatial index of existing ramp slots (UTM) ────────────────────
    # Iterate per column (12 combos) rather than per row × combo — lets us batch
    # the UTM transform once per column instead of once per valid cell.
    ramp_slot_list: list[tuple[int, str, str, int, float, float]] = []
    ramp_slot_points: list[Point] = []

    for side in ("left", "right"):
        for pos in ("start", "end"):
            for n in (1, 2, 3):
                col = f"sidewalk_{side}_curbramp_{pos}_{n}_geometry"
                if col not in gdf.columns:
                    continue
                geom_arr = gdf[col].values
                valid_mask = np.array(
                    [isinstance(g, BaseGeometry) and not g.is_empty for g in geom_arr],
                    dtype=bool,
                )
                if not valid_mask.any():
                    continue
                valid_idx = gdf.index[valid_mask]
                valid_geoms = geom_arr[valid_mask]
                g_xs = np.array([g.x for g in valid_geoms], dtype=np.float64)
                g_ys = np.array([g.y for g in valid_geoms], dtype=np.float64)
                uxs, uys = _to_utm.transform(g_xs, g_ys)
                for i, idx in enumerate(valid_idx):
                    ux, uy = float(uxs[i]), float(uys[i])
                    ramp_slot_list.append((idx, side, pos, n, ux, uy))
                    ramp_slot_points.append(Point(ux, uy))

    ramp_tree = STRtree(ramp_slot_points) if ramp_slot_points else None

    # ── Build road geometry index (UTM) ─────────────────────────────────────
    hw = gdf["highway"].astype(str).str.lower().str.strip()
    is_road = hw.isin(_ROAD_HW)
    road_indices = gdf.index[is_road].tolist()

    road_geoms_utm: list[BaseGeometry] = []
    road_geoms_valid_idx: list[int] = []  # parallel to road_geoms_utm
    for ri in road_indices:
        sg = gdf.at[ri, "street_geometry"]
        if sg is None or not isinstance(sg, BaseGeometry) or sg.is_empty:
            continue
        utm_g = _project_to_utm(sg)
        road_geoms_utm.append(utm_g)
        road_geoms_valid_idx.append(ri)

    road_tree = STRtree(road_geoms_utm) if road_geoms_utm else None

    # ── Per-CSV-ramp loop ───────────────────────────────────────────────────
    matched = 0
    new_ramp = 0
    skipped_no_slot = 0
    radius = config.curb_ramp_match_radius_m

    for lon, lat, row in csv_points:
        ux, uy = _to_utm.transform(lon, lat)
        csv_utm = Point(float(ux), float(uy))

        # Try matching to existing ramp slot
        found_match = False
        if ramp_tree is not None and ramp_slot_points:
            nearby_idxs = ramp_tree.query(csv_utm.buffer(radius))
            if len(nearby_idxs) > 0:
                # Find nearest
                best_dist = float("inf")
                best_i = -1
                for ni in nearby_idxs:
                    d = csv_utm.distance(ramp_slot_points[ni])
                    if d < best_dist:
                        best_dist = d
                        best_i = ni

                road_idx, side, pos, n, _, _ = ramp_slot_list[best_i]
                prefix = f"sidewalk_{side}_curbramp_{pos}_{n}"
                gdf.at[road_idx, f"public_data_id_{prefix}"] = row.get("locID")
                gdf.at[road_idx, f"{prefix}_returnloc"] = row.get("curbReturnLoc")
                gdf.at[road_idx, f"{prefix}_returnposition"] = row.get("positionOnReturn")
                gdf.at[road_idx, f"{prefix}_condition_score"] = row.get("conditionScore")
                gdf.at[road_idx, f"{prefix}_quality"] = "government"
                matched += 1
                found_match = True

        if found_match:
            continue

        # No existing slot match — assign to nearest road
        if road_tree is None or not road_geoms_utm:
            skipped_no_slot += 1
            continue

        near_road_idxs = road_tree.query(csv_utm.buffer(radius * 10))
        if len(near_road_idxs) == 0:
            # Fallback: nearest geometry overall
            near_road_idxs = road_tree.query(csv_utm.buffer(500))
        if len(near_road_idxs) == 0:
            skipped_no_slot += 1
            continue

        # Find nearest road by distance
        best_road_dist = float("inf")
        best_road_local_idx = -1
        for ri_local in near_road_idxs:
            d = csv_utm.distance(road_geoms_utm[ri_local])
            if d < best_road_dist:
                best_road_dist = d
                best_road_local_idx = ri_local

        road_idx = road_geoms_valid_idx[best_road_local_idx]
        road_utm_geom = road_geoms_utm[best_road_local_idx]

        # ── Determine pos (start/end) ──────────────────────────────────────
        start_node_geom = gdf.at[road_idx, "start_node_geometry"]
        end_node_geom = gdf.at[road_idx, "end_node_geometry"]

        d_start = float("inf")
        d_end = float("inf")
        if isinstance(start_node_geom, Point) and not start_node_geom.is_empty:
            snx, sny = _to_utm.transform(start_node_geom.x, start_node_geom.y)
            d_start = csv_utm.distance(Point(float(snx), float(sny)))
        if isinstance(end_node_geom, Point) and not end_node_geom.is_empty:
            enx, eny = _to_utm.transform(end_node_geom.x, end_node_geom.y)
            d_end = csv_utm.distance(Point(float(enx), float(eny)))

        pos = "start" if d_start <= d_end else "end"

        # ── Determine side (left/right) via cross product ───────────────────
        # Project csv_utm onto road UTM geometry, get nearest segment vector
        proj_dist = road_utm_geom.project(csv_utm)
        proj_point = road_utm_geom.interpolate(proj_dist)

        # Get the road segment tangent vector at projection point
        coords = list(road_utm_geom.coords)
        # Find segment containing the projection
        seg_vec = None
        accumulated = 0.0
        for i in range(len(coords) - 1):
            seg_len = Point(coords[i]).distance(Point(coords[i + 1]))
            if accumulated + seg_len >= proj_dist or i == len(coords) - 2:
                seg_vec = (coords[i + 1][0] - coords[i][0],
                           coords[i + 1][1] - coords[i][1])
                break
            accumulated += seg_len

        if seg_vec is None:
            seg_vec = (coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])

        # Offset vector: projection point → csv_utm
        offset_vec = (csv_utm.x - proj_point.x, csv_utm.y - proj_point.y)

        # Cross product: seg_vec × offset_vec
        cross = seg_vec[0] * offset_vec[1] - seg_vec[1] * offset_vec[0]
        side = "left" if cross > 0 else "right"

        # ── Find first available slot n ─────────────────────────────────────
        slot_n = None
        for n in (1, 2, 3):
            col = f"sidewalk_{side}_curbramp_{pos}_{n}_geometry"
            if col not in gdf.columns:
                gdf[col] = None
            val = gdf.at[road_idx, col]
            if val is None or (not isinstance(val, BaseGeometry)) or (isinstance(val, BaseGeometry) and val.is_empty):
                # Also check via pd.isna for scalar None/NaN
                slot_n = n
                break
            elif pd.isna(val):
                slot_n = n
                break

        if slot_n is None:
            skipped_no_slot += 1
            continue

        n = slot_n
        prefix = f"sidewalk_{side}_curbramp_{pos}_{n}"

        # Ensure columns exist
        for suffix in ("_geometry", "_ID", "_quality", "_returnloc",
                       "_returnposition", "_condition_score"):
            c = prefix + suffix
            if c not in gdf.columns:
                gdf[c] = None
        pub_col = f"public_data_id_{prefix}"
        if pub_col not in gdf.columns:
            gdf[pub_col] = None

        # Write values
        gdf.at[road_idx, f"{prefix}_geometry"] = Point(lon, lat)
        _sgid = gdf.at[road_idx, "street_grid_id"]
        gdf.at[road_idx, f"{prefix}_ID"] = f"{_sgid}_CR_{side}_{pos}_{n}"
        gdf.at[road_idx, pub_col] = row.get("locID")
        gdf.at[road_idx, f"{prefix}_returnloc"] = row.get("curbReturnLoc")
        gdf.at[road_idx, f"{prefix}_returnposition"] = row.get("positionOnReturn")
        gdf.at[road_idx, f"{prefix}_condition_score"] = row.get("conditionScore")
        gdf.at[road_idx, f"{prefix}_quality"] = "government"
        new_ramp += 1

    log.info(
        "Step 17: %d government ramps matched to existing slots, %d new slots created, "
        "%d skipped (no coords), %d skipped (no slot)",
        matched, new_ramp, skipped_no_coords, skipped_no_slot,
    )
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(config: PipelineConfig) -> gpd.GeoDataFrame:
    """Execute the full Proximity pipeline, returning the completed GeoDataFrame."""

    log.info("═══ Proximity Pipeline: %s ═══", config.place_name)

    # Phase 1 — Street network foundation
    gdf = step_01_create_parquet(config)
    gdf, crosswalk_cache, G, nodes, edges_reset, walk_edges, bike_edges = step_02_load_streets(gdf, config)
    gdf = step_03_split_deflections(gdf, config)
    gdf = step_04_bearings_and_grid_ids(gdf, config, G=G, nodes=nodes)
    gdf = step_05_traffic_calming(gdf, config)
    gdf = step_06_usgs_elevation(gdf, config)

    # Phase 2 — Sidewalks and bikelanes
    gdf = step_07_populate_facilities(gdf, config, edges_reset=edges_reset,
                                      walk_edges=walk_edges, bike_edges=bike_edges)
    gdf = step_08_swap_left_right(gdf, config)
    gdf = step_09_match_separate_facilities(gdf, config)
    gdf = step_10_consolidate_orphaned_facilities(gdf, config)
    gdf = step_11_facility_grid_ids(gdf, config)
    gdf = step_11b_compute_neighbors(gdf, config)
    gdf = step_12_snap_endpoints(gdf, config)

    # Phase 3 — Intersection analysis
    gdf, hulls = step_13_curb_ramps_and_hulls(gdf, config)
    gdf = step_13b_clear_hull_artifact_sidewalks(gdf, hulls)
    hulls = step_14_merge_hulls(hulls, config)
    gdf = step_15_crosswalk_slots(gdf, hulls, config)
    gdf = step_16_crosswalk_geometries(gdf, hulls, crosswalk_cache, config)

    # Phase 4 — Government data enrichment
    gdf = step_17_enrich_curb_ramps(gdf, config)

    # Pre-write type normalization for object columns:
    #   - Schema-defined list columns (Str|List[Str] or Float|List[Float]):
    #     wrap bare scalars in a single-element list → uniformly list<T>.
    #   - All other object columns: coerce any stray lists to scalar (take first
    #     element) → OSMnx sometimes returns lists for tags that are Str|null.
    _SCHEMA_LIST_COLS: set[str] = {
        "surface",
        "sidewalk_left_surface",    "sidewalk_right_surface",
        "bikeway_left_1_type",      "bikeway_right_1_type",
        "bikeway_left_1_surface",   "bikeway_right_1_surface",
        "bikeway_left_1_permitted", "bikeway_right_1_permitted",
        "bikeway_left_2_type",      "bikeway_right_2_type",
        "bikeway_left_2_surface",   "bikeway_right_2_surface",
        "bikeway_left_2_permitted", "bikeway_right_2_permitted",
        "bikeway_right_2_incline",
    }

    # Serialize list/dict columns to JSON strings so PyArrow writes them as
    # plain strings and DuckDB WASM can read them without STRUCT inference issues.
    import json as _json_ser

    class _NumpyEncoder(_json_ser.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'item'):  # numpy scalar (int64, float64, etc.)
                return obj.item()
            return super().default(obj)

    for _ser_col in ("street_feature_types", "street_feature_attributes",
                     "public_data_id_street_feature"):
        if _ser_col in gdf.columns:
            gdf[_ser_col] = gdf[_ser_col].apply(
                lambda v: _json_ser.dumps(v, cls=_NumpyEncoder) if isinstance(v, (list, dict)) else v
            )

    geom_col_set = set(GEOMETRY_COLUMNS)
    for col in gdf.columns:
        if gdf[col].dtype != object or col in geom_col_set:
            continue
        has_list = any(type(v) is list for v in gdf[col])
        if not has_list:
            continue
        if col in _SCHEMA_LIST_COLS:
            # Uniform list: wrap non-null scalars, keep lists, null → None.
            # Must check pd.isna() — np.nan/pd.NA are not None but must not be wrapped.
            gdf[col] = gdf[col].apply(
                lambda v: None if (not isinstance(v, list) and pd.isna(v))
                          else (v if isinstance(v, list) else [v])
            )
        else:
            # Scalar column: coerce lists → first element
            gdf[col] = gdf[col].apply(_first_if_list)

    # Cast all non-active geometry columns to GeoSeries so geopandas registers them
    # as geometry columns in the parquet file. Plain object columns containing
    # shapely objects are not recognized by PyArrow and cause ArrowInvalid errors.
    active_geom = gdf.geometry.name
    for col in GEOMETRY_COLUMNS:
        if col in gdf.columns and col != active_geom:
            gdf[col] = gpd.GeoSeries(gdf[col], crs="EPSG:4326")

    # Write output with grid origin metadata for RollTracks viewport-based loading
    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    import pyarrow.parquet as _pq
    import pyarrow as _pa

    # Write initial parquet
    gdf.to_parquet(config.output_path)

    # Append proximity_grid_origin to parquet key-value metadata
    grid_origin_x = gdf.attrs.get("grid_origin_x")
    grid_origin_y = gdf.attrs.get("grid_origin_y")
    grid_cell_size = gdf.attrs.get("grid_cell_size")
    if grid_origin_x is not None and grid_origin_y is not None:
        table = _pq.read_table(config.output_path)
        existing_meta = table.schema.metadata or {}
        existing_meta[b"proximity_grid_origin"] = _json.dumps({
            "x": grid_origin_x, "y": grid_origin_y, "cell_size": grid_cell_size,
        }).encode()
        table = table.replace_schema_metadata(existing_meta)
        _pq.write_table(table, config.output_path)

    log.info("═══ Pipeline complete → %s (%d rows) ═══", config.output_path, len(gdf))

    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser()
    _parser.add_argument("--skip-existing", action="store_true",
                         help="Skip regions whose output parquet already exists")
    _args = _parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _configs = [
        PipelineConfig(
            place_name="San Francisco County, California, USA",
            output_path="Output/San_Francisco_County_California_USA_network.parquet",
        ),
        PipelineConfig(
            place_name="Alameda County, California, USA",
            output_path="Output/Alameda_County_California_USA_network.parquet",
        ),
    ]

    for _cfg in _configs:
        if _args.skip_existing and Path(_cfg.output_path).exists():
            log.info("Skipping %s (parquet already exists)", _cfg.place_name)
            continue
        run_pipeline(_cfg)
