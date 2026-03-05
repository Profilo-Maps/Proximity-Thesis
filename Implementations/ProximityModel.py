import math
import pickle
import numpy as np
import osmnx as ox
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

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

# --- Cities Config ---
CITIES_CONFIG = [
    {"place": "San Francisco County, California, USA", "default_lane_width_m": 3.5},
    {"place": "Alameda County, California, USA",       "default_lane_width_m": 3.5},
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


def split_deflected_segments(
    edges_reset: gpd.GeoDataFrame,
    deflection_threshold_deg: float = 45.0,
) -> gpd.GeoDataFrame:
    """Split segments at internal vertices where the deflection angle exceeds
    the threshold.  Returns an augmented GeoDataFrame with split rows appended
    and a fresh RangeIndex.

    Split node IDs are negative sequential integers: -1, -2, ...
    Both halves share all attributes of the parent except u/v/geometry.
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

        # Find the internal vertex with the largest deflection
        max_defl = 0.0
        max_defl_idx = -1
        for i in range(1, len(coords) - 1):
            # Incoming and outgoing vectors
            dx_in = coords[i][0] - coords[i - 1][0]
            dy_in = coords[i][1] - coords[i - 1][1]
            dx_out = coords[i + 1][0] - coords[i][0]
            dy_out = coords[i + 1][1] - coords[i][1]
            len_in = math.hypot(dx_in, dy_in)
            len_out = math.hypot(dx_out, dy_out)
            if len_in < 1e-9 or len_out < 1e-9:
                continue
            # Dot product for angle between vectors
            dot = (dx_in * dx_out + dy_in * dy_out) / (len_in * len_out)
            dot = max(-1.0, min(1.0, dot))
            deflection = math.acos(dot)  # 0 = straight, pi = U-turn
            if deflection > max_defl:
                max_defl = deflection
                max_defl_idx = i

        if max_defl <= threshold_rad or max_defl_idx < 1:
            continue

        # Split at max_defl_idx
        split_counter -= 1
        synthetic_node_id = split_counter  # -1, -2, -3, ...

        row_data = edges_reset.loc[idx].to_dict()
        original_u = row_data["u"]
        original_v = row_data["v"]

        # Row A: original_u → synthetic_node
        row_a = dict(row_data)
        row_a["v"] = synthetic_node_id
        row_a["geometry"] = _LS(coords[: max_defl_idx + 1])

        # Row B: synthetic_node → original_v
        row_b = dict(row_data)
        row_b["u"] = synthetic_node_id
        row_b["geometry"] = _LS(coords[max_defl_idx:])

        rows_to_drop.append(idx)
        new_rows.append(row_a)
        new_rows.append(row_b)

    if not new_rows:
        print(f"Deflection splitting: 0 segments split (threshold={deflection_threshold_deg}°)")
        return edges_reset

    result = edges_reset.drop(index=rows_to_drop)
    new_gdf = gpd.GeoDataFrame(new_rows, crs=edges_reset.crs)
    result = gpd.GeoDataFrame(
        pd.concat([result, new_gdf], ignore_index=True),
        crs=edges_reset.crs,
    )

    n_splits = len(new_rows) // 2
    print(f"Deflection splitting: {n_splits} segments split into "
          f"{len(new_rows)} pieces (threshold={deflection_threshold_deg}°)")

    # Recurse: a split half may itself have a deflection > threshold
    if n_splits > 0:
        result = split_deflected_segments(result, deflection_threshold_deg)

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

    # --- 2. Raw bearings ---
    raw_bearings: dict[int, float] = {}
    for idx in edges_reset.index:
        geom = edges_reset.at[idx, "geometry"]
        if geom is not None:
            b = _linestring_bearing(cast(BaseGeometry, geom))
            if b is not None:
                raw_bearings[idx] = b

    # --- 3. Bearing normalization ---
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

    # Unnamed / unassigned: northward fallback (normalize to [0, 180))
    for idx in unnamed_indices:
        b = raw_bearings[idx]
        edge_bearings[idx] = (b + 180.0) % 360 if b >= 180.0 else b

    # --- 4. Grid ID assignment ---
    cell_seq: dict[tuple[int, int], int] = {}
    edge_grid_ids: dict[int, str] = {}
    for idx in edges_reset.index:
        geom = edges_reset.at[idx, "geometry"]
        if geom is None:
            continue
        mid = cast(BaseGeometry, geom).interpolate(0.5, normalized=True)
        col = int((mid.x - ox_g) / cell)
        row = int((mid.y - oy_g) / cell)
        cell_key = (col, row)
        seq = cell_seq.get(cell_key, 0)
        cell_seq[cell_key] = seq + 1
        edge_grid_ids[idx] = f"{col}_{row}_{seq}"

    # --- 5. Intersection detection ---
    def _is_non_road(data: dict[str, Any]) -> bool:
        hw = data.get("highway", "")
        hw_vals = hw if isinstance(hw, list) else [hw]
        return bool(frozenset(str(h) for h in hw_vals) & _NON_ROAD_HIGHWAY)

    node_degree: dict[int, int] = {}
    for u, v, data in G.edges(data=True):
        if _is_non_road(data):
            continue
        node_degree[u] = node_degree.get(u, 0) + 1
        node_degree[v] = node_degree.get(v, 0) + 1
    intersection_node_ids = {nid for nid, deg in node_degree.items() if deg >= 3}

    print(f"Grid assignments: {len(edge_grid_ids)} segments across "
          f"{len(cell_seq)} grid cells ({n_cols}×{n_rows} @ {cell:.0f}m) "
          f"| {len(intersection_node_ids)} intersection nodes")
    n_oneway = sum(1 for idx in edges_reset.index
                   if idx in edge_bearings and oneway_col.at[idx] in (True, "yes", "1", 1, "-1", "reverse"))
    n_named = sum(len(v) for v in name_groups.values())
    print(f"  Bearings: {n_oneway} one-way, {n_named} named two-way (majority vote), "
          f"{len(unnamed_indices)} unnamed (northward fallback)")

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
    bearings: list[float | None] = []
    grid_ids: list[str | None] = []
    start_inter: list[bool] = []
    end_inter: list[bool] = []

    intersection_ids = grid_result.intersection_node_ids

    for idx in edges_reset.index:
        bearings.append(grid_result.edge_bearings.get(idx))
        grid_ids.append(grid_result.edge_grid_ids.get(idx))

        u = cast(int | float, edges_reset.at[idx, "u"])
        v = cast(int | float, edges_reset.at[idx, "v"])
        # Synthetic split nodes (negative ints) are never intersections
        u_int = int(u) if not (isinstance(u, (int, float)) and u < 0) else u
        v_int = int(v) if not (isinstance(v, (int, float)) and v < 0) else v
        start_inter.append(u_int in intersection_ids)
        end_inter.append(v_int in intersection_ids)

    populated["street_grid_id"] = pd.Series(grid_ids, index=edges_reset.index)
    populated["normalized_bearing"] = pd.Series(bearings, index=edges_reset.index)
    populated["start_node_is_intersection_node"] = pd.Series(start_inter, index=edges_reset.index)
    populated["end_node_is_intersection_node"] = pd.Series(end_inter, index=edges_reset.index)


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


#---Data Pipeline---
def populate_schema(place: str, *, default_lane_width_m: float = 3.5) -> gpd.GeoDataFrame:
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
        "cycleway:left:2:lane", "cycleway:right:2:lane",
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

    # OSM tags that map directly
    for tag in ("name", "highway", "maxspeed", "oneway", "lanes", "lane_width", "surface"):
        populated[tag] = _get(tag)

    # Main street geometry (the edge LineString)
    populated["street_geometry"] = edges_reset["geometry"]

    # Start/end node point geometries looked up from the nodes GDF
    node_geom = nodes["geometry"]
    populated["start_node_geometry"] = edges_reset["u"].map(node_geom)
    populated["end_node_geometry"]   = edges_reset["v"].map(node_geom)

    # Grid columns: street_grid_id, normalized_bearing, intersection node flags
    _populate_grid_columns(populated, edges_reset, grid_result)

    populated = populate_base_bikelanes(populated, edges_reset)
    populated = populate_base_footlanes(populated, edges_reset)
    # Swap left/right facility columns on edges with reversed geometry (deferred
    # until after tag population so all sidewalk/bikeway columns exist)
    populated = swap_facilities_by_bearing(populated, edges_reset)
    populated = _populate_separate_facilities(populated, edges_reset, default_lane_width_m)
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
        export_df = populated.copy()
        for col in present_geom_cols:
            pbar.set_postfix_str(col)
            export_df[col] = export_df[col].apply(
                lambda g: g.wkb_hex if hasattr(g, 'wkb_hex') else None
            )
            pbar.update(1)

        pbar.set_postfix_str("writing parquet")
        # Normalize _buffered columns: pipeline writes True/False booleans while OSM
        # tags provide strings like 'yes'/'no'. Mixed types cause PyArrow serialization
        # failures, so coerce everything to consistent strings before export.
        def _norm_buffered(v):
            if v is True:
                return "yes"
            if v is False:
                return "no"
            try:
                if pd.isna(v):
                    return None
            except (TypeError, ValueError):
                pass
            return str(v) if v is not None else None
        for col in [c for c in export_df.columns if c.endswith("_buffered")]:
            export_df[col] = export_df[col].apply(_norm_buffered)

        # Stringify any object columns that contain list values (pyarrow can't mix list/scalar)
        def _safe_str(x):
            if x is None:
                return None
            if isinstance(x, (list, np.ndarray)):
                return str(x)
            try:
                if pd.isna(x):
                    return None
            except (TypeError, ValueError):
                pass
            return str(x)
        for col in export_df.columns:
            if col in present_geom_cols or col == export_df.geometry.name:
                continue
            if export_df[col].dtype == object:
                has_list = any(isinstance(x, (list, np.ndarray)) for x in export_df[col])
                if has_list:
                    export_df[col] = export_df[col].apply(_safe_str)
        place_slug = place.replace(", ", "_").replace(" ", "_")
        output_path = OUTPUT_DIR / f"{place_slug}_network.parquet"
        export_df.to_parquet(output_path)
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
        "maxspeed", "oneway", "lanes", "lane_width", "surface",
        # Street Centerline Features
        "street_feature_types", "public_data_id_street_feature",
        "street_feature_geometry", "street_feature_geometry_projected",
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
        populated["bikeway_left_1_width"]    = _coalesce("cycleway:left:width",  "cycleway:width")
        populated["bikeway_right_1_width"]   = _coalesce("cycleway:right:width", "cycleway:width")
        populated["bikeway_left_1_seperator"]   = _coalesce("cycleway:left:buffer",  "cycleway:buffer")
        populated["bikeway_right_1_seperator"] = _coalesce("cycleway:right:buffer", "cycleway:buffer")
        pbar.update(1)

        # Permitted / incline
        pbar.set_postfix_str("permitted / incline")
        populated["bikeway_left_1_permitted"]  = _get("bicycle")
        populated["bikeway_right_1_permitted"] = _get("bicycle")
        populated["bikeway_left_1_incline"]    = _get("incline")
        populated["bikeway_right_1_incline"]   = _get("incline")
        pbar.update(1)

        # Secondary bikeway slots (_2)
        pbar.set_postfix_str("secondary bikeway slots")
        populated["bikeway_left_2_quality"]   = _get("cycleway:left:2:lane")
        populated["bikeway_right_2_quality"]  = _get("cycleway:right:2:lane")
        populated["bikeway_left_2_surface"]   = _get("cycleway:left:2:surface")
        populated["bikeway_right_2_surface"]  = _get("cycleway:right:2:surface")
        populated["bikeway_left_2_width"]     = _get("cycleway:left:2:width")
        populated["bikeway_right_2_width"]    = _get("cycleway:right:2:width")
        populated["bikeway_left_2_permitted"] = _get("bicycle")
        populated["bikeway_right_2_permitted"]= _get("bicycle")
        populated["bikeway_left_2_incline"]    = _get("incline")
        populated["bikeway_right_2_incline"]  = _get("incline")
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
        populated["sidewalk_left_width"]      = _get("sidewalk:left:width")
        populated["sidewalk_right_width"]     = _get("sidewalk:right:width")
        populated["sidewalk_left_incline"]    = _get("sidewalk:left:incline")
        populated["sidewalk_right_incline"]   = _get("sidewalk:right:incline")
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
            populated.at[road_idx, f"{prefix}_width"]     = cy_row.get("width",    pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_permitted"] = cy_row.get("bicycle",  pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_incline"]   = cy_row.get("incline",  pd.NA)  # type: ignore[index]
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
            # Deduplication: keep the footway edge with more vertices
            existing = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
            if existing is not None and isinstance(existing, BaseGeometry):
                new_verts = _vertex_count(fw_geom)
                existing_verts = _vertex_count(existing)
                if new_verts > existing_verts:
                    # New edge wins — replace and re-validate
                    populated.at[road_idx, f"{prefix}_geometry"] = fw_geom  # type: ignore[index]
                    winner = fw_geom
                    n_foot_replaced[side] += 1
                    # Re-run centerline check on new winner
                    _is_centerline(winner, cast(BaseGeometry, road_geom), road_idx, prefix, populated)
                    # Check if winner is suspect after centerline check
                    winner_after = populated.at[road_idx, f"{prefix}_geometry"]  # type: ignore[index]
                    if winner_after is not None and isinstance(winner_after, BaseGeometry):
                        if _is_suspect_geometry(winner_after):
                            populated.at[road_idx, f"{prefix}_geometry"] = None  # type: ignore[index]
                            populated.at[road_idx, f"{prefix}_buffered"] = True  # type: ignore[index]
                            n_foot_suspect[side] += 1
                else:
                    # Existing wins — discard new edge
                    n_foot_replaced[side] += 1
            else:
                populated.at[road_idx, f"{prefix}_geometry"] = fw_geom  # type: ignore[index]
            n_foot_merged += 1
        else:
            populated.at[road_idx, f"{prefix}_presence"] = fw_row.get("highway",    pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_surface"]  = fw_row.get("surface",    pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_width"]    = fw_row.get("width",      pd.NA)  # type: ignore[index]
            populated.at[road_idx, f"{prefix}_incline"]  = fw_row.get("incline",    pd.NA)  # type: ignore[index]
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
    _SEPARATE_PRESENCE_VALUES = {"separate", "footway", "pedestrian"}

    # Build unified tree of ALL separate sidewalk geometries (both sides).
    all_sep_sw_geoms: list[BaseGeometry] = []
    all_sep_sw_bearings: list[float] = []
    all_sep_sw_row_indices: list = []  # row index to prevent self-suppression
    for sw_side in ("left", "right"):
        gcol = f"sidewalk_{sw_side}_geometry"
        bcol = f"sidewalk_{sw_side}_buffered"
        if gcol not in populated.columns:
            continue
        for idx in populated.index:
            g = populated.at[idx, gcol]
            if g is None or not hasattr(g, "geom_type"):
                continue
            bval = populated.at[idx, bcol] if bcol in populated.columns else None  # type: ignore[index]
            if bval is True or str(bval).lower() == "yes":
                continue
            bearing = _linestring_bearing(cast(BaseGeometry, g))
            if bearing is None:
                continue
            all_sep_sw_geoms.append(cast(BaseGeometry, g))
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
            geom_before = populated.at[idx, geom_col]
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
    _FACILITY_GEOM_COLS = [
        "street_geometry",
        "bikeway_left_1_geometry",  "bikeway_left_2_geometry",
        "bikeway_right_1_geometry", "bikeway_right_2_geometry",
        "sidewalk_left_geometry",   "sidewalk_right_geometry",
    ]
    for col in _FACILITY_GEOM_COLS:
        if col == own_geom_col or col not in populated.columns:
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

# def create_boundary_grid():
#     return

# def block_assignment():
#     return

# def intersection_analysis():
#     return

if __name__ == "__main__":
    run_multi_city()

