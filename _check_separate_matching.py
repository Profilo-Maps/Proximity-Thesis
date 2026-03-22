"""Diagnostic: why do ~17k road segments have sidewalk_*_presence='separate' but null geometry?

Root cause hypothesis:
  - sidewalk_*_presence is set from OSM centerline tags (sidewalk:left=separate) at population time
  - The footway matching loop tries to match actual footway OSM ways to road segments
  - If no footway is matched, presence stays 'separate' but geometry remains null
  - The buffering pass explicitly SKIPS presence='separate' — so these rows are never backfilled

This script determines whether the footway OSM ways actually exist near these roads.
"""

import pickle
import math
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

# ─── Config ────────────────────────────────────────────────────────────────
PARQUET_PATH = Path("Output/San_Francisco_County_California_USA_network.parquet")
CACHE_PATH = Path("Implementations/.osm_cache/San Francisco County, California, USA.pkl")
SEARCH_RADIUS_M = 30.0  # same as _NAME_MATCH_RADIUS_M in ProximityModel.py
PARALLEL_TOLERANCE_DEG = 30.0  # same as _PARALLEL_BEARING_TOLERANCE_DEG

# ─── Helpers (copied from ProximityModel.py for consistency) ───────────────
def _linestring_bearing(geom: BaseGeometry) -> float | None:
    coords = []
    if isinstance(geom, MultiLineString):
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
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def _bearings_parallel(a: float, b: float, tolerance: float = PARALLEL_TOLERANCE_DEG) -> bool:
    diff = abs(a - b) % 360
    if diff > 180:
        diff = 360 - diff
    if diff > 90:
        diff = 180 - diff
    return diff <= tolerance


def _normalize_name(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().lower()
    return s if s else None


def _coerce_name_raw(val) -> str | None:
    if isinstance(val, (list, np.ndarray)):
        val = next((x for x in val if isinstance(x, str)), None)
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return val if isinstance(val, str) else None


# ─── Step 1: Load parquet ──────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Load parquet and identify problem rows")
print("=" * 70)

df = gpd.read_parquet(PARQUET_PATH)
print(f"Total rows in parquet: {len(df)}")

# Identify rows with separate presence but null geometry
problems: dict[str, pd.Index] = {}
for side in ("left", "right"):
    pres_col = f"sidewalk_{side}_presence"
    geom_col = f"sidewalk_{side}_geometry"

    is_separate = df[pres_col].astype(str).str.lower() == "separate"
    has_geom = pd.Series(
        [isinstance(g, BaseGeometry) for g in df[geom_col].values],
        index=df.index, dtype=bool
    )
    problem_mask = is_separate & ~has_geom
    problems[side] = df.index[problem_mask]

    # Also count rows WITH geometry for context
    has_geom_separate = is_separate & has_geom
    print(f"  sidewalk_{side}: {is_separate.sum()} separate total, "
          f"{has_geom_separate.sum()} have geometry, "
          f"{problem_mask.sum()} missing geometry")

# Combine all problem rows (may overlap)
all_problem_idx = problems["left"].union(problems["right"])
print(f"\nTotal unique rows with at least one side separate+null-geom: {len(all_problem_idx)}")

# ─── Step 2: Load raw OSM edges from cache ─────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Load raw OSM graph from cache & extract footways")
print("=" * 70)

import osmnx as ox

with open(CACHE_PATH, "rb") as f:
    G = pickle.load(f)

nodes, edges = ox.graph_to_gdfs(G)
edges = edges.to_crs("EPSG:32610")
nodes = nodes.to_crs("EPSG:32610")
edges_reset = edges.reset_index()

print(f"Total OSM edges: {len(edges_reset)}")

# Identify footway edges (same classification as ProximityModel.py)
hw = edges_reset.get("highway", pd.Series(dtype=object))
bicycle = edges_reset.get("bicycle", pd.Series(dtype=object))
foot = edges_reset.get("foot", pd.Series(dtype=object))

FOOTWAY_HW = {"footway", "pedestrian", "path", "steps", "corridor"}
BIKEWAY_HW = {"cycleway", "path", "bridleway"}

is_bikeway = (
    hw.isin(BIKEWAY_HW) |
    (hw.isin({"path", "footway"}) & bicycle.isin({"designated", "yes"}))
)
is_footway = (
    hw.isin(FOOTWAY_HW) |
    (hw.isin({"path"}) & foot.isin({"designated", "yes"}))
) & ~bicycle.isin({"designated"}).astype(bool)
is_footway = is_footway & ~is_bikeway

footways = edges_reset[is_footway].copy()
print(f"Footway edges (same filter as pipeline): {len(footways)}")

# Also look at the subset tagged footway=sidewalk specifically
if "footway" in edges_reset.columns:
    fw_tag_counts = edges_reset.loc[is_footway, "footway"].value_counts(dropna=False)
    print(f"  footway tag distribution among footway edges:")
    for val, cnt in fw_tag_counts.head(10).items():
        print(f"    {val}: {cnt}")

# ─── Step 3: Build spatial index of footways ───────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Spatial analysis — do footways exist near problem roads?")
print("=" * 70)

# Build STRtree of footway geometries
fw_geoms = list(footways.geometry)
fw_tree = STRtree(fw_geoms)

# Pre-compute footway bearings and names
fw_bearings = [_linestring_bearing(g) for g in fw_geoms]
fw_names_raw = list(footways["name"]) if "name" in footways.columns else [None] * len(footways)
fw_names = [_coerce_name_raw(n) for n in fw_names_raw]
fw_norm_names = [_normalize_name(n) for n in fw_names]

# ─── Step 4: For each problem road, check for nearby footways ─────────────
print(f"\nChecking {len(all_problem_idx)} problem roads for nearby footways...")

# Categories
cat_no_footway_nearby = []         # No footway within 30m at all
cat_footway_exists_unnamed = []    # Footway nearby, unnamed, but was it matched?
cat_name_mismatch = []             # Footway nearby, but name doesn't match road
cat_bearing_mismatch = []          # Footway nearby, name matches but bearing is off
cat_footway_matched_elsewhere = [] # Footway nearby with matching name (matching should have worked)
cat_road_unnamed = []              # Road itself has no name
cat_footway_nearby_unnamed_road = []  # Road unnamed AND footway nearby

# Get road data
road_geom_col = "street_geometry"
road_name_col = "name"

detail_samples: dict[str, list] = defaultdict(list)  # category -> sample details
MAX_SAMPLES = 10

for i, idx in enumerate(all_problem_idx):
    if i % 2000 == 0 and i > 0:
        print(f"  ... processed {i}/{len(all_problem_idx)}")

    road_geom = df.at[idx, road_geom_col]
    if not isinstance(road_geom, BaseGeometry):
        continue

    road_name = _normalize_name(df.at[idx, road_name_col] if road_name_col in df.columns else None)
    road_bearing = _linestring_bearing(road_geom)

    # Which sides are problematic for this row?
    prob_sides = []
    for side in ("left", "right"):
        if idx in problems[side]:
            prob_sides.append(side)

    # Query footways within search radius
    search_area = road_geom.buffer(SEARCH_RADIUS_M)
    hit_positions = fw_tree.query(search_area)

    if len(hit_positions) == 0:
        cat_no_footway_nearby.append(idx)
        continue

    # Analyze nearby footways
    found_name_match = False
    found_parallel = False
    found_any = False

    nearby_details = []

    for pos in hit_positions:
        fw_geom = fw_geoms[pos]
        actual_dist = road_geom.distance(fw_geom)
        if actual_dist > SEARCH_RADIUS_M:
            continue  # STRtree uses bbox, so filter by actual distance

        found_any = True
        fw_name_norm = fw_norm_names[pos]
        fw_bear = fw_bearings[pos]

        name_matches = (fw_name_norm is not None and fw_name_norm == road_name)
        bearing_ok = True
        if road_bearing is not None and fw_bear is not None:
            bearing_ok = _bearings_parallel(road_bearing, fw_bear)

        if name_matches:
            found_name_match = True
        if bearing_ok:
            found_parallel = True

        nearby_details.append({
            "dist": actual_dist,
            "fw_name": fw_names[pos],
            "name_match": name_matches,
            "bearing_ok": bearing_ok,
            "fw_bearing": fw_bear,
        })

    if not found_any:
        cat_no_footway_nearby.append(idx)
        continue

    # Categorize the failure
    if road_name is None:
        cat_road_unnamed.append(idx)
        cat_footway_nearby_unnamed_road.append(idx)
    elif found_name_match:
        # Name matched AND footway nearby — matching SHOULD have worked
        # Possible reasons: slot taken, bearing mismatch, centerline coincidence, dedup
        cat_footway_matched_elsewhere.append(idx)
        if len(detail_samples["match_should_work"]) < MAX_SAMPLES:
            detail_samples["match_should_work"].append({
                "idx": idx,
                "road_name": road_name,
                "sides": prob_sides,
                "nearby": sorted(nearby_details, key=lambda x: x["dist"])[:3],
            })
    else:
        # Footway nearby but name doesn't match
        # Check: is any nearby footway unnamed?
        has_unnamed_fw = any(fw_norm_names[pos] is None for pos in hit_positions
                           if road_geom.distance(fw_geoms[pos]) <= SEARCH_RADIUS_M)
        has_named_fw = any(fw_norm_names[pos] is not None for pos in hit_positions
                          if road_geom.distance(fw_geoms[pos]) <= SEARCH_RADIUS_M)

        if has_unnamed_fw:
            cat_footway_exists_unnamed.append(idx)
            if len(detail_samples["unnamed_fw"]) < MAX_SAMPLES:
                detail_samples["unnamed_fw"].append({
                    "idx": idx,
                    "road_name": road_name,
                    "sides": prob_sides,
                    "nearby": sorted(nearby_details, key=lambda x: x["dist"])[:3],
                })
        elif has_named_fw:
            cat_name_mismatch.append(idx)
            if len(detail_samples["name_mismatch"]) < MAX_SAMPLES:
                detail_samples["name_mismatch"].append({
                    "idx": idx,
                    "road_name": road_name,
                    "sides": prob_sides,
                    "nearby": sorted(nearby_details, key=lambda x: x["dist"])[:3],
                })

# ─── Step 5: Report ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

total = len(all_problem_idx)
print(f"\nTotal roads with sidewalk_*_presence='separate' + null geometry: {total}")
print()

print("CATEGORY BREAKDOWN:")
print(f"  1. NO footway within 30m (OSM data gap):          {len(cat_no_footway_nearby):>6}  ({100*len(cat_no_footway_nearby)/total:.1f}%)")
print(f"  2. Road is unnamed (can't name-match):            {len(cat_road_unnamed):>6}  ({100*len(cat_road_unnamed)/total:.1f}%)")
print(f"  3. Footway nearby, unnamed (parallel fallback):   {len(cat_footway_exists_unnamed):>6}  ({100*len(cat_footway_exists_unnamed)/total:.1f}%)")
print(f"  4. Footway nearby, name MISMATCH:                 {len(cat_name_mismatch):>6}  ({100*len(cat_name_mismatch)/total:.1f}%)")
print(f"  5. Footway nearby, name MATCHES (should work!):   {len(cat_footway_matched_elsewhere):>6}  ({100*len(cat_footway_matched_elsewhere)/total:.1f}%)")
sum_accounted = (len(cat_no_footway_nearby) + len(cat_road_unnamed) +
                 len(cat_footway_exists_unnamed) + len(cat_name_mismatch) +
                 len(cat_footway_matched_elsewhere))
print(f"  Sum accounted:                                    {sum_accounted:>6}")

# ─── Top streets per category ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("TOP STREETS — Category 1: No footway within 30m (OSM data gap)")
print("-" * 70)
if cat_no_footway_nearby:
    names = [_normalize_name(df.at[idx, "name"]) or "(unnamed)" for idx in cat_no_footway_nearby]
    for name, cnt in Counter(names).most_common(15):
        print(f"  {cnt:>4}  {name}")

print("\n" + "-" * 70)
print("TOP STREETS — Category 2: Road unnamed, footway nearby")
print("-" * 70)
if cat_footway_nearby_unnamed_road:
    # Show the highway types of these unnamed roads
    hw_types = [str(df.at[idx, "highway"]) if "highway" in df.columns else "?" for idx in cat_footway_nearby_unnamed_road]
    for hw, cnt in Counter(hw_types).most_common(10):
        print(f"  {cnt:>4}  highway={hw}")

print("\n" + "-" * 70)
print("TOP STREETS — Category 3: Unnamed footway nearby (parallel fallback path)")
print("-" * 70)
if cat_footway_exists_unnamed:
    names = [_normalize_name(df.at[idx, "name"]) or "(unnamed)" for idx in cat_footway_exists_unnamed]
    for name, cnt in Counter(names).most_common(15):
        print(f"  {cnt:>4}  {name}")

print("\n" + "-" * 70)
print("TOP STREETS — Category 4: Footway nearby but name mismatch")
print("-" * 70)
if cat_name_mismatch:
    names = [_normalize_name(df.at[idx, "name"]) or "(unnamed)" for idx in cat_name_mismatch]
    for name, cnt in Counter(names).most_common(15):
        print(f"  {cnt:>4}  {name}")

print("\n" + "-" * 70)
print("TOP STREETS — Category 5: Name matches but matching still failed")
print("-" * 70)
if cat_footway_matched_elsewhere:
    names = [_normalize_name(df.at[idx, "name"]) or "(unnamed)" for idx in cat_footway_matched_elsewhere]
    for name, cnt in Counter(names).most_common(15):
        print(f"  {cnt:>4}  {name}")

# ─── Sample details ───────────────────────────────────────────────────────
for cat_key, cat_label in [
    ("match_should_work", "Cat 5 — Name matches but failed"),
    ("unnamed_fw", "Cat 3 — Unnamed footway nearby"),
    ("name_mismatch", "Cat 4 — Name mismatch"),
]:
    samples = detail_samples.get(cat_key, [])
    if samples:
        print(f"\n{'-' * 70}")
        print(f"SAMPLE DETAILS: {cat_label}")
        print(f"{'-' * 70}")
        for s in samples[:5]:
            print(f"  Row {s['idx']}  road='{s['road_name']}'  sides={s['sides']}")
            for n in s["nearby"]:
                print(f"    fw @ {n['dist']:.1f}m  name='{n['fw_name']}'  "
                      f"name_match={n['name_match']}  bearing_ok={n['bearing_ok']}  "
                      f"fw_bearing={n['fw_bearing']}")

# ─── Deeper analysis: WHY does category 5 fail? ───────────────────────────
print("\n" + "=" * 70)
print("DEEPER ANALYSIS: Why does matching fail when name matches exist?")
print("=" * 70)

# Check the ORIGIN of the 'separate' presence tag — is it from OSM centerline
# tags (sidewalk:left=separate) or from the footway matching loop?
# The footway matching loop ALSO sets presence='separate' AND geometry.
# So if presence='separate' but geometry is null, it almost certainly came from
# the centerline tag, NOT from the matching loop.

# Let's verify by checking what the original sidewalk:left tag says
print("\nChecking OSM centerline sidewalk tags on problem roads...")
print("(These come from the original OSM way tags, not from footway matching)")

# Reload edges to check the original sidewalk tags
sw_tag_cols = ["sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both"]
available_sw_tags = [c for c in sw_tag_cols if c in edges_reset.columns]
print(f"Available sidewalk tag columns in OSM data: {available_sw_tags}")

if available_sw_tags:
    # How many OSM edges have sidewalk=separate or sidewalk:left=separate etc?
    for col in available_sw_tags:
        val_counts = edges_reset[col].value_counts(dropna=True)
        print(f"\n  {col} value counts:")
        for val, cnt in val_counts.head(10).items():
            print(f"    {val}: {cnt}")

# ─── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The ~17k roads with sidewalk_*_presence='separate' and null geometry are roads
where the OSM WAY ITSELF is tagged sidewalk:left=separate (or :right/:both).

This tag means "there IS a sidewalk, mapped as a separate OSM way." But the
pipeline's footway matching loop did not find and attach a footway geometry to
these road segments.

The buffering pass (which generates offset geometry for 'yes'/'both'/etc.)
explicitly SKIPS rows with presence='separate' — correctly, because
'separate' means the real geometry should come from a separate OSM way.

The question is: does that separate OSM way exist? The breakdown above shows
whether footways actually exist near these roads.
""")
