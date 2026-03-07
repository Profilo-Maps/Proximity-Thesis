# Intersection Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Normalize facility geometries at intersection nodes to create contiguous sidewalk/bikeway networks suitable for Dijkstra pathfinding, generating curb ramps and crosswalks as transition nodes.

**Architecture:** Node-scoped processing — for each intersection node (degree >= 3), build a boundary, classify approaching facilities, resolve contiguity per corner via bisector/bearing meeting points, then generate curb ramps and crosswalks. All facility corrections at a node are resolved in a single pass to prevent ordering dependencies. The entry point is a single function `intersection_analysis()` called from `populate_schema()` after `_assign_facility_grid_ids()` and before export.

**Tech Stack:** Python 3.14, Shapely 2.x (offset_curve, STRtree, unary_union), GeoPandas, NumPy. All geometry in EPSG:32610 (metres). No new dependencies.

---

## Decisions Log

- **Split separate facilities through-intersection:** Merge both halves into street segment rows, delete original separate row if applicable.
- **Review flags:** New `intersetion_review_flag` column in parquet (already in schema spec, typo preserved).
- **Crosswalk caching:** Query OSM early in `populate_schema()`, pass cached GeoDataFrame to `intersection_analysis()`.
- **Acute corners (<60 deg):** Each facility extends along its own bearing until meeting the other or the boundary. No shared bisector meeting point.

## Data Structures

```python
@dataclass
class IntersectionConfig:
    boundary_expansion_m: float = 2.0        # step 12: expand convex hull
    boundary_fallback_radius_m: float = 8.0  # step 12: circle for T-intersections
    search_buffer_m: float = 5.0             # step 13: extra buffer for facility search
    contiguity_tolerance_m: float = 2.0      # steps 14-15: endpoint gap
    acute_angle_threshold_deg: float = 60.0  # step 16: bisector vs bearing cutoff
    curb_ramp_offset_m: float = 0.75         # step 17: min ramp offset
    curb_return_min_m: float = 0.3           # step 17: collapse threshold
    ramp_merge_distance_m: float = 1.0       # step 18: merge nearby ramps
    max_conflict_iterations: int = 3         # step 18: iteration bound

@dataclass
class ApproachingFacility:
    segment_idx: int              # row index in gdf
    facility_col: str             # e.g. "sidewalk_left_geometry"
    facility_kind: str            # "sidewalk" | "bikeway"
    side: str                     # "left" | "right"
    slot: str | None              # "1" | "2" | None (sidewalks have no slot)
    node_end: str                 # "start" | "end" - which end touches this node
    geometry: BaseGeometry        # current facility linestring (EPSG:32610)
    is_buffered: bool             # from *_buffered column
    classification: str           # "overshooting" | "undershooting" | "continuous" | "none"
    trim_point: Point | None      # boundary intersection point
    offset_width_m: float         # perpendicular distance from street centerline
    flagged: bool = False         # needs contiguity correction

@dataclass
class Corner:
    node_id: int
    street_a_idx: int             # gdf row of first bounding street
    street_a_end: str             # "start" | "end"
    street_b_idx: int             # gdf row of second bounding street
    street_b_end: str             # "start" | "end"
    bearing_a: float              # outgoing bearing of street A from node
    bearing_b: float              # outgoing bearing of street B (next clockwise)
    corner_angle: float           # interior angle (degrees)
    bisector_bearing: float       # outward bisector direction
    facilities: list[ApproachingFacility]
    meeting_point: Point | None = None

@dataclass
class IntersectionNode:
    node_id: int
    node_point: Point             # EPSG:32610
    segment_entries: list[tuple[int, str]]  # [(seg_idx, "start"|"end"), ...]
    boundary: BaseGeometry        # convex hull or circle from step 12
    corners: list[Corner]         # ordered clockwise
    exclusion_zone: BaseGeometry  # union of street + corrected facility geometries
```

## Corner-to-Facility Mapping

For corner (A, B) where A and B are ordered clockwise by outgoing bearing:
- **Right side of street A** and **left side of street B** face this corner.
- Street A's facility is at its `start` or `end` depending on `street_a_end`.
- Street B's facility is at its `start` or `end` depending on `street_b_end`.

---

### Task 1: Crosswalk OSM Query + Cache

**Files:**
- Modify: `Implementations/ProximityModel.py:983-1000` (near `_query_traffic_calming_nodes`)
- Modify: `Implementations/ProximityModel.py:1112` (`populate_schema`)

**Step 1: Write `_query_crosswalk_nodes` function**

Add after `_query_traffic_calming_nodes` (line ~1001):

```python
def _query_crosswalk_nodes(place: str) -> gpd.GeoDataFrame:
    """Query OSM for highway=crossing and footway=crossing nodes inside *place*.

    Returns a point GeoDataFrame in EPSG:4326 with all crossing:* tags retained.
    """
    tags: dict[str, bool | str | list[str]] = {"highway": "crossing"}
    gdf = ox.features_from_place(place, tags=tags)
    # Also query footway=crossing
    try:
        gdf2 = ox.features_from_place(place, tags={"footway": "crossing"})
        gdf = pd.concat([gdf, gdf2]).drop_duplicates(subset=["osmid"])
    except Exception:
        pass  # footway query may return empty
    # Keep only points
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    if gdf.empty:
        return gdf
    gdf = gdf.reset_index()
    return gdf.copy()
```

**Step 2: Wire into `populate_schema`**

In `populate_schema`, after the OSM graph load block (~line 1178) and before the schema population, add:

```python
# --- Cache crosswalk data (pipeline step 2 addendum) ---
crosswalk_cache = _query_crosswalk_nodes(place)
if not crosswalk_cache.empty:
    crosswalk_cache = crosswalk_cache.set_crs("EPSG:4326").to_crs("EPSG:32610")
print(f"Crosswalk cache: {len(crosswalk_cache)} crossing nodes found.")
```

**Step 3: Run pipeline on a small test to verify crosswalk query works**

Run: `cd c:\Dev\Proximity && C:\Users\karna\miniconda3\envs\ParkximityENV\python.exe -c "from Implementations.ProximityModel import _query_crosswalk_nodes; gdf = _query_crosswalk_nodes('San Francisco County, California, USA'); print(f'{len(gdf)} crosswalks, columns: {list(gdf.columns)[:10]}')" `

Expected: prints a count of crosswalk nodes and first 10 column names.

**Step 4: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: add crosswalk OSM query and cache in populate_schema (step 2)"
```

---

### Task 2: Data Structures + `intersetion_review_flag` Column

**Files:**
- Modify: `Implementations/ProximityModel.py:68-86` (dataclass area)
- Modify: `Implementations/ProximityModel.py:1338-1345` (`_create_schema_dataframe`)

**Step 1: Add dataclasses**

After `GridResult` (line ~86), add `IntersectionConfig`, `ApproachingFacility`, `Corner`, `IntersectionNode` dataclasses exactly as defined in the Data Structures section above.

**Step 2: Add `intersetion_review_flag` to schema**

In `_create_schema_dataframe` (line ~1340), add `"intersetion_review_flag"` to the columns list, right after the `# Street Centerlines` comment, before `"street_id"`:

```python
columns = [
    # Street Centerlines
    "intersetion_review_flag",
    "street_id", "street_grid_id", ...
```

**Step 3: Verify schema creation includes the new column**

Run: `cd c:\Dev\Proximity && C:\Users\karna\miniconda3\envs\ParkximityENV\python.exe -c "from Implementations.ProximityModel import _create_schema_dataframe; df = _create_schema_dataframe(); assert 'intersetion_review_flag' in df.columns; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: add intersection analysis dataclasses and review flag column"
```

---

### Task 3: Intersection Boundary (Step 12)

**Files:**
- Modify: `Implementations/ProximityModel.py` (after dataclasses, before `populate_schema`)

**Step 1: Write `_ix_get_node_point` helper**

```python
def _ix_get_node_point(
    node_id: int,
    seg_entries: list[tuple[int, str]],
    gdf: gpd.GeoDataFrame,
) -> Point:
    """Get the Point geometry for an intersection node from segment endpoints."""
    for seg_idx, end in seg_entries:
        geom = gdf.at[seg_idx, "street_geometry"]
        if geom is None or geom.is_empty:
            continue
        coords = _flatten_coords(geom)
        if not coords:
            continue
        if end == "start":
            return Point(coords[0])
        else:
            return Point(coords[-1])
    raise ValueError(f"No valid geometry found for node {node_id}")
```

**Step 2: Write `_ix_compute_offset_width` helper**

```python
def _ix_compute_offset_width(
    seg_idx: int,
    gdf: gpd.GeoDataFrame,
    default_lane_width_m: float = 3.5,
) -> float:
    """Compute max perpendicular offset width for a segment (half-road + bike + sidewalk widths)."""
    raw_lanes = gdf.at[seg_idx, "lanes"]
    raw_lw = gdf.at[seg_idx, "lane_width"]
    lanes = _parse_numeric(raw_lanes, 2.0)
    lane_width = _parse_numeric(raw_lw, default_lane_width_m)
    half_road = (lanes * lane_width) / 2.0

    max_offset = half_road
    for side in ("left", "right"):
        bike_w = 0.0
        for slot in ("1", "2"):
            w = gdf.at[seg_idx, f"bikeway_{side}_{slot}_width"]
            if not _is_na(w):
                bike_w += _parse_numeric(w, 0.0)
        sw_w = gdf.at[seg_idx, f"sidewalk_{side}_width"]
        sw_w = _parse_numeric(sw_w, 1.5) if not _is_na(sw_w) else 1.5
        total = half_road + bike_w + sw_w
        if total > max_offset:
            max_offset = total
    return max_offset
```

**Step 3: Write `_ix_build_boundary`**

```python
def _ix_build_boundary(
    node_id: int,
    node_point: Point,
    seg_entries: list[tuple[int, str]],
    gdf: gpd.GeoDataFrame,
    config: IntersectionConfig,
) -> BaseGeometry:
    """Build intersection boundary: convex hull of street endpoints expanded by max offset width.

    Falls back to a circle if fewer than 3 distinct endpoints (T-intersection/collinear).
    """
    endpoints: list[Point] = []
    max_offset = 0.0

    for seg_idx, end in seg_entries:
        geom = gdf.at[seg_idx, "street_geometry"]
        if geom is None or geom.is_empty:
            continue
        coords = _flatten_coords(geom)
        if not coords:
            continue
        pt = Point(coords[0] if end == "start" else coords[-1])
        endpoints.append(pt)
        offset = _ix_compute_offset_width(seg_idx, gdf)
        if offset > max_offset:
            max_offset = offset

    expansion = max(max_offset, config.boundary_expansion_m)

    # Deduplicate endpoints (within 1m)
    unique_pts: list[Point] = []
    for pt in endpoints:
        is_dup = False
        for upt in unique_pts:
            if pt.distance(upt) < 1.0:
                is_dup = True
                break
        if not is_dup:
            unique_pts.append(pt)

    if len(unique_pts) >= 3:
        hull = MultiPoint(unique_pts).convex_hull
        if hull.geom_type == "Polygon":
            return hull.buffer(expansion)
        # Degenerate hull (collinear) — fall through to circle

    # Fallback: circle
    radius = max(expansion, config.boundary_fallback_radius_m)
    return node_point.buffer(radius)
```

**Step 4: Write a quick smoke test**

Run: `cd c:\Dev\Proximity && C:\Users\karna\miniconda3\envs\ParkximityENV\python.exe -c "
from shapely.geometry import Point, MultiPoint
from Implementations.ProximityModel import IntersectionConfig
# Simulate a 4-way intersection
pts = [Point(0, 50), Point(50, 0), Point(0, -50), Point(-50, 0)]
hull = MultiPoint(pts).convex_hull
boundary = hull.buffer(10.0)
print(f'Boundary area: {boundary.area:.1f} m2, type: {boundary.geom_type}')
assert boundary.geom_type == 'Polygon'
assert boundary.area > 0
print('OK')
"`

Expected: prints boundary area and `OK`.

**Step 5: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: intersection boundary computation (step 12)"
```

---

### Task 4: Corner Construction

**Files:**
- Modify: `Implementations/ProximityModel.py` (after boundary functions)

**Step 1: Write `_ix_outgoing_bearing` helper**

```python
def _ix_outgoing_bearing(
    seg_idx: int,
    end: str,
    gdf: gpd.GeoDataFrame,
) -> float:
    """Compute the bearing of a street segment going AWAY from the intersection node.

    If the segment's start node is at the intersection, the outgoing bearing
    is from coords[0] -> coords[1] (the segment goes away from start).
    If the segment's end node is at the intersection, the outgoing bearing
    is from coords[-1] -> coords[-2] (the segment goes away from end).
    """
    geom = gdf.at[seg_idx, "street_geometry"]
    coords = _flatten_coords(geom)
    if len(coords) < 2:
        return 0.0

    if end == "start":
        x0, y0 = coords[0]
        x1, y1 = coords[1]
    else:
        x0, y0 = coords[-1]
        x1, y1 = coords[-2]

    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return 0.0
    return float(np.degrees(np.arctan2(dx, dy)) % 360)
```

**Step 2: Write `_ix_build_corners`**

```python
def _ix_build_corners(
    node_id: int,
    seg_entries: list[tuple[int, str]],
    gdf: gpd.GeoDataFrame,
) -> list[Corner]:
    """Build corners by ordering streets clockwise by outgoing bearing.

    Each corner is the wedge between two adjacent streets. The corner's
    interior angle is the clockwise sweep from street A to street B.
    """
    # Compute outgoing bearings for each segment at this node
    street_bearings: list[tuple[int, str, float]] = []
    for seg_idx, end in seg_entries:
        # Skip non-road facilities (footways, cycleways stored as separate edges)
        hw = gdf.at[seg_idx, "highway"]
        if isinstance(hw, str) and hw in _NON_ROAD_HIGHWAY:
            continue
        bearing = _ix_outgoing_bearing(seg_idx, end, gdf)
        street_bearings.append((seg_idx, end, bearing))

    if len(street_bearings) < 2:
        return []  # Dead-end or single street — no corners

    # Sort clockwise by bearing
    street_bearings.sort(key=lambda x: x[2])

    corners: list[Corner] = []
    n = len(street_bearings)
    for i in range(n):
        seg_a_idx, seg_a_end, bearing_a = street_bearings[i]
        seg_b_idx, seg_b_end, bearing_b = street_bearings[(i + 1) % n]

        # Interior angle: clockwise sweep from A to B
        angle = (bearing_b - bearing_a) % 360
        if angle == 0:
            angle = 360.0  # Parallel streets, full sweep

        # Bisector: midpoint of the angular sweep, pointing outward
        bisector = (bearing_a + angle / 2) % 360

        corners.append(Corner(
            node_id=node_id,
            street_a_idx=seg_a_idx,
            street_a_end=seg_a_end,
            street_b_idx=seg_b_idx,
            street_b_end=seg_b_end,
            bearing_a=bearing_a,
            bearing_b=bearing_b,
            corner_angle=angle,
            bisector_bearing=bisector,
            facilities=[],
            meeting_point=None,
        ))

    return corners
```

**Step 3: Write a unit test**

Run: `cd c:\Dev\Proximity && C:\Users\karna\miniconda3\envs\ParkximityENV\python.exe -c "
# Test: 4-way intersection at 0, 90, 180, 270
# Should produce 4 corners each with 90-degree angle
bearings = [0.0, 90.0, 180.0, 270.0]
n = len(bearings)
for i in range(n):
    a = bearings[i]
    b = bearings[(i+1) % n]
    angle = (b - a) % 360
    if angle == 0: angle = 360.0
    bisector = (a + angle / 2) % 360
    print(f'Corner {i}: A={a}, B={b}, angle={angle}, bisector={bisector}')
    assert angle == 90.0, f'Expected 90, got {angle}'
print('OK')
"`

Expected: 4 corners, each 90 degrees, bisectors at 45, 135, 225, 315. Prints `OK`.

**Step 4: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: corner construction with clockwise ordering (step 12)"
```

---

### Task 5: Facility Classification (Step 13)

**Files:**
- Modify: `Implementations/ProximityModel.py`

**Step 1: Write `_ix_facility_col_prefix` helper**

```python
_IX_FACILITY_COLS = [
    ("sidewalk", "left",  None, "sidewalk_left_geometry",      "sidewalk_left_buffered"),
    ("sidewalk", "right", None, "sidewalk_right_geometry",     "sidewalk_right_buffered"),
    ("bikeway",  "left",  "1",  "bikeway_left_1_geometry",     "bikeway_left_1_buffered"),
    ("bikeway",  "left",  "2",  "bikeway_left_2_geometry",     "bikeway_left_2_buffered"),
    ("bikeway",  "right", "1",  "bikeway_right_1_geometry",    "bikeway_right_1_buffered"),
    ("bikeway",  "right", "2",  "bikeway_right_2_geometry",    "bikeway_right_2_buffered"),
]
```

**Step 2: Write `_ix_classify_facilities`**

```python
def _ix_classify_facilities(
    node_id: int,
    seg_entries: list[tuple[int, str]],
    boundary: BaseGeometry,
    gdf: gpd.GeoDataFrame,
    config: IntersectionConfig,
) -> list[ApproachingFacility]:
    """Classify each facility approaching this intersection node (step 13).

    Categories: overshooting, undershooting, continuous, none.
    For overshooting: trim geometry at boundary.
    For continuous: split at boundary, merge halves into street rows.
    """
    search_zone = boundary.buffer(config.search_buffer_m)
    boundary_ring = boundary.boundary  # LinearRing for intersection tests
    results: list[ApproachingFacility] = []

    for seg_idx, end in seg_entries:
        for kind, side, slot, geom_col, buffered_col in _IX_FACILITY_COLS:
            geom = gdf.at[seg_idx, geom_col]
            if geom is None or (hasattr(geom, "is_empty") and geom.is_empty):
                continue
            if not isinstance(geom, (LineString, MultiLineString)):
                continue

            # Determine the endpoint nearest to the intersection node
            coords = _flatten_coords(geom)
            if not coords:
                continue

            if end == "start":
                near_pt = Point(coords[0])
            else:
                near_pt = Point(coords[-1])

            # Check if facility is within search zone
            if not search_zone.intersects(geom):
                continue  # No intersection — classification "none"

            is_buffered = gdf.at[seg_idx, buffered_col]
            is_buffered = is_buffered in (True, "yes")

            offset_w = _ix_compute_offset_width(seg_idx, gdf)

            # Classify
            crosses_boundary = geom.crosses(boundary_ring) or boundary.contains(geom)
            endpoint_inside = boundary.contains(near_pt)
            endpoint_outside = not endpoint_inside

            # Count intersection points with boundary
            intersection_with_boundary = geom.intersection(boundary_ring)

            classification = "none"
            trim_point = None

            if isinstance(geom, LineString):
                if crosses_boundary:
                    # Check if it's continuous (enters and exits)
                    if not intersection_with_boundary.is_empty:
                        ix_points = []
                        if intersection_with_boundary.geom_type == "Point":
                            ix_points = [intersection_with_boundary]
                        elif intersection_with_boundary.geom_type == "MultiPoint":
                            ix_points = list(intersection_with_boundary.geoms)
                        elif intersection_with_boundary.geom_type in ("LineString", "MultiLineString"):
                            # Tangent case
                            ix_points = []

                        if len(ix_points) >= 2:
                            # Continuous through-intersection
                            classification = "continuous"
                        elif len(ix_points) == 1:
                            # Overshooting from one side
                            classification = "overshooting"
                            trim_point = ix_points[0] if isinstance(ix_points[0], Point) else None
                        else:
                            classification = "overshooting"
                    else:
                        classification = "overshooting"
                elif endpoint_outside and near_pt.distance(boundary) < config.search_buffer_m:
                    classification = "undershooting"
                elif endpoint_inside:
                    classification = "overshooting"

            results.append(ApproachingFacility(
                segment_idx=seg_idx,
                facility_col=geom_col,
                facility_kind=kind,
                side=side,
                slot=slot,
                node_end=end,
                geometry=geom,
                is_buffered=is_buffered,
                classification=classification,
                trim_point=trim_point,
                offset_width_m=offset_w,
                flagged=False,
            ))

    return results
```

**Step 3: Write `_ix_trim_overshooting`**

```python
def _ix_trim_overshooting(
    facility: ApproachingFacility,
    boundary: BaseGeometry,
    gdf: gpd.GeoDataFrame,
) -> None:
    """Trim an overshooting facility at the intersection boundary (step 13). Modifies gdf."""
    geom = facility.geometry
    boundary_ring = boundary.boundary
    intersection = geom.intersection(boundary_ring)

    if intersection.is_empty:
        return

    # Get the trim point (closest intersection point to the segment's far end)
    if intersection.geom_type == "Point":
        trim_pt = intersection
    elif intersection.geom_type == "MultiPoint":
        # Pick the point closest to the near end (intersection node side)
        coords = _flatten_coords(geom)
        if facility.node_end == "start":
            ref_pt = Point(coords[0])
        else:
            ref_pt = Point(coords[-1])
        trim_pt = min(intersection.geoms, key=lambda p: p.distance(ref_pt))
    else:
        return

    # Split the linestring at the trim point and keep the portion outside the boundary
    from shapely.ops import split, snap
    snapped = snap(geom, trim_pt, tolerance=1.0)
    try:
        parts = split(snapped, trim_pt.buffer(0.1))
    except Exception:
        return

    # Keep the part that's mostly outside the boundary
    best_part = None
    best_outside_len = 0.0
    for part in parts.geoms:
        if not isinstance(part, LineString):
            continue
        outside = part.difference(boundary)
        ol = outside.length if not outside.is_empty else 0.0
        if ol > best_outside_len:
            best_outside_len = ol
            best_part = part

    if best_part is not None and not best_part.is_empty:
        # Ensure the trimmed geometry endpoint at the boundary IS the trim point
        facility.geometry = best_part
        facility.trim_point = trim_pt
        gdf.at[facility.segment_idx, facility.facility_col] = best_part
```

**Step 4: Write `_ix_split_continuous`**

```python
def _ix_split_continuous(
    facility: ApproachingFacility,
    boundary: BaseGeometry,
    seg_entries: list[tuple[int, str]],
    gdf: gpd.GeoDataFrame,
) -> list[ApproachingFacility]:
    """Split a continuous through-intersection facility into two halves (step 13).

    Discards the interior portion within the boundary.
    Merges each half into the appropriate street segment row.
    Returns up to 2 new ApproachingFacility objects (one per side).
    """
    geom = facility.geometry
    boundary_ring = boundary.boundary
    intersection = geom.intersection(boundary_ring)

    if intersection.is_empty:
        return [facility]

    ix_points = []
    if intersection.geom_type == "Point":
        ix_points = [intersection]
    elif intersection.geom_type == "MultiPoint":
        ix_points = list(intersection.geoms)

    if len(ix_points) < 2:
        # Tangent or single-point — treat as undershooting
        facility.classification = "undershooting"
        return [facility]

    # Check for degenerate interior segment (< 1m)
    interior = geom.intersection(boundary)
    if interior.length < 1.0:
        facility.classification = "undershooting"
        return [facility]

    # Split at boundary: keep the two outer portions
    from shapely.ops import split, snap
    # Snap the geometry to both intersection points
    snapped = geom
    for pt in ix_points:
        snapped = snap(snapped, pt, tolerance=1.0)

    exterior = geom.difference(boundary)
    if exterior.is_empty:
        return []

    parts: list[LineString] = []
    if exterior.geom_type == "LineString":
        parts = [exterior]
    elif exterior.geom_type == "MultiLineString":
        parts = [p for p in exterior.geoms if isinstance(p, LineString) and p.length > 1.0]

    if len(parts) == 0:
        return []

    # For each part, find the closest street segment and assign
    new_facilities: list[ApproachingFacility] = []
    for part in parts:
        # Find nearest segment
        mid = part.interpolate(0.5, normalized=True)
        best_seg = None
        best_dist = float("inf")
        for seg_idx, end in seg_entries:
            street_geom = gdf.at[seg_idx, "street_geometry"]
            if street_geom is None or street_geom.is_empty:
                continue
            d = street_geom.distance(mid)
            if d < best_dist:
                best_dist = d
                best_seg = (seg_idx, end)

        if best_seg is None:
            continue

        seg_idx, end = best_seg
        # Determine which side of the street this part is on
        side = _road_side(gdf.at[seg_idx, "street_geometry"], mid)
        geom_col = f"{facility.facility_kind}_{side}"
        if facility.facility_kind == "bikeway":
            geom_col += f"_{facility.slot or '1'}"
        geom_col += "_geometry"

        # Write into gdf
        if geom_col in gdf.columns:
            gdf.at[seg_idx, geom_col] = part

        # Find the endpoint on the boundary
        boundary_endpoint = None
        for coord in [part.coords[0], part.coords[-1]]:
            pt = Point(coord)
            if pt.distance(boundary_ring) < 2.0:
                boundary_endpoint = pt
                break

        new_facilities.append(ApproachingFacility(
            segment_idx=seg_idx,
            facility_col=geom_col,
            facility_kind=facility.facility_kind,
            side=side,
            slot=facility.slot,
            node_end=end,
            geometry=part,
            is_buffered=facility.is_buffered,
            classification="trimmed",
            trim_point=boundary_endpoint,
            offset_width_m=facility.offset_width_m,
            flagged=False,
        ))

    return new_facilities
```

**Step 5: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: facility classification - overshooting/undershooting/continuous (step 13)"
```

---

### Task 6: Contiguity Checks (Steps 14-15)

**Files:**
- Modify: `Implementations/ProximityModel.py`

**Step 1: Write `_ix_assign_facilities_to_corners`**

```python
def _ix_assign_facilities_to_corners(
    corners: list[Corner],
    facilities: list[ApproachingFacility],
) -> None:
    """Assign each classified facility to its corner based on segment/side mapping.

    Corner (A, B) clockwise: right side of A, left side of B face this corner.
    """
    for corner in corners:
        corner.facilities = []
        for f in facilities:
            if f.classification == "none":
                continue
            # Check if this facility belongs to this corner
            if f.segment_idx == corner.street_a_idx and f.side == "right":
                corner.facilities.append(f)
            elif f.segment_idx == corner.street_b_idx and f.side == "left":
                corner.facilities.append(f)
```

**Step 2: Write `_ix_check_contiguity`**

```python
def _ix_check_contiguity(
    corner: Corner,
    config: IntersectionConfig,
) -> None:
    """Flag non-contiguous facility pairs at a corner (steps 14-15).

    For each facility kind, if both sides have geometry, check endpoint distance.
    If distance > tolerance, flag both for correction.
    """
    # Group facilities by kind
    by_kind: dict[str, list[ApproachingFacility]] = {}
    for f in corner.facilities:
        key = f.facility_kind
        if f.slot:
            key += f"_{f.slot}"
        by_kind.setdefault(key, []).append(f)

    for kind, facs in by_kind.items():
        if len(facs) < 2:
            # Single facility on one side — flag as undershooting if it exists
            if len(facs) == 1 and facs[0].classification == "undershooting":
                facs[0].flagged = True
            continue

        # Find the two endpoints nearest the intersection
        for i in range(len(facs)):
            for j in range(i + 1, len(facs)):
                fa, fb = facs[i], facs[j]
                if fa.segment_idx == fb.segment_idx:
                    continue  # Same segment, not a corner pair

                # Get endpoints at intersection side
                pt_a = _ix_facility_endpoint_at_node(fa)
                pt_b = _ix_facility_endpoint_at_node(fb)
                if pt_a is None or pt_b is None:
                    continue

                dist = pt_a.distance(pt_b)
                if dist > config.contiguity_tolerance_m:
                    fa.flagged = True
                    fb.flagged = True


def _ix_facility_endpoint_at_node(f: ApproachingFacility) -> Point | None:
    """Get the facility endpoint that faces the intersection node."""
    if f.trim_point is not None:
        return f.trim_point
    coords = _flatten_coords(f.geometry)
    if not coords:
        return None
    if f.node_end == "start":
        return Point(coords[0])
    else:
        return Point(coords[-1])
```

**Step 3: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: contiguity checks for bikelanes and sidewalks (steps 14-15)"
```

---

### Task 7: Constraint Satisfaction - Meeting Points (Step 16)

**Files:**
- Modify: `Implementations/ProximityModel.py`

**Step 1: Write `_ix_find_meeting_point`**

```python
def _ix_find_meeting_point(
    node_point: Point,
    bisector_bearing: float,
    offset_distance: float,
    exclusion_zone: BaseGeometry,
    boundary: BaseGeometry,
) -> Point | None:
    """Walk along bisector from node to find a valid meeting point outside exclusion zone.

    Returns None if no valid point exists within the boundary.
    """
    bearing_rad = np.radians(bisector_bearing)
    dx = np.sin(bearing_rad)
    dy = np.cos(bearing_rad)

    # Start at offset_distance, walk outward in 0.5m steps
    for step_m in np.arange(offset_distance, offset_distance * 3.0, 0.5):
        candidate = Point(node_point.x + dx * step_m, node_point.y + dy * step_m)
        if not boundary.contains(candidate):
            break  # Went past the boundary
        if not exclusion_zone.contains(candidate):
            return candidate

    return None  # No valid point found
```

**Step 2: Write `_ix_extend_facility_to_point`**

```python
def _ix_extend_facility_to_point(
    facility: ApproachingFacility,
    target_point: Point,
    gdf: gpd.GeoDataFrame,
) -> None:
    """Extend a facility linestring to reach the target point. Modifies gdf in-place."""
    coords = _flatten_coords(facility.geometry)
    if not coords:
        return

    if facility.node_end == "start":
        # Prepend target point to start of coords
        new_coords = [(target_point.x, target_point.y)] + coords
    else:
        # Append target point to end of coords
        new_coords = coords + [(target_point.x, target_point.y)]

    new_geom = LineString(new_coords)
    facility.geometry = new_geom
    facility.trim_point = target_point
    gdf.at[facility.segment_idx, facility.facility_col] = new_geom
```

**Step 3: Write `_ix_resolve_corner`**

```python
def _ix_resolve_corner(
    corner: Corner,
    node_point: Point,
    exclusion_zone: BaseGeometry,
    boundary: BaseGeometry,
    gdf: gpd.GeoDataFrame,
    config: IntersectionConfig,
) -> BaseGeometry:
    """Resolve all flagged facilities at a corner (step 16). Returns updated exclusion zone.

    Process order: bikelanes first (add to exclusion zone), then sidewalks.
    For acute corners (<60 deg): each facility extends along its own bearing.
    For normal corners: bisector-based meeting point.
    """
    is_acute = corner.corner_angle < config.acute_angle_threshold_deg

    # Separate facilities by kind
    bikelanes = [f for f in corner.facilities if f.facility_kind == "bikeway" and f.flagged]
    sidewalks = [f for f in corner.facilities if f.facility_kind == "sidewalk" and f.flagged]

    # --- Bikelanes ---
    if len(bikelanes) >= 2:
        exclusion_zone = _ix_resolve_facility_pair(
            bikelanes, corner, node_point, exclusion_zone, boundary, gdf, config, is_acute,
        )
    elif len(bikelanes) == 1:
        _ix_extend_single_facility(bikelanes[0], boundary, exclusion_zone, gdf)

    # --- Sidewalks ---
    if len(sidewalks) >= 2:
        exclusion_zone = _ix_resolve_facility_pair(
            sidewalks, corner, node_point, exclusion_zone, boundary, gdf, config, is_acute,
        )
    elif len(sidewalks) == 1:
        _ix_extend_single_facility(sidewalks[0], boundary, exclusion_zone, gdf)

    return exclusion_zone


def _ix_resolve_facility_pair(
    facilities: list[ApproachingFacility],
    corner: Corner,
    node_point: Point,
    exclusion_zone: BaseGeometry,
    boundary: BaseGeometry,
    gdf: gpd.GeoDataFrame,
    config: IntersectionConfig,
    is_acute: bool,
) -> BaseGeometry:
    """Resolve a pair of flagged facilities at a corner. Returns updated exclusion zone."""
    fa, fb = facilities[0], facilities[1]

    if is_acute:
        # Each extends along its own bearing until meeting the other or boundary
        pt_a = _ix_facility_endpoint_at_node(fa)
        pt_b = _ix_facility_endpoint_at_node(fb)
        if pt_a is not None and pt_b is not None:
            # Extend A's bearing line and B's bearing line, find intersection
            bearing_a = _ix_outgoing_bearing(fa.segment_idx, fa.node_end, gdf)
            bearing_b = _ix_outgoing_bearing(fb.segment_idx, fb.node_end, gdf)

            # Reverse bearings (pointing toward intersection)
            in_a = (bearing_a + 180) % 360
            in_b = (bearing_b + 180) % 360

            # Create extension rays and find intersection
            meet = _ix_ray_intersection(pt_a, in_a, pt_b, in_b)
            if meet is not None and boundary.contains(meet) and not exclusion_zone.contains(meet):
                _ix_extend_facility_to_point(fa, meet, gdf)
                _ix_extend_facility_to_point(fb, meet, gdf)
                corner.meeting_point = meet
            else:
                # Flag for review
                gdf.at[fa.segment_idx, "intersetion_review_flag"] = True
                gdf.at[fb.segment_idx, "intersetion_review_flag"] = True
    else:
        # Bisector approach
        both_buffered = fa.is_buffered and fb.is_buffered
        one_buffered = fa.is_buffered != fb.is_buffered

        if both_buffered:
            avg_offset = (fa.offset_width_m + fb.offset_width_m) / 2.0
            meet = _ix_find_meeting_point(
                node_point, corner.bisector_bearing, avg_offset,
                exclusion_zone, boundary,
            )
            if meet is not None:
                _ix_extend_facility_to_point(fa, meet, gdf)
                _ix_extend_facility_to_point(fb, meet, gdf)
                corner.meeting_point = meet
            else:
                gdf.at[fa.segment_idx, "intersetion_review_flag"] = True
                gdf.at[fb.segment_idx, "intersetion_review_flag"] = True

        elif one_buffered:
            # Extend the buffered one to meet the separate one
            buffered_f = fa if fa.is_buffered else fb
            separate_f = fb if fa.is_buffered else fa
            sep_pt = _ix_facility_endpoint_at_node(separate_f)
            if sep_pt is not None and not exclusion_zone.contains(sep_pt):
                _ix_extend_facility_to_point(buffered_f, sep_pt, gdf)
                corner.meeting_point = sep_pt

        else:
            # Both separate — bisector constrained by exclusion zone
            avg_offset = (fa.offset_width_m + fb.offset_width_m) / 2.0
            meet = _ix_find_meeting_point(
                node_point, corner.bisector_bearing, avg_offset,
                exclusion_zone, boundary,
            )
            if meet is not None:
                _ix_extend_facility_to_point(fa, meet, gdf)
                _ix_extend_facility_to_point(fb, meet, gdf)
                corner.meeting_point = meet

    # Update exclusion zone
    for f in facilities:
        if f.geometry is not None and not f.geometry.is_empty:
            exclusion_zone = exclusion_zone.union(f.geometry.buffer(0.5))

    return exclusion_zone


def _ix_ray_intersection(
    pt_a: Point, bearing_a: float,
    pt_b: Point, bearing_b: float,
) -> Point | None:
    """Find intersection of two rays. Returns None if parallel or degenerate."""
    rad_a = np.radians(bearing_a)
    rad_b = np.radians(bearing_b)
    dx_a, dy_a = np.sin(rad_a), np.cos(rad_a)
    dx_b, dy_b = np.sin(rad_b), np.cos(rad_b)

    # Solve: pt_a + t*dir_a = pt_b + s*dir_b
    det = dx_a * (-dy_b) - dy_a * (-dx_b)
    if abs(det) < 1e-10:
        return None  # Parallel rays

    dpx = pt_b.x - pt_a.x
    dpy = pt_b.y - pt_a.y
    t = (dpx * (-dy_b) - dpy * (-dx_b)) / det

    if t < 0:
        return None  # Intersection is behind ray A

    x = pt_a.x + t * dx_a
    y = pt_a.y + t * dy_a
    return Point(x, y)


def _ix_extend_single_facility(
    facility: ApproachingFacility,
    boundary: BaseGeometry,
    exclusion_zone: BaseGeometry,
    gdf: gpd.GeoDataFrame,
) -> None:
    """Extend a single undershooting facility to the intersection boundary along its bearing."""
    coords = _flatten_coords(facility.geometry)
    if len(coords) < 2:
        return

    # Compute bearing at the node-facing end
    if facility.node_end == "start":
        x0, y0 = coords[1]
        x1, y1 = coords[0]
    else:
        x0, y0 = coords[-2]
        x1, y1 = coords[-1]

    dx, dy = x1 - x0, y1 - y0
    length = (dx**2 + dy**2) ** 0.5
    if length < 1e-10:
        return
    dx /= length
    dy /= length

    # Walk along bearing until hitting boundary edge
    start = Point(x1, y1)
    for dist in np.arange(0.5, 50.0, 0.5):
        candidate = Point(x1 + dx * dist, y1 + dy * dist)
        if not boundary.contains(candidate):
            # Overshoot — back up to boundary
            target = Point(x1 + dx * (dist - 0.5), y1 + dy * (dist - 0.5))
            if not exclusion_zone.contains(target):
                _ix_extend_facility_to_point(facility, target, gdf)
            return
```

**Step 4: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: constraint satisfaction for facility meeting points (step 16)"
```

---

### Task 8: Curb Ramp Generation (Step 17)

**Files:**
- Modify: `Implementations/ProximityModel.py`

**Step 1: Write `_ix_generate_curb_ramps`**

```python
def _ix_generate_curb_ramps(
    corner: Corner,
    node_point: Point,
    gdf: gpd.GeoDataFrame,
    config: IntersectionConfig,
    ramp_counter: list[int],
) -> None:
    """Generate two directional curb ramps per corner (step 17).

    Ramp A: endpoint of sidewalk on street A's right side (facing this corner).
    Ramp B: endpoint of sidewalk on street B's left side (facing this corner).
    Projects ramps outward from meeting point along crosswalk approach directions.
    """
    if corner.meeting_point is None:
        # No sidewalks resolved at this corner — check if any sidewalks exist at all
        sidewalks = [f for f in corner.facilities if f.facility_kind == "sidewalk"]
        if not sidewalks:
            return
        # Use the single sidewalk endpoint as a simple ramp point
        if len(sidewalks) == 1:
            sw = sidewalks[0]
            pt = _ix_facility_endpoint_at_node(sw)
            if pt is None:
                return
            ramp_counter[0] += 1
            _ix_write_curb_ramp(
                gdf, sw.segment_idx, sw.side, sw.node_end, 1,
                ramp_counter[0], pt, None, None,
            )
            return

    meeting = corner.meeting_point

    # Crosswalk approach directions:
    # Ramp A faces toward street B (crosswalk crosses street B at this corner)
    # Ramp B faces toward street A (crosswalk crosses street A at this corner)
    approach_a_bearing = corner.bearing_b  # toward street B
    approach_b_bearing = corner.bearing_a  # toward street A

    # Get sidewalk width for offset
    sw_widths = [f.offset_width_m for f in corner.facilities if f.facility_kind == "sidewalk"]
    sw_width = np.mean(sw_widths) if sw_widths else 1.5
    ramp_offset = max(sw_width / 2.0, config.curb_ramp_offset_m)

    # Project ramp positions
    rad_a = np.radians(approach_a_bearing)
    ramp_a = Point(
        meeting.x + np.sin(rad_a) * ramp_offset,
        meeting.y + np.cos(rad_a) * ramp_offset,
    )
    rad_b = np.radians(approach_b_bearing)
    ramp_b = Point(
        meeting.x + np.sin(rad_b) * ramp_offset,
        meeting.y + np.cos(rad_b) * ramp_offset,
    )

    # Check curb return threshold
    curb_return_length = ramp_a.distance(ramp_b)
    if curb_return_length < config.curb_return_min_m:
        # Collapse to single apex ramp at meeting point
        ramp_counter[0] += 1
        _ix_write_curb_ramp(
            gdf, corner.street_a_idx, "right", corner.street_a_end, 1,
            ramp_counter[0], meeting, None, None,
        )
        ramp_counter[0] += 1
        _ix_write_curb_ramp(
            gdf, corner.street_b_idx, "left", corner.street_b_end, 1,
            ramp_counter[0], meeting, None, None,
        )
        return

    # Compute curb return geometry (arc or line between ramp points)
    curb_return = _ix_compute_curb_return(ramp_a, ramp_b, node_point, corner.corner_angle)

    # Compute return location (compass direction of the curb return)
    mid_return = LineString([ramp_a, ramp_b]).interpolate(0.5, normalized=True)
    returnloc = _ix_compass_direction(node_point, mid_return)

    # Write ramp A (right side of street A)
    ramp_counter[0] += 1
    _ix_write_curb_ramp(
        gdf, corner.street_a_idx, "right", corner.street_a_end, 1,
        ramp_counter[0], ramp_a, returnloc, "Right",
    )

    # Write ramp B (left side of street B)
    ramp_counter[0] += 1
    _ix_write_curb_ramp(
        gdf, corner.street_b_idx, "left", corner.street_b_end, 1,
        ramp_counter[0], ramp_b, returnloc, "Left",
    )

    # Write curb return geometry to appropriate segment
    if "curb_return_geometry" in gdf.columns:
        gdf.at[corner.street_a_idx, "curb_return_geometry"] = curb_return


def _ix_write_curb_ramp(
    gdf: gpd.GeoDataFrame,
    seg_idx: int,
    side: str,  # "left" | "right"
    end: str,   # "start" | "end"
    slot: int,  # 1, 2, or 3
    ramp_id: int,
    point: Point,
    returnloc: str | None,
    returnposition: str | None,
) -> None:
    """Write curb ramp data to the appropriate schema columns."""
    prefix = f"sidewalk_{side}_curbramp_{end}_{slot}"

    col_id = f"{prefix}_ID"
    col_returnloc = f"{prefix}_returnloc"
    col_returnposition = f"{prefix}_returnposition"
    col_geometry = f"{prefix}_geometry"

    if col_id in gdf.columns:
        gdf.at[seg_idx, col_id] = str(ramp_id)
    if col_returnloc in gdf.columns and returnloc is not None:
        gdf.at[seg_idx, col_returnloc] = returnloc
    if col_returnposition in gdf.columns and returnposition is not None:
        gdf.at[seg_idx, col_returnposition] = returnposition
    if col_geometry in gdf.columns:
        gdf.at[seg_idx, col_geometry] = point


def _ix_compute_curb_return(
    ramp_a: Point, ramp_b: Point, center: Point, corner_angle: float,
) -> BaseGeometry:
    """Compute curb return geometry as arc or line between two ramp points."""
    if abs(corner_angle - 180.0) < 15.0:
        # Near-straight corner — straight line
        return LineString([ramp_a, ramp_b])

    # Arc approximation: interpolate points along an arc centered on the node
    radius = (center.distance(ramp_a) + center.distance(ramp_b)) / 2.0
    angle_a = np.arctan2(ramp_a.x - center.x, ramp_a.y - center.y)
    angle_b = np.arctan2(ramp_b.x - center.x, ramp_b.y - center.y)

    # Ensure we go the short way around
    diff = (angle_b - angle_a) % (2 * np.pi)
    if diff > np.pi:
        diff -= 2 * np.pi

    n_points = max(3, int(abs(diff) / np.radians(10)))
    arc_points = []
    for i in range(n_points + 1):
        t = i / n_points
        angle = angle_a + t * diff
        x = center.x + radius * np.sin(angle)
        y = center.y + radius * np.cos(angle)
        arc_points.append((x, y))

    if len(arc_points) < 2:
        return LineString([ramp_a, ramp_b])
    return LineString(arc_points)


def _ix_compass_direction(from_pt: Point, to_pt: Point) -> str:
    """Return compass direction (N, NE, E, SE, S, SW, W, NW) from from_pt to to_pt."""
    dx = to_pt.x - from_pt.x
    dy = to_pt.y - from_pt.y
    angle = np.degrees(np.arctan2(dx, dy)) % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((angle + 22.5) / 45) % 8
    return directions[idx]
```

**Step 2: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: curb ramp generation with curb return geometry (step 17)"
```

---

### Task 9: Multi-Face Conflict Resolution (Step 18)

**Files:**
- Modify: `Implementations/ProximityModel.py`

**Step 1: Write `_ix_resolve_cross_corner_conflicts`**

```python
def _ix_resolve_cross_corner_conflicts(
    node: IntersectionNode,
    gdf: gpd.GeoDataFrame,
    config: IntersectionConfig,
) -> None:
    """Check and resolve conflicts between adjacent corners at a node (step 18).

    Guarantees pullback preserves contiguity. Max iterations = config.max_conflict_iterations.
    """
    corners = node.corners
    n = len(corners)
    if n < 2:
        return

    for iteration in range(config.max_conflict_iterations):
        had_conflict = False

        for i in range(n):
            ca = corners[i]
            cb = corners[(i + 1) % n]

            # Check facility geometry overlaps between adjacent corners
            for fa in ca.facilities:
                for fb in cb.facilities:
                    if fa.facility_kind != fb.facility_kind:
                        continue
                    if fa.geometry is None or fb.geometry is None:
                        continue
                    if fa.geometry.is_empty or fb.geometry.is_empty:
                        continue

                    if fa.geometry.crosses(fb.geometry) or fa.geometry.overlaps(fb.geometry):
                        had_conflict = True
                        # Pull both back to midpoint of overlap
                        intersection = fa.geometry.intersection(fb.geometry)
                        if not intersection.is_empty:
                            mid = intersection.centroid
                            # Trim both facilities to not cross midpoint
                            # This preserves their connection to their own corner's meeting point
                            _ix_trim_to_midpoint(fa, mid, gdf)
                            _ix_trim_to_midpoint(fb, mid, gdf)

            # Check curb ramp separation
            ramps_a = _ix_get_ramp_points(ca, gdf)
            ramps_b = _ix_get_ramp_points(cb, gdf)
            for ra in ramps_a:
                for rb in ramps_b:
                    if ra.distance(rb) < config.ramp_merge_distance_m:
                        had_conflict = True
                        # Merge into single apex ramp at midpoint
                        apex = LineString([ra, rb]).interpolate(0.5, normalized=True)
                        # Update the ramp geometries to the apex point
                        _ix_merge_ramps_to_apex(ca, cb, apex, gdf)

        if not had_conflict:
            break
    else:
        # Max iterations reached — flag for review
        for seg_idx, _ in node.segment_entries:
            gdf.at[seg_idx, "intersetion_review_flag"] = True


def _ix_trim_to_midpoint(
    facility: ApproachingFacility,
    midpoint: Point,
    gdf: gpd.GeoDataFrame,
) -> None:
    """Trim a facility geometry at the midpoint, keeping the portion connected to its corner."""
    from shapely.ops import split, snap
    geom = facility.geometry
    snapped = snap(geom, midpoint, tolerance=1.0)
    try:
        parts = split(snapped, midpoint.buffer(0.1))
    except Exception:
        return

    # Keep the part whose endpoint matches the facility's non-node end
    coords = _flatten_coords(geom)
    if not coords:
        return

    if facility.node_end == "start":
        far_pt = Point(coords[-1])
    else:
        far_pt = Point(coords[0])

    best_part = None
    best_dist = float("inf")
    for part in parts.geoms:
        if not isinstance(part, LineString):
            continue
        d = part.distance(far_pt)
        if d < best_dist:
            best_dist = d
            best_part = part

    if best_part is not None and not best_part.is_empty:
        facility.geometry = best_part
        gdf.at[facility.segment_idx, facility.facility_col] = best_part


def _ix_get_ramp_points(corner: Corner, gdf: gpd.GeoDataFrame) -> list[Point]:
    """Get all curb ramp points at a corner from gdf columns."""
    points: list[Point] = []
    for seg_idx, side, end in [
        (corner.street_a_idx, "right", corner.street_a_end),
        (corner.street_b_idx, "left", corner.street_b_end),
    ]:
        for slot in (1, 2, 3):
            col = f"sidewalk_{side}_curbramp_{end}_{slot}_geometry"
            if col in gdf.columns:
                pt = gdf.at[seg_idx, col]
                if pt is not None and isinstance(pt, Point):
                    points.append(pt)
    return points


def _ix_merge_ramps_to_apex(
    ca: Corner, cb: Corner, apex: Point, gdf: gpd.GeoDataFrame,
) -> None:
    """Merge two close ramps from adjacent corners into a single apex ramp."""
    # Set the ramp point to the apex for the shared-edge ramps
    # Corner A's ramp on street B side, Corner B's ramp on street A side
    # These are the ramps closest to each other
    for seg_idx, side, end in [
        (ca.street_b_idx, "left", ca.street_b_end),   # CA's ramp toward CB
        (cb.street_a_idx, "right", cb.street_a_end),   # CB's ramp toward CA
    ]:
        col = f"sidewalk_{side}_curbramp_{end}_1_geometry"
        if col in gdf.columns:
            gdf.at[seg_idx, col] = apex

    # Clear curb return geometry for merged ramps
    gdf.at[ca.street_a_idx, "curb_return_geometry"] = None
    gdf.at[cb.street_a_idx, "curb_return_geometry"] = None
```

**Step 2: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: multi-face conflict resolution with contiguity preservation (step 18)"
```

---

### Task 10: Crosswalk Population (Steps 19-20)

**Files:**
- Modify: `Implementations/ProximityModel.py`

**Step 1: Write `_ix_populate_crosswalks`**

```python
# OSM crossing tag -> schema column mapping
_CROSSWALK_TAG_MAP = {
    "crossing": "type",
    "crossing:markings": "markings",
    "crossing:signals": "signals",
    "crossing:island": "island",
    "crossing:continuous": "continuous",
    "button_operated": "signals",  # merged into signals field
    "traffic_signals:sound": "signals",
    "traffic_signals:vibration": "signals",
    "flashing_lights": "signals",
    "kerb": "kerb",
    "tactile_paving": "tactile_paving",
    "traffic_calming": "traffic_calming",
}


def _ix_populate_crosswalks(
    node: IntersectionNode,
    gdf: gpd.GeoDataFrame,
    crosswalk_cache: gpd.GeoDataFrame | None,
    config: IntersectionConfig,
    crosswalk_counter: list[int],
) -> None:
    """Match cached crosswalks to intersection and populate schema columns (steps 19-20)."""
    # Step 19: explicit crosswalks from OSM/government data
    matched_segments: set[tuple[int, str]] = set()  # (seg_idx, "start"|"end") already populated

    if crosswalk_cache is not None and not crosswalk_cache.empty:
        # Find crosswalks within the intersection boundary bbox
        bbox = node.boundary.bounds  # (minx, miny, maxx, maxy)
        candidates = crosswalk_cache.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        for _, cw_row in candidates.iterrows():
            cw_pt: Point = cw_row.geometry
            if not node.boundary.contains(cw_pt):
                continue

            # Find the closest street segment
            best_seg = None
            best_dist = float("inf")
            best_end = None
            for seg_idx, end in node.segment_entries:
                street_geom = gdf.at[seg_idx, "street_geometry"]
                if street_geom is None or street_geom.is_empty:
                    continue
                d = street_geom.distance(cw_pt)
                if d < best_dist:
                    best_dist = d
                    best_seg = seg_idx
                    best_end = end

            if best_seg is None:
                continue

            crosswalk_counter[0] += 1
            prefix = f"crosswalk_{best_end}"

            # Populate attributes
            gdf.at[best_seg, f"{prefix}_id"] = str(crosswalk_counter[0])

            # Map OSM tags to schema columns
            controlled = "no"
            marked = "no"
            for tag, schema_field in _CROSSWALK_TAG_MAP.items():
                val = cw_row.get(tag)
                if val is not None and not _is_na(val):
                    col = f"{prefix}_{schema_field}"
                    if col in gdf.columns:
                        # Append to signals field if multiple signal tags
                        if schema_field == "signals":
                            existing = gdf.at[best_seg, col]
                            if existing is not None and not _is_na(existing):
                                gdf.at[best_seg, col] = f"{existing},{tag}={val}"
                            else:
                                gdf.at[best_seg, col] = f"{tag}={val}"
                        else:
                            gdf.at[best_seg, col] = str(val)

            # Controlled/marked detection
            crossing_type = cw_row.get("crossing")
            if crossing_type in ("traffic_signals", "controlled"):
                gdf.at[best_seg, f"{prefix}_controlled"] = "yes"
            elif crossing_type == "uncontrolled":
                gdf.at[best_seg, f"{prefix}_controlled"] = "no"
            if cw_row.get("crossing:markings") not in (None, "no", ""):
                gdf.at[best_seg, f"{prefix}_marked"] = "yes"

            # Draw geometry: left_ramp -> crosswalk_point -> right_ramp
            _ix_draw_crosswalk_geometry(best_seg, best_end, cw_pt, gdf)

            matched_segments.add((best_seg, best_end))

    # Step 20: implicit crosswalks where ramps exist but no OSM data
    for seg_idx, end in node.segment_entries:
        if (seg_idx, end) in matched_segments:
            continue

        left_ramp = gdf.at[seg_idx, f"sidewalk_left_curbramp_{end}_1_geometry"]
        right_ramp = gdf.at[seg_idx, f"sidewalk_right_curbramp_{end}_1_geometry"]

        if (left_ramp is not None and isinstance(left_ramp, Point) and
                right_ramp is not None and isinstance(right_ramp, Point)):
            # Draw geometry only, no attributes
            _ix_draw_crosswalk_geometry(seg_idx, end, None, gdf)


def _ix_draw_crosswalk_geometry(
    seg_idx: int,
    end: str,
    crosswalk_point: Point | None,
    gdf: gpd.GeoDataFrame,
) -> None:
    """Draw crosswalk geometry as segments: left_ramp -> [crosswalk_point] -> right_ramp."""
    left_ramp = gdf.at[seg_idx, f"sidewalk_left_curbramp_{end}_1_geometry"]
    right_ramp = gdf.at[seg_idx, f"sidewalk_right_curbramp_{end}_1_geometry"]

    if not isinstance(left_ramp, Point) or not isinstance(right_ramp, Point):
        return

    if crosswalk_point is not None:
        # Two segments: left_ramp -> crosswalk_point -> right_ramp
        geom = LineString([left_ramp, crosswalk_point, right_ramp])
    else:
        # Single segment: left_ramp -> right_ramp
        geom = LineString([left_ramp, right_ramp])

    col = f"crosswalk_{end}_geometry"
    if col in gdf.columns:
        gdf.at[seg_idx, col] = geom
```

**Step 2: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: crosswalk population from OSM cache and implicit generation (steps 19-20)"
```

---

### Task 11: Main Entry Point + Pipeline Wiring

**Files:**
- Modify: `Implementations/ProximityModel.py:2529` (replace placeholder)
- Modify: `Implementations/ProximityModel.py:1254-1256` (wire into `populate_schema`)

**Step 1: Write `intersection_analysis` main function**

Replace the placeholder at line ~2529:

```python
def intersection_analysis(
    gdf: gpd.GeoDataFrame,
    grid_result: GridResult,
    crosswalk_cache: gpd.GeoDataFrame | None = None,
    config: IntersectionConfig | None = None,
) -> gpd.GeoDataFrame:
    """Process all intersection nodes (pipeline steps 12-20). Modifies gdf in-place.

    For each intersection node (degree >= 3):
    1. Build intersection boundary (step 12)
    2. Classify approaching facilities (step 13)
    3. Build corners and check contiguity (steps 14-15)
    4. Resolve flagged facilities per corner (step 16)
    5. Generate curb ramps (step 17)
    6. Resolve cross-corner conflicts (step 18)
    7. Populate crosswalks (steps 19-20)
    """
    if config is None:
        config = IntersectionConfig()

    # Build node -> segment index mapping
    from collections import defaultdict
    node_segments: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for idx in gdf.index:
        if gdf.at[idx, "start_node_is_intersection_node"] == True:  # noqa: E712
            nid = gdf.at[idx, "start_node_id"]
            if not _is_na(nid):
                node_segments[int(nid)].append((idx, "start"))
        if gdf.at[idx, "end_node_is_intersection_node"] == True:  # noqa: E712
            nid = gdf.at[idx, "end_node_id"]
            if not _is_na(nid):
                node_segments[int(nid)].append((idx, "end"))

    ramp_counter = [0]
    crosswalk_counter = [0]
    processed = 0
    flagged = 0

    for node_id in tqdm(node_segments, desc="Intersection analysis", unit="node"):
        seg_entries = node_segments[node_id]
        if len(seg_entries) < 2:
            continue  # Need at least 2 segments to form a corner

        try:
            # Step 12: Build boundary
            node_point = _ix_get_node_point(node_id, seg_entries, gdf)
            boundary = _ix_build_boundary(node_id, node_point, seg_entries, gdf, config)

            # Step 13: Classify facilities
            facilities = _ix_classify_facilities(node_id, seg_entries, boundary, gdf, config)

            # Apply trimming/splitting
            processed_facilities: list[ApproachingFacility] = []
            for f in facilities:
                if f.classification == "overshooting":
                    _ix_trim_overshooting(f, boundary, gdf)
                    processed_facilities.append(f)
                elif f.classification == "continuous":
                    split_results = _ix_split_continuous(f, boundary, seg_entries, gdf)
                    processed_facilities.extend(split_results)
                else:
                    processed_facilities.append(f)

            # Build corners
            corners = _ix_build_corners(node_id, seg_entries, gdf)
            if not corners:
                continue

            # Assign facilities to corners
            _ix_assign_facilities_to_corners(corners, processed_facilities)

            # Steps 14-15: Contiguity checks
            for corner in corners:
                _ix_check_contiguity(corner, config)

            # Step 16: Resolve per corner
            street_geoms = [gdf.at[idx, "street_geometry"] for idx, _ in seg_entries]
            valid_geoms = [g.buffer(0.5) for g in street_geoms if g is not None and not g.is_empty]
            from shapely.ops import unary_union
            exclusion_zone = unary_union(valid_geoms) if valid_geoms else Point(0, 0).buffer(0.1)

            for corner in corners:
                exclusion_zone = _ix_resolve_corner(
                    corner, node_point, exclusion_zone, boundary, gdf, config,
                )

            # Step 17: Curb ramps
            for corner in corners:
                _ix_generate_curb_ramps(corner, node_point, gdf, config, ramp_counter)

            # Step 18: Cross-corner conflicts
            node_data = IntersectionNode(
                node_id=node_id,
                node_point=node_point,
                segment_entries=seg_entries,
                boundary=boundary,
                corners=corners,
                exclusion_zone=exclusion_zone,
            )
            _ix_resolve_cross_corner_conflicts(node_data, gdf, config)

            # Steps 19-20: Crosswalks
            _ix_populate_crosswalks(node_data, gdf, crosswalk_cache, config, crosswalk_counter)

            processed += 1

        except Exception as e:
            # Flag all segments at this node for review
            for seg_idx, _ in seg_entries:
                gdf.at[seg_idx, "intersetion_review_flag"] = True
            flagged += 1
            if flagged <= 5:
                print(f"  [IX_ERROR] node {node_id}: {e}")

    print(f"Intersection analysis: {processed} nodes processed, "
          f"{flagged} flagged for review, "
          f"{ramp_counter[0]} curb ramps, {crosswalk_counter[0]} crosswalks.")

    return gdf
```

**Step 2: Wire into `populate_schema`**

In `populate_schema`, after `_assign_facility_grid_ids` (line ~1255) and before the export block:

```python
    populated = _assign_facility_grid_ids(populated)

    # --- Intersection analysis (pipeline steps 12-20) ---
    populated = intersection_analysis(populated, grid_result, crosswalk_cache=crosswalk_cache)

    # --- Export ---
```

**Step 3: Run the full pipeline on a test city**

Run: `cd c:\Dev\Proximity && C:\Users\karna\miniconda3\envs\ParkximityENV\python.exe -c "from Implementations.ProximityModel import populate_schema; gdf = populate_schema('San Francisco County, California, USA'); print(f'Curb ramps populated: {gdf[\"sidewalk_left_curbramp_start_1_geometry\"].notna().sum() + gdf[\"sidewalk_right_curbramp_start_1_geometry\"].notna().sum()}'); print(f'Review flags: {gdf[\"intersetion_review_flag\"].sum()}')" `

Expected: Pipeline completes without crash. Prints count of populated curb ramps and review flags.

**Step 4: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: intersection_analysis entry point wired into pipeline (steps 12-20)"
```

---

### Task 12: Test Map Visualization Update

**Files:**
- Modify: `Implementations/test_maps.py`

**Step 1: Update test_maps to render curb ramps and crosswalks**

The existing test map visualization already has curb ramp rendering scaffolding (lines 59-62 define `_CURBRAMP_SIDES`, `_CURBRAMP_POSITIONS`, `_CURBRAMP_INDICES`). Verify the existing rendering code picks up the newly populated curb ramp geometry columns and crosswalk geometry columns. Add crosswalk rendering if not present.

**Step 2: Generate updated test maps**

Run: `cd c:\Dev\Proximity && C:\Users\karna\miniconda3\envs\ParkximityENV\python.exe Implementations/test_maps.py`

**Step 3: Visual inspection**

Open the generated HTML maps in `Output/test_maps/` and verify:
- Curb ramps appear as orange dots at intersection corners
- Crosswalks appear as lines connecting left/right ramps
- Sidewalk geometries are trimmed/extended at intersection boundaries
- Review-flagged intersections can be identified

**Step 4: Commit**

```bash
git add Implementations/test_maps.py
git commit -m "feat: test map visualization for curb ramps and crosswalks"
```

---

### Task 13: Integration Testing + Debugging

**Step 1: Run full pipeline on both counties**

Run: `cd c:\Dev\Proximity && C:\Users\karna\miniconda3\envs\ParkximityENV\python.exe -c "from Implementations.ProximityModel import run_all_cities; run_all_cities()"`

**Step 2: Check review flag rates**

If review flags exceed ~5% of intersection nodes, investigate common failure modes and adjust `IntersectionConfig` defaults.

**Step 3: Spot-check the 4 screenshot locations**

Generate test maps for each of the 4 locations from the original screenshots:
1. SF Market/Haight area (5+ way intersection)
2. Berkeley Allston Way (standard grid)
3. SF Embarcadero (complex plaza area)
4. SF Embarcadero south (waterfront)

Verify the intersection normalization produces reasonable results at each.

**Step 4: Commit final adjustments**

```bash
git add -A
git commit -m "feat: intersection analysis integration tested and tuned"
```

---

## Notes for the Implementing Engineer

1. **Import `shapely.ops.split` and `shapely.ops.snap`** — these are used in trimming/splitting. Import them at the top of the file alongside existing Shapely imports.

2. **Import `collections.defaultdict`** — used in the main loop. Add to top-level imports.

3. **The `_ix_` prefix** namespaces all intersection analysis helpers to avoid collisions with existing `_` prefixed helpers in the file.

4. **EPSG:32610** — all geometry operations are in metres. The export step at the end of `populate_schema` handles WKB serialization.

5. **The `intersetion_review_flag` column** preserves the typo from `ProximitySchema.md` (`intersetion` not `intersection`). This is intentional per the spec.

6. **Performance consideration** — for large counties (SF has ~4800 intersection nodes, Alameda ~4400), the per-node loop with per-corner geometry operations may take 5-15 minutes. Use `tqdm` progress bars throughout. If too slow, the spatial operations in `_ix_classify_facilities` can be optimized with STRtree pre-indexing of all facility geometries.

7. **The `crosswalk_cache` must be in EPSG:32610** before being passed to `intersection_analysis`. This conversion happens in `populate_schema` right after the query.
