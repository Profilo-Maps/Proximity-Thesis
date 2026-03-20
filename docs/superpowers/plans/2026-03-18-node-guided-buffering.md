# Node-Guided Sequential Buffering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current independent-per-segment buffering with a node-guided sequential approach that produces connected sidewalk/bikelane geometries by construction, using OSM node topology to determine processing order and anchor points.

**Architecture:** The new system processes buffer candidates in BFS order from a seed segment, computing meeting points at each shared node before writing the buffer geometry. At each node, it first tries to anchor to an existing separate or previously-buffered geometry; if none exists, it falls back to a bisector + offset-width meeting point. The intersection analysis step is simplified to a verification/fixup role for corner geometry, rather than being the primary connector.

**Tech Stack:** Python 3.14, Shapely 2.x, GeoPandas, NumPy, pandas

---

## File Structure

All changes are in a single file:

- **Modify:** `Implementations/ProximityModel.py`
  - Replace the buffering loop in `_populate_separate_facilities()` (lines ~2210-2361) with the new node-guided system
  - Add new functions: `_node_guided_buffering()`, `_compute_node_meeting_point()`, `_buffer_segment_to_endpoints()`
  - Simplify `_validate_buffered_connectivity()` to a diagnostic/audit function (or remove)
  - Adjust `_ix_resolve_corner()` to verify pre-connected buffers and only fix failures
  - Update `populate_schema()` pipeline call sequence

- **Test:** `Implementations/test_maps.py` — visual verification via existing test map infrastructure (no unit test framework currently exists in the project)

---

## Conceptual Overview

### Current Flow
```
_populate_separate_facilities() → independent offset_curve() per segment
  → _validate_buffered_connectivity() → post-hoc snap/discard
  → intersection_analysis() → _ix_resolve_corner() does full corner meeting-point computation
```

### New Flow
```
_populate_separate_facilities() → separate geometry matching (unchanged)
  → _node_guided_buffering() → BFS-ordered buffering with meeting points at nodes
  → _audit_buffered_connectivity() → diagnostic pass (flag, don't fix)
  → intersection_analysis() → _ix_resolve_corner() verifies & fixes remaining gaps only
```

### Node Meeting Point Priority
At each shared node between two segments, when determining where a buffer endpoint should land:

1. **Anchor to existing separate geometry** — If an adjacent segment at this node already has a non-buffered (separate) sidewalk/bikelane geometry, compute where the new buffer should terminate to meet it.
2. **Anchor to previously-buffered geometry** — If an adjacent segment was already buffered in an earlier BFS step, snap the new buffer's node-side endpoint to that buffer's endpoint at the shared node.
3. **Bisector + offset default** — If no adjacent geometry exists yet at this node (first arrival), compute the corner meeting point using the angular bisector and offset widths (same logic currently in `_ix_find_meeting_point`). Store this meeting point so subsequent segments arriving at the same node can anchor to it.

### Dead-End Handling
At dead-end nodes (degree 1), `_compute_node_meeting_point()` returns `None`. The buffer endpoint at that node stays wherever `offset_curve()` places it — this is correct behavior since there is no adjacent facility to connect to. The audit function exempts degree-1 nodes from connectivity checks, matching the existing `_validate_buffered_connectivity()` exemption.

### CRS Assumption
All meeting point computations use direct coordinate arithmetic (adding metres to projected coordinates). This is valid because the pipeline reprojects to EPSG:32610 (UTM Zone 10N, metres) at line 1451 before any geometry operations.

---

## Task 1: Build Node Adjacency Graph for Buffer Candidates

**Files:**
- Modify: `Implementations/ProximityModel.py` (add after line ~2384, after `_FACILITY_SLOTS`)

### Purpose
Build a graph of which buffer-candidate segments share intersection nodes. This graph drives BFS ordering. Only segments that need buffering (have type/presence data but no geometry) participate. Segments with separate geometries are not in the BFS graph but their endpoints ARE registered as anchor points.

- [ ] **Step 1: Write `_build_buffer_adjacency()` function**

This function takes the populated GeoDataFrame (after separate facility matching) and returns:
- `buffer_candidates`: `dict[str, set[int]]` — per facility slot, the set of row indices needing buffers
- `node_to_segments`: `dict[int, list[tuple[int, str]]]` — node_id → list of (row_idx, "start"|"end") for ALL segments (not just candidates)
- `node_anchors`: `dict[tuple[int, str], dict[int, Point]]` — `(node_id, facility_slot)` → existing endpoint from separate geometries at that node

```python
def _build_buffer_adjacency(
    populated: gpd.GeoDataFrame,
) -> tuple[
    dict[str, set[int]],                           # buffer_candidates per slot
    dict[int, list[tuple[int, str]]],               # node_to_segments
    dict[tuple[int, str], dict[int, Point]],        # node_anchors: (node_id, slot) → {row_idx: Point}
]:
    """Build the node adjacency graph and identify anchor points for guided buffering.

    Scans all segments to build:
    1. Which segments need buffers (have data but no geometry) per facility slot
    2. Which segments share each intersection node
    3. Which nodes already have facility endpoints from separate (non-buffered) geometries

    Anchor points are the endpoints of existing separate sidewalk/bikelane geometries,
    keyed by (node_id, facility_slot). These are ground-truth positions that new
    buffers should try to connect to.
    """
    from collections import defaultdict

    _NEGATIVE_VALUES = {"no", "none"}
    _SEPARATE_PRESENCE_VALUES = {"separate"}

    buffer_candidates: dict[str, set[int]] = {}
    node_to_segments: dict[int, list[tuple[int, str]]] = defaultdict(list)
    node_anchors: dict[tuple[int, str], dict[int, Point]] = defaultdict(dict)

    # 1. Build node → segment mapping from start_node_id / end_node_id
    for idx in populated.index:
        sn_raw = populated.at[idx, "start_node_id"]
        en_raw = populated.at[idx, "end_node_id"]
        if _is_na(sn_raw) or _is_na(en_raw):
            continue
        sn, en = int(sn_raw), int(en_raw)
        node_to_segments[sn].append((idx, "start"))
        node_to_segments[en].append((idx, "end"))

    # 2. Identify buffer candidates and existing anchors per slot
    for kind, side, slot in _FACILITY_SLOTS:
        sub_id = f"{kind}_{side}_{slot}" if slot else f"{kind}_{side}"
        geom_col = f"{sub_id}_geometry"
        buff_col = f"{sub_id}_buffered"
        data_col = f"{sub_id}_type" if kind == "bikeway" else f"{sub_id}_presence"

        if geom_col not in populated.columns or data_col not in populated.columns:
            continue

        skip_values = _NEGATIVE_VALUES | _SEPARATE_PRESENCE_VALUES if kind == "sidewalk" else _NEGATIVE_VALUES
        has_data = populated[data_col].notna() & ~populated[data_col].astype(str).str.lower().isin(skip_values)

        _gvals = populated[geom_col]
        has_geom = _gvals.notna() & _gvals.map(type).isin([LineString, MultiLineString])

        # Candidates: have data but no geometry
        candidates = has_data & ~has_geom
        buffer_candidates[sub_id] = set(populated.index[candidates])

        # Anchors: have real (non-buffered) geometry — register their endpoints at nodes
        is_buffered = pd.Series(False, index=populated.index)
        if buff_col in populated.columns:
            is_buffered = populated[buff_col].isin([True, "yes", "Yes"])
        real_geom_mask = has_geom & ~is_buffered

        for idx in populated.index[real_geom_mask]:
            geom = populated.at[idx, geom_col]
            if not isinstance(geom, (LineString, MultiLineString)):
                continue
            coords = _flatten_coords(geom)
            if len(coords) < 2:
                continue

            street_geom = populated.at[idx, "street_geometry"]
            if street_geom is None or not hasattr(street_geom, "coords"):
                continue
            street_coords = _flatten_coords(street_geom)
            if len(street_coords) < 2:
                continue

            sn_raw = populated.at[idx, "start_node_id"]
            en_raw = populated.at[idx, "end_node_id"]
            if _is_na(sn_raw) or _is_na(en_raw):
                continue
            sn, en = int(sn_raw), int(en_raw)

            # Determine which facility endpoint is near which street endpoint (node)
            fac_start = Point(coords[0])
            fac_end = Point(coords[-1])
            street_start = Point(street_coords[0])
            street_end = Point(street_coords[-1])

            if fac_start.distance(street_start) < fac_start.distance(street_end):
                node_anchors[(sn, sub_id)][idx] = fac_start
                node_anchors[(en, sub_id)][idx] = fac_end
            else:
                node_anchors[(sn, sub_id)][idx] = fac_end
                node_anchors[(en, sub_id)][idx] = fac_start

    return buffer_candidates, dict(node_to_segments), dict(node_anchors)
```

- [ ] **Step 2: Verify function runs without error**

Add a temporary call in `_populate_separate_facilities()` after the separate facility matching (before the current buffering loop at line ~2210) and print the counts:

```python
buf_cands, node_segs, node_anch = _build_buffer_adjacency(populated)
for slot, idxs in buf_cands.items():
    print(f"  [NODE_BUF] {slot}: {len(idxs)} candidates")
print(f"  [NODE_BUF] {len(node_segs)} nodes, {len(node_anch)} anchor entries")
```

Run: `C:/Users/karna/miniconda3/envs/ParkXimityENV/python.exe Implementations/ProximityModel.py`

Expected: Prints candidate counts per slot matching the existing `candidates:` output, plus node/anchor counts.

- [ ] **Step 3: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: add _build_buffer_adjacency for node-guided buffering"
```

---

## Task 2: Implement Node Meeting Point Computation

**Files:**
- Modify: `Implementations/ProximityModel.py`

### Purpose
Given two segments meeting at a node and a facility slot, compute the meeting point where both buffers should terminate. This encapsulates the 3-tier priority: anchor to separate → anchor to buffered → bisector default.

- [ ] **Step 1: Write `_compute_node_meeting_point()` function**

```python
@dataclass
class NodeMeetingPoint:
    """Result of computing where buffer endpoints should meet at a node."""
    point: Point
    source: str  # "separate_anchor", "buffered_anchor", "bisector_default"


def _compute_node_meeting_point(
    node_id: int,
    facility_slot: str,
    segment_idx: int,
    segment_end: str,
    populated: gpd.GeoDataFrame,
    node_to_segments: dict[int, list[tuple[int, str]]],
    node_anchors: dict[tuple[int, str], dict[int, Point]],
    node_meeting_cache: dict[tuple[int, str], NodeMeetingPoint],
    default_lane_width_m: float,
) -> NodeMeetingPoint | None:
    """Compute where a buffer endpoint should land at a given node.

    Priority:
    1. Anchor to existing separate geometry endpoint at this node
    2. Anchor to previously-buffered geometry endpoint at this node
    3. Compute bisector + offset meeting point (requires >=2 streets at node)

    Results are cached in node_meeting_cache so all segments arriving at the
    same (node, slot) converge to the same point.
    """
    cache_key = (node_id, facility_slot)

    # Check cache first — reuse previously computed meeting point
    if cache_key in node_meeting_cache:
        return node_meeting_cache[cache_key]

    # --- Priority 1: Anchor to separate geometry ---
    if cache_key in node_anchors:
        # Pick the anchor from the closest adjacent segment (not self)
        anchor_entries = node_anchors[cache_key]
        for other_idx, anchor_pt in anchor_entries.items():
            if other_idx != segment_idx:
                result = NodeMeetingPoint(point=anchor_pt, source="separate_anchor")
                node_meeting_cache[cache_key] = result
                return result
        # If only self has an anchor, use it (degenerate but valid)
        if anchor_entries:
            pt = next(iter(anchor_entries.values()))
            result = NodeMeetingPoint(point=pt, source="separate_anchor")
            node_meeting_cache[cache_key] = result
            return result

    # --- Priority 2: Anchor to previously-buffered geometry ---
    # Check if any adjacent segment at this node already has a buffered geometry
    # for this facility slot
    parts = facility_slot.split("_")
    kind = parts[0]
    side = parts[1]
    slot_num = parts[2] if len(parts) > 2 else None
    geom_col = f"{facility_slot}_geometry"
    buff_col = f"{facility_slot}_buffered"

    for other_idx, other_end in node_to_segments.get(node_id, []):
        if other_idx == segment_idx:
            continue
        if geom_col not in populated.columns:
            continue
        other_geom = populated.at[other_idx, geom_col]
        if not isinstance(other_geom, (LineString, MultiLineString)):
            continue
        # Confirm it's buffered (was processed in an earlier BFS step)
        if buff_col in populated.columns:
            if not populated.at[other_idx, buff_col] in (True, "yes", "Yes"):
                continue

        coords = _flatten_coords(other_geom)
        if len(coords) < 2:
            continue

        # Determine which endpoint of the other buffer faces this node
        other_street = populated.at[other_idx, "street_geometry"]
        if other_street is None or not hasattr(other_street, "coords"):
            continue
        other_street_coords = _flatten_coords(other_street)
        if len(other_street_coords) < 2:
            continue

        # other_end tells us which end of the OTHER segment faces this node
        if other_end == "start":
            street_node_pt = Point(other_street_coords[0])
        else:
            street_node_pt = Point(other_street_coords[-1])

        fac_start = Point(coords[0])
        fac_end = Point(coords[-1])
        anchor_pt = fac_start if fac_start.distance(street_node_pt) < fac_end.distance(street_node_pt) else fac_end

        result = NodeMeetingPoint(point=anchor_pt, source="buffered_anchor")
        node_meeting_cache[cache_key] = result
        return result

    # --- Priority 3: Bisector + offset default ---
    # Need at least 2 road segments at this node to compute a bisector
    seg_entries = node_to_segments.get(node_id, [])
    road_bearings: list[tuple[int, str, float]] = []
    for seg_idx_other, end_other in seg_entries:
        hw = populated.at[seg_idx_other, "highway"] if "highway" in populated.columns else None
        if isinstance(hw, str) and hw in _NON_ROAD_HIGHWAY:
            continue
        street_geom = populated.at[seg_idx_other, "street_geometry"]
        if street_geom is None or not hasattr(street_geom, "coords"):
            continue
        coords = _flatten_coords(street_geom)
        if len(coords) < 2:
            continue
        # Outgoing bearing from node
        if end_other == "start":
            dx = coords[1][0] - coords[0][0]
            dy = coords[1][1] - coords[0][1]
        else:
            dx = coords[-2][0] - coords[-1][0]
            dy = coords[-2][1] - coords[-1][1]
        bearing = math.degrees(math.atan2(dx, dy)) % 360
        road_bearings.append((seg_idx_other, end_other, bearing))

    if len(road_bearings) < 2:
        return None  # Dead-end — no meeting point needed

    # Sort clockwise, find the corner that contains our segment
    road_bearings.sort(key=lambda x: x[2])

    # Find where our segment sits in the clockwise order
    my_pos = None
    for i, (idx_b, end_b, _) in enumerate(road_bearings):
        if idx_b == segment_idx and end_b == segment_end:
            my_pos = i
            break

    if my_pos is None:
        return None

    # The corner for "left" side is between this segment and the previous one (CCW neighbor)
    # The corner for "right" side is between this segment and the next one (CW neighbor)
    n = len(road_bearings)
    if side == "right":
        neighbor_pos = (my_pos + 1) % n
    else:  # left
        neighbor_pos = (my_pos - 1) % n

    my_bearing = road_bearings[my_pos][2]
    neighbor_bearing = road_bearings[neighbor_pos][2]

    # Interior angle and bisector (same logic as _ix_build_corners)
    if side == "right":
        angle = (neighbor_bearing - my_bearing) % 360
    else:
        angle = (my_bearing - neighbor_bearing) % 360
    if angle == 0:
        angle = 360.0

    if side == "right":
        bisector = (my_bearing + angle / 2) % 360
    else:
        bisector = (neighbor_bearing + angle / 2) % 360

    # Compute offset distance
    row = populated.loc[segment_idx]
    raw_lanes = row.get("lanes")
    raw_lane_width = row.get("lane_width")
    lanes = _parse_numeric(raw_lanes, 2.0)
    lane_width = _parse_numeric(raw_lane_width, default_lane_width_m)
    half_road = (lanes * lane_width) / 2.0

    if kind == "bikeway":
        offset_m = half_road
    elif kind == "sidewalk":
        bike_width = 0.0
        for s in ("1", "2"):
            w = row.get(f"bikeway_{side}_{s}_width")
            bike_width += _parse_numeric(w, 0.0) if not _is_na(w) else _DEFAULT_BIKE_WIDTH_M \
                if not _is_na(row.get(f"bikeway_{side}_{s}_type")) else 0.0
        offset_m = half_road + bike_width
    else:
        return None

    # Also get neighbor offset to average.
    # The neighbor's side facing this corner depends on the angular relationship:
    # In clockwise ordering, the corner between street A and street B (next CW)
    # is bounded by A's right side and B's left side. So:
    #   - If we are computing for our "right" side, the neighbor (CW next) faces
    #     this corner with its "left" side.
    #   - If we are computing for our "left" side, the neighbor (CCW prev) faces
    #     this corner with its "right" side.
    # This matches _ix_build_corners / _ix_assign_facilities_to_corners logic.
    neighbor_idx, neighbor_end, _ = road_bearings[neighbor_pos]
    n_row = populated.loc[neighbor_idx]
    n_lanes = _parse_numeric(n_row.get("lanes"), 2.0)
    n_lane_width = _parse_numeric(n_row.get("lane_width"), default_lane_width_m)
    n_half_road = (n_lanes * n_lane_width) / 2.0
    # Determine which side of the neighbor faces this corner
    # Our "right" corner → neighbor's "left"; our "left" corner → neighbor's "right"
    n_side = "left" if side == "right" else "right"
    if kind == "bikeway":
        n_offset = n_half_road
    elif kind == "sidewalk":
        n_bike_width = 0.0
        for s in ("1", "2"):
            w = n_row.get(f"bikeway_{n_side}_{s}_width")
            n_bike_width += _parse_numeric(w, 0.0) if not _is_na(w) else _DEFAULT_BIKE_WIDTH_M \
                if not _is_na(n_row.get(f"bikeway_{n_side}_{s}_type")) else 0.0
        n_offset = n_half_road + n_bike_width
    else:
        n_offset = half_road

    avg_offset = (offset_m + n_offset) / 2.0

    # Walk along bisector to find the meeting point
    street_geom = populated.at[segment_idx, "street_geometry"]
    if street_geom is None:
        return None
    street_coords = _flatten_coords(street_geom)
    if segment_end == "start":
        node_pt = Point(street_coords[0])
    else:
        node_pt = Point(street_coords[-1])

    bearing_rad = math.radians(bisector)
    dx = math.sin(bearing_rad)
    dy = math.cos(bearing_rad)
    meeting_pt = Point(node_pt.x + dx * avg_offset, node_pt.y + dy * avg_offset)

    result = NodeMeetingPoint(point=meeting_pt, source="bisector_default")
    node_meeting_cache[cache_key] = result
    return result
```

- [ ] **Step 2: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: add _compute_node_meeting_point with 3-tier priority"
```

---

## Task 3: Implement BFS-Ordered Buffer Segment Writer

**Files:**
- Modify: `Implementations/ProximityModel.py`

### Purpose
Create the buffer geometry for a single segment, but instead of using `offset_curve()` endpoints as-is, override the node-side endpoints with the computed meeting points. This ensures connectivity by construction.

- [ ] **Step 1: Write `_buffer_segment_to_endpoints()` function**

This is a modified version of `_buffer_segment()` that:
1. Creates the parallel offset geometry (same as current `_buffer_segment`)
2. Replaces the start/end points with the meeting points computed by `_compute_node_meeting_point()`
3. Still performs collision checks

```python
def _buffer_segment_to_endpoints(
    idx: int,
    sub_facility_id: str,
    street_geom: BaseGeometry,
    populated: gpd.GeoDataFrame,
    side: str,
    start_meeting: NodeMeetingPoint | None,
    end_meeting: NodeMeetingPoint | None,
    debug: dict[str, Any],
    default_lane_width_m: float = _DEFAULT_LANE_WIDTH_M,
) -> tuple[gpd.GeoDataFrame, Point | None, Point | None]:
    """Create a buffered facility geometry with endpoints snapped to meeting points.

    1. Compute offset distance (same as _buffer_segment)
    2. Generate parallel-offset geometry via offset_curve()
    3. Replace the start/end coords with meeting points (if provided)
    4. Run collision checks
    5. Write geometry and set buffered flag

    Returns (populated, start_node_endpoint, end_node_endpoint) where the
    endpoint Points map to the street's start/end node respectively,
    accounting for possible buffer orientation reversal. Returns (populated, None, None)
    on failure.
    """
    row = populated.loc[idx]
    parts = sub_facility_id.split("_")
    facility_kind = parts[0]

    # --- Compute offset distance ---
    raw_lanes = row.get("lanes")
    raw_lane_width = row.get("lane_width")
    lanes = _parse_numeric(raw_lanes, 2.0)
    lane_width = _parse_numeric(raw_lane_width, default_lane_width_m)
    half_road = (lanes * lane_width) / 2.0

    if facility_kind == "bikeway":
        offset_m = half_road
    elif facility_kind == "sidewalk":
        bike_width = 0.0
        for slot in ("1", "2"):
            w = row.get(f"bikeway_{side}_{slot}_width")
            bike_width += _parse_numeric(w, 0.0) if not _is_na(w) else _DEFAULT_BIKE_WIDTH_M \
                if not _is_na(row.get(f"bikeway_{side}_{slot}_type")) else 0.0
        offset_m = half_road + bike_width
    else:
        return populated, None, None

    # --- Generate parallel offset ---
    sign = 1 if side == "left" else -1
    buffered_geom: BaseGeometry | None = None
    for fraction in (1.0, 0.75, 0.5, 0.25):
        try:
            cur_offset = sign * offset_m * fraction
            if hasattr(street_geom, "offset_curve"):
                candidate = street_geom.offset_curve(cur_offset)
            else:
                candidate = street_geom.parallel_offset(
                    abs(cur_offset), side=side, resolution=16, join_style=2,
                )
            if not candidate.is_empty:
                buffered_geom = candidate
                break
        except Exception:
            continue

    if buffered_geom is None:
        debug["n_empty_offset"] = debug.get("n_empty_offset", 0) + 1
        return populated, None, None

    # --- Snap endpoints to meeting points ---
    coords = _flatten_coords(buffered_geom)
    if len(coords) < 2:
        debug["n_empty_offset"] = debug.get("n_empty_offset", 0) + 1
        return populated, None, None

    modified = False
    # "start" of the street → coords[0] of the buffer (if buffer follows street direction)
    # Need to check which end of the buffer is near the street start vs end
    street_coords = _flatten_coords(street_geom)
    buf_start = Point(coords[0])
    buf_end = Point(coords[-1])
    street_start = Point(street_coords[0])
    street_end = Point(street_coords[-1])

    # Determine orientation: does buffer[0] correspond to street start or end?
    if buf_start.distance(street_start) <= buf_start.distance(street_end):
        # Buffer is same direction as street
        if start_meeting is not None:
            coords[0] = (start_meeting.point.x, start_meeting.point.y)
            modified = True
        if end_meeting is not None:
            coords[-1] = (end_meeting.point.x, end_meeting.point.y)
            modified = True
    else:
        # Buffer is reversed relative to street
        if start_meeting is not None:
            coords[-1] = (start_meeting.point.x, start_meeting.point.y)
            modified = True
        if end_meeting is not None:
            coords[0] = (end_meeting.point.x, end_meeting.point.y)
            modified = True

    if modified:
        buffered_geom = LineString(coords)

    # --- Collision checks (same as _buffer_segment) ---
    own_geom_col = f"{sub_facility_id}_geometry"
    endpoint_buffer = buffered_geom.boundary.buffer(1e-6)

    def _mid_intersection_clear(a: BaseGeometry, b: BaseGeometry) -> bool:
        if not a.intersects(b):
            return True
        return a.intersection(b).within(endpoint_buffer)

    same_side_sidewalk_col = f"sidewalk_{side}_geometry"
    _FACILITY_GEOM_COLS = [
        "street_geometry",
        "bikeway_left_1_geometry", "bikeway_left_2_geometry",
        "bikeway_right_1_geometry", "bikeway_right_2_geometry",
        "sidewalk_left_geometry", "sidewalk_right_geometry",
    ]
    for col in _FACILITY_GEOM_COLS:
        if col == own_geom_col or col not in populated.columns:
            continue
        if facility_kind == "bikeway" and col == same_side_sidewalk_col:
            continue
        other_geom = populated.at[idx, col]
        if other_geom is None or not hasattr(other_geom, "intersects"):
            continue
        if not _mid_intersection_clear(buffered_geom, cast(BaseGeometry, other_geom)):
            debug.setdefault("collision_counts", {})
            debug["collision_counts"][col] = debug["collision_counts"].get(col, 0) + 1
            return populated, None, None

    # --- Write geometry ---
    geom_col = own_geom_col
    if geom_col in populated.columns:
        populated.at[idx, geom_col] = buffered_geom  # type: ignore[index]
    buffered_col = f"{sub_facility_id}_buffered"
    if buffered_col in populated.columns:
        populated.at[idx, buffered_col] = True  # type: ignore[index]

    # --- Return endpoint mapping for anchor registration ---
    # The caller needs to know which endpoint corresponds to which node,
    # accounting for possible orientation reversal of the buffer vs street.
    final_coords = _flatten_coords(buffered_geom)
    if buf_start.distance(street_start) <= buf_start.distance(street_end):
        # Buffer same direction as street: coords[0] → start_node, coords[-1] → end_node
        start_node_pt = Point(final_coords[0])
        end_node_pt = Point(final_coords[-1])
    else:
        # Buffer reversed: coords[-1] → start_node, coords[0] → end_node
        start_node_pt = Point(final_coords[-1])
        end_node_pt = Point(final_coords[0])

    return populated, start_node_pt, end_node_pt
```

- [ ] **Step 2: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: add _buffer_segment_to_endpoints with meeting point snapping"
```

---

## Task 4: Implement Main BFS Buffering Orchestrator

**Files:**
- Modify: `Implementations/ProximityModel.py`

### Purpose
The main function that replaces the current buffering loop. Processes segments in BFS order per facility slot, computing meeting points at each node before writing geometries.

- [ ] **Step 1: Write `_node_guided_buffering()` function**

```python
def _node_guided_buffering(
    populated: gpd.GeoDataFrame,
    default_lane_width_m: float,
) -> gpd.GeoDataFrame:
    """Buffer facility geometries using node-guided BFS ordering.

    For each facility slot, processes candidates in BFS order from seed segments.
    At each shared node, computes a meeting point (anchor to separate → anchor to
    buffered → bisector default) and writes the buffer with endpoints snapped to
    those meeting points.

    Replaces the previous independent-per-segment buffering loop and the
    _validate_buffered_connectivity() post-hoc fix.
    """
    from collections import deque

    buf_cands, node_segs, node_anch = _build_buffer_adjacency(populated)

    _NEGATIVE_VALUES = {"no", "none"}
    _SEPARATE_PRESENCE_VALUES = {"separate"}

    total_buffered = 0
    total_skipped = 0
    _buffer_debug: dict[str, Any] = {"n_empty_offset": 0, "collision_counts": {}}

    # Build the unified separate sidewalk tree for suppression (same as current code)
    # ... [reuse existing suppression logic from lines 2229-2264] ...
    all_sep_sw_geoms: list[BaseGeometry] = []
    all_sep_sw_bearings: list[float] = []
    all_sep_sw_row_indices: list = []
    all_sep_sw_names: list[str | None] = []
    _has_name_col = "name" in populated.columns
    for sw_side in ("left", "right"):
        gcol = f"sidewalk_{sw_side}_geometry"
        bcol = f"sidewalk_{sw_side}_buffered"
        if gcol not in populated.columns:
            continue
        _gcol_vals = populated[gcol]
        has_geom_mask = _gcol_vals.notna() & _gcol_vals.map(type).isin([LineString, MultiLineString, Point])
        if bcol in populated.columns:
            not_buffered_mask = ~(populated[bcol].isin([True, "yes", "Yes"]))
        else:
            not_buffered_mask = pd.Series(True, index=populated.index)
        candidate_mask = has_geom_mask & not_buffered_mask
        candidate_indices = populated.index[candidate_mask]
        candidate_geoms = populated.loc[candidate_indices, gcol]
        for c_idx, g in zip(candidate_indices, candidate_geoms):
            bearing = _linestring_bearing(cast(BaseGeometry, g))
            if bearing is None:
                continue
            all_sep_sw_geoms.append(cast(BaseGeometry, g))
            all_sep_sw_bearings.append(bearing)
            all_sep_sw_row_indices.append(c_idx)
            if _has_name_col:
                raw_name = populated.at[c_idx, "name"]
                all_sep_sw_names.append(
                    str(raw_name).strip().lower() if raw_name is not None and not _is_na(raw_name) else None
                )
            else:
                all_sep_sw_names.append(None)

    sep_sw_tree = STRtree(all_sep_sw_geoms) if all_sep_sw_geoms else None

    # Process each facility slot in order (bikes first, then sidewalks)
    for kind, side, slot in _FACILITY_SLOTS:
        sub_id = f"{kind}_{side}_{slot}" if slot else f"{kind}_{side}"
        candidates = buf_cands.get(sub_id, set())
        if not candidates:
            continue

        geom_col = f"{sub_id}_geometry"
        node_meeting_cache: dict[tuple[int, str], NodeMeetingPoint] = {}
        visited: set[int] = set()
        n_buffered_slot = 0
        n_suppressed = 0
        n_no_street = 0

        # BFS from each unvisited candidate
        # Seeds: candidates that share a node with an existing anchor
        # (prioritize segments adjacent to real geometries)
        seed_queue: deque[int] = deque()
        non_seed: list[int] = []

        for idx in candidates:
            sn = populated.at[idx, "start_node_id"]
            en = populated.at[idx, "end_node_id"]
            if _is_na(sn) or _is_na(en):
                continue
            has_anchor = (int(sn), sub_id) in node_anch or (int(en), sub_id) in node_anch
            if has_anchor:
                seed_queue.append(idx)
            else:
                non_seed.append(idx)

        # Process seeds first, then remaining
        queue: deque[int] = deque()
        queue.extend(seed_queue)
        queue.extend(non_seed)

        while queue:
            idx = queue.popleft()
            if idx in visited:
                continue
            visited.add(idx)

            street_geom = populated.at[idx, "street_geometry"]
            if street_geom is None or not hasattr(street_geom, "geom_type"):
                n_no_street += 1
                continue

            # Sidewalk suppression check (same as current code)
            if kind == "sidewalk" and sep_sw_tree is not None:
                street_bearing = _linestring_bearing(cast(BaseGeometry, street_geom))
                if street_bearing is not None:
                    _cand_name: str | None = None
                    if _has_name_col:
                        _raw_cand = populated.at[idx, "name"]
                        _cand_name = str(_raw_cand).strip().lower() if _raw_cand is not None and not _is_na(_raw_cand) else None
                    search_area = street_geom.buffer(NEARBY_SEPARATE_SIDEWALK_THRESHOLD_M)
                    hit_indices = sep_sw_tree.query(search_area)
                    skip = False
                    for hi in hit_indices:
                        if all_sep_sw_row_indices[hi] == idx:
                            continue
                        sup_name = all_sep_sw_names[hi]
                        if _cand_name is not None and sup_name is not None and _cand_name != sup_name:
                            continue
                        if not _bearings_parallel(street_bearing, all_sep_sw_bearings[hi]):
                            continue
                        fac_mid = all_sep_sw_geoms[hi].interpolate(0.5, normalized=True)
                        if (street_geom.distance(fac_mid) <= NEARBY_SEPARATE_SIDEWALK_THRESHOLD_M
                                and _road_side(street_geom, fac_mid) == side):
                            skip = True
                            break
                    if skip:
                        n_suppressed += 1
                        total_skipped += 1
                        continue

            # Compute meeting points at start and end nodes
            sn_raw = populated.at[idx, "start_node_id"]
            en_raw = populated.at[idx, "end_node_id"]
            start_meeting: NodeMeetingPoint | None = None
            end_meeting: NodeMeetingPoint | None = None

            if not _is_na(sn_raw):
                sn = int(sn_raw)
                start_meeting = _compute_node_meeting_point(
                    sn, sub_id, idx, "start", populated, node_segs, node_anch,
                    node_meeting_cache, default_lane_width_m,
                )

            if not _is_na(en_raw):
                en = int(en_raw)
                end_meeting = _compute_node_meeting_point(
                    en, sub_id, idx, "end", populated, node_segs, node_anch,
                    node_meeting_cache, default_lane_width_m,
                )

            # Buffer with meeting point endpoints
            populated, start_pt, end_pt = _buffer_segment_to_endpoints(
                idx, sub_id, cast(BaseGeometry, street_geom), populated, side,
                start_meeting, end_meeting, _buffer_debug, default_lane_width_m,
            )

            # Always enqueue unvisited neighbors regardless of buffer success/failure.
            # This prevents BFS chain breaks: if this segment fails, its neighbors
            # can still be reached and may succeed with bisector defaults.
            for nid in [int(sn_raw) if not _is_na(sn_raw) else None,
                        int(en_raw) if not _is_na(en_raw) else None]:
                if nid is None:
                    continue
                for neighbor_idx, _ in node_segs.get(nid, []):
                    if neighbor_idx not in visited and neighbor_idx in candidates:
                        queue.appendleft(neighbor_idx)  # prioritize connected neighbors

            # Check if buffer was written successfully
            geom_after = populated.at[idx, geom_col]  # type: ignore[index]
            if geom_after is not None and hasattr(geom_after, "geom_type"):
                if _is_suspect_geometry(cast(BaseGeometry, geom_after)):
                    populated.at[idx, geom_col] = None  # type: ignore[index]
                else:
                    n_buffered_slot += 1
                    total_buffered += 1

                    # Register this buffer's endpoints as anchors for subsequent
                    # BFS neighbors. Uses the orientation-aware endpoints returned
                    # by _buffer_segment_to_endpoints (start_pt → start_node,
                    # end_pt → end_node).
                    if not _is_na(sn_raw) and start_pt is not None:
                        node_anch.setdefault((int(sn_raw), sub_id), {})[idx] = start_pt
                    if not _is_na(en_raw) and end_pt is not None:
                        node_anch.setdefault((int(en_raw), sub_id), {})[idx] = end_pt

        print(f"  [{sub_id}] buffered={n_buffered_slot}, suppressed={n_suppressed}, "
              f"no_street_geom={n_no_street}")

    print(f"Node-guided buffering: {total_buffered} segments buffered "
          f"({total_skipped} suppressed by nearby separate sidewalks).")
    print(f"  Failures: empty_offset={_buffer_debug.get('n_empty_offset', 0)}, "
          f"collisions={_buffer_debug.get('collision_counts', {})}")
    return populated
```

- [ ] **Step 2: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: add _node_guided_buffering BFS orchestrator"
```

---

## Task 5: Replace Old Buffering Loop and Update Pipeline

**Files:**
- Modify: `Implementations/ProximityModel.py:1541-1542` (pipeline calls)
- Modify: `Implementations/ProximityModel.py:2210-2361` (old buffering loop)

### Purpose
Wire the new system into the pipeline. Remove the old buffering loop from `_populate_separate_facilities()`. Convert `_validate_buffered_connectivity()` to a diagnostic-only audit function.

- [ ] **Step 1: Remove the old buffering loop from `_populate_separate_facilities()`**

Delete or comment out lines ~2210-2361 (the "Buffering pass" section) from `_populate_separate_facilities()`. The function should return `populated` immediately after the separate facility matching and diagnostic printing (line ~2209).

The function signature and all separate-facility matching logic (lines 1876-2209) remain unchanged.

- [ ] **Step 2: Update `populate_schema()` pipeline to call `_node_guided_buffering()`**

In `populate_schema()`, replace:
```python
populated = _populate_separate_facilities(populated, edges_reset, default_lane_width_m)
populated = _validate_buffered_connectivity(populated)
```

With:
```python
populated = _populate_separate_facilities(populated, edges_reset, default_lane_width_m)
populated = _node_guided_buffering(populated, default_lane_width_m)
```

The `_validate_buffered_connectivity()` call is removed because connectivity is now handled by construction.

- [ ] **Step 3: Rename `_validate_buffered_connectivity()` to `_audit_buffered_connectivity()`**

Convert it to a diagnostic function that only logs/prints statistics about connectivity without modifying geometries. Change all `populated.at[idx, gcol] = ...` writes to just increment counters. This is useful for verifying the new approach works but should not modify data.

Add a call after the merge/dedup steps:
```python
populated = _node_guided_buffering(populated, default_lane_width_m)
populated = _merge_disjointed_facility_segments(populated)
populated = _remove_circuitous_duplicate_sidewalks(populated)
_audit_buffered_connectivity(populated)  # diagnostic only
populated = _assign_facility_grid_ids(populated)
```

- [ ] **Step 4: Run the pipeline and compare output**

Run: `C:/Users/karna/miniconda3/envs/ParkXimityENV/python.exe Implementations/ProximityModel.py`

Check:
- "Node-guided buffering:" output shows similar total counts to previous "Buffering pass:" output
- Audit function shows improved connectivity (fewer disconnected, fewer discarded)
- No crashes

- [ ] **Step 5: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: replace independent buffering with node-guided BFS approach"
```

---

## Task 6: Update Intersection Analysis to Verify-and-Fix Mode

**Files:**
- Modify: `Implementations/ProximityModel.py:4450-4483` (`_ix_resolve_corner`)

### Purpose
The intersection analysis should now confirm that buffered endpoints are already connected and only intervene when they're not. This means `_ix_resolve_corner()` should check if facilities are already contiguous before running its full meeting-point computation.

- [ ] **Step 1: Add pre-check to `_ix_check_contiguity()`**

Before flagging, check if the endpoints are already within tolerance. The existing code does this — the change is to add logging when contiguity is already satisfied (showing the new buffering worked):

At the top of `_ix_check_contiguity()`, add a counter for pre-connected pairs:

```python
n_preconnected = 0
# ... in the loop:
if dist <= config.contiguity_tolerance_m:
    n_preconnected += 1
    _ix_logger.debug("  [CONTIGUITY] seg%d↔seg%d %s: pre-connected (%.2fm)",
                     fa.segment_idx, fb.segment_idx, kind, dist)
```

- [ ] **Step 2: Add early-exit in `_ix_resolve_facility_pair()` when already connected**

At the start of `_ix_resolve_facility_pair()`, check if the two facilities are already within tolerance:

```python
pt_a = _ix_facility_endpoint_at_node(fa)
pt_b = _ix_facility_endpoint_at_node(fb)
if pt_a is not None and pt_b is not None:
    if pt_a.distance(pt_b) <= config.contiguity_tolerance_m:
        _ix_logger.debug("  [RESOLVE] seg%d↔seg%d already connected (%.2fm), skipping",
                         fa.segment_idx, fb.segment_idx, pt_a.distance(pt_b))
        # Still add to exclusion zone
        for f in facilities:
            if f.geometry is not None and not f.geometry.is_empty:
                exclusion_zone = exclusion_zone.union(f.geometry.buffer(0.5))
        return exclusion_zone
```

- [ ] **Step 3: Run pipeline and verify intersection analysis logs show pre-connected pairs**

Run: `C:/Users/karna/miniconda3/envs/ParkXimityENV/python.exe Implementations/ProximityModel.py`

Check `Output/intersection_diag.log` for "[CONTIGUITY] ... pre-connected" and "[RESOLVE] ... already connected" messages. A high fraction of these indicates the node-guided buffering is producing correct meeting points.

- [ ] **Step 4: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "feat: intersection analysis now verifies pre-connected buffers before resolving"
```

---

## Task 7: Generate Test Maps and Visual Verification

**Files:**
- Modify: `Implementations/test_maps.py` (if needed for new diagnostics)

### Purpose
Run the full pipeline and generate test maps to visually verify that:
1. Buffered sidewalks meet at intersection corners
2. No visible gaps between adjacent sidewalk segments
3. Curb ramps are placed at connected endpoints

- [ ] **Step 1: Run full pipeline for San Francisco**

```bash
C:/Users/karna/miniconda3/envs/ParkXimityENV/python.exe Implementations/ProximityModel.py
```

- [ ] **Step 2: Generate test maps**

```bash
C:/Users/karna/miniconda3/envs/ParkXimityENV/python.exe Implementations/test_maps.py
```

- [ ] **Step 3: Open test maps and verify connectivity visually**

Open `Output/test_maps/*.html` files in a browser. Check:
- Sidewalk lines (blue/green) meet at corners — no floating endpoints
- Bikelane lines run parallel and connect at intersections
- Curb ramp markers sit at sidewalk endpoints, not floating in space

- [ ] **Step 4: Compare stats**

Check the pipeline output for:
- "Node-guided buffering: X segments buffered" — should be similar count to before
- Audit connectivity: "disconnected=0" or very low
- Intersection analysis: "flagged for review" count should decrease

- [ ] **Step 5: Final commit**

```bash
git add Implementations/ProximityModel.py Implementations/test_maps.py
git commit -m "feat: verify node-guided buffering via test maps"
```

---

## Task 8: Clean Up and Remove Dead Code

**Files:**
- Modify: `Implementations/ProximityModel.py`

### Purpose
Remove the old `_buffer_segment()` function and the original `_validate_buffered_connectivity()` once the new approach is confirmed working.

- [ ] **Step 1: Delete `_buffer_segment()` function (lines ~2577-2731)**

This function is fully replaced by `_buffer_segment_to_endpoints()`.

- [ ] **Step 2: Delete or archive `_validate_buffered_connectivity()` if audit version is sufficient**

If the audit function confirms connectivity is good, the old validation function can be removed entirely.

- [ ] **Step 3: Run Pyright type checking**

Use the VS Code Pylance extension to verify no type errors were introduced.

- [ ] **Step 4: Run full pipeline one more time to confirm clean execution**

```bash
C:/Users/karna/miniconda3/envs/ParkXimityENV/python.exe Implementations/ProximityModel.py
```

- [ ] **Step 5: Commit**

```bash
git add Implementations/ProximityModel.py
git commit -m "refactor: remove dead buffering code after node-guided migration"
```
