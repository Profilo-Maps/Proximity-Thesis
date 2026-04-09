# Hull-Guided Crosswalk Generation — Design

**Date:** 2026-03-24
**Status:** Approved
**Scope:** Refactor `_create_crosswalk_geometries` (pipeline step 13)

## Problem

The existing implementation searches for crosswalk evidence bottom-up (footway chains, sidewalk crossings, ramp pairs) and stores whatever it finds. This produces two failure modes:

1. **Missing crosswalks** — slots with evidence are missed because the evidence search is not slot-scoped, so conflicts between N-S and E-W arms cause evidence to be skipped.
2. **Spurious/long crosswalks** — the spanning-sidewalk fallback accepts any sidewalk whose endpoints straddle a street centreline, including full-block sidewalks, producing crosswalks far outside any intersection hull.

## Design: Top-Down Slots, Bottom-Up Evidence Fills

### Principle

The outer loop is over **slots**, not evidence. A slot is one arm of an intersection. Every slot that has pedestrian infrastructure nearby gets a crosswalk geometry — "has pedestrian infrastructure" is satisfied implicitly: if Cases A/B/C and the fallback tiers all fail to find anchors on both sides, the slot stays empty with no geometry. The quality of a filled slot's geometry depends on what evidence was available.

---

### Section 1 — Slot Enumeration

For every node in `node_to_segs`, enumerate one slot per arm: `(street_row, position, slot_zone_polygon)`.

**Nodes with a single arm (dead-ends) are skipped** — there is no ring to complete.

**Slot zone polygon:**
- Get the arm direction unit vector `d̂` (node → far end of the arm, i.e. the opposite endpoint of the road row from this intersection node).
- The crosswalk direction is `d̂⊥` (perpendicular to the arm).
- Build an axis-aligned rectangle in the rotated frame centered on the node:
  - Along `d̂`: ±`strip_half_width` = half the arm's computed lane-width (e.g. `_sidewalk_max_offset_m(arm_row) / 2`, defaulting to `default_lane_width_m / 2 ≈ 1.75 m`).
  - Along `d̂⊥`: ±`hull_fallback_r` (the global fallback constant, e.g. 15 m — large enough to span any intersection regardless of arm-specific offset variation).
- Clip to the intersection hull polygon → **slot zone polygon** (an irregular polygon at angled or T-intersections; the full clipped result is used for all subsequent intersection tests).

**Slot ordering:** Before processing slots at a node, sort by descending count of footway/sidewalk candidates that intersect the slot zone polygon. Slots with more candidates run first and claim shared evidence near the centre before adjacent slots can.

**Overlapping zones at T-intersections:** The base arm of a T will have a wide slot zone that overlaps the two lateral arm zones. The lateral arms run first (more candidates), absorbing the crossing footways. What remains for the base arm's slot is only evidence genuinely crossing the base arm centreline.

---

### Section 2 — Evidence Search (Cases A → B → C per Slot)

For each slot, try cases in priority order. First success fills the slot.

**"Arm centreline" definition:** The arm centreline used for intersection tests is the **half-segment from the intersection node toward the arm's far endpoint** — specifically a LineString constructed as `[node_pt, far_end_pt]`. This prevents false positives from mid-block footways that happen to lie near a distant end of the same road row.

**Case A — Footway chains** (highest priority):
- Initial candidates: footway rows (`highway` ∈ `_FOOTWAY_CROSSING_HW_TYPES`) whose geometry intersects the **slot zone polygon**.
- BFS-chain initial candidates by shared OSM node IDs (same logic as current); trace from degree-1 endpoint; merge coordinates into a single LineString.
- **Centreline filter applies at the merged chain level**: accept the chain only if the merged LineString point-intersects the **half-segment arm centreline** within the slot zone. Individual segments do not need to cross the centreline — only the merged result does. This preserves multi-segment crossings where stub pieces sit entirely on one side.
- Absorbed rows are marked and skipped by subsequent slots at the same node.
- Curb-return segments (those that do NOT cross any local street centreline) are classified and stored in `curb_return_{pos}_{n}_geometry` as before; they are absorbed but not used as crosswalk evidence.

**Case B — Sidewalk crossing:**
- Candidates: sidewalk geometries intersecting the slot zone AND crossing the half-segment arm centreline.
- Extract sub-segment between flanking curb ramps (existing logic, hull-constrained ramp search).
- Spanning-sidewalk fallback applies, but is **gated by the slot zone**: the resulting crosswalk midpoint must lie inside the slot zone polygon, otherwise skip. This eliminates the long-crosswalk regression.
- Accept first valid sub-segment.

**Case C — Curb ramp pair:**
- Find two curb ramps on opposite sides of the half-segment arm centreline, both within the slot zone polygon.
- Straight line between them.
- Accept if line point-intersects the half-segment arm centreline.

---

### Section 3 — Fallback Synthesis

When Cases A/B/C all fail, synthesize a geometry from the best available anchor points.

**"Adjacent-arm rows" lookup:** All `(row_idx, _)` entries in `node_to_segs[node_key]` — iterate every arm row at this node and search each for ramps/endpoints within the slot zone.

**Tier 1 — Curb ramp pair** (`source = "ramp_pair"`):
- Search `sidewalk_{side}_curbramp_{start|end}_1_geometry` on all arm rows at this node, keeping ramps inside the slot zone.
- Both sides of the half-segment arm centreline must have at least one ramp. Straight line between the closest pair on opposite sides.

**Tier 2 — Sidewalk endpoint pair** (`source = "sw_endpoints"`):
- No ramps available; find sidewalk geometry endpoints within the slot zone on each side of the arm centreline.
- Both sides must have an endpoint. Straight line between them.
- Each endpoint is **promoted to a curb ramp**: written to `sidewalk_{side}_curbramp_{start|end}_1_geometry` if that slot is currently null/NA. Promoted ramps are flagged with a companion boolean column `sidewalk_{side}_curbramp_{start|end}_1_synthesized = True` so downstream step 14 can distinguish them from survey-derived ramps.

**No Tier 3.** If neither tier finds anchors on both sides the slot remains empty — no geometry is synthesized from pure street geometry.

---

### Section 4 — Out-of-Zone Validation

After all slots are filled, check each stored crosswalk geometry against its **slot zone polygon** (not just the hull — a stricter check that catches geometries inside the hull but misaligned with the arm):

| Condition | `crosswalk_{pos}_quality` value |
|---|---|
| Midpoint outside the slot zone polygon | `"out_of_zone"` + log warning |
| Length > 2 × `hull_fallback_r` | `"suspect_length"` |
| Otherwise | `"ok"` |

`crosswalk_{pos}_quality` is a new nullable string column, populated only when a crosswalk geometry exists. Flagged crosswalks are retained — downstream consumers filter on this column.

`crosswalk_{pos}_source` records how the geometry was produced: `"case_a"` | `"case_b"` | `"case_c"` | `"ramp_pair"` | `"sw_endpoints"`.

---

## Schema Changes

| Column | Type | Notes |
|---|---|---|
| `crosswalk_{pos}_source` | string (nullable) | New. Insert after `crosswalk_{pos}_condition` and before `crosswalk_{pos}_geometry` in the ordered schema column list (lines ~1672–1683 of ProximityModel.py). |
| `crosswalk_{pos}_quality` | string (nullable) | New. Insert after `crosswalk_{pos}_source` and before `crosswalk_{pos}_geometry`. |
| `sidewalk_{side}_curbramp_{pos}_{n}_synthesized` | bool (nullable) | New. Companion to promoted Tier-2 ramp writes. |
| `curb_return_{pos}_{n}_geometry` | geometry (nullable) | Existing (added previous session). Already in `SECONDARY_GEOM_COLS`. |

`crosswalk_{pos}_source` and `crosswalk_{pos}_quality` are strings, not geometries — they do not belong in `SECONDARY_GEOM_COLS`.

## Affected Code

- `_create_crosswalk_geometries` in `Implementations/ProximityModel.py` — full restructure of outer loop and evidence cases.
- `_build_intersection_hulls` — unchanged; slot zone computation uses existing hull + arm-direction data already returned.
- `specs/ProximityPipelineOutline.md` — update step 13 description after implementation.
