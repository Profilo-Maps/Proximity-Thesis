# Street–Sidewalk Overlap Deduplication Design

**Date:** 2026-05-04
**Branch:** feat/intersection-analysis
**Location in pipeline:** inside `step_09_match_separate_facilities`, after the match loop, before `_generate_offset_geometries`

---

## Problem

Some OSM contributors tag separately-mapped sidewalk paths with road highway values (e.g., `highway=residential`) instead of `highway=footway`. When OSMnx loads the network, these mistagged rows enter the GDF as road rows. Step 09 then treats them as road candidates, not as facilities to match. The result is duplicate street centerlines sitting on top of or immediately adjacent to the true road centerline — visible as overlapping red lines in the output map.

---

## Trigger Condition

A road row (highway in `_ROAD_HW`) is a mistagged sidewalk when **more than 80% of its length** falls within a **3 m buffer** of any footway row (highway in `_FOOTWAY_HW`).

---

## Algorithm

The detection is folded into the existing per-footway loop in `step_09_match_separate_facilities`. No separate step or second pass is added.

### Inside the footway loop

For each footway segment, after querying candidate roads within `_MATCH_RADIUS_M` (30 m):

1. Compute a **tight buffer** of the footway geometry at **3 m** in UTM.
2. For each candidate road, compute:
   ```
   coverage = road_geom.intersection(tight_buf).length / road_geom.length
   ```
3. Partition candidates:
   - `coverage > 0.80` → **coincident** (mistagged duplicate)
   - `coverage ≤ 0.80` → **true road candidate**
4. Run the existing best-road scoring (bearing parallelism + distance + name bonus) **only on true road candidates**. This ensures the coincident duplicate is never selected as the matching parent.
5. If `best_road` is found among true candidates:
   - Record every coincident candidate in `duplicate_to_centerline: dict[int, int]` mapping `dup_idx → best_road`.
   - Assign the footway geometry to `best_road`'s sidewalk slot as before.
6. If **no true road candidate** exists (all candidates are coincident, or no candidates at all): skip deduplication for this footway. Do not flag the coincident roads — there is no safe centerline to consolidate into.

### After the loop, before `_generate_offset_geometries`

1. **Attribute consolidation** — for each `(dup_idx, center_idx)` pair, copy sidewalk tags from the duplicate into the centerline using a **fill-null** strategy (only write if the centerline's slot is `None`/`pd.NA`). Columns consolidated:
   - `sidewalk_{side}_presence`
   - `sidewalk_{side}_surface`
   - `sidewalk_{side}_condition`
   - `sidewalk_{side}_width`
   - `sidewalk_{side}_incline`
   - `sidewalk_{side}_seperator`
   - `sidewalk_{side}_geometry` (if the dup somehow received geometry earlier in the loop)

   for both `side` in `("left", "right")`.

2. **Drop** — `gdf.drop(index=list(duplicate_to_centerline.keys()), inplace=False)`.

3. **Logging** — emit count of removed rows at `log.info` level, consistent with other step 09 log lines.

---

## Constants

| Name | Value | Rationale |
|---|---|---|
| `_DEDUP_COVERAGE_THRESHOLD` | `0.80` | Strict — 80% of the road must coincide before it is deleted |
| `_DEDUP_BUFFER_M` | `3.0` | Tight buffer — sidewalk paths are typically 0–2 m from their OSM centerline |

Both constants are module-level, placed near the existing step 09 constants (`_MATCH_RADIUS_M`, `_PARALLEL_THRESHOLD_DEG`).

---

## What Is Not Changed

- The `_generate_offset_geometries` call and all downstream steps are unchanged.
- Bikeway (cycleway) matching is unchanged — deduplication only applies to footway rows.
- No new pipeline step is added to `run_pipeline`.
