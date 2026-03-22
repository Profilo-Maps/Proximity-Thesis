"""
Diagnose sidewalk network connectivity with iterative snap analysis.

Phase 1: Build initial connectivity graph (same-node endpoints within corner threshold).
Phase 2: Iterative bearing-aware snapping — starting with singletons, then smallest
         islands, find nearby endpoints from other islands via KD-tree. Only snap if
         parent segments have compatible bearings (prevents cross-street joins).
"""

import math
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely import wkb
from shapely.geometry import LineString, MultiLineString

# -- CONFIG --------------------------------------------------------------------
PARQUET = "Output/San_Francisco_County_California_USA_network.parquet"
CORNER_THRESHOLD_M = 15.0   # same-node endpoints closer than this = connected
SNAP_THRESHOLD_M = 20.0     # max distance for bearing-aware snapping
BEARING_TOLERANCE_DEG = 45.0  # max bearing difference for snap (0=parallel only)
BEARING_EXEMPT_M = 5.0       # endpoints closer than this skip bearing check (same corner)


SegId = tuple[int, str]   # (row_index, 'left'|'right')


def _bearing_from_coords(
    start: tuple[float, float], end: tuple[float, float]
) -> float:
    """Compute bearing (0-360°) from start to end in UTM coordinates."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.degrees(math.atan2(dx, dy)) % 360.0
    return angle


def _bearings_compatible(b1: float, b2: float, tolerance: float) -> bool:
    """Check if two bearings are roughly parallel (same or reverse direction).

    Two segments continuing along the same street will have bearings that are
    either similar (~0° apart) or opposite (~180° apart). Cross-street segments
    will be ~90° apart.
    """
    diff = abs(b1 - b2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    # Accept if nearly parallel (diff < tolerance) or nearly anti-parallel
    # (diff near 180°, meaning |diff - 180| < tolerance)
    return diff < tolerance or abs(diff - 180.0) < tolerance


def parse_segment(val: object) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Parse WKB hex to (start_xy, end_xy, bearing)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        s = str(val).strip()
        if not s:
            return None
        geom = wkb.loads(s, hex=True)
    except Exception:
        return None
    if isinstance(geom, LineString):
        coords = list(geom.coords)
    elif isinstance(geom, MultiLineString):
        coords = [c for line in geom.geoms for c in line.coords]
    else:
        return None
    if len(coords) < 2:
        return None
    start = (float(coords[0][0]), float(coords[0][1]))
    end = (float(coords[-1][0]), float(coords[-1][1]))
    bearing = _bearing_from_coords(start, end)
    return (start, end, bearing)


# -- Union-Find ----------------------------------------------------------------
class UnionFind:
    def __init__(self, elements: set[SegId]):
        self.parent: dict[SegId, SegId] = {e: e for e in elements}
        self.rank: dict[SegId, int] = {e: 0 for e in elements}

    def find(self, x: SegId) -> SegId:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: SegId, b: SegId) -> bool:
        """Merge sets of a and b. Returns True if they were in different sets."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def components(self) -> dict[SegId, set[SegId]]:
        groups: dict[SegId, set[SegId]] = defaultdict(set)
        for e in self.parent:
            groups[self.find(e)].add(e)
        return groups

    def same(self, a: SegId, b: SegId) -> bool:
        return self.find(a) == self.find(b)


def component_sizes(uf: UnionFind) -> list[int]:
    return sorted((len(c) for c in uf.components().values()), reverse=True)


def size_distribution(sizes: list[int]) -> str:
    lines = []
    lines.append(f"  Singletons (1):   {sum(1 for s in sizes if s == 1):>6,}")
    lines.append(f"  Small (2-5):      {sum(1 for s in sizes if 2 <= s <= 5):>6,}")
    lines.append(f"  Medium (6-20):    {sum(1 for s in sizes if 6 <= s <= 20):>6,}")
    lines.append(f"  Large (21-100):   {sum(1 for s in sizes if 21 <= s <= 100):>6,}")
    lines.append(f"  Very large (>100):{sum(1 for s in sizes if s > 100):>6,}")
    return '\n'.join(lines)


def main() -> None:
    t0 = time.perf_counter()
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

    # -- 1. Parse sidewalk endpoints + bearings --------------------------------
    # seg_data: seg_id -> (start_node, end_node, start_xy, end_xy, bearing)
    seg_data: dict[SegId, tuple[int, int, tuple[float, float], tuple[float, float], float]] = {}
    presence_no_geom: list[tuple[int, str, str | None]] = []

    for side in ('left', 'right'):
        geom_col = f'sidewalk_{side}_geometry'
        pres_col = f'sidewalk_{side}_presence'
        has_geom = df[geom_col].notna()
        has_presence = df[pres_col].notna() & ~df[pres_col].isin(['no', 'none', ''])

        missing_geom_mask = has_presence & ~has_geom
        for idx in df[missing_geom_mask].index:
            raw_name = df.at[idx, 'name']
            street_name = str(raw_name) if pd.notna(raw_name) else None
            presence_no_geom.append((idx, side, street_name))

        print(f"Parsing {side} sidewalk endpoints ({has_geom.sum():,} segments)...")
        for idx in df[has_geom].index:
            result = parse_segment(df.at[idx, geom_col])
            if result is None:
                continue
            start_xy, end_xy, bearing = result
            seg_id: SegId = (idx, side)
            start_node = int(str(df.at[idx, 'start_node_id']))
            end_node = int(str(df.at[idx, 'end_node_id']))
            seg_data[seg_id] = (start_node, end_node, start_xy, end_xy, bearing)

    all_seg_ids = set(seg_data.keys())
    print(f"\nTotal sidewalk segments with geometry: {len(all_seg_ids):,}")
    print(f"Sidewalk segments with presence but NO geometry: {len(presence_no_geom):,}")

    # -- 2. Initial connectivity: same-node, within corner threshold ----------
    print(f"\n{'-'*60}")
    print(f"PHASE 1: Initial same-node connectivity (threshold={CORNER_THRESHOLD_M:.0f}m)")
    print(f"{'-'*60}")

    node_endpoints: dict[int, list[tuple[SegId, tuple[float, float]]]] = defaultdict(list)
    for seg_id, (sn, en, sxy, exy, _brg) in seg_data.items():
        node_endpoints[sn].append((seg_id, sxy))
        node_endpoints[en].append((seg_id, exy))

    uf = UnionFind(all_seg_ids)
    initial_edges = 0
    for _node_id, endpoints in node_endpoints.items():
        n = len(endpoints)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                sid_a, xy_a = endpoints[i]
                sid_b, xy_b = endpoints[j]
                if sid_a == sid_b:
                    continue
                dist = np.sqrt((xy_a[0] - xy_b[0]) ** 2 + (xy_a[1] - xy_b[1]) ** 2)
                if dist < CORNER_THRESHOLD_M:
                    if uf.union(sid_a, sid_b):
                        initial_edges += 1

    sizes = component_sizes(uf)
    print(f"Edges added: {initial_edges:,}")
    print(f"Islands: {len(sizes):,}")
    print(f"Largest: {sizes[0]:,} ({100 * sizes[0] / len(all_seg_ids):.1f}%)")
    print(size_distribution(sizes))

    # -- 3. Build KD-tree from ALL endpoints ----------------------------------
    print(f"\n{'-'*60}")
    print(f"PHASE 2: Bearing-aware iterative snap (threshold={SNAP_THRESHOLD_M:.0f}m, "
          f"bearing tolerance={BEARING_TOLERANCE_DEG:.0f}°)")
    print(f"{'-'*60}")

    # Each endpoint: (seg_id, 'start'|'end')
    endpoint_list: list[tuple[SegId, str]] = []
    endpoint_coords: list[tuple[float, float]] = []
    for seg_id, (sn, en, sxy, exy, _brg) in seg_data.items():
        endpoint_list.append((seg_id, 'start'))
        endpoint_coords.append(sxy)
        endpoint_list.append((seg_id, 'end'))
        endpoint_coords.append(exy)

    coords_arr = np.array(endpoint_coords)
    print(f"Building KD-tree from {len(coords_arr):,} endpoints...")
    tree = KDTree(coords_arr)

    print(f"Querying neighbors within {SNAP_THRESHOLD_M:.0f}m...")
    neighbor_lists = tree.query_ball_point(coords_arr, SNAP_THRESHOLD_M)

    # Build neighbor map with distances (filter self)
    endpoint_neighbors: dict[int, list[tuple[int, float]]] = {}
    for ep_idx in range(len(endpoint_list)):
        seg_id_a = endpoint_list[ep_idx][0]
        neighbors = []
        for nb_idx in neighbor_lists[ep_idx]:
            seg_id_b = endpoint_list[nb_idx][0]
            if seg_id_b == seg_id_a:
                continue
            dx = coords_arr[ep_idx, 0] - coords_arr[nb_idx, 0]
            dy = coords_arr[ep_idx, 1] - coords_arr[nb_idx, 1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            neighbors.append((nb_idx, dist))
        if neighbors:
            endpoint_neighbors[ep_idx] = neighbors

    # Map seg_id → its endpoint indices
    seg_to_ep: dict[SegId, list[int]] = defaultdict(list)
    for ep_idx, (seg_id, _) in enumerate(endpoint_list):
        seg_to_ep[seg_id].append(ep_idx)

    # -- 4. Iterative bearing-aware snapping ----------------------------------
    total_snaps = 0
    total_bearing_rejects = 0
    round_num = 0

    while True:
        round_num += 1
        comps = uf.components()
        sorted_comps = sorted(comps.values(), key=len)

        snaps_this_round = 0
        bearing_rejects_this_round = 0
        snap_details: list[tuple[SegId, SegId, float]] = []

        for comp in sorted_comps:
            for seg_id in comp:
                bearing_a = seg_data[seg_id][4]
                for ep_idx in seg_to_ep[seg_id]:
                    if ep_idx not in endpoint_neighbors:
                        continue
                    best_nb_idx: int | None = None
                    best_dist = float('inf')
                    for nb_idx, dist in endpoint_neighbors[ep_idx]:
                        nb_seg_id = endpoint_list[nb_idx][0]
                        if uf.same(seg_id, nb_seg_id):
                            continue
                        # Bearing check: only for endpoints >BEARING_EXEMPT_M apart.
                        # Closer endpoints are on the same corner (intersection) and
                        # should always connect regardless of bearing difference.
                        if dist > BEARING_EXEMPT_M:
                            bearing_b = seg_data[nb_seg_id][4]
                            if not _bearings_compatible(bearing_a, bearing_b, BEARING_TOLERANCE_DEG):
                                bearing_rejects_this_round += 1
                                continue
                        if dist < best_dist:
                            best_dist = dist
                            best_nb_idx = nb_idx

                    if best_nb_idx is not None:
                        nb_seg_id = endpoint_list[best_nb_idx][0]
                        if uf.union(seg_id, nb_seg_id):
                            snaps_this_round += 1
                            snap_details.append((seg_id, nb_seg_id, best_dist))

        total_snaps += snaps_this_round
        total_bearing_rejects += bearing_rejects_this_round
        sizes = component_sizes(uf)

        print(f"\n  Round {round_num}: {snaps_this_round:,} snaps, "
              f"{bearing_rejects_this_round:,} bearing-rejected")
        print(f"    Islands: {len(sizes):,}")
        print(f"    Largest: {sizes[0]:,} ({100 * sizes[0] / len(all_seg_ids):.1f}%)")
        print(size_distribution(sizes))

        if snaps_this_round > 0:
            dists = [d for _, _, d in snap_details]
            print(f"    Snap distances: min={min(dists):.1f}m, "
                  f"median={sorted(dists)[len(dists)//2]:.1f}m, "
                  f"max={max(dists):.1f}m")

        if snaps_this_round == 0:
            break

    # -- 5. Final report ------------------------------------------------------
    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"FINAL CONNECTIVITY REPORT")
    print(f"{'='*60}")
    sizes = component_sizes(uf)
    print(f"Total segments:     {len(all_seg_ids):,}")
    print(f"Total snaps:        {total_snaps:,}")
    print(f"Bearing rejects:    {total_bearing_rejects:,}")
    print(f"Rounds:             {round_num}")
    print(f"Final islands:      {len(sizes):,}")
    print(f"Largest island:     {sizes[0]:,} ({100 * sizes[0] / len(all_seg_ids):.1f}%)")
    if len(sizes) > 1:
        print(f"2nd largest:        {sizes[1]:,}")
    print(f"\n{size_distribution(sizes)}")

    # Remaining singletons analysis
    remaining_singletons: list[SegId] = []
    for comp in uf.components().values():
        if len(comp) == 1:
            remaining_singletons.append(next(iter(comp)))

    if remaining_singletons:
        print(f"\n{'-'*60}")
        print(f"REMAINING SINGLETONS: {len(remaining_singletons):,}")
        print(f"{'-'*60}")

        # For each remaining singleton, find nearest bearing-compatible neighbor
        min_dists_any: list[float] = []
        min_dists_compatible: list[float] = []
        for seg_id in remaining_singletons:
            bearing_a = seg_data[seg_id][4]
            best_any = float('inf')
            best_compatible = float('inf')
            for ep_idx in seg_to_ep[seg_id]:
                dists_arr, idxs_arr = tree.query(coords_arr[ep_idx], k=20)
                for d, nb_idx in zip(dists_arr, idxs_arr, strict=False):
                    nb_seg_id = endpoint_list[nb_idx][0]
                    if nb_seg_id == seg_id:
                        continue
                    best_any = min(best_any, float(d))
                    bearing_b = seg_data[nb_seg_id][4]
                    if _bearings_compatible(bearing_a, bearing_b, BEARING_TOLERANCE_DEG):
                        best_compatible = min(best_compatible, float(d))
                    break  # only need closest non-self for "any"
                # Also scan for closest bearing-compatible specifically
                for d, nb_idx in zip(dists_arr, idxs_arr, strict=False):
                    nb_seg_id = endpoint_list[nb_idx][0]
                    if nb_seg_id == seg_id:
                        continue
                    bearing_b = seg_data[nb_seg_id][4]
                    if _bearings_compatible(bearing_a, bearing_b, BEARING_TOLERANCE_DEG):
                        best_compatible = min(best_compatible, float(d))
                        break
            min_dists_any.append(best_any)
            min_dists_compatible.append(best_compatible)

        any_arr = np.array(min_dists_any)
        compat_arr = np.array(min_dists_compatible)
        print(f"\nNearest neighbor distances:")
        print(f"  {'Threshold':>10}  {'Any':>12}  {'Bearing-OK':>12}")
        for threshold in [10, 15, 20, 25, 30, 50, 100, 500]:
            c_any = int(np.sum(any_arr <= threshold))
            c_compat = int(np.sum(compat_arr <= threshold))
            print(f"  <={threshold:>4}m:    {c_any:>5,} ({100*c_any/len(min_dists_any):>4.0f}%)  "
                  f"  {c_compat:>5,} ({100*c_compat/len(min_dists_compatible):>4.0f}%)")

        truly_isolated = int(np.sum(any_arr > 500))
        print(f"  >500m:      {truly_isolated:>5,} ({100*truly_isolated/len(min_dists_any):>4.0f}%)")

        # Show close-but-unsnapped singletons
        close_ones = [(seg_id, d_any, d_compat)
                      for seg_id, d_any, d_compat in zip(remaining_singletons, min_dists_any, min_dists_compatible)
                      if d_any <= 50]
        close_ones.sort(key=lambda x: x[1])
        if close_ones:
            print(f"\n  Closest unsnapped singletons (within 50m):")
            print(f"  {'dist_any':>8}  {'dist_ok':>8}  {'side':>5}  {'grid_id':>16}  name")
            for seg_id, d_any, d_compat in close_ones[:25]:
                row_idx, side = seg_id
                name = df.at[row_idx, 'name'] if pd.notna(df.at[row_idx, 'name']) else '(unnamed)'
                grid_id = df.at[row_idx, 'street_grid_id']
                d_compat_str = f"{d_compat:.1f}m" if d_compat < 1e6 else "none"
                print(f"  {d_any:>7.1f}m  {d_compat_str:>8}  {side:>5}  {grid_id:>16}  {name}")

    # Presence-but-no-geometry
    if presence_no_geom:
        print(f"\n{'-'*60}")
        print(f"PRESENCE BUT NO GEOMETRY: {len(presence_no_geom):,}")
        print(f"{'-'*60}")
        name_counts = Counter(p[2] for p in presence_no_geom if p[2] is not None)
        unnamed_count = sum(1 for p in presence_no_geom if p[2] is None)
        print(f"  Named streets: {sum(name_counts.values()):,} across {len(name_counts):,} streets")
        print(f"  Unnamed:       {unnamed_count:,}")
        print(f"  Top 15 streets:")
        for name, count in name_counts.most_common(15):
            print(f"    {count:>4}  {name}")

    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
