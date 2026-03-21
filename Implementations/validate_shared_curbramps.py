"""Validate that shared curb ramps are recorded in both touching edge segments.

A physical curb ramp at an intersection is used by two sidewalk segments.
Both segments should record this ramp in their appropriate slot.
"""

import pandas as pd
from shapely import wkb
from shapely.geometry import Point

def parse_geom_hex(s: str):
    """Parse WKB hex string to geometry."""
    if isinstance(s, str) and all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0:
        try:
            return wkb.loads(s, hex=True)
        except:
            return None
    return None


def get_curbramp_assignments(row: pd.Series) -> dict:
    """Extract all curb ramp assignments from an edge row.

    Returns: {(side, position, index): geometry} for all non-null ramps
    """
    assignments = {}

    for side in ('left', 'right'):
        for position in ('start', 'end'):
            for idx in (1, 2, 3):
                col = f'sidewalk_{side}_curbramp_{position}_{idx}_geometry'
                if col in row.index:
                    raw = row[col]
                    if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                        geom = parse_geom_hex(str(raw))
                        if geom:
                            assignments[(side, position, idx)] = geom

    return assignments


def validate_node_ramps(node_id: int, expected_ramps: dict) -> None:
    """Validate curb ramp assignments at a node.

    Args:
        node_id: OSM node ID
        expected_ramps: {point_name: (should_have_ramp, description)}
            e.g., {'point_3': (True, 'Shared by Jackson St segments')}
    """
    data = pd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")

    # Get all edges touching this node
    mask = (data['start_node_id'] == node_id) | (data['end_node_id'] == node_id)
    edges = data[mask].copy()

    if edges.empty:
        print(f"Node {node_id} not found")
        return

    print(f"\n{'='*80}")
    print(f"NODE {node_id} CURB RAMP VALIDATION")
    print(f"{'='*80}")
    print(f"Touching {len(edges)} edges\n")

    # Collect all curb ramps at this node
    all_ramps = {}  # {coords: list of (edge_name, side, position, idx)}

    for edge_idx, (_, row) in enumerate(edges.iterrows(), 1):
        street = row.get('name', '(unnamed)')
        assignments = get_curbramp_assignments(row)

        print(f"Edge {edge_idx}: {street}")
        if assignments:
            for (side, position, idx), geom in sorted(assignments.items()):
                coord_key = (round(geom.x, 4), round(geom.y, 4))
                if coord_key not in all_ramps:
                    all_ramps[coord_key] = []
                all_ramps[coord_key].append((street, side, position, idx))
                print(f"  - {side} {position} #{idx}")
        else:
            print(f"  (no curb ramps)")

    # Check for duplicates at same location (should be shared by 2 edges)
    print(f"\n{'-'*80}")
    print("RAMPS BY LOCATION (should see 2 edges per ramp):")
    print(f"{'-'*80}")

    for coord_key in sorted(all_ramps.keys()):
        entries = all_ramps[coord_key]
        print(f"\nLocation {coord_key}:")
        for street, side, position, idx in entries:
            print(f"  {street:20s} {side} {position} #{idx}")

        if len(entries) == 2:
            print(f"  [OK] SHARED by 2 edges (correct)")
        elif len(entries) == 1:
            print(f"  [!] Only in 1 edge (should be in both?)")
        else:
            print(f"  [ERROR] In {len(entries)} edges (unexpected)")

    # Validation summary
    print(f"\n{'-'*80}")
    print("VALIDATION AGAINST EXPECTATIONS:")
    print(f"{'-'*80}")

    num_ramps = len(all_ramps)
    print(f"Actual ramps found: {num_ramps}")
    print(f"Expected ramps: {sum(1 for should_have, _ in expected_ramps.values() if should_have)}")

    for point_name, (should_have, description) in expected_ramps.items():
        status = "[OK]" if (should_have and num_ramps > 0) or (not should_have and num_ramps == 0) else "[FAIL]"
        print(f"  {status} {point_name}: {description}")


if __name__ == '__main__':
    print("\nVALIDATING SHARED CURB RAMPS ON JACKSON STREET")

    # Node A validation
    validate_node_ramps(
        65336487,
        expected_ramps={
            'point_3': (True, 'Shared by Jackson St & Battery St segments'),
            'points_2_4_6_8': (False, 'Should NOT have ramps'),
        }
    )

    # Node B validation
    validate_node_ramps(
        65327381,
        expected_ramps={
            'points_1_3_7_9': (True, 'Should have ramps'),
            'points_2_4_5_6_8': (False, 'Should NOT have ramps'),
        }
    )
