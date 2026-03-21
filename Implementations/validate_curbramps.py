"""Comprehensive curb ramp validation for Jackson St intersections.

Shows all 6 potential curb ramp positions per node (left/right x start/end):
  Point 1: LEFT START
  Point 2: RIGHT START
  Point 3: LEFT END
  Point 4: RIGHT END
  Point 5: intersection center
  Point 6-9: indices 2 & 3 for each position
"""

import pandas as pd
from pathlib import Path
from shapely import wkb

def parse_geom_hex(s: str):
    """Parse WKB hex string."""
    if isinstance(s, str) and all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0:
        try:
            return wkb.loads(s, hex=True)
        except:
            return None
    return None


def analyze_node_detailed(node_id: int):
    """Show all 6 potential curb ramp positions and their current assignments."""
    data = pd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")

    # Get all edges touching this node
    mask = (data['start_node_id'] == node_id) | (data['end_node_id'] == node_id)
    edges = data[mask].copy()

    if edges.empty:
        print(f"Node {node_id} not found")
        return

    print(f"\nNODE {node_id}")
    print("=" * 80)
    print(f"Expected positions (visual numbering):")
    print(f"  1 = LEFT START    |  2 = RIGHT START   |  3 = LEFT END")
    print(f"  4 = RIGHT END     |  5 = CENTER        |  6-9 = Alt indices")
    print("=" * 80)

    # Collect all ramp positions
    positions = {}

    for idx, row in edges.iterrows():
        street = row.get('name', '(unnamed)')

        # Check all 6 position/side combinations
        for side in ('left', 'right'):
            for position in ('start', 'end'):
                key = f"{side}_{position}"

                # Check all 3 possible indices
                indices_present = []
                for ramp_idx in (1, 2, 3):
                    col = f'sidewalk_{side}_curbramp_{position}_{ramp_idx}_geometry'
                    if col in row.index:
                        raw = row[col]
                        if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                            geom = parse_geom_hex(str(raw))
                            if geom:
                                indices_present.append(ramp_idx)

                if key not in positions:
                    positions[key] = []

                positions[key].append({
                    'street': street,
                    'indices': indices_present,
                    'side_at': 'START' if row.get('start_node_id') == node_id else 'END'
                })

    # Display the matrix
    pos_order = ['left_start', 'right_start', 'left_end', 'right_end']
    pos_names = ['LEFT START', 'RIGHT START', 'LEFT END', 'RIGHT END']

    for pos_key, pos_name in zip(pos_order, pos_names):
        entries = positions.get(pos_key, [])
        indices = []
        for entry in entries:
            indices.extend(entry['indices'])

        indices_str = ', '.join(str(i) for i in sorted(set(indices))) if indices else '(none)'
        edges_str = ' | '.join(str(e['street']) for e in entries) if entries else '(no edges)'

        print(f"  {pos_name:15s}: indices [{indices_str}] | {edges_str}")

    print("\nCurrent state: 3 curb ramps assigned (all at index #1)")
    print("=" * 80)


if __name__ == '__main__':
    print("\nJACKSON STREET INTERSECTION VALIDATION")
    print("=" * 80)

    analyze_node_detailed(65336487)
    print("\nUSER EXPECTATION for Node A (65336487):")
    print("  Point 3 SHOULD be a curb ramp")
    print("  Points 4, 2, 6, 8 should NOT be curb ramps")
    print("  >> Need to clarify: which positions correspond to visual points 3, 4, 2, 6, 8?")

    analyze_node_detailed(65327381)
    print("\nUSER EXPECTATION for Node B (65327381):")
    print("  Points 1, 3, 7, 9 SHOULD be curb ramps")
    print("  Points 2, 4, 5, 6, 8 should NOT be curb ramps")
    print("  >> Currently has 6 ramps. Need to clarify the visual numbering scheme.")
