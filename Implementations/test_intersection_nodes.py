"""Validation tests for specific intersection nodes on Jackson St, SF.

These tests verify that curb ramp assignments are correct at specific intersections.
The point numbers refer to the visual numbering on the map visualization.
"""

import pandas as pd
from pathlib import Path
from shapely.geometry.base import BaseGeometry
from shapely import wkb, wkt

def load_network() -> pd.DataFrame:
    """Load the SF network parquet."""
    parquet_path = Path("Output/San_Francisco_County_California_USA_network.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def parse_geom(val) -> BaseGeometry | None:
    """Parse geometry from various formats."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if hasattr(val, 'geom_type'):
            return val
        if isinstance(val, bytes):
            return wkb.loads(val)
        s = str(val).strip()
        if not s or s.lower() in ('none', 'nan', 'null'):
            return None
        # Try WKB hex first (common in parquets)
        if all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0:
            try:
                return wkb.loads(s, hex=True)
            except Exception:
                pass
        # Fall back to WKT
        return wkt.loads(s)
    except Exception:
        return None


def get_curbramp_cols(row: pd.Series, side: str, position: str) -> dict[int, tuple[bool, BaseGeometry | None]]:
    """Extract curb ramp presence for all indices at a given side/position.

    Returns: {index: (is_present, geom)} for indices 1, 2, 3
    """
    result = {}
    for idx in (1, 2, 3):
        col = f'sidewalk_{side}_curbramp_{position}_{idx}_geometry'
        raw_geom = row.get(col)
        geom = parse_geom(raw_geom)
        is_present = geom is not None
        result[idx] = (is_present, geom)
    return result


def analyze_node(data: pd.DataFrame, node_id: int) -> dict:
    """Analyze all curb ramps at a given intersection node ID.

    Returns a dict with the structure:
    {
        'start_node_is_int': bool,
        'end_node_is_int': bool,
        'left_start': {1: bool, 2: bool, 3: bool},
        'left_end': {1: bool, 2: bool, 3: bool},
        'right_start': {1: bool, 2: bool, 3: bool},
        'right_end': {1: bool, 2: bool, 3: bool},
        'edges': [list of edges touching this node]
    }
    """
    # Find all edges touching this node (either start or end)
    start_mask = data['start_node_id'] == node_id
    end_mask   = data['end_node_id'] == node_id

    touching = data[start_mask | end_mask].copy()
    if touching.empty:
        raise ValueError(f"Node {node_id} not found in network")

    results = {
        'node_id': node_id,
        'touching_edges': len(touching),
        'edges': []
    }

    all_ramps = []  # Collect all ramps with coordinates for sorting

    for idx, row in touching.iterrows():
        edge_info = {
            'street_name': row.get('name', '(unnamed)'),
            'at_start': bool(row.get('start_node_id') == node_id),
            'at_end': bool(row.get('end_node_id') == node_id),
            'left': {},
            'right': {}
        }

        # For start edges, check start position
        if edge_info['at_start']:
            edge_info['left']['start'] = get_curbramp_cols(row, 'left', 'start')
            edge_info['right']['start'] = get_curbramp_cols(row, 'right', 'start')

        # For end edges, check end position
        if edge_info['at_end']:
            edge_info['left']['end'] = get_curbramp_cols(row, 'left', 'end')
            edge_info['right']['end'] = get_curbramp_cols(row, 'right', 'end')

        results['edges'].append(edge_info)

        # Collect all ramps for sorting by coordinates
        for side in ('left', 'right'):
            for position in ('start', 'end'):
                if position not in edge_info[side]:
                    continue
                ramps = edge_info[side][position]
                for idx_num, (is_present, geom) in ramps.items():
                    if is_present and geom:
                        coords = geom.coords[0] if hasattr(geom, 'coords') else None
                        all_ramps.append({
                            'side': side,
                            'position': position,
                            'index': idx_num,
                            'edge': edge_info['street_name'],
                            'geom': geom,
                            'coords': coords
                        })

    results['all_ramps'] = sorted(all_ramps, key=lambda r: (r['coords'][1], r['coords'][0]) if r['coords'] else (0, 0))
    return results


def print_node_analysis(results: dict) -> None:
    """Pretty-print the curb ramp analysis for a node."""
    node_id = results['node_id']
    print(f"\n{'='*70}")
    print(f"Node {node_id}")
    print(f"Touching {results['touching_edges']} edges")
    print(f"{'='*70}")

    # Show all curb ramps sorted by coordinates
    all_ramps_list = results.get('all_ramps', [])
    if all_ramps_list:
        print(f"\nAll curb ramps at this node (sorted N->S, W->E):")
        print(f"{'-'*70}")
        for visual_idx, ramp in enumerate(all_ramps_list, 1):
            print(f"  {visual_idx}. {ramp['side'].upper()} {ramp['position'].upper():5s} #{ramp['index']} | "
                  f"{ramp['edge']} | {ramp['coords']}")
    else:
        print("\nNo curb ramps assigned at this node")

    # Also show breakdown by edge
    print(f"\n{'-'*70}")
    print("Breakdown by edge:")

    for edge_idx, edge in enumerate(results['edges']):
        street = edge['street_name']
        at = "START" if edge['at_start'] else "END" if edge['at_end'] else "?"
        print(f"\nEdge {edge_idx + 1}: {street} [{at}]")

        for side in ('left', 'right'):
            for position in ('start', 'end'):
                if position not in edge[side]:
                    continue
                indices = edge[side][position]
                present = [i for i, (p, _) in indices.items() if p]
                absent = [i for i, (p, _) in indices.items() if not p]

                status = f"Present: {present if present else 'none'} | Absent: {absent if absent else 'none'}"
                print(f"  {side.upper()} {position.upper():5s}: {status}")


def test_node_65336487():
    """Test node 65336487 (Jackson St, marked 'A' on map).

    Expected:
    - Point 3 should be a curb ramp
    - Points 4, 2, 6, 8 should NOT be curb ramps
    """
    print("\n" + "="*70)
    print("TEST NODE 65336487 (Jackson St - A)")
    print("="*70)
    print("Expected: Point 3 IS ramp, Points 2, 4, 6, 8 are NOT ramps")

    data = load_network()
    results = analyze_node(data, 65336487)
    print_node_analysis(results)

    # TODO: Add assertions once we understand the point numbering scheme


def test_node_65327381():
    """Test node 65327381 (Jackson St, marked 'B' on map).

    Expected:
    - Points 1, 3, 7, 9 should be curb ramps
    - Points 2, 4, 5, 6, 8 should NOT be curb ramps
    """
    print("\n" + "="*70)
    print("TEST NODE 65327381 (Jackson St - B)")
    print("="*70)
    print("Expected: Points 1, 3, 7, 9 ARE ramps, Points 2, 4, 5, 6, 8 are NOT ramps")

    data = load_network()
    results = analyze_node(data, 65327381)
    print_node_analysis(results)

    # TODO: Add assertions once we understand the point numbering scheme


if __name__ == '__main__':
    try:
        test_node_65336487()
        test_node_65327381()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
