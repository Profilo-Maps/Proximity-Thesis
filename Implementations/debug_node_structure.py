"""Debug: Show which edges touch each node and their sidewalk endpoints."""

import pandas as pd
from shapely import wkb
from pyproj import Transformer

_to_wgs = Transformer.from_crs('EPSG:32610', 'EPSG:4326', always_xy=True)

def parse_geom_bytes(raw):
    """Parse geometry from bytes."""
    if isinstance(raw, bytes):
        try:
            return wkb.loads(raw)
        except:
            return None
    return None

def utc_to_latlon(x, y):
    lon, lat = _to_wgs.transform(x, y)
    return lat, lon

data = pd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")

for node_id, label in [(65336487, "A"), (65327381, "B")]:
    print(f"\n{'='*80}")
    print(f"NODE {label} ({node_id}) - EDGE STRUCTURE")
    print(f"{'='*80}\n")

    mask = (data['start_node_id'] == node_id) | (data['end_node_id'] == node_id)
    edges = data[mask].copy()

    for edge_idx, (_, row) in enumerate(edges.iterrows(), 1):
        street = row.get('name', '(unnamed)')
        start_node = row.get('start_node_id')
        end_node = row.get('end_node_id')
        at_node = 'START' if start_node == node_id else 'END'

        print(f"Edge {edge_idx}: {street}")
        print(f"  Nodes: {start_node} -> {end_node} (touches at {at_node})")

        # Check sidewalks
        for side in ('left', 'right'):
            sw_col = f'sidewalk_{side}_geometry'
            if sw_col in row.index:
                sw_geom = parse_geom_bytes(row[sw_col])
                if sw_geom:
                    coords = list(sw_geom.coords)
                    start = coords[0]
                    end = coords[-1]

                    # Which endpoint is closest to the node?
                    lat_start, lon_start = utc_to_latlon(*start)
                    lat_end, lon_end = utc_to_latlon(*end)

                    print(f"  {side.upper()} sidewalk:")
                    print(f"    Start: ({lat_start:.6f}, {lon_start:.6f})")
                    print(f"    End:   ({lat_end:.6f}, {lon_end:.6f})")

                    # Show assigned curb ramps
                    for position in ('start', 'end'):
                        for idx in (1, 2, 3):
                            col = f'sidewalk_{side}_curbramp_{position}_{idx}_geometry'
                            if col in row.index:
                                raw = row[col]
                                if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                                    geom = parse_geom_bytes(raw)
                                    if geom:
                                        lat, lon = utc_to_latlon(geom.x, geom.y)
                                        print(f"    Curb ramp {position}#{idx}: ({lat:.6f}, {lon:.6f})")
                else:
                    print(f"  {side.upper()} sidewalk: (none)")
        print()
