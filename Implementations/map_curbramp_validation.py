"""Create a validation map showing current curb ramp assignments."""

import pandas as pd
import folium
from pathlib import Path
from pyproj import Transformer
from shapely import wkb

# Coordinate transformers
_to_wgs = Transformer.from_crs('EPSG:32610', 'EPSG:4326', always_xy=True)


def parse_geom_hex(s: str):
    """Parse WKB hex string to geometry."""
    if isinstance(s, str) and all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0:
        try:
            return wkb.loads(s, hex=True)
        except:
            return None
    return None


def utm_to_latlon(x, y):
    """Convert UTM to lat/lon."""
    lon, lat = _to_wgs.transform(x, y)
    return [lat, lon]


def create_validation_map(node_id: int, node_name: str, center_lat: float, center_lon: float):
    """Create a folium map showing current curb ramp assignments."""
    data = pd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")

    # Get all edges touching this node
    mask = (data['start_node_id'] == node_id) | (data['end_node_id'] == node_id)
    edges = data[mask].copy()

    if edges.empty:
        print(f"Node {node_id} not found")
        return

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=20,
        tiles='CartoDB positron'
    )

    # Collect all curb ramps
    ramp_count = 0
    ramp_locations = {}  # {(x, y): list of (street, side, position, idx)}

    for _, row in edges.iterrows():
        street = row.get('name', '(unnamed)')

        for side in ('left', 'right'):
            for position in ('start', 'end'):
                for idx in (1, 2, 3):
                    col = f'sidewalk_{side}_curbramp_{position}_{idx}_geometry'
                    if col in row.index:
                        raw = row[col]
                        if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                            geom = parse_geom_hex(str(raw))
                            if geom:
                                ramp_count += 1
                                coord_key = (round(geom.x, 2), round(geom.y, 2))
                                if coord_key not in ramp_locations:
                                    ramp_locations[coord_key] = []
                                ramp_locations[coord_key].append((street, side, position, idx))

    # Add ramp markers numbered by location
    for visual_num, (coord_key, entries) in enumerate(sorted(ramp_locations.items()), 1):
        utm_x, utm_y = coord_key
        lat, lon = utm_to_latlon(utm_x, utm_y)

        # Color based on number of edges using this ramp
        num_edges = len(entries)
        if num_edges == 2:
            color = 'green'  # Properly shared
            icon_num = f'<b style="color:white;font-size:16px">{visual_num}</b>'
        elif num_edges == 1:
            color = 'orange'  # Only in one edge
            icon_num = f'<b style="color:white;font-size:16px">{visual_num}!</b>'
        else:
            color = 'red'  # In too many edges
            icon_num = f'<b style="color:white;font-size:16px">{visual_num}X</b>'

        popup_text = f"<b>Curb Ramp #{visual_num}</b><br>"
        popup_text += f"In {num_edges} edge(s):<br>"
        for street, side, position, idx in entries:
            popup_text += f"  {street} {side} {position}#{idx}<br>"

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=color, icon_color='white', prefix='fa'),
            tooltip=f"Point {visual_num}",
        ).add_to(m)

        # Add numbered circle on top
        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            color='darkred',
            fill=True,
            fill_color='transparent',
            fill_opacity=0,
            weight=2,
            popup=popup_text,
        ).add_to(m)

    # Add center marker for the node
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=8,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.3,
        weight=2,
        popup=f"<b>Node {node_id}</b><br>{node_name}",
        tooltip=f"Intersection node {node_id}",
    ).add_to(m)

    # Add legend
    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 280px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px; border-radius: 5px;">
    <b>Node {node_name} ({node_id})</b><br>
    <b>Curb Ramp Validation</b><br><br>
    <i style="color:green">Green (2 edges)</i> = Properly shared<br>
    <i style="color:orange">Orange! (1 edge)</i> = Only in one edge<br>
    <i style="color:red">Red X (3+ edges)</i> = Too many edges<br><br>

    <b>Points are numbered 1-{ramp_count}</b><br>
    Compare to your annotated map.<br><br>
    <small>Blue circle = intersection node</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


if __name__ == '__main__':
    # Node A - Jackson & Battery
    print("Creating validation map for Node 65336487 (A)...")
    m_a = create_validation_map(
        65336487,
        "Jackson & Battery (A)",
        center_lat=37.796781,
        center_lon=-122.400702
    )
    m_a.save("Output/test_maps/validation_node_a.html")
    print("  -> Output/test_maps/validation_node_a.html")

    # Node B - Jackson & Front
    print("Creating validation map for Node 65327381 (B)...")
    m_b = create_validation_map(
        65327381,
        "Jackson & Front (B)",
        center_lat=37.796942,
        center_lon=-122.399515
    )
    m_b.save("Output/test_maps/validation_node_b.html")
    print("  -> Output/test_maps/validation_node_b.html")

    print("\nOpen these maps in your browser and compare to your annotated screenshots.")
    print("Tell me which point numbers match your visual points, and which ones should/shouldn't exist.")
