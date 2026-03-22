"""
Validate incline data and create a simplified heatmap of topography.
Aggregates street inclines into a grid for fast rendering.
"""

import os
import pandas as pd
import folium
from folium import plugins
import numpy as np
from shapely import wkb
from pyproj import Transformer

# ── CONFIG ──────────────────────────────────────────────────────────────────
PARQUET_PATH = "Output/San_Francisco_County_California_USA_network.parquet"
OUTPUT_DIR = "Output/test_maps"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "incline_validation.html")

# Center on SF
CENTER_LAT = 37.7749
CENTER_LON = -122.4194
ZOOM = 12

# Grid resolution (degrees - smaller = finer grid)
GRID_SIZE = 0.01  # ~1km cells at this latitude

# Coordinate transformer (UTM Zone 10 -> WGS84)
TRANSFORMER = Transformer.from_crs('EPSG:32610', 'EPSG:4326', always_xy=True)

# ── HELPERS ─────────────────────────────────────────────────────────────────
def parse_incline(val) -> float | None:
    """Parse incline value to absolute float (percent)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return abs(float(val))
    s = str(val).strip().replace('%', '').replace('+', '')
    try:
        return abs(float(s))
    except ValueError:
        return None


def parse_geom(val):
    """Parse WKB hex string to shapely geometry."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if hasattr(val, 'x'):  # Already parsed
            return val
        s = str(val).strip()
        if not s:
            return None
        return wkb.loads(s, hex=True)
    except Exception:
        return None


def geom_to_latlon(geom):
    """Convert UTM geometry point to [lat, lon]."""
    if geom is None or not hasattr(geom, 'x'):
        return None
    lon, lat = TRANSFORMER.transform(geom.x, geom.y)
    return [lat, lon]


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows")

    # Parse incline
    print("Parsing incline data...")
    df['incline_val'] = df['street_incline'].apply(parse_incline)
    valid_inclines = df[df['incline_val'].notna()]

    print(f"\n=== INCLINE VALIDATION ===")
    print(f"Street incline coverage: {len(valid_inclines)} / {len(df)} ({100*len(valid_inclines)/len(df):.1f}%)")
    print(f"Range: {valid_inclines['incline_val'].min():.2f}% to {valid_inclines['incline_val'].max():.2f}%")
    print(f"Mean: {valid_inclines['incline_val'].mean():.2f}%")

    # Extract lat/lon from start nodes
    print("Extracting coordinates...")
    df['geom'] = df['start_node_geometry'].apply(parse_geom)
    df['latlon'] = df['geom'].apply(geom_to_latlon)
    df['lat'] = df['latlon'].apply(lambda x: x[0] if x else None)
    df['lon'] = df['latlon'].apply(lambda x: x[1] if x else None)

    # Filter to rows with both coords and incline
    valid = df[(df['lat'].notna()) & (df['lon'].notna()) & (df['incline_val'].notna())].copy()
    print(f"Valid rows with coords + incline: {len(valid)}")

    # Create grid
    print(f"Creating grid with {GRID_SIZE} degree cells...")
    valid['lat_bin'] = (valid['lat'] / GRID_SIZE).astype(int) * GRID_SIZE
    valid['lon_bin'] = (valid['lon'] / GRID_SIZE).astype(int) * GRID_SIZE

    # Aggregate by grid cell
    grid = valid.groupby(['lat_bin', 'lon_bin'])['incline_val'].agg(['mean', 'count']).reset_index()
    grid.columns = ['lat', 'lon', 'incline', 'count']
    grid = grid[grid['count'] >= 1]  # At least 1 point per cell

    print(f"Grid cells with data: {len(grid)}")

    # Create map
    print("Generating map...")
    m = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=ZOOM,
        tiles='CartoDB positron'
    )

    # Prepare heatmap data: [[lat, lon, incline], ...]
    heat_data = grid[['lat', 'lon', 'incline']].values.tolist()

    # Add heatmap layer
    plugins.HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=30,
        blur=25,
        gradient={
            0.0: '#0000ff',   # blue - flat
            0.25: '#00ffff',  # cyan
            0.5: '#ffff00',   # yellow - moderate
            0.75: '#ff8800',  # orange
            1.0: '#ff0000'    # red - steep
        }
    ).add_to(m)

    # Add legend
    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 280px; height: 200px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px;
                font-family: monospace;">
        <p style="margin: 0 0 8px 0;"><b>Street Incline Heatmap</b></p>
        <p style="margin: 0 0 12px 0; font-size: 11px; color: #666;">
            Aggregated to ~1km grid cells (Gaussian blur applied)
        </p>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 20px; height: 12px; background: #0000ff; margin-right: 8px;"></div>
            <span>Flat (0%)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 20px; height: 12px; background: #00ffff; margin-right: 8px;"></div>
            <span>Gentle (~7%)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 20px; height: 12px; background: #ffff00; margin-right: 8px;"></div>
            <span>Moderate (~14%)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 20px; height: 12px; background: #ff8800; margin-right: 8px;"></div>
            <span>Steep (~21%)</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 12px; background: #ff0000; margin-right: 8px;"></div>
            <span>Very steep (40%+)</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    m.save(OUTPUT_FILE)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
