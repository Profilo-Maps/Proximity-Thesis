import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 80)

gdf = gpd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")
GRID_COL = 'street_grid_id'

# Get segment 6_8_498
seg498 = gdf[gdf[GRID_COL] == '6_8_498'].iloc[0]
geom498 = seg498['street_geometry']
coords498 = list(geom498.coords)
bearing498 = seg498['normalized_bearing']

print("=" * 80)
print("SEGMENT 6_8_498 SUMMARY")
print("=" * 80)
print(f"  OSM ID: {seg498['street_id']}")
print(f"  highway: {seg498['highway']}")
print(f"  name: {seg498['name']} (is nan: {pd.isna(seg498['name'])})")
print(f"  bearing: {bearing498:.2f}")
print(f"  start coords: {coords498[0]}")
print(f"  end coords: {coords498[-1]}")
print(f"  start_node_is_intersection: {seg498['start_node_is_intersection_node']}")
print(f"  end_node_is_intersection: {seg498['end_node_is_intersection_node']}")
print()

# Find Santiago Street segments with 'separate' that have missing left geometry
print("=" * 80)
print("Santiago Street segments in 6_8 with 'separate' presence and MISSING geometry")
print("=" * 80)
santiago = gdf[
    (gdf['name'].astype(str).str.contains('Santiago', case=False, na=False)) &
    (gdf[GRID_COL].astype(str).str.startswith('6_8', na=False))
]

for idx, row in santiago.iterrows():
    left_sep = row['sidewalk_left_presence'] == 'separate'
    right_sep = row['sidewalk_right_presence'] == 'separate'
    
    left_geom_missing = True
    right_geom_missing = True
    
    left_geom = row.get('sidewalk_left_geometry', None)
    right_geom = row.get('sidewalk_right_geometry', None)
    
    if left_geom is not None and hasattr(left_geom, 'is_empty') and not left_geom.is_empty:
        left_geom_missing = False
    elif isinstance(left_geom, str):
        left_geom_missing = False
        
    if right_geom is not None and hasattr(right_geom, 'is_empty') and not right_geom.is_empty:
        right_geom_missing = False
    elif isinstance(right_geom, str):
        right_geom_missing = False
    
    if (left_sep and left_geom_missing) or (right_sep and right_geom_missing):
        road_geom = row['street_geometry']
        road_coords = list(road_geom.coords)
        # Distance from 498 to this road
        dist = geom498.distance(road_geom)
        
        print(f"\n  {GRID_COL}={row[GRID_COL]}, bearing={row['normalized_bearing']:.2f}")
        print(f"    left_presence={row['sidewalk_left_presence']}, left_geom_missing={left_geom_missing}")
        print(f"    right_presence={row['sidewalk_right_presence']}, right_geom_missing={right_geom_missing}")
        print(f"    left_buffered={row['sidewalk_left_buffered']}, right_buffered={row['sidewalk_right_buffered']}")
        print(f"    distance to 6_8_498: {dist:.2f} m")
        print(f"    road start: ({road_coords[0][0]:.2f}, {road_coords[0][1]:.2f})")
        print(f"    road end:   ({road_coords[-1][0]:.2f}, {road_coords[-1][1]:.2f})")

# Also check: are there other footways near 6_8_498 that DID get matched?
print()
print("=" * 80)
print("ALL footway segments in 6_8 grid - check which ones have sidewalk association")
print("=" * 80)
footways = gdf[
    (gdf['highway'] == 'footway') &
    (gdf[GRID_COL].astype(str).str.startswith('6_8_', na=False))
]
print(f"Total footway segments in 6_8: {len(footways)}")

# Check how many of these have sidewalk data set somewhere
# A footway that IS a separate sidewalk should appear as a sidewalk_*_geometry on a road
# Let's check if 6_8_498's geometry matches any sidewalk geometry on any road
print()
print("=" * 80)
print("CHECKING: Is 6_8_498 referenced as sidewalk geometry on ANY road segment?")
print("=" * 80)

# Check all rows where sidewalk_left_grid_ID or sidewalk_right_grid_ID references this
for col in ['sidewalk_left_grid_ID', 'sidewalk_right_grid_ID']:
    matches = gdf[gdf[col].astype(str) == '6_8_498']
    print(f"  {col} == '6_8_498': {len(matches)} rows")
    
# Also check if the footway's OSM ID appears anywhere
osm_id = seg498['street_id']
print(f"\n  Checking OSM ID {osm_id}...")
for col in gdf.columns:
    if 'public_data_id' in col or 'feature_id' in col:
        try:
            matches = gdf[gdf[col].astype(str).str.contains(str(osm_id), na=False)]
            if len(matches) > 0:
                print(f"  {col}: {len(matches)} matches")
        except Exception:
            pass

