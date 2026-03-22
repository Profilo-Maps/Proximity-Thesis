import geopandas as gpd
import pandas as pd
import numpy as np

gdf = gpd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")
GRID_COL = 'street_grid_id'

# Show details of 6_8_79 (the service road at 0m distance)
seg79 = gdf[gdf[GRID_COL] == '6_8_79'].iloc[0]
seg498 = gdf[gdf[GRID_COL] == '6_8_498'].iloc[0]

print("6_8_79 (service road — nearest to 6_8_498):")
print(f"  name: {seg79['name']}")
print(f"  highway: {seg79['highway']}")
print(f"  bearing: {seg79['normalized_bearing']:.2f}")
geom79 = seg79['street_geometry']
coords79 = list(geom79.coords)
print(f"  coords: {[(round(c[0],2), round(c[1],2)) for c in coords79]}")
print(f"  sidewalk_left_presence: {seg79['sidewalk_left_presence']}")
print(f"  sidewalk_right_presence: {seg79['sidewalk_right_presence']}")

print()
print("6_8_498 (the unmatched footway):")
print(f"  bearing: {seg498['normalized_bearing']:.2f}")
geom498 = seg498['street_geometry']
coords498 = list(geom498.coords)
print(f"  coords: {[(round(c[0],2), round(c[1],2)) for c in coords498]}")

# 6_8_79 has bearing ~178 (N-S), 6_8_498 has bearing ~86 (E-W).
# They are PERPENDICULAR, sharing a node. 
# The footway gets matched to this perpendicular service road instead of
# the parallel Santiago Street road.

# Check if they share a node
start79 = seg79['start_node_id']
end79 = seg79['end_node_id']
start498 = seg498['start_node_id']
end498 = seg498['end_node_id']
print(f"\n6_8_79 nodes: start={start79}, end={end79}")
print(f"6_8_498 nodes: start={start498}, end={end498}")
shared = set([start79, end79]) & set([start498, end498])
print(f"Shared nodes: {shared}")

# Also check 6_8_234 (another service road at 0m)
print()
seg234 = gdf[gdf[GRID_COL] == '6_8_234'].iloc[0]
print("6_8_234 (service road — also at 0m distance):")
print(f"  name: {seg234['name']}")
print(f"  highway: {seg234['highway']}")
print(f"  bearing: {seg234['normalized_bearing']:.2f}")
geom234 = seg234['street_geometry']
coords234 = list(geom234.coords)
print(f"  coords: {[(round(c[0],2), round(c[1],2)) for c in coords234]}")
start234 = seg234['start_node_id']
end234 = seg234['end_node_id']
print(f"  nodes: start={start234}, end={end234}")
shared234 = set([start234, end234]) & set([start498, end498])
print(f"  Shared nodes with 6_8_498: {shared234}")

# So the unnamed footway 6_8_498 (bearing ~86, E-W along Santiago)
# gets matched to the nearest ROAD which is a perpendicular unnamed service road
# at 0m distance (sharing a node), not Santiago Street at ~9m distance.
# This is the root cause: the proximity fallback picks the geometrically
# nearest road regardless of bearing compatibility.

