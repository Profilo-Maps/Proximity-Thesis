import geopandas as gpd
import pandas as pd
import numpy as np

gdf = gpd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")
GRID_COL = 'street_grid_id'

seg498 = gdf[gdf[GRID_COL] == '6_8_498'].iloc[0]
geom498 = seg498['street_geometry']
mid498 = geom498.interpolate(0.5, normalized=True)

# The matching logic builds a spatial index of ROADS only (non-footway, non-bikeway).
# Let's simulate: find the nearest road to 6_8_498
is_footway = gdf['highway'].isin(['footway', 'pedestrian', 'path', 'steps'])
is_cycleway = gdf['highway'].isin(['cycleway'])
is_separate = is_footway | is_cycleway
roads = gdf[~is_separate.to_numpy(dtype=bool)].copy()

print(f"Total roads (non-separate): {len(roads)}")

# Find nearest road
from shapely import STRtree
road_geoms = roads['street_geometry'].values
road_tree = STRtree(road_geoms)
nearest_pos = road_tree.nearest(geom498)
nearest_road = roads.iloc[nearest_pos]
nearest_road_geom = nearest_road['street_geometry']
dist = nearest_road_geom.distance(geom498)

print(f"\nNearest road to 6_8_498:")
print(f"  grid_id: {nearest_road[GRID_COL]}")
print(f"  name: {nearest_road['name']}")
print(f"  highway: {nearest_road['highway']}")
print(f"  distance: {dist:.2f}m")
print(f"  bearing: {nearest_road['normalized_bearing']:.2f}")

# Also find roads within 30m (the NAME_MATCH_RADIUS_M)
search_area = geom498.buffer(30.0)
hit_positions = road_tree.query(search_area)
print(f"\nRoads within 30m of 6_8_498: {len(hit_positions)}")
for pos in hit_positions:
    r = roads.iloc[pos]
    d = r['street_geometry'].distance(geom498)
    print(f"  {r[GRID_COL]}: name='{r['name']}', highway={r['highway']}, dist={d:.2f}m, bearing={r['normalized_bearing']:.2f}")

# Now check: since 6_8_498 is UNNAMED, the code goes to the fallback path
# which calls _sindex_nearest_idx → returns nearest road.
# That should be the Santiago Street segment at ~9m.
# So WHY wasn't it matched?

# Let's check if the nearest road already has its sidewalk slot filled
print(f"\nChecking if nearest road's sidewalk slots are already filled:")
for side in ['left', 'right']:
    pres = nearest_road[f'sidewalk_{side}_presence']
    geom_val = nearest_road[f'sidewalk_{side}_geometry']
    has_geom = geom_val is not None and hasattr(geom_val, 'is_empty') and not geom_val.is_empty
    if isinstance(geom_val, str):
        has_geom = True
    print(f"  sidewalk_{side}: presence={pres}, has_geometry={has_geom}, buffered={nearest_road[f'sidewalk_{side}_buffered']}")

# Check what _road_side would return for 6_8_498 relative to its nearest road
from shapely.geometry import Point
road_line = nearest_road_geom
# Crude side determination: project footway midpoint onto road and check cross product
road_start = Point(list(road_line.coords)[0])
road_end = Point(list(road_line.coords)[-1])
road_dx = road_end.x - road_start.x
road_dy = road_end.y - road_start.y
mid_dx = mid498.x - road_start.x
mid_dy = mid498.y - road_start.y
cross = road_dx * mid_dy - road_dy * mid_dx
side = "right" if cross < 0 else "left"
print(f"\n  Side determination for 6_8_498 relative to nearest road: {side} (cross={cross:.2f})")

