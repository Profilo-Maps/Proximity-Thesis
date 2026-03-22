import geopandas as gpd
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 80)

gdf = gpd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")
GRID_COL = 'street_grid_id'

seg498 = gdf[gdf[GRID_COL] == '6_8_498'].iloc[0]
geom498 = seg498['street_geometry']

# 6_8_498 bearing is 85.92 and it runs from ~(546094, 4177628) to ~(546150, 4177632)
# The closest Santiago segments with missing separate sidewalk geometries:
# 6_8_67:  dist=9.16,  bearing=86.47, right_sep missing  (road: 546143->546094)
# 6_8_68:  dist=9.10,  bearing=86.46, left_sep missing   (road: 546143->546150)
# 6_8_72:  dist=9.10,  bearing=86.54, right_sep missing  (road: 546164->546150)
# 6_8_77:  dist=9.10,  bearing=86.46, right_sep missing  (road: 546150->546143)
# 6_8_78:  dist=9.10,  bearing=86.54, left_sep missing   (road: 546150->546164)
# 6_8_180: dist=9.16,  bearing=86.47, left_sep missing   (road: 546094->546143)
# 6_8_179: dist=9.63,  bearing=86.75, right_sep missing  (road: 546094->546084)

# So 6_8_498 spans from x=546094 to x=546150.
# 6_8_67 (Santiago) goes from x=546143 to x=546094 -- overlaps perfectly!
# 6_8_180 (Santiago) goes from x=546094 to x=546143 -- overlaps perfectly!
# Both are at ~9m distance with similar bearing.

# The key question: 6_8_498 is UNNAMED and is a footway.
# Santiago Street has highway=residential.
# Let's understand the name-matching logic.

# Let's look at what road segments DID get their separate sidewalks matched
print("=" * 80)
print("Santiago segments where separate sidewalk WAS successfully matched")
print("=" * 80)

santiago = gdf[
    (gdf['name'].astype(str).str.contains('Santiago', case=False, na=False)) &
    (gdf[GRID_COL].astype(str).str.startswith('6_8_', na=False))
]

for idx, row in santiago.iterrows():
    left_sep = row['sidewalk_left_presence'] == 'separate'
    right_sep = row['sidewalk_right_presence'] == 'separate'
    
    left_geom = row.get('sidewalk_left_geometry', None)
    right_geom = row.get('sidewalk_right_geometry', None)
    
    left_has = left_geom is not None and (isinstance(left_geom, str) or (hasattr(left_geom, 'is_empty') and not left_geom.is_empty))
    right_has = right_geom is not None and (isinstance(right_geom, str) or (hasattr(right_geom, 'is_empty') and not right_geom.is_empty))
    
    if (left_sep and left_has) or (right_sep and right_has):
        print(f"\n  {GRID_COL}={row[GRID_COL]}, bearing={row['normalized_bearing']:.2f}")
        if left_sep and left_has:
            print(f"    LEFT SEPARATE matched! buffered={row['sidewalk_left_buffered']}")
        if right_sep and right_has:
            print(f"    RIGHT SEPARATE matched! buffered={row['sidewalk_right_buffered']}")

# Now let's check: what footways ARE named (have names matching roads)?
print()
print("=" * 80)
print("Named footways in 6_8 grid (footway with a name value)")
print("=" * 80)
footways = gdf[
    (gdf['highway'] == 'footway') &
    (gdf[GRID_COL].astype(str).str.startswith('6_8_', na=False))
]
named_footways = footways[footways['name'].notna() & (footways['name'].astype(str) != 'nan')]
print(f"Total footways in 6_8: {len(footways)}")
print(f"Named footways in 6_8: {len(named_footways)}")
if len(named_footways) > 0:
    for idx, row in named_footways.head(20).iterrows():
        print(f"  {row[GRID_COL]}: name='{row['name']}', bearing={row['normalized_bearing']:.2f}")

# Check: what footways in 6_8 DID get matched as sidewalks (by checking 
# if their geometry appears as a sidewalk_*_geometry on a road)
# We can check indirectly: footways with no name that share similar coords with 
# a road's sidewalk geometry

print()
print("=" * 80)  
print("Footways near 6_8_498 (within 15m)")
print("=" * 80)
nearby = footways[footways['street_geometry'].distance(geom498) < 15]
for idx, row in nearby.iterrows():
    print(f"  {row[GRID_COL]}: name={row['name']}, bearing={row['normalized_bearing']:.2f}, "
          f"dist={row['street_geometry'].distance(geom498):.2f}m, "
          f"osm_id={row['street_id']}, "
          f"coords: {list(row['street_geometry'].coords)[0]} -> {list(row['street_geometry'].coords)[-1]}")

