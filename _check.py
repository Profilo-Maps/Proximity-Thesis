import geopandas as gpd
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 80)

gdf = gpd.read_parquet("Output/San_Francisco_County_California_USA_network.parquet")

GRID_COL = 'street_grid_id'
GEOM_COL = gdf.geometry.name  # active geometry column name

print(f"Active geometry column: {GEOM_COL}")
print(f"Total rows: {len(gdf)}")
print()

def print_sidewalk_info(row):
    core_sw = ['sidewalk_left_presence', 'sidewalk_right_presence',
               'sidewalk_left_buffered', 'sidewalk_right_buffered']
    for c in core_sw:
        print(f"  {c}: {row.get(c, 'N/A')}")
    for c in ['sidewalk_left_geometry', 'sidewalk_right_geometry']:
        val = row.get(c, None)
        if val is None or (hasattr(val, 'is_empty') and val.is_empty):
            print(f"  {c}: None/Empty")
        elif hasattr(val, 'geom_type'):
            print(f"  {c}: {val.geom_type} (present)")
        else:
            print(f"  {c}: {val}")
    for c in ['sidewalk_left_ID', 'sidewalk_right_ID',
              'sidewalk_left_grid_ID', 'sidewalk_right_grid_ID']:
        if c in gdf.columns:
            val = row.get(c, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                print(f"  {c}: {val}")

def print_row(idx, row):
    geom = row[GEOM_COL]
    print(f"\n--- Row index={idx}, {GRID_COL}={row[GRID_COL]} ---")
    print(f"  name: {row.get('name', 'N/A')}")
    print(f"  highway: {row.get('highway', 'N/A')}")
    if hasattr(geom, 'geom_type'):
        print(f"  geometry type: {geom.geom_type}")
    print_sidewalk_info(row)

# 1) Rows where grid_id contains "6_8_498"
print("=" * 80)
print("QUERY 1: street_grid_id contains '6_8_498'")
print("=" * 80)
mask1 = gdf[GRID_COL].astype(str).str.contains('6_8_498', na=False)
subset1 = gdf[mask1]
print(f"Found {len(subset1)} rows")
for idx, row in subset1.iterrows():
    print_row(idx, row)

# 2) Rows where grid_id contains "6_8_180"
print()
print("=" * 80)
print("QUERY 2: street_grid_id contains '6_8_180'")
print("=" * 80)
mask2 = gdf[GRID_COL].astype(str).str.contains('6_8_180', na=False)
subset2 = gdf[mask2]
print(f"Found {len(subset2)} rows")
for idx, row in subset2.iterrows():
    print_row(idx, row)

# 3) Rows where name contains "Santiago" and grid_id starts with "6_8"
print()
print("=" * 80)
print("QUERY 3: name contains 'Santiago' AND street_grid_id starts with '6_8'")
print("=" * 80)
mask3 = gdf['name'].astype(str).str.contains('Santiago', case=False, na=False) & gdf[GRID_COL].astype(str).str.startswith('6_8', na=False)
subset3 = gdf[mask3]
print(f"Found {len(subset3)} rows")
for idx, row in subset3.iterrows():
    print_row(idx, row)

# 4) Deep dive on 6_8_498
print()
print("=" * 80)
print("DEEP DIVE: segment 6_8_498")
print("=" * 80)
exact = gdf[gdf[GRID_COL].astype(str) == '6_8_498']
if len(exact) == 0:
    exact = gdf[gdf[GRID_COL].astype(str).str.contains('6_8_498', na=False)]
    print(f"(No exact match, using partial: {len(exact)} rows)")
else:
    print(f"Found {len(exact)} exact match rows")

for idx, row in exact.iterrows():
    print(f"\n--- Row index={idx}, {GRID_COL}={row[GRID_COL]} ---")
    name_val = row.get('name', None)
    has_name = name_val is not None and not (isinstance(name_val, float) and np.isnan(name_val)) and str(name_val) not in ('None', 'nan', '')
    print(f"  has name? {has_name}")
    print(f"  name value: '{name_val}'")
    print(f"  normalized_bearing: {row.get('normalized_bearing', 'N/A')}")
    
    geom = row[GEOM_COL]
    if hasattr(geom, 'geom_type'):
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            print(f"  geometry coords count: {len(coords)}")
            print(f"  start point (lon,lat): ({coords[0][0]:.7f}, {coords[0][1]:.7f})")
            print(f"  end point (lon,lat):   ({coords[-1][0]:.7f}, {coords[-1][1]:.7f})")
            if len(coords) <= 10:
                print(f"  all coords: {[(round(c[0],7), round(c[1],7)) for c in coords]}")
        elif geom.geom_type == 'Point':
            print(f"  point coords: ({geom.x:.7f}, {geom.y:.7f})")
    
    print(f"\n  ALL NON-NULL COLUMNS:")
    for c in gdf.columns:
        val = row[c]
        if hasattr(val, 'is_empty'):
            if val is not None and not val.is_empty:
                if val.geom_type == 'LineString':
                    cds = list(val.coords)
                    print(f"    {c}: {val.geom_type}, {len(cds)} coords")
                else:
                    print(f"    {c}: {val.geom_type}")
            # skip None/empty
        elif val is not None and not (isinstance(val, float) and np.isnan(val)):
            print(f"    {c}: {val}")

