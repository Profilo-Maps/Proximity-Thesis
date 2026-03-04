"""
Diagnostic script: Investigate Adeline Street and Ellis Street in Berkeley
from the Alameda County parquet file.
"""

import geopandas as gpd
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)

PARQUET_PATH = r"c:\Dev\Proximity\Output\Alameda_County_California_USA_network.parquet"

print("Reading parquet...")
gdf = gpd.read_parquet(PARQUET_PATH)
print(f"Total rows: {len(gdf)}")
print(f"Columns: {list(gdf.columns)}\n")

# --- Identify relevant sidewalk columns ---
sw_cols = [c for c in gdf.columns if 'sidewalk' in c.lower()]
print(f"Sidewalk-related columns ({len(sw_cols)}):")
for c in sw_cols:
    print(f"  {c}")
print()

# --- Columns we want to inspect ---
inspect_cols = [
    'street_id', 'name', 'highway',
    'sidewalk_left_presence', 'sidewalk_right_presence',
    'sidewalk_left_buffered', 'sidewalk_right_buffered',
]
# Add geometry presence columns dynamically
geom_cols = ['sidewalk_left_geometry', 'sidewalk_right_geometry', 'street_geometry']
available_inspect = [c for c in inspect_cols if c in gdf.columns]
available_geom = [c for c in geom_cols if c in gdf.columns]

def print_street_info(df: gpd.GeoDataFrame, label: str):
    print(f"\n{'='*80}")
    print(f"  {label}  ({len(df)} rows)")
    print(f"{'='*80}")
    if len(df) == 0:
        print("  No rows found.")
        return

    for idx, row in df.iterrows():
        print(f"\n--- Row index {idx} ---")
        for col in available_inspect:
            print(f"  {col}: {row[col]}")
        # Geometry presence
        for gc in available_geom:
            val = row.get(gc, None)
            is_present = val is not None and (not hasattr(val, 'is_empty') or not val.is_empty)
            print(f"  {gc} present: {is_present}")
        # Check if sidewalk geometry differs from street geometry (i.e., separate sidewalk)
        if 'sidewalk_left_geometry' in gdf.columns and 'street_geometry' in gdf.columns:
            sw_l = row.get('sidewalk_left_geometry', None)
            st = row.get('street_geometry', None)
            if sw_l is not None and st is not None and not getattr(sw_l, 'is_empty', True):
                print(f"  sidewalk_left is separate from street: {not sw_l.equals(st)}")
            else:
                print(f"  sidewalk_left is separate from street: N/A (geometry missing)")
        if 'sidewalk_right_geometry' in gdf.columns and 'street_geometry' in gdf.columns:
            sw_r = row.get('sidewalk_right_geometry', None)
            st = row.get('street_geometry', None)
            if sw_r is not None and st is not None and not getattr(sw_r, 'is_empty', True):
                print(f"  sidewalk_right is separate from street: {not sw_r.equals(st)}")
            else:
                print(f"  sidewalk_right is separate from street: N/A (geometry missing)")
        print()


# --- Filter: Adeline Street, secondary, Berkeley area ---
# Berkeley is roughly lat 37.85-37.89, lon -122.30 to -122.25
adeline_mask = (
    gdf['name'].str.contains('Adeline', case=False, na=False)
    & (gdf['highway'] == 'secondary')
)
adeline = gdf[adeline_mask]
print_street_info(adeline.head(20), "Adeline Street (highway=secondary)")

# --- Filter: Ellis Street, residential ---
ellis_mask = (
    gdf['name'].str.contains('Ellis', case=False, na=False)
    & (gdf['highway'] == 'residential')
)
ellis = gdf[ellis_mask]
print_street_info(ellis.head(20), "Ellis Street (highway=residential)")

# --- Summary stats ---
print("\n" + "="*80)
print("  SUMMARY")
print("="*80)

for label, subset in [("Adeline (secondary)", adeline), ("Ellis (residential)", ellis)]:
    print(f"\n{label}: {len(subset)} segments")
    if len(subset) > 0:
        if 'sidewalk_left_presence' in subset.columns:
            print(f"  sidewalk_left_presence values: {subset['sidewalk_left_presence'].value_counts().to_dict()}")
        if 'sidewalk_right_presence' in subset.columns:
            print(f"  sidewalk_right_presence values: {subset['sidewalk_right_presence'].value_counts().to_dict()}")
        if 'sidewalk_left_buffered' in subset.columns:
            print(f"  sidewalk_left_buffered values: {subset['sidewalk_left_buffered'].value_counts().to_dict()}")
        if 'sidewalk_right_buffered' in subset.columns:
            print(f"  sidewalk_right_buffered values: {subset['sidewalk_right_buffered'].value_counts().to_dict()}")
        if 'sidewalk_left_geometry' in subset.columns:
            non_null = subset['sidewalk_left_geometry'].notna() & ~subset['sidewalk_left_geometry'].is_empty
            print(f"  sidewalk_left_geometry non-null/non-empty: {non_null.sum()}/{len(subset)}")
        if 'sidewalk_right_geometry' in subset.columns:
            non_null = subset['sidewalk_right_geometry'].notna() & ~subset['sidewalk_right_geometry'].is_empty
            print(f"  sidewalk_right_geometry non-null/non-empty: {non_null.sum()}/{len(subset)}")

print("\nDone.")
