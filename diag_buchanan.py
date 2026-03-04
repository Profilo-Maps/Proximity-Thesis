"""Diagnostic script to investigate missing buffered sidewalks on Buchanan Street."""

import geopandas as gpd
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

PARQUET = r"c:\Dev\Proximity\Output\San_Francisco_County_California_USA_network.parquet"

gdf = gpd.read_parquet(PARQUET)

mask = gdf["name"].str.contains("Buchanan", case=False, na=False)
buchanan = gdf.loc[mask].copy()

print(f"{'='*80}")
print(f"BUCHANAN STREET DIAGNOSTIC")
print(f"{'='*80}\n")

for idx, row in buchanan.iterrows():
    print(f"--- Row index: {idx} ---")
    print(f"  name:              {row.get('name')}")
    print(f"  highway:           {row.get('highway')}")
    print(f"  street_id:         {row.get('street_id')}")
    print(f"  normalized_bearing:{row.get('normalized_bearing')}")

    lp = row.get("sidewalk_left_presence")
    rp = row.get("sidewalk_right_presence")
    print(f"  sidewalk_left_presence:  {lp}")
    print(f"  sidewalk_right_presence: {rp}")

    lg = row.get("sidewalk_left_geometry")
    rg = row.get("sidewalk_right_geometry")
    left_geom_populated = lg is not None and not (hasattr(lg, "is_empty") and lg.is_empty)
    right_geom_populated = rg is not None and not (hasattr(rg, "is_empty") and rg.is_empty)
    print(f"  sidewalk_left_geometry populated:  {left_geom_populated}")
    print(f"  sidewalk_right_geometry populated: {right_geom_populated}")

    lb = row.get("sidewalk_left_buffered")
    rb = row.get("sidewalk_right_buffered")
    print(f"  sidewalk_left_buffered:  {lb}")
    print(f"  sidewalk_right_buffered: {rb}")
    print()

# --- Summary ---
total = len(buchanan)
print(f"{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Total Buchanan segments: {total}")

if total == 0:
    print("No Buchanan segments found.")
else:
    has_left_presence = buchanan["sidewalk_left_presence"].notna().sum()
    has_right_presence = buchanan["sidewalk_right_presence"].notna().sum()
    print(f"Segments with left presence data:  {has_left_presence}/{total}")
    print(f"Segments with right presence data: {has_right_presence}/{total}")

    def geom_populated(series):
        return series.apply(
            lambda g: g is not None and not (hasattr(g, "is_empty") and g.is_empty)
        ).sum()

    has_left_geom = geom_populated(buchanan["sidewalk_left_geometry"])
    has_right_geom = geom_populated(buchanan["sidewalk_right_geometry"])
    print(f"Segments with left geometry:       {has_left_geom}/{total}")
    print(f"Segments with right geometry:      {has_right_geom}/{total}")

    has_left_buffered = (buchanan["sidewalk_left_buffered"] == True).sum()
    has_right_buffered = (buchanan["sidewalk_right_buffered"] == True).sum()
    print(f"Segments with left buffered=True:  {has_left_buffered}/{total}")
    print(f"Segments with right buffered=True: {has_right_buffered}/{total}")

    unique_left = buchanan["sidewalk_left_presence"].unique().tolist()
    unique_right = buchanan["sidewalk_right_presence"].unique().tolist()
    print(f"Unique left presence values:  {unique_left}")
    print(f"Unique right presence values: {unique_right}")
