"""Test _buffer_segment logic on specific Roosevelt Avenue rows."""
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import wkb
from shapely.geometry.base import BaseGeometry

OUTPUT_DIR = Path("Output")

path = OUTPUT_DIR / "Alameda_County_California_USA_network.parquet"
gdf = gpd.read_parquet(path)
for col in gdf.columns:
    if col.endswith("_geometry") and col != gdf.geometry.name:
        gdf[col] = pd.Series([wkb.loads(bytes.fromhex(h)) if isinstance(h, str) else h for h in gdf[col]], index=gdf.index)

mask = gdf["name"].astype(str).str.lower() == "roosevelt avenue"
rows = gdf[mask]

# Pick first candidate with "both" presence
cands = rows[rows["sidewalk_left_presence"].astype(str).str.lower() == "both"]
print(f"Roosevelt Ave candidates with 'both' presence: {len(cands)}")

for idx in cands.index[:3]:
    row = gdf.loc[idx]
    sg = row["street_geometry"]
    print(f"\n--- Row {idx} ---")
    print(f"  street_geometry type: {sg.geom_type if sg else 'None'}")
    print(f"  street_geometry length: {sg.length:.2f}m" if sg else "  N/A")
    print(f"  coords: {list(sg.coords)[:3]}..." if hasattr(sg, 'coords') else "  no coords")

    # Simulate _buffer_segment
    lanes = 2.0
    lane_width = 3.5
    half_road = (lanes * lane_width) / 2.0
    offset_m = half_road  # 3.5m for sidewalk without bikeways

    for side_name, sign in [("left", 1), ("right", -1)]:
        cur_offset = sign * offset_m
        try:
            if hasattr(sg, "offset_curve"):
                candidate = sg.offset_curve(cur_offset)
            else:
                candidate = sg.parallel_offset(abs(cur_offset), side=side_name)

            print(f"  {side_name} offset_curve({cur_offset:.1f}): type={candidate.geom_type}, "
                  f"empty={candidate.is_empty}, length={candidate.length:.2f}m")

            # Check collision with street_geometry
            if not candidate.is_empty and sg.intersects(candidate):
                endpoint_buffer = candidate.boundary.buffer(1e-6)
                intersection = sg.intersection(candidate)
                within_endpoints = intersection.within(endpoint_buffer)
                print(f"    Collision with street_geometry: intersects=True, "
                      f"within_endpoints={within_endpoints}")
                if not within_endpoints:
                    print(f"    *** COLLISION WOULD BLOCK BUFFERING ***")
                    print(f"    intersection type: {intersection.geom_type}, "
                          f"length: {intersection.length:.4f}")
            else:
                print(f"    No collision with street_geometry")

            # Check collision with other facility geometries
            for fcol in ["sidewalk_left_geometry", "sidewalk_right_geometry",
                         "bikeway_left_1_geometry", "bikeway_left_2_geometry",
                         "bikeway_right_1_geometry", "bikeway_right_2_geometry"]:
                if fcol not in gdf.columns:
                    continue
                other = gdf.at[idx, fcol]
                if not isinstance(other, BaseGeometry):
                    continue
                if not candidate.is_empty and other.intersects(candidate):
                    endpoint_buffer2 = candidate.boundary.buffer(1e-6)
                    ix = other.intersection(candidate)
                    within = ix.within(endpoint_buffer2)
                    print(f"    Collision with {fcol}: within_endpoints={within}")
                    if not within:
                        print(f"    *** {fcol} COLLISION WOULD BLOCK BUFFERING ***")

        except Exception as e:
            print(f"  {side_name} offset_curve({cur_offset:.1f}): EXCEPTION: {e}")
