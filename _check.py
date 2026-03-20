"""Check vertex counts in stored geometries."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString

gdf = gpd.read_parquet('Output/San_Francisco_County_California_USA_network.parquet')

counts = []
for g in gdf['street_geometry']:
    if g is not None and hasattr(g, 'coords'):
        try:
            counts.append(len(list(g.coords)))
        except:
            pass

counts = np.array(counts)
print('Street geometry vertex counts:')
print(f'  min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}, median={np.median(counts):.0f}')
print(f'  2-vertex (straight): {(counts == 2).sum()} / {len(counts)} ({100*(counts == 2).sum()/len(counts):.1f}%)')
print(f'  3+ vertices (curved): {(counts > 2).sum()} / {len(counts)} ({100*(counts > 2).sum()/len(counts):.1f}%)')

for col in ['sidewalk_left_geometry', 'sidewalk_right_geometry']:
    sw_counts = []
    for g in gdf[col]:
        if g is not None and not (hasattr(g, 'is_empty') and g.is_empty):
            try:
                if isinstance(g, MultiLineString):
                    sw_counts.append(sum(len(list(ls.coords)) for ls in g.geoms))
                elif hasattr(g, 'coords'):
                    sw_counts.append(len(list(g.coords)))
            except:
                pass
    if sw_counts:
        sw_counts = np.array(sw_counts)
        print(f'{col} vertex counts:')
        print(f'  min={sw_counts.min()}, max={sw_counts.max()}, mean={sw_counts.mean():.1f}')
        print(f'  2-vertex: {(sw_counts == 2).sum()} / {len(sw_counts)} ({100*(sw_counts == 2).sum()/len(sw_counts):.1f}%)')
