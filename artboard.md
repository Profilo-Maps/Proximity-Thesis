
## The "Budget Premium" Material Strategy
    Base Board: Choose a Black Solder Mask (standard price).
    Network 1 (Streets): Exposed Copper (Silver/Gold)
        The Look: Shiny, metallic, and reflective.
        How to do it: Draw these in your software on the "Top Copper" layer AND the "Top Solder Mask" layer. The hole in the mask reveals the metal.
    Network 2 (Sidewalks): White Silkscreen
        The Look: Flat, matte white ink. This sits on top of everything else.
        How to do it: Draw these on the "Top Silk" layer.
    Network 3 (Bikelanes): Solder Mask "Ghosting"
        The Look: A subtle, textured "black-on-black" or "etched" look.
        How to do it: Draw these on the Top Copper layer ONLY. Because the black solder mask covers the copper, it creates a slightly raised, embossed effect that catches the light differently than the flat black background.




## Python Script: GIS to PCB Slicer

import geopandas as gpd
from shapely.geometry import box

# 1. Load Data
gdf = gpd.read_parquet('urban_data.parquet')

# 2. Rescale Coordinates to 0-600mm (for a 2x4 layout)
# We assume a 2-column, 4-row grid of 300mm boards = 600mm x 1200mm total
bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
width = bounds[2] - bounds[0]
height = bounds[3] - bounds[1]

# Scale factor to fit 1200mm (the 4ft side)
scale_factor = 1200 / max(width, height)

gdf['geometry'] = gdf['geometry'].translate(-bounds[0], -bounds[1]) # Move to 0,0
gdf['geometry'] = gdf['geometry'].scale(scale_factor, scale_factor, origin=(0,0))

# 3. Slice into 300mm x 300mm quadrants (8 total)
cols, rows = 2, 4
board_w, board_h = 300, 300

for c in range(cols):
    for r in range(rows):
        minx, miny = c * board_w, r * board_h
        maxx, maxy = minx + board_w, miny + board_h
        
        clipper = box(minx, miny, maxx, maxy)
        chunk = gdf.clip(clipper)
        
        # Shift back to (0,0) for the PCB software import
        chunk['geometry'] = chunk['geometry'].translate(-minx, -miny)
        
        if not chunk.empty:
            chunk.to_file(f"board_c{c}_r{r}.svg", driver='SVG')
            print(f"Saved Board Col {c}, Row {r}")
