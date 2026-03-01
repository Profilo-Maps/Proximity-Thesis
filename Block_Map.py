import os
import json
import math
import hashlib
import pandas as pd
import folium
from shapely import wkb as shapely_wkb
from shapely import wkt as shapely_wkt
from shapely.geometry import shape
from pyproj import Transformer
from collections import defaultdict

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(SCRIPT_DIR, "Output",
                            "San_Francisco_County_California_USA_network.parquet")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "Output", "test_maps")
OUTPUT_NAME  = "block_map"

CENTER_LAT = 37 + 46/60 + 16.6/3600
CENTER_LON = -(122 + 25/60 + 27.1/3600)
ZOOM       = 14

PALETTE = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
]

# ── CRS TRANSFORMERS ───────────────────────────────────────────────────────────
_to_wgs = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)

def _utm_to_latlon(x, y):
    lon, lat = _to_wgs.transform(x, y)
    return [lat, lon]

# ── GEOMETRY PARSER ────────────────────────────────────────────────────────────
def parse_geom(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if hasattr(val, "geom_type"):
            return val
        if isinstance(val, bytes):
            return shapely_wkb.loads(val)
        if isinstance(val, dict):
            return shape(val)
        s = str(val).strip()
        if not s or s.lower() in ("none", "nan", "null"):
            return None
        if s.startswith("{"):
            return shape(json.loads(s))
        if all(c in "0123456789abcdefABCDEF" for c in s) and len(s) % 2 == 0 and len(s) >= 2:
            return shapely_wkb.loads(s, hex=True)
        return shapely_wkt.loads(s)
    except Exception:
        return None

def geom_to_latlons(geom):
    if geom is None:
        return []
    if geom.geom_type == "LineString":
        return [_utm_to_latlon(c[0], c[1]) for c in geom.coords]
    if geom.geom_type == "MultiLineString":
        coords = []
        for line in geom.geoms:
            coords.extend([_utm_to_latlon(c[0], c[1]) for c in line.coords])
        return coords
    return []

# ── LOAD PARQUET ───────────────────────────────────────────────────────────────
print(f"Loading parquet: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} rows")

if df.empty:
    raise ValueError(f"Parquet is empty: {PARQUET_PATH}")

# ── PARSE block_ids COLUMN ─────────────────────────────────────────────────────
def extract_block_ids(val):
    """Return (left_id, right_id) strings or None from the block_ids struct/str."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None, None
    if isinstance(val, dict):
        return val.get("left"), val.get("right")
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null", "{}"):
        return None, None
    # Try JSON-like parsing after normalising single-quoted Python dicts
    try:
        s_json = s.replace("'", '"').replace("None", "null")
        d = json.loads(s_json)
        if isinstance(d, dict):
            return d.get("left"), d.get("right")
    except Exception:
        pass
    return None, None

# ── BUILD ADJACENCY GRAPH & COLLECT BLOCK DATA ─────────────────────────────────
# block_adjacency: block_id -> set of adjacent block_ids
block_adjacency = defaultdict(set)
# block_segments: block_id -> list of row indices (for label placement)
block_segments  = defaultdict(list)

rows_list = []
for idx, row in df.iterrows():
    left_id, right_id = extract_block_ids(row.get("block_ids"))
    rows_list.append((idx, left_id, right_id))
    if left_id is not None:
        block_segments[left_id].append(idx)
    if right_id is not None:
        block_segments[right_id].append(idx)
    if left_id is not None and right_id is not None:
        block_adjacency[left_id].add(right_id)
        block_adjacency[right_id].add(left_id)

all_blocks = set(block_segments.keys())
print(f"Unique block IDs: {len(all_blocks)}")

# Ensure every block appears in adjacency (even isolated ones)
for b in all_blocks:
    if b not in block_adjacency:
        block_adjacency[b] = set()

# ── GREEDY GRAPH COLORING (sort by descending degree) ─────────────────────────
sorted_blocks = sorted(all_blocks,
                       key=lambda b: len(block_adjacency[b]),
                       reverse=True)

block_color = {}

def _hash_color(block_id):
    h = hashlib.md5(str(block_id).encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"#{r:02x}{g:02x}{b:02x}"

for block_id in sorted_blocks:
    neighbor_colors = {block_color[nb] for nb in block_adjacency[block_id]
                       if nb in block_color}
    assigned = None
    for color in PALETTE:
        if color not in neighbor_colors:
            assigned = color
            break
    if assigned is None:
        assigned = _hash_color(block_id)
    block_color[block_id] = assigned

colors_used = set(block_color.values())
palette_colors_used = colors_used & set(PALETTE)
fallback_colors_used = colors_used - set(PALETTE)
print(f"Colors used: {len(colors_used)} "
      f"({len(palette_colors_used)} palette, {len(fallback_colors_used)} fallback)")

# ── BUILD FOLIUM MAP ───────────────────────────────────────────────────────────
m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=ZOOM,
               tiles="CartoDB positron")

fg_segments = folium.FeatureGroup(name="Block segments",  show=True)
fg_labels   = folium.FeatureGroup(name="Block ID labels", show=True)

segment_count = 0

# For label placement: track first-occurrence midpoint per block
block_label_placed = {}  # block_id -> True once placed

def _midpoint_latlon(geom):
    """Return the UTM midpoint of a LineString/MultiLineString, then convert."""
    if geom is None:
        return None
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
    elif geom.geom_type == "MultiLineString":
        coords = []
        for line in geom.geoms:
            coords.extend(list(line.coords))
    else:
        return None
    if not coords:
        return None
    mid_idx = len(coords) // 2
    x, y = coords[mid_idx][0], coords[mid_idx][1]
    return x, y

def _offset_utm(x, y, bearing_deg, offset_m=0.5):
    """Offset a UTM point perpendicular (left) by offset_m metres."""
    perp_deg = (bearing_deg - 90.0) % 360.0
    rad = math.radians(perp_deg)
    dx = math.sin(rad) * offset_m
    dy = math.cos(rad) * offset_m
    return x + dx, y + dy

for idx, left_id, right_id in rows_list:
    row = df.iloc[idx] if isinstance(idx, int) else df.loc[idx]

    street_geom = parse_geom(row.get("street_geometry"))
    if street_geom is None:
        continue

    coords = geom_to_latlons(street_geom)
    if len(coords) < 2:
        continue

    seg_color = block_color.get(left_id, "#888888") if left_id is not None else "#888888"
    name = str(row.get("name") or "").strip()
    bearing = row.get("normalized_bearing")
    bearing_str = f"{bearing:.1f}°" if bearing is not None and not (
        isinstance(bearing, float) and math.isnan(bearing)) else "—"

    tooltip_text = (
        f"Left: {left_id} | Right: {right_id} | {name or '(unnamed)'}"
    )

    folium.PolyLine(
        coords,
        color=seg_color,
        weight=4,
        opacity=0.80,
        tooltip=tooltip_text,
    ).add_to(fg_segments)
    segment_count += 1

    # ── Label: one CircleMarker per block_id (left side) ──────────────────
    for block_id, is_left in ((left_id, True), (right_id, False)):
        if block_id is None:
            continue
        if block_id in block_label_placed:
            continue
        block_label_placed[block_id] = True

        mid_utm = _midpoint_latlon(street_geom)
        if mid_utm is None:
            continue
        mx, my = mid_utm

        # Offset slightly to left or right of segment
        if bearing is not None and not (isinstance(bearing, float) and math.isnan(bearing)):
            try:
                off_bearing = float(bearing)
                # left side: perpendicular left; right side: perpendicular right
                perp = (off_bearing - 90.0) % 360.0 if is_left else (off_bearing + 90.0) % 360.0
                rad = math.radians(perp)
                ox = mx + math.sin(rad) * 0.8
                oy = my + math.cos(rad) * 0.8
                label_latlon = _utm_to_latlon(ox, oy)
            except (TypeError, ValueError):
                label_latlon = _utm_to_latlon(mx, my)
        else:
            label_latlon = _utm_to_latlon(mx, my)

        label_color = block_color.get(block_id, "#888888")
        side_str = "left" if is_left else "right"

        popup_html = (
            f'<div style="font-family:sans-serif;font-size:12px;min-width:160px">'
            f'<b>Block ID:</b> {block_id}<br>'
            f'<b>Side:</b> {side_str}<br>'
            f'<b>Street:</b> {name or "(unnamed)"}<br>'
            f'<b>Bearing:</b> {bearing_str}'
            f'</div>'
        )

        folium.CircleMarker(
            location=label_latlon,
            radius=5,
            color=label_color,
            fill=True,
            fill_color=label_color,
            fill_opacity=0.9,
            weight=1.5,
            tooltip=f"Block: {block_id}",
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(fg_labels)

fg_segments.add_to(m)
fg_labels.add_to(m)

# ── CENTER MARKER ──────────────────────────────────────────────────────────────
folium.Marker(
    [CENTER_LAT, CENTER_LON],
    popup="Map Center",
    icon=folium.Icon(color="orange", icon="map-marker"),
).add_to(m)

# ── LEGEND ─────────────────────────────────────────────────────────────────────
map_var      = m.get_name()
js_segments  = fg_segments.get_name()
js_labels    = fg_labels.get_name()

palette_swatches = ""
for i, color in enumerate(PALETTE):
    palette_swatches += (
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<span style="display:inline-block;width:18px;height:12px;'
        f'background:{color};border:1px solid #999;border-radius:2px"></span>'
        f'Color {i+1}'
        f'</div>'
    )
if fallback_colors_used:
    palette_swatches += (
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<span style="display:inline-block;width:18px;height:12px;'
        f'background:#aaa;border:1px solid #999;border-radius:2px"></span>'
        f'Fallback ({len(fallback_colors_used)} extra colors)'
        f'</div>'
    )

legend_html = f"""
<div id="bm-legend" style="
    position:fixed;bottom:30px;left:30px;z-index:9999;
    background:white;padding:10px 14px;border:2px solid #aaa;
    border-radius:6px;font-size:13px;font-family:sans-serif;line-height:1.8;
    max-width:260px;">
  <b style="font-size:14px">Block ID Map</b><br>
  <span style="color:#555;font-size:12px">
    {len(all_blocks)} blocks &nbsp;|&nbsp;
    {segment_count} segments<br>
    {len(palette_colors_used)} palette + {len(fallback_colors_used)} fallback colors
  </span>
  <hr style="margin:6px 0;border:none;border-top:1px solid #ddd">
  {palette_swatches}
  <hr style="margin:6px 0;border:none;border-top:1px solid #ddd">

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_seg" checked
           onchange="bmToggle('{js_segments}', this.checked)">
    Block segments (polylines)
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_lbl" checked
           onchange="bmToggle('{js_labels}', this.checked)">
    Block ID labels (dots)
  </label>
</div>

<script>
function bmToggle(fgName, show) {{
  var mapObj = window['{map_var}'];
  var fg = window[fgName];
  if (!fg) return;
  if (show) {{
    if (!mapObj.hasLayer(fg)) mapObj.addLayer(fg);
  }} else {{
    if (mapObj.hasLayer(fg)) mapObj.removeLayer(fg);
  }}
}}
</script>
"""

m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m)

# ── SAVE ───────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.html")
m.save(out_path)
print(f"\nBlock map saved -> {out_path}")
print(f"Blocks: {len(all_blocks)}, Segments drawn: {segment_count}, "
      f"Labels placed: {len(block_label_placed)}")