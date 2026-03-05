import os
import json
import math
from typing import cast

import pandas as pd
import folium
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import shape, box, Point
from shapely.geometry.base import BaseGeometry

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "Output/test_maps"

LOCATIONS = [
    {
        "name":   "sf_test_map",
        "lat":    37 + 46/60 + 16.6/3600,      # 37°46'16.6"N
        "lon":    -(122 + 25/60 + 27.1/3600),   # 122°25'27.1"W
        "zoom":   19,
        "bbox_m": 750,
        "parquet": "Output/San_Francisco_County_California_USA_network.parquet",
    },
    {
        "name":   "sf_embarcadero",
        "lat":    37 + 47/60 + 42.9/3600,      # 37°47'42.9"N
        "lon":    -(122 + 23/60 + 38.3/3600),   # 122°23'38.3"W
        "zoom":   19,
        "bbox_m": 750,
        "parquet": "Output/San_Francisco_County_California_USA_network.parquet",
    },
    {
        "name":   "alameda_test_map",
        "lat":    37 + 52/60 + 16.4/3600,      # 37°52'16.4"N
        "lon":    -(122 + 16/60 + 4.8/3600),    # 122°16'04.8"W
        "zoom":   19,
        "bbox_m": 750,
        "parquet": "Output/Alameda_County_California_USA_network.parquet",
    },
    {
        "name":   "alameda_ashby",
        "lat":    37 + 50/60 + 49.5/3600,      # 37°50'49.5"N
        "lon":    -(122 + 16/60 + 18.8/3600),   # 122°16'18.8"W
        "zoom":   19,
        "bbox_m": 750,
        "parquet": "Output/Alameda_County_California_USA_network.parquet",
    },
]

# ── COORDINATE TRANSFORMERS ────────────────────────────────────────────────────
_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
_to_wgs = Transformer.from_crs('EPSG:32610', 'EPSG:4326', always_xy=True)

# ── CURB RAMP SCHEMA CONSTANTS ─────────────────────────────────────────────────
_CURBRAMP_SIDES     = ('left', 'right')
_CURBRAMP_POSITIONS = ('start', 'end')
_CURBRAMP_INDICES   = (1, 2, 3)

# ── PARQUET CACHE ──────────────────────────────────────────────────────────────
_parquet_cache: dict[str, pd.DataFrame] = {}

_GEOM_COLS = ['street_geometry', 'sidewalk_left_geometry', 'sidewalk_right_geometry',
              'bikeway_left_1_geometry', 'bikeway_right_1_geometry']

def load_parquet(path: str) -> pd.DataFrame:
    """Load a parquet file, printing a column summary on first load. Cached by path."""
    if path in _parquet_cache:
        return _parquet_cache[path]
    data = pd.read_parquet(path)
    print(f"Loaded {len(data)} rows from {os.path.basename(path)}")
    if data.empty:
        raise ValueError(f"Parquet is empty: {path}")
    for col in _GEOM_COLS:
        if col in data.columns:
            print(f"  {col}: {data[col].dropna().__len__()} non-null values")
        else:
            print(f"  {col}: COLUMN MISSING")
    _parquet_cache[path] = data
    return data

# ── GEOMETRY HELPERS ───────────────────────────────────────────────────────────
def parse_geom(val) -> BaseGeometry | None:
    """Parse a geometry from WKB bytes, WKB hex string, WKT string, GeoJSON, or shapely object."""
    from shapely import wkb
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if hasattr(val, 'geom_type'):
            return cast(BaseGeometry, val)
        if isinstance(val, bytes):
            return wkb.loads(val)
        if isinstance(val, dict):
            return shape(val)
        s = str(val).strip()
        if not s or s.lower() in ('none', 'nan', 'null'):
            return None
        if s.startswith('{'):
            return shape(json.loads(s))
        if all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0:
            return wkb.loads(s, hex=True)
        return wkt.loads(s)
    except Exception:
        return None


def _utm_to_latlon(x, y):
    lon, lat = _to_wgs.transform(x, y)
    return [lat, lon]


def geom_to_latlons(geom):
    """Return list of [lat, lon] pairs for LineString / MultiLineString (UTM -> WGS84)."""
    if geom is None:
        return []
    if geom.geom_type == 'LineString':
        return [_utm_to_latlon(c[0], c[1]) for c in geom.coords]
    if geom.geom_type == 'MultiLineString':
        coords = []
        for line in geom.geoms:
            coords.extend([_utm_to_latlon(c[0], c[1]) for c in line.coords])
        return coords
    return []


def add_endpoints(coords, color, target):
    """Draw a small circle at the start and end of a segment."""
    for pt in (coords[0], coords[-1]):
        folium.CircleMarker(
            location=pt, radius=3,
            color=color, fill=True, fill_color=color, fill_opacity=1.0,
            weight=1
        ).add_to(target)


def make_popup(title, attrs):
    """Build an HTML popup table from a title and a list of (label, value) pairs."""
    rows = ''.join(
        f'<tr><td style="padding:2px 8px 2px 0;color:#555;white-space:nowrap">'
        f'<b>{k}</b></td>'
        f'<td style="padding:2px 0">{v if (v is not None and str(v).strip() not in ("", "nan", "None")) else "<i>—</i>"}</td></tr>'
        for k, v in attrs
    )
    html = (
        f'<div style="font-family:sans-serif;font-size:12px;min-width:180px">'
        f'<b style="font-size:13px">{title}</b>'
        f'<table style="border-collapse:collapse;margin-top:4px">{rows}</table>'
        f'</div>'
    )
    return folium.Popup(html, max_width=320)


def _sw_attrs_from_row(row, side):
    return [
        ('Side',        side),
        ('Presence',    row.get(f'sidewalk_{side}_presence')),
        ('Width (m)',   row.get(f'sidewalk_{side}_width')),
        ('Surface',     row.get(f'sidewalk_{side}_surface')),
        ('Condition',   row.get(f'sidewalk_{side}_condition')),
        ('Kerb',        row.get(f'sidewalk_{side}_kerb')),
        ('Obstacle',    row.get(f'sidewalk_{side}_obstacle')),
        ('Street name', row.get('name')),
    ]


def _is_buffered(val):
    """Return True if a buffered column value is truthy (True, 'yes', non-empty string)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s not in ('', 'none', 'nan', 'null', 'no', '0', 'false')


def _curbramp_col(side, position, index, suffix):
    """Return the canonical column name for a curb ramp attribute."""
    return f'sidewalk_{side}_curbramp_{position}_{index}_{suffix}'


# ── MAP GENERATION ─────────────────────────────────────────────────────────────
def generate_map(data: pd.DataFrame, center_lat: float, center_lon: float,
                 output_name: str, zoom: int = 19, bbox_m: int = 750) -> str:
    """Build and save a folium map centered on (center_lat, center_lon).

    Returns the path to the saved HTML file.
    """
    # -- bounding box in UTM, degrees for rectangle drawing
    cx, cy  = _to_utm.transform(center_lon, center_lat)
    bbox    = box(cx - bbox_m, cy - bbox_m, cx + bbox_m, cy + bbox_m)
    lat_deg = bbox_m / 111320
    lon_deg = bbox_m / (111320 * math.cos(math.radians(center_lat)))

    def in_bbox(geom):
        return geom is not None and bbox.intersects(geom)

    def draw_sidewalk(coords, row, side, label, target_sep, target_buf):
        presence = row.get(f'sidewalk_{side}_presence', '') if side else ''
        width    = row.get(f'sidewalk_{side}_width', '')    if side else ''
        buffered = _is_buffered(row.get(f'sidewalk_{side}_buffered')) if side else False
        color    = '#add8e6' if buffered else '#00008b'
        kind     = 'buffered' if buffered else 'separate'
        target   = target_buf if buffered else target_sep
        attrs    = _sw_attrs_from_row(row, side) if side else [
            ('Name',       row.get('name')),
            ('Highway',    row.get('highway')),
            ('OSM ID',     row.get('osmid')),
            ('Length (m)', row.get('length')),
            ('Surface',    row.get('surface')),
            ('Access',     row.get('access')),
        ]
        folium.PolyLine(
            coords, color=color, weight=2, opacity=0.85,
            tooltip=f"Sidewalk {label} [{kind}]: presence={presence}, width={width}",
            popup=make_popup(f'Sidewalk ({label}) [{kind}]', attrs)
        ).add_to(target)
        add_endpoints(coords, color, target)

    # -- map + feature groups
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles='CartoDB positron')

    fg_streets   = folium.FeatureGroup(name='Streets',              show=True)
    fg_sw_sep    = folium.FeatureGroup(name='Sidewalks – separate', show=True)
    fg_sw_buf    = folium.FeatureGroup(name='Sidewalks – buffered', show=True)
    fg_bk_sep    = folium.FeatureGroup(name='Bikeways – separate',  show=True)
    fg_bk_buf    = folium.FeatureGroup(name='Bikeways – buffered',  show=True)
    fg_curbramps = folium.FeatureGroup(name='Curb Ramps',           show=True)
    fg_bbox      = folium.FeatureGroup(name=f'{bbox_m} m bbox',     show=True)

    folium.Rectangle(
        bounds=[
            [center_lat - lat_deg, center_lon - lon_deg],
            [center_lat + lat_deg, center_lon + lon_deg],
        ],
        color='gray', weight=1.5, fill=False, dash_array='6 4',
        tooltip=f"{bbox_m} m bounding box"
    ).add_to(fg_bbox)

    counts = {
        'street': 0,
        'sidewalk_separate': 0,
        'sidewalk_buffered': 0,
        'bikeway_separate': 0,
        'bikeway_buffered': 0,
        'curbramp': 0,
    }

    for _, row in data.iterrows():
        hw         = str(row.get('highway') or '').strip().lower()
        is_footway = hw == 'footway'

        # Streets (red) — skip footway edges
        street_geom = parse_geom(row.get('street_geometry'))
        if in_bbox(street_geom):
            coords = geom_to_latlons(street_geom)
            if coords:
                name = row.get('name', '') or ''
                if is_footway:
                    is_buf = _is_buffered(row.get('sidewalk_buffered', False))
                    draw_sidewalk(coords, row, None, f'footway – {name or "(unnamed)"}',
                                  fg_sw_sep, fg_sw_buf)
                    counts['sidewalk_buffered' if is_buf else 'sidewalk_separate'] += 1
                else:
                    street_attrs = [
                        ('Name',       row.get('name')),
                        ('Highway',    row.get('highway')),
                        ('OSM ID',     row.get('osmid')),
                        ('Lanes',      row.get('lanes')),
                        ('Max speed',  row.get('maxspeed')),
                        ('Oneway',     row.get('oneway')),
                        ('Length (m)', row.get('length')),
                        ('Surface',    row.get('surface')),
                        ('Access',     row.get('access')),
                    ]
                    folium.PolyLine(
                        coords, color='red', weight=3, opacity=0.85,
                        tooltip=f"Street: {name} ({hw})",
                        popup=make_popup(f'Street: {name or "(unnamed)"}', street_attrs)
                    ).add_to(fg_streets)
                    add_endpoints(coords, 'red', fg_streets)
                    counts['street'] += 1

        # Sidewalks (blue)
        for side in ('left', 'right'):
            sw_geom = parse_geom(row.get(f'sidewalk_{side}_geometry'))
            if in_bbox(sw_geom):
                coords = geom_to_latlons(sw_geom)
                if coords:
                    is_sw_buf = _is_buffered(row.get(f'sidewalk_{side}_buffered'))
                    draw_sidewalk(coords, row, side, side, fg_sw_sep, fg_sw_buf)
                    counts['sidewalk_buffered' if is_sw_buf else 'sidewalk_separate'] += 1

        # Bikeways (green)
        for side in ('left', 'right'):
            bk_buffered = _is_buffered(row.get(f'bikeway_{side}_buffered'))
            bk_color    = '#90ee90' if bk_buffered else '#006400'
            bk_kind     = 'buffered' if bk_buffered else 'separate'
            bk_target   = fg_bk_buf if bk_buffered else fg_bk_sep
            for num in (1, 2):
                bk_geom = parse_geom(row.get(f'bikeway_{side}_{num}_geometry'))
                if in_bbox(bk_geom):
                    coords = geom_to_latlons(bk_geom)
                    if coords:
                        bk_type  = row.get(f'bikeway_{side}_{num}_type', '')
                        bk_attrs = [
                            ('Side',        side),
                            ('Lane #',      num),
                            ('Kind',        bk_kind),
                            ('Type',        row.get(f'bikeway_{side}_{num}_type')),
                            ('Width (m)',   row.get(f'bikeway_{side}_{num}_width')),
                            ('Surface',     row.get(f'bikeway_{side}_{num}_surface')),
                            ('Condition',   row.get(f'bikeway_{side}_{num}_condition')),
                            ('Separation',  row.get(f'bikeway_{side}_{num}_separation')),
                            ('Street name', row.get('name')),
                        ]
                        folium.PolyLine(
                            coords, color=bk_color, weight=2, opacity=0.85,
                            tooltip=f"Bikeway {side}-{num} [{bk_kind}]: {bk_type}",
                            popup=make_popup(f'Bikeway ({side}-{num}) [{bk_kind}]', bk_attrs)
                        ).add_to(bk_target)
                        add_endpoints(coords, bk_color, bk_target)
                        counts['bikeway_buffered' if bk_buffered else 'bikeway_separate'] += 1

        # Curb Ramps (orange)
        for side in _CURBRAMP_SIDES:
            for position in _CURBRAMP_POSITIONS:
                for index in _CURBRAMP_INDICES:
                    geom_col = _curbramp_col(side, position, index, 'geometry')
                    raw_val  = row.get(geom_col)
                    if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                        continue

                    ramp_geom = parse_geom(raw_val)
                    if ramp_geom is None or not in_bbox(ramp_geom):
                        continue
                    if ramp_geom.geom_type != 'Point':
                        continue

                    ramp_point = cast(Point, ramp_geom)
                    lat_lon    = _utm_to_latlon(ramp_point.x, ramp_point.y)

                    ramp_attrs = [
                        ('Ramp ID',          row.get(_curbramp_col(side, position, index, 'ID'))),
                        ('Side',             side),
                        ('Position',         position),
                        ('Index',            index),
                        ('Return direction', row.get(_curbramp_col(side, position, index, 'returnloc'))),
                        ('Return position',  row.get(_curbramp_col(side, position, index, 'returnposition'))),
                        ('Condition score',  row.get(_curbramp_col(side, position, index, 'condition_score'))),
                        ('Street name',      row.get('name')),
                    ]
                    folium.CircleMarker(
                        location=lat_lon, radius=5,
                        color='orange', fill=True, fill_color='orange',
                        fill_opacity=0.9, weight=1,
                        tooltip=(f"Curb ramp {side}-{position}-{index}: "
                                 f"returnloc={row.get(_curbramp_col(side, position, index, 'returnloc'))}, "
                                 f"returnpos={row.get(_curbramp_col(side, position, index, 'returnposition'))}"),
                        popup=make_popup(f'Curb Ramp ({side} {position} #{index})', ramp_attrs)
                    ).add_to(fg_curbramps)
                    counts['curbramp'] += 1

    # -- assemble map
    for fg in (fg_bbox, fg_streets, fg_bk_sep, fg_bk_buf, fg_sw_sep, fg_sw_buf, fg_curbramps):
        fg.add_to(m)

    folium.Marker(
        [center_lat, center_lon],
        popup=output_name,
        icon=folium.Icon(color='orange', icon='map-marker')
    ).add_to(m)

    # -- interactive legend
    map_var      = m.get_name()
    js_street    = fg_streets.get_name()
    js_sw_sep    = fg_sw_sep.get_name()
    js_sw_buf    = fg_sw_buf.get_name()
    js_bk_sep    = fg_bk_sep.get_name()
    js_bk_buf    = fg_bk_buf.get_name()
    js_curbramps = fg_curbramps.get_name()
    js_bbox      = fg_bbox.get_name()

    legend_html = f"""
<div id="px-legend" style="
    position:fixed;bottom:30px;left:30px;z-index:9999;
    background:white;padding:10px 14px;border:2px solid #aaa;
    border-radius:6px;font-size:13px;font-family:sans-serif;line-height:1.9;">
  <b style="font-size:14px">Legend</b><br>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_curbramps" checked
           onchange="toggleFG('{js_curbramps}', this.checked)">
    <span style="color:orange;font-size:18px;line-height:1">&#9679;</span>
    Curb Ramps ({counts['curbramp']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_sw_buf" checked
           onchange="toggleFG('{js_sw_buf}', this.checked)">
    <span style="color:#add8e6;font-size:18px;line-height:1">&#9644;</span>
    Sidewalks – buffered ({counts['sidewalk_buffered']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_sw_sep" checked
           onchange="toggleFG('{js_sw_sep}', this.checked)">
    <span style="color:#00008b;font-size:18px;line-height:1">&#9644;</span>
    Sidewalks – separate ({counts['sidewalk_separate']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_bk_buf" checked
           onchange="toggleFG('{js_bk_buf}', this.checked)">
    <span style="color:#90ee90;font-size:18px;line-height:1">&#9644;</span>
    Bikeways – buffered ({counts['bikeway_buffered']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_bk_sep" checked
           onchange="toggleFG('{js_bk_sep}', this.checked)">
    <span style="color:#006400;font-size:18px;line-height:1">&#9644;</span>
    Bikeways – separate ({counts['bikeway_separate']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_streets" checked
           onchange="toggleFG('{js_street}', this.checked)">
    <span style="color:red;font-size:18px;line-height:1">&#9644;</span>
    Streets ({counts['street']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_bbox" checked
           onchange="toggleFG('{js_bbox}', this.checked)">
    <span style="color:gray;font-size:18px;line-height:1">&#9645;</span>
    {bbox_m} m bbox
  </label>

</div>

<script>
function toggleFG(fgName, show) {{
  var mapObj = window['{map_var}'];
  if (show) {{
    if (!mapObj.hasLayer(window[fgName])) mapObj.addLayer(window[fgName]);
  }} else {{
    if (mapObj.hasLayer(window[fgName]))  mapObj.removeLayer(window[fgName]);
  }}
}}
</script>
"""

    m.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[attr-defined]
    folium.LayerControl(collapsed=False).add_to(m)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{output_name}.html")
    m.save(out_path)
    return out_path


# ── BLOCK MAP GENERATION ──────────────────────────────────────────────────────

# Palette large enough that greedy graph-coloring rarely needs to wrap.
_BLOCK_PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075', '#a9a9a9',
]


def _graph_color_blocks(data: pd.DataFrame) -> dict[str, int]:
    """Assign a color index to each block ID such that no two adjacent blocks
    share the same colour (greedy graph colouring).

    Two blocks are adjacent when they appear on left/right of the same edge.
    """
    adjacency: dict[str, set[str]] = {}
    for _, row in data.iterrows():
        bid = row.get('block_ids')
        if not isinstance(bid, dict):
            continue
        left  = bid.get('left')
        right = bid.get('right')
        if left is not None:
            adjacency.setdefault(left, set())
        if right is not None:
            adjacency.setdefault(right, set())
        if left is not None and right is not None and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)

    # Greedy colouring: process nodes in descending adjacency-count order
    color_map: dict[str, int] = {}
    n_colors = len(_BLOCK_PALETTE)
    for block_id in sorted(adjacency, key=lambda b: len(adjacency[b]), reverse=True):
        used = {color_map[nb] for nb in adjacency[block_id] if nb in color_map}
        for c in range(n_colors):
            if c not in used:
                color_map[block_id] = c
                break
        else:
            color_map[block_id] = 0  # wrap if palette exhausted

    return color_map


def generate_block_map(data: pd.DataFrame, center_lat: float, center_lon: float,
                       output_name: str, zoom: int = 17, bbox_m: int = 750) -> str:
    """Build a folium map showing street segments coloured by block assignment,
    with a block ID label placed at the centroid of each block's total geometry.
    """
    from shapely.ops import unary_union

    cx, cy = _to_utm.transform(center_lon, center_lat)
    bbox   = box(cx - bbox_m, cy - bbox_m, cx + bbox_m, cy + bbox_m)
    lat_deg = bbox_m / 111320
    lon_deg = bbox_m / (111320 * math.cos(math.radians(center_lat)))

    def in_bbox(geom):
        return geom is not None and bbox.intersects(geom)

    # Filter to edges inside bbox
    visible_rows = []
    for idx, row in data.iterrows():
        street_geom = parse_geom(row.get('street_geometry'))
        if in_bbox(street_geom):
            visible_rows.append(idx)

    subset    = data.loc[visible_rows]
    color_map = _graph_color_blocks(subset)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles='CartoDB positron')

    fg_blocks     = folium.FeatureGroup(name='Block-coloured streets', show=True)
    fg_unassigned = folium.FeatureGroup(name='Unassigned streets',     show=True)
    fg_labels     = folium.FeatureGroup(name='Block ID labels',        show=True)
    fg_bbox       = folium.FeatureGroup(name=f'{bbox_m} m bbox',       show=True)

    folium.Rectangle(
        bounds=[
            [center_lat - lat_deg, center_lon - lon_deg],
            [center_lat + lat_deg, center_lon + lon_deg],
        ],
        color='gray', weight=1.5, fill=False, dash_array='6 4',
        tooltip=f"{bbox_m} m bounding box"
    ).add_to(fg_bbox)

    # Draw coloured streets and accumulate geometries per block ID
    block_geoms: dict[str, list] = {}
    n_assigned   = 0
    n_unassigned = 0

    for _, row in subset.iterrows():
        street_geom = parse_geom(row.get('street_geometry'))
        if street_geom is None:
            continue
        coords = geom_to_latlons(street_geom)
        if not coords:
            continue

        bid      = row.get('block_ids')
        left_id  = bid.get('left')  if isinstance(bid, dict) else None
        right_id = bid.get('right') if isinstance(bid, dict) else None
        primary_id = left_id or right_id
        name = row.get('name', '') or ''

        if primary_id is not None and primary_id in color_map:
            color  = _BLOCK_PALETTE[color_map[primary_id] % len(_BLOCK_PALETTE)]
            target = fg_blocks
            n_assigned += 1
        else:
            color  = '#cccccc'
            target = fg_unassigned
            n_unassigned += 1

        attrs = [
            ('Name',         name),
            ('Highway',      row.get('highway')),
            ('Block left',   left_id),
            ('Block right',  right_id),
            ('Block side L', (row.get('block_sides') or {}).get('left')  if isinstance(row.get('block_sides'), dict) else None),
            ('Block side R', (row.get('block_sides') or {}).get('right') if isinstance(row.get('block_sides'), dict) else None),
            ('Bearing',      row.get('normalized_bearing')),
        ]
        folium.PolyLine(
            coords, color=color, weight=4, opacity=0.9,
            tooltip=f"{name} | L={left_id} R={right_id}",
            popup=make_popup(f'Street: {name or "(unnamed)"}', attrs),
        ).add_to(target)

        # Accumulate geometries per block side for centroid labels
        if isinstance(bid, dict):
            for side_id in (left_id, right_id):
                if side_id is not None:
                    block_geoms.setdefault(str(side_id), []).append(street_geom)

    # Place block ID label at centroid of each block's unioned geometry
    n_labels = 0
    for block_id, geoms in block_geoms.items():
        union    = unary_union(geoms)
        centroid = union.centroid
        lat_lon  = _utm_to_latlon(centroid.x, centroid.y)
        folium.Marker(
            lat_lon,
            icon=folium.DivIcon(
                html=(
                    f'<div style="'
                    f'font-size:9px;font-weight:bold;color:#222;'
                    f'white-space:nowrap;'
                    f'background:rgba(255,255,255,0.75);'
                    f'padding:1px 3px;border-radius:2px;'
                    f'border:1px solid #999;line-height:1.2'
                    f'">{block_id}</div>'
                ),
                icon_size=(80, 18),
                icon_anchor=(40, 9),
            ),
            tooltip=f"Block {block_id}",
        ).add_to(fg_labels)
        n_labels += 1

    # Assemble
    for fg in (fg_bbox, fg_blocks, fg_unassigned, fg_labels):
        fg.add_to(m)

    folium.Marker(
        [center_lat, center_lon],
        popup=output_name,
        icon=folium.Icon(color='orange', icon='map-marker')
    ).add_to(m)

    # Legend
    map_var       = m.get_name()
    js_blocks     = fg_blocks.get_name()
    js_unassigned = fg_unassigned.get_name()
    js_labels     = fg_labels.get_name()
    js_bbox       = fg_bbox.get_name()

    legend_html = f"""
<div id="px-legend" style="
    position:fixed;bottom:30px;left:30px;z-index:9999;
    background:white;padding:10px 14px;border:2px solid #aaa;
    border-radius:6px;font-size:13px;font-family:sans-serif;line-height:1.9;">
  <b style="font-size:14px">Block Map</b><br>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" checked
           onchange="toggleFG('{js_blocks}', this.checked)">
    <span style="font-size:18px;line-height:1">&#9644;</span>
    Block-coloured ({n_assigned})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" checked
           onchange="toggleFG('{js_unassigned}', this.checked)">
    <span style="color:#cccccc;font-size:18px;line-height:1">&#9644;</span>
    Unassigned ({n_unassigned})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" checked
           onchange="toggleFG('{js_labels}', this.checked)">
    <span style="font-size:11px;font-weight:bold;border:1px solid #999;padding:0 3px">ID</span>
    Block ID labels ({n_labels})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" checked
           onchange="toggleFG('{js_bbox}', this.checked)">
    <span style="color:gray;font-size:18px;line-height:1">&#9645;</span>
    {bbox_m} m bbox
  </label>

</div>

<script>
function toggleFG(fgName, show) {{
  var mapObj = window['{map_var}'];
  if (show) {{
    if (!mapObj.hasLayer(window[fgName])) mapObj.addLayer(window[fgName]);
  }} else {{
    if (mapObj.hasLayer(window[fgName]))  mapObj.removeLayer(window[fgName]);
  }}
}}
</script>
"""

    m.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[attr-defined]
    folium.LayerControl(collapsed=False).add_to(m)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{output_name}.html")
    m.save(out_path)
    return out_path


BLOCK_MAP_LOCATIONS = [
    {
        "name":    "sf_block_map",
        "lat":     37 + 46/60 + 16.6/3600,
        "lon":     -(122 + 25/60 + 27.1/3600),
        "zoom":    17,
        "bbox_m":  750,
        "parquet": "Output/San_Francisco_County_California_USA_network.parquet",
    },
    {
        "name":    "alameda_block_map",
        "lat":     37 + 52/60 + 16.4/3600,
        "lon":     -(122 + 16/60 + 4.8/3600),
        "zoom":    17,
        "bbox_m":  750,
        "parquet": "Output/Alameda_County_California_USA_network.parquet",
    },
]


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
# Group all map jobs by parquet path so each file is loaded exactly once.
_facility_jobs: list[tuple[str, dict]] = [(loc["parquet"], loc) for loc in LOCATIONS]
_block_jobs:    list[tuple[str, dict]] = [(loc["parquet"], loc) for loc in BLOCK_MAP_LOCATIONS]

_all_parquets: list[str] = list(dict.fromkeys(
    p for p, _ in _facility_jobs + _block_jobs
))

if not _all_parquets:
    import warnings
    warnings.warn("No map locations configured — no maps will be generated.", stacklevel=1)

for parquet_path in _all_parquets:
    data = load_parquet(parquet_path)

    for pq, loc in _facility_jobs:
        if pq != parquet_path:
            continue
        path = generate_map(
            data        = data,
            center_lat  = loc["lat"],
            center_lon  = loc["lon"],
            output_name = loc["name"],
            zoom        = loc.get("zoom", 19),
            bbox_m      = loc.get("bbox_m", 750),
        )
        print(f"Map saved -> {path}")

    for pq, loc in _block_jobs:
        if pq != parquet_path:
            continue
        path = generate_block_map(
            data        = data,
            center_lat  = loc["lat"],
            center_lon  = loc["lon"],
            output_name = loc["name"],
            zoom        = loc.get("zoom", 17),
            bbox_m      = loc.get("bbox_m", 750),
        )
        print(f"Block map saved -> {path}")
