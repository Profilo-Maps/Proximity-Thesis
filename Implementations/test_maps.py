import ast
import os
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

from tqdm import tqdm

import pandas as pd
import folium
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import shape, box, Point, LineString, MultiLineString, MultiPoint
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


def geom_to_latlons(geom: BaseGeometry | None) -> list[list[float]]:
    """Batch-transform all coords of a LineString/MultiLineString (UTM → WGS84).

    Uses a single pyproj array call per geometry instead of one call per point.
    """
    if geom is None:
        return []
    if isinstance(geom, LineString):
        raw = list(geom.coords)
    elif isinstance(geom, MultiLineString):
        raw = [c for line in geom.geoms for c in line.coords]
    else:
        return []
    if not raw:
        return []
    xs = [c[0] for c in raw]
    ys = [c[1] for c in raw]
    lons, lats = _to_wgs.transform(xs, ys)
    return [[lat, lon] for lat, lon in zip(lats, lons)]


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
        ('ID',          row.get(f'sidewalk_{side}_ID')),
        ('Grid ID',     row.get(f'sidewalk_{side}_grid_ID')),
        ('Presence',    row.get(f'sidewalk_{side}_presence')),
        ('Width (m)',   row.get(f'sidewalk_{side}_width')),
        ('Surface',     row.get(f'sidewalk_{side}_surface')),
        ('Quality',     row.get(f'sidewalk_{side}_quality')),
        ('Incline',     row.get(f'sidewalk_{side}_incline')),
        ('Separator',   row.get(f'sidewalk_{side}_seperator')),
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


# ── INCLINE GRADIENT HELPERS ───────────────────────────────────────────────────
_INCLINE_COLS = [
    'street_incline',
    'sidewalk_left_incline', 'sidewalk_right_incline',
    'bikeway_left_1_incline', 'bikeway_left_2_incline',
    'bikeway_right_1_incline', 'bikeway_right_2_incline',
]

_COLOR_NAMES: dict[str, str] = {
    'red':    '#ff0000',
    'orange': '#ffa500',
}


def _parse_incline(val) -> float | None:
    """Parse an incline value to an absolute float (percent)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return abs(float(val))
    s = str(val).strip().replace('%', '').replace('+', '')
    try:
        return abs(float(s))
    except ValueError:
        return None


def _blend_with_black(hex_color: str, opacity: float) -> str:
    """Darken hex_color by compositing black at the given opacity (0–1)."""
    h = _COLOR_NAMES.get(hex_color, hex_color).lstrip('#')
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    f = 1.0 - max(0.0, min(1.0, opacity))
    return f'#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}'


# ── MAP GENERATION ─────────────────────────────────────────────────────────────
def generate_map(data: pd.DataFrame, center_lat: float, center_lon: float,
                 output_name: str, zoom: int = 19, bbox_m: int = 750,
                 bar_pos: int = 0) -> str:
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
        buffered    = _is_buffered(row.get(f'sidewalk_{side}_buffered')) if side else False
        base_color  = '#add8e6' if buffered else '#00008b'
        incline_raw = row.get(f'sidewalk_{side}_incline') if side else row.get('street_incline')
        color       = _seg_color(base_color, incline_raw)
        kind        = 'buffered' if buffered else 'separate'
        target   = target_buf if buffered else target_sep
        attrs    = _sw_attrs_from_row(row, side) if side else [
            ('Name',        row.get('name')),
            ('Highway',     row.get('highway')),
            ('OSM ID',      row.get('osmid')),
            ('Grid ID',     row.get('street_grid_id')),
            ('Bearing (°)', row.get('normalized_bearing')),
            ('Length (m)',  row.get('length')),
            ('Lanes',       row.get('lanes')),
            ('Lane width',  row.get('lane_width')),
            ('Surface',     row.get('surface')),
            ('Incline',     row.get('street_incline')),
            ('Access',      row.get('access')),
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
    fg_nodes         = folium.FeatureGroup(name='Intersection Nodes',   show=True)
    fg_curbramps     = folium.FeatureGroup(name='Curb Ramps',           show=True)
    fg_traffic_calm  = folium.FeatureGroup(name='Traffic Calming',      show=True)
    fg_bbox          = folium.FeatureGroup(name=f'{bbox_m} m bbox',     show=True)

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
        'traffic_calming': 0,
        'intersection_node': 0,
    }

    _geom_cols_to_parse = (
        [f'sidewalk_{s}_geometry' for s in ('left', 'right')]
        + [f'bikeway_{s}_{n}_geometry' for s in ('left', 'right') for n in (1, 2)]
        + [_curbramp_col(s, p, i, 'geometry')
           for s in _CURBRAMP_SIDES for p in _CURBRAMP_POSITIONS for i in _CURBRAMP_INDICES]
        + ['street_feature_geometry']
    )

    def _parse_list_col(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return ast.literal_eval(str(v))
        except Exception:
            return None

    # Single progress bar covering all pipeline stages.
    # total is updated after bbox filtering once row count is known.
    with tqdm(total=4, desc=f'{output_name} · filtering',
              position=bar_pos, leave=True, unit='step') as pbar:

        # ── Stage 1: filter to bbox ───────────────────────────────────────────
        _parsed_street = data['street_geometry'].apply(parse_geom)  # type: ignore[arg-type]
        _in_bbox_mask  = _parsed_street.apply(lambda g: g is not None and bbox.intersects(g))  # type: ignore[arg-type]
        data = data.loc[_in_bbox_mask].copy()
        data['_street_geom'] = _parsed_street[_in_bbox_mask]
        _present_geom_cols = [c for c in _geom_cols_to_parse if c in data.columns]
        pbar.total = 4 + len(_present_geom_cols) + len(data)
        pbar.refresh()
        pbar.update(1)

        # ── Stage 2: incline normalization ────────────────────────────────────
        pbar.set_description(f'{output_name} · inclines')
        _incline_vals: list[float] = []
        for _icol in _INCLINE_COLS:
            if _icol in data.columns:
                _incline_vals.extend(
                    v for v in data[_icol].apply(_parse_incline) if v is not None
                )
        _incline_min = min(_incline_vals) if _incline_vals else 0.0
        _incline_max = max(_incline_vals) if _incline_vals else 1.0

        def _seg_color(base: str, raw_incline) -> str:
            parsed = _parse_incline(raw_incline)
            if parsed is None or _incline_max <= _incline_min:
                return base
            norm = (parsed - _incline_min) / (_incline_max - _incline_min)
            return _blend_with_black(base, 0.5 * norm)

        pbar.update(1)

        # ── Stage 3: pre-parse geometry columns (one step per column) ─────────
        for _col in _present_geom_cols:
            pbar.set_description(f'{output_name} · {_col}')
            data[f'_p_{_col}'] = data[_col].apply(parse_geom)  # type: ignore[arg-type]
            pbar.update(1)

        # ── Stage 4: pre-parse feature list columns ───────────────────────────
        pbar.set_description(f'{output_name} · feature lists')
        if 'street_feature_types' in data.columns:
            data['_feat_types'] = data['street_feature_types'].apply(_parse_list_col)
        if 'street_feature_attributes' in data.columns:
            data['_feat_attrs'] = data['street_feature_attributes'].apply(_parse_list_col)
        pbar.update(1)

        # ── Stage 5: build map elements (one step per row) ────────────────────
        pbar.set_description(f'{output_name} · building')
        for row in data.to_dict('records'):
            hw           = str(row.get('highway') or '').strip().lower()
            is_footway   = hw == 'footway'
            is_cycleway  = hw == 'cycleway'
            # Streets (red) — skip footway/cycleway edges
            street_geom = row.get('_street_geom')
            if street_geom is not None:
                coords = geom_to_latlons(street_geom)
                if coords:
                    name = row.get('name', '') or ''
                    if is_cycleway:
                        bk_buf   = _is_buffered(row.get('bikeway_left_1_buffered') or row.get('bikeway_right_1_buffered'))
                        bk_color = '#90ee90' if bk_buf else '#006400'
                        bk_kind  = 'buffered' if bk_buf else 'separate'
                        bk_target = fg_bk_buf if bk_buf else fg_bk_sep
                        folium.PolyLine(
                            coords, color=bk_color, weight=2, opacity=0.85,
                            tooltip=f"Bikeway [separate cycleway]: {name or '(unnamed)'}",
                            popup=make_popup(f'Bikeway (separate cycleway): {name or "(unnamed)"}', [
                                ('Name',    row.get('name')),
                                ('Highway', hw),
                                ('OSM ID',  row.get('osmid')),
                                ('Grid ID', row.get('street_grid_id')),
                            ])
                        ).add_to(bk_target)
                        add_endpoints(coords, bk_color, bk_target)
                        counts['bikeway_buffered' if bk_buf else 'bikeway_separate'] += 1
                    elif is_footway:
                        is_buf = _is_buffered(row.get('sidewalk_buffered', False))
                        draw_sidewalk(coords, row, None, f'footway – {name or "(unnamed)"}',
                                      fg_sw_sep, fg_sw_buf)
                        counts['sidewalk_buffered' if is_buf else 'sidewalk_separate'] += 1
                    else:
                        start_is_int = bool(row.get('start_node_is_intersection_node'))
                        end_is_int   = bool(row.get('end_node_is_intersection_node'))
                        street_attrs = [
                            ('Name',        row.get('name')),
                            ('Highway',     row.get('highway')),
                            ('OSM ID',      row.get('osmid')),
                            ('Grid ID',     row.get('street_grid_id')),
                            ('Bearing (°)', row.get('normalized_bearing')),
                            ('Lanes',       row.get('lanes')),
                            ('Lane width',  row.get('lane_width')),
                            ('Max speed',   row.get('maxspeed')),
                            ('Oneway',      row.get('oneway')),
                            ('Length (m)',  row.get('length')),
                            ('Surface',     row.get('surface')),
                            ('Incline',     row.get('street_incline')),
                            ('Access',      row.get('access')),
                            ('Start node ID',           row.get('start_node_id')),
                            ('Start node intersection', 'yes' if start_is_int else 'no'),
                            ('End node ID',             row.get('end_node_id')),
                            ('End node intersection',   'yes' if end_is_int else 'no'),
                        ]
                        street_color = _seg_color('red', row.get('street_incline'))
                        folium.PolyLine(
                            coords, color=street_color, weight=3, opacity=0.85,
                            tooltip=(f"Street: {name} ({hw}) | "
                                     f"start={'⬟' if start_is_int else '·'} "
                                     f"end={'⬟' if end_is_int else '·'}"),
                            popup=make_popup(f'Street: {name or "(unnamed)"}', street_attrs)
                        ).add_to(fg_streets)
                        add_endpoints(coords, street_color, fg_streets)
                        counts['street'] += 1

                        # Intersection nodes (dark red)
                        _NODE_COLOR = '#8b0000'
                        if start_is_int:
                            folium.CircleMarker(
                                location=coords[0], radius=5,
                                color=_NODE_COLOR, fill=True,
                                fill_color=_NODE_COLOR, fill_opacity=1.0, weight=1,
                                tooltip=f"Intersection node (start): {row.get('start_node_id')}",
                            ).add_to(fg_nodes)
                            counts['intersection_node'] += 1
                        if end_is_int:
                            folium.CircleMarker(
                                location=coords[-1], radius=5,
                                color=_NODE_COLOR, fill=True,
                                fill_color=_NODE_COLOR, fill_opacity=1.0, weight=1,
                                tooltip=f"Intersection node (end): {row.get('end_node_id')}",
                            ).add_to(fg_nodes)
                            counts['intersection_node'] += 1

            # Sidewalks (blue)
            for side in ('left', 'right'):
                sw_geom = row.get(f'_p_sidewalk_{side}_geometry')
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
                    bk_geom = row.get(f'_p_bikeway_{side}_{num}_geometry')
                    if in_bbox(bk_geom):
                        coords = geom_to_latlons(bk_geom)
                        if coords:
                            bk_type  = row.get(f'bikeway_{side}_{num}_type', '')
                            bk_attrs = [
                                ('Side',        side),
                                ('ID',          row.get(f'bikeway_{side}_{num}_id')),
                                ('Grid ID',     row.get(f'bikeway_{side}_{num}_grid_id')),
                                ('Lane #',      num),
                                ('Kind',        bk_kind),
                                ('Type',        row.get(f'bikeway_{side}_{num}_type')),
                                ('Width (m)',   row.get(f'bikeway_{side}_{num}_width')),
                                ('Surface',     row.get(f'bikeway_{side}_{num}_surface')),
                                ('Quality',     row.get(f'bikeway_{side}_{num}_quality')),
                                ('Incline',     row.get(f'bikeway_{side}_{num}_incline')),
                                ('Permitted',   row.get(f'bikeway_{side}_{num}_permitted')),
                                ('Separator',   row.get(f'bikeway_{side}_{num}_seperator')),
                                ('Condition',   row.get(f'bikeway_{side}_{num}_condition')),
                                ('Separation',  row.get(f'bikeway_{side}_{num}_separation')),
                                ('Street name', row.get('name')),
                            ]
                            bk_draw_color = _seg_color(bk_color, row.get(f'bikeway_{side}_{num}_incline'))
                            folium.PolyLine(
                                coords, color=bk_draw_color, weight=2, opacity=0.85,
                                tooltip=f"Bikeway {side}-{num} [{bk_kind}]: {bk_type}",
                                popup=make_popup(f'Bikeway ({side}-{num}) [{bk_kind}]', bk_attrs)
                            ).add_to(bk_target)
                            add_endpoints(coords, bk_draw_color, bk_target)
                            counts['bikeway_buffered' if bk_buffered else 'bikeway_separate'] += 1

            # Curb Ramps (orange)
            for side in _CURBRAMP_SIDES:
                for position in _CURBRAMP_POSITIONS:
                    for index in _CURBRAMP_INDICES:
                        geom_col = _curbramp_col(side, position, index, 'geometry')
                        ramp_geom = row.get(f'_p_{geom_col}')
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

            # Traffic Calming (purple) — from street_feature_* columns
            feat_types = row.get('_feat_types')
            mp         = row.get('_p_street_feature_geometry')
            if feat_types and mp is not None:
                try:
                    feat_attrs_list = row.get('_feat_attrs') or [None] * len(feat_types)
                    if mp.geom_type == 'MultiPoint':
                        points: list[Point] = list(cast(MultiPoint, mp).geoms)  # type: ignore[arg-type]
                    elif mp.geom_type == 'Point':
                        points = [cast(Point, mp)]
                    else:
                        points = []
                    for idx, (ftype, fpt) in enumerate(zip(feat_types, points)):
                        if not str(ftype).strip().lower().startswith('traffic_calming'):
                            continue
                        if not in_bbox(fpt):
                            continue
                        lat_lon = _utm_to_latlon(fpt.x, fpt.y)
                        fattr = feat_attrs_list[idx] if idx < len(feat_attrs_list) else None
                        popup_attrs = [('Street', row.get('name')), ('Type', ftype)]
                        if isinstance(fattr, dict):
                            popup_attrs += list(fattr.items())
                        elif fattr:
                            popup_attrs.append(('Attributes', str(fattr)))
                        folium.CircleMarker(
                            location=lat_lon, radius=5,
                            color='#6a0dad', fill=True, fill_color='#6a0dad',
                            fill_opacity=0.85, weight=1.5,
                            tooltip=f"Traffic calming ({ftype}): {row.get('name', '') or '(unnamed)'}",
                            popup=make_popup('Traffic Calming', popup_attrs)
                        ).add_to(fg_traffic_calm)
                        counts['traffic_calming'] += 1
                except Exception:
                    pass

            pbar.update(1)

        # ── Stage 6: assemble + save ──────────────────────────────────────────
        pbar.set_description(f'{output_name} · saving')
        for fg in (fg_bbox, fg_streets, fg_bk_sep, fg_bk_buf, fg_sw_sep, fg_sw_buf, fg_curbramps, fg_traffic_calm, fg_nodes):
            fg.add_to(m)

        folium.Marker(
            [center_lat, center_lon],
            popup=output_name,
            icon=folium.Icon(color='orange', icon='map-marker')
        ).add_to(m)

        map_var      = m.get_name()
        js_street    = fg_streets.get_name()
        js_sw_sep    = fg_sw_sep.get_name()
        js_sw_buf    = fg_sw_buf.get_name()
        js_bk_sep    = fg_bk_sep.get_name()
        js_bk_buf    = fg_bk_buf.get_name()
        js_curbramps    = fg_curbramps.get_name()
        js_traffic_calm = fg_traffic_calm.get_name()
        js_nodes        = fg_nodes.get_name()
        js_bbox         = fg_bbox.get_name()

        legend_html = f"""
<div id="px-legend" style="
    position:fixed;bottom:30px;left:30px;z-index:9999;
    background:white;padding:10px 14px;border:2px solid #aaa;
    border-radius:6px;font-size:13px;font-family:sans-serif;line-height:1.9;">
  <b style="font-size:14px">Legend</b><br>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_nodes" checked
           onchange="toggleFG('{js_nodes}', this.checked)">
    <span style="color:#8b0000;font-size:18px;line-height:1">&#9679;</span>
    Intersection Nodes ({counts['intersection_node']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_curbramps" checked
           onchange="toggleFG('{js_curbramps}', this.checked)">
    <span style="color:orange;font-size:18px;line-height:1">&#9679;</span>
    Curb Ramps ({counts['curbramp']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_traffic_calm" checked
           onchange="toggleFG('{js_traffic_calm}', this.checked)">
    <span style="color:#6a0dad;font-size:18px;line-height:1">&#9679;</span>
    Traffic Calming ({counts['traffic_calming']})
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

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{output_name}.html")
        m.save(out_path)
        pbar.update(1)

    return out_path


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
# Group all map jobs by parquet path so each file is loaded exactly once.
_facility_jobs: list[tuple[str, dict]] = [(loc["parquet"], loc) for loc in LOCATIONS]

_all_parquets: list[str] = list(dict.fromkeys(
    p for p, _ in _facility_jobs
))

if not _all_parquets:
    import warnings
    warnings.warn("No map locations configured — no maps will be generated.", stacklevel=1)

for parquet_path in _all_parquets:
    data = load_parquet(parquet_path)

    facility_locs = [loc for pq, loc in _facility_jobs if pq == parquet_path]

    with ThreadPoolExecutor() as ex:
        future_to_name = {
            ex.submit(generate_map, data,
                      loc["lat"], loc["lon"], loc["name"],
                      loc.get("zoom", 19), loc.get("bbox_m", 750),
                      bar_pos=i): loc["name"]
            for i, loc in enumerate(facility_locs)
        }
        for f in as_completed(future_to_name):
            print(f"Map saved -> {f.result()}")
