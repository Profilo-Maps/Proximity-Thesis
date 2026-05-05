import ast
import os
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import numpy as np

from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import folium
from pyproj import Transformer
import shapely
from shapely import wkt
from shapely.geometry import shape, box, Point, LineString, MultiLineString, MultiPoint, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from ProximityModel import (
    PipelineConfig as _PipelineConfig,
    step_12_curb_ramps_and_hulls as _step_12,
    step_13_merge_hulls as _step_13,
    step_14_crosswalk_slots as _step_14,
)

_DEFAULT_PIPELINE_CONFIG = _PipelineConfig(
    place_name="test_maps",
    output_path="",
)

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
        "name":   "sf_outer_sunset",
        "lat":    37 + 44/60 + 57.9/3600,      # 37°44'57.9"N
        "lon":    -(122 + 28/60 + 17.0/3600),   # 122°28'17.0"W
        "zoom":   19,
        "bbox_m": 750,
        "parquet": "Output/San_Francisco_County_California_USA_network.parquet",
    },
    # TODO: re-enable when Alameda parquet is ready
    # {
    #     "name":   "alameda_test_map",
    #     "lat":    37 + 52/60 + 16.4/3600,      # 37°52'16.4"N
    #     "lon":    -(122 + 16/60 + 4.8/3600),    # 122°16'04.8"W
    #     "zoom":   19,
    #     "bbox_m": 750,
    #     "parquet": "Output/Alameda_County_California_USA_network.parquet",
    # },
    # {
    #     "name":   "alameda_ashby",
    #     "lat":    37 + 50/60 + 49.5/3600,      # 37°50'49.5"N
    #     "lon":    -(122 + 16/60 + 18.8/3600),   # 122°16'18.8"W
    #     "zoom":   19,
    #     "bbox_m": 750,
    #     "parquet": "Output/Alameda_County_California_USA_network.parquet",
    # },
]

# ── COORDINATE TRANSFORMERS ────────────────────────────────────────────────────
_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
_to_wgs = Transformer.from_crs('EPSG:32610', 'EPSG:4326', always_xy=True)

# ── CURB RAMP SCHEMA CONSTANTS ─────────────────────────────────────────────────
_CURBRAMP_SIDES     = ('left', 'right')
_CURBRAMP_POSITIONS = ('start', 'end')
_CURBRAMP_INDICES   = (1, 2, 3)

# Pre-computed column name tuples for the 12 curb-ramp combinations.
# Each entry: (side, position, index, parsed_geom_col, id_col, returnloc_col,
#              returnposition_col, condition_score_col)
_CURBRAMP_SPECS: list[tuple[str, str, int, str, str, str, str, str]] = [
    (s, p, i,
     f'_p_sidewalk_{s}_curbramp_{p}_{i}_geometry',
     f'sidewalk_{s}_curbramp_{p}_{i}_ID',
     f'sidewalk_{s}_curbramp_{p}_{i}_returnloc',
     f'sidewalk_{s}_curbramp_{p}_{i}_returnposition',
     f'sidewalk_{s}_curbramp_{p}_{i}_condition_score')
    for s in _CURBRAMP_SIDES
    for p in _CURBRAMP_POSITIONS
    for i in _CURBRAMP_INDICES
]

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
    """Return [lat, lon] from a WGS84 point (x=lon, y=lat)."""
    return [y, x]


def parse_geom_series(series: pd.Series) -> pd.Series:
    """Vectorized geometry parsing using shapely bulk ops where possible.

    Detects column format from the first non-null value and dispatches to
    shapely.from_wkb (bytes) or shapely.from_wkt (WKT strings).  Falls back
    to row-by-row parse_geom for GeoJSON / WKB-hex / mixed formats.
    """
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series([None] * len(series), index=series.index, dtype=object)

    first = non_null.iloc[0]
    # to_numpy(dtype=object) gives a plain ndarray and converts NaN → None
    arr: 'np.ndarray[Any, np.dtype[np.object_]]' = series.to_numpy(dtype=object)

    # WKB bytes — typical for parquet-stored geometries
    if isinstance(first, bytes):
        return pd.Series(
            shapely.from_wkb(arr, on_invalid='warn'),  # type: ignore[arg-type]
            index=series.index, dtype=object,
        )

    # Already shapely geometry objects
    if hasattr(first, 'geom_type'):
        return series

    # WKT strings
    if isinstance(first, str):
        _WKT_PREFIXES = ('POINT', 'LINESTRING', 'POLYGON', 'MULTI', 'GEOMETRYCOLLECTION', 'LINEARRING')
        if first.strip().upper().startswith(_WKT_PREFIXES):
            return pd.Series(
                shapely.from_wkt(arr, on_invalid='warn'),  # type: ignore[arg-type]
                index=series.index, dtype=object,
            )

    # Fallback: row-by-row for GeoJSON / WKB-hex / mixed formats
    return series.apply(parse_geom)  # type: ignore[arg-type]


def geom_to_latlons(geom: BaseGeometry | None) -> list[list[float]]:
    """Extract [[lat, lon], ...] pairs from a WGS84 LineString/MultiLineString."""
    if geom is None:
        return []
    if isinstance(geom, LineString):
        return [[lat, lon] for lon, lat in geom.coords]
    if isinstance(geom, MultiLineString):
        return [[lat, lon] for line in geom.geoms for lon, lat in line.coords]
    return []


def polygon_exterior_latlons(poly: BaseGeometry) -> list[list[float]]:
    """Return [[lat, lon], ...] from a WGS84 Polygon exterior ring."""
    if isinstance(poly, Polygon) and not poly.is_empty:
        return [[lat, lon] for lon, lat in poly.exterior.coords]
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


def _is_offset(val):
    """Return True if an offset column value is truthy (True, 'yes', non-empty string)."""
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


def _parse_incline_series(series: pd.Series) -> pd.Series:
    """Vectorized version of _parse_incline — returns absolute float values, NaN for unparseable."""
    numeric = pd.to_numeric(series, errors='coerce').abs()
    needs_parse = numeric.isna() & series.notna()
    if needs_parse.any():
        cleaned = (
            series[needs_parse].astype(str)
            .str.strip()
            .str.replace('%', '', regex=False)
            .str.replace('+', '', regex=False)
        )
        numeric = numeric.copy()
        numeric[needs_parse] = pd.to_numeric(cleaned, errors='coerce').abs()
    return numeric


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
                 bar_pos: int = 0,
                 default_lane_width_m: float = 3.5,
                 hull_fallback_r: float = 15.0) -> str:
    """Build and save a folium map centered on (center_lat, center_lon).

    Returns the path to the saved HTML file.
    """
    # -- bounding box in WGS84 degrees (geometries are EPSG:4326)
    lat_deg = bbox_m / 111320
    lon_deg = bbox_m / (111320 * math.cos(math.radians(center_lat)))
    bbox    = box(center_lon - lon_deg, center_lat - lat_deg,
                  center_lon + lon_deg, center_lat + lat_deg)

    def in_bbox(geom):
        return geom is not None and bbox.intersects(geom)

    def draw_sidewalk(coords, row, side, label, target_sep, target_off):
        presence = row.get(f'sidewalk_{side}_presence', '') if side else ''
        width    = row.get(f'sidewalk_{side}_width', '')    if side else ''
        is_offset   = _is_offset(row.get(f'sidewalk_{side}_offset')) if side else False
        base_color  = '#add8e6' if is_offset else '#00008b'
        incline_raw = row.get(f'sidewalk_{side}_incline') if side else row.get('street_incline')
        color       = _seg_color(base_color, incline_raw)
        kind        = 'offset' if is_offset else 'separate'
        target   = target_off if is_offset else target_sep
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
        _sw_layer = folium.PolyLine(
            coords, color=color, weight=2, opacity=0.85,
            tooltip=f"Sidewalk {label} [{kind}]: presence={presence}, width={width}",
            popup=make_popup(f'Sidewalk ({label}) [{kind}]', attrs)
        ).add_to(target)
        add_endpoints(coords, color, target)
        _sw_id = str(row.get(f'sidewalk_{side}_grid_ID') or '') if side else str(row.get('street_grid_id') or '')
        if _sw_id and _sw_id not in _seg_index:
            _seg_index[_sw_id] = {'v': _sw_layer.get_name(), 'c': coords[len(coords) // 2], 'ow': 2, 'oc': color}

    # -- map + feature groups
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   max_zoom=22, tiles='CartoDB positron', zoom_control=False)

    fg_streets   = folium.FeatureGroup(name='Streets',              show=True)
    fg_sw_sep    = folium.FeatureGroup(name='Sidewalks – separate', show=True)
    fg_sw_off    = folium.FeatureGroup(name='Sidewalks – offset', show=True)
    fg_bk_sep    = folium.FeatureGroup(name='Bikeways – separate',  show=True)
    fg_bk_off    = folium.FeatureGroup(name='Bikeways – offset',  show=True)
    fg_nodes         = folium.FeatureGroup(name='Intersection Nodes',   show=True)
    fg_hulls         = folium.FeatureGroup(name='Intersection Hulls',   show=True)
    fg_slots         = folium.FeatureGroup(name='Crosswalk Slots',      show=True)
    fg_curbramps     = folium.FeatureGroup(name='Curb Ramps',           show=True)
    fg_crosswalks    = folium.FeatureGroup(name='Crosswalks',            show=True)
    fg_curb_returns  = folium.FeatureGroup(name='Curb Returns',         show=True)
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
        'sidewalk_offset': 0,
        'bikeway_separate': 0,
        'bikeway_offset': 0,
        'curbramp': 0,
        'crosswalk': 0,
        'curb_return': 0,
        'traffic_calming': 0,
        'intersection_node': 0,
        'hull': 0,
        'slot': 0,
    }

    _geom_cols_to_parse = (
        [f'sidewalk_{s}_geometry' for s in ('left', 'right')]
        + [f'bikeway_{s}_{n}_geometry' for s in ('left', 'right') for n in (1, 2)]
        + [_curbramp_col(s, p, i, 'geometry')
           for s in _CURBRAMP_SIDES for p in _CURBRAMP_POSITIONS for i in _CURBRAMP_INDICES]
        + ['street_feature_geometry']
        + ['crosswalk_start_geometry', 'crosswalk_end_geometry']
        + ['curb_return_geometry']
        + ['start_node_geometry', 'end_node_geometry']
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
        _parsed_street = parse_geom_series(data['street_geometry'])
        _in_bbox_mask  = pd.Series(
            shapely.intersects(_parsed_street.to_numpy(dtype=object), bbox),  # type: ignore[arg-type]
            index=data.index,
        ).astype(bool)
        data = data.loc[_in_bbox_mask].copy()
        data['_street_geom'] = _parsed_street[_in_bbox_mask]
        _present_geom_cols = [c for c in _geom_cols_to_parse if c in data.columns]
        pbar.total = 5 + len(_present_geom_cols) + len(data)
        pbar.refresh()
        pbar.update(1)

        # ── Stage 2: incline normalization ────────────────────────────────────
        pbar.set_description(f'{output_name} · inclines')
        _incline_vals: list[float] = []
        for _icol in _INCLINE_COLS:
            if _icol in data.columns:
                _incline_vals.extend(_parse_incline_series(data[_icol]).dropna().tolist())
        _incline_min = min(_incline_vals) if _incline_vals else 0.0
        _incline_max = max(_incline_vals) if _incline_vals else 1.0

        def _seg_color(base: str, raw_incline) -> str:
            parsed = _parse_incline(raw_incline)
            if parsed is None or _incline_max <= _incline_min:
                return base
            norm = (parsed - _incline_min) / (_incline_max - _incline_min)
            return _blend_with_black(base, 0.5 * norm)

        pbar.update(1)

        # ── Stage 3: pre-parse geometry columns (parallel, one thread per col) ──
        pbar.set_description(f'{output_name} · geometry cols')
        _col_results: dict[str, pd.Series] = {}
        with ThreadPoolExecutor() as _geom_ex:
            _geom_futures = {
                _geom_ex.submit(parse_geom_series, data[col]): col
                for col in _present_geom_cols
            }
            for _f in as_completed(_geom_futures):
                _col_results[_geom_futures[_f]] = _f.result()
                pbar.update(1)
        for _col, _result in _col_results.items():
            data[f'_p_{_col}'] = _result

        # ── Stage 3.5: intersection hulls and crosswalk slots ─────────────────
        pbar.set_description(f'{output_name} · hulls & slots')

        # Build a minimal GeoDataFrame for step_12/13/14 from pre-parsed cols
        _hull_geom_cols = [
            'start_node_geometry', 'end_node_geometry',
            'sidewalk_left_geometry', 'sidewalk_right_geometry',
            *[f'sidewalk_{s}_curbramp_{p}_{n}_geometry'
              for s in ('left', 'right') for p in ('start', 'end') for n in (1, 2, 3)]
        ]
        _hull_attr_cols = [
            'start_node_id', 'end_node_id',
            'start_node_is_intersection_node', 'end_node_is_intersection_node',
            'name', 'highway',
            'sidewalk_left_presence', 'sidewalk_right_presence',
        ]
        _hull_df = gpd.GeoDataFrame(index=data.index, geometry=data['_street_geom'], crs='EPSG:4326')
        _hull_df['street_geometry'] = data['_street_geom'].values
        _null_arr = [None] * len(data)
        for _col in _hull_geom_cols:
            _parsed = f'_p_{_col}'
            _hull_df[_col] = data[_parsed].values if _parsed in data.columns else _null_arr
        for _col in _hull_attr_cols:
            _hull_df[_col] = data[_col].values if _col in data.columns else _null_arr
        _, _hulls_gdf = _step_12(_hull_df, _DEFAULT_PIPELINE_CONFIG)
        _hulls_gdf = _step_13(_hulls_gdf, _DEFAULT_PIPELINE_CONFIG)
        _step_14(_hull_df, _hulls_gdf, _DEFAULT_PIPELINE_CONFIG)
        _slot_list = _hulls_gdf.attrs.get('crosswalk_slots', [])

        # Render hull polygons (amber, translucent) — hulls are WGS84 (lon, lat)
        for _hrow in _hulls_gdf.itertuples():
            _hull_geom = _hrow.geometry
            if not in_bbox(_hull_geom):
                continue
            if isinstance(_hull_geom, Polygon):
                _hcoords = [[c[1], c[0]] for c in _hull_geom.exterior.coords]
                if _hcoords:
                    _clon, _clat = _hull_geom.centroid.x, _hull_geom.centroid.y
                    folium.Polygon(
                        locations=_hcoords,
                        color='#b45309', weight=1.5, opacity=0.8,
                        fill=True, fill_color='#fbbf24', fill_opacity=0.15,
                        tooltip=f"Intersection hull ({_clon:.5f}, {_clat:.5f})",
                    ).add_to(fg_hulls)
                    counts['hull'] += 1

        # Render crosswalk slot zones (cyan, translucent) — slots are WGS84
        for _slot in _slot_list:
            _slot_poly = _slot.get('slot_geom')
            if not isinstance(_slot_poly, (Polygon, MultiPolygon)):
                continue
            if not in_bbox(_slot_poly):
                continue
            if not isinstance(_slot_poly, Polygon):
                continue
            _zcoords = [[c[1], c[0]] for c in _slot_poly.exterior.coords]
            if not _zcoords:
                continue
            folium.Polygon(
                locations=_zcoords,
                color='#0891b2', weight=1.5, opacity=0.8,
                fill=True, fill_color='#7dd3fc', fill_opacity=0.2,
                tooltip="Crosswalk slot",
            ).add_to(fg_slots)
            counts['slot'] += 1
        pbar.update(1)

        # ── Stage 4: pre-parse feature list columns ───────────────────────────
        pbar.set_description(f'{output_name} · feature lists')
        if 'street_feature_types' in data.columns:
            data['_feat_types'] = data['street_feature_types'].apply(_parse_list_col)
        if 'street_feature_attributes' in data.columns:
            data['_feat_attrs'] = data['street_feature_attributes'].apply(_parse_list_col)
        pbar.update(1)

        # ── Stage 5: build map elements (one step per row) ────────────────────
        _seg_index:  dict[str, dict] = {}
        _node_index: dict[str, dict] = {}
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
                        bk_buf   = _is_offset(row.get('bikeway_left_1_offset') or row.get('bikeway_right_1_offset'))
                        bk_color = '#90ee90' if bk_buf else '#006400'
                        bk_kind  = 'offset' if bk_buf else 'separate'
                        bk_target = fg_bk_off if bk_buf else fg_bk_sep
                        _cy_layer = folium.PolyLine(
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
                        counts['bikeway_offset' if bk_buf else 'bikeway_separate'] += 1
                        _cy_id = str(row.get('street_grid_id') or '')
                        if _cy_id and _cy_id not in _seg_index:
                            _seg_index[_cy_id] = {'v': _cy_layer.get_name(), 'c': coords[len(coords) // 2], 'ow': 2, 'oc': bk_color}
                    elif is_footway:
                        is_buf = _is_offset(row.get('sidewalk_offset', False))
                        draw_sidewalk(coords, row, None, f'footway – {name or "(unnamed)"}',
                                      fg_sw_sep, fg_sw_off)
                        counts['sidewalk_offset' if is_buf else 'sidewalk_separate'] += 1
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
                        _st_layer = folium.PolyLine(
                            coords, color=street_color, weight=3, opacity=0.85,
                            tooltip=(f"Street: {name} ({hw}) | "
                                     f"start={'⬟' if start_is_int else '·'} "
                                     f"end={'⬟' if end_is_int else '·'}"),
                            popup=make_popup(f'Street: {name or "(unnamed)"}', street_attrs)
                        ).add_to(fg_streets)
                        add_endpoints(coords, street_color, fg_streets)
                        counts['street'] += 1
                        _st_id = str(row.get('street_grid_id') or '')
                        if _st_id and _st_id not in _seg_index:
                            _seg_index[_st_id] = {'v': _st_layer.get_name(), 'c': coords[len(coords) // 2], 'ow': 3, 'oc': street_color}

                        # Intersection nodes (dark red)
                        _NODE_COLOR = '#8b0000'
                        if start_is_int:
                            _sn_id = row.get('start_node_id')
                            _sn_layer = folium.CircleMarker(
                                location=coords[0], radius=5,
                                color=_NODE_COLOR, fill=True,
                                fill_color=_NODE_COLOR, fill_opacity=1.0, weight=1,
                                tooltip=f"Intersection node (start): {_sn_id}",
                                popup=make_popup(f'Intersection Node: {_sn_id}', [
                                    ('Node ID',  _sn_id),
                                    ('Position', 'start'),
                                    ('Street',   row.get('name')),
                                ])
                            ).add_to(fg_nodes)
                            counts['intersection_node'] += 1
                            if _sn_id and str(_sn_id) not in _node_index:
                                _node_index[str(_sn_id)] = {'v': _sn_layer.get_name(), 'c': list(coords[0]), 'ow': 1, 'oc': _NODE_COLOR}
                        if end_is_int:
                            _en_id = row.get('end_node_id')
                            _en_layer = folium.CircleMarker(
                                location=coords[-1], radius=5,
                                color=_NODE_COLOR, fill=True,
                                fill_color=_NODE_COLOR, fill_opacity=1.0, weight=1,
                                tooltip=f"Intersection node (end): {_en_id}",
                                popup=make_popup(f'Intersection Node: {_en_id}', [
                                    ('Node ID',  _en_id),
                                    ('Position', 'end'),
                                    ('Street',   row.get('name')),
                                ])
                            ).add_to(fg_nodes)
                            counts['intersection_node'] += 1
                            if _en_id and str(_en_id) not in _node_index:
                                _node_index[str(_en_id)] = {'v': _en_layer.get_name(), 'c': list(coords[-1]), 'ow': 1, 'oc': _NODE_COLOR}

            # Sidewalks (blue)
            for side in ('left', 'right'):
                sw_geom = row.get(f'_p_sidewalk_{side}_geometry')
                if in_bbox(sw_geom):
                    coords = geom_to_latlons(sw_geom)
                    if coords:
                        is_sw_buf = _is_offset(row.get(f'sidewalk_{side}_offset'))
                        draw_sidewalk(coords, row, side, side, fg_sw_sep, fg_sw_off)
                        counts['sidewalk_offset' if is_sw_buf else 'sidewalk_separate'] += 1

            # Bikeways (green)
            for side in ('left', 'right'):
                bk_is_offset = _is_offset(row.get(f'bikeway_{side}_offset'))
                bk_color     = '#90ee90' if bk_is_offset else '#006400'
                bk_kind      = 'offset' if bk_is_offset else 'separate'
                bk_target    = fg_bk_off if bk_is_offset else fg_bk_sep
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
                            _bk_layer = folium.PolyLine(
                                coords, color=bk_draw_color, weight=2, opacity=0.85,
                                tooltip=f"Bikeway {side}-{num} [{bk_kind}]: {bk_type}",
                                popup=make_popup(f'Bikeway ({side}-{num}) [{bk_kind}]', bk_attrs)
                            ).add_to(bk_target)
                            add_endpoints(coords, bk_draw_color, bk_target)
                            counts['bikeway_offset' if bk_is_offset else 'bikeway_separate'] += 1
                            _bk_id = str(row.get(f'bikeway_{side}_{num}_grid_id') or '')
                            if _bk_id and _bk_id not in _seg_index:
                                _seg_index[_bk_id] = {'v': _bk_layer.get_name(), 'c': coords[len(coords) // 2], 'ow': 2, 'oc': bk_draw_color}

            # Curb Ramps (orange)
            for side, position, index, p_geom_col, id_col, rloc_col, rpos_col, cond_col in _CURBRAMP_SPECS:
                ramp_geom = row.get(p_geom_col)
                if ramp_geom is None or not in_bbox(ramp_geom):
                    continue
                if ramp_geom.geom_type != 'Point':
                    continue
                ramp_point = cast(Point, ramp_geom)
                lat_lon    = _utm_to_latlon(ramp_point.x, ramp_point.y)
                rloc       = row.get(rloc_col)
                rpos       = row.get(rpos_col)
                ramp_attrs = [
                    ('Ramp ID',          row.get(id_col)),
                    ('Side',             side),
                    ('Position',         position),
                    ('Index',            index),
                    ('Return direction', rloc),
                    ('Return position',  rpos),
                    ('Condition score',  row.get(cond_col)),
                    ('Street name',      row.get('name')),
                ]
                folium.CircleMarker(
                    location=lat_lon, radius=5,
                    color='orange', fill=True, fill_color='orange',
                    fill_opacity=0.9, weight=1,
                    tooltip=(f"Curb ramp {side}-{position}-{index}: "
                             f"returnloc={rloc}, returnpos={rpos}"),
                    popup=make_popup(f'Curb Ramp ({side} {position} #{index})', ramp_attrs)
                ).add_to(fg_curbramps)
                counts['curbramp'] += 1

            # Crosswalks (magenta)
            for cw_pos in ('start', 'end'):
                cw_geom = row.get(f'_p_crosswalk_{cw_pos}_geometry')
                if in_bbox(cw_geom):
                    cw_coords = geom_to_latlons(cw_geom)
                    if cw_coords:
                        cw_type = row.get(f'crosswalk_{cw_pos}_type', '')
                        cw_attrs = [
                            ('ID',           row.get(f'crosswalk_{cw_pos}_id')),
                            ('Position',     cw_pos),
                            ('Type',         cw_type),
                            ('Controlled',   row.get(f'crosswalk_{cw_pos}_controlled')),
                            ('Marked',       row.get(f'crosswalk_{cw_pos}_marked')),
                            ('Markings',     row.get(f'crosswalk_{cw_pos}_markings')),
                            ('Signals',      row.get(f'crosswalk_{cw_pos}_signals')),
                            ('Island',       row.get(f'crosswalk_{cw_pos}_island')),
                            ('Kerb',         row.get(f'crosswalk_{cw_pos}_kerb')),
                            ('Tactile',      row.get(f'crosswalk_{cw_pos}_tactile_paving')),
                            ('Street name',  row.get('name')),
                        ]
                        folium.PolyLine(
                            cw_coords, color='#ff00ff', weight=3, opacity=0.8,
                            dash_array='5 5',
                            tooltip=f"Crosswalk ({cw_pos}): {cw_type or '(implicit)'}",
                            popup=make_popup(f'Crosswalk ({cw_pos})', cw_attrs)
                        ).add_to(fg_crosswalks)
                        counts['crosswalk'] += 1

            # Curb Returns (teal)
            cr_geom = row.get('_p_curb_return_geometry')
            if in_bbox(cr_geom):
                cr_coords = geom_to_latlons(cr_geom)
                if cr_coords:
                    folium.PolyLine(
                        cr_coords, color='#008080', weight=2, opacity=0.7,
                        tooltip=f"Curb return: {row.get('name', '') or '(unnamed)'}",
                    ).add_to(fg_curb_returns)
                    counts['curb_return'] += 1

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
        for fg in (fg_bbox, fg_hulls, fg_slots, fg_streets, fg_bk_sep, fg_bk_off, fg_sw_sep, fg_sw_off, fg_curbramps, fg_crosswalks, fg_curb_returns, fg_traffic_calm, fg_nodes):
            fg.add_to(m)

        folium.Marker(
            [center_lat, center_lon],
            popup=output_name,
            icon=folium.Icon(color='orange', icon='map-marker')
        ).add_to(m)

        map_var      = m.get_name()
        js_street    = fg_streets.get_name()
        js_sw_sep    = fg_sw_sep.get_name()
        js_sw_buf    = fg_sw_off.get_name()
        js_bk_sep    = fg_bk_sep.get_name()
        js_bk_buf    = fg_bk_off.get_name()
        js_curbramps    = fg_curbramps.get_name()
        js_crosswalks   = fg_crosswalks.get_name()
        js_curb_returns = fg_curb_returns.get_name()
        js_traffic_calm = fg_traffic_calm.get_name()
        js_nodes        = fg_nodes.get_name()
        js_hulls        = fg_hulls.get_name()
        js_slots        = fg_slots.get_name()
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
    <input type="checkbox" id="cb_hulls" checked
           onchange="toggleFG('{js_hulls}', this.checked)">
    <span style="color:#b45309;font-size:18px;line-height:1">&#9647;</span>
    Intersection Hulls ({counts['hull']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_slots" checked
           onchange="toggleFG('{js_slots}', this.checked)">
    <span style="color:#0891b2;font-size:18px;line-height:1">&#9647;</span>
    Crosswalk Slots ({counts['slot']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_curbramps" checked
           onchange="toggleFG('{js_curbramps}', this.checked)">
    <span style="color:orange;font-size:18px;line-height:1">&#9679;</span>
    Curb Ramps ({counts['curbramp']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_crosswalks" checked
           onchange="toggleFG('{js_crosswalks}', this.checked)">
    <span style="color:#ff00ff;font-size:18px;line-height:1">&#9644;</span>
    Crosswalks ({counts['crosswalk']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_curb_returns" checked
           onchange="toggleFG('{js_curb_returns}', this.checked)">
    <span style="color:#008080;font-size:18px;line-height:1">&#9644;</span>
    Curb Returns ({counts['curb_return']})
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
    Sidewalks – offset ({counts['sidewalk_offset']})
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
    Bikeways – offset ({counts['bikeway_offset']})
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

        _search_html = f"""
<div id="px-search" style="
    position:fixed;top:70px;right:10px;z-index:9999;
    background:white;padding:10px 14px;border:2px solid #aaa;
    border-radius:6px;font-size:13px;font-family:sans-serif;line-height:1.6;">
  <b style="font-size:14px">Search</b><br>
  <div style="display:flex;gap:4px;margin:6px 0 4px;">
    <button id="px-mode-seg" onclick="pxSetMode('seg')"
            style="flex:1;padding:3px 8px;border:1px solid #888;border-radius:4px;cursor:pointer;background:#1a73e8;color:white;font-size:12px;">
      Segment ID
    </button>
    <button id="px-mode-node" onclick="pxSetMode('node')"
            style="flex:1;padding:3px 8px;border:1px solid #888;border-radius:4px;cursor:pointer;background:white;color:#333;font-size:12px;">
      Node ID
    </button>
  </div>
  <div style="display:flex;gap:4px;">
    <input id="px-search-input" type="text" placeholder="e.g. 12_34_0"
           style="flex:1;padding:4px 6px;border:1px solid #ccc;border-radius:4px;font-size:12px;min-width:140px;"
           onkeydown="if(event.key==='Enter')pxSearch()">
    <button onclick="pxSearch()"
            style="padding:4px 10px;border:1px solid #888;border-radius:4px;cursor:pointer;background:#f5f5f5;font-size:12px;">
      Go
    </button>
  </div>
  <div id="px-search-status" style="margin-top:5px;font-size:11px;color:#555;min-height:14px;"></div>
</div>
<script>
var _pxMode = 'seg';
var _pxSegIndex  = {json.dumps(_seg_index)};
var _pxNodeIndex = {json.dumps(_node_index)};
var _pxTimer = null;
function pxSetMode(mode) {{
  _pxMode = mode;
  var s = document.getElementById('px-mode-seg');
  var n = document.getElementById('px-mode-node');
  s.style.background = mode==='seg'  ? '#1a73e8' : 'white';
  s.style.color       = mode==='seg'  ? 'white'   : '#333';
  n.style.background = mode==='node' ? '#1a73e8' : 'white';
  n.style.color       = mode==='node' ? 'white'   : '#333';
  document.getElementById('px-search-input').placeholder = mode==='seg' ? 'e.g. 12_34_0' : 'e.g. 123456789';
  document.getElementById('px-search-status').textContent = '';
}}
function pxSearch() {{
  var id = document.getElementById('px-search-input').value.trim();
  if (!id) return;
  var idx = _pxMode === 'seg' ? _pxSegIndex : _pxNodeIndex;
  var entry = idx[id];
  var el = document.getElementById('px-search-status');
  if (!entry) {{ el.style.color='#c00'; el.textContent='Not found: '+id; return; }}
  el.style.color='#555'; el.textContent='\u2192 Found: '+id;
  var mapObj = window['{map_var}'];
  mapObj.flyTo(entry.c, 20);
  var layer = window[entry.v];
  if (!layer) return;
  if (_pxTimer) clearTimeout(_pxTimer);
  try {{ layer.setStyle({{color:'#ffff00', weight: entry.ow * 2.5}}); }} catch(e) {{}}
  _pxTimer = setTimeout(function() {{
    try {{ layer.setStyle({{color: entry.oc, weight: entry.ow}}); }} catch(e) {{}}
  }}, 2500);
  try {{ if (typeof layer.openPopup === 'function') layer.openPopup(entry.c); }} catch(e) {{}}
}}
</script>
"""
        m.get_root().html.add_child(folium.Element(_search_html))  # type: ignore[attr-defined]

        # --- Click-to-show-coordinates widget (bottom-right) ---
        _coord_html = f"""
<div id="px-coords" style="
    position:fixed;bottom:30px;right:30px;z-index:9999;
    background:rgba(255,255,255,0.92);padding:6px 12px;border-radius:6px;
    font:12px/1.4 monospace;color:#333;box-shadow:0 1px 4px rgba(0,0,0,0.2);
    pointer-events:none;display:none;">
</div>
<script>
document.addEventListener('DOMContentLoaded', function() {{
  var box = document.getElementById('px-coords');
  var map = window['{map_var}'];
  if (!map) return;
  map.on('click', function(e) {{
    box.style.display = 'block';
    box.textContent = e.latlng.lat.toFixed(7) + ', ' + e.latlng.lng.toFixed(7);
  }});
}});
</script>
"""
        m.get_root().html.add_child(folium.Element(_coord_html))  # type: ignore[attr-defined]

        # --- Click-to-probe nearby segments panel ---
        # Uses plain string + .replace() to avoid f-string escaping mangling JS
        _probe_html = """
<div id="px-probe" style="
    position:fixed;top:50px;left:10px;z-index:10002;
    background:white;border:2px solid #aaa;border-radius:6px;
    font:12px/1.5 sans-serif;color:#333;box-shadow:0 2px 6px rgba(0,0,0,0.2);
    max-width:280px;min-width:200px;display:none;">
  <div style="background:#1a73e8;color:white;padding:5px 10px;border-radius:4px 4px 0 0;
              display:flex;justify-content:space-between;align-items:center;">
    <b>Nearby Segments</b>
    <span id="px-probe-close" style="cursor:pointer;font-size:14px;line-height:1;">&times;</span>
  </div>
  <div style="padding:6px 10px;">
    <label style="font-size:11px;color:#666;">
      Radius:
      <input id="px-probe-radius" type="range" min="5" max="100" value="20"
             style="width:90px;vertical-align:middle;">
      <span id="px-probe-radius-val">20</span> m
    </label>
  </div>
  <div id="px-probe-list" style="padding:0 8px 8px;max-height:220px;overflow-y:auto;"></div>
</div>
<button id="px-probe-toggle" style="
    position:fixed;top:10px;left:10px;z-index:10001;
    padding:5px 12px;border:2px solid #aaa;border-radius:4px;cursor:pointer;
    background:white;color:#333;font:12px sans-serif;
    box-shadow:0 1px 4px rgba(0,0,0,0.15);">
  &#128269; Probe Segments
</button>
<script>
(function() {
  var _probeActive = false;
  var _probeTimer = {};

  function pxProbeToggle() {
    _probeActive = !_probeActive;
    var btn = document.getElementById('px-probe-toggle');
    btn.style.background  = _probeActive ? '#1a73e8' : 'white';
    btn.style.color       = _probeActive ? 'white'   : '#333';
    btn.style.borderColor = _probeActive ? '#1a73e8' : '#aaa';
    btn.textContent       = _probeActive ? '\\u{1F50D} Probe ON \\u2014 click map' : '\\u{1F50D} Probe Segments';
    if (!_probeActive) {
      document.getElementById('px-probe').style.display = 'none';
    }
  }

  document.getElementById('px-probe-toggle').addEventListener('click', function(e) {
    e.stopPropagation();
    pxProbeToggle();
  });

  function haverDist(lat1, lon1, lat2, lon2) {
    var R = 6371000;
    var dLat = (lat2-lat1)*Math.PI/180;
    var dLon = (lon2-lon1)*Math.PI/180;
    var a = Math.sin(dLat/2)*Math.sin(dLat/2) +
            Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*
            Math.sin(dLon/2)*Math.sin(dLon/2);
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  }

  function flashSegment(segId) {
    var entry = _pxSegIndex[segId];
    if (!entry) return;
    var layer = window[entry.v];
    if (!layer) return;
    if (_probeTimer[segId]) clearTimeout(_probeTimer[segId]);
    try { layer.setStyle({color:'#ff4400', weight: entry.ow * 3}); } catch(e) {}
    _probeTimer[segId] = setTimeout(function() {
      try { layer.setStyle({color: entry.oc, weight: entry.ow}); } catch(e) {}
    }, 2500);
  }

  var _hidden = {};
  function hideSegment(segId) {
    var entry = _pxSegIndex[segId];
    if (!entry) return;
    var layer = window[entry.v];
    if (!layer) return;
    if (_hidden[segId]) {
      try { layer.setStyle({opacity:1, weight: entry.ow}); } catch(e) {}
      delete _hidden[segId];
    } else {
      try { layer.setStyle({opacity:0, weight:0}); } catch(e) {}
      _hidden[segId] = true;
    }
    // Update button text in probe list
    var btn = document.getElementById('px-hide-' + segId);
    if (btn) btn.textContent = _hidden[segId] ? 'Show' : 'Hide';
  }

  function probeAt(latlng, radiusM) {
    var found = [];
    for (var id in _pxSegIndex) {
      var e = _pxSegIndex[id];
      var d = haverDist(latlng.lat, latlng.lng, e.c[0], e.c[1]);
      if (d <= radiusM) found.push({id: id, dist: d, entry: e});
    }
    found.sort(function(a,b){ return a.dist - b.dist; });
    return found;
  }

  function renderProbe(found) {
    var panel = document.getElementById('px-probe');
    var list  = document.getElementById('px-probe-list');
    if (found.length === 0) {
      list.innerHTML = '<div style="color:#999;font-size:11px;padding:4px 2px;">No segments found</div>';
    } else {
      list.innerHTML = found.map(function(f) {
        var col = f.entry.oc || '#333';
        return '<div style="display:flex;align-items:center;gap:6px;padding:3px 2px;border-bottom:1px solid #eee;">' +
          '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;flex-shrink:0;background:' + col + '"></span>' +
          '<code style="flex:1;font-size:11px;user-select:all;">' + f.id + '</code>' +
          '<span style="font-size:10px;color:#999;">' + Math.round(f.dist) + 'm</span>' +
          '<button onclick="flashSegment(\\'' + f.id + '\\')" style="font-size:10px;padding:1px 5px;cursor:pointer;border:1px solid #ccc;border-radius:3px;background:#f5f5f5;">Flash</button>' +
          '<button id="px-hide-' + f.id + '" onclick="hideSegment(\\'' + f.id + '\\')" style="font-size:10px;padding:1px 5px;cursor:pointer;border:1px solid #ccc;border-radius:3px;background:#f5f5f5;">' + (_hidden[f.id] ? 'Show' : 'Hide') + '</button>' +
        '</div>';
      }).join('');
    }
    panel.style.display = 'block';
  }

  function pxProbeInit() {
    var mapObj = window['%%MAP_VAR%%'];
    if (!mapObj) { setTimeout(pxProbeInit, 100); return; }

    var radiusInput = document.getElementById('px-probe-radius');
    var radiusVal   = document.getElementById('px-probe-radius-val');
    radiusInput.addEventListener('input', function() {
      radiusVal.textContent = radiusInput.value;
    });

    mapObj.on('click', function(e) {
      if (!_probeActive) return;
      var r = parseInt(radiusInput.value, 10);
      var found = probeAt(e.latlng, r);
      renderProbe(found);
    });

    document.getElementById('px-probe-close').addEventListener('click', function() {
      document.getElementById('px-probe').style.display = 'none';
      _probeActive = false;
      var btn = document.getElementById('px-probe-toggle');
      btn.style.background  = 'white';
      btn.style.color       = '#333';
      btn.style.borderColor = '#aaa';
      btn.textContent       = '\\u{1F50D} Probe Segments';
    });
  }
  setTimeout(pxProbeInit, 0);

  window.flashSegment = flashSegment;
  window.hideSegment = hideSegment;
})();
</script>
""".replace("%%MAP_VAR%%", map_var)
        m.get_root().html.add_child(folium.Element(_probe_html))  # type: ignore[attr-defined]

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{output_name}.html")
        m.save(out_path)
        pbar.update(1)

    return out_path


# ── COUNTY MAP (DuckDB WASM) ────────────────────────────────────────────────────
# Serve the Output/ directory with `python -m http.server 8080 --bind 127.0.0.1` and open
# http://localhost:8080/test_maps/<output_name>.html
# DuckDB WASM queries the parquet directly via HTTP range requests.
_COUNTY_MAP_TEMPLATE = """\
<!-- Serve: cd Output && python -m http.server 8080 --bind 127.0.0.1 -->
<!-- Open:  http://localhost:8080/test_maps/%%TITLE%%.html   -->
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>%%TITLE%%</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
#map { width: 100%; height: 100vh; }
#status {
  position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
  z-index: 9999; background: white; padding: 5px 14px;
  border: 1px solid #ccc; border-radius: 4px;
  font-family: sans-serif; font-size: 12px; white-space: nowrap;
  pointer-events: none;
}
#error-banner {
  display: none; position: fixed; top: 0; left: 0; right: 0;
  background: #b00; color: white; padding: 10px 16px;
  z-index: 10000; font-family: sans-serif; font-size: 13px; text-align: center;
}
#legend {
  position: fixed; bottom: 30px; left: 30px; z-index: 9999;
  background: white; padding: 10px 14px; border: 2px solid #aaa;
  border-radius: 6px; font-size: 13px; font-family: sans-serif; line-height: 1.9;
}
#px-search {
  position: fixed; top: 70px; right: 10px; z-index: 9999;
  background: white; padding: 10px 14px; border: 2px solid #aaa;
  border-radius: 6px; font-size: 13px; font-family: sans-serif; line-height: 1.6;
}
#px-probe-toggle {
  position: fixed; top: 10px; left: 10px; z-index: 10001;
  padding: 5px 12px; border: 2px solid #aaa; border-radius: 4px; cursor: pointer;
  background: white; color: #333; font: 12px sans-serif;
  box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}
#px-probe {
  position: fixed; top: 42px; left: 10px; z-index: 10002;
  background: white; border: 2px solid #aaa; border-radius: 6px;
  font: 12px/1.5 sans-serif; color: #333; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
  max-width: 280px; min-width: 200px; display: none;
}
</style>
</head>
<body>
<div id="error-banner"></div>
<div id="status">Initializing DuckDB WASM\u2026</div>
<div id="map"></div>

<div id="px-coords" style="
    position:fixed;bottom:30px;right:30px;z-index:9999;
    background:rgba(255,255,255,0.92);padding:6px 12px;border-radius:6px;
    font:12px/1.4 monospace;color:#333;box-shadow:0 1px 4px rgba(0,0,0,0.2);
    pointer-events:none;display:none;">
</div>

<button id="px-probe-toggle">&#128269; Probe</button>

<div id="px-probe">
  <div style="background:#1a73e8;color:white;padding:5px 10px;border-radius:4px 4px 0 0;
              display:flex;justify-content:space-between;align-items:center;">
    <b>Probe</b>
    <span id="px-probe-close" style="cursor:pointer;font-size:14px;line-height:1;">&times;</span>
  </div>
  <div style="padding:6px 10px;">
    <label style="font-size:11px;color:#666;">
      Radius:
      <input id="px-probe-radius" type="range" min="5" max="100" value="20"
             style="width:90px;vertical-align:middle;">
      <span id="px-probe-radius-val">20</span> m
    </label>
  </div>
  <div style="padding:0 8px 4px;">
    <b style="font-size:11px;color:#555;">Nearby Segments</b>
    <div id="px-probe-segs" style="max-height:200px;overflow-y:auto;margin-top:2px;"></div>
  </div>
  <hr style="margin:4px 8px;border:none;border-top:1px solid #ddd;">
  <div style="padding:0 8px 8px;">
    <b style="font-size:11px;color:#555;">Nearby Points</b>
    <div id="px-probe-pts" style="max-height:160px;overflow-y:auto;margin-top:2px;"></div>
  </div>
</div>

<div id="legend">
  <b style="font-size:14px">Legend</b><br>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_nodes" checked>
    <span style="color:#8b0000;font-size:18px;line-height:1">&#9679;</span> Intersection Nodes
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_ramps" checked>
    <span style="color:#ff8c00;font-size:18px;line-height:1">&#9679;</span> Curb Ramps
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_calm" checked>
    <span style="color:#6a0dad;font-size:18px;line-height:1">&#9679;</span> Traffic Calming
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_sw_off" checked>
    <span style="color:#add8e6;font-size:18px;line-height:1">&#9644;</span> Sidewalks \u2013 offset
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_sw_sep" checked>
    <span style="color:#00008b;font-size:18px;line-height:1">&#9644;</span> Sidewalks \u2013 separate
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_bk_off" checked>
    <span style="color:#90ee90;font-size:18px;line-height:1">&#9644;</span> Bikeways \u2013 offset
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_bk_sep" checked>
    <span style="color:#006400;font-size:18px;line-height:1">&#9644;</span> Bikeways \u2013 separate
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_streets" checked>
    <span style="color:#c0392b;font-size:18px;line-height:1">&#9644;</span> Streets
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_slots" checked>
    <span style="color:#0891b2;font-size:16px;line-height:1">&#9634;</span> Crosswalk Slots
  </label>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_hulls" checked>
    <span style="color:#b45309;font-size:16px;line-height:1">&#9634;</span> Intersection Hulls
  </label>
</div>

<div id="px-search">
  <b style="font-size:14px">Search</b><br>
  <div style="display:flex;gap:4px;margin:6px 0 4px;">
    <button id="px-mode-seg" style="flex:1;padding:3px 8px;border:1px solid #888;border-radius:4px;cursor:pointer;background:#1a73e8;color:white;font-size:12px;">
      Segment ID
    </button>
    <button id="px-mode-node" style="flex:1;padding:3px 8px;border:1px solid #888;border-radius:4px;cursor:pointer;background:white;color:#333;font-size:12px;">
      Node ID
    </button>
  </div>
  <div style="display:flex;gap:4px;">
    <input id="px-search-input" type="text" placeholder="e.g. 12_34_0"
           style="flex:1;padding:4px 6px;border:1px solid #ccc;border-radius:4px;font-size:12px;min-width:140px;"
           onkeydown="if(event.key==='Enter')pxSearch()">
    <button onclick="pxSearch()"
            style="padding:4px 10px;border:1px solid #888;border-radius:4px;cursor:pointer;background:#f5f5f5;font-size:12px;">
      Go
    </button>
  </div>
  <div id="px-search-status" style="margin-top:5px;font-size:11px;color:#555;min-height:14px;"></div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/proj4@2.11.0/dist/proj4.min.js"></script>
<script type="module">
import * as duckdb from 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.29.0/+esm';

const PARQUET_URL = new URL('%%PARQUET_URL%%', window.location.href).href;
const HULL_FC = %%HULL_GEOJSON%%;
const SLOT_FC = %%SLOT_GEOJSON%%;

// ── Coordinate transforms (all parquet geometries are EPSG:32610 UTM Zone 10N) ──
proj4.defs('EPSG:32610', '+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs');

function viewportToUTM(bounds) {
  const sw = proj4('EPSG:4326', 'EPSG:32610', [bounds.getWest(),  bounds.getSouth()]);
  const ne = proj4('EPSG:4326', 'EPSG:32610', [bounds.getEast(),  bounds.getNorth()]);
  return { minX: sw[0], minY: sw[1], maxX: ne[0], maxY: ne[1] };
}

// Recursively transform GeoJSON coordinates from UTM to WGS84 lon/lat
function xfGeom(g) {
  if (!g) return null;
  const xf = c => proj4('EPSG:32610', 'EPSG:4326', c);
  if (g.type === 'Point')           return { type: 'Point',           coordinates: xf(g.coordinates) };
  if (g.type === 'LineString')      return { type: 'LineString',      coordinates: g.coordinates.map(xf) };
  if (g.type === 'MultiLineString') return { type: 'MultiLineString', coordinates: g.coordinates.map(r => r.map(xf)) };
  if (g.type === 'MultiPoint')      return { type: 'MultiPoint',      coordinates: g.coordinates.map(xf) };
  return g;
}

// ── Incline-based colour darkening (mirrors _blend_with_black in Python) ─────
function blendBlack(hex, opacity) {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
  const f = 1 - Math.max(0, Math.min(1, opacity));
  return '#' + [r,g,b].map(v => Math.round(v*f).toString(16).padStart(2,'0')).join('');
}
function inclineColor(base, raw) {
  const v = Math.abs(parseFloat(raw));
  if (isNaN(v)) return base;
  return blendBlack(base, Math.min(v / 30, 1) * 0.5);
}

// ── Colours (matching generate_map) ──────────────────────────────────────────
const C = {
  street:  '#c0392b',
  bk_sep:  '#006400', bk_off: '#90ee90',
  sw_sep:  '#00008b', sw_off: '#add8e6',
  node:    '#8b0000',
  ramp:    '#ff8c00',
  calm:    '#6a0dad',
  cross:   '#ff00ff',
  cret:    '#008080',
};

// ── Zoom tier thresholds ──────────────────────────────────────────────────────
function getTier(zoom) {
  if (zoom >= 18) return 5;
  if (zoom >= 17) return 4;
  if (zoom >= 15) return 3;
  if (zoom >= 13) return 2;
  return 1;
}

const HW_RE = {
  1: "(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link)",
  2: "(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|busway|cycleway)",
};

// ── Leaflet map ───────────────────────────────────────────────────────────────
const map = L.map('map').setView([37.7749, -122.4194], 12);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 20
}).addTo(map);

const lg = {
  streets: L.layerGroup().addTo(map),
  bk_sep:  L.layerGroup().addTo(map),
  bk_off:  L.layerGroup().addTo(map),
  sw_sep:  L.layerGroup().addTo(map),
  sw_off:  L.layerGroup().addTo(map),
  nodes:   L.layerGroup().addTo(map),
  ramps:   L.layerGroup().addTo(map),
  calm:    L.layerGroup().addTo(map),
  hulls:   L.layerGroup().addTo(map),
  slots:   L.layerGroup().addTo(map),
};

// ── Search index ─────────────────────────────────────────────────────────────
const segIndex   = new Map();   // street_grid_id / sidewalk_*_ID / bk_* → { layer, mid, baseColor, baseWeight, label }
const nodeIndex  = new Map();   // node_id (string) → { mid }
const pointIndex = new Map();   // node_* / ramp_* → { layer, lat, lon, label, baseColor, baseRadius }
let   _pxMode    = 'seg';

// ── Popup builder (matches generate_map style) ────────────────────────────────
function makePopup(title, attrs) {
  const fmt = v => (v == null || String(v).trim() === '' || String(v) === 'nan' || String(v) === 'None')
    ? '<i>\u2014</i>' : v;
  const rows = attrs.map(([k,v]) =>
    `<tr><td style="padding:2px 8px 2px 0;color:#555;white-space:nowrap"><b>${k}</b></td>` +
    `<td style="padding:2px 0">${fmt(v)}</td></tr>`).join('');
  return `<div style="font-family:sans-serif;font-size:12px;min-width:180px">` +
         `<b style="font-size:13px">${title}</b>` +
         `<table style="border-collapse:collapse;margin-top:4px">${rows}</table></div>`;
}

// ── Coordinate on click ────────────────────────────────────────────────────────
const _coordBox = document.getElementById('px-coords');
map.on('click', e => {
  _coordBox.style.display = 'block';
  _coordBox.textContent = e.latlng.lat.toFixed(7) + ', ' + e.latlng.lng.toFixed(7);
  if (_probeActive) {
    const r = parseInt(_probeRadiusInput.value, 10);
    renderProbe(probeSegments(e.latlng, r), probePoints(e.latlng, r));
  }
});
map.on('contextmenu', () => { _coordBox.style.display = 'none'; });

// ── Hull and slot rendering (pre-computed, viewport-filtered, zoom-gated) ──────
function renderHullsAndSlots() {
  lg.hulls.clearLayers();
  lg.slots.clearLayers();
  if (map.getZoom() < 17) return;
  const b = map.getBounds();
  const inView = f => {
    const ring = f.geometry.coordinates[0];
    return ring.some(([lon, lat]) => b.contains([lat, lon]));
  };
  L.geoJSON({ type: 'FeatureCollection', features: HULL_FC.features.filter(inView) }, {
    style: () => ({ color: '#b45309', weight: 1.5, opacity: 0.8, fill: true, fillColor: '#fbbf24', fillOpacity: 0.15 }),
    onEachFeature: (f, l) => l.bindTooltip(f.properties.tip),
  }).addTo(lg.hulls);
  L.geoJSON({ type: 'FeatureCollection', features: SLOT_FC.features.filter(inView) }, {
    style: () => ({ color: '#0891b2', weight: 1.5, opacity: 0.8, fill: true, fillColor: '#7dd3fc', fillOpacity: 0.2 }),
    onEachFeature: (f, l) => l.bindTooltip(f.properties.tip),
  }).addTo(lg.slots);
}

// ── Layer rendering ───────────────────────────────────────────────────────────
function clearAll() {
  ['streets','bk_sep','bk_off','sw_sep','sw_off','nodes','ramps','calm'].forEach(k => lg[k].clearLayers());
  segIndex.clear();
  nodeIndex.clear();
  pointIndex.clear();
}

function midLatLon(geojson) {
  // Return approximate [lat, lon] midpoint of a GeoJSON geometry (already WGS84)
  const coords = geojson.type === 'LineString'      ? geojson.coordinates
               : geojson.type === 'MultiLineString' ? geojson.coordinates.flat()
               : geojson.type === 'Point'           ? [geojson.coordinates]
               : geojson.type === 'MultiPoint'      ? geojson.coordinates
               : [];
  if (!coords.length) return [0, 0];
  const mid = coords[Math.floor(coords.length / 2)];
  return [mid[1], mid[0]];  // [lat, lon]
}

function renderRows(rows, tier) {
  clearAll();
  for (const r of rows) {
    // ── Streets ──────────────────────────────────────────────────────────────
    if (r.street_geom) {
      const g    = xfGeom(JSON.parse(r.street_geom));
      const col  = inclineColor(C.street, r.street_incline);
      const opts = { style: () => ({ color: col, weight: tier >= 2 ? 2 : 1.5, opacity: 0.8 }) };
      const layer = L.geoJSON(g, opts);
      if (tier >= 2) {
        layer.bindTooltip(String(r.name || r.highway || ''));
        layer.bindPopup(makePopup('Street', [
          ['Name',      r.name],       ['Highway',  r.highway],
          ['Grid ID',   r.street_grid_id], ['Incline', r.street_incline],
          ['Lanes',     r.lanes],       ['Surface',  r.surface],
          ['Max speed', r.maxspeed],    ['Oneway',   r.oneway],
        ]));
        const mid = midLatLon(g);
        segIndex.set(String(r.street_grid_id), { layer, mid, baseColor: col, baseWeight: 2, label: String(r.name || r.highway || r.street_grid_id) });
      }
      layer.addTo(lg.streets);
    }

    if (tier >= 2) {
      // ── Bikeways ────────────────────────────────────────────────────────────
      for (const [key, side, idx] of [
        ['bk_l1_geom','left',1], ['bk_l2_geom','left',2],
        ['bk_r1_geom','right',1],['bk_r2_geom','right',2],
      ]) {
        if (!r[key]) continue;
        const isOff = r[`bikeway_${side}_${idx}_offset`] === 'yes';
        const base  = isOff ? C.bk_off : C.bk_sep;
        const col   = inclineColor(base, r[`bikeway_${side}_${idx}_incline`]);
        const g     = xfGeom(JSON.parse(r[key]));
        const bkMid = midLatLon(g);
        const bkKey = `bk_${side}_${idx}_${r.street_grid_id}`;
        const layer = L.geoJSON(g, { style: () => ({ color: col, weight: 2, opacity: 0.85 }) })
          .bindTooltip(`Bikeway ${side}-${idx}: ${r[`bikeway_${side}_${idx}_type`] || ''}`)
          .bindPopup(makePopup(`Bikeway (${side} ${idx})`, [
            ['Type',      r[`bikeway_${side}_${idx}_type`]],
            ['Offset',    r[`bikeway_${side}_${idx}_offset`]],
            ['Width (m)', r[`bikeway_${side}_${idx}_width`]],
            ['Surface',   r[`bikeway_${side}_${idx}_surface`]],
            ['Incline',   r[`bikeway_${side}_${idx}_incline`]],
            ['Permitted', r[`bikeway_${side}_${idx}_permitted`]],
            ['Separator', r[`bikeway_${side}_${idx}_seperator`]],
            ['Street',    r.name],
          ]));
        segIndex.set(bkKey, { layer, mid: bkMid, baseColor: col, baseWeight: 2, label: `Bikeway ${side}-${idx}: ${r[`bikeway_${side}_${idx}_type`] || ''} (${r.name || ''})` });
        layer.addTo(isOff ? lg.bk_off : lg.bk_sep);
      }
    }

    if (tier >= 3) {
      // ── Sidewalks ────────────────────────────────────────────────────────────
      for (const [key, side] of [['sw_l_geom','left'],['sw_r_geom','right']]) {
        if (!r[key]) continue;
        const isOff = r[`sidewalk_${side}_offset`] === 'yes';
        const base  = isOff ? C.sw_off : C.sw_sep;
        const col   = inclineColor(base, r[`sidewalk_${side}_incline`]);
        const g     = xfGeom(JSON.parse(r[key]));
        const layer = L.geoJSON(g, { style: () => ({ color: col, weight: 2, opacity: 0.85 }) })
          .bindTooltip(`Sidewalk ${side}: presence=${r[`sidewalk_${side}_presence`] || ''}`)
          .bindPopup(makePopup(`Sidewalk (${side})`, [
            ['ID',        r[`sidewalk_${side}_ID`]],
            ['Presence',  r[`sidewalk_${side}_presence`]],
            ['Surface',   r[`sidewalk_${side}_surface`]],
            ['Quality',   r[`sidewalk_${side}_quality`]],
            ['Width (m)', r[`sidewalk_${side}_width`]],
            ['Incline',   r[`sidewalk_${side}_incline`]],
            ['Separator', r[`sidewalk_${side}_seperator`]],
            ['Offset',    r[`sidewalk_${side}_offset`]],
            ['Street',    r.name],
          ]));
        const mid = midLatLon(g);
        const swId = String(r[`sidewalk_${side}_ID`] || '');
        if (swId) segIndex.set(swId, { layer, mid, baseColor: col, baseWeight: 2, label: `Sidewalk ${side}: ${r.name || swId}` });
        layer.addTo(isOff ? lg.sw_off : lg.sw_sep);
      }

      // ── Crosswalks ────────────────────────────────────────────────────────────
      for (const [key, pos] of [['xw_start_geom','start'],['xw_end_geom','end']]) {
        if (!r[key]) continue;
        const g = xfGeom(JSON.parse(r[key]));
        L.geoJSON(g, { style: () => ({ color: C.cross, weight: 2, opacity: 0.8 }) })
          .bindTooltip(`Crosswalk ${pos}`)
          .bindPopup(makePopup(`Crosswalk (${pos})`, [
            ['ID',         r[`crosswalk_${pos}_id`]],
            ['Type',       r[`crosswalk_${pos}_type`]],
            ['Controlled', r[`crosswalk_${pos}_controlled`]],
            ['Marked',     r[`crosswalk_${pos}_marked`]],
            ['Signals',    r[`crosswalk_${pos}_signals`]],
          ]))
          .addTo(lg.streets);
      }

      // ── Curb returns ──────────────────────────────────────────────────────────
      if (r.curb_return_geom) {
        const g = xfGeom(JSON.parse(r.curb_return_geom));
        L.geoJSON(g, { style: () => ({ color: C.cret, weight: 2, opacity: 0.7 }) })
          .addTo(lg.streets);
      }

      // ── Traffic calming ───────────────────────────────────────────────────────
      if (r.feat_geom && r.street_feature_types) {
        try {
          const g     = xfGeom(JSON.parse(r.feat_geom));
          const types = JSON.parse(r.street_feature_types.replace(/'/g, '"'));
          const attrs = r.street_feature_attributes
            ? JSON.parse(r.street_feature_attributes.replace(/'/g, '"')) : [];
          const coords = g.type === 'MultiPoint' ? g.coordinates : [g.coordinates];
          coords.forEach((c, i) => {
            if (!String(types[i] || '').startsWith('traffic_calming')) return;
            L.circleMarker([c[1], c[0]], {
              radius: 5, color: C.calm, fillColor: C.calm, fillOpacity: 0.85, weight: 1.5,
            })
              .bindTooltip(`Traffic calming: ${types[i]}`)
              .bindPopup(makePopup('Traffic Calming', [
                ['Street', r.name], ['Type', types[i]],
                ...(typeof attrs[i] === 'object' && attrs[i]
                    ? Object.entries(attrs[i]) : []),
              ]))
              .addTo(lg.calm);
          });
        } catch(_) {}
      }
    }

    if (tier >= 4) {
      // ── Intersection nodes ────────────────────────────────────────────────────
      for (const [key, idField, intField] of [
        ['start_node_geom', 'start_node_id', 'start_node_is_intersection_node'],
        ['end_node_geom',   'end_node_id',   'end_node_is_intersection_node'],
      ]) {
        if (!r[key] || !r[intField]) continue;
        const g   = xfGeom(JSON.parse(r[key]));
        const mid = [g.coordinates[1], g.coordinates[0]];
        const nid = String(r[idField]);
        if (!nodeIndex.has(nid)) {
          const nodeMk = L.circleMarker(mid, {
            radius: 4, color: C.node, fillColor: C.node, fillOpacity: 0.9, weight: 1,
          })
            .bindTooltip(`Node ${nid}`)
            .addTo(lg.nodes);
          nodeIndex.set(nid, { mid });
          pointIndex.set(`node_${nid}`, { layer: nodeMk, lat: mid[0], lon: mid[1], label: `Node ${nid}`, baseColor: C.node, baseRadius: 4 });
        }
      }
    }

    if (tier >= 5) {
      // ── Curb ramps ────────────────────────────────────────────────────────────
      for (const s of ['left','right']) for (const p of ['start','end']) for (const i of [1,2,3]) {
        const key  = `cr_${s}_${p}_${i}_geom`;
        if (!r[key]) continue;
        const base = `sidewalk_${s}_curbramp_${p}_${i}`;
        const g    = xfGeom(JSON.parse(r[key]));
        const mid  = [g.coordinates[1], g.coordinates[0]];
        const rloc = r[`${base}_returnloc`];
        const rpos = r[`${base}_returnposition`];
        const rampKey = `ramp_${s}_${p}_${i}_${r.street_grid_id}`;
        const rampMk = L.circleMarker(mid, {
          radius: 5, color: C.ramp, fillColor: C.ramp, fillOpacity: 0.9, weight: 1,
        })
          .bindTooltip(`Curb ramp ${s}-${p}-${i}: loc=${rloc || ''}, pos=${rpos || ''}`)
          .bindPopup(makePopup(`Curb Ramp (${s} ${p} #${i})`, [
            ['ID',               r[`${base}_ID`]],
            ['Side',             s],
            ['Position',         p],
            ['Index',            i],
            ['Return direction', rloc],
            ['Return position',  rpos],
            ['Condition score',  r[`${base}_condition_score`]],
            ['Street',           r.name],
          ]))
          .addTo(lg.ramps);
        pointIndex.set(rampKey, { layer: rampMk, lat: mid[0], lon: mid[1], label: `Ramp ${s}-${p}-${i} (${r.name || ''})`, baseColor: C.ramp, baseRadius: 5 });
      }
    }
  }
}

// ── SQL builder ───────────────────────────────────────────────────────────────
function buildSQL(tier, bounds) {
  // Helper: CASE-guard geometry columns (null-safe) → GeoJSON alias
  // Secondary geometry columns are stored as WKB hex (VARCHAR); decode before ST_AsGeoJSON.
  const wkb  = (col, alias) =>
    `CASE WHEN ${col} IS NOT NULL THEN ST_AsGeoJSON(ST_GeomFromHEXWKB(CAST(${col} AS VARCHAR))) END AS ${alias}`;
  const bk   = (s, n) => wkb(`bikeway_${s}_${n}_geometry`,             `bk_${s[0]}${n}_geom`);
  const sw   = s      => wkb(`sidewalk_${s}_geometry`,                  `sw_${s[0]}_geom`);
  const xw   = p      => wkb(`crosswalk_${p}_geometry`,                 `xw_${p}_geom`);
  const nd   = s      => wkb(`${s}_node_geometry`,                      `${s}_node_geom`);
  const cr   = (s,p,i)=> wkb(`sidewalk_${s}_curbramp_${p}_${i}_geometry`,`cr_${s}_${p}_${i}_geom`);

  const cols = [
    'street_grid_id', 'highway',
    'ST_AsGeoJSON(street_geometry) AS street_geom',
  ];

  if (tier >= 2) {
    cols.push(
      'name','lanes','lane_width','surface','maxspeed','oneway','street_incline',
      ...['left','right'].flatMap(s => [1,2].flatMap(n => [
        `bikeway_${s}_${n}_type`,`bikeway_${s}_${n}_offset`,`bikeway_${s}_${n}_incline`,
        `bikeway_${s}_${n}_permitted`,`bikeway_${s}_${n}_width`,`bikeway_${s}_${n}_seperator`,
        bk(s, n),
      ]))
    );
  }

  if (tier >= 3) {
    cols.push(
      ...['left','right'].flatMap(s => [
        `sidewalk_${s}_ID`,`sidewalk_${s}_presence`,`sidewalk_${s}_surface`,
        `sidewalk_${s}_quality`,`sidewalk_${s}_width`,`sidewalk_${s}_incline`,
        `sidewalk_${s}_seperator`,`sidewalk_${s}_offset`, sw(s),
      ]),
      ...['start','end'].flatMap(p => [
        `crosswalk_${p}_id`,`crosswalk_${p}_type`,`crosswalk_${p}_controlled`,
        `crosswalk_${p}_marked`,`crosswalk_${p}_signals`, xw(p),
      ]),
      wkb('curb_return_geometry', 'curb_return_geom'),
      'street_feature_types','street_feature_attributes',
      wkb('street_feature_geometry','feat_geom'),
      'sidewalk_left_feature_types',
      wkb('sidewalk_left_feature_geometry','sw_l_feat_geom'),
      'sidewalk_right_feature_types',
      wkb('sidewalk_right_feature_geometry','sw_r_feat_geom'),
    );
  }

  if (tier >= 4) {
    cols.push(
      'start_node_id','start_node_is_intersection_node', nd('start'),
      'end_node_id',  'end_node_is_intersection_node',   nd('end'),
    );
  }

  if (tier >= 5) {
    for (const s of ['left','right'])
      for (const p of ['start','end'])
        for (const i of [1,2,3]) {
          const base = `sidewalk_${s}_curbramp_${p}_${i}`;
          cols.push(
            `${base}_ID`,`${base}_returnloc`,`${base}_returnposition`,`${base}_condition_score`,
            cr(s, p, i),
          );
        }
  }

  // WHERE clause: highway filter (tiers 1-2) + viewport bbox (tiers 2+)
  const hwClause   = tier <= 2 ? `regexp_matches(highway, '${HW_RE[tier]}')` : null;
  let   bboxClause = null;
  if (tier >= 2) {
    const { minX, minY, maxX, maxY } = viewportToUTM(bounds);
    bboxClause = `ST_Intersects(street_geometry, ST_MakeEnvelope(${minX}, ${minY}, ${maxX}, ${maxY}))`;
  }
  const where = [hwClause, bboxClause].filter(Boolean).join(' AND ') || 'true';

  return `SELECT ${cols.join(', ')} FROM read_parquet('network.parquet') WHERE ${where}`;
}

// ── DuckDB init ───────────────────────────────────────────────────────────────
let conn          = null;
let _refreshTimer = null;

async function initDuckDB() {
  try {
    setStatus('Loading DuckDB WASM\u2026');
    const bundles  = duckdb.getJsDelivrBundles();
    const bundle   = await duckdb.selectBundle(bundles);
    const workerUrl = URL.createObjectURL(
      new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' })
    );
    const worker = new Worker(workerUrl);
    const db     = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    URL.revokeObjectURL(workerUrl);
    await db.registerFileURL('network.parquet', PARQUET_URL, duckdb.DuckDBDataProtocol.HTTP, false);
    conn = await db.connect();
    setStatus('Loading spatial extension\u2026');
    await conn.query('LOAD spatial;');
    setStatus('Ready \u2014 loading streets\u2026');
    await refreshMap();
  } catch (e) {
    showError(
      'DuckDB failed to load. Ensure you have internet access and are serving via ' +
      'http://, not file://. Error: ' + e.message
    );
  }
}

async function refreshMap() {
  if (!conn) return;
  const zoom   = map.getZoom();
  const tier   = getTier(zoom);
  const bounds = map.getBounds();
  setStatus(`Loading tier ${tier}\u2026`);
  try {
    const result = await conn.query(buildSQL(tier, bounds));
    const rows   = result.toArray().map(r => r.toJSON());
    setStatus(`Rendering ${rows.length} segments\u2026`);
    renderRows(rows, tier);
    setStatus(`Tier ${tier} \u00b7 ${rows.length} segments`);
  } catch (e) {
    const msg = e?.message || String(e);
    setStatus('Query error: ' + msg.slice(0, 120));
    console.error('County map query error:', e);
  }
}

map.on('zoomend moveend', () => {
  clearTimeout(_refreshTimer);
  _refreshTimer = setTimeout(() => { refreshMap(); renderHullsAndSlots(); }, 300);
});

// ── Legend toggles ────────────────────────────────────────────────────────────
[
  ['cb_streets', 'streets'], ['cb_bk_sep', 'bk_sep'], ['cb_bk_off', 'bk_off'],
  ['cb_sw_sep',  'sw_sep'],  ['cb_sw_off', 'sw_off'],
  ['cb_nodes',   'nodes'],   ['cb_ramps',  'ramps'],   ['cb_calm', 'calm'],
  ['cb_hulls',   'hulls'],   ['cb_slots',  'slots'],
].forEach(([id, key]) => {
  document.getElementById(id).addEventListener('change', e => {
    e.target.checked ? map.addLayer(lg[key]) : map.removeLayer(lg[key]);
  });
});

// ── Search ────────────────────────────────────────────────────────────────────
let _flashTimer = null;

function pxSearch() {
  const id    = document.getElementById('px-search-input').value.trim();
  const el    = document.getElementById('px-search-status');
  const idx   = _pxMode === 'seg' ? segIndex : nodeIndex;
  const entry = idx.get(id);
  if (!entry) {
    el.style.color = '#c00'; el.textContent = 'Not found: ' + id; return;
  }
  el.style.color = '#555'; el.textContent = '\u2192 Found: ' + id;
  map.flyTo(entry.mid, 20);
  if (entry.layer) {
    clearTimeout(_flashTimer);
    try { entry.layer.setStyle({ color: '#ffff00', weight: (entry.baseWeight || 2) * 2.5 }); } catch(_) {}
    _flashTimer = setTimeout(() => {
      try { entry.layer.setStyle({ color: entry.baseColor, weight: entry.baseWeight }); } catch(_) {}
    }, 2500);
    try { entry.layer.openPopup(entry.mid); } catch(_) {}
  }
}

function pxSetMode(mode) {
  _pxMode = mode;
  const s = document.getElementById('px-mode-seg');
  const n = document.getElementById('px-mode-node');
  s.style.background = mode === 'seg'  ? '#1a73e8' : 'white';
  s.style.color       = mode === 'seg'  ? 'white'   : '#333';
  n.style.background = mode === 'node' ? '#1a73e8' : 'white';
  n.style.color       = mode === 'node' ? 'white'   : '#333';
  document.getElementById('px-search-input').placeholder =
    mode === 'seg' ? 'e.g. 12_34_0' : 'e.g. 123456789';
  document.getElementById('px-search-status').textContent = '';
}

document.getElementById('px-mode-seg').onclick  = () => pxSetMode('seg');
document.getElementById('px-mode-node').onclick = () => pxSetMode('node');

// ── Probe ─────────────────────────────────────────────────────────────────────
let _probeActive = false;
const _probeTimerSeg = {};
const _probeTimerPt  = {};
const _hiddenSeg     = {};
const _hiddenPt      = {};

function haverDist(lat1, lon1, lat2, lon2) {
  const R = 6371000, r = Math.PI / 180;
  const dLat = (lat2 - lat1) * r, dLon = (lon2 - lon1) * r;
  const a = Math.sin(dLat/2)**2 + Math.cos(lat1*r)*Math.cos(lat2*r)*Math.sin(dLon/2)**2;
  return R * 2 * Math.asin(Math.sqrt(a));
}

function probeSegments(latlng, radiusM) {
  const found = [];
  segIndex.forEach((e, key) => {
    const d = haverDist(latlng.lat, latlng.lng, e.mid[0], e.mid[1]);
    if (d <= radiusM) found.push({ key, dist: d, entry: e });
  });
  found.sort((a, b) => a.dist - b.dist);
  return found;
}

function probePoints(latlng, radiusM) {
  const found = [];
  pointIndex.forEach((e, key) => {
    const d = haverDist(latlng.lat, latlng.lng, e.lat, e.lon);
    if (d <= radiusM) found.push({ key, dist: d, entry: e });
  });
  found.sort((a, b) => a.dist - b.dist);
  return found;
}

function flashFeature(key, kind) {
  if (kind === 'seg') {
    const e = segIndex.get(key); if (!e || !e.layer) return;
    if (_probeTimerSeg[key]) clearTimeout(_probeTimerSeg[key]);
    try { e.layer.setStyle({ color: '#ff4400', weight: e.baseWeight * 3 }); } catch(_) {}
    _probeTimerSeg[key] = setTimeout(() => {
      try { e.layer.setStyle({ color: e.baseColor, weight: e.baseWeight }); } catch(_) {}
    }, 2500);
  } else {
    const e = pointIndex.get(key); if (!e || !e.layer) return;
    if (_probeTimerPt[key]) clearTimeout(_probeTimerPt[key]);
    try { e.layer.setStyle({ color: '#ff4400', fillColor: '#ff4400', radius: e.baseRadius * 2.5 }); } catch(_) {}
    _probeTimerPt[key] = setTimeout(() => {
      try { e.layer.setStyle({ color: e.baseColor, fillColor: e.baseColor, radius: e.baseRadius }); } catch(_) {}
    }, 2500);
  }
}

function toggleHide(key, kind) {
  const safeKey = key.replace(/[^a-zA-Z0-9_]/g, '_');
  const btn = document.getElementById('px-hide-' + kind + '-' + safeKey);
  if (kind === 'seg') {
    const e = segIndex.get(key); if (!e || !e.layer) return;
    if (_hiddenSeg[key]) {
      try { e.layer.setStyle({ color: e.baseColor, weight: e.baseWeight, opacity: 0.8 }); } catch(_) {}
      delete _hiddenSeg[key];
      if (btn) btn.textContent = 'Hide';
    } else {
      try { e.layer.setStyle({ opacity: 0, weight: 0 }); } catch(_) {}
      _hiddenSeg[key] = true;
      if (btn) btn.textContent = 'Show';
    }
  } else {
    const e = pointIndex.get(key); if (!e || !e.layer) return;
    if (_hiddenPt[key]) {
      try { e.layer.setStyle({ color: e.baseColor, fillColor: e.baseColor, fillOpacity: 0.9, opacity: 1, radius: e.baseRadius }); } catch(_) {}
      delete _hiddenPt[key];
      if (btn) btn.textContent = 'Hide';
    } else {
      try { e.layer.setStyle({ opacity: 0, fillOpacity: 0 }); } catch(_) {}
      _hiddenPt[key] = true;
      if (btn) btn.textContent = 'Show';
    }
  }
}

function _probeRow(key, dist, entry, kind) {
  const safeKey = key.replace(/[^a-zA-Z0-9_]/g, '_');
  const isHidden = kind === 'seg' ? !!_hiddenSeg[key] : !!_hiddenPt[key];
  return '<div style="display:flex;align-items:center;gap:4px;padding:3px 2px;border-bottom:1px solid #f0f0f0;font-size:11px;">' +
    '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;flex-shrink:0;background:' + (entry.baseColor || '#888') + ';"></span>' +
    '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="' + (entry.label || key) + '">' + (entry.label || key) + '</span>' +
    '<span style="color:#888;flex-shrink:0;">' + Math.round(dist) + 'm</span>' +
    '<button onclick="flashFeature(\'' + key + '\',\'' + kind + '\')" style="padding:1px 5px;font-size:10px;cursor:pointer;border:1px solid #ccc;border-radius:3px;">Flash</button>' +
    '<button id="px-hide-' + kind + '-' + safeKey + '" onclick="toggleHide(\'' + key + '\',\'' + kind + '\')" style="padding:1px 5px;font-size:10px;cursor:pointer;border:1px solid #ccc;border-radius:3px;">' + (isHidden ? 'Show' : 'Hide') + '</button>' +
    '</div>';
}

function renderProbe(segResults, ptResults) {
  const panel   = document.getElementById('px-probe');
  const segList = document.getElementById('px-probe-segs');
  const ptList  = document.getElementById('px-probe-pts');
  segList.innerHTML = segResults.length
    ? segResults.map(f => _probeRow(f.key, f.dist, f.entry, 'seg')).join('')
    : '<div style="color:#999;font-size:11px;padding:4px 2px;">No segments found</div>';
  ptList.innerHTML  = ptResults.length
    ? ptResults.map(f => _probeRow(f.key, f.dist, f.entry, 'pt')).join('')
    : '<div style="color:#999;font-size:11px;padding:4px 2px;">No points found</div>';
  panel.style.display = 'block';
}

function _setProbeActive(active) {
  _probeActive = active;
  const btn = document.getElementById('px-probe-toggle');
  btn.style.background  = active ? '#1a73e8' : 'white';
  btn.style.color       = active ? 'white'   : '#333';
  btn.style.borderColor = active ? '#1a73e8' : '#aaa';
  btn.textContent       = active ? '\\u{1F50D} Probe ON \\u2014 click map' : '\\u{1F50D} Probe';
  if (!active) document.getElementById('px-probe').style.display = 'none';
}

document.getElementById('px-probe-toggle').addEventListener('click', e => {
  e.stopPropagation();
  _setProbeActive(!_probeActive);
});

document.getElementById('px-probe-close').addEventListener('click', () => _setProbeActive(false));

const _probeRadiusInput = document.getElementById('px-probe-radius');
const _probeRadiusVal   = document.getElementById('px-probe-radius-val');
_probeRadiusInput.addEventListener('input', () => { _probeRadiusVal.textContent = _probeRadiusInput.value; });

// ── Status / error helpers ────────────────────────────────────────────────────
function setStatus(msg) { document.getElementById('status').textContent = msg; }
function showError(msg)  {
  const el = document.getElementById('error-banner');
  el.textContent = msg; el.style.display = 'block';
  setStatus('Error \u2014 see banner');
}

renderHullsAndSlots();
initDuckDB();
</script>
</body>
</html>
"""


def generate_county_map(parquet_path: str, data: pd.DataFrame, output_name: str = 'sf_county') -> str:
    """Generate a county-wide interactive map backed by DuckDB WASM.

    Layers load progressively by zoom tier:
      z<=12  streets (arterials only, county-wide)
      z13-14 + bikeways
      z15-16 + sidewalks, crosswalks, traffic calming
      z17    + intersection nodes
      z18+   + curb ramps

    To open the map:
        cd Output && python -m http.server 8080 --bind 127.0.0.1
        http://localhost:8080/test_maps/<output_name>.html
    """
    # ── Pre-compute intersection hulls and crosswalk slot zones ─────────────────
    _hull_geom_cols_county = [
        'start_node_geometry', 'end_node_geometry',
        'sidewalk_left_geometry', 'sidewalk_right_geometry',
        *[f'sidewalk_{s}_curbramp_{p}_{n}_geometry'
          for s in ('left', 'right') for p in ('start', 'end') for n in (1, 2, 3)]
    ]
    _hull_attr_cols_county = [
        'start_node_id', 'end_node_id',
        'start_node_is_intersection_node', 'end_node_is_intersection_node',
        'name', 'highway',
        'sidewalk_left_presence', 'sidewalk_right_presence',
    ]
    _hull_street_geom = parse_geom_series(data['street_geometry']) if 'street_geometry' in data.columns else gpd.GeoSeries(index=data.index, dtype=object)
    _hull_df = gpd.GeoDataFrame(index=data.index, geometry=_hull_street_geom, crs='EPSG:4326')
    _hull_df['street_geometry'] = _hull_street_geom
    for _col in _hull_geom_cols_county:
        if _col in data.columns:
            _hull_df[_col] = parse_geom_series(data[_col])
    for _col in _hull_attr_cols_county:
        if _col in data.columns:
            _hull_df[_col] = data[_col]

    _, _hulls_gdf = _step_12(_hull_df, _DEFAULT_PIPELINE_CONFIG)
    _hulls_gdf = _step_13(_hulls_gdf, _DEFAULT_PIPELINE_CONFIG)
    _step_14(_hull_df, _hulls_gdf, _DEFAULT_PIPELINE_CONFIG)
    _slot_list = _hulls_gdf.attrs.get('crosswalk_slots', [])

    def _wgs_poly_to_ring(poly: BaseGeometry) -> list | None:
        """WGS84 Polygon → [[lon, lat], ...] exterior ring for GeoJSON, or None."""
        if poly is None or poly.is_empty:
            return None
        if hasattr(poly, 'geoms'):
            poly = max(poly.geoms, key=lambda p: p.area)  # type: ignore[attr-defined]
        if not hasattr(poly, 'exterior'):
            return None
        return [[x, y] for x, y in poly.exterior.coords]  # type: ignore[union-attr]

    hull_features: list[dict] = []
    for _hrow in _hulls_gdf.itertuples():
        _geom = cast(BaseGeometry, _hrow.geometry)
        _ring = _wgs_poly_to_ring(_geom)
        if _ring:
            _clon, _clat = _geom.centroid.x, _geom.centroid.y
            hull_features.append({
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [_ring]},
                'properties': {'tip': f'Hull ({_clon:.5f}, {_clat:.5f})'},
            })

    slot_features: list[dict] = []
    for _slot in _slot_list:
        _slot_poly = _slot.get('slot_geom')
        _ring = _wgs_poly_to_ring(_slot_poly)
        if not _ring:
            continue
        slot_features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [_ring]},
            'properties': {'tip': 'Crosswalk slot'},
        })

    hull_json = json.dumps({'type': 'FeatureCollection', 'features': hull_features})
    slot_json = json.dumps({'type': 'FeatureCollection', 'features': slot_features})

    parquet_basename = os.path.basename(parquet_path)
    html = (
        _COUNTY_MAP_TEMPLATE
        .replace('%%PARQUET_URL%%', f'../{parquet_basename}')
        .replace('%%TITLE%%', output_name)
        .replace('%%HULL_GEOJSON%%', hull_json)
        .replace('%%SLOT_GEOJSON%%', slot_json)
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{output_name}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'County map saved -> {out_path}')
    print(f'  Serve: cd Output && python -m http.server 8080 --bind 127.0.0.1')
    print(f'  Open:  http://localhost:8080/test_maps/{output_name}.html')
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
                      bar_pos=i,
                      default_lane_width_m=loc.get("default_lane_width_m", 3.5),
                      hull_fallback_r=loc.get("hull_fallback_r", 15.0)): loc["name"]
            for i, loc in enumerate(facility_locs)
        }
        for f in as_completed(future_to_name):
            print(f"Map saved -> {f.result()}")

    # County-wide DuckDB WASM map (one per unique parquet)
    _county_name = os.path.splitext(os.path.basename(parquet_path))[0].replace('_network', '_county_map')
    generate_county_map(parquet_path, data, _county_name)

