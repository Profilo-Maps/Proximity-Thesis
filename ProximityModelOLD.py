"""
ProximityModel.py — Data pipeline per OSMNetworkDescription.md.
Generates [city]_sanity.parquet and [city]_network.parquet per city.
All geometries stored as WKB bytes in EPSG:4326.
Pipeline works internally in a local projected (metric) CRS and
converts back to EPSG:4326 only at export time.
"""

# --- Imports ---

import json
import os
import sys
import warnings
import gc
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import traceback
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import (
    Point, LineString, Polygon, MultiPoint, MultiLineString,
    box as shapely_box
)
from shapely import wkb
from shapely.ops import nearest_points, unary_union, transform as shapely_transform
from shapely.strtree import STRtree
import osmnx as ox
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from pyproj import Transformer as _ProjTransformer
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore')

ox.settings.log_console = False
ox.settings.use_cache = True

# --- GPU Detection ---

GPU_AVAILABLE = False
_gpu_backend = None
GPU_MEMORY_LIMIT_PERCENT = 0.7

try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        GPU_AVAILABLE = True
        _gpu_backend = 'cupy'
        try:
            mempool = cp.get_default_memory_pool()
            device_id = cp.cuda.Device().id
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            total_memory = props['totalGlobalMem']
            mempool.set_limit(size=int(total_memory * GPU_MEMORY_LIMIT_PERCENT))
        except Exception:
            pass
except (ImportError, Exception):
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            _gpu_backend = 'torch'
            try:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_LIMIT_PERCENT)
            except Exception:
                pass
    except (ImportError, Exception):
        pass


class GPUContext:
    """Thin wrapper for GPU operations with graceful CPU fallback."""

    def __init__(self, use_gpu=True):
        self.use_gpu = GPU_AVAILABLE and use_gpu
        self.backend = _gpu_backend if self.use_gpu else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def nearest_neighbors(self, reference_points, query_points, k=1):
        if not self.use_gpu:
            tree = cKDTree(reference_points)
            distances, indices = tree.query(query_points, k=k)
            return distances, indices
        try:
            if self.backend == 'cupy':
                ref_gpu = cp.asarray(reference_points)
                query_gpu = cp.asarray(query_points)
                diff = query_gpu[:, cp.newaxis, :] - ref_gpu[cp.newaxis, :, :]
                distances_all = cp.sqrt(cp.sum(diff ** 2, axis=2))
                indices_gpu = cp.argpartition(distances_all, k-1, axis=1)[:, :k]
                distances_gpu = cp.take_along_axis(distances_all, indices_gpu, axis=1)
                distances_result = cp.asnumpy(distances_gpu)
                indices_result = cp.asnumpy(indices_gpu)
                del ref_gpu, query_gpu, diff, distances_all, indices_gpu, distances_gpu
                cp.get_default_memory_pool().free_all_blocks()
                return distances_result, indices_result
            elif self.backend == 'torch':
                ref_gpu = torch.from_numpy(reference_points).cuda()
                query_gpu = torch.from_numpy(query_points).cuda()
                diff = query_gpu.unsqueeze(1) - ref_gpu.unsqueeze(0)
                distances_all = torch.sqrt(torch.sum(diff ** 2, dim=2))
                distances_gpu, indices_gpu = torch.topk(distances_all, k, largest=False, dim=1)
                distances_result = distances_gpu.cpu().numpy()
                indices_result = indices_gpu.cpu().numpy()
                del ref_gpu, query_gpu, diff, distances_all, distances_gpu, indices_gpu
                torch.cuda.empty_cache()
                return distances_result, indices_result
        except Exception as e:
            print(f"  GPU computation failed, falling back to CPU: {e}")
            tree = cKDTree(reference_points)
            distances, indices = tree.query(query_points, k=k)
            return distances, indices
        # Fallback: unknown backend (should not happen)
        tree = cKDTree(reference_points)
        distances, indices = tree.query(query_points, k=k)
        return distances, indices


def get_gpu_info():
    info = {'available': GPU_AVAILABLE, 'backend': _gpu_backend, 'gpu_names': [], 'total_memory_gb': []}
    if not GPU_AVAILABLE:
        return info
    try:
        if _gpu_backend == 'cupy':
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                info['gpu_names'].append(props['name'].decode('utf-8'))
                info['total_memory_gb'].append(props['totalGlobalMem'] / (1024**3))
        elif _gpu_backend == 'torch':
            import torch
            info['gpu_names'].append(torch.cuda.get_device_name(0))
            info['total_memory_gb'].append(torch.cuda.get_device_properties(0).total_memory / 1e9)
    except Exception:
        pass
    return info

# --- Constants ---

HIGHWAY_SANITY = {
    'motorway': {'lane_width': 3.7, 'buffer': 3.0},
    'trunk': {'lane_width': 3.7, 'buffer': 3.0},
    'primary': {'lane_width': 3.4, 'buffer': 2.5},
    'secondary': {'lane_width': 3.3, 'buffer': 2.0},
    'tertiary': {'lane_width': 3.0, 'buffer': 1.5},
    'residential': {'lane_width': 3.0, 'buffer': 1.2},
    'service': {'lane_width': 2.7, 'buffer': 0.5},
    'unclassified': {'lane_width': 3.0, 'buffer': 1.2},
    'living_street': {'lane_width': 2.7, 'buffer': 0.5},
}

DEFAULT_LANE_WIDTHS = {
    'motorway': 3.7, 'trunk': 3.7, 'primary': 3.4,
    'secondary': 3.3, 'tertiary': 3.0, 'residential': 3.0,
    'service': 2.7, 'unclassified': 3.0, 'living_street': 2.7,
}

MIN_SANITY_FLOOR = 1.5
RAMP_SEARCH_RADIUS_M = 30.0
DEFLECTION_THRESHOLD_DEG = 45.0  
INTERSECTION_BBOX_M = 30.0
DEFAULT_OUTER_TRUST_BUFFER_M = 15.0
DEFAULT_INNER_TRUST_BUFFER_M = 5.0
DEFAULT_CURB_RAMP_SPLIT_OFFSET = 0.75  # meters, half sidewalk width default
MIN_CURB_RETURN_LENGTH = 0.3  # meters, below this collapse to single apex ramp
MIN_RAMP_SEPARATION = 1.0  # meters, minimum distance between ramps from adjacent corners
CORNER_SKIP_ANGLE = 15.0  # degrees, skip facility generation for near-parallel merges
CONTIGUITY_TOLERANCE = 2.0  # meters, default tolerance for contiguity checks

# --- CRS Utilities ---

def estimate_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Estimate the best UTM EPSG code for a GeoDataFrame based on its centroid.
    Input must be in EPSG:4326 (geographic). Returns an EPSG string like 'EPSG:32618'."""
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy) = (min_lon, min_lat, max_lon, max_lat)
    center_lon = (bounds[0] + bounds[2]) / 2.0
    center_lat = (bounds[1] + bounds[3]) / 2.0
    # UTM zone number from longitude
    zone_number = int((center_lon + 180) / 6) + 1
    # EPSG code: 326xx for northern hemisphere, 327xx for southern
    if center_lat >= 0:
        epsg_code = 32600 + zone_number
    else:
        epsg_code = 32700 + zone_number
    return f"EPSG:{epsg_code}"


def reproject_to_working_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to the working (metric) CRS.
    Handles None CRS by assuming EPSG:4326."""
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

# --- Configuration Dataclasses ---

@dataclass
class ColumnMappingConfig:
    """Maps government dataset column names to canonical fields."""
    street_id: Optional[str] = None
    street_name: Optional[str] = None
    street_highway: Optional[str] = None
    street_maxspeed: Optional[str] = None
    street_lanes: Optional[str] = None
    street_surface: Optional[str] = None
    sidewalk_id: Optional[str] = None
    sidewalk_surface: Optional[str] = None
    sidewalk_width: Optional[str] = None
    sidewalk_incline: Optional[str] = None
    bikelane_id: Optional[str] = None
    bikelane_type: Optional[str] = None
    bikelane_surface: Optional[str] = None
    bikelane_width: Optional[str] = None
    curbramp_id: Optional[str] = None
    curbramp_return_loc: Optional[str] = None
    curbramp_position: Optional[str] = None
    curbramp_condition: Optional[str] = None
    feature_id: Optional[str] = None
    feature_type: Optional[str] = None


@dataclass
class GovernmentDataPaths:
    """Optional file paths for government data layers."""
    street_centerlines: Optional[str] = None
    intersection_nodes: Optional[str] = None
    sidewalks: Optional[str] = None
    bikelanes: Optional[str] = None
    curb_ramps: Optional[str] = None
    crosswalks: Optional[str] = None
    parcels: Optional[str] = None
    street_features: Optional[str] = None
    sidewalk_features: Optional[str] = None
    bikeway_features: Optional[str] = None


@dataclass
class CityConfig:
    """Configuration for a single city."""
    name: str
    government_data_paths: GovernmentDataPaths = field(default_factory=GovernmentDataPaths)
    column_mappings: ColumnMappingConfig = field(default_factory=ColumnMappingConfig)
    curbramp_trustworthy: bool = False


@dataclass
class GlobalConfig:
    """Global configuration for all cities."""
    output_dir: str
    cities: List[CityConfig]
    default_max_speed: int = 25
    curb_ramp_trustworthiness_outer_buffer: float = DEFAULT_OUTER_TRUST_BUFFER_M
    curb_ramp_trustworthiness_inner_buffer: float = DEFAULT_INNER_TRUST_BUFFER_M
    use_gpu: bool = True
    python_env: str = "python"


def load_config(config_dict: Dict[str, Any]) -> GlobalConfig:
    cities = []
    for city_dict in config_dict.get('cities', []):
        gov_paths = GovernmentDataPaths(**city_dict.get('government_data_paths', {}))
        col_map = ColumnMappingConfig(**city_dict.get('column_mappings', {}))
        city = CityConfig(
            name=city_dict['name'],
            government_data_paths=gov_paths,
            column_mappings=col_map,
            curbramp_trustworthy=city_dict.get('curbramp_trustworthy', False)
        )
        cities.append(city)
    return GlobalConfig(
        output_dir=config_dict.get('output_dir', 'Output'),
        cities=cities,
        default_max_speed=config_dict.get('default_max_speed', 25),
        curb_ramp_trustworthiness_outer_buffer=config_dict.get(
            'curb_ramp_trustworthiness_outer_buffer', DEFAULT_OUTER_TRUST_BUFFER_M),
        curb_ramp_trustworthiness_inner_buffer=config_dict.get(
            'curb_ramp_trustworthiness_inner_buffer', DEFAULT_INNER_TRUST_BUFFER_M),
        use_gpu=config_dict.get('use_gpu', True),
        python_env=config_dict.get('python_env', 'python'),
    )

# --- Sequential ID Counter ---

class SequentialIDCounter:
    def __init__(self, start: int = 1):
        self._next = start
        self._next_neg = -1

    def next(self) -> int:
        current = self._next
        self._next += 1
        return current

    def next_negative(self) -> int:
        current = self._next_neg
        self._next_neg -= 1
        return current


# Module-level counters (reset per city via _reset_pipeline_counters)
sidewalk_counter = SequentialIDCounter()
sidewalk_feature_counter = SequentialIDCounter()
bikeway_counter = SequentialIDCounter()
bikeway_feature_counter = SequentialIDCounter()
curbramp_counter = SequentialIDCounter()
crosswalk_counter = SequentialIDCounter()
split_node_counter = SequentialIDCounter()

# --- Helper Functions ---

def normalize_tag(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) > 0 else None
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ['', 'none', 'nan', 'unknown'] else None


def resolve_lane_width(edge_row, highway: str) -> float:
    if 'width' in edge_row.index and edge_row['width'] is not None:
        try:
            w = float(edge_row['width'])
            if w > 0:
                return w
        except (ValueError, TypeError):
            pass
    lanes = edge_row.get('lanes', 1)
    try:
        lanes = int(lanes) if lanes else 1
    except (ValueError, TypeError):
        lanes = 1
    if lanes < 1:
        lanes = 1
    default_width = DEFAULT_LANE_WIDTHS.get(highway, 3.0)
    return lanes * default_width


def compute_bearing(geometry: LineString) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    coords = list(geometry.coords)
    if len(coords) < 2:
        return 0.0
    start = coords[0]
    end = coords[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    bearing = np.degrees(np.arctan2(dx, dy))
    return (bearing + 360) % 360


def compute_bearings_vectorized(geometries: gpd.GeoSeries) -> np.ndarray:
    n = len(geometries)
    bearings = np.zeros(n)

    # Batch coordinate extraction
    starts = np.zeros((n, 2))
    ends = np.zeros((n, 2))
    valid = np.ones(n, dtype=bool)

    for idx, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            valid[idx] = False
        else:
            coords = np.asarray(geom.coords)
            if len(coords) >= 2:
                starts[idx] = coords[0]
                ends[idx] = coords[-1]
            else:
                valid[idx] = False

    # Vectorized bearing calculation on all geometries at once
    dx = ends[:, 0] - starts[:, 0]
    dy = ends[:, 1] - starts[:, 1]
    angles = np.degrees(np.arctan2(dx, dy))
    bearings = np.where(valid, (angles + 360) % 360, 0)

    return bearings


def geom_to_wkb(geom) -> Optional[bytes]:
    if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
        return None
    try:
        return geom.wkb
    except Exception:
        return None


def project_features_to_segment(feature_geometry: Point, segment_geometry: LineString) -> Optional[Point]:
    if feature_geometry is None or segment_geometry is None or segment_geometry.is_empty:
        return None
    try:
        return segment_geometry.interpolate(segment_geometry.project(feature_geometry))
    except Exception:
        return None


def compute_circular_mean_bearing(bearings: List[float]) -> float:
    if not bearings:
        return 0.0
    radians = np.deg2rad(bearings)
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    mean_rad = np.arctan2(sin_mean, cos_mean)
    return (np.rad2deg(mean_rad) + 360) % 360


def _angle_between_bearings(b1: float, b2: float) -> float:
    """Compute the positive angular difference from b1 to b2 going clockwise."""
    return (b2 - b1) % 360


def _angular_bisector(b1: float, b2: float) -> float:
    """Compute the angular bisector direction between two bearings (interior angle)."""
    diff = (b2 - b1) % 360
    return (b1 + diff / 2) % 360


def _point_along_bearing(origin: Point, bearing_deg: float, distance: float) -> Point:
    """Compute a point at a given distance along a bearing from an origin point.
    bearing_deg is measured clockwise from north (positive Y axis).
    Works in projected (metric) coordinates."""
    rad = math.radians(bearing_deg)
    dx = distance * math.sin(rad)
    dy = distance * math.cos(rad)
    return Point(origin.x + dx, origin.y + dy)

# --- Sanity Buffer ---

def _compute_parcel_distance_worker(args):
    chunk_data, parcel_tree, parcel_geoms, use_gpu = args
    results = []
    if use_gpu and GPU_AVAILABLE:
        try:
            with GPUContext(use_gpu=use_gpu) as ctx:
                centroids = np.array([row['geometry'].centroid.coords[0] for _, row in chunk_data.iterrows()])
                parcel_coords = np.array([g.centroid.coords[0] for g in parcel_geoms])
                distances, indices = ctx.nearest_neighbors(parcel_coords, centroids, k=1)
                for i, (idx, row) in enumerate(chunk_data.iterrows()):
                    street_geom = row['geometry']
                    if street_geom is None or street_geom.is_empty:
                        results.append((row['osmid'], MIN_SANITY_FLOOR))
                        continue
                    nearest_parcel = parcel_geoms[indices[i, 0]]
                    distance = street_geom.distance(nearest_parcel)
                    results.append((row['osmid'], max(distance, MIN_SANITY_FLOOR)))
        except Exception:
            use_gpu = False
    if not use_gpu:
        for idx, row in chunk_data.iterrows():
            street_geom = row['geometry']
            if street_geom is None or street_geom.is_empty:
                results.append((row['osmid'], MIN_SANITY_FLOOR))
                continue
            nearest_idx = parcel_tree.query(street_geom.centroid.coords[0])[1]
            distance = street_geom.distance(parcel_geoms[nearest_idx])
            results.append((row['osmid'], max(distance, MIN_SANITY_FLOOR)))
    return results


def compute_sanity_buffer_from_parcels(network: gpd.GeoDataFrame, parcel_path: str, n_jobs: int = None) -> pd.DataFrame:
    print(f"  Loading parcels from {parcel_path}...")
    parcels = gpd.read_file(parcel_path)
    if parcels.crs != network.crs:
        parcels = parcels.to_crs(network.crs)
    print("  Aggregating parcels into building footprints...")
    try:
        parcels = parcels.dissolve()
        if len(parcels) == 1:
            parcels = parcels.explode(index_parts=False)
    except Exception:
        aggregated_geom = parcels.geometry.unary_union
        if aggregated_geom.geom_type == 'MultiPolygon':
            parcels = gpd.GeoDataFrame(geometry=list(aggregated_geom.geoms), crs=parcels.crs)
        else:
            parcels = gpd.GeoDataFrame(geometry=[aggregated_geom], crs=parcels.crs)
    print(f"  Aggregated to {len(parcels)} building footprints")
    parcel_geoms = list(parcels.geometry)
    parcel_coords = np.array([g.centroid.coords[0] for g in parcel_geoms])
    parcel_tree = cKDTree(parcel_coords)
    use_gpu = GPU_AVAILABLE
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    chunks = np.array_split(network, n_jobs)
    args = [(chunk, parcel_tree, parcel_geoms, use_gpu) for chunk in chunks]
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(_compute_parcel_distance_worker, args), total=len(args), desc="  Parcel distances"))
    all_results = [r for chunk_results in results for r in chunk_results]
    return pd.DataFrame(all_results, columns=['osmid', 'max_offset_width'])


def compute_sanity_buffer_from_highway(network: gpd.GeoDataFrame) -> pd.DataFrame:
    print("  Computing sanity buffer from highway classification...")
    FORMULAS = {
        'motorway': (1.85, 3.0), 'trunk': (1.85, 3.0),
        'primary': (1.7, 2.5), 'secondary': (1.65, 2.0),
        'tertiary': (1.5, 1.5), 'residential': (1.5, 1.2),
        'service': (1.35, 0.5), 'unclassified': (1.5, 1.2), 'living_street': (1.35, 0.5),
    }
    highway_normalized = network['highway'].apply(
        lambda x: normalize_tag(x) if pd.notna(x) else 'residential'
    ).apply(lambda x: x if x in FORMULAS else 'residential')

    def parse_lanes(val):
        try:
            return max(1, int(val)) if pd.notna(val) else 1
        except (ValueError, TypeError):
            return 1

    lanes_normalized = network.get('lanes', 1).apply(parse_lanes)
    max_offset_width = [
        max((lanes * FORMULAS[hw][0]) + FORMULAS[hw][1], MIN_SANITY_FLOOR)
        for hw, lanes in zip(highway_normalized, lanes_normalized)
    ]
    return pd.DataFrame({'osmid': network['osmid'].values, 'max_offset_width': max_offset_width})


def export_sanity_buffer(sanity_df: pd.DataFrame, city_name: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    schema = pa.schema([('osmid', pa.int64()), ('max_offset_width', pa.float64())])
    table = pa.Table.from_pandas(sanity_df, schema=schema)
    pq.write_table(table, output_path)
    print(f"  Exported sanity buffer: {output_path}")

# --- OSM Data Fetch (Phase 1, Step 1) ---

def fetch_osm_network(city_config: CityConfig, working_crs: Optional[str] = None) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, str]:
    """
    Fetch OSM network data for a city.
    Returns: (edges_gdf, crossings_cache, working_crs)

    Per Phase 1 Step 1: Pull Street, Cycleway, footway, and crossing data.
    Crossing data is cached separately.

    If working_crs is None, estimates a UTM zone from the data centroid.
    All returned GeoDataFrames are in the working (metric) CRS.
    """
    print(f"  Fetching OSM data for {city_config.name}...")

    highway_tags = {
        'highway': [
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'residential', 'service', 'unclassified', 'living_street',
            'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
            'footway', 'path', 'pedestrian', 'steps',
            'cycleway',
        ]
    }

    print("  Fetching street geometries...")
    edges = ox.features_from_place(city_config.name, tags=highway_tags)

    if not edges.empty:
        if 'area' in edges.columns:
            edges = edges[edges['area'] != 'yes']
        edges = edges[edges.geometry.type == 'LineString'].copy()
        if 'footway' in edges.columns:
            edges = edges[edges['footway'] != 'crossing']
        if 'highway' in edges.columns:
            edges = edges[edges['highway'] != 'crossing']
        if 'osmid' not in edges.columns and edges.index.name == 'osmid':
            edges = edges.reset_index()
        elif 'osmid' not in edges.columns:
            edges['osmid'] = range(len(edges))

    print(f"  Found {len(edges)} street segments")

    # Estimate working CRS from the data if not provided
    if edges.crs is None:
        raise ValueError("Street edges GeoDataFrame has no CRS set. Assign a CRS before calling this function.")
    if working_crs is None:
        working_crs = estimate_utm_crs(edges)
    print(f"  Working CRS: {working_crs}")

    # Reproject to metric CRS for all spatial operations
    edges = edges.to_crs(working_crs)

    # Assign start/end node IDs from endpoint coordinates
    # Round to 2 decimal places (cm precision in metric CRS)
    coord_to_id = {}
    node_id_counter = 1
    start_node_ids = []
    end_node_ids = []

    for row in edges.itertuples():
        geom = row.geometry
        if geom and hasattr(geom, 'coords'):
            start_key = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
            if start_key not in coord_to_id:
                coord_to_id[start_key] = node_id_counter
                node_id_counter += 1
            end_key = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
            if end_key not in coord_to_id:
                coord_to_id[end_key] = node_id_counter
                node_id_counter += 1
            start_node_ids.append(coord_to_id[start_key])
            end_node_ids.append(coord_to_id[end_key])
        else:
            start_node_ids.append(None)
            end_node_ids.append(None)

    edges['start_node_osmid'] = start_node_ids
    edges['end_node_osmid'] = end_node_ids

    # Initialize node geometry columns
    edges['start_node_geometry'] = edges['geometry'].apply(
        lambda g: Point(list(g.coords)[0]) if g is not None and hasattr(g, 'coords') else None)
    edges['end_node_geometry'] = edges['geometry'].apply(
        lambda g: Point(list(g.coords)[-1]) if g is not None and hasattr(g, 'coords') else None)
    edges['public_data_id_start_end_nodes'] = None

    # Initialize street feature columns
    for col in ['street_feature_types', 'public_data_id_street_feature',
                'street_feature_geometry', 'street_feature_geometry_projected']:
        edges[col] = None

    # Fetch crossing data separately and cache
    print("  Fetching crossing data...")
    crossings_cache = gpd.GeoDataFrame()
    try:
        crossing_tags = {'highway': ['crossing'], 'footway': ['crossing']}
        crossings = ox.features_from_place(city_config.name, tags=crossing_tags)
        if not crossings.empty:
            crossings_cache = crossings[crossings.geometry.type.isin(['Point', 'LineString'])].copy()
            # Reproject crossings to working CRS
            if crossings_cache.crs is None:
                crossings_cache = crossings_cache.set_crs("EPSG:4326")
            crossings_cache = crossings_cache.to_crs(working_crs)
            print(f"  Found {len(crossings_cache)} crossing features")
    except Exception as e:
        print(f"  Warning: Could not fetch crossings: {e}")

    print(f"  Completed fetch: {len(edges)} edges")
    return edges, crossings_cache, working_crs


# --- Government Data Integration (Phase 1, Step 2) ---

def load_geospatial_file(filepath: str, lat_col: Optional[str] = None, lon_col: Optional[str] = None,
                         geom_col: Optional[str] = None, wkt_col: Optional[str] = None) -> Optional[gpd.GeoDataFrame]:
    if not filepath or not os.path.exists(filepath):
        return None
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext in ['.geojson', '.json', '.shp']:
        try:
            return gpd.read_file(filepath)
        except Exception as e:
            print(f"  Warning: Error reading {filepath}: {e}")
            return None
    if file_ext == '.csv':
        try:
            df = pd.read_csv(filepath, engine="python", on_bad_lines="warn")
            if wkt_col and wkt_col in df.columns:
                from shapely import wkt as shapely_wkt
                geometries = df[wkt_col].apply(lambda x: shapely_wkt.loads(x) if pd.notna(x) else None)
                gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                return gdf[gdf.geometry.notna()]
            if geom_col and geom_col in df.columns:
                geometries = df[geom_col].apply(lambda x: wkb.loads(x) if pd.notna(x) else None)
                gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                return gdf[gdf.geometry.notna()]
            if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
                valid = df.dropna(subset=[lat_col, lon_col])
                return gpd.GeoDataFrame(valid, geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]), crs="EPSG:4326")
            lat_candidates = ['latitude', 'lat', 'y', 'yloc']
            lon_candidates = ['longitude', 'lon', 'lng', 'long', 'x', 'xloc']
            df_cols_lower = {col.lower(): col for col in df.columns}
            lat_found = next((df_cols_lower[c] for c in lat_candidates if c in df_cols_lower), None)
            lon_found = next((df_cols_lower[c] for c in lon_candidates if c in df_cols_lower), None)
            if lat_found and lon_found:
                valid = df.dropna(subset=[lat_found, lon_found])
                return gpd.GeoDataFrame(valid, geometry=gpd.points_from_xy(valid[lon_found], valid[lat_found]), crs="EPSG:4326")
            return None
        except Exception as e:
            print(f"  Warning: Error reading CSV {filepath}: {e}")
            return None
    return None


def _load_gov_layer(city_config: CityConfig, path_attr: str, label: str,
                    working_crs: Optional[str] = None) -> Optional[gpd.GeoDataFrame]:
    """Generic loader for government data layers. Reprojects to working_crs if provided."""
    path = getattr(city_config.government_data_paths, path_attr, None)
    if path is None or not os.path.exists(path):
        return None
    print(f"  Loading {label} from {path}...")
    gdf = load_geospatial_file(path)
    if gdf is not None and gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf is not None and working_crs is not None:
        gdf = reproject_to_working_crs(gdf, working_crs)
    return gdf

def load_government_centerlines(city_config, working_crs=None): return _load_gov_layer(city_config, 'street_centerlines', 'government centerlines', working_crs)
def load_government_sidewalks(city_config, working_crs=None): return _load_gov_layer(city_config, 'sidewalks', 'government sidewalks', working_crs)
def load_government_bikelanes(city_config, working_crs=None): return _load_gov_layer(city_config, 'bikelanes', 'government bikelanes', working_crs)
def load_government_intersection_nodes(city_config, working_crs=None): return _load_gov_layer(city_config, 'intersection_nodes', 'government intersection nodes', working_crs)
def load_government_crosswalks(city_config, working_crs=None): return _load_gov_layer(city_config, 'crosswalks', 'government crosswalks', working_crs)
def load_street_features(city_config, working_crs=None): return _load_gov_layer(city_config, 'street_features', 'street features', working_crs)
def load_sidewalk_features(city_config, working_crs=None): return _load_gov_layer(city_config, 'sidewalk_features', 'sidewalk features', working_crs)
def load_bikeway_features(city_config, working_crs=None): return _load_gov_layer(city_config, 'bikeway_features', 'bikeway features', working_crs)


def merge_government_centerlines(osm_edges: gpd.GeoDataFrame, gov_lines: gpd.GeoDataFrame, city_config: CityConfig) -> gpd.GeoDataFrame:
    if gov_lines is None or gov_lines.empty:
        return osm_edges
    print("  Merging government centerlines with OSM data...")
    if gov_lines.crs != osm_edges.crs:
        gov_lines = gov_lines.to_crs(osm_edges.crs)
    osm_tree = STRtree(osm_edges.geometry)
    col_map = city_config.column_mappings
    updates = {}
    for idx in tqdm(gov_lines.index, total=len(gov_lines), desc="  Matching centerlines"):
        gov_row = gov_lines.loc[idx]
        gov_geom = gov_row.geometry
        if gov_geom is None or gov_geom.is_empty:
            continue
        nearby_indices = osm_tree.query(gov_geom, predicate='dwithin', distance=20.0)
        if len(nearby_indices) == 0:
            continue
        best_idx = None
        best_dist = float('inf')
        for osm_idx in nearby_indices:
            osm_geom = osm_edges.iloc[osm_idx].geometry
            try:
                dist = gov_geom.hausdorff_distance(osm_geom)
                if dist < best_dist and dist < 20.0:
                    best_dist = dist
                    best_idx = osm_idx
            except Exception:
                continue
        if best_idx is not None:
            osm_edge_idx = osm_edges.index[best_idx]
            if osm_edge_idx not in updates:
                updates[osm_edge_idx] = {}
            updates[osm_edge_idx]['geometry'] = gov_geom
            for attr, col_name in [('street_id', 'public_data_id_street'), ('street_name', 'name'),
                                   ('street_highway', 'highway'), ('street_maxspeed', 'maxspeed'),
                                   ('street_lanes', 'lanes'), ('street_surface', 'surface')]:
                mapped = getattr(col_map, attr, None)
                if mapped and mapped in gov_row.index:
                    updates[osm_edge_idx][col_name] = gov_row[mapped]
    for osm_idx, cols in updates.items():
        for col, value in cols.items():
            osm_edges.at[osm_idx, col] = value
    print(f"  Merged {len(updates)} government centerlines")
    return osm_edges


def merge_government_intersection_nodes(edges: gpd.GeoDataFrame, gov_nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gov_nodes is None or gov_nodes.empty:
        return edges
    print("  Joining government intersection nodes to edge endpoints...")
    if gov_nodes.crs != edges.crs:
        gov_nodes = gov_nodes.to_crs(edges.crs)
    gov_tree = STRtree(gov_nodes.geometry)
    match_threshold = 10.0

    def _get_gov_id(row):
        for candidate in ['id', 'ID', 'OBJECTID', 'node_id', 'NodeID', 'FID']:
            val = row.get(candidate)
            if val is not None:
                return str(val)
        return None

    matched = 0
    for idx, row in edges.iterrows():
        geom = row['geometry']
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])
        start_gov_id = end_gov_id = None
        nearby = gov_tree.query(start_pt, predicate='dwithin', distance=match_threshold)
        if len(nearby) > 0:
            gov_row = gov_nodes.iloc[nearby[0]]
            edges.at[idx, 'start_node_geometry'] = gov_row.geometry
            start_gov_id = _get_gov_id(gov_row)
        nearby = gov_tree.query(end_pt, predicate='dwithin', distance=match_threshold)
        if len(nearby) > 0:
            gov_row = gov_nodes.iloc[nearby[0]]
            edges.at[idx, 'end_node_geometry'] = gov_row.geometry
            end_gov_id = _get_gov_id(gov_row)
        if start_gov_id is not None or end_gov_id is not None:
            edges.at[idx, 'public_data_id_start_end_nodes'] = (start_gov_id, end_gov_id)
            matched += 1
    print(f"  Matched government intersection nodes to {matched} edges")
    return edges

# --- Government data merge helpers ---

def _determine_facility_side(facility_geom: LineString, street_geom: LineString) -> str:
    """Determine whether a facility geometry is on the left or right side of a street.
    Uses the cross product of the street direction vector and the vector from the
    street to the facility midpoint. Positive cross product = left, negative = right."""
    if facility_geom is None or street_geom is None:
        return 'left'
    try:
        # Get the midpoint of the facility
        facility_mid = facility_geom.interpolate(0.5, normalized=True)
        # Project onto the street to find the closest point and local direction
        proj_dist = street_geom.project(facility_mid)
        # Sample two points along the street near the projection to determine direction
        # Use a small fraction of street length to work in any CRS (projected or geographic)
        sample_offset = max(street_geom.length * 0.01, 1e-7)
        d1 = max(0, proj_dist - sample_offset)
        d2 = min(street_geom.length, proj_dist + sample_offset)
        if d2 - d1 < 1e-12:
            return 'left'
        p1 = street_geom.interpolate(d1)
        p2 = street_geom.interpolate(d2)
        # Street direction vector
        dx_street = p2.x - p1.x
        dy_street = p2.y - p1.y
        # Vector from street to facility midpoint
        dx_fac = facility_mid.x - p1.x
        dy_fac = facility_mid.y - p1.y
        # Cross product: positive = left, negative = right
        cross = dx_street * dy_fac - dy_street * dx_fac
        return 'left' if cross > 0 else 'right'
    except Exception:
        return 'left'


def merge_government_sidewalks(edges: gpd.GeoDataFrame, gov_sidewalks: gpd.GeoDataFrame, city_config: CityConfig) -> gpd.GeoDataFrame:
    print("  Merging government sidewalk data...")
    col_map = city_config.column_mappings
    if gov_sidewalks.crs != edges.crs:
        gov_sidewalks = gov_sidewalks.to_crs(edges.crs)
    edges_buffered = gpd.GeoDataFrame(geometry=edges.geometry.buffer(10.0), index=edges.index, crs=edges.crs)
    joined = gpd.sjoin(gov_sidewalks, edges_buffered, how='inner', predicate='intersects')
    del edges_buffered
    for edge_idx in joined['index_right'].unique():
        matches = joined[joined['index_right'] == edge_idx]
        street_geom = edges.loc[edge_idx, 'geometry']
        for _, match in matches.iterrows():
            sidewalk_geom = match['geometry']
            if isinstance(sidewalk_geom, LineString):
                side = _determine_facility_side(sidewalk_geom, street_geom)
                edges.at[edge_idx, f'sidewalk_{side}_geometry'] = sidewalk_geom
                id_col = col_map.sidewalk_id if col_map.sidewalk_id else 'id'
                gov_id = match.get(id_col, match.get('ID', match.get('OBJECTID', None)))
                edges.at[edge_idx, f'public_data_id_sidewalk_{side}'] = str(gov_id) if gov_id is not None else None
                if col_map.sidewalk_surface and col_map.sidewalk_surface in match.index:
                    edges.at[edge_idx, f'sidewalk_{side}_surface'] = match[col_map.sidewalk_surface]
                if col_map.sidewalk_width and col_map.sidewalk_width in match.index:
                    edges.at[edge_idx, f'sidewalk_{side}_width'] = match[col_map.sidewalk_width]
                if col_map.sidewalk_incline and col_map.sidewalk_incline in match.index:
                    edges.at[edge_idx, f'sidewalk_{side}_incline'] = match[col_map.sidewalk_incline]
    return edges


def merge_government_bikelanes(edges: gpd.GeoDataFrame, gov_bikelanes: gpd.GeoDataFrame, city_config: CityConfig) -> gpd.GeoDataFrame:
    print("  Merging government bikelane data...")
    col_map = city_config.column_mappings
    if gov_bikelanes.crs != edges.crs:
        gov_bikelanes = gov_bikelanes.to_crs(edges.crs)
    edges_buffered = gpd.GeoDataFrame(geometry=edges.geometry.buffer(10.0), index=edges.index, crs=edges.crs)
    joined = gpd.sjoin(gov_bikelanes, edges_buffered, how='inner', predicate='intersects')
    del edges_buffered
    # Track how many bikelanes have been assigned per side per edge
    side_lane_count: Dict[Tuple[int, str], int] = {}
    for edge_idx in joined['index_right'].unique():
        matches = joined[joined['index_right'] == edge_idx]
        street_geom = edges.loc[edge_idx, 'geometry']
        for _, match in matches.iterrows():
            bikelane_geom = match['geometry']
            if isinstance(bikelane_geom, LineString):
                side = _determine_facility_side(bikelane_geom, street_geom)
                count_key = (edge_idx, side)
                current_count = side_lane_count.get(count_key, 0)
                lane_n = min(current_count + 1, 2)  # Max 2 lanes per side
                if current_count >= 2:
                    continue  # Skip if both lane slots already filled
                side_lane_count[count_key] = current_count + 1
                edges.at[edge_idx, f'bikeway_{side}_{lane_n}_geometry'] = bikelane_geom
                id_col = col_map.bikelane_id if col_map.bikelane_id else 'id'
                gov_id = match.get(id_col, match.get('ID', match.get('OBJECTID', None)))
                edges.at[edge_idx, f'public_data_id_bikeway_{side}_{lane_n}'] = str(gov_id) if gov_id is not None else None
                if col_map.bikelane_type and col_map.bikelane_type in match.index:
                    edges.at[edge_idx, f'bikeway_{side}_{lane_n}_type'] = match[col_map.bikelane_type]
                if col_map.bikelane_surface and col_map.bikelane_surface in match.index:
                    edges.at[edge_idx, f'bikeway_{side}_{lane_n}_surface'] = match[col_map.bikelane_surface]
                if col_map.bikelane_width and col_map.bikelane_width in match.index:
                    edges.at[edge_idx, f'bikeway_{side}_{lane_n}_width'] = match[col_map.bikelane_width]
    return edges


def populate_street_features(edges: gpd.GeoDataFrame, street_features: gpd.GeoDataFrame, city_config: CityConfig) -> gpd.GeoDataFrame:
    print("  Populating street features...")
    if street_features.crs != edges.crs:
        street_features = street_features.to_crs(edges.crs)
    for col in ['street_feature_types', 'public_data_id_street_feature', 'street_feature_geometry', 'street_feature_geometry_projected']:
        edges[col] = [[] for _ in range(len(edges))]
    if len(edges) == 0:
        return edges
    col_map = city_config.column_mappings
    for _, feature in street_features.iterrows():
        feature_point = feature['geometry']
        distances = edges.geometry.distance(feature_point)
        if len(distances) == 0:
            continue
        nearest_idx = distances.idxmin()
        if distances[nearest_idx] > 20.0:
            continue
        feature_type_col = col_map.feature_type if col_map.feature_type else 'feature_type'
        feature_type = feature.get(feature_type_col, feature.get('type', feature.get('amenity', 'unknown')))
        feature_id_col = col_map.feature_id if col_map.feature_id else 'id'
        gov_id = feature.get(feature_id_col, feature.get('ID', feature.get('OBJECTID', None)))
        street_geom = edges.loc[nearest_idx, 'geometry']
        projected_point = street_geom.interpolate(street_geom.project(feature_point))
        edges.at[nearest_idx, 'street_feature_types'].append(str(feature_type))
        edges.at[nearest_idx, 'public_data_id_street_feature'].append(str(gov_id) if gov_id is not None else None)
        edges.at[nearest_idx, 'street_feature_geometry'].append(feature_point)
        edges.at[nearest_idx, 'street_feature_geometry_projected'].append(projected_point)
    for idx in edges.index:
        fg = edges.at[idx, 'street_feature_geometry']
        pg = edges.at[idx, 'street_feature_geometry_projected']
        edges.at[idx, 'street_feature_geometry'] = MultiPoint(fg) if fg else MultiPoint([])
        edges.at[idx, 'street_feature_geometry_projected'] = MultiPoint(pg) if pg else MultiPoint([])
    return edges


def populate_sidewalk_features(edges: gpd.GeoDataFrame, sidewalk_features: gpd.GeoDataFrame,
                               city_config: CityConfig, sw_feature_counter: SequentialIDCounter) -> gpd.GeoDataFrame:
    print("  Populating sidewalk features...")
    if sidewalk_features.crs != edges.crs:
        sidewalk_features = sidewalk_features.to_crs(edges.crs)
    for side in ['left', 'right']:
        for col in [f'sidewalk_{side}_feature_ids', f'sidewalk_{side}_feature_types',
                    f'public_data_id_sidewalk_{side}_feature',
                    f'sidewalk_{side}_feature_geometry', f'sidewalk_{side}_feature_geometry_projected']:
            edges[col] = [[] for _ in range(len(edges))]
    if len(edges) == 0 or len(sidewalk_features) == 0:
        return edges

    # Build STRtree index over all sidewalk geometries (left and right)
    geom_entries = []  # (geometry, edge_idx, side)
    for idx in edges.index:
        for side in ['left', 'right']:
            sw_geom = edges.loc[idx, f'sidewalk_{side}_geometry']
            if sw_geom is not None and isinstance(sw_geom, LineString):
                geom_entries.append((sw_geom, idx, side))

    if not geom_entries:
        return edges

    tree_geoms = [e[0] for e in geom_entries]
    tree = STRtree(tree_geoms)

    for _, feature in sidewalk_features.iterrows():
        feature_point = feature['geometry']
        # Query nearest geometry from STRtree
        nearest_idx_in_tree = tree.nearest(feature_point)
        nearest_geom = tree_geoms[nearest_idx_in_tree]
        d = nearest_geom.distance(feature_point)
        if d > 10.0:
            continue
        _, edge_idx, nearest_side = geom_entries[nearest_idx_in_tree]
        feature_type = feature.get('type', feature.get('amenity', feature.get('feature_type', 'unknown')))
        gov_id = feature.get('id', feature.get('ID', feature.get('OBJECTID', None)))
        fid = sw_feature_counter.next()
        projected_point = nearest_geom.interpolate(nearest_geom.project(feature_point))
        edges.at[edge_idx, f'sidewalk_{nearest_side}_feature_ids'].append(str(fid))
        edges.at[edge_idx, f'sidewalk_{nearest_side}_feature_types'].append(str(feature_type))
        edges.at[edge_idx, f'public_data_id_sidewalk_{nearest_side}_feature'].append(str(gov_id) if gov_id else None)
        edges.at[edge_idx, f'sidewalk_{nearest_side}_feature_geometry'].append(feature_point)
        edges.at[edge_idx, f'sidewalk_{nearest_side}_feature_geometry_projected'].append(projected_point)

    for idx in edges.index:
        for side in ['left', 'right']:
            fg = edges.at[idx, f'sidewalk_{side}_feature_geometry']
            pg = edges.at[idx, f'sidewalk_{side}_feature_geometry_projected']
            edges.at[idx, f'sidewalk_{side}_feature_geometry'] = MultiPoint(fg) if fg else MultiPoint([])
            edges.at[idx, f'sidewalk_{side}_feature_geometry_projected'] = MultiPoint(pg) if pg else MultiPoint([])
    return edges


def populate_bikeway_features(edges: gpd.GeoDataFrame, bikeway_features: gpd.GeoDataFrame, city_config: CityConfig) -> gpd.GeoDataFrame:
    print("  Populating bikeway features...")
    if bikeway_features.crs != edges.crs:
        bikeway_features = bikeway_features.to_crs(edges.crs)
    for side in ['left', 'right']:
        for n in [1, 2]:
            for col in [f'bikeway_{side}_{n}_feature_ids', f'bikeway_{side}_{n}_feature_types',
                        f'public_data_id_bikeway_{side}_{n}_features',
                        f'bikeway_{side}_{n}_feature_geometry', f'bikeway_{side}_{n}_feature_geometry_projected']:
                edges[col] = [[] for _ in range(len(edges))]
    if len(edges) == 0 or len(bikeway_features) == 0:
        return edges

    # Build STRtree index over all bikeway geometries (left/right x 1/2)
    geom_entries = []  # (geometry, edge_idx, side, lane)
    for idx in edges.index:
        for side in ['left', 'right']:
            for n in [1, 2]:
                bw_geom = edges.loc[idx, f'bikeway_{side}_{n}_geometry']
                if bw_geom is not None and isinstance(bw_geom, LineString):
                    geom_entries.append((bw_geom, idx, side, n))

    if not geom_entries:
        return edges

    tree_geoms = [e[0] for e in geom_entries]
    tree = STRtree(tree_geoms)

    for _, feature in bikeway_features.iterrows():
        feature_point = feature['geometry']
        nearest_idx_in_tree = tree.nearest(feature_point)
        nearest_geom = tree_geoms[nearest_idx_in_tree]
        d = nearest_geom.distance(feature_point)
        if d > 10.0:
            continue
        _, edge_idx, nearest_side, nearest_lane = geom_entries[nearest_idx_in_tree]
        feature_type = feature.get('type', feature.get('amenity', feature.get('feature_type', 'unknown')))
        gov_id = feature.get('id', feature.get('ID', feature.get('OBJECTID', None)))
        projected_point = nearest_geom.interpolate(nearest_geom.project(feature_point))
        edges.at[edge_idx, f'bikeway_{nearest_side}_{nearest_lane}_feature_ids'].append('')
        edges.at[edge_idx, f'bikeway_{nearest_side}_{nearest_lane}_feature_types'].append(str(feature_type))
        edges.at[edge_idx, f'public_data_id_bikeway_{nearest_side}_{nearest_lane}_features'].append(str(gov_id) if gov_id else None)
        edges.at[edge_idx, f'bikeway_{nearest_side}_{nearest_lane}_feature_geometry'].append(feature_point)
        edges.at[edge_idx, f'bikeway_{nearest_side}_{nearest_lane}_feature_geometry_projected'].append(projected_point)

    for idx in edges.index:
        for side in ['left', 'right']:
            for n in [1, 2]:
                fg = edges.at[idx, f'bikeway_{side}_{n}_feature_geometry']
                pg = edges.at[idx, f'bikeway_{side}_{n}_feature_geometry_projected']
                edges.at[idx, f'bikeway_{side}_{n}_feature_geometry'] = MultiPoint(fg) if fg else MultiPoint([])
                edges.at[idx, f'bikeway_{side}_{n}_feature_geometry_projected'] = MultiPoint(pg) if pg else MultiPoint([])
    return edges

# --- Offset Geometry Generation (Phase 1, Steps 3-4) ---

def load_sanity_buffer(sanity_path: str) -> Dict[int, float]:
    df = pd.read_parquet(sanity_path)
    return dict(zip(df['osmid'], df['max_offset_width']))


def compute_offset_distance(edge_row, side: str, facility_type: str, sanity_buffer: Dict[int, float]) -> float:
    osmid = edge_row['osmid']
    highway = normalize_tag(edge_row.get('highway', 'residential'))
    lane_width = resolve_lane_width(edge_row, highway)
    base_offset = lane_width / 2.0
    bikeway_width = 0.0
    if facility_type == 'sidewalk':
        for lane_n in (1, 2):
            bw_type = edge_row.get(f'bikeway_{side}_{lane_n}_type')
            if bw_type is None:
                continue
            w = 1.5
            w_val = edge_row.get(f'bikeway_{side}_{lane_n}_width')
            if w_val is not None:
                try:
                    w = float(w_val)
                except (ValueError, TypeError):
                    pass
            bikeway_width += w
    offset = base_offset + bikeway_width
    if osmid in sanity_buffer:
        offset = min(offset, sanity_buffer[osmid])
    return max(offset, 0.5)


def generate_offset_geometry(line_geom: LineString, offset_distance: float, side: str) -> Optional[LineString]:
    if line_geom is None or line_geom.is_empty:
        return None
    try:
        sign = 1 if side == 'left' else -1
        offset_geom = line_geom.offset_curve(sign * offset_distance, quad_segs=16, join_style=2, mitre_limit=5.0)
        if offset_geom is None or offset_geom.is_empty:
            return None
        if isinstance(offset_geom, MultiLineString):
            offset_geom = max(offset_geom.geoms, key=lambda g: g.length)
        return offset_geom if isinstance(offset_geom, LineString) else None
    except Exception:
        return None


def _enrich_bikeway_chunk(chunk_data):
    chunk, sanity_buffer = chunk_data
    for idx in chunk.index:
        row = chunk.loc[idx]
        for side in ('left', 'right'):
            if row.get(f'bikeway_{side}_1_type') is not None and row.get(f'bikeway_{side}_1_geometry') is None:
                offset_dist = compute_offset_distance(row, side, 'bikeway', sanity_buffer)
                geom = generate_offset_geometry(row['geometry'], offset_dist, side)
                chunk.at[idx, f'bikeway_{side}_1_geometry'] = geom
                chunk.at[idx, f'bikeway_{side}_buffered'] = True
            if row.get(f'bikeway_{side}_2_type') is not None and row.get(f'bikeway_{side}_2_geometry') is None:
                offset_lane1 = compute_offset_distance(row, side, 'bikeway', sanity_buffer)
                w1 = 1.5
                try:
                    w1_raw = row.get(f'bikeway_{side}_1_width')
                    if w1_raw is not None:
                        w1 = float(w1_raw)
                except (ValueError, TypeError):
                    pass
                offset_lane2 = offset_lane1 + w1
                osmid = row.get('osmid')
                if osmid in sanity_buffer:
                    offset_lane2 = min(offset_lane2, sanity_buffer[osmid])
                offset_lane2 = max(offset_lane2, 0.5)
                geom2 = generate_offset_geometry(row['geometry'], offset_lane2, side)
                chunk.at[idx, f'bikeway_{side}_2_geometry'] = geom2
                chunk.at[idx, f'bikeway_{side}_buffered'] = True
    return chunk


def enrich_bikeway_geometries(network: gpd.GeoDataFrame, sanity_buffer: Dict[int, float], n_jobs: int = None) -> gpd.GeoDataFrame:
    print("  Generating bikeway offset geometries...")
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    chunks = np.array_split(network, n_jobs)
    args = [(chunk, sanity_buffer) for chunk in chunks]
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(_enrich_bikeway_chunk, args), total=len(args), desc="  Bikeway geometries"))
    result_df = pd.concat(results, ignore_index=False)
    if isinstance(network, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(result_df, geometry='geometry', crs=network.crs)
    return result_df


def _enrich_sidewalk_chunk(chunk_data):
    chunk, sanity_buffer = chunk_data
    for idx in chunk.index:
        row = chunk.loc[idx]
        if row.get('sidewalk_left_presence') is True and row.get('sidewalk_left_geometry') is None:
            offset_dist = compute_offset_distance(row, 'left', 'sidewalk', sanity_buffer)
            chunk.at[idx, 'sidewalk_left_geometry'] = generate_offset_geometry(row['geometry'], offset_dist, 'left')
            chunk.at[idx, 'sidewalk_left_buffered'] = True
        if row.get('sidewalk_right_presence') is True and row.get('sidewalk_right_geometry') is None:
            offset_dist = compute_offset_distance(row, 'right', 'sidewalk', sanity_buffer)
            chunk.at[idx, 'sidewalk_right_geometry'] = generate_offset_geometry(row['geometry'], offset_dist, 'right')
            chunk.at[idx, 'sidewalk_right_buffered'] = True
    return chunk


def enrich_sidewalk_geometries(network: gpd.GeoDataFrame, sanity_buffer: Dict[int, float], n_jobs: int = None) -> gpd.GeoDataFrame:
    print("  Generating sidewalk offset geometries...")
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    chunks = np.array_split(network, n_jobs)
    args = [(chunk, sanity_buffer) for chunk in chunks]
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(_enrich_sidewalk_chunk, args), total=len(args), desc="  Sidewalk geometries"))
    result_df = pd.concat(results, ignore_index=False)
    if isinstance(network, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(result_df, geometry='geometry', crs=network.crs)
    return result_df

# --- Vertex Deflection Splitting (Phase 1, Step 5) ---

def _detect_deflection_worker(args):
    idx, row, threshold = args
    geom = row['geometry']
    if geom is None or geom.is_empty or not isinstance(geom, LineString):
        return idx, []
    coords = list(geom.coords)
    if len(coords) < 3:
        return idx, []
    split_indices = []
    for i in range(1, len(coords) - 1):
        p_prev = np.array(coords[i-1])
        p_curr = np.array(coords[i])
        p_next = np.array(coords[i+1])
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm < 1e-9 or v2_norm < 1e-9:
            continue
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_product))
        deflection = 180.0 - angle_deg
        if deflection > threshold:
            split_indices.append(i)
    return idx, split_indices


def detect_and_split_deflections(edges: gpd.GeoDataFrame, counter: SequentialIDCounter) -> gpd.GeoDataFrame:
    """Split edges at vertices with deflection > 45 degrees (per spec)."""
    print("  Detecting and splitting vertex deflections...")
    n_jobs = max(1, mp.cpu_count() - 1)
    args = [(idx, row, DEFLECTION_THRESHOLD_DEG) for idx, row in edges.iterrows()]
    with mp.Pool(n_jobs) as pool:
        deflection_results = list(tqdm(pool.imap(_detect_deflection_worker, args), total=len(args), desc="  Detecting deflections"))
    edges_to_split = {idx: si for idx, si in deflection_results if si}
    if not edges_to_split:
        print(f"  No deflections detected")
        return edges
    coord_to_split_node: Dict[Tuple[float, float], int] = {}
    new_edges = []
    for idx in tqdm(edges.index, total=len(edges), desc="  Splitting edges"):
        row = edges.loc[idx]
        if idx not in edges_to_split:
            new_edges.append(row)
            continue
        geom = row['geometry']
        coords = list(geom.coords)
        split_indices = edges_to_split[idx]
        segments = []
        start_idx = 0
        for split_idx in split_indices:
            seg_coords = coords[start_idx:split_idx+1]
            if len(seg_coords) >= 2:
                segments.append(seg_coords)
            start_idx = split_idx
        seg_coords = coords[start_idx:]
        if len(seg_coords) >= 2:
            segments.append(seg_coords)
        for seg_idx, seg_coords in enumerate(segments):
            new_row = row.copy()
            new_row['geometry'] = LineString(seg_coords)
            if seg_idx > 0:
                prev_split_coord = tuple(seg_coords[0])
                if prev_split_coord not in coord_to_split_node:
                    coord_to_split_node[prev_split_coord] = counter.next_negative()
                new_row['start_node_osmid'] = coord_to_split_node[prev_split_coord]
            if seg_idx < len(segments) - 1:
                split_coord = tuple(seg_coords[-1])
                if split_coord not in coord_to_split_node:
                    coord_to_split_node[split_coord] = counter.next_negative()
                new_row['end_node_osmid'] = coord_to_split_node[split_coord]
            new_edges.append(new_row)
    result = gpd.GeoDataFrame(new_edges, crs=edges.crs).reset_index(drop=True)
    print(f"  Split {len(result) - len(edges)} segments due to deflection")
    return result

# --- Layer-Aware Planarization (Phase 1, Step 6) ---

def planarize_by_layer(edges: gpd.GeoDataFrame, counter: SequentialIDCounter) -> gpd.GeoDataFrame:
    """
    Conduct layer-aware planarization based on OSM layers to ensure that all
    crossing street, independent bikeway, and independent footway segments are
    assigned node geometries within each layer.

    Segments on different OSM 'layer' values do not interact (e.g. a bridge
    over a road). Within the same layer, crossing linestrings are split at
    their intersection points and new shared node IDs are assigned.
    """
    print("  Performing layer-aware planarization...")

    # Group edges by layer (default layer=0)
    def _get_layer(row):
        layer = row.get('layer', 0)
        try:
            return int(layer) if pd.notna(layer) else 0
        except (ValueError, TypeError):
            return 0

    edges['_layer'] = edges.apply(_get_layer, axis=1)
    layers = edges['_layer'].unique()

    new_edges_all = []
    total_splits = 0

    for layer_val in sorted(layers):
        layer_mask = edges['_layer'] == layer_val
        layer_edges = edges[layer_mask]

        if len(layer_edges) < 2:
            new_edges_all.append(layer_edges)
            continue

        # Build STRtree for this layer
        geom_list = list(layer_edges.geometry)
        idx_list = list(layer_edges.index)
        tree = STRtree(geom_list)

        split_map: Dict[int, List[LineString]] = {}  # original idx -> list of sub-segments

        for i, (orig_idx, row) in enumerate(layer_edges.iterrows()):
            geom_a = row['geometry']
            if geom_a is None or geom_a.is_empty:
                continue

            # Find crossing geometries in same layer
            nearby = tree.query(geom_a)
            intersection_points = []

            for j in nearby:
                if j == i:
                    continue
                geom_b = geom_list[j]
                if geom_b is None or geom_b.is_empty:
                    continue
                try:
                    ix = geom_a.intersection(geom_b)
                    if ix.is_empty:
                        continue
                    if ix.geom_type == 'Point':
                        # Only split at interior points, not shared endpoints
                        if not geom_a.touches(geom_b):
                            intersection_points.append(ix)
                    elif ix.geom_type == 'MultiPoint':
                        for pt in ix.geoms:
                            if not geom_a.touches(geom_b):
                                intersection_points.append(pt)
                except Exception:
                    continue

            if intersection_points:
                # Split geom_a at all intersection points
                split_distances = sorted(set(geom_a.project(pt) for pt in intersection_points))
                # Filter out distances at start/end
                split_distances = [d for d in split_distances if 0.1 < d < geom_a.length - 0.1]

                if split_distances:
                    sub_segments = []
                    prev_dist = 0.0
                    for d in split_distances:
                        seg = _substring(geom_a, prev_dist, d)
                        if seg is not None and seg.length > 0.01:
                            sub_segments.append(seg)
                        prev_dist = d
                    # Final segment
                    seg = _substring(geom_a, prev_dist, geom_a.length)
                    if seg is not None and seg.length > 0.01:
                        sub_segments.append(seg)

                    if len(sub_segments) > 1:
                        split_map[orig_idx] = sub_segments

        # Rebuild edges for this layer
        coord_to_node: Dict[Tuple[float, float], int] = {}
        for orig_idx, row in layer_edges.iterrows():
            if orig_idx in split_map:
                for seg_i, seg_geom in enumerate(split_map[orig_idx]):
                    new_row = row.copy()
                    new_row['geometry'] = seg_geom
                    # Assign node IDs at split points
                    start_key = (round(seg_geom.coords[0][0], 2), round(seg_geom.coords[0][1], 2))
                    end_key = (round(seg_geom.coords[-1][0], 2), round(seg_geom.coords[-1][1], 2))
                    if seg_i > 0:
                        if start_key not in coord_to_node:
                            coord_to_node[start_key] = counter.next_negative()
                        new_row['start_node_osmid'] = coord_to_node[start_key]
                        new_row['start_node_is_block_node'] = True
                    if seg_i < len(split_map[orig_idx]) - 1:
                        if end_key not in coord_to_node:
                            coord_to_node[end_key] = counter.next_negative()
                        new_row['end_node_osmid'] = coord_to_node[end_key]
                        new_row['end_node_is_block_node'] = True
                    new_edges_all.append(pd.DataFrame([new_row]))
                    total_splits += 1
                total_splits -= 1  # Don't count the original
            else:
                new_edges_all.append(pd.DataFrame([row]))

    if total_splits > 0:
        result = gpd.GeoDataFrame(pd.concat(new_edges_all, ignore_index=True), crs=edges.crs)
    else:
        result = edges.copy()

    if '_layer' in result.columns:
        result = result.drop(columns=['_layer'])

    print(f"  Planarization created {total_splits} new split segments")
    return result


def _substring(line: LineString, start_dist: float, end_dist: float) -> Optional[LineString]:
    """Extract a substring of a LineString between two distances along it."""
    if start_dist >= end_dist:
        return None
    try:
        coords = list(line.coords)
        new_coords = []
        current_dist = 0.0

        # Add interpolated start point
        start_pt = line.interpolate(start_dist)
        new_coords.append(start_pt.coords[0])

        # Add intermediate vertices
        for i in range(1, len(coords)):
            seg_start = Point(coords[i-1])
            seg_end = Point(coords[i])
            seg_len = seg_start.distance(seg_end)
            next_dist = current_dist + seg_len

            if next_dist > start_dist and current_dist < end_dist:
                if current_dist >= start_dist:
                    new_coords.append(coords[i-1])
                if next_dist <= end_dist:
                    new_coords.append(coords[i])

            current_dist = next_dist

        # Add interpolated end point
        end_pt = line.interpolate(end_dist)
        new_coords.append(end_pt.coords[0])

        # Deduplicate consecutive identical coords
        deduped = [new_coords[0]]
        for c in new_coords[1:]:
            if c != deduped[-1]:
                deduped.append(c)

        if len(deduped) >= 2:
            return LineString(deduped)
        return None
    except Exception:
        return None

# --- Tag Extraction (Phase 1) ---

def normalize_left_right_tags(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize bearings. Left/right swap happens later via reassign_facility_data_by_bearing."""
    print("  Computing bearings...")
    network['bearing'] = compute_bearings_vectorized(network['geometry'])
    network['normalized_bearing'] = network['bearing']
    return network


def extract_sidewalk_tags(network: gpd.GeoDataFrame, sw_counter: SequentialIDCounter) -> gpd.GeoDataFrame:
    print("  Extracting sidewalk tags...")
    n = len(network)
    for side in ['left', 'right']:
        network[f'sidewalk_{side}_ID'] = None
        network[f'sidewalk_{side}_block_ID'] = None
        network[f'sidewalk_{side}_presence'] = False
        network[f'public_data_id_sidewalk_{side}'] = None
        network[f'sidewalk_{side}_surface'] = None
        network[f'sidewalk_{side}_quality'] = None
        network[f'sidewalk_{side}_width'] = None
        network[f'sidewalk_{side}_incline'] = None
        network[f'sidewalk_{side}_buffered'] = False
        network[f'sidewalk_{side}_geometry'] = None
        network[f'sidewalk_{side}_feature_ids'] = [[] for _ in range(n)]
        network[f'sidewalk_{side}_feature_types'] = [[] for _ in range(n)]
        network[f'public_data_id_sidewalk_{side}_feature'] = [[] for _ in range(n)]
        network[f'sidewalk_{side}_feature_geometry'] = [MultiPoint([]) for _ in range(n)]
        network[f'sidewalk_{side}_feature_geometry_projected'] = [MultiPoint([]) for _ in range(n)]

    # Vectorized tag normalization
    hw_norm = network['highway'].apply(normalize_tag) if 'highway' in network.columns else pd.Series([None] * n, index=network.index)
    fw_norm = network['footway'].apply(normalize_tag) if 'footway' in network.columns else pd.Series([None] * n, index=network.index)
    surface_norm = network['surface'].apply(normalize_tag) if 'surface' in network.columns else pd.Series([None] * n, index=network.index)
    smoothness_norm = network['smoothness'].apply(normalize_tag) if 'smoothness' in network.columns else pd.Series([None] * n, index=network.index)
    width_norm = network['width'].apply(normalize_tag) if 'width' in network.columns else pd.Series([None] * n, index=network.index)
    incline_norm = network['incline'].apply(normalize_tag) if 'incline' in network.columns else pd.Series([None] * n, index=network.index)

    # Mask: footway=sidewalk rows (separate geometry)
    is_footway_type = hw_norm.isin(['footway', 'path', 'pedestrian', 'steps'])
    is_crossing = fw_norm == 'crossing'
    is_sidewalk_separate = is_footway_type & (fw_norm == 'sidewalk') & ~is_crossing
    sep_indices = network.index[is_sidewalk_separate]

    # Batch-assign IDs for separate sidewalks (need sequential IDs)
    sep_ids_left = [sw_counter.next() for _ in range(len(sep_indices))]
    sep_ids_right = [sw_counter.next() for _ in range(len(sep_indices))]

    for side, sep_ids in [('left', sep_ids_left), ('right', sep_ids_right)]:
        network.loc[sep_indices, f'sidewalk_{side}_presence'] = True
        network.loc[sep_indices, f'sidewalk_{side}_ID'] = sep_ids
        network.loc[sep_indices, f'sidewalk_{side}_buffered'] = False
        network.loc[sep_indices, f'sidewalk_{side}_geometry'] = network.loc[sep_indices, 'geometry']
        network.loc[sep_indices, f'sidewalk_{side}_surface'] = surface_norm.loc[sep_indices]
        network.loc[sep_indices, f'sidewalk_{side}_quality'] = smoothness_norm.loc[sep_indices]
        network.loc[sep_indices, f'sidewalk_{side}_width'] = width_norm.loc[sep_indices]
        network.loc[sep_indices, f'sidewalk_{side}_incline'] = incline_norm.loc[sep_indices]

    # Mask: regular street rows (not footway types)
    is_street = ~is_footway_type
    street_indices = network.index[is_street]

    for side in ['left', 'right']:
        # Build the sidewalk tag value for this side
        side_col = f'sidewalk:{side}'
        both_col = 'sidewalk:both'
        base_col = 'sidewalk'
        tag_series = (network.loc[street_indices, side_col] if side_col in network.columns else pd.Series(dtype=object, index=street_indices))
        if both_col in network.columns:
            tag_series = tag_series.fillna(network.loc[street_indices, both_col])
        if base_col in network.columns:
            tag_series = tag_series.fillna(network.loc[street_indices, base_col])
        val_series = tag_series.apply(normalize_tag)

        presence = val_series.isin(['yes', 'both', 'separate', 'seperate', side])
        present_indices = street_indices[presence]
        network.loc[present_indices, f'sidewalk_{side}_presence'] = True

        # Assign sequential IDs for present sidewalks
        if len(present_indices) > 0:
            sw_ids = [sw_counter.next() for _ in range(len(present_indices))]
            network.loc[present_indices, f'sidewalk_{side}_ID'] = sw_ids

            # Surface/width/incline from side-specific or generic tags
            for attr, osm_key in [('surface', 'surface'), ('width', 'width'), ('incline', 'incline')]:
                side_attr_col = f'sidewalk:{side}:{osm_key}'
                generic_attr_col = f'sidewalk:{osm_key}'
                attr_series = (network.loc[present_indices, side_attr_col].apply(normalize_tag)
                               if side_attr_col in network.columns
                               else pd.Series([None] * len(present_indices), index=present_indices))
                if generic_attr_col in network.columns:
                    fallback = network.loc[present_indices, generic_attr_col].apply(normalize_tag)
                    attr_series = attr_series.fillna(fallback)
                network.loc[present_indices, f'sidewalk_{side}_{attr}'] = attr_series

    return network


def extract_cycleway_tags(network: gpd.GeoDataFrame, bw_counter: SequentialIDCounter) -> gpd.GeoDataFrame:
    print("  Extracting cycleway tags...")
    network['oneway_bicycle'] = None
    for side in ['left', 'right']:
        for n in [1, 2]:
            for col in [f'bikeway_{side}_{n}_id', f'bikeway_{side}_{n}_block_id',
                        f'public_data_id_bikeway_{side}_{n}', f'bikeway_{side}_{n}_type',
                        f'bikeway_{side}_{n}_surface', f'bikeway_{side}_{n}_quality',
                        f'bikeway_{side}_{n}_permitted', f'bikeway_{side}_{n}_width',
                        f'bikeway_{side}_{n}_incline', f'bikeway_{side}_{n}_geometry']:
                network[col] = None
            network[f'bikeway_{side}_{n}_feature_ids'] = [[] for _ in range(len(network))]
            network[f'bikeway_{side}_{n}_feature_types'] = [[] for _ in range(len(network))]
            network[f'public_data_id_bikeway_{side}_{n}_features'] = [[] for _ in range(len(network))]
            network[f'bikeway_{side}_{n}_feature_geometry'] = [MultiPoint([]) for _ in range(len(network))]
            network[f'bikeway_{side}_{n}_feature_geometry_projected'] = [MultiPoint([]) for _ in range(len(network))]
        network[f'bikeway_{side}_buffered'] = False

    # Vectorized: normalize highway column once
    hw_norm = network['highway'].apply(normalize_tag) if 'highway' in network.columns else pd.Series([None] * len(network), index=network.index)
    is_cycleway = hw_norm == 'cycleway'
    is_not_cycleway = ~is_cycleway
    cycleway_indices = network.index[is_cycleway]
    non_cycleway_indices = network.index[is_not_cycleway]

    # --- Dedicated cycleway rows (highway=cycleway) ---
    if len(cycleway_indices) > 0:
        for side in ['left', 'right']:
            bw_ids = [bw_counter.next() for _ in range(len(cycleway_indices))]
            network.loc[cycleway_indices, f'bikeway_{side}_1_id'] = bw_ids
            network.loc[cycleway_indices, f'bikeway_{side}_1_type'] = 'track'
            network.loc[cycleway_indices, f'bikeway_{side}_buffered'] = False
            network.loc[cycleway_indices, f'bikeway_{side}_1_geometry'] = network.loc[cycleway_indices, 'geometry']
            for attr, osm_key in [('surface', 'surface'), ('quality', 'smoothness'), ('width', 'width'), ('incline', 'incline'), ('permitted', 'bicycle')]:
                if osm_key in network.columns:
                    network.loc[cycleway_indices, f'bikeway_{side}_1_{attr}'] = network.loc[cycleway_indices, osm_key].apply(normalize_tag)

        # oneway_bicycle for cycleways: prefer oneway:bicycle, fallback to oneway
        if 'oneway:bicycle' in network.columns:
            ow_bic = network.loc[cycleway_indices, 'oneway:bicycle'].apply(normalize_tag)
        else:
            ow_bic = pd.Series([None] * len(cycleway_indices), index=cycleway_indices)
        if 'oneway' in network.columns:
            ow_fallback = network.loc[cycleway_indices, 'oneway'].apply(normalize_tag)
            ow_bic = ow_bic.fillna(ow_fallback)
        network.loc[cycleway_indices, 'oneway_bicycle'] = ow_bic

    # --- Non-cycleway rows: check cycleway:left / cycleway:right tags ---
    if len(non_cycleway_indices) > 0:
        if 'oneway:bicycle' in network.columns:
            network.loc[non_cycleway_indices, 'oneway_bicycle'] = network.loc[non_cycleway_indices, 'oneway:bicycle'].apply(normalize_tag)

        for side in ['left', 'right']:
            # Primary cycleway tag: prefer cycleway:{side}, fallback to cycleway
            side_col = f'cycleway:{side}'
            generic_col = 'cycleway'
            if side_col in network.columns:
                cw_val = network.loc[non_cycleway_indices, side_col].apply(normalize_tag)
            else:
                cw_val = pd.Series([None] * len(non_cycleway_indices), index=non_cycleway_indices)
            if generic_col in network.columns:
                cw_fallback = network.loc[non_cycleway_indices, generic_col].apply(normalize_tag)
                cw_val = cw_val.fillna(cw_fallback)

            has_cycleway = cw_val.notna() & ~cw_val.isin(['no', 'none'])
            present_indices = non_cycleway_indices[has_cycleway]

            if len(present_indices) > 0:
                bw_ids = [bw_counter.next() for _ in range(len(present_indices))]
                network.loc[present_indices, f'bikeway_{side}_1_id'] = bw_ids
                network.loc[present_indices, f'bikeway_{side}_1_type'] = cw_val[has_cycleway].values

                for attr, suffix in [('surface', 'surface'), ('width', 'width'), ('incline', 'incline')]:
                    side_attr_col = f'cycleway:{side}:{suffix}'
                    generic_attr_col = f'cycleway:{suffix}'
                    if side_attr_col in network.columns:
                        attr_vals = network.loc[present_indices, side_attr_col].apply(normalize_tag)
                    else:
                        attr_vals = pd.Series([None] * len(present_indices), index=present_indices)
                    if generic_attr_col in network.columns:
                        attr_fallback = network.loc[present_indices, generic_attr_col].apply(normalize_tag)
                        attr_vals = attr_vals.fillna(attr_fallback)
                    network.loc[present_indices, f'bikeway_{side}_1_{attr}'] = attr_vals

                if 'bicycle' in network.columns:
                    network.loc[present_indices, f'bikeway_{side}_1_permitted'] = network.loc[present_indices, 'bicycle'].apply(normalize_tag)

            # Secondary cycleway lane (cycleway:{side}:2)
            cw2_col = f'cycleway:{side}:2'
            if cw2_col in network.columns:
                cw2_val = network.loc[non_cycleway_indices, cw2_col].apply(normalize_tag)
                has_cw2 = cw2_val.notna() & ~cw2_val.isin(['no', 'none'])
                present_2 = non_cycleway_indices[has_cw2]
                if len(present_2) > 0:
                    bw2_ids = [bw_counter.next() for _ in range(len(present_2))]
                    network.loc[present_2, f'bikeway_{side}_2_id'] = bw2_ids
                    network.loc[present_2, f'bikeway_{side}_2_type'] = cw2_val[has_cw2].values

    return network

# --- Block Detection (Phase 2) ---

def extract_nodes_from_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Vectorized coordinate extraction
    valid = edges['geometry'].notna() & edges['geometry'].apply(lambda g: g is not None and hasattr(g, 'coords'))
    valid_edges = edges.loc[valid]

    start_coords = valid_edges['geometry'].apply(lambda g: g.coords[0])
    end_coords = valid_edges['geometry'].apply(lambda g: g.coords[-1])

    # Build node dict from start and end points (first occurrence wins)
    nodes_dict = {}
    for osmid, coord in zip(valid_edges['start_node_osmid'], start_coords):
        if osmid not in nodes_dict:
            nodes_dict[osmid] = {'osmid': osmid, 'x': coord[0], 'y': coord[1], 'geometry': Point(coord)}
    for osmid, coord in zip(valid_edges['end_node_osmid'], end_coords):
        if osmid not in nodes_dict:
            nodes_dict[osmid] = {'osmid': osmid, 'x': coord[0], 'y': coord[1], 'geometry': Point(coord)}
    return gpd.GeoDataFrame(list(nodes_dict.values()), crs=edges.crs)


def build_adjacency_graph(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame = None) -> Dict[int, List[Tuple[int, int, float]]]:
    print("  Building adjacency graph...")
    adj = defaultdict(list)
    for idx, start_node, end_node, bearing in zip(edges.index, edges['start_node_osmid'], edges['end_node_osmid'], edges['bearing']):
        adj[start_node].append((idx, end_node, bearing))
        adj[end_node].append((idx, start_node, (bearing + 180) % 360))
    return dict(adj)


def create_grid_bounding_box(edges: gpd.GeoDataFrame, grid_size: float = 1000.0) -> Tuple[List, Dict]:
    print("  Creating grid bounding box...")
    bounds = edges.total_bounds
    minx, miny, maxx, maxy = bounds
    n_cols = int(np.ceil((maxx - minx) / grid_size))
    n_rows = int(np.ceil((maxy - miny) / grid_size))
    print(f"  Grid: {n_cols} columns x {n_rows} rows ({n_cols * n_rows} cells)")
    grid_cells = []
    for col in range(n_cols):
        for row in range(n_rows):
            x_min = minx + col * grid_size
            y_min = miny + row * grid_size
            grid_cells.append((col, row, shapely_box(x_min, y_min, x_min + grid_size, y_min + grid_size)))
    grid_metadata = {'bounds': bounds, 'n_cols': n_cols, 'n_rows': n_rows, 'grid_size': grid_size, 'minx': minx, 'miny': miny}
    return grid_cells, grid_metadata


def _preassign_nodes_to_grid(node_coords: Dict[int, Tuple[float, float]], grid_metadata: Dict) -> Dict[Tuple[int, int], List[int]]:
    minx = grid_metadata['minx']
    miny = grid_metadata['miny']
    gs = grid_metadata['grid_size']
    n_cols = grid_metadata['n_cols']
    n_rows = grid_metadata['n_rows']
    cell_node_map: Dict[Tuple[int, int], List[int]] = {}
    for nid, (lat, lon) in node_coords.items():
        col = min(max(int((lon - minx) / gs), 0), n_cols - 1)
        row_idx = min(max(int((lat - miny) / gs), 0), n_rows - 1)
        cell_node_map.setdefault((col, row_idx), []).append(nid)
    return cell_node_map


def _detect_blocks_worker(args):
    """
    Phase A worker: left-turn traversal within one grid cell.
    Returns (grid_col, grid_row, block_map, visited_edges, open_chains, node_face_map)

    Per spec Phase 2 Step 2: Uses leftmost turn at each node. Accumulates
    shoelace cross-product. Positive (CCW) = interior face (real block),
    negative (CW) = exterior face (discard). Maintains node -> set of block face IDs.
    """
    grid_col, grid_row, cell_node_ids, adj, node_coords = args
    cell_node_ids_set = set(cell_node_ids)
    visited_edges: Set = set()
    block_map: Dict[str, List[int]] = {}
    open_chains = []
    block_sequence = 1
    # NEW: Track node -> set of block face IDs for intersection node classification
    node_face_map: Dict[int, Set[str]] = defaultdict(set)

    sorted_nodes = sorted(cell_node_ids, key=lambda n: node_coords[n])

    for start_node in sorted_nodes:
        if start_node not in adj:
            continue
        for start_edge_idx, next_node, start_bearing in adj[start_node]:
            edge_key = (start_edge_idx, start_node, next_node)
            if edge_key in visited_edges:
                continue
            cycle = []
            chain_visited = set()
            current_node = start_node
            current_edge_idx = start_edge_idx
            incoming_bearing = start_bearing
            prev_node = None
            _next_node = next_node
            # Shoelace accumulator
            shoelace_sum = 0.0
            traversed_nodes = [start_node]
            max_steps = 1000
            steps = 0
            exited_cell = False
            cycle_closed = False

            while steps < max_steps:
                cycle.append(current_edge_idx)
                ek = (current_edge_idx, current_node, _next_node)
                visited_edges.add(ek)
                chain_visited.add(ek)

                # Accumulate shoelace cross-product
                if current_node in node_coords and _next_node in node_coords:
                    y1, x1 = node_coords[current_node]
                    y2, x2 = node_coords[_next_node]
                    shoelace_sum += (x1 * y2 - x2 * y1)

                prev_node = current_node
                current_node = _next_node
                traversed_nodes.append(current_node)

                if current_node not in cell_node_ids_set:
                    open_chains.append({
                        'start_node': start_node, 'start_edge_idx': start_edge_idx,
                        'cycle': list(cycle), 'visited': set(chain_visited),
                        'current_node': current_node, 'current_edge_idx': current_edge_idx,
                        'incoming_bearing': incoming_bearing, 'prev_node': prev_node,
                        'next_node': current_node, 'shoelace_sum': shoelace_sum,
                        'traversed_nodes': list(traversed_nodes),
                    })
                    exited_cell = True
                    break

                if current_node not in adj:
                    break

                candidates = [(e_idx, other, ob) for e_idx, other, ob in adj[current_node]
                              if not (e_idx == current_edge_idx and other == prev_node)]
                if not candidates:
                    break

                # Leftmost turn: smallest clockwise angle from incoming bearing
                turn_angles = sorted(((ob - incoming_bearing) % 360, e_idx, other, ob) for e_idx, other, ob in candidates)
                _, next_edge_idx, next_node_id, next_bearing = turn_angles[0]

                if next_node_id == start_node and next_edge_idx == start_edge_idx:
                    # Cycle closed — check winding order
                    # Positive shoelace = CCW = interior face (real block)
                    # Negative = CW = exterior face (discard)
                    # Zero = degenerate dead-end block (store)
                    cycle_closed = True
                    if shoelace_sum >= 0:
                        block_id = f"{grid_col}_{grid_row}_{block_sequence}"
                        block_map[block_id] = cycle
                        block_sequence += 1
                        # Record node -> face membership
                        for n in traversed_nodes:
                            node_face_map[n].add(block_id)
                    # Discard CW (exterior) faces only
                    break

                current_edge_idx = next_edge_idx
                _next_node = next_node_id
                incoming_bearing = next_bearing
                steps += 1
            else:
                # max_steps reached — terminally bounded
                if cycle:
                    block_id = f"{grid_col}_{grid_row}_{block_sequence}"
                    block_map[block_id] = cycle
                    block_sequence += 1
                    for n in traversed_nodes:
                        node_face_map[n].add(block_id)

            # Dead-end blocks that didn't exit cell and didn't close a cycle
            if not exited_cell and not cycle_closed and cycle and steps < max_steps:
                already_stored = any(cycle == v for v in block_map.values())
                if not already_stored:
                    block_id = f"{grid_col}_{grid_row}_{block_sequence}"
                    block_map[block_id] = cycle
                    block_sequence += 1
                    for n in traversed_nodes:
                        node_face_map[n].add(block_id)

    return grid_col, grid_row, block_map, visited_edges, open_chains, dict(node_face_map)


def _stitch_open_chains(open_chains, adj, node_coords, global_visited, grid_metadata):
    if not open_chains:
        return {}, {}
    minx = grid_metadata['minx']
    miny = grid_metadata['miny']
    gs = grid_metadata['grid_size']
    n_cols = grid_metadata['n_cols']
    n_rows = grid_metadata['n_rows']

    def _node_to_cell(node_id):
        lat, lon = node_coords[node_id]
        col = min(max(int((lon - minx) / gs), 0), n_cols - 1)
        row = min(max(int((lat - miny) / gs), 0), n_rows - 1)
        return col, row

    block_map: Dict[str, List[int]] = {}
    node_face_map: Dict[int, Set[str]] = defaultdict(set)
    cell_seq_counters: Dict[Tuple[int, int], int] = {}
    stitch_visited: Set = set(global_visited)

    for chain in open_chains:
        start_node = chain['start_node']
        start_edge_idx = chain['start_edge_idx']
        cycle = list(chain['cycle'])
        current_node = chain['current_node']
        current_edge_idx = chain['current_edge_idx']
        incoming_bearing = chain['incoming_bearing']
        prev_node = chain['prev_node']
        shoelace_sum = chain.get('shoelace_sum', 0.0)
        traversed_nodes = list(chain.get('traversed_nodes', []))

        resume_ek = (current_edge_idx, prev_node, current_node)
        if resume_ek in stitch_visited:
            continue

        max_steps = 2000
        steps = 0
        stored = False
        while steps < max_steps:
            if current_node not in adj:
                break
            candidates = [(e_idx, other, ob) for e_idx, other, ob in adj[current_node]
                          if not (e_idx == current_edge_idx and other == prev_node)]
            if not candidates:
                break
            turn_angles = sorted(((ob - incoming_bearing) % 360, e_idx, other, ob) for e_idx, other, ob in candidates)
            _, next_edge_idx, next_node_id, next_bearing = turn_angles[0]
            ek = (next_edge_idx, current_node, next_node_id)
            if ek in stitch_visited:
                break
            cycle.append(next_edge_idx)
            stitch_visited.add(ek)
            # Accumulate shoelace
            if current_node in node_coords and next_node_id in node_coords:
                y1, x1 = node_coords[current_node]
                y2, x2 = node_coords[next_node_id]
                shoelace_sum += (x1 * y2 - x2 * y1)
            traversed_nodes.append(next_node_id)

            if next_node_id == start_node and next_edge_idx == start_edge_idx:
                # Only store CCW (interior) and zero-area (degenerate dead-end) faces
                if shoelace_sum >= 0:
                    owner_col, owner_row = _node_to_cell(start_node)
                    cell_key = (owner_col, owner_row)
                    if cell_key not in cell_seq_counters:
                        cell_seq_counters[cell_key] = 1
                    block_id = f"{owner_col}_{owner_row}_s{cell_seq_counters[cell_key]}"
                    cell_seq_counters[cell_key] += 1
                    block_map[block_id] = cycle
                    for n in traversed_nodes:
                        node_face_map[n].add(block_id)
                stored = True
                break

            prev_node = current_node
            current_node = next_node_id
            current_edge_idx = next_edge_idx
            incoming_bearing = next_bearing
            steps += 1
        else:
            if cycle:
                owner_col, owner_row = _node_to_cell(start_node)
                cell_key = (owner_col, owner_row)
                if cell_key not in cell_seq_counters:
                    cell_seq_counters[cell_key] = 1
                block_id = f"{owner_col}_{owner_row}_s{cell_seq_counters[cell_key]}"
                cell_seq_counters[cell_key] += 1
                block_map[block_id] = cycle
                for n in traversed_nodes:
                    node_face_map[n].add(block_id)
                stored = True

        # Store partial chains that broke out without being stored
        if not stored and cycle:
            cycle_fs = frozenset(cycle)
            already_stored = any(frozenset(v) == cycle_fs for v in block_map.values())
            if not already_stored:
                owner_col, owner_row = _node_to_cell(start_node)
                cell_key = (owner_col, owner_row)
                if cell_key not in cell_seq_counters:
                    cell_seq_counters[cell_key] = 1
                block_id = f"{owner_col}_{owner_row}_s{cell_seq_counters[cell_key]}"
                cell_seq_counters[cell_key] += 1
                block_map[block_id] = cycle
                for n in traversed_nodes:
                    node_face_map[n].add(block_id)

    return block_map, dict(node_face_map)


def detect_blocks(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame = None,
                  grid_size: float = 1000.0, n_jobs: int = None) -> Tuple[Dict[str, List[int]], Dict[int, Set[str]]]:
    """
    Detect blocks. Returns (block_map, node_face_map).
    node_face_map: node_id -> set of block face IDs that touch this node.
    Per spec: intersection nodes are those referenced by 3+ distinct block faces.
    """
    print("  Detecting blocks with two-phase approach...")
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    adj = build_adjacency_graph(edges, nodes)
    node_coords: Dict[int, Tuple[float, float]] = {}
    for idx, row in edges.iterrows():
        geom = row['geometry']
        if geom:
            coords = list(geom.coords)
            sid = row['start_node_osmid']
            eid = row['end_node_osmid']
            if sid not in node_coords:
                node_coords[sid] = (coords[0][1], coords[0][0])
            if eid not in node_coords:
                node_coords[eid] = (coords[-1][1], coords[-1][0])

    grid_cells, grid_metadata = create_grid_bounding_box(edges, grid_size)
    cell_node_map = _preassign_nodes_to_grid(node_coords, grid_metadata)
    grid_keys_sorted = sorted(cell_node_map.keys(), key=lambda k: (k[0], k[1]))
    worker_args = [(col, row, cell_node_map[(col, row)], adj, node_coords) for col, row in grid_keys_sorted]

    print(f"  Phase A: Processing {len(worker_args)} non-empty grid cells...")
    block_map: Dict[str, List[int]] = {}
    all_open_chains = []
    global_visited: Set = set()
    global_node_face_map: Dict[int, Set[str]] = defaultdict(set)

    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(_detect_blocks_worker, worker_args,
                      chunksize=max(1, len(worker_args) // (n_jobs * 4))),
                      total=len(worker_args), desc="  Phase A: grid cells"))

    for grid_col, grid_row, cell_blocks, cell_visited, cell_open_chains, cell_nfm in results:
        for bid, eids in cell_blocks.items():
            block_map[bid] = eids
        global_visited.update(cell_visited)
        all_open_chains.extend(cell_open_chains)
        for nid, faces in cell_nfm.items():
            global_node_face_map[nid].update(faces)

    closed_count = len(block_map)
    print(f"  Phase A complete: {closed_count} closed blocks, {len(all_open_chains)} open chains")

    if all_open_chains:
        print(f"  Phase B: Stitching {len(all_open_chains)} boundary-crossing traversals...")
        stitched_blocks, stitched_nfm = _stitch_open_chains(all_open_chains, adj, node_coords, global_visited, grid_metadata)
        block_map.update(stitched_blocks)
        for nid, faces in stitched_nfm.items():
            global_node_face_map[nid].update(faces)
        print(f"  Phase B complete: {len(stitched_blocks)} additional blocks")

    # Dedup
    print("  Deduplicating blocks...")
    edge_set_to_candidates: Dict[frozenset, List[Tuple[str, List[int]]]] = {}
    for bid, eids in block_map.items():
        key = frozenset(eids)
        edge_set_to_candidates.setdefault(key, []).append((bid, eids))
    deduped_map: Dict[str, List[int]] = {}
    for key, candidates in edge_set_to_candidates.items():
        if len(candidates) == 1:
            deduped_map[candidates[0][0]] = candidates[0][1]
        else:
            def _parse_cell(bid):
                parts = bid.replace('s', '').split('_')
                try:
                    return (int(parts[1]), int(parts[0]))
                except (ValueError, IndexError):
                    return (999999, 999999)
            candidates.sort(key=lambda c: _parse_cell(c[0]))
            deduped_map[candidates[0][0]] = candidates[0][1]

    print(f"  Detected {len(deduped_map)} blocks")
    return deduped_map, dict(global_node_face_map)

# --- Block Side Assignment & Node Classification (Phase 2, Steps 2-4) ---

def _compute_block_side_worker(args):
    block_id, edge_ids, edges_coords = args
    coords = [coord for eid in edge_ids for coord in (edges_coords.get(eid) or [])]
    if len(coords) < 3:
        return block_id, []
    coords_array = np.array(coords, dtype=np.float64)
    x = coords_array[:, 0]
    y = coords_array[:, 1]
    signed_area = 0.5 * (np.sum(x[:-1] * y[1:]) + x[-1] * y[0] - np.sum(y[:-1] * x[1:]) - y[-1] * x[0])
    side = 'left' if signed_area > 0 else 'right'
    return block_id, [(eid, side, block_id) for eid in edge_ids]


def assign_block_sides_shoelace(block_map, edges, node_face_map):
    """
    Assign block IDs and sides. Classify block nodes and intersection nodes.
    Per spec Phase 2 Step 3: set start/end_node_is_block_node for vertices of
    detected block polygons. Set start/end_node_is_intersection_node for nodes
    referenced by 3+ distinct block faces.
    """
    print("  Assigning block sides using shoelace formula...")
    all_edge_ids = set()
    for eids in block_map.values():
        all_edge_ids.update(eids)
    minimal_coords = {}
    for eid in all_edge_ids:
        if eid in edges.index:
            geom = edges.loc[eid, 'geometry']
            if geom is not None and not geom.is_empty:
                minimal_coords[eid] = list(geom.coords)

    n_jobs = max(1, mp.cpu_count() - 1)
    sorted_blocks = sorted(block_map.items(), key=lambda x: len(x[1]), reverse=True)
    args = [(bid, eids, minimal_coords) for bid, eids in sorted_blocks]
    edge_block_membership = defaultdict(lambda: [None, None])

    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(_compute_block_side_worker, args, chunksize=max(1, len(args) // (n_jobs * 4))),
                      total=len(args), desc="  Shoelace assignment"))

    for block_id, edge_assignments in results:
        for eid, side, bid in edge_assignments:
            if side == 'left':
                edge_block_membership[eid][0] = bid
            else:
                edge_block_membership[eid][1] = bid

    # Block side labels from circular mean of bearings
    print("  Computing block sides from circular mean of bearings...")
    edge_bearings = {}
    for eid in edge_block_membership:
        if eid in edges.index:
            geom = edges.loc[eid, 'geometry']
            if geom is not None and not geom.is_empty:
                edge_bearings[eid] = compute_bearing(geom) % 180

    block_side_edges = defaultdict(list)
    for eid, (lb, rb) in edge_block_membership.items():
        if lb:
            block_side_edges[(lb, 'left')].append(eid)
        if rb:
            block_side_edges[(rb, 'right')].append(eid)

    block_side_bearings = {}
    for (bid, side), elist in block_side_edges.items():
        bearings = [edge_bearings[e] for e in elist if e in edge_bearings]
        if bearings:
            block_side_bearings[(bid, side)] = compute_circular_mean_bearing(bearings)

    def bearing_to_label(b):
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        return directions[int((b + 22.5) / 45) % 8]

    edge_block_side_membership = {}
    for eid, (lb, rb) in edge_block_membership.items():
        ls = f"{lb}_{bearing_to_label(block_side_bearings.get((lb, 'left'), 0))}" if lb else None
        rs = f"{rb}_{bearing_to_label(block_side_bearings.get((rb, 'right'), 0))}" if rb else None
        edge_block_side_membership[eid] = (ls, rs)

    return dict(edge_block_membership), edge_block_side_membership, node_face_map


def apply_block_ids_to_network(edges, edge_block_membership, edge_block_side_membership, node_face_map):
    """
    Apply block IDs, block sides, block node flags, and intersection node flags.
    Per spec: start/end_node_is_intersection_node = True if node referenced by 3+ block faces.
    """
    print("  Applying block IDs and node classification to network...")
    edges['block_ids'] = None
    edges['block_sides'] = None
    edges['start_node_is_block_node'] = False
    edges['end_node_is_block_node'] = False
    edges['start_node_is_intersection_node'] = False
    edges['end_node_is_intersection_node'] = False
    for side in ['left', 'right']:
        edges[f'sidewalk_{side}_block_ID'] = None
        edges[f'bikeway_{side}_1_block_id'] = None
        edges[f'bikeway_{side}_2_block_id'] = None

    block_vertex_nodes = set()
    for eid, (lb, rb) in edge_block_membership.items():
        if eid in edges.index and (lb or rb):
            block_vertex_nodes.add(edges.loc[eid, 'start_node_osmid'])
            block_vertex_nodes.add(edges.loc[eid, 'end_node_osmid'])

    # Intersection nodes: referenced by 3+ distinct block faces
    intersection_nodes = {nid for nid, faces in node_face_map.items() if len(faces) >= 3}

    # Vectorized block membership assignment
    membership_idx = edges.index.intersection(list(edge_block_membership.keys()))
    if len(membership_idx) > 0:
        lid_series = pd.Series({eid: edge_block_membership[eid][0] for eid in membership_idx}, dtype=object)
        rid_series = pd.Series({eid: edge_block_membership[eid][1] for eid in membership_idx}, dtype=object)
        ls_series = pd.Series({eid: edge_block_side_membership.get(eid, (None, None))[0] for eid in membership_idx}, dtype=object)
        rs_series = pd.Series({eid: edge_block_side_membership.get(eid, (None, None))[1] for eid in membership_idx}, dtype=object)

        edges.loc[membership_idx, 'block_ids'] = list(zip(lid_series, rid_series))
        edges.loc[membership_idx, 'block_sides'] = list(zip(ls_series, rs_series))
        edges.loc[membership_idx, 'sidewalk_left_block_ID'] = lid_series.values
        edges.loc[membership_idx, 'sidewalk_right_block_ID'] = rid_series.values
        edges.loc[membership_idx, 'bikeway_left_1_block_id'] = lid_series.values
        edges.loc[membership_idx, 'bikeway_right_1_block_id'] = rid_series.values
        edges.loc[membership_idx, 'bikeway_left_2_block_id'] = lid_series.values
        edges.loc[membership_idx, 'bikeway_right_2_block_id'] = rid_series.values

    # Vectorized node classification using .isin()
    sn = edges['start_node_osmid']
    en = edges['end_node_osmid']
    edges['start_node_is_block_node'] = sn.isin(block_vertex_nodes)
    edges['end_node_is_block_node'] = en.isin(block_vertex_nodes)
    edges['start_node_is_intersection_node'] = sn.isin(intersection_nodes)
    edges['end_node_is_intersection_node'] = en.isin(intersection_nodes)

    print(f"  {len(block_vertex_nodes)} block nodes, {len(intersection_nodes)} intersection nodes")
    return edges


def reassign_facility_data_by_bearing(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Swap left/right facility data for segments with bearing >= 180."""
    print("  Reassigning facility data based on normalized bearing...")
    swap_mask = (edges['bearing'] >= 180) & (edges['bearing'] < 360)
    if not swap_mask.any():
        return edges
    swap_indices = edges.index[swap_mask]

    # Build pairs of columns to swap
    swap_pairs = []
    for slot in ['', '_ID', '_block_ID', '_presence', '_surface', '_quality', '_width', '_incline', '_buffered', '_geometry']:
        swap_pairs.append((f'sidewalk_left{slot}', f'sidewalk_right{slot}'))
    for slot in ['_feature_ids', '_feature_types', '_feature_geometry', '_feature_geometry_projected']:
        swap_pairs.append((f'sidewalk_left{slot}', f'sidewalk_right{slot}'))
    swap_pairs.append(('public_data_id_sidewalk_left', 'public_data_id_sidewalk_right'))
    swap_pairs.append(('public_data_id_sidewalk_left_feature', 'public_data_id_sidewalk_right_feature'))
    for pos in [1, 2, 3]:
        for slot in ['start', 'end']:
            for attr in ['_ID', '_returnloc', '_returnposition', '_condition_score', '_geometry']:
                swap_pairs.append((f'sidewalk_left_curbramp_{slot}_{pos}{attr}', f'sidewalk_right_curbramp_{slot}_{pos}{attr}'))
            swap_pairs.append((f'public_data_id_sidewalk_left_curbramp_{slot}_{pos}', f'public_data_id_sidewalk_right_curbramp_{slot}_{pos}'))
    for n in [1, 2]:
        for slot in ['_id', '_block_id', '_type', '_surface', '_quality', '_permitted', '_width', '_incline', '_geometry']:
            swap_pairs.append((f'bikeway_left_{n}{slot}', f'bikeway_right_{n}{slot}'))
        for slot in ['_feature_ids', '_feature_types', '_feature_geometry', '_feature_geometry_projected']:
            swap_pairs.append((f'bikeway_left_{n}{slot}', f'bikeway_right_{n}{slot}'))
        swap_pairs.append((f'public_data_id_bikeway_left_{n}', f'public_data_id_bikeway_right_{n}'))
        swap_pairs.append((f'public_data_id_bikeway_left_{n}_features', f'public_data_id_bikeway_right_{n}_features'))
    swap_pairs.append(('bikeway_left_buffered', 'bikeway_right_buffered'))

    for left_col, right_col in swap_pairs:
        if left_col in edges.columns and right_col in edges.columns:
            lv = edges.loc[swap_indices, left_col].copy()
            rv = edges.loc[swap_indices, right_col].copy()
            edges.loc[swap_indices, left_col] = rv
            edges.loc[swap_indices, right_col] = lv
    return edges

# --- Crosswalk Tag Extraction (Phase 1 cache + Phase 4) ---

def extract_crosswalk_tags(network, crossings_cache, cw_counter, gov_crosswalks=None):
    print("  Extracting crosswalk tags...")
    for slot in ['start', 'end']:
        for col in [f'crosswalk_{slot}_id', f'crosswalk_{slot}_block_ids', f'crosswalk_{slot}_type',
                    f'public_data_id_crosswalk_{slot}', f'crosswalk_{slot}_controlled',
                    f'crosswalk_{slot}_marked', f'crosswalk_{slot}_markings', f'crosswalk_{slot}_signals',
                    f'crosswalk_{slot}_island', f'crosswalk_{slot}_kerb', f'crosswalk_{slot}_tactile_paving',
                    f'crosswalk_{slot}_traffic_calming', f'crosswalk_{slot}_continuous',
                    f'crosswalk_{slot}_condition', f'crosswalk_{slot}_geometry',
                    f'crosswalk_{slot}_island_geometry']:
            network[col] = None
    for side in ['left', 'right']:
        for slot in ['start', 'end']:
            for pos in [1, 2, 3]:
                col = f'sidewalk_{side}_curbramp_{slot}_{pos}_geometry'
                if col not in network.columns:
                    network[col] = None

    gov_tree = None
    if gov_crosswalks is not None and not gov_crosswalks.empty:
        gov_tree = STRtree(gov_crosswalks.geometry)
    crossing_tree = None
    if crossings_cache is not None and not crossings_cache.empty:
        crossing_tree = STRtree(crossings_cache.geometry)
    if gov_tree is None and crossing_tree is None:
        print("  No crossing data available")
        return network

    def _extract_crossing_attrs(crossing, slot_prefix, idx, is_government):
        network.at[idx, f'{slot_prefix}_id'] = cw_counter.next()
        if is_government:
            gov_id = crossing.get('id', crossing.get('ID', crossing.get('OBJECTID', None)))
            network.at[idx, f'public_data_id_{slot_prefix.replace("crosswalk_", "crosswalk_")}'] = str(gov_id) if gov_id else None
        network.at[idx, f'{slot_prefix}_type'] = normalize_tag(crossing.get('crossing', crossing.get('type', None)))
        network.at[idx, f'{slot_prefix}_controlled'] = normalize_tag(crossing.get('traffic_signals', None))
        cv = normalize_tag(crossing.get('crossing', crossing.get('type', None)))
        if cv in ['marked', 'zebra', 'tiger']:
            network.at[idx, f'{slot_prefix}_marked'] = 'yes'
        elif cv == 'unmarked':
            network.at[idx, f'{slot_prefix}_marked'] = 'no'
        network.at[idx, f'{slot_prefix}_markings'] = normalize_tag(crossing.get('crossing:markings', None))
        network.at[idx, f'{slot_prefix}_tactile_paving'] = normalize_tag(crossing.get('tactile_paving', None))
        network.at[idx, f'{slot_prefix}_island'] = normalize_tag(crossing.get('crossing:island', None))
        network.at[idx, f'{slot_prefix}_kerb'] = normalize_tag(crossing.get('kerb', None))
        network.at[idx, f'{slot_prefix}_traffic_calming'] = normalize_tag(crossing.get('traffic_calming', None))
        network.at[idx, f'{slot_prefix}_continuous'] = normalize_tag(crossing.get('crossing:continuous', None))
        network.at[idx, f'{slot_prefix}_condition'] = normalize_tag(crossing.get('condition', None))
        signals_present = normalize_tag(crossing.get('crossing:signals', None))
        if signals_present is None and normalize_tag(crossing.get('traffic_signals', None)) is not None:
            signals_present = 'yes'
        network.at[idx, f'{slot_prefix}_signals'] = [
            signals_present, normalize_tag(crossing.get('button_operated', None)),
            normalize_tag(crossing.get('traffic_signals:sound', None)),
            normalize_tag(crossing.get('traffic_signals:vibration', None)),
            normalize_tag(crossing.get('flashing_lights', None))]
        lb = network.at[idx, 'sidewalk_left_block_ID']
        rb = network.at[idx, 'sidewalk_right_block_ID']
        network.at[idx, f'{slot_prefix}_block_ids'] = (rb, lb)

    for idx in tqdm(network.index, total=len(network), desc="  Matching crosswalks"):
        geom = network.at[idx, 'geometry']
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        for slot, pt in [('start', Point(coords[0])), ('end', Point(coords[-1]))]:
            crossing = crossing_point = None
            is_gov = False
            if gov_tree is not None:
                nearby = gov_tree.query(pt, predicate='dwithin', distance=10.0)
                if len(nearby) > 0:
                    crossing = gov_crosswalks.iloc[nearby[0]]
                    crossing_point = crossing.geometry
                    is_gov = True
            if crossing is None and crossing_tree is not None:
                nearby = crossing_tree.query(pt, predicate='dwithin', distance=10.0)
                if len(nearby) > 0:
                    crossing = crossings_cache.iloc[nearby[0]]
                    crossing_point = crossing.geometry
            if crossing is not None:
                _extract_crossing_attrs(crossing, f'crosswalk_{slot}', idx, is_gov)
                if crossing_point.geom_type == 'LineString':
                    network.at[idx, f'crosswalk_{slot}_geometry'] = crossing_point
                else:
                    lr = network.at[idx, f'sidewalk_left_curbramp_{slot}_1_geometry']
                    rr = network.at[idx, f'sidewalk_right_curbramp_{slot}_1_geometry']
                    if lr and rr:
                        network.at[idx, f'crosswalk_{slot}_geometry'] = LineString([
                            lr.coords[0], crossing_point.coords[0], rr.coords[0]])
    return network

# --- Intersection Analysis (Phase 3) ---

def _enumerate_corners(approaching_segments, edges):
    """Sort approaching segments by bearing, identify angular sectors between consecutive pairs."""
    if len(approaching_segments) < 2:
        return []

    # Compute outgoing bearing for each segment at the intersection
    seg_bearings = []
    for edge_idx, pos in approaching_segments:
        bearing = edges.loc[edge_idx, 'bearing']
        if pos == 'end':
            bearing = (bearing + 180) % 360  # Reverse bearing for segments arriving at their end
        seg_bearings.append((bearing, edge_idx, pos))

    seg_bearings.sort(key=lambda x: x[0])

    # Detect T-intersection stem
    stem_idx = None
    if len(seg_bearings) == 3:
        for i, (b, eidx, pos) in enumerate(seg_bearings):
            opposite = (b + 180) % 360
            has_match = any(abs(_angle_between_bearings(ob, opposite)) < 30 or abs(_angle_between_bearings(opposite, ob)) < 30
                           for j, (ob, _, _) in enumerate(seg_bearings) if j != i)
            if not has_match:
                stem_idx = i
                break

    corners = []
    n = len(seg_bearings)
    for i in range(n):
        j = (i + 1) % n
        b_a, eidx_a, pos_a = seg_bearings[i]
        b_b, eidx_b, pos_b = seg_bearings[j]

        interior_angle = (b_b - b_a) % 360

        # Skip near-parallel merges (< 15 degrees)
        if interior_angle < CORNER_SKIP_ANGLE or interior_angle > (360 - CORNER_SKIP_ANGLE):
            continue

        bisector = _angular_bisector(b_a, b_b)

        # Determine which side of each segment faces this corner
        # The corner is to the right of seg_a (clockwise from its bearing)
        # and to the left of seg_b
        side_a = 'right'
        side_b = 'left'

        is_flat = False
        if len(seg_bearings) == 3 and stem_idx is not None:
            # Flat corner: the corner spanning the through-street across from the stem
            if i != stem_idx and j != stem_idx:
                # This corner is between the two through-street segments
                if interior_angle > 150:
                    is_flat = True

        corners.append({
            'seg_a': (eidx_a, pos_a), 'seg_b': (eidx_b, pos_b),
            'side_a': side_a, 'side_b': side_b,
            'interior_angle': interior_angle, 'bisector_bearing': bisector,
            'is_flat_corner': is_flat,
        })

    return corners


def _compute_intersection_boundary(node_geom, approaching_segments, edges, sanity_buffer):
    """
    Phase 3 Step 1: Define intersection boundary as convex hull of centerline
    endpoints expanded by max offset width. Falls back to circle if < 3 distinct
    endpoints (e.g. T-intersection with collinear endpoints).
    """
    endpoints = []
    max_offset = 0.0
    for edge_idx, pos in approaching_segments:
        geom = edges.loc[edge_idx, 'geometry']
        if geom is None:
            continue
        coords = list(geom.coords)
        pt = Point(coords[0] if pos == 'start' else coords[-1])
        endpoints.append(pt)
        osmid = edges.loc[edge_idx, 'osmid']
        offset = sanity_buffer.get(osmid, 5.0)
        if offset > max_offset:
            max_offset = offset

    if len(endpoints) < 3:
        # Fall back to circle
        return node_geom.buffer(max_offset)

    try:
        hull = MultiPoint(endpoints).convex_hull
        return hull.buffer(max_offset)
    except Exception:
        return node_geom.buffer(max_offset)


def _classify_facility_at_boundary(facility_geom, boundary, pos):
    """
    Phase 3 Step 2: Classify a facility geometry relative to the intersection boundary.
    Returns: (classification, facility_geom)
    Classifications:
      'overshooting' — crosses boundary and continues past intersection node
      'undershooting' — terminates before reaching boundary
      'continuous' — passes through intersection without terminating (both endpoints outside)
      'at_boundary' — endpoint is at or near the boundary
      'none' — does not interact with boundary
    """
    if facility_geom is None or not isinstance(facility_geom, LineString):
        return 'none', facility_geom

    try:
        if not facility_geom.intersects(boundary):
            return 'none', facility_geom

        intersection = facility_geom.intersection(boundary.boundary)
        if intersection.is_empty:
            if boundary.contains(facility_geom):
                return 'undershooting', facility_geom
            return 'none', facility_geom

        coords = list(facility_geom.coords)
        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])
        start_inside = boundary.contains(start_pt)
        end_inside = boundary.contains(end_pt)

        # Continuous through-intersection: both endpoints outside boundary
        # but the linestring crosses through it (two intersection points)
        if not start_inside and not end_inside:
            # Check if the facility actually passes through (not just tangent)
            interior_segment = facility_geom.intersection(boundary)
            if not interior_segment.is_empty:
                interior_length = interior_segment.length if hasattr(interior_segment, 'length') else 0
                if interior_length >= 1.0:
                    return 'continuous', facility_geom
                else:
                    # Tangent or barely clips — treat as undershooting on longer side
                    return 'undershooting', facility_geom
            return 'none', facility_geom

        # Check the endpoint relevant to this segment's position
        endpoint = start_pt if pos == 'start' else end_pt
        if boundary.contains(endpoint):
            return 'overshooting', facility_geom
        else:
            dist_to_boundary = endpoint.distance(boundary.boundary)
            if dist_to_boundary > CONTIGUITY_TOLERANCE:
                return 'undershooting', facility_geom
            return 'at_boundary', facility_geom
    except Exception:
        return 'none', facility_geom



def _trim_to_boundary(facility_geom, boundary, pos):
    """Trim an overshooting facility linestring back to the intersection boundary."""
    try:
        intersection_line = facility_geom.intersection(boundary.boundary)
        if intersection_line.is_empty:
            return facility_geom
        # Find the intersection point closest to the facility endpoint
        coords = list(facility_geom.coords)
        if pos == 'start':
            # Trim from start
            trim_point = facility_geom.interpolate(facility_geom.project(intersection_line))
            dist = facility_geom.project(trim_point)
            return _substring(facility_geom, 0, dist) or facility_geom
        else:
            trim_point = facility_geom.interpolate(facility_geom.project(intersection_line))
            dist = facility_geom.project(trim_point)
            return _substring(facility_geom, dist, facility_geom.length) or facility_geom
    except Exception:
        return facility_geom


def _build_exclusion_zone(corner, node_geom, edges, sanity_buffer):
    """
    Phase 3 Step 7: Build corner exclusion zone — union of street centerline
    buffers (half-street-width), sanity buffer boundary, and any non-buffered
    facility geometry.
    """
    zones = []
    for seg_key in ['seg_a', 'seg_b']:
        edge_idx, pos = corner[seg_key]
        geom = edges.loc[edge_idx, 'geometry']
        if geom is None:
            continue
        highway = normalize_tag(edges.loc[edge_idx].get('highway', 'residential'))
        lane_width = resolve_lane_width(edges.loc[edge_idx], highway)
        half_width = lane_width / 2.0
        try:
            zones.append(geom.buffer(half_width))
        except Exception:
            pass
        # Add non-buffered facility geometries
        side = corner['side_a'] if seg_key == 'seg_a' else corner['side_b']
        for fac in ['sidewalk', 'bikeway']:
            if fac == 'sidewalk':
                fgeom = edges.loc[edge_idx].get(f'sidewalk_{side}_geometry')
                buffered = edges.loc[edge_idx].get(f'sidewalk_{side}_buffered', False)
            else:
                fgeom = edges.loc[edge_idx].get(f'bikeway_{side}_1_geometry')
                buffered = edges.loc[edge_idx].get(f'bikeway_{side}_buffered', False)
            if fgeom is not None and not buffered and isinstance(fgeom, LineString):
                try:
                    zones.append(fgeom.buffer(0.3))
                except Exception:
                    pass
    if zones:
        try:
            return unary_union(zones)
        except Exception:
            pass
    return node_geom.buffer(1.0)


def run_intersection_analysis(network, nodes, sanity_buffer, cr_counter, n_jobs=None):
    """Phase 3: Corner enumeration, geometry normalization, curb ramps, conflict resolution."""
    print("  Running intersection analysis (Phase 3)...")

    # Initialize all curb ramp and curb return columns
    for side in ['left', 'right']:
        for slot in ['start', 'end']:
            for pos in [1, 2, 3]:
                for attr in ['_ID', '_returnloc', '_returnposition', '_condition_score', '_geometry']:
                    col = f'sidewalk_{side}_curbramp_{slot}_{pos}{attr}'
                    if col not in network.columns:
                        network[col] = None
                col = f'public_data_id_sidewalk_{side}_curbramp_{slot}_{pos}'
                if col not in network.columns:
                    network[col] = None
    if 'curb_return_geometry' not in network.columns:
        network['curb_return_geometry'] = None

    # Find intersection nodes from the network
    intersection_nodes = set()
    for idx in network.index:
        if network.at[idx, 'start_node_is_intersection_node']:
            sn = network.at[idx, 'start_node_osmid']
            intersection_nodes.add(sn)
        if network.at[idx, 'end_node_is_intersection_node']:
            en = network.at[idx, 'end_node_osmid']
            intersection_nodes.add(en)

    if not intersection_nodes:
        print("  No intersection nodes found — skipping.")
        return network

    # Build node -> edges index
    node_edges = defaultdict(list)
    for idx in network.index:
        sn = network.at[idx, 'start_node_osmid']
        en = network.at[idx, 'end_node_osmid']
        node_edges[sn].append((idx, 'start'))
        node_edges[en].append((idx, 'end'))

    print(f"  Processing {len(intersection_nodes)} intersection nodes...")
    total_ramps = 0
    total_extensions = 0

    for node_id in tqdm(intersection_nodes, desc="  Intersection analysis"):
        if node_id not in node_edges:
            continue
        approaching = node_edges[node_id]
        if len(approaching) < 2:
            continue

        # Get node geometry
        node_geom = None
        for edge_idx, pos in approaching:
            geom = network.at[edge_idx, 'geometry']
            if geom:
                coords = list(geom.coords)
                node_geom = Point(coords[0] if pos == 'start' else coords[-1])
                break
        if node_geom is None:
            continue

        # Step 0: Enumerate corners
        corners = _enumerate_corners(approaching, network)
        if not corners:
            continue

        # Step 1: Compute intersection boundary
        boundary = _compute_intersection_boundary(node_geom, approaching, network, sanity_buffer)

        # Step 2: Normalize facility geometry at boundary
        # Handle overshooting, continuous-through-intersection, and undershooting
        new_rows_to_add = []
        for edge_idx, pos in approaching:
            for side in ['left', 'right']:
                for fac_col in [f'sidewalk_{side}_geometry', f'bikeway_{side}_1_geometry']:
                    fgeom = network.at[edge_idx, fac_col]
                    if fgeom is None or not isinstance(fgeom, LineString):
                        continue
                    classification, _ = _classify_facility_at_boundary(fgeom, boundary, pos)
                    if classification == 'overshooting':
                        trimmed = _trim_to_boundary(fgeom, boundary, pos)
                        network.at[edge_idx, fac_col] = trimmed
                    elif classification == 'continuous':
                        # Step 3: Split continuous facility at boundary intersection points
                        try:
                            boundary_line = boundary.boundary
                            ix = fgeom.intersection(boundary_line)
                            if ix.is_empty:
                                continue
                            # Collect intersection points
                            ix_points = []
                            if ix.geom_type == 'Point':
                                ix_points = [ix]
                            elif ix.geom_type == 'MultiPoint':
                                ix_points = list(ix.geoms)
                            if len(ix_points) >= 2:
                                # Sort by distance along the linestring
                                dists = sorted([fgeom.project(p) for p in ix_points])
                                # Split into: before boundary, inside boundary (discard), after boundary
                                seg_before = _substring(fgeom, 0, dists[0])
                                seg_after = _substring(fgeom, dists[-1], fgeom.length)
                                if seg_before and seg_before.length > 0.01 and seg_after and seg_after.length > 0.01:
                                    # Keep the segment on this edge's side, create new row for the other
                                    # Determine which segment belongs to this edge
                                    orig_start = Point(fgeom.coords[0])
                                    if pos == 'start':
                                        # This edge starts at the intersection — seg_before is the other side
                                        network.at[edge_idx, fac_col] = seg_after
                                        # Create new row for seg_before
                                        new_row = network.loc[edge_idx].copy()
                                        new_row[fac_col] = seg_before
                                        new_rows_to_add.append(new_row)
                                    else:
                                        # This edge ends at the intersection — seg_after is the other side
                                        network.at[edge_idx, fac_col] = seg_before
                                        new_row = network.loc[edge_idx].copy()
                                        new_row[fac_col] = seg_after
                                        new_rows_to_add.append(new_row)
                        except Exception:
                            pass

        # Add any new rows from continuous-through-intersection splits
        if new_rows_to_add:
            new_df = gpd.GeoDataFrame(new_rows_to_add, crs=network.crs)
            network = pd.concat([network, new_df], ignore_index=True)
            # Rebuild node_edges for this node since we added rows
            node_edges[node_id] = []
            for idx in network.index:
                sn = network.at[idx, 'start_node_osmid']
                en = network.at[idx, 'end_node_osmid']
                if sn == node_id:
                    node_edges[node_id].append((idx, 'start'))
                if en == node_id:
                    node_edges[node_id].append((idx, 'end'))

        # Steps 4-9: Single-pass constraint satisfaction per corner
        exclusion_zone = None
        corner_ramp_positions = []  # For multi-face conflict resolution (step 11)

        for corner in corners:
            eidx_a, pos_a = corner['seg_a']
            eidx_b, pos_b = corner['seg_b']
            side_a = corner['side_a']
            side_b = corner['side_b']
            bisector = corner['bisector_bearing']
            is_flat = corner['is_flat_corner']

            if exclusion_zone is None:
                exclusion_zone = _build_exclusion_zone(corner, node_geom, network, sanity_buffer)

            # Step 5: Check bikelane contiguity
            bw_a = network.at[eidx_a, f'bikeway_{side_a}_1_geometry']
            bw_b = network.at[eidx_b, f'bikeway_{side_b}_1_geometry']
            bw_buf_a = network.at[eidx_a, f'bikeway_{side_a}_buffered']
            bw_buf_b = network.at[eidx_b, f'bikeway_{side_b}_buffered']

            if bw_a is not None and bw_b is not None and (bw_buf_a or bw_buf_b):
                try:
                    ca = list(bw_a.coords)
                    cb = list(bw_b.coords)
                    pa = Point(ca[0] if pos_a == 'start' else ca[-1])
                    pb = Point(cb[0] if pos_b == 'start' else cb[-1])
                    if pa.distance(pb) > CONTIGUITY_TOLERANCE:
                        if bw_buf_a and bw_buf_b:
                            # Step 8: Both buffered — compute meeting point via angular bisector
                            offset_a = compute_offset_distance(network.loc[eidx_a], side_a, 'bikeway', sanity_buffer)
                            offset_b = compute_offset_distance(network.loc[eidx_b], side_b, 'bikeway', sanity_buffer)
                            avg_offset = (offset_a + offset_b) / 2.0
                            max_clamp = min(sanity_buffer.get(network.at[eidx_a, 'osmid'], 10.0),
                                            sanity_buffer.get(network.at[eidx_b, 'osmid'], 10.0))
                            avg_offset = min(avg_offset, max_clamp)
                            meeting_pt_bw = _point_along_bearing(node_geom, bisector, avg_offset)

                            # Walk outward if inside exclusion zone
                            for _ in range(20):
                                if exclusion_zone is None or not exclusion_zone.contains(meeting_pt_bw):
                                    break
                                avg_offset += 0.5
                                if avg_offset > max_clamp:
                                    meeting_pt_bw = None
                                    break
                                meeting_pt_bw = _point_along_bearing(node_geom, bisector, avg_offset)

                            if meeting_pt_bw is not None:
                                for eidx, pos, bw_geom, side in [(eidx_a, pos_a, bw_a, side_a), (eidx_b, pos_b, bw_b, side_b)]:
                                    if bw_geom and isinstance(bw_geom, LineString):
                                        c = list(bw_geom.coords)
                                        if pos == 'end':
                                            c.append(meeting_pt_bw.coords[0])
                                        else:
                                            c.insert(0, meeting_pt_bw.coords[0])
                                        network.at[eidx, f'bikeway_{side}_1_geometry'] = LineString(c)
                                try:
                                    exclusion_zone = unary_union([exclusion_zone, meeting_pt_bw.buffer(0.5)])
                                except Exception:
                                    pass
                                total_extensions += 1
                            else:
                                # Step 9: No valid placement — flag corner for manual review
                                corner['needs_manual_review'] = True
                                print(f"    Warning: No valid bikelane placement at node {node_id}, corner flagged for manual review")
                        else:
                            # One buffered, one stored — extend buffered to meet stored
                            if bw_buf_a and not bw_buf_b:
                                target_pt = pb
                                extend_eidx, extend_pos, extend_side = eidx_a, pos_a, side_a
                            else:
                                target_pt = pa
                                extend_eidx, extend_pos, extend_side = eidx_b, pos_b, side_b
                            ext_geom = network.at[extend_eidx, f'bikeway_{extend_side}_1_geometry']
                            if ext_geom and isinstance(ext_geom, LineString):
                                c = list(ext_geom.coords)
                                if extend_pos == 'end':
                                    c.append(target_pt.coords[0])
                                else:
                                    c.insert(0, target_pt.coords[0])
                                network.at[extend_eidx, f'bikeway_{extend_side}_1_geometry'] = LineString(c)
                                try:
                                    exclusion_zone = unary_union([exclusion_zone, target_pt.buffer(0.5)])
                                except Exception:
                                    pass
                                total_extensions += 1
                except Exception:
                    pass

            # Step 6: Check sidewalk contiguity
            sw_a = network.at[eidx_a, f'sidewalk_{side_a}_geometry']
            sw_b = network.at[eidx_b, f'sidewalk_{side_b}_geometry']
            sw_buf_a = network.at[eidx_a, f'sidewalk_{side_a}_buffered']
            sw_buf_b = network.at[eidx_b, f'sidewalk_{side_b}_buffered']

            meeting_pt = None
            if sw_a is not None and sw_b is not None:
                try:
                    ca = list(sw_a.coords)
                    cb = list(sw_b.coords)
                    pa = Point(ca[0] if pos_a == 'start' else ca[-1])
                    pb = Point(cb[0] if pos_b == 'start' else cb[-1])

                    if pa.distance(pb) > CONTIGUITY_TOLERANCE or (sw_buf_a or sw_buf_b):
                        # Compute meeting point via angular bisector
                        offset_a = compute_offset_distance(network.loc[eidx_a], side_a, 'sidewalk', sanity_buffer)
                        offset_b = compute_offset_distance(network.loc[eidx_b], side_b, 'sidewalk', sanity_buffer)
                        avg_offset = (offset_a + offset_b) / 2.0
                        max_clamp = min(sanity_buffer.get(network.at[eidx_a, 'osmid'], 10.0),
                                        sanity_buffer.get(network.at[eidx_b, 'osmid'], 10.0))
                        avg_offset = min(avg_offset, max_clamp)
                        meeting_pt = _point_along_bearing(node_geom, bisector, avg_offset)

                        for _ in range(20):
                            if exclusion_zone is None or not exclusion_zone.contains(meeting_pt):
                                break
                            avg_offset += 0.5
                            if avg_offset > max_clamp:
                                meeting_pt = None
                                break
                            meeting_pt = _point_along_bearing(node_geom, bisector, avg_offset)

                        if meeting_pt is not None:
                            for eidx, pos, sw_geom, side in [(eidx_a, pos_a, sw_a, side_a), (eidx_b, pos_b, sw_b, side_b)]:
                                if sw_geom and isinstance(sw_geom, LineString):
                                    c = list(sw_geom.coords)
                                    if pos == 'end':
                                        c.append(meeting_pt.coords[0])
                                    else:
                                        c.insert(0, meeting_pt.coords[0])
                                    network.at[eidx, f'sidewalk_{side}_geometry'] = LineString(c)
                            total_extensions += 1
                        else:
                            # Step 9: No valid placement — flag corner for manual review
                            corner['needs_manual_review'] = True
                            print(f"    Warning: No valid sidewalk placement at node {node_id}, corner flagged for manual review")
                except Exception:
                    pass

            # Step 8 (undershooting): Extend single undershooting sidewalks to boundary
            for seg_key, seg_side in [('seg_a', side_a), ('seg_b', side_b)]:
                seg_eidx, seg_pos = corner[seg_key]
                sw_geom = network.at[seg_eidx, f'sidewalk_{seg_side}_geometry']
                if sw_geom is None or not isinstance(sw_geom, LineString):
                    continue
                classification, _ = _classify_facility_at_boundary(sw_geom, boundary, seg_pos)
                if classification == 'undershooting':
                    # Extend along existing bearing to boundary, constrained by exclusion zone
                    c = list(sw_geom.coords)
                    if seg_pos == 'start':
                        bearing = compute_bearing(sw_geom)
                        ext_pt = _point_along_bearing(Point(c[0]), (bearing + 180) % 360, CONTIGUITY_TOLERANCE)
                    else:
                        bearing = compute_bearing(sw_geom)
                        ext_pt = _point_along_bearing(Point(c[-1]), bearing, CONTIGUITY_TOLERANCE)
                    if exclusion_zone is None or not exclusion_zone.contains(ext_pt):
                        if seg_pos == 'end':
                            c.append(ext_pt.coords[0])
                        else:
                            c.insert(0, ext_pt.coords[0])
                        network.at[seg_eidx, f'sidewalk_{seg_side}_geometry'] = LineString(c)

            # Step 10: Generate two directional curb ramps per corner
            if is_flat and sw_a is not None and sw_b is not None:
                # Flat corner (T-intersection): single connecting ramp or continuous
                if meeting_pt is not None:
                    ramp_id = cr_counter.next()
                    network.at[eidx_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_ID'] = ramp_id
                    network.at[eidx_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_geometry'] = meeting_pt
                    network.at[eidx_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_ID'] = ramp_id
                    network.at[eidx_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_geometry'] = meeting_pt
                    corner_ramp_positions.append(meeting_pt)
                    total_ramps += 1
            elif meeting_pt is not None:
                # Normal corner: two directional ramps split from meeting point
                sw_width_a = 0.75
                try:
                    w = network.at[eidx_a, f'sidewalk_{side_a}_width']
                    if w is not None:
                        sw_width_a = max(float(w) / 2.0, 0.75)
                except (ValueError, TypeError):
                    pass

                bearing_a = network.at[eidx_a, 'bearing']
                if pos_a == 'end':
                    bearing_a = (bearing_a + 180) % 360
                bearing_b = network.at[eidx_b, 'bearing']
                if pos_b == 'end':
                    bearing_b = (bearing_b + 180) % 360

                ramp_a_pt = _point_along_bearing(meeting_pt, bearing_a, sw_width_a)
                ramp_b_pt = _point_along_bearing(meeting_pt, bearing_b, sw_width_a)

                # Check if curb return is long enough
                curb_return_length = ramp_a_pt.distance(ramp_b_pt)
                if curb_return_length < MIN_CURB_RETURN_LENGTH:
                    # Collapse to single apex ramp
                    ramp_id = cr_counter.next()
                    network.at[eidx_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_ID'] = ramp_id
                    network.at[eidx_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_geometry'] = meeting_pt
                    network.at[eidx_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_ID'] = ramp_id
                    network.at[eidx_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_geometry'] = meeting_pt
                    corner_ramp_positions.append(meeting_pt)
                    total_ramps += 1
                else:
                    # Two directional ramps
                    ramp_a_id = cr_counter.next()
                    ramp_b_id = cr_counter.next()
                    network.at[eidx_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_ID'] = ramp_a_id
                    network.at[eidx_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_geometry'] = ramp_a_pt
                    network.at[eidx_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_ID'] = ramp_b_id
                    network.at[eidx_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_geometry'] = ramp_b_pt

                    # Curb return geometry: arc/line between the two ramps
                    curb_return = LineString([ramp_a_pt.coords[0], ramp_b_pt.coords[0]])
                    # Store on the segment that "owns" this corner
                    network.at[eidx_a, 'curb_return_geometry'] = curb_return

                    corner_ramp_positions.extend([ramp_a_pt, ramp_b_pt])
                    total_ramps += 2

        # Step 11: Multi-face conflict resolution
        # Build owner list by re-walking corners (mirrors the ramp generation order above)
        ramp_position_owners = []
        for corner in corners:
            eidx_a, pos_a = corner['seg_a']
            eidx_b, pos_b = corner['seg_b']
            side_a = corner['side_a']
            side_b = corner['side_b']
            is_flat = corner['is_flat_corner']
            remaining = len(corner_ramp_positions) - len(ramp_position_owners)
            if remaining <= 0:
                break
            # Check if this corner produced a single ramp (flat or collapsed) or two
            next_idx = len(ramp_position_owners)
            if next_idx + 1 < len(corner_ramp_positions) and \
               corner_ramp_positions[next_idx] is not None and \
               corner_ramp_positions[next_idx + 1] is not None and \
               corner_ramp_positions[next_idx].distance(corner_ramp_positions[next_idx + 1]) > 0.001:
                # Two distinct ramps
                ramp_position_owners.append((eidx_a, side_a, pos_a, 1))
                ramp_position_owners.append((eidx_b, side_b, pos_b, 1))
            else:
                # Single ramp (flat corner, collapsed, or last odd ramp)
                ramp_position_owners.append((eidx_a, side_a, pos_a, 1))
                if next_idx + 1 < len(corner_ramp_positions):
                    ramp_position_owners.append((eidx_b, side_b, pos_b, 1))

        if len(corner_ramp_positions) >= 2:
            for i in range(len(corner_ramp_positions)):
                for j in range(i + 1, len(corner_ramp_positions)):
                    if corner_ramp_positions[i] is None or corner_ramp_positions[j] is None:
                        continue
                    if corner_ramp_positions[i].distance(corner_ramp_positions[j]) < MIN_RAMP_SEPARATION:
                        # Merge into midpoint apex ramp, clear curb_return_geometry
                        mid = Point(
                            (corner_ramp_positions[i].x + corner_ramp_positions[j].x) / 2,
                            (corner_ramp_positions[i].y + corner_ramp_positions[j].y) / 2)
                        corner_ramp_positions[i] = mid
                        corner_ramp_positions[j] = mid
                        # Write merged geometry back to the network
                        for k in (i, j):
                            if k < len(ramp_position_owners):
                                eidx_k, side_k, slot_k, pos_k = ramp_position_owners[k]
                                network.at[eidx_k, f'sidewalk_{side_k}_curbramp_{slot_k}_{pos_k}_geometry'] = mid
                                # Snap sidewalk endpoint to merged ramp
                                sw_geom = network.at[eidx_k, f'sidewalk_{side_k}_geometry']
                                if sw_geom is not None and isinstance(sw_geom, LineString):
                                    c = list(sw_geom.coords)
                                    if slot_k == 'start':
                                        c[0] = mid.coords[0]
                                    else:
                                        c[-1] = mid.coords[0]
                                    network.at[eidx_k, f'sidewalk_{side_k}_geometry'] = LineString(c)
                        # Clear curb_return_geometry for merged ramps
                        if i < len(ramp_position_owners):
                            network.at[ramp_position_owners[i][0], 'curb_return_geometry'] = None
                        if j < len(ramp_position_owners):
                            network.at[ramp_position_owners[j][0], 'curb_return_geometry'] = None

        # Check for overlapping corrected facility geometry between adjacent corners
        if len(corners) >= 2:
            for ci in range(len(corners)):
                cj = (ci + 1) % len(corners)
                ca = corners[ci]
                cb = corners[cj]
                # Check if bikelane extensions from adjacent corners cross
                for fac in ['bikeway', 'sidewalk']:
                    col_a = f'{fac}_{ca["side_b"]}_1_geometry' if fac == 'bikeway' else f'{fac}_{ca["side_b"]}_geometry'
                    col_b = f'{fac}_{cb["side_a"]}_1_geometry' if fac == 'bikeway' else f'{fac}_{cb["side_a"]}_geometry'
                    eidx_a_b = ca['seg_b'][0]
                    eidx_b_a = cb['seg_a'][0]
                    geom_a = network.at[eidx_a_b, col_a] if col_a in network.columns else None
                    geom_b = network.at[eidx_b_a, col_b] if col_b in network.columns else None
                    if geom_a is not None and geom_b is not None:
                        try:
                            if isinstance(geom_a, LineString) and isinstance(geom_b, LineString) and geom_a.crosses(geom_b):
                                # Pull both back to midpoint of overlap
                                ix = geom_a.intersection(geom_b)
                                if not ix.is_empty and ix.geom_type == 'Point':
                                    # Trim both to the intersection point
                                    d_a = geom_a.project(ix)
                                    d_b = geom_b.project(ix)
                                    trimmed_a = _substring(geom_a, 0, d_a)
                                    trimmed_b = _substring(geom_b, 0, d_b)
                                    if trimmed_a and trimmed_a.length > 0.01:
                                        network.at[eidx_a_b, col_a] = trimmed_a
                                    if trimmed_b and trimmed_b.length > 0.01:
                                        network.at[eidx_b_a, col_b] = trimmed_b
                        except Exception:
                            pass

    print(f"  Generated {total_ramps} curb ramps, {total_extensions} facility extensions")
    return network

# --- Government Curb Ramp Processing (Phase 3, Steps 12-16) ---

def load_government_curb_ramps(city_config, working_crs=None):
    path = city_config.government_data_paths.curb_ramps
    if path is None or path == '' or not os.path.exists(path):
        return None
    print(f"  Loading curb ramps from {path}...")
    gdf = load_geospatial_file(path)
    if gdf is None or gdf.empty:
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if working_crs is not None:
        gdf = reproject_to_working_crs(gdf, working_crs)
    print(f"  Loaded {len(gdf)} curb ramp records")
    return gdf


def process_government_curb_ramps(network, gov_curb_ramps, city_config, global_config, cr_counter):
    """Phase 3 Steps 12-16: Government curb ramp integration."""
    if gov_curb_ramps is None or gov_curb_ramps.empty:
        return network
    if not city_config.curbramp_trustworthy:
        print("  Curb ramps marked as untrustworthy — skipping government curb ramp processing")
        return network

    print(f"  Processing {len(gov_curb_ramps)} government curb ramps...")
    inner_buffer = global_config.curb_ramp_trustworthiness_inner_buffer
    outer_buffer = global_config.curb_ramp_trustworthiness_outer_buffer

    if gov_curb_ramps.crs != network.crs:
        gov_curb_ramps = gov_curb_ramps.to_crs(network.crs)

    col_map = city_config.column_mappings
    ramp_id_col = col_map.curbramp_id or 'id'
    return_loc_col = col_map.curbramp_return_loc or 'return_loc'
    return_pos_col = col_map.curbramp_position or 'return_position'
    condition_col = col_map.curbramp_condition or 'condition_score'

    processed = inner_count = replaced_count = 0

    # Build STRtree spatial index on government curb ramp geometries
    ramp_geoms = list(gov_curb_ramps.geometry)
    ramp_tree = STRtree(ramp_geoms)
    # Pre-extract ramp attributes into lists for fast indexed access
    ramp_ids = [gov_curb_ramps.iloc[i].get(ramp_id_col, str(gov_curb_ramps.index[i])) for i in range(len(gov_curb_ramps))]
    ramp_return_locs = [gov_curb_ramps.iloc[i].get(return_loc_col, None) for i in range(len(gov_curb_ramps))]
    ramp_return_positions = [gov_curb_ramps.iloc[i].get(return_pos_col, None) for i in range(len(gov_curb_ramps))]
    ramp_conditions = [gov_curb_ramps.iloc[i].get(condition_col, None) for i in range(len(gov_curb_ramps))]

    for idx in tqdm(network.index, desc="  Processing curb ramps"):
        for side in ['left', 'right']:
            for slot in ['start', 'end']:
                default_geom = network.loc[idx, f'sidewalk_{side}_curbramp_{slot}_1_geometry']
                default_id = network.loc[idx, f'sidewalk_{side}_curbramp_{slot}_1_ID']
                if default_geom is None or pd.isna(default_id):
                    continue

                # Query STRtree for ramps within outer_buffer distance
                candidate_indices = ramp_tree.query(default_geom.buffer(outer_buffer))
                nearby_ramps = []
                for ci in candidate_indices:
                    d = default_geom.distance(ramp_geoms[ci])
                    if d <= outer_buffer:
                        nearby_ramps.append({
                            'distance': d, 'geometry': ramp_geoms[ci],
                            'id': ramp_ids[ci],
                            'return_loc': ramp_return_locs[ci],
                            'return_position': ramp_return_positions[ci],
                            'condition_score': ramp_conditions[ci]})
                if not nearby_ramps:
                    continue
                nearby_ramps.sort(key=lambda x: x['distance'])

                # Step 12: Inner buffer
                if nearby_ramps[0]['distance'] <= inner_buffer:
                    r = nearby_ramps[0]
                    network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_1'] = r['id']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_returnloc'] = r['return_loc']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_returnposition'] = r['return_position']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_condition_score'] = r['condition_score']
                    inner_count += 1
                    processed += 1
                    continue

                # Steps 13-14: Goldilocks zone
                goldilocks = [r for r in nearby_ramps if inner_buffer < r['distance'] <= outer_buffer]
                if not goldilocks:
                    continue

                if len(goldilocks) <= 2:
                    # Step 14: Replace closest ramp(s)
                    for ri, r in enumerate(goldilocks[:2]):
                        pos_n = ri + 1
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_{pos_n}_geometry'] = r['geometry']
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_{pos_n}_ID'] = cr_counter.next()
                        network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_{pos_n}'] = r['id']
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_{pos_n}_returnloc'] = r['return_loc']
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_{pos_n}_returnposition'] = r['return_position']
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_{pos_n}_condition_score'] = r['condition_score']
                        # Snap sidewalk endpoint
                        sw_geom = network.loc[idx, f'sidewalk_{side}_geometry']
                        if sw_geom and isinstance(sw_geom, LineString) and pos_n == 1:
                            c = list(sw_geom.coords)
                            if slot == 'start':
                                c[0] = r['geometry'].coords[0]
                            else:
                                c[-1] = r['geometry'].coords[0]
                            network.at[idx, f'sidewalk_{side}_geometry'] = LineString(c)
                    # Recompute curb return if two ramps
                    if len(goldilocks) >= 2:
                        pt1 = goldilocks[0]['geometry']
                        pt2 = goldilocks[1]['geometry']
                        network.at[idx, 'curb_return_geometry'] = LineString([pt1.coords[0], pt2.coords[0]])
                    replaced_count += 1
                    processed += 1

                # Step 15: Additional ramps to slot 3
                if len(goldilocks) >= 3:
                    r = goldilocks[2]
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_ID'] = cr_counter.next()
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_geometry'] = r['geometry']
                    network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_3'] = r['id']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_returnloc'] = r['return_loc']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_returnposition'] = r['return_position']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_condition_score'] = r['condition_score']

                # Step 16: Single apex ramp from government — merge two defaults into one
                if len(nearby_ramps) == 1 and nearby_ramps[0]['distance'] <= outer_buffer:
                    # Check if we have two default ramps at this corner
                    ramp_2_id = network.at[idx, f'sidewalk_{side}_curbramp_{slot}_2_ID']
                    if ramp_2_id is not None and not pd.isna(ramp_2_id):
                        r = nearby_ramps[0]
                        # Merge: update both sidewalk endpoints to government ramp point
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_geometry'] = r['geometry']
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_ID'] = cr_counter.next()
                        network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_1'] = r['id']
                        # Clear second ramp slot
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_2_ID'] = None
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_2_geometry'] = None
                        network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_2'] = None
                        # Snap sidewalk endpoint
                        sw_geom = network.loc[idx, f'sidewalk_{side}_geometry']
                        if sw_geom and isinstance(sw_geom, LineString):
                            c = list(sw_geom.coords)
                            if slot == 'start':
                                c[0] = r['geometry'].coords[0]
                            else:
                                c[-1] = r['geometry'].coords[0]
                            network.at[idx, f'sidewalk_{side}_geometry'] = LineString(c)
                        # Store curb return between original sidewalk endpoints
                        network.at[idx, 'curb_return_geometry'] = None

    print(f"  Processed {processed} curb ramp slots ({inner_count} inner, {replaced_count} replaced)")
    return network


@dataclass
class CurbRamp:
    loc_id: str
    cnn: str
    curb_return_loc: str
    position_on_return: str
    latitude: float
    longitude: float
    condition_score: float
    geometry: Point


def load_curb_ramps(city_config):
    path = city_config.government_data_paths.curb_ramps
    if path is None or path == '' or not os.path.exists(path):
        return []
    try:
        print(f"  Loading curb ramps from: {path}")
        df = pd.read_csv(path)
        curb_ramps = []
        for _, row in df.iterrows():
            if pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')):
                continue
            try:
                lat, lon = float(row['Latitude']), float(row['Longitude'])
            except (ValueError, TypeError):
                continue
            curb_ramps.append(CurbRamp(
                loc_id=str(row.get('LocID', '')), cnn=str(row.get('CNN', '')),
                curb_return_loc=str(row.get('CurbReturnLoc', '')),
                position_on_return=str(row.get('PositionOnReturn', '')),
                latitude=lat, longitude=lon,
                condition_score=float(row.get('conditionScore', -2)),
                geometry=Point(lon, lat)))
        print(f"  Loaded {len(curb_ramps)} curb ramps")
        return curb_ramps
    except Exception as e:
        print(f"  Error loading curb ramps: {e}")
        return []

# --- Implicit Crosswalk Geometry (Phase 4, Step 2) ---

def generate_implicit_crosswalk_geometries(network, cw_counter):
    """Phase 4 Step 2: Draw crosswalk geometry between paired curb ramps where no crosswalk exists."""
    print("  Generating implicit crosswalk geometries...")
    generated = 0
    for slot in ['start', 'end']:
        cw_geom_col = f'crosswalk_{slot}_geometry'
        lr_col = f'sidewalk_left_curbramp_{slot}_1_geometry'
        rr_col = f'sidewalk_right_curbramp_{slot}_1_geometry'

        # Check required columns exist
        if cw_geom_col not in network.columns or lr_col not in network.columns or rr_col not in network.columns:
            continue

        # Build boolean mask: no existing crosswalk AND both ramps are Points
        no_cw = network[cw_geom_col].isna() | network[cw_geom_col].apply(lambda v: v is None)
        has_lr = network[lr_col].apply(lambda v: isinstance(v, Point))
        has_rr = network[rr_col].apply(lambda v: isinstance(v, Point))
        mask = no_cw & has_lr & has_rr
        candidate_indices = network.index[mask]

        if len(candidate_indices) == 0:
            continue

        # Build crosswalk geometries vectorized via list comprehension
        lr_geoms = network.loc[candidate_indices, lr_col]
        rr_geoms = network.loc[candidate_indices, rr_col]
        cw_geoms = [LineString([lr.coords[0], rr.coords[0]]) for lr, rr in zip(lr_geoms, rr_geoms)]
        cw_ids = [cw_counter.next() for _ in range(len(candidate_indices))]

        network.loc[candidate_indices, cw_geom_col] = cw_geoms
        network.loc[candidate_indices, f'crosswalk_{slot}_id'] = cw_ids

        # Island geometry where crosswalk_{slot}_island == 'yes'
        island_col = f'crosswalk_{slot}_island'
        if island_col in network.columns:
            island_mask = network.loc[candidate_indices, island_col] == 'yes'
            island_indices = candidate_indices[island_mask]
            if len(island_indices) > 0:
                island_geoms = [MultiPoint([cw.interpolate(0.5, normalized=True)])
                                for cw in network.loc[island_indices, cw_geom_col]]
                network.loc[island_indices, f'crosswalk_{slot}_island_geometry'] = island_geoms

        generated += len(candidate_indices)
    print(f"  Generated {generated} implicit crosswalk geometries")
    return network


# --- Assemble Network Schema ---

def assemble_network_schema(network: gpd.GeoDataFrame, global_config: GlobalConfig = None) -> gpd.GeoDataFrame:
    """Ensure all spec columns exist, populate derived columns."""
    print("  Assembling network schema...")

    # Street centerlines
    street_cols = [
        'block_ids', 'street_id', 'block_sides', 'public_data_id_street',
        'start_node_osmid', 'start_node_is_block_node', 'start_node_is_intersection_node',
        'end_node_osmid', 'end_node_is_block_node', 'end_node_is_intersection_node',
        'public_data_id_start_end_nodes', 'normalized_bearing', 'name', 'highway',
        'maxspeed', 'oneway', 'lanes', 'lane_width', 'surface',
    ]
    # Street features
    street_feature_cols = [
        'street_feature_types', 'public_data_id_street_feature',
        'street_feature_geometry', 'street_feature_geometry_projected',
    ]
    # Sidewalk columns (left and right)
    sidewalk_cols = []
    for side in ['left', 'right']:
        sidewalk_cols.extend([
            f'sidewalk_{side}_ID', f'sidewalk_{side}_block_ID',
            f'sidewalk_{side}_presence', f'public_data_id_sidewalk_{side}',
            f'sidewalk_{side}_surface', f'sidewalk_{side}_quality',
            f'sidewalk_{side}_width', f'sidewalk_{side}_incline',
            f'sidewalk_{side}_buffered',
        ])
        # Curb ramps (3 positions x start/end)
        for slot in ['start', 'end']:
            for pos in [1, 2, 3]:
                sidewalk_cols.extend([
                    f'sidewalk_{side}_curbramp_{slot}_{pos}_ID',
                    f'public_data_id_sidewalk_{side}_curbramp_{slot}_{pos}',
                    f'sidewalk_{side}_curbramp_{slot}_{pos}_returnloc',
                    f'sidewalk_{side}_curbramp_{slot}_{pos}_returnposition',
                    f'sidewalk_{side}_curbramp_{slot}_{pos}_condition_score',
                    f'sidewalk_{side}_curbramp_{slot}_{pos}_geometry',
                ])
        # Sidewalk features
        sidewalk_cols.extend([
            f'sidewalk_{side}_feature_ids', f'sidewalk_{side}_feature_types',
            f'public_data_id_sidewalk_{side}_feature',
            f'sidewalk_{side}_feature_geometry', f'sidewalk_{side}_feature_geometry_projected',
        ])

    # Crosswalk columns (start and end)
    crosswalk_cols = []
    for slot in ['start', 'end']:
        crosswalk_cols.extend([
            f'crosswalk_{slot}_id', f'crosswalk_{slot}_block_ids',
            f'crosswalk_{slot}_type', f'public_data_id_crosswalk_{slot}',
            f'crosswalk_{slot}_controlled', f'crosswalk_{slot}_marked',
            f'crosswalk_{slot}_markings', f'crosswalk_{slot}_signals',
            f'crosswalk_{slot}_island', f'crosswalk_{slot}_kerb',
            f'crosswalk_{slot}_tactile_paving', f'crosswalk_{slot}_traffic_calming',
            f'crosswalk_{slot}_continuous', f'crosswalk_{slot}_condition',
            f'crosswalk_{slot}_geometry', f'crosswalk_{slot}_island_geometry',
        ])

    # Bikeway columns (left/right x 1/2)
    bikeway_cols = []
    for side in ['left', 'right']:
        for n in [1, 2]:
            bikeway_cols.extend([
                f'bikeway_{side}_{n}_id', f'bikeway_{side}_{n}_block_id',
                f'public_data_id_bikeway_{side}_{n}',
                f'bikeway_{side}_{n}_type', f'bikeway_{side}_{n}_surface',
                f'bikeway_{side}_{n}_quality', f'bikeway_{side}_{n}_permitted',
                f'bikeway_{side}_{n}_width', f'bikeway_{side}_{n}_incline',
            ])
            bikeway_cols.extend([
                f'bikeway_{side}_{n}_feature_ids', f'bikeway_{side}_{n}_feature_types',
                f'public_data_id_bikeway_{side}_{n}_features',
                f'bikeway_{side}_{n}_feature_geometry', f'bikeway_{side}_{n}_feature_geometry_projected',
            ])
        bikeway_cols.append(f'bikeway_{side}_buffered')

    # Main geometry columns
    geom_cols = [
        'street_geometry', 'start_node_geometry', 'end_node_geometry',
        'sidewalk_left_geometry', 'sidewalk_right_geometry',
        'curb_return_geometry',
        'bikeway_left_1_geometry', 'bikeway_left_2_geometry',
        'bikeway_right_1_geometry', 'bikeway_right_2_geometry',
    ]

    all_cols = street_cols + street_feature_cols + sidewalk_cols + crosswalk_cols + bikeway_cols + geom_cols
    added = 0
    for col in all_cols:
        if col not in network.columns:
            network[col] = None
            added += 1

    # Populate derived columns
    if 'street_id' not in network.columns or network['street_id'].isna().all():
        if 'osmid' in network.columns:
            network['street_id'] = network['osmid']
    if 'normalized_bearing' not in network.columns or network['normalized_bearing'].isna().all():
        if 'bearing' in network.columns:
            network['normalized_bearing'] = network['bearing']

    # Populate lane_width from resolve_lane_width where missing
    if 'lane_width' in network.columns and 'highway' in network.columns:
        missing_lw = network['lane_width'].isna()
        if missing_lw.any():
            for idx in network.index[missing_lw]:
                hw = normalize_tag(network.at[idx, 'highway']) or 'residential'
                network.at[idx, 'lane_width'] = resolve_lane_width(network.loc[idx], hw)

    # Apply default maxspeed from config where missing
    if global_config is not None and 'maxspeed' in network.columns:
        default_speed = global_config.default_max_speed
        network['maxspeed'] = network['maxspeed'].fillna(default_speed)

    # Populate geometry columns from source columns (vectorized)
    if 'geometry' in network.columns:
        network['street_geometry'] = network['geometry']
        valid_geom = network['geometry'].notna() & network['geometry'].apply(lambda g: g is not None and not g.is_empty)
        valid_idx = network.index[valid_geom]
        if len(valid_idx) > 0:
            geoms = network.loc[valid_idx, 'geometry']
            network.loc[valid_idx, 'start_node_geometry'] = geoms.apply(lambda g: Point(g.coords[0]))
            network.loc[valid_idx, 'end_node_geometry'] = geoms.apply(lambda g: Point(g.coords[-1]))

    print(f"  Schema assembled: {added} columns added, {len(network.columns)} total")
    return network


# --- Export Network Parquet ---

def _reproject_geom_to_4326(geom, transformer):
    """Reproject a single Shapely geometry to EPSG:4326 using a pyproj Transformer."""
    if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
        return geom
    try:
        return shapely_transform(transformer.transform, geom)
    except Exception:
        return geom


def export_network_parquet(network: gpd.GeoDataFrame, city_name: str, output_path: str) -> None:
    """Export network to parquet with geometry columns serialized as WKB in EPSG:4326."""
    print(f"  Exporting network parquet for {city_name}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Build a transformer from the working CRS to EPSG:4326
    source_crs = network.crs
    needs_reproject = source_crs is not None and str(source_crs) != "EPSG:4326"
    transformer = None
    if needs_reproject:
        transformer = _ProjTransformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        print(f"  Reprojecting geometries from {source_crs} to EPSG:4326 for export...")

    geometry_columns = [
        'street_geometry', 'start_node_geometry', 'end_node_geometry',
        'sidewalk_left_geometry', 'sidewalk_right_geometry',
        'curb_return_geometry',
        'bikeway_left_1_geometry', 'bikeway_left_2_geometry',
        'bikeway_right_1_geometry', 'bikeway_right_2_geometry',
        'street_feature_geometry', 'street_feature_geometry_projected',
        'crosswalk_start_geometry', 'crosswalk_end_geometry',
        'crosswalk_start_island_geometry', 'crosswalk_end_island_geometry',
    ]
    # Add sidewalk feature geometry columns
    for side in ['left', 'right']:
        geometry_columns.extend([
            f'sidewalk_{side}_feature_geometry', f'sidewalk_{side}_feature_geometry_projected',
        ])
        for slot in ['start', 'end']:
            for pos in [1, 2, 3]:
                geometry_columns.append(f'sidewalk_{side}_curbramp_{slot}_{pos}_geometry')
    # Add bikeway feature geometry columns
    for side in ['left', 'right']:
        for n in [1, 2]:
            geometry_columns.extend([
                f'bikeway_{side}_{n}_feature_geometry', f'bikeway_{side}_{n}_feature_geometry_projected',
            ])

    df = network.copy()

    # Reproject the main geometry column first
    if needs_reproject and 'geometry' in df.columns:
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=source_crs).to_crs("EPSG:4326")

    # Reproject and serialize all geometry columns to WKB in EPSG:4326
    for col in geometry_columns:
        if col in df.columns:
            if needs_reproject:
                df[col] = df[col].apply(lambda g: geom_to_wkb(_reproject_geom_to_4326(g, transformer)))
            else:
                df[col] = df[col].apply(geom_to_wkb)

    # Drop the GeoDataFrame geometry column (already stored as street_geometry)
    if 'geometry' in df.columns:
        df = pd.DataFrame(df.drop(columns=['geometry']))

    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression='snappy')
    print(f"  Exported {len(df)} rows to {output_path}")


def _write_network_checkpoint(network: gpd.GeoDataFrame, city_name: str, stage: str, output_dir: str) -> None:
    """Write an intermediate checkpoint parquet for debugging."""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'{city_name}_{stage}.parquet')
    try:
        df = pd.DataFrame(network.drop(columns=['geometry'], errors='ignore'))
        # Convert geometry objects to WKB for serialization
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(1)
                if len(sample) > 0 and hasattr(sample.iloc[0], 'wkb'):
                    df[col] = df[col].apply(geom_to_wkb)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression='snappy')
        print(f"    Checkpoint: {path}")
    except Exception as e:
        print(f"    Checkpoint failed ({stage}): {e}")


# --- Pipeline Stage Functions ---

def _display_gpu_status():
    info = get_gpu_info()
    if info['available']:
        print(f"  GPU: {info['backend']} — {', '.join(info['gpu_names'])}")
        for i, mem in enumerate(info['total_memory_gb']):
            print(f"    Device {i}: {mem:.1f} GB")
    else:
        print("  GPU: not available, using CPU")


def _reset_pipeline_counters():
    return {
        'sidewalk': SequentialIDCounter(),
        'bikeway': SequentialIDCounter(),
        'curbramp': SequentialIDCounter(),
        'crosswalk': SequentialIDCounter(),
        'sw_feature': SequentialIDCounter(),
        'node': SequentialIDCounter(),
    }


def _stage_banner(n, label):
    print(f"\n{'='*60}\n  STAGE {n}: {label}\n{'='*60}")


def _stage_load_and_init(city_config: CityConfig):
    _stage_banner(1, f"Load & Initialize — {city_config.name}")
    edges, crossings_cache, working_crs = fetch_osm_network(city_config)
    print(f"  Loaded {len(edges)} edges")

    # Load and merge government centerlines if available
    gov_lines = load_government_centerlines(city_config, working_crs)
    if gov_lines is not None:
        edges = merge_government_centerlines(edges, gov_lines, city_config)

    # Load and merge government intersection nodes if available
    gov_nodes = load_government_intersection_nodes(city_config, working_crs)
    if gov_nodes is not None:
        edges = merge_government_intersection_nodes(edges, gov_nodes)

    return edges, crossings_cache, working_crs


def _stage_sanity_buffer(edges: gpd.GeoDataFrame, city_config: CityConfig, global_config: GlobalConfig):
    _stage_banner(2, "Sanity Buffer")
    parcel_path = city_config.government_data_paths.parcels
    if parcel_path and os.path.exists(parcel_path):
        sanity_df = compute_sanity_buffer_from_parcels(edges, parcel_path)
    else:
        print("  No parcel data — computing from highway classification")
        sanity_df = compute_sanity_buffer_from_highway(edges)

    output_dir = global_config.output_dir
    sanity_path = os.path.join(output_dir, f'{city_config.name}_sanity.parquet')
    export_sanity_buffer(sanity_df, city_config.name, sanity_path)
    sanity_buffer = dict(zip(sanity_df['osmid'], sanity_df['max_offset_width']))
    return sanity_buffer, sanity_path


def _stage_government_data(edges: gpd.GeoDataFrame, city_config: CityConfig, counters: Dict, working_crs: str):
    _stage_banner(3, "Government Data Integration")

    # Sidewalks
    gov_sidewalks = load_government_sidewalks(city_config, working_crs)
    if gov_sidewalks is not None:
        edges = merge_government_sidewalks(edges, gov_sidewalks, city_config)

    # Bikelanes
    gov_bikelanes = load_government_bikelanes(city_config, working_crs)
    if gov_bikelanes is not None:
        edges = merge_government_bikelanes(edges, gov_bikelanes, city_config)

    # Street features
    street_features = load_street_features(city_config, working_crs)
    if street_features is not None:
        edges = populate_street_features(edges, street_features, city_config)

    # Sidewalk features
    sidewalk_features = load_sidewalk_features(city_config, working_crs)
    if sidewalk_features is not None:
        edges = populate_sidewalk_features(edges, sidewalk_features, city_config, counters['sw_feature'])

    # Bikeway features
    bikeway_features = load_bikeway_features(city_config, working_crs)
    if bikeway_features is not None:
        edges = populate_bikeway_features(edges, bikeway_features, city_config)

    # Crosswalks (loaded for later use)
    gov_crosswalks = load_government_crosswalks(city_config, working_crs)

    return edges, gov_crosswalks


def _stage_tag_extraction(edges: gpd.GeoDataFrame, counters: Dict):
    _stage_banner(4, "Tag Extraction")
    edges = normalize_left_right_tags(edges)
    edges = extract_sidewalk_tags(edges, counters['sidewalk'])
    edges = extract_cycleway_tags(edges, counters['bikeway'])
    return edges


def _stage_deflection_split(edges: gpd.GeoDataFrame, counters: Dict):
    _stage_banner(5, "Vertex Deflection Splitting")
    edges = detect_and_split_deflections(edges, counters['node'])
    return edges


def _stage_planarization(edges: gpd.GeoDataFrame, counters: Dict):
    _stage_banner(6, "Layer-Aware Planarization")
    edges = planarize_by_layer(edges, counters['node'])
    return edges


def _stage_offset_and_features(edges: gpd.GeoDataFrame, sanity_buffer: Dict[int, float]):
    _stage_banner(7, "Offset Geometry Generation")
    edges = enrich_bikeway_geometries(edges, sanity_buffer)
    edges = enrich_sidewalk_geometries(edges, sanity_buffer)
    return edges


def _stage_bearing_and_blocks(edges: gpd.GeoDataFrame):
    """Stage 8: Compute bearings, detect blocks, assign block sides."""
    _stage_banner(8, "Bearings & Block Detection")

    # Compute bearings
    print("  Computing bearings...")
    edges['bearing'] = compute_bearings_vectorized(edges['geometry'])

    # Extract nodes for block detection
    nodes = extract_nodes_from_edges(edges)

    # Detect blocks — returns (block_map, node_face_map)
    block_map, node_face_map = detect_blocks(edges, nodes)

    # Assign block sides — takes node_face_map, returns 3 values
    edge_block_membership, edge_block_side_membership, node_face_map = assign_block_sides_shoelace(
        block_map, edges, node_face_map)

    # Apply to network — takes node_face_map as 4th arg
    edges = apply_block_ids_to_network(edges, edge_block_membership, edge_block_side_membership, node_face_map)

    # Reassign facility data based on bearing
    edges = reassign_facility_data_by_bearing(edges)

    return edges


def _stage_crosswalks(edges: gpd.GeoDataFrame, crossings_cache, counters: Dict, gov_crosswalks=None):
    _stage_banner(9, "Crosswalk Tag Extraction")
    edges = extract_crosswalk_tags(edges, crossings_cache, counters['crosswalk'], gov_crosswalks)
    return edges


def _stage_intersection_analysis(edges: gpd.GeoDataFrame, sanity_buffer: Dict[int, float], counters: Dict):
    _stage_banner(10, "Intersection Analysis")
    nodes = extract_nodes_from_edges(edges)
    edges = run_intersection_analysis(edges, nodes, sanity_buffer, counters['curbramp'])
    return edges


def _stage_curb_ramps(edges: gpd.GeoDataFrame, city_config: CityConfig, global_config: GlobalConfig, counters: Dict, working_crs: str):
    _stage_banner(11, "Government Curb Ramp Integration")
    gov_curb_ramps = load_government_curb_ramps(city_config, working_crs)
    if gov_curb_ramps is not None:
        edges = process_government_curb_ramps(edges, gov_curb_ramps, city_config, global_config, counters['curbramp'])
    else:
        print("  No government curb ramp data — skipping")
    return edges


def _stage_finalize(edges: gpd.GeoDataFrame, city_config: CityConfig, global_config: GlobalConfig):
    _stage_banner(12, "Finalize & Export")
    edges = assemble_network_schema(edges, global_config)
    output_dir = global_config.output_dir
    output_path = os.path.join(output_dir, f'{city_config.name}_network.parquet')
    export_network_parquet(edges, city_config.name, output_path)
    return edges


# --- Main Entry Point ---

def main():
    """Main pipeline entry point."""
    print("=" * 60)
    print("  Proximity Model — OSM Network Builder")
    print("=" * 60)

    _display_gpu_status()

    # Load configuration
    config_path = os.environ.get('PROXIMITY_CONFIG', 'config.json')
    if not os.path.exists(config_path):
        print(f"  Config not found at {config_path}")
        print("  Set PROXIMITY_CONFIG env var or place config.json in working directory")
        return

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    global_config = load_config(config_dict)
    os.makedirs(global_config.output_dir, exist_ok=True)

    print(f"  Processing {len(global_config.cities)} cities")
    print(f"  Export path: {global_config.output_dir}")

    for city_config in global_config.cities:
        print(f"\n{'#'*60}")
        print(f"  CITY: {city_config.name}")
        print(f"{'#'*60}")

        try:
            counters = _reset_pipeline_counters()

            # Stage 1: Load & Init
            edges, crossings_cache, working_crs = _stage_load_and_init(city_config)

            # Stage 2: Sanity Buffer
            sanity_buffer, sanity_path = _stage_sanity_buffer(edges, city_config, global_config)

            # Stage 3: Government Data
            edges, gov_crosswalks = _stage_government_data(edges, city_config, counters, working_crs)

            # Stage 4: Tag Extraction
            edges = _stage_tag_extraction(edges, counters)

            # Stage 5: Deflection Splitting
            edges = _stage_deflection_split(edges, counters)

            # Stage 6: Layer-Aware Planarization (NEW)
            edges = _stage_planarization(edges, counters)

            # Stage 7: Offset Geometry
            edges = _stage_offset_and_features(edges, sanity_buffer)

            # Stage 8: Bearings & Blocks
            edges = _stage_bearing_and_blocks(edges)

            # Stage 9: Crosswalks
            edges = _stage_crosswalks(edges, crossings_cache, counters, gov_crosswalks)

            # Stage 10: Intersection Analysis
            edges = _stage_intersection_analysis(edges, sanity_buffer, counters)

            # Stage 11: Government Curb Ramps
            edges = _stage_curb_ramps(edges, city_config, global_config, counters, working_crs)

            # Stage 11b: Generate implicit crosswalk geometries (Phase 4 Step 2)
            # Must run after intersection analysis and curb ramp processing
            print(f"\n{'='*60}")
            print(f"  STAGE 11b: Implicit Crosswalk Geometry")
            print(f"{'='*60}")
            edges = generate_implicit_crosswalk_geometries(edges, counters['crosswalk'])

            # Stage 12: Finalize & Export
            edges = _stage_finalize(edges, city_config, global_config)

            print(f"\n  {city_config.name} complete: {len(edges)} segments")

        except Exception as e:
            print(f"\n  ERROR processing {city_config.name}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("  Pipeline complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
