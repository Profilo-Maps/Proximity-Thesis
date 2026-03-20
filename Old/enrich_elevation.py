"""
Street Centerline Elevation Enrichment
=======================================
Adds start/end point elevation and derived slope to each segment.

Supports:
  - USA: USGS 3DEP EPQS REST API — direct point query, no raster, no API key
  - Canada: AWS Terrain Tiles / OpenTopoData fallback (SRTM/CDEM)

Requirements:
    pip install geopandas pyarrow shapely pyproj requests tqdm

Usage:
    python enrich_elevation.py \
        --input  streets.parquet \
        --output streets_with_elevation.parquet \
        --crs    EPSG:4326
"""

import argparse
import math
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Point
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bounding box for contiguous US (used to route to the right data source)
US_BBOX = {"minx": -125.0, "miny": 24.0, "maxx": -66.9, "maxy": 49.4}
# Canada bbox (rough)
CA_BBOX = {"minx": -141.0, "miny": 41.7, "maxx": -52.6, "maxy": 83.0}

# OpenTopoData endpoint — self-hostable, free, covers SRTM/ASTER globally
OPENTOPO_URL = "https://api.opentopodata.org/v1/{dataset}"
# Dataset choices: "aster30m" | "srtm30m" | "etopo1"
OPENTOPO_DATASET_US = "ned10m"       # 10m NED (same source as 3DEP, slower)
OPENTOPO_DATASET_CA = "cdem"         # Canadian Digital Elevation Model
OPENTOPO_BATCH_SIZE = 100            # max locations per request


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def get_endpoints(geom):
    """Return (start_lon, start_lat, end_lon, end_lat) for a LineString."""
    coords = list(geom.coords)
    return coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]


def point_in_us(lon, lat):
    return (US_BBOX["minx"] <= lon <= US_BBOX["maxx"] and
            US_BBOX["miny"] <= lat <= US_BBOX["maxy"])


def point_in_canada(lon, lat):
    return (CA_BBOX["minx"] <= lon <= CA_BBOX["maxx"] and
            CA_BBOX["miny"] <= lat <= CA_BBOX["maxy"])


# ---------------------------------------------------------------------------
# Elevation backends
# ---------------------------------------------------------------------------

USGS_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
USGS_EPQS_MAX_RETRIES = 3
USGS_EPQS_RETRY_DELAY = 2   # seconds between retries


def query_usgs_epqs(points_lonlat: list[tuple]) -> list[float | None]:
    """
    Query the USGS Elevation Point Query Service (3DEP) directly via REST.

    GET  https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&wkid=4326
    Returns elevation in metres. No API key, no raster sampling — true point lookup.

    Processes points concurrently using a thread pool for throughput.
    """
    import concurrent.futures

    def fetch_one(lon_lat: tuple) -> float | None:
        lon, lat = lon_lat
        params = {"x": lon, "y": lat, "wkid": 4326, "includeDate": "false"}
        for attempt in range(USGS_EPQS_MAX_RETRIES):
            try:
                r = requests.get(USGS_EPQS_URL, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()
                val = data.get("value")
                if val is None or val == -1000000:  # USGS sentinel for no-data
                    return None
                return round(float(val), 3)
            except requests.exceptions.Timeout:
                if attempt < USGS_EPQS_MAX_RETRIES - 1:
                    time.sleep(USGS_EPQS_RETRY_DELAY * (attempt + 1))
            except Exception as e:
                print(f"  [usgs_epqs] ({lon:.5f},{lat:.5f}) attempt {attempt+1}: {e}")
                if attempt < USGS_EPQS_MAX_RETRIES - 1:
                    time.sleep(USGS_EPQS_RETRY_DELAY)
        return None

    results: list[float | None] = [None] * len(points_lonlat)
    # 10 concurrent workers keeps throughput high without hammering the endpoint
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {
            executor.submit(fetch_one, pt): i
            for i, pt in enumerate(points_lonlat)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  [usgs_epqs] Unhandled error at index {idx}: {e}")
    return results


def query_opentopo(points_lonlat: list[tuple], dataset: str) -> list[float | None]:
    """
    Query OpenTopoData REST API in batches.
    Free tier: 1 req/s, 100 locations/req, 1000 req/day.
    For production volume, self-host: https://www.opentopodata.org/#host-your-own
    """
    results = []
    for i in range(0, len(points_lonlat), OPENTOPO_BATCH_SIZE):
        batch = points_lonlat[i : i + OPENTOPO_BATCH_SIZE]
        locations = "|".join(f"{lat},{lon}" for lon, lat in batch)
        url = OPENTOPO_URL.format(dataset=dataset)
        try:
            r = requests.get(url, params={"locations": locations}, timeout=30)
            r.raise_for_status()
            data = r.json()
            for result in data.get("results", []):
                elev = result.get("elevation")
                results.append(float(elev) if elev is not None else None)
        except Exception as e:
            print(f"  [opentopodata/{dataset}] Error: {e}")
            results.extend([None] * len(batch))
        time.sleep(1.1)  # rate limit: 1 req/s on free tier
    return results


def query_aws_terrain(points_lonlat: list[tuple]) -> list[float | None]:
    """
    Sample elevation from AWS Terrain RGB tiles (Mapbox/Terrarium encoding).
    No API key required. Good for Canada + global coverage.
    Resolution ~30m globally, ~10m in US.
    """
    try:
        import mercantile  # type: ignore
        from PIL import Image
        import io

        results: list[float | None] = []
        session = requests.Session()

        for lon, lat in points_lonlat:
            try:
                # Zoom 12 ≈ 10m/px in mid-latitudes
                tile = mercantile.tile(lon, lat, 12)
                url = (
                    f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/"
                    f"{tile.z}/{tile.x}/{tile.y}.png"
                )
                r = session.get(url, timeout=10)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")

                # Pixel position within tile
                bounds = mercantile.xy_bounds(tile)
                px = mercantile.xy(lon, lat)
                col = int((px.x - bounds.left) / (bounds.right - bounds.left) * 256)
                row = int((bounds.top - px.y) / (bounds.top - bounds.bottom) * 256)
                col = max(0, min(255, col))
                row = max(0, min(255, row))

                pixel = img.getpixel((col, row))
                if isinstance(pixel, tuple):
                    r_val, g_val, b_val = pixel
                    # Terrarium encoding: elevation = R*256 + G + B/256 - 32768
                    elevation = r_val * 256 + g_val + b_val / 256 - 32768
                    results.append(round(elevation, 2))
                else:
                    results.append(None)
            except Exception as e:
                print(f"  [aws_terrain] ({lon:.4f},{lat:.4f}): {e}")
                results.append(None)

        return results

    except ImportError:
        print("  [aws_terrain] Requires: pip install mercantile Pillow")
        return [None] * len(points_lonlat)  # type: ignore


# ---------------------------------------------------------------------------
# Routing logic: pick the right backend per point
# ---------------------------------------------------------------------------

def fetch_elevations(points_lonlat: list[tuple], backend: str = "auto") -> list[float | None]:
    """
    Route each point to the appropriate elevation source based on location,
    then merge results back in order.

    backend options:
        "auto"        — usgs_epqs for US, aws_terrain for Canada (recommended)
        "opentopo"    — OpenTopoData REST API for everything (rate-limited)
        "aws_terrain" — AWS Terrain Tiles for everything (no key, no rate limit)
    """
    n = len(points_lonlat)
    elevations: list[float | None] = [None] * n

    if backend == "opentopo":
        return query_opentopo(points_lonlat, dataset="aster30m")

    if backend == "aws_terrain":
        return query_aws_terrain(points_lonlat)

    # "auto" — route by geography
    us_idx, us_pts = [], []
    ca_idx, ca_pts = [], []
    other_idx, other_pts = [], []

    for i, (lon, lat) in enumerate(points_lonlat):
        if point_in_us(lon, lat):
            us_idx.append(i); us_pts.append((lon, lat))
        elif point_in_canada(lon, lat):
            ca_idx.append(i); ca_pts.append((lon, lat))
        else:
            other_idx.append(i); other_pts.append((lon, lat))

    if us_pts:
        print(f"  → {len(us_pts)} US points via USGS 3DEP EPQS (direct REST)")
        us_elev = query_usgs_epqs(us_pts)
        for i, e in zip(us_idx, us_elev):
            elevations[i] = e  # type: ignore

    if ca_pts:
        print(f"  → {len(ca_pts)} Canada points via AWS Terrain Tiles")
        ca_elev = query_aws_terrain(ca_pts)
        for i, e in zip(ca_idx, ca_elev):
            elevations[i] = e  # type: ignore

    if other_pts:
        print(f"  → {len(other_pts)} other points via AWS Terrain Tiles")
        oth_elev = query_aws_terrain(other_pts)
        for i, e in zip(other_idx, oth_elev):
            elevations[i] = e  # type: ignore

    return elevations


# ---------------------------------------------------------------------------
# Slope calculation
# ---------------------------------------------------------------------------

def haversine_m(lon1, lat1, lon2, lat2) -> float:
    """Horizontal distance between two WGS84 points in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_slope(row) -> float | None:
    """
    Returns slope in % (rise/run * 100).
    Positive = uphill start→end, negative = downhill.
    """
    e_start = row["elev_start_m"]
    e_end   = row["elev_end_m"]
    if e_start is None or e_end is None:
        return None
    dist = haversine_m(row["start_lon"], row["start_lat"],
                       row["end_lon"],   row["end_lat"])
    if dist < 1:  # avoid divide-by-zero on degenerate segments
        return None
    return round((e_end - e_start) / dist * 100, 4)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def enrich(input_path: str, output_path: str, crs: str, backend: str,
           chunk_size: int = 500):

    print(f"Reading {input_path} …")
    gdf = gpd.read_parquet(input_path)

    # Reproject to WGS84 for elevation queries
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        print(f"Reprojecting from {gdf.crs} → EPSG:4326 …")
        gdf = gdf.to_crs("EPSG:4326")

    # Extract endpoints
    print("Extracting endpoints …")
    gdf[["start_lon", "start_lat", "end_lon", "end_lat"]] = gdf["geometry"].apply(
        lambda g: pd.Series(get_endpoints(g))
    )

    # Collect all unique points (dedup saves API calls at intersections)
    start_pts = list(zip(gdf["start_lon"], gdf["start_lat"]))
    end_pts   = list(zip(gdf["end_lon"],   gdf["end_lat"]))
    all_pts_raw = start_pts + end_pts

    unique_pts = list(set(all_pts_raw))
    print(f"  {len(gdf)} segments → {len(unique_pts)} unique endpoints to query")

    # Fetch elevations in chunks (progress bar)
    print(f"Fetching elevations (backend={backend}, chunk={chunk_size}) …")
    elev_map: dict[tuple, float | None] = {}

    for i in tqdm(range(0, len(unique_pts), chunk_size)):
        chunk = unique_pts[i : i + chunk_size]
        elevs = fetch_elevations(chunk, backend=backend)
        for pt, elev in zip(chunk, elevs):
            elev_map[pt] = elev

    # Map back
    gdf["elev_start_m"] = [elev_map.get(p) for p in start_pts]
    gdf["elev_end_m"]   = [elev_map.get(p) for p in end_pts]

    # Derived columns
    gdf["slope_pct"] = gdf.apply(compute_slope, axis=1)
    gdf["elev_delta_m"] = (gdf["elev_end_m"] - gdf["elev_start_m"]).round(2)

    print(f"Writing {output_path} …")
    gdf.to_parquet(output_path, index=False)

    # Summary
    n_ok = gdf["elev_start_m"].notna().sum()
    print(f"\n✓ Done. Elevation resolved for {n_ok}/{len(gdf)} segments "
          f"({n_ok/len(gdf)*100:.1f}%)")
    print(gdf[["elev_start_m","elev_end_m","slope_pct","elev_delta_m"]].describe())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich street centerlines with elevation.")
    parser.add_argument("--input",  required=True,  help="Input parquet path")
    parser.add_argument("--output", required=True,  help="Output parquet path")
    parser.add_argument("--crs",    default="EPSG:4326", help="CRS if not set in file")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "aws_terrain", "opentopo"],
                        help="Elevation backend (default: auto)")
    parser.add_argument("--chunk",  type=int, default=500,
                        help="Points per API batch (default: 500)")
    args = parser.parse_args()

    enrich(args.input, args.output, args.crs, args.backend, args.chunk)
