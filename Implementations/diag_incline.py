"""
diag_incline.py — Diagnostic: verify street_incline values in the output parquet.

Checks:
  1. Column presence and null rate
  2. Distribution stats and histogram
  3. Artifact flags (values near/beyond ±40% cap)
  4. Spot-recompute a random sample of segments via live USGS 3DEP and compare
  5. Short-segment audit (< DEM_RESOLUTION_M = 10 m)

Run from the repo root:
  python Implementations/diag_incline.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

PARQUET = Path("Output/San_Francisco_County_California_USA_network.parquet")
DEM_RESOLUTION_M = 10.0
SLOPE_CAP_PCT = 40.0
SPOT_SAMPLE_N = 30        # segments to re-check against live 3DEP
RANDOM_SEED = 42

# Pinned coordinate to always include in spot-recompute (DMS: 37°44'57.8"N 122°28'17.1"W)
PINNED_LON = -(122 + 28/60 + 17.1/3600)   # -122.47142°
PINNED_LAT =   37 + 44/60 + 57.8/3600     #  37.74939°

# ── helpers ──────────────────────────────────────────────────────────────────

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── 1. Load ───────────────────────────────────────────────────────────────────

section("Loading parquet")
if not PARQUET.exists():
    sys.exit(f"ERROR: {PARQUET} not found")

gdf = gpd.read_parquet(PARQUET)
print(f"  Rows: {len(gdf):,}  |  CRS: {gdf.crs}")

if "street_incline" not in gdf.columns:
    sys.exit("ERROR: 'street_incline' column is missing from the parquet")


# ── 2. Null rate ──────────────────────────────────────────────────────────────

section("Null / coverage audit")
total = len(gdf)
n_null = gdf["street_incline"].isna().sum()
n_valid = total - n_null
print(f"  Total segments : {total:,}")
print(f"  With slope     : {n_valid:,}  ({n_valid/total*100:.1f}%)")
print(f"  Null           : {n_null:,}   ({n_null/total*100:.1f}%)")


# ── 3. Distribution stats ─────────────────────────────────────────────────────

section("Distribution stats  (valid slopes only)")
slopes = gdf["street_incline"].dropna().astype(float)

if slopes.empty:
    print("  WARNING: no valid slope values — nothing to analyse")
else:
    arr = np.asarray(slopes.values, dtype=np.float64)
    pcts = np.quantile(arr, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    print(f"  Min    : {arr.min():.4f}%")
    print(f"  P01    : {pcts[0]:.4f}%")
    print(f"  P05    : {pcts[1]:.4f}%")
    print(f"  P25    : {pcts[2]:.4f}%")
    print(f"  Median : {pcts[3]:.4f}%")
    print(f"  P75    : {pcts[4]:.4f}%")
    print(f"  P95    : {pcts[5]:.4f}%")
    print(f"  P99    : {pcts[6]:.4f}%")
    print(f"  Max    : {arr.max():.4f}%")
    print(f"  Mean   : {arr.mean():.4f}%  |  Std: {arr.std():.4f}%")
    print(f"  Uphill (>0) : {(arr>0).sum():,}  |  Downhill (<0): {(arr<0).sum():,}  |  Flat (==0): {(arr==0).sum():,}")

    # ASCII histogram (absolute value buckets)
    abs_arr = np.abs(arr)
    buckets = [0, 2, 5, 10, 15, 20, 30, 40, np.inf]
    labels  = ["<2%","2-5%","5-10%","10-15%","15-20%","20-30%","30-40%",">40%(artifact)"]
    print("\n  |slope| distribution:")
    for lo, hi, lbl in zip(buckets, buckets[1:], labels):
        count = int(((abs_arr >= lo) & (abs_arr < hi)).sum())
        bar = "#" * min(40, count // max(1, n_valid // 400))
        print(f"    {lbl:>18s}  {count:6,}  {bar}")


# ── 4. Artifact flags ─────────────────────────────────────────────────────────

section("Artifact / cap audit")
if not slopes.empty:
    near_cap = gdf[gdf["street_incline"].abs() >= SLOPE_CAP_PCT * 0.9]
    over_cap  = gdf[gdf["street_incline"].abs() >= SLOPE_CAP_PCT]
    print(f"  Segments within 10% of ±{SLOPE_CAP_PCT:.0f}% cap : {len(near_cap):,}")
    print(f"  Segments AT or OVER ±{SLOPE_CAP_PCT:.0f}% cap     : {len(over_cap):,}  (should be 0 — cap should null these)")
    if not over_cap.empty:
        print("  WARNING: values at/over cap were NOT nulled correctly!")
        print(over_cap[["street_id", "street_incline"]].head(10).to_string(index=False))


# ── 5. Short-segment audit ────────────────────────────────────────────────────

section(f"Short-segment audit  (< {DEM_RESOLUTION_M:.0f} m horizontal)")

# Reproject to a metric CRS for accurate length
gdf_metric = gdf.to_crs("EPSG:32610")
seg_lengths = gdf_metric.geometry.length

short_mask = seg_lengths < DEM_RESOLUTION_M
n_short = short_mask.sum()
short_with_slope = (short_mask & gdf["street_incline"].notna()).sum()

print(f"  Segments shorter than {DEM_RESOLUTION_M:.0f} m : {n_short:,}")
print(f"    Of those, have a slope value    : {short_with_slope:,}")
print(f"    (These were filled via neighbor propagation or should be null)")
if short_with_slope > 0:
    sample_short = gdf[short_mask & gdf["street_incline"].notna()][["street_id","street_incline"]].head(5)
    print(sample_short.to_string(index=False))


# ── 6. Spot-recompute from live 3DEP ─────────────────────────────────────────

section(f"Spot-recompute check  (n={SPOT_SAMPLE_N} segments via live USGS 3DEP)")
print("  NOTE: requires network access and py3dep / pyproj installed")

try:
    import py3dep
    import xarray as xr
    from pyproj import Transformer

    # Pick segments that are long enough, have a stored slope, and are in WGS-84
    eligible = gdf[
        (seg_lengths >= DEM_RESOLUTION_M * 2) &
        gdf["street_incline"].notna()
    ].copy()

    # Find the segment nearest to the pinned coordinate.
    # eligible is in EPSG:32610 (metric), so project the pin point to match.
    from shapely.geometry import Point as ShapelyPoint
    from pyproj import Transformer as _Transformer
    _to_utm = _Transformer.from_crs("EPSG:4326", eligible.crs, always_xy=True)
    _pin_x, _pin_y = _to_utm.transform(PINNED_LON, PINNED_LAT)
    pinned_pt_utm = ShapelyPoint(_pin_x, _pin_y)
    distances_to_pin = eligible.geometry.distance(pinned_pt_utm)
    pinned_idx = distances_to_pin.idxmin()
    pinned_row = eligible.loc[[pinned_idx]]
    print(f"  Pinned segment: index={pinned_idx}  street_id={pinned_row['street_id'].iloc[0]}  "
          f"stored incline={pinned_row['street_incline'].iloc[0]:.4f}%  "
          f"dist_to_pin={distances_to_pin[pinned_idx]:.1f} m")

    # Random sample from the rest, then prepend the pinned segment
    rest = eligible.drop(index=pinned_idx)
    n_random = max(0, SPOT_SAMPLE_N - 1)
    if len(rest) < n_random:
        print(f"  Only {len(rest)} other eligible segments (need {n_random}) — using all")
        _frames = [pinned_row, rest]
    else:
        _frames = [pinned_row, rest.sample(n_random, random_state=RANDOM_SEED)]
    sample = gpd.GeoDataFrame(
        pd.concat(_frames),
        geometry=eligible.geometry.name,
        crs=eligible.crs,
    )

    if sample.empty:
        print("  No eligible segments — skipping spot check")
    else:
        # Ensure WGS-84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            sample_wgs = sample.to_crs("EPSG:4326")
        else:
            sample_wgs = sample

        to_5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

        # Collect unique endpoints
        geom_col = str(sample_wgs.geometry.name)
        pts_wgs = {}  # index → (start_lonlat, end_lonlat)
        for idx, row in sample_wgs.iterrows():
            coords = list(row[geom_col].coords)
            s = (coords[0][0], coords[0][1])
            e = (coords[-1][0], coords[-1][1])
            pts_wgs[idx] = (s, e)

        unique_wgs = list({p for pair in pts_wgs.values() for p in pair})
        lons = np.array([p[0] for p in unique_wgs])
        lats = np.array([p[1] for p in unique_wgs])

        buf = 0.001
        bbox = (lons.min()-buf, lats.min()-buf, lons.max()+buf, lats.max()+buf)
        print(f"  Fetching DEM tile for bbox {bbox[0]:.4f},{bbox[1]:.4f} → {bbox[2]:.4f},{bbox[3]:.4f} ...")
        dem = py3dep.get_dem(bbox, crs="EPSG:4326", resolution=10)
        print("  DEM fetched — sampling ...")

        xs_5070, ys_5070 = to_5070.transform(lons, lats)
        xs_da = xr.DataArray(xs_5070, dims="points")
        ys_da = xr.DataArray(ys_5070, dims="points")
        elevs = dem.interp(x=xs_da, y=ys_da, method="nearest").values
        elev_map = {pt: (None if (np.isnan(e) or float(e) < -1000) else round(float(e), 3))
                    for pt, e in zip(unique_wgs, elevs)}

        # Compare
        print(f"\n  {'street_id':>20s}  {'chord_m':>7s}  {'stored':>8s}  {'recomp':>8s}  {'delta':>8s}  {'flag'}")
        print("  " + "-"*72)
        n_ok = n_miss = n_large_delta = 0
        deltas = []
        for idx, (s, e) in pts_wgs.items():
            stored = float(np.asarray(sample.loc[idx, "street_incline"], dtype=np.float64))
            elev_s = elev_map.get(s)
            elev_e = elev_map.get(e)
            # Use haversine endpoint distance to match the pipeline's denominator,
            # NOT the full polyline path length (seg_lengths uses path length).
            chord_m = haversine_m(s[0], s[1], e[0], e[1])
            seg_len = float(np.asarray(seg_lengths.loc[idx], dtype=np.float64))
            sid = str(sample.loc[idx, "street_id"])[:20]

            if elev_s is None or elev_e is None:
                recomp = None
                flag = "NO_ELEV"
                n_miss += 1
            else:
                rise = elev_e - elev_s
                recomp = round(rise / chord_m * 100, 4) if chord_m >= DEM_RESOLUTION_M else None
                if recomp is None:
                    flag = "CHORD_TOO_SHORT"
                    n_miss += 1
                else:
                    delta = abs(stored - recomp)
                    deltas.append(delta)
                    flag = "OK" if delta < 0.5 else ("LARGE_DELTA" if delta < 5 else "ERROR")
                    if flag != "OK":
                        n_large_delta += 1
                    n_ok += 1
            recomp_s = f"{recomp:.4f}%" if recomp is not None else "N/A"
            delta_s  = f"{abs(stored-recomp):.4f}" if recomp is not None else "N/A"
            # chord_ratio: path_len / chord — >1.1 means the segment curves significantly
            chord_ratio = seg_len / chord_m if chord_m > 0 else float("inf")
            curve_flag = f" [curve={chord_ratio:.2f}x]" if chord_ratio > 1.05 else ""
            print(f"  {sid:>20s}  {chord_m:7.1f}  {stored:8.4f}  {recomp_s:>8s}  {delta_s:>8s}  {flag}{curve_flag}")

        print(f"\n  Summary: {n_ok} recomputed | {n_miss} missing elevation | {n_large_delta} large deltas (>0.5%)")
        if deltas:
            print(f"  Delta stats: mean={np.mean(deltas):.4f}  median={np.median(deltas):.4f}  max={np.max(deltas):.4f}")

except ImportError as exc:
    print(f"  SKIPPED — missing dependency: {exc}")
except Exception as exc:
    print(f"  FAILED — {type(exc).__name__}: {exc}")

# ── 7. Pinned segment deep-dive ───────────────────────────────────────────────

section("Pinned segment deep-dive  (street_id=8919297 / 37°44'57.8\"N 122°28'17.1\"W)")
print("  Traces the exact values the pipeline would have used for this segment.")

try:
    import py3dep as _py3dep
    import xarray as _xr
    from pyproj import Transformer as _T2

    _pinned_row = gdf[gdf["street_id"] == "8919297"]
    if _pinned_row.empty:
        print("  WARNING: street_id 8919297 not found in parquet")
    else:
        _pinned_row = _pinned_row.iloc[[0]]
        _stored_slope = float(_pinned_row["street_incline"].iloc[0])
        print(f"  Stored street_incline : {_stored_slope:.6f}%")

        # --- Geometry in native CRS (EPSG:32610) ---
        _geom_utm = _pinned_row.geometry.values[0]
        _coords_utm = list(_geom_utm.coords)
        print(f"  Geometry vertices (EPSG:32610): {len(_coords_utm)}")
        print(f"    start : ({_coords_utm[0][0]:.3f}, {_coords_utm[0][1]:.3f})")
        print(f"    end   : ({_coords_utm[-1][0]:.3f}, {_coords_utm[-1][1]:.3f})")

        # --- Transform to WGS-84 exactly as the pipeline does ---
        _to_wgs = _T2.from_crs(_pinned_row.crs, "EPSG:4326", always_xy=True)
        _sx_utm, _sy_utm = _coords_utm[0][0], _coords_utm[0][1]
        _ex_utm, _ey_utm = _coords_utm[-1][0], _coords_utm[-1][1]
        (_slon, _slat), (_elon, _elat) = (
            _to_wgs.transform(_sx_utm, _sy_utm),
            _to_wgs.transform(_ex_utm, _ey_utm),
        )
        print(f"  Endpoints in WGS-84 (pipeline transform):")
        print(f"    start : lon={_slon:.8f}  lat={_slat:.8f}")
        print(f"    end   : lon={_elon:.8f}  lat={_elat:.8f}")

        # --- WGS-84 from GeoDataFrame.to_crs (diagnostic transform) ---
        _pinned_wgs = _pinned_row.to_crs("EPSG:4326")
        _coords_wgs = list(_pinned_wgs.geometry.values[0].coords)
        _slon_d, _slat_d = _coords_wgs[0][0], _coords_wgs[0][1]
        _elon_d, _elat_d = _coords_wgs[-1][0], _coords_wgs[-1][1]
        print(f"  Endpoints in WGS-84 (GeoDataFrame.to_crs):")
        print(f"    start : lon={_slon_d:.8f}  lat={_slat_d:.8f}")
        print(f"    end   : lon={_elon_d:.8f}  lat={_elat_d:.8f}")
        print(f"  Coord delta (pipeline vs to_crs):")
        print(f"    start : Δlon={abs(_slon-_slon_d):.2e}  Δlat={abs(_slat-_slat_d):.2e}")
        print(f"    end   : Δlon={abs(_elon-_elon_d):.2e}  Δlat={abs(_elat-_elat_d):.2e}")

        # --- Haversine chord ---
        _chord = haversine_m(_slon, _slat, _elon, _elat)
        print(f"  Haversine chord (pipeline denominator) : {_chord:.3f} m")

        # --- Fetch a fresh DEM with a small bbox (just this segment) ---
        _buf = 0.002
        _bbox_pin = (
            min(_slon, _elon) - _buf, min(_slat, _elat) - _buf,
            max(_slon, _elon) + _buf, max(_slat, _elat) + _buf,
        )
        print(f"  Fetching small DEM tile (just this segment) "
              f"bbox {_bbox_pin[0]:.5f},{_bbox_pin[1]:.5f} → {_bbox_pin[2]:.5f},{_bbox_pin[3]:.5f} ...")
        _dem_pin = _py3dep.get_dem(_bbox_pin, crs="EPSG:4326", resolution=10)
        _to5070 = _T2.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

        def _sample_elev(lon, lat):
            x5, y5 = _to5070.transform(lon, lat)
            val = float(_dem_pin.interp(
                x=_xr.DataArray([x5], dims="p"),
                y=_xr.DataArray([y5], dims="p"),
                method="nearest",
            ).values[0])
            return None if (np.isnan(val) or val < -1000) else round(val, 3)

        _es = _sample_elev(_slon, _slat)
        _ee = _sample_elev(_elon, _elat)
        print(f"  Elevations (small tile):")
        print(f"    start : {_es} m")
        print(f"    end   : {_ee} m")
        if _es is not None and _ee is not None:
            _rise_small = _ee - _es
            _slope_small = round(_rise_small / _chord * 100, 6)
            print(f"    rise  : {_rise_small:.3f} m  →  slope = {_slope_small:.6f}%")
            print(f"  Stored slope   : {_stored_slope:.6f}%")
            print(f"  Small-tile recomp: {_slope_small:.6f}%")
            print(f"  Delta          : {abs(_stored_slope - _slope_small):.6f}%")
            print()
            print("  NOTE: If delta is still large, the pipeline's DEM tile (full SF county)")
            print("  returned different elevation values at these coordinates — tile mosaic artifact.")
            print("  If delta is ~0, the pipeline stored the wrong slope for a different reason.")

except ImportError as _exc:
    print(f"  SKIPPED — missing dependency: {_exc}")
except Exception as _exc:
    import traceback
    print(f"  FAILED — {type(_exc).__name__}: {_exc}")
    traceback.print_exc()


# ── 8. Flat segment geography audit ──────────────────────────────────────────

section("Flat segment geography audit  (slope == 0.0000)")
_flat = gdf[gdf["street_incline"] == 0.0].copy()
print(f"  Total flat segments (slope=0.0) : {len(_flat):,}")

# Centroids in WGS-84 for neighbourhood binning
_flat_wgs = _flat.to_crs("EPSG:4326")
_flat_wgs["_cx"] = np.asarray(_flat_wgs.geometry.centroid.x.values, dtype=np.float64)
_flat_wgs["_cy"] = np.asarray(_flat_wgs.geometry.centroid.y.values, dtype=np.float64)

# Rough SF neighbourhood boxes (lon, lat bounds) — enough to distinguish flat vs hilly areas
_ZONES = {
    "Marina/Fishermans Wharf  (flat expected)": (-122.44, 37.800, -122.40, 37.815),
    "Embarcadero/SOMA         (flat expected)": (-122.41, 37.775, -122.38, 37.800),
    "Mission/Castro           (hilly)":         (-122.44, 37.755, -122.41, 37.775),
    "Noe Valley/Glen Park     (hilly)":         (-122.44, 37.735, -122.42, 37.755),
    "Twin Peaks/Diamond Hts   (very hilly)":    (-122.45, 37.745, -122.43, 37.760),
    "Outer Sunset             (flat expected)": (-122.51, 37.745, -122.46, 37.770),
}

print(f"\n  Distribution of flat segments by zone:")
for zone, (w, s, e, n) in _ZONES.items():
    in_zone = (
        (_flat_wgs["_cx"] >= w) & (_flat_wgs["_cx"] <= e) &
        (_flat_wgs["_cy"] >= s) & (_flat_wgs["_cy"] <= n)
    )
    total_in_zone = (
        (_flat_wgs["_cx"] >= w) & (_flat_wgs["_cx"] <= e) &
        (_flat_wgs["_cy"] >= s) & (_flat_wgs["_cy"] <= n)
    )
    # Count all segments in zone (not just flat)
    all_in_zone = (
        (gdf.to_crs("EPSG:4326").geometry.centroid.x >= w) &
        (gdf.to_crs("EPSG:4326").geometry.centroid.x <= e) &
        (gdf.to_crs("EPSG:4326").geometry.centroid.y >= s) &
        (gdf.to_crs("EPSG:4326").geometry.centroid.y <= n)
    )
    n_flat_zone = int(in_zone.sum())
    n_all_zone  = int(all_in_zone.sum())
    pct = n_flat_zone / n_all_zone * 100 if n_all_zone > 0 else 0
    print(f"    {zone:<42s}  flat={n_flat_zone:5,} / {n_all_zone:5,}  ({pct:.1f}%)")

# Short-segment share of flat
_flat_short = _flat[seg_lengths[_flat.index] < DEM_RESOLUTION_M]
print(f"\n  Flat segments that are short (< {DEM_RESOLUTION_M:.0f} m, i.e. propagated): "
      f"{len(_flat_short):,}  ({len(_flat_short)/len(_flat)*100:.1f}% of flat)")
print(f"  Flat segments that are long  (≥ {DEM_RESOLUTION_M:.0f} m, i.e. direct DEM): "
      f"{len(_flat)-len(_flat_short):,}  ({(len(_flat)-len(_flat_short))/len(_flat)*100:.1f}% of flat)")


# ── 9. Short-segment propagation quality ─────────────────────────────────────

section("Short-segment propagation quality")
_short_with_slope = gdf[short_mask & gdf["street_incline"].notna()].copy()
print(f"  Short segments with propagated slope : {len(_short_with_slope):,}")

_prop_slopes = _short_with_slope["street_incline"].astype(float)
_prop_arr = np.asarray(_prop_slopes.values, dtype=np.float64)

print(f"  Slope=0.0  : {(_prop_arr == 0.0).sum():,}  ({(_prop_arr == 0.0).mean()*100:.1f}%)")
print(f"  |slope|<1% : {(np.abs(_prop_arr) < 1.0).sum():,}  ({(np.abs(_prop_arr) < 1.0).mean()*100:.1f}%)")
print(f"  |slope|≥5% : {(np.abs(_prop_arr) >= 5.0).sum():,}  ({(np.abs(_prop_arr) >= 5.0).mean()*100:.1f}%)")
print(f"  Mean |slope|: {np.abs(_prop_arr).mean():.4f}%  |  Max |slope|: {np.abs(_prop_arr).max():.4f}%")

# Check how many short segments had BOTH endpoints with valid elevation
# but were still given 0.0 — these may be suspicious
_both_zero_neighbors = (
    (_prop_arr == 0.0) &
    (np.asarray(seg_lengths[_short_with_slope.index].values, dtype=np.float64) < DEM_RESOLUTION_M)
)
print(f"\n  Short segments with propagated slope=0.0 (neighbor was flat): "
      f"{_both_zero_neighbors.sum():,}")
print("  (If a short segment is on a hilly block, its neighbors should not be 0.0 —")
print("   a high count here means propagation is absorbing real elevation change.)")

# Sample 5 suspicious ones: short, high |slope| (neighbors were steep)
_steep_short = _short_with_slope[np.abs(_prop_arr) >= 10.0]
if not _steep_short.empty:
    print(f"\n  Short segments with |propagated slope| ≥ 10% (steep neighbors): {len(_steep_short):,}")
    print(_steep_short[["street_id", "street_incline", "name"]].head(8).to_string(index=False))
else:
    print("\n  No short segments with |propagated slope| ≥ 10%.")


# ── 10. Hilly-zone short segment investigation ───────────────────────────────

section("Hilly-zone short segment investigation  (slope=0.0 in hilly areas)")
print("  For each hilly zone, finds short (propagated) segments with slope=0.0")
print("  and traces their nearest connected long-segment neighbor's slope.")

_HILLY_ZONES = {
    "Mission/Castro":        (-122.44, 37.755, -122.41, 37.775),
    "Noe Valley/Glen Park":  (-122.44, 37.735, -122.42, 37.755),
    "Twin Peaks/Diamond Hts":(-122.45, 37.745, -122.43, 37.760),
}

# Build node → position lookup for the full GDF
_start_ids = gdf["start_node_id"].tolist() if "start_node_id" in gdf.columns else []
_end_ids   = gdf["end_node_id"].tolist()   if "end_node_id"   in gdf.columns else []
_node_to_pos: dict = {}
for _pi, (_u, _v) in enumerate(zip(_start_ids, _end_ids)):
    if pd.notna(_u):
        _node_to_pos.setdefault(_u, []).append(_pi)
    if pd.notna(_v):
        _node_to_pos.setdefault(_v, []).append(_pi)

# WGS-84 centroids of all segments for spatial filtering
_gdf_wgs = gdf.to_crs("EPSG:4326")
_cx_all = np.asarray(_gdf_wgs.geometry.centroid.x.values, dtype=np.float64)
_cy_all = np.asarray(_gdf_wgs.geometry.centroid.y.values, dtype=np.float64)
_slopes_arr = np.asarray(gdf["street_incline"].astype(float).fillna(float('nan')), dtype=np.float64)
_seg_lens   = np.asarray(seg_lengths.values, dtype=np.float64)  # aligned with gdf positional index

if not _start_ids:
    print("  SKIPPED — start_node_id / end_node_id columns not present")
else:
    for _zone, (_w, _s, _e, _n) in _HILLY_ZONES.items():
        # Positional indices of short, zero-slope segments in this zone
        _in_zone = (
            (_cx_all >= _w) & (_cx_all <= _e) &
            (_cy_all >= _s) & (_cy_all <= _n)
        )
        _zero_short = (
            _in_zone &
            (_seg_lens < DEM_RESOLUTION_M) &
            (_slopes_arr == 0.0)
        )
        _candidates = np.where(_zero_short)[0]  # positional indices

        print(f"\n  Zone: {_zone}  —  {int(_zero_short.sum()):,} zero-slope short segments")
        if len(_candidates) == 0:
            print("    (none found)")
            continue

        # Sample up to 10 for reporting
        _sample_pos = _candidates[:10]
        print(f"  {'street_id':>20s}  {'seg_m':>6s}  {'stored':>8s}  {'nearest_long_slope':>18s}  {'long_street_id':>20s}")
        print("  " + "-"*80)

        for _pos in _sample_pos:
            _sid = str(gdf.iloc[_pos]["street_id"])[:20]
            _seg_m = float(_seg_lens[_pos])
            _stored = float(_slopes_arr[_pos]) if _slopes_arr[_pos] is not None else float("nan")

            # BFS to nearest long segment (seg_len >= DEM_RESOLUTION_M) with a slope
            import heapq as _hq
            _u0, _v0 = _start_ids[_pos], _end_ids[_pos]
            _heap: list = []
            _visited: set = set()
            for _nd in (_u0, _v0):
                if pd.notna(_nd):
                    _hq.heappush(_heap, (0.0, _nd))

            _found_slope: float | None = None
            _found_sid: str = "N/A"
            while _heap:
                _nd_dist, _nd = _hq.heappop(_heap)
                if _nd in _visited:
                    continue
                _visited.add(_nd)
                for _np2 in _node_to_pos.get(_nd, []):
                    if _np2 == _pos:
                        continue
                    _nbr_slope = float(_slopes_arr[_np2]) if not np.isnan(_slopes_arr[_np2]) else None
                    _nbr_len   = float(_seg_lens[_np2])
                    if _nbr_slope is not None and _nbr_len >= DEM_RESOLUTION_M:
                        _found_slope = float(_nbr_slope)
                        _found_sid   = str(gdf.iloc[_np2]["street_id"])[:20]
                        break
                    # Traverse through this segment
                    _nu, _nv = _start_ids[_np2], _end_ids[_np2]
                    _far = _nv if _nu == _nd else _nu
                    if pd.notna(_far) and _far not in _visited:
                        _hq.heappush(_heap, (_nd_dist + _nbr_len, _far))
                if _found_slope is not None:
                    break

            _ls = f"{_found_slope:.4f}%" if _found_slope is not None else "NOT FOUND"
            print(f"  {_sid:>20s}  {_seg_m:6.1f}  {_stored:8.4f}  {_ls:>18s}  {_found_sid:>20s}")

    print(f"\n  Interpretation:")
    print(f"  If 'nearest_long_slope' is non-zero for segments in hilly zones,")
    print(f"  the BFS propagation fix should replace those 0.0 values with real slopes.")
    print(f"  If 'nearest_long_slope' is also 0.0, the entire local street block is flat")
    print(f"  (e.g. the bottom of a valley) and slope=0.0 is correct.")


section("Done")
