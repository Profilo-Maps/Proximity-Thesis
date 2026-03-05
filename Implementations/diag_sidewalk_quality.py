"""Diagnostic: identify scrambled/overlapping separate sidewalk geometries in the parquet.

Geometry columns are stored as WKB hex strings in EPSG:32610 (UTM 10N).

Outputs:
  - Console report with sinuosity distribution and coincidence group stats
  - Output/diag_sidewalk_quality.csv  -- per-geometry quality table
  - Output/diag_sidewalk_quality.html -- Folium map
"""
import io
import sys

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import math
from collections import Counter, defaultdict
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import STRtree, wkb
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

PARQUET = Path("Output/San_Francisco_County_California_USA_network.parquet")
OUT_HTML = Path("Output/diag_sidewalk_quality.html")
OUT_CSV  = Path("Output/diag_sidewalk_quality.csv")

# ── thresholds ────────────────────────────────────────────────────────────────
SINUOSITY_THRESHOLD = 3.0   # length / crow-fly; above this = suspect
MIN_CROW_FLY_M      = 5.0   # ignore tiny stubs when judging sinuosity
OVERLAP_BUFFER_M    = 4.0   # how close two lines must be (metres) to count as coincident
OVERLAP_FRACTION    = 0.60  # fraction of shorter line within buffer for coincidence

# ── coordinate transformer UTM -> WGS84 (for Folium) ─────────────────────────
_to_wgs = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True).transform

def _reproj(geom: BaseGeometry) -> BaseGeometry:
    return transform(_to_wgs, geom)

# ── geometry helpers ──────────────────────────────────────────────────────────

def _parse_wkb(val) -> BaseGeometry | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        if isinstance(val, (bytes, bytearray)):
            return wkb.loads(val)
        if isinstance(val, str):
            return wkb.loads(bytes.fromhex(val))
        if isinstance(val, BaseGeometry):
            return val
        return None
    except Exception:
        return None


def _vertex_count(geom: BaseGeometry) -> int:
    if isinstance(geom, LineString):
        return len(geom.coords)
    if isinstance(geom, MultiLineString):
        return sum(len(ls.coords) for ls in geom.geoms)
    return 0


def _crow_fly_m(geom: BaseGeometry) -> float:
    if isinstance(geom, LineString):
        coords = list(geom.coords)
    elif isinstance(geom, MultiLineString):
        coords = [c for ls in geom.geoms for c in ls.coords]
    else:
        return 0.0
    if len(coords) < 2:
        return 0.0
    p0, p1 = coords[0], coords[-1]
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def _sinuosity(geom: BaseGeometry) -> float:
    cf = _crow_fly_m(geom)
    if cf < 1e-3:
        return float("inf")
    return float(geom.length) / cf


def _overlap_fraction(a: BaseGeometry, b: BaseGeometry) -> float:
    """Fraction of the shorter line that lies within OVERLAP_BUFFER_M of the other."""
    if a.length <= b.length:
        shorter, other = a, b
    else:
        shorter, other = b, a
    try:
        buf = other.buffer(OVERLAP_BUFFER_M)
        overlap = shorter.intersection(buf)
        return overlap.length / (shorter.length + 1e-9)
    except Exception:
        return 0.0


# ── load parquet ──────────────────────────────────────────────────────────────

print("Loading parquet ...")
df = gpd.read_parquet(PARQUET)
print(f"  {len(df)} rows")

# Check column presence
for side in ["left", "right"]:
    bc = f"sidewalk_{side}_buffered"
    pc = f"sidewalk_{side}_presence"
    gc = f"sidewalk_{side}_geometry"
    counts = {
        "geometry_non_null": df[gc].notna().sum() if gc in df.columns else "MISSING",
        "buffered_col": bc in df.columns,
        "presence_col": pc in df.columns,
    }
    print(f"  {gc}: {counts}")

# ── collect geometry records ──────────────────────────────────────────────────

records: list[dict] = []

for side in ["left", "right"]:
    geom_col     = f"sidewalk_{side}_geometry"
    buffered_col = f"sidewalk_{side}_buffered"
    presence_col = f"sidewalk_{side}_presence"

    if geom_col not in df.columns:
        continue

    for row_idx, row in df.iterrows():
        geom = _parse_wkb(row[geom_col])
        if geom is None or geom.is_empty:
            continue

        raw_buf = row[buffered_col] if buffered_col in df.columns else None
        is_buffered = str(raw_buf).lower() in ("yes", "true", "1") if raw_buf is not None else False
        presence     = str(row.get(presence_col, "")) if presence_col in df.columns else ""
        street_name  = str(row.get("name", ""))

        length  = float(geom.length)
        crow    = _crow_fly_m(geom)
        sinu    = _sinuosity(geom)
        n_verts = _vertex_count(geom)
        is_self = not geom.is_simple

        records.append({
            "row_idx":         row_idx,
            "side":            side,
            "is_buffered":     is_buffered,
            "presence":        presence,
            "street_name":     street_name,
            "length_m":        round(length, 2),
            "crow_fly_m":      round(crow, 2),
            "sinuosity":       round(sinu, 3),
            "n_vertices":      n_verts,
            "self_intersecting": is_self,
            "geom_utm":        geom,
        })

n_sep  = sum(1 for r in records if not r["is_buffered"])
n_buf  = sum(1 for r in records if r["is_buffered"])
print(f"Collected {len(records)} sidewalk geometry entries ({n_sep} separate, {n_buf} buffered)")

# ── flag suspect geometries ───────────────────────────────────────────────────

for r in records:
    r["suspect"] = bool(
        (r["sinuosity"] > SINUOSITY_THRESHOLD and r["crow_fly_m"] > MIN_CROW_FLY_M)
        or r["self_intersecting"]
    )

print(f"Suspect geometries: {sum(r['suspect'] for r in records)} / {len(records)}")

# ── find coincident groups (Union-Find) ───────────────────────────────────────

print("Building coincidence groups ...")

geoms_utm    = [r["geom_utm"] for r in records]
buf_geoms    = [g.buffer(OVERLAP_BUFFER_M) for g in geoms_utm]
tree         = STRtree(buf_geoms)

parent: list[int] = list(range(len(records)))

def _find(x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def _union(a: int, b: int) -> None:
    ra, rb = _find(a), _find(b)
    if ra != rb:
        parent[rb] = ra

for i, bg in enumerate(buf_geoms):
    for j in tree.query(bg):
        if j <= i:
            continue
        if _overlap_fraction(geoms_utm[i], geoms_utm[j]) >= OVERLAP_FRACTION:
            _union(i, j)

group_members: dict[int, list[int]] = defaultdict(list)
for i in range(len(records)):
    group_members[_find(i)].append(i)

coincident_groups = {k: v for k, v in group_members.items() if len(v) > 1}
print(f"Coincident groups (>=2 overlapping segments): {len(coincident_groups)}")

# Annotate with group id and keep/skip recommendation
for i, r in enumerate(records):
    r["group_id"] = _find(i)

for gid, members in coincident_groups.items():
    best = max(members, key=lambda i: records[i]["n_vertices"])
    for i in members:
        records[i]["group_recommended"] = "keep" if i == best else "skip_redundant"

for r in records:
    if "group_recommended" not in r:
        r["group_recommended"] = "keep_solo"

# ── console report ────────────────────────────────────────────────────────────

print("\n=== SINUOSITY DISTRIBUTION (separate sidewalks only) ===")
sep_sinu = sorted(
    r["sinuosity"] for r in records
    if not r["is_buffered"] and r["sinuosity"] != float("inf")
)
if sep_sinu:
    for pct in [50, 75, 90, 95, 99]:
        print(f"  p{pct}: {np.percentile(sep_sinu, pct):.3f}")
    print(f"  max (finite): {max(sep_sinu):.3f}")

print("\n=== COINCIDENT GROUP SIZE DISTRIBUTION ===")
sizes = [len(v) for v in coincident_groups.values()]
for sz, cnt in sorted(Counter(sizes).items()):
    print(f"  groups of size {sz}: {cnt}")

print("\n=== RECOMMENDATIONS ===")
for label in ["keep_solo", "keep", "skip_redundant"]:
    print(f"  {label}: {sum(1 for r in records if r['group_recommended'] == label)}")
print(f"  suspect AND recommended-keep: "
      f"{sum(1 for r in records if r['suspect'] and 'keep' in r['group_recommended'])}")

# ── export CSV ────────────────────────────────────────────────────────────────

csv_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("geom_utm",)}
    for r in records
])
csv_df.to_csv(OUT_CSV, index=False)
print(f"\nCSV written -> {OUT_CSV}")

# ── Folium map ────────────────────────────────────────────────────────────────

print("Rendering map ...")

m = folium.Map(location=[37.775, -122.42], zoom_start=14, tiles="CartoDB positron")

COLOR = {
    "keep_solo":      "#2196F3",  # blue
    "keep":           "#4CAF50",  # green (best of group)
    "skip_redundant": "#FF5722",  # orange
}
SUSPECT_COLOR = "#9C27B0"  # purple

for r in records:
    geom_wgs = _reproj(r["geom_utm"])
    rec      = r["group_recommended"]
    color    = SUSPECT_COLOR if r["suspect"] else COLOR.get(rec, "#999")

    try:
        if isinstance(geom_wgs, LineString):
            coords = [[c[1], c[0]] for c in geom_wgs.coords]
        elif isinstance(geom_wgs, MultiLineString):
            coords = [[c[1], c[0]] for ls in geom_wgs.geoms for c in ls.coords]
        else:
            continue
    except Exception:
        continue

    tooltip = (
        f"row={r['row_idx']} side={r['side']} name={r['street_name']!r}<br>"
        f"len={r['length_m']}m crow={r['crow_fly_m']}m "
        f"sinu={r['sinuosity']} verts={r['n_vertices']}<br>"
        f"buffered={r['is_buffered']} suspect={r['suspect']} rec={rec}"
    )

    folium.PolyLine(coords, color=color, weight=2, opacity=0.75, tooltip=tooltip).add_to(m)

legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
            padding:10px;border-radius:6px;font-size:12px;line-height:1.8">
  <b>Sidewalk geometry quality</b><br>
  <span style="color:#2196F3">&#9644;</span> keep_solo (unique)<br>
  <span style="color:#4CAF50">&#9644;</span> keep (most vertices in group)<br>
  <span style="color:#FF5722">&#9644;</span> skip_redundant (fewer vertices)<br>
  <span style="color:#9C27B0">&#9644;</span> suspect (high sinuosity / self-intersecting)<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[attr-defined]
m.save(str(OUT_HTML))
print(f"Map written -> {OUT_HTML}")
print("Done.")
