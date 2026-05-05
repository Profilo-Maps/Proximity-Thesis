"""County Map Editor — FastAPI server.

Serves parquet data as GeoJSON, accepts edit changesets via POST /save,
and re-runs pipeline stages on affected rows. The React frontend (editor/web)
runs on Vite dev server and proxies API calls here.

Run:
    cd editor/server && uv run uvicorn editor_server:app --reload --port 8000
"""
from __future__ import annotations

import json as _json
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

import re

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from shapely.geometry import MultiPoint, Point, box
from shapely.geometry.base import BaseGeometry

# Ensure Implementations/ is on sys.path so we can import ProximityModel
_SERVER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVER_DIR.parent.parent
_IMPL_DIR = _PROJECT_ROOT / "Implementations"
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

from ProximityModel import (
    PipelineConfig,
    step_10_snap_endpoints,
    step_12_curb_ramps_and_hulls,
    step_13_merge_hulls,
    step_14_crosswalk_slots,
    step_15_crosswalk_geometries,
)

# Buffer (metres) used to expand edit bboxes before filtering affected rows.
_EDIT_PROXIMITY_M = 20.0

# Shared config for pipeline step re-runs on edited patches.
_PIPELINE_CONFIG = PipelineConfig(
    place_name="editor",
    output_path="",
)

OUTPUT_DIR = _PROJECT_ROOT / "Output"

app = FastAPI(title="Proximity Editor")

# ── In-memory parquet cache ──────────────────────────────────────────────────
_gdf_cache: dict[str, gpd.GeoDataFrame] = {}


def _load_gdf(parquet_name: str) -> gpd.GeoDataFrame:
    """Load a parquet into the cache (or return cached copy)."""
    if parquet_name not in _gdf_cache:
        path = OUTPUT_DIR / parquet_name
        if not path.exists():
            raise HTTPException(404, f"Parquet not found: {parquet_name}")
        _gdf_cache[parquet_name] = gpd.read_parquet(path)
    return _gdf_cache[parquet_name]


# ── Routes ────────────────────────────────────────────────────────────────────
# In dev mode, the React app runs on Vite (port 5173) and proxies /api → here.
# In production, serve the built React app from editor/web/dist/.
_DIST_DIR = _SERVER_DIR.parent / "web" / "dist"


@app.get("/", response_class=HTMLResponse)
async def serve_editor():
    index = _DIST_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h3>Run <code>cd editor/web && npm run build</code> first, "
            "or use <code>npm run dev</code> for development.</h3>"
        )
    return HTMLResponse(index.read_text(encoding="utf-8"))


from fastapi.staticfiles import StaticFiles  # noqa: E402

if _DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST_DIR / "assets")), name="assets")


@app.get("/parquets")
async def list_parquets():
    """Return list of available parquet files in the Output directory."""
    files = sorted(p.name for p in OUTPUT_DIR.glob("*.parquet"))
    return {"files": files}


@app.get("/parquet/{filename}")
async def serve_parquet(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(path, media_type="application/octet-stream")


@app.get("/hulls/{parquet_name}")
async def get_hulls(
    parquet_name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
):
    """Return hull polygons and crosswalk slots for the requested viewport bbox."""
    gdf = _load_gdf(parquet_name)
    bbox = BBox(minX=min_lon, minY=min_lat, maxX=max_lon, maxY=max_lat)
    # Buffer ~200 m in degrees so edge-straddling intersections are included
    patch_idx = _filter_bbox(gdf, bbox, buffer_m=0.002)
    if len(patch_idx) == 0:
        return {"hulls": [], "slots": []}

    patch_df = gdf.loc[patch_idx].copy()
    _, hulls = step_12_curb_ramps_and_hulls(patch_df, _PIPELINE_CONFIG)
    if hulls.empty:
        return {"hulls": [], "slots": []}
    hulls = step_13_merge_hulls(hulls, _PIPELINE_CONFIG)
    step_14_crosswalk_slots(patch_df, hulls, _PIPELINE_CONFIG)

    hull_features = []
    for idx in hulls.index:
        geom = hulls.at[idx, "geometry"]
        if geom is None or geom.is_empty:
            continue
        node_id = hulls.at[idx, "node_id"]
        hull_features.append({
            "type": "Feature",
            "geometry": geom.__geo_interface__,
            "properties": {"tip": f"Intersection {node_id}", "node_id": str(node_id)},
        })

    slot_features = []
    for slot in hulls.attrs.get("crosswalk_slots", []):
        geom = slot.get("slot_geom")
        if geom is None or geom.is_empty:
            continue
        slot_features.append({
            "type": "Feature",
            "geometry": geom.__geo_interface__,
            "properties": {"tip": f"Crosswalk slot · node {slot.get('node_id', '')}"},
        })

    return {"hulls": hull_features, "slots": slot_features}


# ── Feature serving ──────────────────────────────────────────────────────────
# Colour constants (matching client-side C object)
_FC = {
    "street": "#c0392b", "bk_sep": "#006400", "bk_off": "#90ee90",
    "sw_sep": "#00008b", "sw_off": "#add8e6", "node": "#8b0000",
    "ramp": "#ff8c00", "calm": "#6a0dad", "cross": "#ff00ff", "cret": "#008080",
}

_HW_RE = {
    1: re.compile(
        r"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link"
        r"|secondary|secondary_link)$", re.I),
    2: re.compile(
        r"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link"
        r"|secondary|secondary_link|tertiary|tertiary_link|busway|cycleway)$", re.I),
}


def _blend_black(hex_color: str, opacity: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    f = 1 - max(0.0, min(1.0, opacity))
    return f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"


def _incline_color(base: str, raw: Any) -> str:
    try:
        v = abs(float(raw))
    except (TypeError, ValueError):
        return base
    if np.isnan(v):
        return base
    return _blend_black(base, min(v / 30.0, 1.0) * 0.5)


def _geo(val: Any) -> dict | None:
    """Geometry → GeoJSON dict, or None if missing/empty."""
    if val is None or not isinstance(val, BaseGeometry) or val.is_empty:
        return None
    return val.__geo_interface__


def _midpoint(geom: BaseGeometry) -> list[float] | None:
    """Return [lat, lon] midpoint for segment index building."""
    if isinstance(geom, Point):
        return [geom.y, geom.x]
    coords = list(geom.coords) if hasattr(geom, "coords") else []
    if not coords:
        return None
    m = coords[len(coords) // 2]
    return [m[1], m[0]]


def _safe_str(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return str(val)


def _get_tier(zoom: float) -> int:
    if zoom >= 18:
        return 5
    if zoom >= 17:
        return 4
    if zoom >= 15:
        return 3
    if zoom >= 13:
        return 2
    return 1


def _parse_feature_types(val: Any) -> list[str] | None:
    """Parse street_feature_types from list, JSON string, or repr."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            r = _json.loads(val)
            if isinstance(r, list):
                return r
        except Exception:
            pass
        return [val]
    return None


def _has(gdf: gpd.GeoDataFrame, col: str) -> bool:
    return col in gdf.columns


def _val(gdf: gpd.GeoDataFrame, idx: int, col: str) -> Any:
    return gdf.at[idx, col] if col in gdf.columns else None


def _build_features(
    gdf: gpd.GeoDataFrame, indices: pd.Index, tier: int,
) -> list[dict]:
    """Build GeoJSON features for the given row indices and tier level."""
    features: list[dict] = []
    seen_nodes: set[str] = set()
    seen_ramps: set[str] = set()

    for idx in indices:
        seg_id = str(gdf.at[idx, "street_grid_id"] or "")

        # ── Streets (always) ─────────────────────────────────────────────
        street_geom = gdf.at[idx, "street_geometry"]
        sg = _geo(street_geom)
        if not sg:
            continue
        col = _incline_color(_FC["street"], _val(gdf, idx, "street_incline"))
        features.append({
            "type": "Feature", "geometry": sg,
            "properties": {
                "_t": "street", "_color": col,
                "_seg_id": seg_id, "_fid": seg_id,
                "_mid": _midpoint(street_geom),
                "name": _safe_str(_val(gdf, idx, "name")),
                "highway": _safe_str(_val(gdf, idx, "highway")),
            },
        })

        # ── Bikeways (tier ≥ 2) ──────────────────────────────────────────
        if tier >= 2:
            for side in ("left", "right"):
                for n in (1, 2):
                    gcol = f"bikeway_{side}_{n}_geometry"
                    bg = _geo(_val(gdf, idx, gcol))
                    if not bg:
                        continue
                    is_off = _safe_str(_val(gdf, idx, f"bikeway_{side}_{n}_offset")) == "yes"
                    base = _FC["bk_off"] if is_off else _FC["bk_sep"]
                    bk_col = _incline_color(base, _val(gdf, idx, f"bikeway_{side}_{n}_incline"))
                    bk_key = f"bk_{side}_{n}_{seg_id}"
                    bk_geom = gdf.at[idx, gcol]
                    features.append({
                        "type": "Feature", "geometry": bg,
                        "properties": {
                            "_t": "bikeway", "_color": bk_col,
                            "_off": "yes" if is_off else "no",
                            "_seg_id": seg_id, "_fid": bk_key,
                            "_mid": _midpoint(bk_geom) if isinstance(bk_geom, BaseGeometry) else None,
                            "_layerId": "px-bk-off" if is_off else "px-bk-sep",
                        },
                    })

        # ── Sidewalks (tier ≥ 3) ─────────────────────────────────────────
        if tier >= 3:
            for side in ("left", "right"):
                gcol = f"sidewalk_{side}_geometry"
                sw_geom = _val(gdf, idx, gcol)
                swg = _geo(sw_geom)
                if not swg:
                    continue
                is_off = _safe_str(_val(gdf, idx, f"sidewalk_{side}_offset")) == "yes"
                sw_col = _FC["sw_off"] if is_off else _FC["sw_sep"]
                sw_id = _safe_str(_val(gdf, idx, f"sidewalk_{side}_ID")) or ""
                sw_fid = sw_id or f"sw_{side}_{seg_id}"
                features.append({
                    "type": "Feature", "geometry": swg,
                    "properties": {
                        "_t": "sidewalk", "_color": sw_col,
                        "_off": "yes" if is_off else "no",
                        "_seg_id": seg_id, "_fid": sw_fid,
                        "_sw_id": sw_id, "_side": side,
                        "_mid": _midpoint(sw_geom) if isinstance(sw_geom, BaseGeometry) else None,
                        "_layerId": "px-sw-off" if is_off else "px-sw-sep",
                    },
                })

            # ── Crosswalks ────────────────────────────────────────────────
            for pos in ("start", "end"):
                xwg = _geo(_val(gdf, idx, f"crosswalk_{pos}_geometry"))
                if xwg:
                    features.append({
                        "type": "Feature", "geometry": xwg,
                        "properties": {"_t": "crosswalk", "_seg_id": seg_id, "_xw_pos": pos},
                    })

            # ── Curb returns ──────────────────────────────────────────────
            crg = _geo(_val(gdf, idx, "curb_return_geometry"))
            if crg:
                features.append({
                    "type": "Feature", "geometry": crg,
                    "properties": {"_t": "cret", "_seg_id": seg_id},
                })

            # ── Traffic calming ───────────────────────────────────────────
            feat_geom = _val(gdf, idx, "street_feature_geometry")
            feat_types = _val(gdf, idx, "street_feature_types")
            if feat_geom is not None and feat_types is not None:
                types = _parse_feature_types(feat_types)
                if types and isinstance(feat_geom, BaseGeometry) and not feat_geom.is_empty:
                    pts = list(feat_geom.geoms) if isinstance(feat_geom, MultiPoint) else [feat_geom]
                    for i, pt in enumerate(pts):
                        if i < len(types) and str(types[i]).startswith("traffic_calming"):
                            calm_fid = f"calm_{i}_{seg_id}"
                            features.append({
                                "type": "Feature",
                                "geometry": pt.__geo_interface__,
                                "properties": {"_t": "calm", "_fid": calm_fid, "_seg_id": seg_id, "_calm_type": str(types[i])},
                            })

        # ── Intersection nodes (tier ≥ 4) ─────────────────────────────────
        if tier >= 4:
            for prefix in ("start", "end"):
                int_col = f"{prefix}_node_is_intersection_node"
                if not _val(gdf, idx, int_col):
                    continue
                gcol = f"{prefix}_node_geometry"
                node_geom = _val(gdf, idx, gcol)
                ng = _geo(node_geom)
                if not ng:
                    continue
                nid = _safe_str(_val(gdf, idx, f"{prefix}_node_id")) or ""
                if nid in seen_nodes:
                    continue
                seen_nodes.add(nid)
                features.append({
                    "type": "Feature", "geometry": ng,
                    "properties": {"_t": "node", "_node_id": nid, "_seg_id": seg_id, "_fid": nid},
                })

        # ── Curb ramps (tier ≥ 5) ─────────────────────────────────────────
        if tier >= 5:
            for side in ("left", "right"):
                for pos in ("start", "end"):
                    for n in (1, 2, 3):
                        gcol = f"sidewalk_{side}_curbramp_{pos}_{n}_geometry"
                        rg = _geo(_val(gdf, idx, gcol))
                        if not rg:
                            continue
                        ramp_key = f"{side}_{pos}_{n}_{seg_id}"
                        if ramp_key in seen_ramps:
                            continue
                        seen_ramps.add(ramp_key)
                        features.append({
                            "type": "Feature", "geometry": rg,
                            "properties": {
                                "_t": "ramp", "_color": _FC["ramp"],
                                "_fid": ramp_key,
                                "_seg_id": seg_id, "_side": side,
                                "_position": pos, "_index": n,
                                "_ramp_disabled": "no",
                            },
                        })

    return features


@app.get("/features/{parquet_name}")
async def get_features(
    parquet_name: str,
    zoom: float,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
):
    """Return GeoJSON FeatureCollection for the viewport at the given zoom."""
    gdf = _load_gdf(parquet_name)
    tier = _get_tier(zoom)

    # Spatial filter — always use viewport bbox
    search_box = box(min_lon, min_lat, max_lon, max_lat)
    mask = gdf["street_geometry"].intersects(search_box)
    indices = gdf.index[mask]

    # Highway-type filter for lower tiers
    if tier <= 2:
        hw = gdf.loc[indices, "highway"].astype(str)
        hw_mask = hw.str.fullmatch(_HW_RE[tier], na=False)
        indices = indices[hw_mask]

    features = _build_features(gdf, indices, tier)

    return JSONResponse(_json_safe({"type": "FeatureCollection", "features": features}))


# ── Attribute rows for the table ─────────────────────────────────────────────
_ROW_COLS = [
    # IDs
    "street_grid_id",
    # Common
    "name", "highway", "surface",
    # Street / bikeway
    "lanes", "lane_width", "maxspeed", "oneway", "street_incline",
    # Crosswalk
    "crosswalk_start_type", "crosswalk_end_type",
    "crosswalk_start_marked", "crosswalk_end_marked",
    # Sidewalk / ramp placement
    "_side", "_position",
    # Node
    "is_intersection", "node_type",
    # Ramp
    "accessible",
]


def _json_safe(v: Any) -> Any:
    """Coerce a pandas/numpy value to a JSON-serializable Python type."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (list, np.ndarray)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, np.ndarray):
        return v.tolist()
    if pd.isna(v):
        return None
    return v


@app.get("/rows/{parquet_name}")
async def get_rows(
    parquet_name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
):
    """Return attribute rows for segments in the viewport (no geometry)."""
    gdf = _load_gdf(parquet_name)
    search_box = box(min_lon, min_lat, max_lon, max_lat)
    mask = gdf["street_geometry"].intersects(search_box)
    indices = gdf.index[mask]

    rows: list[dict] = []
    for idx in indices:
        row: dict[str, Any] = {}
        for col in _ROW_COLS:
            if col not in gdf.columns:
                row[col] = None
                continue
            row[col] = _json_safe(gdf.at[idx, col])
        rows.append(row)
    return {"rows": rows}


# ── Config management ─────────────────────────────────────────────────────────
# Global config: stored in Output/proximity_config_global.json
# City config:   stored in Output/{parquet_stem}_config.json
# City values override global values which override PipelineConfig defaults.

_GLOBAL_CONFIG_PATH = OUTPUT_DIR / "proximity_config_global.json"

# Fields excluded from the config UI (location/path fields set during pipeline run)
_CONFIG_UI_EXCLUDE = frozenset({"place_name", "output_path", "custom_filter"})


def _config_field_defs() -> list[dict]:
    """Return metadata for all editable PipelineConfig fields."""
    from dataclasses import MISSING
    defs = []
    for f in dc_fields(PipelineConfig):
        if f.name in _CONFIG_UI_EXCLUDE:
            continue
        ftype = "bool" if f.type == "bool" else (
            "int" if f.type == "int" else (
            "str" if f.type in ("str", "str | None") else "float"))
        default = f.default if f.default is not MISSING else None
        defs.append({"name": f.name, "type": ftype, "default": default})
    return defs


def _city_config_path(parquet_name: str) -> Path:
    stem = Path(parquet_name).stem
    return OUTPUT_DIR / f"{stem}_config.json"


def _load_json(path: Path) -> dict:
    if path.exists():
        return _json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.write_text(_json.dumps(data, indent=2), encoding="utf-8")


@app.get("/config/{parquet_name}")
async def get_config(parquet_name: str):
    """Return field definitions, global overrides, and city overrides."""
    global_overrides = _load_json(_GLOBAL_CONFIG_PATH)
    city_overrides = _load_json(_city_config_path(parquet_name))
    return {
        "fields": _config_field_defs(),
        "global": global_overrides,
        "city": city_overrides,
    }


class ConfigUpdate(BaseModel):
    tier: str  # "global" | "city"
    values: dict[str, Any]


@app.post("/config/{parquet_name}")
async def update_config(parquet_name: str, req: ConfigUpdate):
    """Update global or city config overrides."""
    if req.tier == "global":
        existing = _load_json(_GLOBAL_CONFIG_PATH)
        existing.update(req.values)
        # Remove keys set back to None (= revert to default)
        existing = {k: v for k, v in existing.items() if v is not None}
        _save_json(_GLOBAL_CONFIG_PATH, existing)
    elif req.tier == "city":
        path = _city_config_path(parquet_name)
        existing = _load_json(path)
        existing.update(req.values)
        existing = {k: v for k, v in existing.items() if v is not None}
        _save_json(path, existing)
    else:
        raise HTTPException(400, f"Unknown config tier: {req.tier}")

    # Rebuild the active pipeline config from merged values
    _rebuild_pipeline_config(parquet_name)
    return {"status": "ok"}


def _rebuild_pipeline_config(parquet_name: str) -> None:
    """Merge global + city overrides onto PipelineConfig defaults, update _PIPELINE_CONFIG."""
    global _PIPELINE_CONFIG
    merged: dict[str, Any] = {}
    for f in dc_fields(PipelineConfig):
        if f.name in _CONFIG_UI_EXCLUDE:
            continue
        merged[f.name] = f.default
    # Layer global overrides
    for k, v in _load_json(_GLOBAL_CONFIG_PATH).items():
        if k in merged:
            merged[k] = v
    # Layer city overrides
    for k, v in _load_json(_city_config_path(parquet_name)).items():
        if k in merged:
            merged[k] = v
    from dataclasses import replace as _dc_replace
    _PIPELINE_CONFIG = _dc_replace(
        _PIPELINE_CONFIG,
        **{k: v for k, v in merged.items() if k not in _CONFIG_UI_EXCLUDE},
    )


# ── Pipeline stage definitions ────────────────────────────────────────────────
STAGE_SNAP_ENDPOINTS  = 0
STAGE_BUILD_HULLS     = 1
STAGE_ASSIGN_RAMPS    = 2
STAGE_CREATE_XWALKS   = 3

STAGE_NAMES = {
    STAGE_SNAP_ENDPOINTS: "SNAP_ENDPOINTS",
    STAGE_BUILD_HULLS:    "BUILD_HULLS",
    STAGE_ASSIGN_RAMPS:   "ASSIGN_CURB_RAMPS",
    STAGE_CREATE_XWALKS:  "CREATE_CROSSWALKS",
}


# ── Request / response models ────────────────────────────────────────────────
class BBox(BaseModel):
    minX: float
    minY: float
    maxX: float
    maxY: float


class AddedNode(BaseModel):
    x: float
    y: float


class ConsolidatedHull(BaseModel):
    node_keys: list[list[float]]  # [[x1,y1], [x2,y2], ...]
    buffer_m: float | None = None  # None = auto-compute from gap distance


class MovedEndpoint(BaseModel):
    node_id: str
    new_x: float
    new_y: float
    rubber_band_segments: list[str]  # street_grid_id values


class ToggledCurbRamp(BaseModel):
    segment_id: str
    side: str       # "left" | "right"
    position: str   # "start" | "end"
    index: int      # 1, 2, or 3
    enabled: bool


class RampRef(BaseModel):
    segment_id: str
    side: str
    position: str
    index: int


class DrawnCrosswalk(BaseModel):
    ramp_a: RampRef
    ramp_b: RampRef


class EditedHull(BaseModel):
    node_id: str
    geometry: dict  # GeoJSON Polygon
    utm_anchor: dict | None = None  # {x, y} for bbox computation


class AttrEdit(BaseModel):
    street_grid_id: str
    col: str
    old_value: str | None = None
    new_value: str | None = None


class DeletedPoint(BaseModel):
    node_id: str
    x: float
    y: float


class DeletedHull(BaseModel):
    node_id: str


class MergedSegments(BaseModel):
    surviving_seg_id: str
    consumed_seg_id: str
    shared_node_id: str


class DrawnSegment(BaseModel):
    coordinates: list[list[float]]  # [[lng, lat], ...]


class Edits(BaseModel):
    added_nodes: list[AddedNode] = []
    consolidated_hulls: list[ConsolidatedHull] = []
    moved_endpoints: list[MovedEndpoint] = []
    toggled_curb_ramps: list[ToggledCurbRamp] = []
    drawn_crosswalks: list[DrawnCrosswalk] = []
    edited_hulls: list[EditedHull] = []
    attr_edits: list[AttrEdit] = []
    deleted_points: list[DeletedPoint] = []
    deleted_hulls: list[DeletedHull] = []
    merged_segments: list[MergedSegments] = []
    drawn_segments: list[DrawnSegment] = []


# Columns that callers may not overwrite (IDs, geometries, derived cols)
_ATTR_EDIT_BLOCKLIST = frozenset({
    "street_grid_id", "start_node_id", "end_node_id",
    "start_node_is_intersection_node", "end_node_is_intersection_node",
})


class SaveRequest(BaseModel):
    parquet: str
    dirty_from_stage: int
    bbox: BBox
    edits: Edits


class SaveResponse(BaseModel):
    status: str
    stages_run: list[int]
    rows_affected: int
    bbox_used: BBox


# ── Bbox filtering helper ────────────────────────────────────────────────────
def _filter_bbox(gdf: gpd.GeoDataFrame, bbox: BBox, buffer_m: float = 0.0) -> pd.Index:
    """Return index labels of rows whose street_geometry intersects the buffered bbox."""
    search_box = box(
        bbox.minX - buffer_m, bbox.minY - buffer_m,
        bbox.maxX + buffer_m, bbox.maxY + buffer_m,
    )
    geom_col = gdf.geometry if gdf.geometry.name == "street_geometry" else gdf["street_geometry"]
    mask = geom_col.intersects(search_box)
    return gdf.index[mask]


# ── Edit application functions ────────────────────────────────────────────────
def _apply_moved_endpoints(gdf: gpd.GeoDataFrame, edits: list[MovedEndpoint]) -> None:
    """Move node positions and rubber-band connected segment geometries in-place."""
    for me in edits:
        node_id = me.node_id
        new_pt = Point(me.new_x, me.new_y)
        for prefix in ("start", "end"):
            id_col = f"{prefix}_node_id"
            geom_col = f"{prefix}_node_geometry"
            if id_col not in gdf.columns or geom_col not in gdf.columns:
                continue
            node_mask = gdf[id_col].astype(str) == str(node_id)
            if not node_mask.any():
                continue
            gdf.loc[node_mask, geom_col] = new_pt
            for idx in gdf.index[node_mask]:
                seg_id = str(gdf.at[idx, "street_grid_id"])
                if seg_id not in me.rubber_band_segments:
                    continue
                street_geom = gdf.at[idx, "street_geometry"]
                if street_geom is None or street_geom.is_empty:
                    continue
                coords = list(street_geom.coords)
                if prefix == "start":
                    coords[0] = (me.new_x, me.new_y)
                else:
                    coords[-1] = (me.new_x, me.new_y)
                from shapely.geometry import LineString
                gdf.at[idx, "street_geometry"] = LineString(coords)


def _apply_added_nodes(gdf: gpd.GeoDataFrame, patch_idx: pd.Index,
                       edits: list[AddedNode]) -> None:
    """Mark nearest segment endpoints as intersection nodes for added points."""
    if not edits:
        return
    for an in edits:
        placed = Point(an.x, an.y)
        best_dist = float("inf")
        best_idx = None
        best_prefix = None
        for idx in patch_idx:
            for prefix in ("start", "end"):
                geom_col = f"{prefix}_node_geometry"
                if geom_col not in gdf.columns:
                    continue
                pt = gdf.at[idx, geom_col]
                if pt is None or (hasattr(pt, "is_empty") and pt.is_empty):
                    continue
                d = placed.distance(pt)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
                    best_prefix = prefix
        if best_idx is not None and best_dist < _EDIT_PROXIMITY_M * 2:
            gdf.at[best_idx, f"{best_prefix}_node_is_intersection_node"] = True


def _apply_toggled_ramps(gdf: gpd.GeoDataFrame, edits: list[ToggledCurbRamp]) -> None:
    """Enable/disable curb ramps by setting/clearing geometry columns."""
    for tr in edits:
        mask = gdf["street_grid_id"].astype(str) == str(tr.segment_id)
        if not mask.any():
            continue
        base = f"sidewalk_{tr.side}_curbramp_{tr.position}_{tr.index}"
        geom_col = f"{base}_geometry"
        if geom_col not in gdf.columns:
            continue
        if not tr.enabled:
            gdf.loc[mask, geom_col] = None


def _apply_drawn_crosswalks(gdf: gpd.GeoDataFrame, edits: list[DrawnCrosswalk]) -> None:
    """Insert straight-line crosswalk geometries between two curb ramps."""
    for dc in edits:
        a_mask = gdf["street_grid_id"].astype(str) == str(dc.ramp_a.segment_id)
        b_mask = gdf["street_grid_id"].astype(str) == str(dc.ramp_b.segment_id)
        if not a_mask.any() or not b_mask.any():
            continue
        a_idx = gdf.index[a_mask][0]
        b_idx = gdf.index[b_mask][0]
        a_geom_col = f"sidewalk_{dc.ramp_a.side}_curbramp_{dc.ramp_a.position}_{dc.ramp_a.index}_geometry"
        b_geom_col = f"sidewalk_{dc.ramp_b.side}_curbramp_{dc.ramp_b.position}_{dc.ramp_b.index}_geometry"
        a_pt = gdf.at[a_idx, a_geom_col] if a_geom_col in gdf.columns else None
        b_pt = gdf.at[b_idx, b_geom_col] if b_geom_col in gdf.columns else None
        if a_pt is None or b_pt is None:
            continue
        from shapely.geometry import LineString
        xw_line = LineString([a_pt, b_pt])
        pos = dc.ramp_b.position
        xw_geom_col = f"crosswalk_{pos}_geometry"
        xw_type_col = f"crosswalk_{pos}_type"
        if xw_geom_col in gdf.columns:
            gdf.at[b_idx, xw_geom_col] = xw_line
        if xw_type_col in gdf.columns:
            gdf.at[b_idx, xw_type_col] = "manual"


# ── Attribute edits ──────────────────────────────────────────────────────────
def _apply_deleted_points(gdf: gpd.GeoDataFrame, edits: list[DeletedPoint]) -> None:
    """Mark intersection nodes as non-intersection and clear their geometry."""
    for dp in edits:
        node_id = dp.node_id
        for prefix in ("start", "end"):
            id_col = f"{prefix}_node_id"
            int_col = f"{prefix}_node_is_intersection_node"
            geom_col = f"{prefix}_node_geometry"
            if id_col not in gdf.columns:
                continue
            mask = gdf[id_col].astype(str) == str(node_id)
            if not mask.any():
                continue
            if int_col in gdf.columns:
                gdf.loc[mask, int_col] = False
            if geom_col in gdf.columns:
                gdf.loc[mask, geom_col] = None


def _apply_deleted_hulls(gdf: gpd.GeoDataFrame, edits: list[DeletedHull]) -> None:
    """Mark nodes as non-intersection so hulls are not generated for them."""
    for dh in edits:
        node_id = dh.node_id
        for prefix in ("start", "end"):
            id_col = f"{prefix}_node_id"
            int_col = f"{prefix}_node_is_intersection_node"
            if id_col not in gdf.columns:
                continue
            mask = gdf[id_col].astype(str) == str(node_id)
            if not mask.any():
                continue
            if int_col in gdf.columns:
                gdf.loc[mask, int_col] = False


def _apply_merged_segments(gdf: gpd.GeoDataFrame, edits: list[MergedSegments]) -> gpd.GeoDataFrame:
    """Merge two segments: concatenate geometries onto the survivor, drop the consumed row."""
    from shapely.geometry import LineString
    drop_indices = []
    for ms in edits:
        surv_mask = gdf["street_grid_id"].astype(str) == str(ms.surviving_seg_id)
        cons_mask = gdf["street_grid_id"].astype(str) == str(ms.consumed_seg_id)
        if not surv_mask.any() or not cons_mask.any():
            continue
        surv_idx = gdf.index[surv_mask][0]
        cons_idx = gdf.index[cons_mask][0]

        surv_geom = gdf.at[surv_idx, "street_geometry"]
        cons_geom = gdf.at[cons_idx, "street_geometry"]
        if surv_geom is None or cons_geom is None:
            continue

        sc = list(surv_geom.coords)
        cc = list(cons_geom.coords)

        # Determine connection order via shared node
        eps = 1e-7
        def _close(a, b):
            return abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps

        if _close(sc[-1], cc[0]):
            merged_coords = sc + cc[1:]
        elif _close(sc[-1], cc[-1]):
            merged_coords = sc + list(reversed(cc))[1:]
        elif _close(sc[0], cc[0]):
            merged_coords = list(reversed(sc)) + cc[1:]
        elif _close(sc[0], cc[-1]):
            merged_coords = cc + sc[1:]
        else:
            continue  # not connected

        gdf.at[surv_idx, "street_geometry"] = LineString(merged_coords)

        # Transfer end-node info: the surviving segment inherits the consumed segment's
        # non-shared endpoint node
        shared_nid = str(ms.shared_node_id)
        for prefix_s in ("start", "end"):
            s_nid_col = f"{prefix_s}_node_id"
            if s_nid_col not in gdf.columns:
                continue
            if str(gdf.at[surv_idx, s_nid_col]) == shared_nid:
                # This end of the survivor is the shared node — replace with consumed's other end
                for prefix_c in ("start", "end"):
                    c_nid_col = f"{prefix_c}_node_id"
                    if c_nid_col not in gdf.columns:
                        continue
                    if str(gdf.at[cons_idx, c_nid_col]) != shared_nid:
                        # Copy node info from consumed's non-shared end
                        for suffix in ("_id", "_geometry", "_is_intersection_node"):
                            src = f"{prefix_c}_node{suffix}"
                            dst = f"{prefix_s}_node{suffix}"
                            if src in gdf.columns and dst in gdf.columns:
                                gdf.at[surv_idx, dst] = gdf.at[cons_idx, src]
                        break
                break

        drop_indices.append(cons_idx)

    if drop_indices:
        gdf = gdf.drop(index=drop_indices).reset_index(drop=True)
    return gdf


def _apply_drawn_segments(gdf: gpd.GeoDataFrame, edits: list[DrawnSegment]) -> gpd.GeoDataFrame:
    """Add new segment rows to the GeoDataFrame from user-drawn lines."""
    from shapely.geometry import LineString, Point
    if not edits:
        return gdf

    # Find max existing grid_id to generate new sequential IDs
    existing_ids = gdf["street_grid_id"].astype(str).tolist()
    max_numeric = 0
    for sid in existing_ids:
        # Extract trailing digits
        digits = "".join(c for c in sid if c.isdigit())
        if digits:
            max_numeric = max(max_numeric, int(digits))

    new_rows = []
    for i, ds in enumerate(edits):
        coords = [(c[0], c[1]) for c in ds.coordinates]
        if len(coords) < 2:
            continue
        line = LineString(coords)
        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])
        max_numeric += 1
        new_id = f"drawn_{max_numeric}"

        row: dict = {"street_grid_id": new_id, "street_geometry": line}
        # Set node endpoints
        if "start_node_id" in gdf.columns:
            row["start_node_id"] = f"{new_id}_start"
        if "end_node_id" in gdf.columns:
            row["end_node_id"] = f"{new_id}_end"
        if "start_node_geometry" in gdf.columns:
            row["start_node_geometry"] = start_pt
        if "end_node_geometry" in gdf.columns:
            row["end_node_geometry"] = end_pt
        if "start_node_is_intersection_node" in gdf.columns:
            row["start_node_is_intersection_node"] = False
        if "end_node_is_intersection_node" in gdf.columns:
            row["end_node_is_intersection_node"] = False
        if "highway" in gdf.columns:
            row["highway"] = "unclassified"
        if "name" in gdf.columns:
            row["name"] = None

        # Fill remaining columns with NaN/None
        for col in gdf.columns:
            if col not in row:
                row[col] = None

        new_rows.append(row)

    if new_rows:
        new_df = gpd.GeoDataFrame(new_rows, geometry="street_geometry", crs=gdf.crs)
        gdf = pd.concat([gdf, new_df], ignore_index=True)

    return gdf


def _apply_attr_edits(gdf: gpd.GeoDataFrame, edits: list[AttrEdit]) -> None:
    """Apply scalar attribute edits to non-geometry, non-ID columns."""
    for ae in edits:
        if ae.col.endswith("_geometry") or ae.col in _ATTR_EDIT_BLOCKLIST:
            continue
        if ae.col not in gdf.columns:
            continue
        mask = gdf["street_grid_id"].astype(str) == str(ae.street_grid_id)
        if not mask.any():
            continue
        dtype = gdf[ae.col].dtype
        try:
            val: Any = ae.new_value
            if val is not None:
                if pd.api.types.is_integer_dtype(dtype):
                    val = int(val)
                elif pd.api.types.is_float_dtype(dtype):
                    val = float(val)
        except (ValueError, TypeError):
            pass
        gdf.loc[mask, ae.col] = val


# ── Hull vertex edits ─────────────────────────────────────────────────────────
def _apply_edited_hulls(
    hulls: gpd.GeoDataFrame,
    edits: list[EditedHull],
) -> gpd.GeoDataFrame:
    """Override hull geometries with user-edited polygons."""
    if not edits:
        return hulls
    from shapely.geometry import shape as _shape
    for eh in edits:
        mask = hulls["node_id"].astype(str) == str(eh.node_id)
        if not mask.any():
            continue
        try:
            new_geom = _shape(eh.geometry)
            hulls.loc[mask, "geometry"] = new_geom
        except Exception:
            continue
    return hulls


# ── Hull consolidation ────────────────────────────────────────────────────────
def _compute_consolidation_buffer(
    edits: list[ConsolidatedHull],
    patch_df: gpd.GeoDataFrame,
) -> float | None:
    """Return an override hull_merge_buffer_m that forces selected nodes to merge.

    When the user picks explicit node_keys to consolidate, we widen the merge
    buffer to at least half the gap between the farthest pair of hulls in the
    edit set so that step_13 unions them. Returns None if no edits.
    """
    if not edits:
        return None

    # Run step_12 once to get current hulls (in WGS84)
    _, current_hulls = step_12_curb_ramps_and_hulls(patch_df.copy(), _PIPELINE_CONFIG)
    if current_hulls.empty:
        return None

    hulls_utm = current_hulls.to_crs("EPSG:32610")
    from shapely.geometry import Point as _Pt

    override_buf: float | None = None
    for ch in edits:
        # Resolve each (x, y) key to the nearest hull by centroid distance
        resolved_idxs: list[int] = []
        for k in ch.node_keys:
            target = _Pt(k[0], k[1])
            best_i = None
            best_d = float("inf")
            for i in hulls_utm.index:
                c = hulls_utm.at[i, "geometry"].centroid
                d = c.distance(target)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None:
                resolved_idxs.append(best_i)

        if len(resolved_idxs) < 2:
            continue

        if ch.buffer_m is not None:
            buf = ch.buffer_m
        else:
            max_gap = 0.0
            for i in range(len(resolved_idxs)):
                for j in range(i + 1, len(resolved_idxs)):
                    gi = hulls_utm.at[resolved_idxs[i], "geometry"]
                    gj = hulls_utm.at[resolved_idxs[j], "geometry"]
                    d = gi.distance(gj)
                    if d > max_gap:
                        max_gap = d
            buf = (max_gap / 2.0) + 0.5  # small epsilon to guarantee overlap

        if override_buf is None or buf > override_buf:
            override_buf = buf

    return override_buf


# ── POST /save endpoint ───────────────────────────────────────────────────────
_EMPTY_CROSSWALK_CACHE = gpd.GeoDataFrame(
    columns=["geometry", "type", "controlled", "marked", "markings",
             "signals", "island", "kerb", "tactile_paving",
             "traffic_calming", "continuous"],
    geometry="geometry",
    crs="EPSG:4326",
)


def _collect_affected_node_ids(edits: Edits) -> set[str]:
    """Extract all intersection node IDs that were directly affected by edits.

    This determines which intersections need pipeline re-analysis. Only segments
    connected to these nodes will have their hulls/ramps/crosswalks regenerated.
    """
    nids: set[str] = set()

    for me in edits.moved_endpoints:
        nid = me.node_id
        # Vertex drags use "segId:v0" format — extract the node from the segment
        if ":v" in nid:
            # Mid-vertex edit doesn't affect intersection analysis directly,
            # but endpoint edits (v0 or last) do. Include all for safety.
            nids.add(nid)
        else:
            nids.add(nid)

    for dp in edits.deleted_points:
        nids.add(dp.node_id)

    for dh in edits.deleted_hulls:
        nids.add(dh.node_id)

    for ms in edits.merged_segments:
        nids.add(ms.shared_node_id)

    for eh in edits.edited_hulls:
        nids.add(eh.node_id)

    for ch in edits.consolidated_hulls:
        # node_keys are [x, y] pairs — we'll resolve them spatially later
        pass

    for tr in edits.toggled_curb_ramps:
        # Ramp toggles affect the intersection at that segment endpoint
        nids.add(f"_ramp_{tr.segment_id}_{tr.position}")

    for dc in edits.drawn_crosswalks:
        nids.add(f"_xw_{dc.ramp_a.segment_id}_{dc.ramp_a.position}")
        nids.add(f"_xw_{dc.ramp_b.segment_id}_{dc.ramp_b.position}")

    # added_nodes and drawn_segments affect new geometry — include them via bbox
    return nids


def _filter_affected_rows(
    gdf: gpd.GeoDataFrame,
    patch_idx: pd.Index,
    affected_nids: set[str],
    edits: Edits,
) -> pd.Index:
    """From the bbox patch, return only rows connected to affected intersections.

    A row is included if its start_node_id or end_node_id matches an affected
    node, or if its street_grid_id was referenced in a merge/ramp/crosswalk edit.
    Falls back to the full patch_idx if no specific nodes were identified.
    """
    if not affected_nids and not edits.added_nodes and not edits.drawn_segments:
        return patch_idx

    # Collect affected segment IDs directly from edits
    affected_seg_ids: set[str] = set()
    for ms in edits.merged_segments:
        affected_seg_ids.add(ms.surviving_seg_id)
        affected_seg_ids.add(ms.consumed_seg_id)
    for tr in edits.toggled_curb_ramps:
        affected_seg_ids.add(tr.segment_id)
    for dc in edits.drawn_crosswalks:
        affected_seg_ids.add(dc.ramp_a.segment_id)
        affected_seg_ids.add(dc.ramp_b.segment_id)
    for ds in edits.drawn_segments:
        # Newly drawn segments may have been appended — include all drawn_ IDs
        pass
    for me in edits.moved_endpoints:
        for seg_id in me.rubber_band_segments:
            affected_seg_ids.add(seg_id)

    # Resolve node IDs: filter to real node IDs (not synthetic _ramp_/_xw_ keys)
    real_nids = {n for n in affected_nids if not n.startswith("_")}

    # Also resolve vertex-drag node IDs: "segId:v0" → find which nodes are at
    # the endpoints of that segment
    vertex_seg_ids: set[str] = set()
    for nid in list(real_nids):
        if ":v" in nid:
            seg_part = nid.split(":v")[0]
            vertex_seg_ids.add(seg_part)
            real_nids.discard(nid)

    # Build the row mask
    patch_df = gdf.loc[patch_idx]
    mask = pd.Series(False, index=patch_idx)

    # Match by node ID
    if real_nids:
        for prefix in ("start", "end"):
            id_col = f"{prefix}_node_id"
            if id_col in patch_df.columns:
                mask |= patch_df[id_col].astype(str).isin(real_nids)

    # Match by segment ID (direct references + vertex-drag segments)
    all_seg_refs = affected_seg_ids | vertex_seg_ids
    if all_seg_refs and "street_grid_id" in patch_df.columns:
        mask |= patch_df["street_grid_id"].astype(str).isin(all_seg_refs)

    # For vertex drags on segments, also include rows sharing a node with the
    # affected segment (the adjacent intersection)
    if vertex_seg_ids:
        seg_mask = patch_df["street_grid_id"].astype(str).isin(vertex_seg_ids)
        for prefix in ("start", "end"):
            id_col = f"{prefix}_node_id"
            if id_col not in patch_df.columns:
                continue
            endpoint_nids = set(patch_df.loc[seg_mask, id_col].astype(str).dropna())
            endpoint_nids.discard("")
            endpoint_nids.discard("nan")
            if endpoint_nids:
                for prefix2 in ("start", "end"):
                    id_col2 = f"{prefix2}_node_id"
                    if id_col2 in patch_df.columns:
                        mask |= patch_df[id_col2].astype(str).isin(endpoint_nids)

    # Always include newly added/drawn features (they're in the bbox already)
    if edits.added_nodes or edits.drawn_segments:
        # New features don't have known node_ids yet — include all rows near
        # added points by falling back to the full bbox patch
        for an in edits.added_nodes:
            placed = Point(an.x, an.y)
            for idx in patch_idx:
                for prefix in ("start", "end"):
                    gcol = f"{prefix}_node_geometry"
                    if gcol not in gdf.columns:
                        continue
                    pt = gdf.at[idx, gcol]
                    if pt is not None and hasattr(pt, "distance") and pt.distance(placed) < 0.0003:
                        mask.at[idx] = True

    result = patch_idx[mask]
    # If filtering produced nothing (e.g. all synthetic keys), fall back to full patch
    return result if len(result) > 0 else patch_idx


@app.post("/save", response_model=SaveResponse)
async def save_edits(req: SaveRequest):
    gdf = _load_gdf(req.parquet)
    buffer_m = _EDIT_PROXIMITY_M
    working_bbox = BBox(
        minX=req.bbox.minX - buffer_m,
        minY=req.bbox.minY - buffer_m,
        maxX=req.bbox.maxX + buffer_m,
        maxY=req.bbox.maxY + buffer_m,
    )
    parquet_path = OUTPUT_DIR / req.parquet
    patch_idx = _filter_bbox(gdf, req.bbox, buffer_m)
    if patch_idx.empty:
        return SaveResponse(status="ok", stages_run=[], rows_affected=0, bbox_used=working_bbox)

    # ── Apply direct edits ────────────────────────────────────────────────────
    _apply_moved_endpoints(gdf, req.edits.moved_endpoints)
    _apply_added_nodes(gdf, patch_idx, req.edits.added_nodes)
    _apply_toggled_ramps(gdf, req.edits.toggled_curb_ramps)
    _apply_drawn_crosswalks(gdf, req.edits.drawn_crosswalks)
    _apply_attr_edits(gdf, req.edits.attr_edits)
    _apply_deleted_points(gdf, req.edits.deleted_points)
    _apply_deleted_hulls(gdf, req.edits.deleted_hulls)
    gdf = _apply_merged_segments(gdf, req.edits.merged_segments)
    gdf = _apply_drawn_segments(gdf, req.edits.drawn_segments)
    # Update cache after structural changes (merge/draw may change indices)
    _gdf_cache[req.parquet] = gdf

    patch_idx = _filter_bbox(gdf, req.bbox, buffer_m)
    if patch_idx.empty:
        gdf.to_parquet(parquet_path)
        return SaveResponse(status="ok", stages_run=[], rows_affected=0, bbox_used=working_bbox)

    # ── Identify affected intersections ───────────────────────────────────────
    affected_nids = _collect_affected_node_ids(req.edits)
    intersection_idx = _filter_affected_rows(gdf, patch_idx, affected_nids, req.edits)

    # Full bbox patch for geometry-level ops (snap endpoints)
    full_patch_df = gdf.loc[patch_idx].copy()
    # Narrow patch for intersection analysis (hulls, ramps, crosswalks)
    intersection_patch_df = gdf.loc[intersection_idx].copy()

    # ── Hull consolidation: derive override merge buffer from the edit set ────
    override_buf = _compute_consolidation_buffer(req.edits.consolidated_hulls, intersection_patch_df)
    from dataclasses import replace as _dc_replace
    run_config = _PIPELINE_CONFIG
    if override_buf is not None:
        run_config = _dc_replace(_PIPELINE_CONFIG, hull_merge_buffer_m=override_buf)

    stages_run: list[int] = []

    # stage 99 = attr-only edits; skip all pipeline re-runs
    if req.dirty_from_stage >= 99:
        gdf.to_parquet(parquet_path)
        _gdf_cache[req.parquet] = gdf
        return SaveResponse(status="ok", stages_run=[], rows_affected=len(patch_idx), bbox_used=working_bbox)

    # Step 10: snap endpoints — runs on full bbox patch (geometry-wide)
    if req.dirty_from_stage <= STAGE_SNAP_ENDPOINTS:
        full_patch_df = step_10_snap_endpoints(full_patch_df, run_config)
        stages_run.append(STAGE_SNAP_ENDPOINTS)
        # Update the intersection patch with snapped results
        common = intersection_idx.intersection(patch_idx)
        intersection_patch_df = full_patch_df.loc[
            full_patch_df.index.isin(intersection_idx)
        ].copy()

    # Steps 12-15: intersection analysis — runs only on affected rows
    hulls: gpd.GeoDataFrame | None = None
    if req.dirty_from_stage <= STAGE_ASSIGN_RAMPS:
        intersection_patch_df, hulls = step_12_curb_ramps_and_hulls(intersection_patch_df, run_config)
        hulls = step_13_merge_hulls(hulls, run_config)
        hulls = _apply_edited_hulls(hulls, req.edits.edited_hulls)
        if req.dirty_from_stage <= STAGE_BUILD_HULLS:
            stages_run.append(STAGE_BUILD_HULLS)
        stages_run.append(STAGE_ASSIGN_RAMPS)

    if req.dirty_from_stage <= STAGE_CREATE_XWALKS:
        if hulls is None:
            intersection_patch_df, hulls = step_12_curb_ramps_and_hulls(intersection_patch_df, run_config)
            hulls = step_13_merge_hulls(hulls, run_config)
            hulls = _apply_edited_hulls(hulls, req.edits.edited_hulls)
        intersection_patch_df = step_14_crosswalk_slots(intersection_patch_df, hulls, run_config)
        intersection_patch_df = step_15_crosswalk_geometries(
            intersection_patch_df, hulls, _EMPTY_CROSSWALK_CACHE, run_config,
        )
        stages_run.append(STAGE_CREATE_XWALKS)

    # Write back: full patch for snap, intersection patch for analysis
    gdf.loc[patch_idx] = full_patch_df
    gdf.loc[intersection_idx] = intersection_patch_df
    gdf.to_parquet(parquet_path)
    _gdf_cache[req.parquet] = gdf

    n_total = len(patch_idx)
    n_intersection = len(intersection_idx)
    print(f"[save] {n_total} rows in bbox, {n_intersection} rows for intersection analysis "
          f"({len(affected_nids)} affected nodes)")

    return SaveResponse(
        status="ok",
        stages_run=stages_run,
        rows_affected=n_intersection,
        bbox_used=working_bbox,
    )


def _save_stream(req: SaveRequest):
    """Generator that yields NDJSON progress lines and runs the save pipeline."""
    import json as _j

    def _emit(stage: str, pct: int, **kw) -> str:
        return _j.dumps({"stage": stage, "pct": pct, **kw}) + "\n"

    gdf = _load_gdf(req.parquet)
    parquet_path = _OUTPUT_DIR / req.parquet
    buffer_m = _EDIT_PROXIMITY_M

    yield _emit("applying_edits", 5)

    working_bbox = BBox(
        minX=req.bbox.minX - buffer_m / 111_000,
        minY=req.bbox.minY - buffer_m / 111_000,
        maxX=req.bbox.maxX + buffer_m / 111_000,
        maxY=req.bbox.maxY + buffer_m / 111_000,
    )
    _apply_all_edits(gdf, req.edits)
    patch_idx = _filter_bbox(gdf, req.bbox, buffer_m)

    if patch_idx.empty:
        gdf.to_parquet(parquet_path)
        yield _emit("done", 100, stages_run=[], rows_affected=0)
        return

    affected_nids = _collect_affected_node_ids(req.edits)
    intersection_idx = _filter_affected_rows(gdf, patch_idx, affected_nids, req.edits)

    full_patch_df = gdf.loc[patch_idx].copy()
    intersection_patch_df = gdf.loc[intersection_idx].copy()

    override_buf = _compute_consolidation_buffer(req.edits.consolidated_hulls, intersection_patch_df)
    from dataclasses import replace as _dc_replace
    run_config = _PIPELINE_CONFIG
    if override_buf is not None:
        run_config = _dc_replace(_PIPELINE_CONFIG, hull_merge_buffer_m=override_buf)

    stages_run: list[int] = []

    if req.dirty_from_stage >= 99:
        yield _emit("writing_parquet", 90)
        gdf.to_parquet(parquet_path)
        _gdf_cache[req.parquet] = gdf
        yield _emit("done", 100, stages_run=[], rows_affected=len(patch_idx))
        return

    if req.dirty_from_stage <= STAGE_SNAP_ENDPOINTS:
        yield _emit("snap_endpoints", 20)
        full_patch_df = step_10_snap_endpoints(full_patch_df, run_config)
        stages_run.append(STAGE_SNAP_ENDPOINTS)
        intersection_patch_df = full_patch_df.loc[
            full_patch_df.index.isin(intersection_idx)
        ].copy()

    hulls = None
    if req.dirty_from_stage <= STAGE_ASSIGN_RAMPS:
        yield _emit("build_hulls", 40)
        intersection_patch_df, hulls = step_12_curb_ramps_and_hulls(intersection_patch_df, run_config)
        hulls = step_13_merge_hulls(hulls, run_config)
        hulls = _apply_edited_hulls(hulls, req.edits.edited_hulls)
        if req.dirty_from_stage <= STAGE_BUILD_HULLS:
            stages_run.append(STAGE_BUILD_HULLS)
        yield _emit("assign_ramps", 60)
        stages_run.append(STAGE_ASSIGN_RAMPS)

    if req.dirty_from_stage <= STAGE_CREATE_XWALKS:
        if hulls is None:
            intersection_patch_df, hulls = step_12_curb_ramps_and_hulls(intersection_patch_df, run_config)
            hulls = step_13_merge_hulls(hulls, run_config)
            hulls = _apply_edited_hulls(hulls, req.edits.edited_hulls)
        yield _emit("create_crosswalks", 75)
        intersection_patch_df = step_14_crosswalk_slots(intersection_patch_df, hulls, run_config)
        intersection_patch_df = step_15_crosswalk_geometries(
            intersection_patch_df, hulls, _EMPTY_CROSSWALK_CACHE, run_config,
        )
        stages_run.append(STAGE_CREATE_XWALKS)

    yield _emit("writing_parquet", 90)
    gdf.loc[patch_idx] = full_patch_df
    gdf.loc[intersection_idx] = intersection_patch_df
    gdf.to_parquet(parquet_path)
    _gdf_cache[req.parquet] = gdf

    n_intersection = len(intersection_idx)
    yield _emit("done", 100, stages_run=stages_run, rows_affected=n_intersection)


@app.post("/save/stream")
async def save_edits_stream(req: SaveRequest):
    import asyncio
    import queue
    import threading

    async def async_gen():
        q: queue.Queue[str | None] = queue.Queue()

        def run():
            try:
                for line in _save_stream(req):
                    q.put(line)
            except Exception as exc:
                import json as _j
                q.put(_j.dumps({"stage": "error", "pct": 0, "message": str(exc)}) + "\n")
            finally:
                q.put(None)  # sentinel

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield item

        thread.join()

    return StreamingResponse(async_gen(), media_type="application/x-ndjson")
