"""County Map Editor — local server.

Serves the editor HTML and parquet files, accepts edit changesets via
POST /save, and re-runs pipeline stages on affected rows.

Run:
    cd Implementations && uv run uvicorn editor_server:app --reload --port 8081
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from shapely.geometry import Point, box

# Ensure Implementations/ is on sys.path so we can import ProximityModel
_IMPL_DIR = Path(__file__).resolve().parent
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

OUTPUT_DIR = _IMPL_DIR.parent / "Output"
EDITOR_HTML = _IMPL_DIR / "county_editor.html"

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
@app.get("/", response_class=HTMLResponse)
async def serve_editor():
    if not EDITOR_HTML.exists():
        raise HTTPException(404, "county_editor.html not found")
    return HTMLResponse(EDITOR_HTML.read_text(encoding="utf-8"))


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


class Edits(BaseModel):
    added_nodes: list[AddedNode] = []
    consolidated_hulls: list[ConsolidatedHull] = []
    moved_endpoints: list[MovedEndpoint] = []
    toggled_curb_ramps: list[ToggledCurbRamp] = []
    drawn_crosswalks: list[DrawnCrosswalk] = []


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
    patch_idx = _filter_bbox(gdf, req.bbox, buffer_m)
    if patch_idx.empty:
        return SaveResponse(status="ok", stages_run=[], rows_affected=0, bbox_used=working_bbox)

    # ── Apply direct edits ────────────────────────────────────────────────────
    _apply_moved_endpoints(gdf, req.edits.moved_endpoints)
    _apply_added_nodes(gdf, patch_idx, req.edits.added_nodes)
    _apply_toggled_ramps(gdf, req.edits.toggled_curb_ramps)
    _apply_drawn_crosswalks(gdf, req.edits.drawn_crosswalks)

    patch_df = gdf.loc[patch_idx].copy()

    # ── Hull consolidation: derive override merge buffer from the edit set ────
    override_buf = _compute_consolidation_buffer(req.edits.consolidated_hulls, patch_df)
    from dataclasses import replace as _dc_replace
    run_config = _PIPELINE_CONFIG
    if override_buf is not None:
        run_config = _dc_replace(_PIPELINE_CONFIG, hull_merge_buffer_m=override_buf)

    stages_run: list[int] = []

    if req.dirty_from_stage <= STAGE_SNAP_ENDPOINTS:
        patch_df = step_10_snap_endpoints(patch_df, run_config)
        stages_run.append(STAGE_SNAP_ENDPOINTS)

    # step_12 now places curb ramps AND builds hulls in one shot, so
    # STAGE_BUILD_HULLS and STAGE_ASSIGN_RAMPS collapse onto the same call.
    hulls: gpd.GeoDataFrame | None = None
    if req.dirty_from_stage <= STAGE_ASSIGN_RAMPS:
        patch_df, hulls = step_12_curb_ramps_and_hulls(patch_df, run_config)
        hulls = step_13_merge_hulls(hulls, run_config)
        if req.dirty_from_stage <= STAGE_BUILD_HULLS:
            stages_run.append(STAGE_BUILD_HULLS)
        stages_run.append(STAGE_ASSIGN_RAMPS)

    if req.dirty_from_stage <= STAGE_CREATE_XWALKS:
        if hulls is None:
            patch_df, hulls = step_12_curb_ramps_and_hulls(patch_df, run_config)
            hulls = step_13_merge_hulls(hulls, run_config)
        patch_df = step_14_crosswalk_slots(patch_df, hulls, run_config)
        patch_df = step_15_crosswalk_geometries(
            patch_df, hulls, _EMPTY_CROSSWALK_CACHE, run_config,
        )
        stages_run.append(STAGE_CREATE_XWALKS)

    gdf.loc[patch_idx] = patch_df
    parquet_path = OUTPUT_DIR / req.parquet
    gdf.to_parquet(parquet_path)

    return SaveResponse(
        status="ok",
        stages_run=stages_run,
        rows_affected=len(patch_idx),
        bbox_used=working_bbox,
    )
