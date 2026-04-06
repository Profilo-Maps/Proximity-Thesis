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
    _build_intersection_hulls,
    _assign_curb_ramp_geometries,
    _create_crosswalk_geometries,
    _snap_offset_endpoints,
    _CURB_RAMP_PROXIMITY_M,
)

OUTPUT_DIR = _IMPL_DIR.parent / "Output"
EDITOR_HTML = OUTPUT_DIR / "test_maps" / "county_editor.html"

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
        if best_idx is not None and best_dist < _CURB_RAMP_PROXIMITY_M * 2:
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


# ── POST /save endpoint ───────────────────────────────────────────────────────
@app.post("/save", response_model=SaveResponse)
async def save_edits(req: SaveRequest):
    gdf = _load_gdf(req.parquet)
    buffer_m = _CURB_RAMP_PROXIMITY_M
    working_bbox = BBox(
        minX=req.bbox.minX - buffer_m,
        minY=req.bbox.minY - buffer_m,
        maxX=req.bbox.maxX + buffer_m,
        maxY=req.bbox.maxY + buffer_m,
    )
    patch_idx = _filter_bbox(gdf, req.bbox, buffer_m)
    if patch_idx.empty:
        return SaveResponse(status="ok", stages_run=[], rows_affected=0, bbox_used=working_bbox)

    _apply_moved_endpoints(gdf, req.edits.moved_endpoints)
    _apply_added_nodes(gdf, patch_idx, req.edits.added_nodes)
    _apply_toggled_ramps(gdf, req.edits.toggled_curb_ramps)
    _apply_drawn_crosswalks(gdf, req.edits.drawn_crosswalks)

    patch_df = gdf.loc[patch_idx].copy()
    stages_run = []

    if req.dirty_from_stage <= STAGE_SNAP_ENDPOINTS:
        patch_df = _snap_offset_endpoints(patch_df)
        stages_run.append(STAGE_SNAP_ENDPOINTS)

    hulls_data = None
    if req.dirty_from_stage <= STAGE_BUILD_HULLS:
        patch_df, hulls_data = _build_intersection_hulls(patch_df)
        stages_run.append(STAGE_BUILD_HULLS)

    if req.dirty_from_stage <= STAGE_ASSIGN_RAMPS:
        patch_df = _assign_curb_ramp_geometries(
            patch_df,
            _precomputed_hulls=hulls_data,
        )
        stages_run.append(STAGE_ASSIGN_RAMPS)

    if req.dirty_from_stage <= STAGE_CREATE_XWALKS:
        patch_df, _ = _create_crosswalk_geometries(
            patch_df,
            _precomputed_hulls=hulls_data,
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
