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
