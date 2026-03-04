"""Diagnostic script: why are Page St (SF), Roosevelt Ave & Dover St (Alameda)
missing buffered sidewalk geometries?"""

import pandas as pd
import geopandas as gpd
from shapely import wkb
from pathlib import Path

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

SF_PATH = Path(r"c:\Dev\Proximity\Output\San_Francisco_County_California_USA_network.parquet")
ALAMEDA_PATH = Path(r"c:\Dev\Proximity\Output\Alameda_County_California_USA_network.parquet")

COLS_OF_INTEREST = [
    "name", "highway", "street_id",
    "sidewalk_left_presence", "sidewalk_right_presence",
    "sidewalk_left_geometry", "sidewalk_right_geometry",
    "sidewalk_left_buffered", "sidewalk_right_buffered",
]

# Extra cols to check for STRtree suppression clues
EXTRA_COLS = [
    "sidewalk_left_ID", "sidewalk_right_ID",
]


def load_parquet(path: Path) -> gpd.GeoDataFrame:
    print(f"\n{'='*80}")
    print(f"Loading: {path}")
    gdf = gpd.read_parquet(path)
    print(f"  Total rows: {len(gdf)}")
    print(f"  Columns ({len(gdf.columns)}): ...truncated...")
    return gdf


def has_geom(val) -> bool:
    """Check if a WKB hex string or geometry object is non-null."""
    if val is None:
        return False
    if isinstance(val, str):
        try:
            g = wkb.loads(val, hex=True)
            return g is not None and not g.is_empty
        except Exception:
            return False
    if hasattr(val, "is_empty"):
        return not val.is_empty
    try:
        return not pd.isna(val)
    except (TypeError, ValueError):
        return val is not None


def diagnose_street(gdf: gpd.GeoDataFrame, street_name: str, dataset_label: str):
    print(f"\n{'─'*80}")
    print(f"DIAGNOSING: '{street_name}' in {dataset_label}")
    print(f"{'─'*80}")

    # Case-insensitive match
    mask = gdf["name"].astype(str).str.lower() == street_name.lower()
    subset = gdf[mask]
    print(f"  Rows matching name: {len(subset)}")

    if len(subset) == 0:
        # Try partial match
        partial = gdf["name"].astype(str).str.lower().str.contains(street_name.lower(), na=False)
        partial_rows = gdf[partial]
        print(f"  Partial matches: {len(partial_rows)}")
        if len(partial_rows) > 0:
            print(f"  Unique names found: {partial_rows['name'].unique()[:10]}")
        return

    # Presence values
    for side in ("left", "right"):
        col = f"sidewalk_{side}_presence"
        if col in subset.columns:
            vc = subset[col].value_counts(dropna=False)
            print(f"\n  sidewalk_{side}_presence value_counts:")
            for val, cnt in vc.items():
                print(f"    {repr(val)}: {cnt}")

    # Geometry non-null counts
    for side in ("left", "right"):
        gcol = f"sidewalk_{side}_geometry"
        if gcol in subset.columns:
            has_g = subset[gcol].apply(has_geom).sum()
            print(f"\n  sidewalk_{side}_geometry non-null: {has_g} / {len(subset)}")

    # Buffered counts
    for side in ("left", "right"):
        bcol = f"sidewalk_{side}_buffered"
        if bcol in subset.columns:
            vc = subset[bcol].value_counts(dropna=False)
            print(f"\n  sidewalk_{side}_buffered value_counts:")
            for val, cnt in vc.items():
                print(f"    {repr(val)}: {cnt}")

    # Highway types for these rows
    if "highway" in subset.columns:
        print(f"\n  highway value_counts:")
        for val, cnt in subset["highway"].value_counts(dropna=False).items():
            print(f"    {repr(val)}: {cnt}")

    # Sample rows
    available = [c for c in COLS_OF_INTEREST if c in subset.columns]
    print(f"\n  Sample rows (first 5):")
    sample = subset[available].head(5)
    # Truncate geometry columns for readability
    for gcol in [c for c in available if "geometry" in c]:
        sample[gcol] = sample[gcol].apply(lambda v: f"<WKB:{len(str(v))}chars>" if has_geom(v) else None)
    print(sample.to_string(index=False))


def check_nearby_footways(gdf: gpd.GeoDataFrame, street_name: str, dataset_label: str):
    """Check if there are separate footway edges near this street that might
    trigger STRtree suppression in the buffering pass."""
    print(f"\n  --- STRtree suppression analysis for '{street_name}' ---")

    # Find all footway/pedestrian/path rows in the dataset
    hw_col = gdf["highway"].astype(str).str.lower()
    footway_mask = hw_col.isin(["footway", "pedestrian", "path", "steps", "corridor"])
    footways = gdf[footway_mask]
    print(f"  Total footway-class rows in dataset: {len(footways)}")

    # Find the target street rows
    street_mask = gdf["name"].astype(str).str.lower() == street_name.lower()
    street_rows = gdf[street_mask]

    if len(street_rows) == 0:
        print(f"  No rows for '{street_name}' — cannot check nearby footways.")
        return

    # Check: do any of these street rows themselves have highway=footway?
    street_hw = street_rows["highway"].value_counts(dropna=False)
    print(f"  Highway types for '{street_name}' rows: {dict(street_hw)}")

    # Now check: are these street rows classified as roads (not footways)?
    # The pipeline filters out is_separate edges from the roads GDF.
    # If Page St segments are classified as footways, they won't be in 'roads'
    # and won't get buffered sidewalks at all.
    BIKEWAY_HW = {"cycleway", "path", "bridleway"}
    FOOTWAY_HW = {"footway", "pedestrian", "path", "steps", "corridor"}

    for idx, row in street_rows.head(10).iterrows():
        hw = str(row.get("highway", "")).lower()
        is_bw = hw in BIKEWAY_HW
        is_fw = hw in FOOTWAY_HW and not is_bw
        is_sep = is_bw or is_fw
        presence_l = row.get("sidewalk_left_presence", None)
        presence_r = row.get("sidewalk_right_presence", None)
        buffered_l = row.get("sidewalk_left_buffered", None)
        buffered_r = row.get("sidewalk_right_buffered", None)
        geom_l = has_geom(row.get("sidewalk_left_geometry", None))
        geom_r = has_geom(row.get("sidewalk_right_geometry", None))
        print(f"    Row {idx}: hw={hw}, is_separate={is_sep}, "
              f"pres_L={presence_l}, pres_R={presence_r}, "
              f"buf_L={buffered_l}, buf_R={buffered_r}, "
              f"has_geom_L={geom_l}, has_geom_R={geom_r}")

    # Check the presence values that would trigger/skip buffering
    # _NEGATIVE_VALUES = {"no", "none"}
    # _SEPARATE_PRESENCE_VALUES = {"separate", "footway", "pedestrian"}
    # skip_values = _NEGATIVE_VALUES | _SEPARATE_PRESENCE_VALUES for sidewalks
    SKIP_VALUES = {"no", "none", "separate", "footway", "pedestrian"}

    for side in ("left", "right"):
        pcol = f"sidewalk_{side}_presence"
        gcol = f"sidewalk_{side}_geometry"
        if pcol not in street_rows.columns:
            continue
        presence_vals = street_rows[pcol].astype(str).str.lower()
        has_data = street_rows[pcol].notna() & ~presence_vals.isin(SKIP_VALUES)
        no_geom = street_rows[gcol].apply(lambda g: not has_geom(g)) if gcol in street_rows.columns else pd.Series(True, index=street_rows.index)
        candidates = has_data & no_geom
        print(f"\n  {side} side buffering candidates (has_data & no_geom): {candidates.sum()} / {len(street_rows)}")
        print(f"    has_data (presence not null and not in skip_values): {has_data.sum()}")
        print(f"    no_geom: {no_geom.sum()}")
        # Show what presence values are causing issues
        non_candidate_presence = street_rows.loc[~has_data & street_rows[pcol].notna(), pcol]
        if len(non_candidate_presence) > 0:
            print(f"    Presence values excluded by skip_values: {dict(non_candidate_presence.value_counts())}")


def main():
    # ── SF: Page Street ──
    if SF_PATH.exists():
        sf = load_parquet(SF_PATH)
        diagnose_street(sf, "Page Street", "SF")
        check_nearby_footways(sf, "Page Street", "SF")
    else:
        print(f"SF parquet not found at {SF_PATH}")

    # ── Alameda: Roosevelt Avenue & Dover Street ──
    if ALAMEDA_PATH.exists():
        al = load_parquet(ALAMEDA_PATH)
        diagnose_street(al, "Roosevelt Avenue", "Alameda")
        check_nearby_footways(al, "Roosevelt Avenue", "Alameda")
        diagnose_street(al, "Dover Street", "Alameda")
        check_nearby_footways(al, "Dover Street", "Alameda")
    else:
        print(f"Alameda parquet not found at {ALAMEDA_PATH}")


if __name__ == "__main__":
    main()
