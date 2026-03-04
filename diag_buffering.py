"""Diagnose why buffering candidates aren't getting buffered geometry."""
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import wkb

OUTPUT_DIR = Path("Output")

def load_parquet(name):
    path = OUTPUT_DIR / f"{name}_network.parquet"
    gdf = gpd.read_parquet(path)
    # Deserialize WKB geometry columns
    for col in gdf.columns:
        if col.endswith("_geometry") and col != gdf.geometry.name:
            gdf[col] = pd.Series([wkb.loads(bytes.fromhex(h)) if isinstance(h, str) else h for h in gdf[col]], index=gdf.index)
    return gdf

def check_street(gdf, street_name, dataset_name):
    print(f"\n{'='*80}")
    print(f"CHECKING: '{street_name}' in {dataset_name}")
    print(f"{'='*80}")

    mask = gdf["name"].astype(str).str.lower() == street_name.lower()
    rows = gdf[mask]
    print(f"  Total rows: {len(rows)}")

    # Check street_geometry existence
    has_street_geom = rows["street_geometry"].apply(lambda g: g is not None and hasattr(g, "geom_type")).sum()
    print(f"  street_geometry non-null: {has_street_geom} / {len(rows)}")

    # Check what geom types street_geometry has
    geom_types = rows["street_geometry"].apply(lambda g: g.geom_type if g is not None and hasattr(g, "geom_type") else "None")
    print(f"  street_geometry types: {dict(geom_types.value_counts())}")

    # Focus on "both" presence rows (valid buffering candidates)
    _NEGATIVE_VALUES = {"no", "none"}
    _SEPARATE_PRESENCE_VALUES = {"separate", "footway", "pedestrian"}
    skip_values = _NEGATIVE_VALUES | _SEPARATE_PRESENCE_VALUES

    for side in ("left", "right"):
        pres_col = f"sidewalk_{side}_presence"
        geom_col = f"sidewalk_{side}_geometry"
        buff_col = f"sidewalk_{side}_buffered"

        has_data = rows[pres_col].notna() & ~rows[pres_col].astype(str).str.lower().isin(skip_values)
        no_geom = rows[geom_col].apply(lambda g: g is None or not hasattr(g, "geom_type"))
        candidates = has_data & no_geom
        n_cand = candidates.sum()

        print(f"\n  --- {side.upper()} side ---")
        print(f"  Buffering candidates: {n_cand}")

        if n_cand == 0:
            continue

        # For each candidate, check what would happen in the buffering pass
        cand_rows = rows[candidates]

        # 1. How many have street_geometry?
        has_sg = cand_rows["street_geometry"].apply(lambda g: g is not None and hasattr(g, "geom_type")).sum()
        print(f"  Candidates with street_geometry: {has_sg} / {n_cand}")

        if has_sg == 0:
            print(f"  *** BUG: All candidates have NULL street_geometry! ***")
            continue

        # 2. Check STRtree suppression - count how many nearby separate sidewalks exist
        # Build a simplified version of the suppression check
        all_sep_geoms = []
        all_sep_names = []
        all_sep_sids = []
        for sw_s in ("left", "right"):
            gc = f"sidewalk_{sw_s}_geometry"
            bc = f"sidewalk_{sw_s}_buffered"
            for idx in gdf.index:
                g = gdf.at[idx, gc]
                if g is None or not hasattr(g, "geom_type"):
                    continue
                bval = gdf.at[idx, bc]
                if bval is True or str(bval).lower() in ("yes", "true"):
                    continue
                all_sep_geoms.append(g)
                all_sep_names.append(gdf.at[idx, "name"] if "name" in gdf.columns else None)
                all_sep_sids.append(gdf.at[idx, "street_id"])

        print(f"  Total separate sidewalk geometries in dataset: {len(all_sep_geoms)}")

        if len(all_sep_geoms) > 0:
            from shapely import STRtree
            import numpy as np
            sep_tree = STRtree(all_sep_geoms)

            THRESHOLD = 20.0
            n_suppressed = 0
            n_no_street_bearing = 0
            n_passed = 0

            for idx in cand_rows.index[:5]:  # Check first 5 candidates
                sg = gdf.at[idx, "street_geometry"]
                if sg is None or not hasattr(sg, "geom_type"):
                    continue

                this_sid = gdf.at[idx, "street_id"]
                this_name = gdf.at[idx, "name"]

                # Get bearing
                coords = list(sg.coords) if hasattr(sg, "coords") else []
                if len(coords) < 2:
                    n_no_street_bearing += 1
                    continue

                search_area = sg.buffer(THRESHOLD)
                hits = sep_tree.query(search_area)

                suppression_hits = []
                for hi in hits:
                    hit_sid = all_sep_sids[hi]
                    hit_name = all_sep_names[hi]

                    # Same street_id or same name → skip (don't suppress)
                    if hit_sid == this_sid:
                        continue
                    if (this_name is not None and hit_name is not None
                        and str(this_name).lower() == str(hit_name).lower()):
                        continue

                    # Check distance
                    fac_mid = all_sep_geoms[hi].interpolate(0.5, normalized=True)
                    dist = sg.distance(fac_mid)
                    if dist <= THRESHOLD:
                        suppression_hits.append({
                            "hit_name": hit_name,
                            "hit_sid": hit_sid,
                            "dist": round(dist, 1),
                        })

                if suppression_hits:
                    n_suppressed += 1
                    print(f"    Row {idx}: WOULD BE SUPPRESSED by {len(suppression_hits)} hits:")
                    for h in suppression_hits[:3]:
                        print(f"      name={h['hit_name']}, sid={h['hit_sid']}, dist={h['dist']}m")
                else:
                    n_passed += 1
                    print(f"    Row {idx}: would NOT be suppressed (hits={len(hits)}, same-street filtered)")

            print(f"  Of 5 sampled: {n_suppressed} suppressed, {n_passed} passed, {n_no_street_bearing} no bearing")


# Run checks
for dataset, streets in [
    ("San_Francisco_County_California_USA", ["Page Street"]),
    ("Alameda_County_California_USA", ["Roosevelt Avenue", "Dover Street"]),
]:
    gdf = load_parquet(dataset)
    for s in streets:
        check_street(gdf, s, dataset)
