#!/usr/bin/env python3
"""
Verify the structure and content of network parquet files.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely import wkb as shapely_wkb
from shapely import Point

# ============================================================================
# Configuration
# ============================================================================

# List of parquet files to verify
PARQUET_FILES = [
    {
        'path': "Notebooks/Karna/Proximity Model/Output/San_Francisco_County_CA_network.parquet",
        'sanity_path': "Notebooks/Karna/Proximity Model/Output/San_Francisco_County_CA_sanity.parquet",
        'name': "San Francisco County",
        'map_location': {
            'lat': 37 + 46/60 + 16.6/3600,  # 37°46'16.6"N
            'lon': -(122 + 25/60 + 27.1/3600),  # 122°25'27.1"W
            'label': 'Market Street Block'
        },
        'additional_locations': [
            {
                'lat': 37 + 42/60 + 41.0/3600,  # 37°42'41.0"N
                'lon': -(122 + 27/60 + 46.9/3600),  # 122°27'46.9"W
                'label': 'Brotherhood Alemany'
            },
            {
                'lat': 37 + 47/60 + 42.9/3600,  # 37°47'42.9"N
                'lon': -(122 + 23/60 + 38.1/3600),  # 122°23'38.1"W
                'label': 'Embarcadero'
            }
        ]
    },
    {
        'path': "Notebooks/Karna/Proximity Model/Output/Alameda_County_CA_network.parquet",
        'sanity_path': "Notebooks/Karna/Proximity Model/Output/Alameda_County_CA_sanity.parquet",
        'name': "Alameda County",
        'map_location': {
            'lat': 37 + 51/60 + 55.7/3600,  # 37°51'55.7"N
            'lon': -(122 + 16/60 + 12.3/3600),  # 122°16'12.3"W
            'label': 'Oakland/San Leandro Area'
        },
        'additional_locations': [
            {
                'lat': 37 + 50/60 + 49.7/3600,  # 37°50'49.7"N
                'lon': -(122 + 16/60 + 18.8/3600),  # 122°16'18.8"W
                'label': 'Alameda Diagnostic'
            }
        ]
    }
]

# ============================================================================
# Helper Functions
# ============================================================================

def geometry_intersects_bbox(wkb_bytes, bbox):
    """Check if WKB geometry intersects bounding box."""
    if wkb_bytes is None:
        return False
    try:
        geom = shapely_wkb.loads(bytes(wkb_bytes))
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        return not (bounds[2] < bbox['minx'] or bounds[0] > bbox['maxx'] or
                   bounds[3] < bbox['miny'] or bounds[1] > bbox['maxy'])
    except:
        return False


def verify_parquet_file(config):
    """Verify a single parquet file."""
    parquet_path = Path(config['path'])
    sanity_path = Path(config.get('sanity_path', ''))
    city_name = config['name']

    print("\n" + "=" * 80)
    print(f"VERIFYING: {city_name}")
    print("=" * 80)

    # Check network file
    if not parquet_path.exists():
        print(f"❌ Network file not found: {parquet_path}")
        return None

    print(f"✓ Found network parquet: {parquet_path}")
    print(f"  File size: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Read the dataframe
    df = pd.read_parquet(parquet_path)
    
    # Check sanity file
    if sanity_path.exists():
        print(f"✓ Found sanity parquet: {sanity_path}")
        print(f"  File size: {sanity_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Verify sanity buffer structure
        sanity_df = pd.read_parquet(sanity_path)
        print(f"\nSANITY BUFFER VERIFICATION")
        print("-" * 80)
        print(f"  Rows: {len(sanity_df)}")
        print(f"  Columns: {list(sanity_df.columns)}")
        
        if 'osmid' in sanity_df.columns and 'max_offset_width' in sanity_df.columns:
            print(f"  ✓ Required columns present")
            print(f"  Max offset range: {sanity_df['max_offset_width'].min():.2f}m - {sanity_df['max_offset_width'].max():.2f}m")
            print(f"  Mean offset: {sanity_df['max_offset_width'].mean():.2f}m")
        else:
            print(f"  ❌ Missing required columns")
    else:
        print(f"⚠ Sanity file not found: {sanity_path}")
    
    print()
    
    # Read parquet metadata
    parquet_file = pq.ParquetFile(parquet_path)
    arrow_schema = parquet_file.schema_arrow
    
    # Check for expected columns
    print("\nEXPECTED COLUMNS CHECK (ProximityModel.py)")
    print("-" * 80)
    
    expected_columns = [
        # Street Centerline
        'block_ids', 'street_id', 'block_sides', 'public_data_id_street',
        'start_node_id', 'start_node_is_block_node', 'start_node_is_intersection_node',
        'end_node_id', 'end_node_is_block_node', 'end_node_is_intersection_node',
        'public_data_id_start_end_nodes',
        'normalized_bearing', 'name', 'highway', 'maxspeed', 'oneway', 'lanes', 'lane_width', 'surface',

        # Street Centerline Features
        'street_feature_types', 'public_data_id_street_feature',
        'street_feature_geometry', 'street_feature_geometry_projected',

        # Sidewalk Centerlines (Left)
        'sidewalk_left_ID', 'sidewalk_left_block_ID', 'sidewalk_left_presence',
        'public_data_id_sidewalk_left', 'sidewalk_left_surface', 'sidewalk_left_quality',
        'sidewalk_left_width', 'sidewalk_left_incline', 'sidewalk_left_buffered',

        # Curb Ramp (Left, Start, 1-3)
        'sidewalk_left_curbramp_start_1_ID', 'public_data_id_sidewalk_left_curbramp_start_1',
        'sidewalk_left_curbramp_start_1_returnloc', 'sidewalk_left_curbramp_start_1_returnposition',
        'sidewalk_left_curbramp_start_1_condition_score', 'sidewalk_left_curbramp_start_1_geometry',
        'sidewalk_left_curbramp_start_2_ID', 'public_data_id_sidewalk_left_curbramp_start_2',
        'sidewalk_left_curbramp_start_2_returnloc', 'sidewalk_left_curbramp_start_2_returnposition',
        'sidewalk_left_curbramp_start_2_condition_score', 'sidewalk_left_curbramp_start_2_geometry',
        'sidewalk_left_curbramp_start_3_ID', 'public_data_id_sidewalk_left_curbramp_start_3',
        'sidewalk_left_curbramp_start_3_returnloc', 'sidewalk_left_curbramp_start_3_returnposition',
        'sidewalk_left_curbramp_start_3_condition_score', 'sidewalk_left_curbramp_start_3_geometry',

        # Curb Ramp (Left, End, 1-3)
        'sidewalk_left_curbramp_end_1_ID', 'public_data_id_sidewalk_left_curbramp_end_1',
        'sidewalk_left_curbramp_end_1_returnloc', 'sidewalk_left_curbramp_end_1_returnposition',
        'sidewalk_left_curbramp_end_1_condition_score', 'sidewalk_left_curbramp_end_1_geometry',
        'sidewalk_left_curbramp_end_2_ID', 'public_data_id_sidewalk_left_curbramp_end_2',
        'sidewalk_left_curbramp_end_2_returnloc', 'sidewalk_left_curbramp_end_2_returnposition',
        'sidewalk_left_curbramp_end_2_condition_score', 'sidewalk_left_curbramp_end_2_geometry',
        'sidewalk_left_curbramp_end_3_ID', 'public_data_id_sidewalk_left_curbramp_end_3',
        'sidewalk_left_curbramp_end_3_returnloc', 'sidewalk_left_curbramp_end_3_returnposition',
        'sidewalk_left_curbramp_end_3_condition_score', 'sidewalk_left_curbramp_end_3_geometry',

        # Sidewalk Centerline Features (Left)
        'sidewalk_left_feature_ids', 'sidewalk_left_feature_types',
        'public_data_id_sidewalk_left_feature', 'sidewalk_left_feature_geometry',
        'sidewalk_left_feature_geometry_projected',

        # Sidewalk Centerlines (Right)
        'sidewalk_right_ID', 'sidewalk_right_block_ID', 'sidewalk_right_presence',
        'public_data_id_sidewalk_right', 'sidewalk_right_surface', 'sidewalk_right_quality',
        'sidewalk_right_width', 'sidewalk_right_incline', 'sidewalk_right_buffered',

        # Curb Ramp (Right, Start, 1-3)
        'sidewalk_right_curbramp_start_1_ID', 'public_data_id_sidewalk_right_curbramp_start_1',
        'sidewalk_right_curbramp_start_1_returnloc', 'sidewalk_right_curbramp_start_1_returnposition',
        'sidewalk_right_curbramp_start_1_condition_score', 'sidewalk_right_curbramp_start_1_geometry',
        'sidewalk_right_curbramp_start_2_ID', 'public_data_id_sidewalk_right_curbramp_start_2',
        'sidewalk_right_curbramp_start_2_returnloc', 'sidewalk_right_curbramp_start_2_returnposition',
        'sidewalk_right_curbramp_start_2_condition_score', 'sidewalk_right_curbramp_start_2_geometry',
        'sidewalk_right_curbramp_start_3_ID', 'public_data_id_sidewalk_right_curbramp_start_3',
        'sidewalk_right_curbramp_start_3_returnloc', 'sidewalk_right_curbramp_start_3_returnposition',
        'sidewalk_right_curbramp_start_3_condition_score', 'sidewalk_right_curbramp_start_3_geometry',

        # Curb Ramp (Right, End, 1-3)
        'sidewalk_right_curbramp_end_1_ID', 'public_data_id_sidewalk_right_curbramp_end_1',
        'sidewalk_right_curbramp_end_1_returnloc', 'sidewalk_right_curbramp_end_1_returnposition',
        'sidewalk_right_curbramp_end_1_condition_score', 'sidewalk_right_curbramp_end_1_geometry',
        'sidewalk_right_curbramp_end_2_ID', 'public_data_id_sidewalk_right_curbramp_end_2',
        'sidewalk_right_curbramp_end_2_returnloc', 'sidewalk_right_curbramp_end_2_returnposition',
        'sidewalk_right_curbramp_end_2_condition_score', 'sidewalk_right_curbramp_end_2_geometry',
        'sidewalk_right_curbramp_end_3_ID', 'public_data_id_sidewalk_right_curbramp_end_3',
        'sidewalk_right_curbramp_end_3_returnloc', 'sidewalk_right_curbramp_end_3_returnposition',
        'sidewalk_right_curbramp_end_3_condition_score', 'sidewalk_right_curbramp_end_3_geometry',

        # Sidewalk Centerline Features (Right)
        'sidewalk_right_feature_ids', 'sidewalk_right_feature_types',
        'public_data_id_sidewalk_right_feature', 'sidewalk_right_feature_geometry',
        'sidewalk_right_feature_geometry_projected',

        # Crosswalk (Start)
        'crosswalk_start_id', 'crosswalk_start_block_ids', 'crosswalk_start_type',
        'public_data_id_crosswalk_start', 'crosswalk_start_controlled', 'crosswalk_start_marked',
        'crosswalk_start_markings', 'crosswalk_start_signals', 'crosswalk_start_island',
        'crosswalk_start_kerb', 'crosswalk_start_tactile_paving', 'crosswalk_start_traffic_calming',
        'crosswalk_start_continuous', 'crosswalk_start_condition', 'crosswalk_start_geometry',
        'crosswalk_start_island_geometry',

        # Crosswalk (End)
        'crosswalk_end_id', 'crosswalk_end_block_ids', 'crosswalk_end_type',
        'public_data_id_crosswalk_end', 'crosswalk_end_controlled', 'crosswalk_end_marked',
        'crosswalk_end_markings', 'crosswalk_end_signals', 'crosswalk_end_island',
        'crosswalk_end_kerb', 'crosswalk_end_tactile_paving', 'crosswalk_end_traffic_calming',
        'crosswalk_end_continuous', 'crosswalk_end_condition', 'crosswalk_end_geometry',
        'crosswalk_end_island_geometry',

        # Bikeway Centerline (Left, 1)
        'bikeway_left_1_id', 'bikeway_left_1_block_id', 'public_data_id_bikeway_left_1',
        'bikeway_left_1_type', 'bikeway_left_1_surface', 'bikeway_left_1_quality',
        'bikeway_left_1_permitted', 'bikeway_left_1_width', 'bikeway_left_1_incline',
        'bikeway_left_buffered',

        # Bikeway Centerline (Left, 2)
        'bikeway_left_2_id', 'public_data_id_bikeway_left_2', 'bikeway_left_2_type',
        'bikeway_left_2_surface', 'bikeway_left_2_quality', 'bikeway_left_2_permitted',
        'bikeway_left_2_width', 'bikeway_left_2_incline',

        # Bikeway Centerline Features (Left, 1)
        'bikeway_left_1_feature_ids', 'bikeway_left_1_feature_types',
        'public_data_id_bikeway_left_1_features', 'bikeway_left_1_feature_geometry',
        'bikeway_left_1_feature_geometry_projected',

        # Bikeway Centerline Features (Left, 2)
        'bikeway_left_2_feature_types', 'public_data_id_bikeway_left_2_features',
        'bikeway_left_2_feature_geometry', 'bikeway_left_2_feature_geometry_projected',

        # Bikeway Centerline (Right, 1)
        'bikeway_right_1_id', 'bikeway_right_1_block_id', 'public_data_id_bikeway_right_1',
        'bikeway_right_1_type', 'bikeway_right_1_surface', 'bikeway_right_1_quality',
        'bikeway_right_1_permitted', 'bikeway_right_1_width', 'bikeway_right_1_incline',
        'bikeway_right_buffered',

        # Bikeway Centerline (Right, 2)
        'bikeway_right_2_id', 'public_data_id_bikeway_right_2', 'bikeway_right_2_type',
        'bikeway_right_2_surface', 'bikeway_right_2_quality', 'bikeway_right_2_permitted',
        'bikeway_right_2_width', 'bikeway_right_2_incline',

        # Bikeway Centerline Features (Right, 1)
        'bikeway_right_1_feature_ids', 'bikeway_right_1_feature_types',
        'public_data_id_bikeway_right_1_features', 'bikeway_right_1_feature_geometry',
        'bikeway_right_1_feature_geometry_projected',

        # Bikeway Centerline Features (Right, 2)
        'bikeway_right_2_feature_types', 'public_data_id_bikeway_right_2_features',
        'bikeway_right_2_feature_geometry', 'bikeway_right_2_feature_geometry_projected',

        # Main Geometry Columns
        'street_geometry', 'start_node_geometry', 'end_node_geometry',
        'sidewalk_left_geometry', 'sidewalk_right_geometry', 'curb_return_geometry',
        'bikeway_left_1_geometry', 'bikeway_left_2_geometry',
        'bikeway_right_1_geometry', 'bikeway_right_2_geometry',
    ]
    
    missing_cols = []
    for col in expected_columns:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (MISSING)")
            missing_cols.append(col)
    
    if len(missing_cols) == 0:
        print(f"\n✓ All {len(expected_columns)} expected columns present")
    else:
        print(f"\n⚠ {len(missing_cols)} columns missing: {missing_cols}")
    
    # Verify geometry columns are WKB bytes
    print("\nGEOMETRY COLUMN VERIFICATION")
    print("-" * 80)
    geom_cols = ['street_geometry', 'start_node_geometry', 'end_node_geometry',
                 'sidewalk_left_geometry', 'sidewalk_right_geometry',
                 'bikeway_left_1_geometry', 'bikeway_left_2_geometry',
                 'bikeway_right_1_geometry', 'bikeway_right_2_geometry',
                 'curb_return_geometry']
    
    for col in geom_cols:
        if col in df.columns:
            sample = df[col].dropna()
            if len(sample) > 0:
                try:
                    geom = shapely_wkb.loads(bytes(sample.iloc[0]))
                    print(f"  ✓ {col}: Valid WKB ({geom.geom_type})")
                except Exception as e:
                    print(f"  ✗ {col}: Invalid WKB - {e}")
            else:
                print(f"  ⚠ {col}: No data")
        else:
            print(f"  ✗ {col}: Column missing")
    
    # Verify sequential IDs
    print("\nSEQUENTIAL ID VERIFICATION")
    print("-" * 80)
    id_cols = ['sidewalk_left_ID', 'sidewalk_right_ID', 
               'bikeway_left_1_id', 'bikeway_left_2_id',
               'bikeway_right_1_id', 'bikeway_right_2_id',
               'crosswalk_start_id', 'crosswalk_end_id']
    
    for col in id_cols:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                # Check if values are numeric
                try:
                    numeric_valid = pd.to_numeric(valid, errors='coerce')
                    non_numeric = valid[numeric_valid.isna()]
                    if len(non_numeric) > 0:
                        print(f"  ⚠ {col}: Contains non-numeric values (e.g., {non_numeric.iloc[0]})")
                        numeric_valid = numeric_valid.dropna()
                    
                    if len(numeric_valid) > 0 and (numeric_valid > 0).all():
                        print(f"  ✓ {col}: All positive integers (min={numeric_valid.min()}, max={numeric_valid.max()})")
                    elif len(numeric_valid) > 0:
                        print(f"  ✗ {col}: Contains non-positive values")
                    else:
                        print(f"  ⚠ {col}: No valid numeric data")
                except Exception as e:
                    print(f"  ✗ {col}: Error checking values - {str(e)}")
            else:
                print(f"  ⚠ {col}: No data")
        else:
            print(f"  ✗ {col}: Column missing")

    return df



def find_feature_examples(df, city_name):
    """Find examples of different feature types for diagnostic maps."""
    examples = {}
    
    print(f"\nSearching for feature examples in {city_name}...")
    print("-" * 80)
    
    # 1. Find block assignment example
    blocks_assigned = df[df['block_ids'].notna()]
    if len(blocks_assigned) > 0:
        row = blocks_assigned.iloc[0]
        geom = shapely_wkb.loads(bytes(row['street_geometry']))
        centroid = geom.centroid
        examples['block_assignment'] = {
            'lat': centroid.y,
            'lon': centroid.x,
            'label': f'Block Assignment',
            'row_idx': blocks_assigned.index[0]
        }
        print(f"  ✓ Found block assignment at row {blocks_assigned.index[0]}")
        print(f"    Block IDs: {row['block_ids']}")
    else:
        print(f"  ✗ No block assignment found")
    
    # 2. Find buffered bikeway (bikeway_left_buffered or bikeway_right_buffered)
    buffered_bikeway = df[(df['bikeway_left_buffered'] == True) | (df['bikeway_right_buffered'] == True)]
    if len(buffered_bikeway) > 0:
        # Find one with actual geometry data
        for idx in buffered_bikeway.index[:100]:  # Check first 100
            row = buffered_bikeway.loc[idx]
            if row['bikeway_left_1_geometry'] is not None or row['bikeway_right_1_geometry'] is not None:
                geom = shapely_wkb.loads(bytes(row['street_geometry']))
                centroid = geom.centroid
                examples['buffered_bikeway'] = {
                    'lat': centroid.y,
                    'lon': centroid.x,
                    'label': f'Buffered Bikeway',
                    'row_idx': idx
                }
                print(f"  ✓ Found buffered bikeway at row {idx}")
                print(f"    Left buffered: {row['bikeway_left_buffered']}, Right buffered: {row['bikeway_right_buffered']}")
                break
        else:
            print(f"  ✗ No buffered bikeway with valid geometry found")
    else:
        print(f"  ✗ No buffered bikeway found")
    
    # 3. Find buffered sidewalk (sidewalk_left_buffered or sidewalk_right_buffered)
    buffered_sidewalk = df[(df['sidewalk_left_buffered'] == True) | (df['sidewalk_right_buffered'] == True)]
    if len(buffered_sidewalk) > 0:
        # Find one with actual geometry data
        for idx in buffered_sidewalk.index[:100]:  # Check first 100
            row = buffered_sidewalk.loc[idx]
            if row['sidewalk_left_geometry'] is not None or row['sidewalk_right_geometry'] is not None:
                geom = shapely_wkb.loads(bytes(row['street_geometry']))
                centroid = geom.centroid
                examples['buffered_sidewalk'] = {
                    'lat': centroid.y,
                    'lon': centroid.x,
                    'label': f'Buffered Sidewalk',
                    'row_idx': idx
                }
                print(f"  ✓ Found buffered sidewalk at row {idx}")
                print(f"    Left buffered: {row['sidewalk_left_buffered']}, Right buffered: {row['sidewalk_right_buffered']}")
                break
        else:
            print(f"  ✗ No buffered sidewalk with valid geometry found")
    else:
        print(f"  ✗ No buffered sidewalk found")
    
    # 4. Find default curb ramp slot assignment (curb ramps without public_data_id)
    default_curbramp = None
    for idx, row in df.iterrows():
        for side in ['left', 'right']:
            for slot in ['start', 'end']:
                ramp_id = row.get(f'sidewalk_{side}_curbramp_{slot}_1_ID')
                public_id = row.get(f'public_data_id_sidewalk_{side}_curbramp_{slot}_1')
                ramp_geom = row.get(f'sidewalk_{side}_curbramp_{slot}_1_geometry')
                
                # Default curb ramp: has ID and geometry but no public_data_id
                if ramp_id is not None and ramp_geom is not None and (public_id is None or pd.isna(public_id)):
                    geom = shapely_wkb.loads(bytes(row['street_geometry']))
                    centroid = geom.centroid
                    examples['default_curbramp'] = {
                        'lat': centroid.y,
                        'lon': centroid.x,
                        'label': f'Default Curb Ramp Slot',
                        'row_idx': idx
                    }
                    print(f"  ✓ Found default curb ramp at row {idx}")
                    print(f"    Side: {side}, Slot: {slot}, ID: {ramp_id}")
                    default_curbramp = True
                    break
            if default_curbramp:
                break
        if default_curbramp:
            break
    
    if not default_curbramp:
        print(f"  ✗ No default curb ramp slot found")
    
    # 5. Find government curb ramp (has public_data_id)
    gov_curbramp = None
    for idx, row in df.iterrows():
        for side in ['left', 'right']:
            for slot in ['start', 'end']:
                public_id = row.get(f'public_data_id_sidewalk_{side}_curbramp_{slot}_1')
                ramp_geom = row.get(f'sidewalk_{side}_curbramp_{slot}_1_geometry')
                
                # Government curb ramp: has public_data_id
                if public_id is not None and not pd.isna(public_id) and ramp_geom is not None:
                    geom = shapely_wkb.loads(bytes(row['street_geometry']))
                    centroid = geom.centroid
                    examples['government_curbramp'] = {
                        'lat': centroid.y,
                        'lon': centroid.x,
                        'label': f'Government Curb Ramp',
                        'row_idx': idx
                    }
                    print(f"  ✓ Found government curb ramp at row {idx}")
                    print(f"    Side: {side}, Slot: {slot}, Public ID: {public_id}")
                    gov_curbramp = True
                    break
            if gov_curbramp:
                break
        if gov_curbramp:
            break
    
    if not gov_curbramp:
        print(f"  ✗ No government curb ramp found")
    
    print()
    return examples


def generate_diagnostic_map(df, city_name, feature_type, feature_info, output_dir):
    """Generate a diagnostic map showing a specific feature type."""
    print(f"\n  Generating {feature_type} map...")
    
    target_lat = feature_info['lat']
    target_lon = feature_info['lon']
    label = feature_info['label']
    
    # Use larger bounding box for block_assignment to show multiple blocks
    if feature_type == 'block_assignment':
        buffer_deg = 0.0045  # ~500 meters for block visualization
    else:
        buffer_deg = 0.001  # ~100 meters for other features
    
    bbox = {
        'minx': target_lon - buffer_deg,
        'maxx': target_lon + buffer_deg,
        'miny': target_lat - buffer_deg,
        'maxy': target_lat + buffer_deg
    }
    
    # Filter streets within bounding box
    df['in_bbox'] = df['street_geometry'].apply(lambda x: geometry_intersects_bbox(x, bbox))
    filtered_df = df[df['in_bbox']].copy()
    
    if len(filtered_df) == 0:
        print(f"    ⚠ No streets found in bounding box")
        return
    
    print(f"    Found {len(filtered_df)} streets in bounding box")
    
    # For block_assignment, prepare color mapping
    block_colors = {}
    block_segments = {}
    if feature_type == 'block_assignment':
        # Extract all unique block IDs
        all_block_ids = set()
        for idx, row in filtered_df.iterrows():
            if row['block_ids'] is not None:
                if isinstance(row['block_ids'], (list, tuple)):
                    for block_id in row['block_ids']:
                        if block_id is not None and block_id > 0:
                            all_block_ids.add(block_id)
        
        # Build adjacency graph for coloring
        block_adjacency = {block_id: set() for block_id in all_block_ids}
        for idx, row in filtered_df.iterrows():
            if row['block_ids'] is not None and isinstance(row['block_ids'], (list, tuple)) and len(row['block_ids']) >= 2:
                left_block, right_block = row['block_ids'][0], row['block_ids'][1]
                if left_block and right_block and left_block > 0 and right_block > 0:
                    block_adjacency[left_block].add(right_block)
                    block_adjacency[right_block].add(left_block)
        
        # Greedy coloring
        available_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
        for block_id in sorted(all_block_ids):
            used_colors = {block_colors.get(adj_block) for adj_block in block_adjacency[block_id] 
                          if adj_block in block_colors}
            for i, color in enumerate(available_colors):
                if i not in used_colors:
                    block_colors[block_id] = (i, color)
                    block_segments[block_id] = []
                    break
    
    # Create map
    fig, ax = plt.subplots(figsize=(20, 20) if feature_type == 'block_assignment' else (16, 16))
    
    # Track what features are actually present for dynamic legend
    features_present = {
        'carway': False,
        'footway': False,
        'cycleway': False,
        'sidewalk_buffered': False,
        'sidewalk_separate': False,
        'bikeway_buffered': False,
        'bikeway_separate': False,
        'curb_ramp_empty': False,
        'curb_ramp_populated': False,
        'curb_return': False
    }
    
    # Plot each street centerline
    for idx, row in filtered_df.iterrows():
        try:
            geom = shapely_wkb.loads(bytes(row['street_geometry']))
            
            # Special rendering for block_assignment - color by block
            if feature_type == 'block_assignment' and row['block_ids'] is not None:
                if geom.geom_type == 'LineString':
                    x, y = geom.xy
                    
                    if isinstance(row['block_ids'], (list, tuple)) and len(row['block_ids']) >= 2:
                        left_block, right_block = row['block_ids'][0], row['block_ids'][1]
                        
                        # Draw with block colors
                        if left_block and left_block > 0 and left_block in block_colors:
                            _, color = block_colors[left_block]
                            ax.plot(x, y, color=color, linewidth=4, alpha=0.8, zorder=2, solid_capstyle='round')
                            block_segments[left_block].append(geom)
                        
                        if right_block and right_block > 0 and right_block in block_colors:
                            _, color = block_colors[right_block]
                            ax.plot(x, y, color=color, linewidth=4, alpha=0.6, zorder=1, solid_capstyle='round')
                            block_segments[right_block].append(geom)
                    else:
                        ax.plot(x, y, color='gray', linewidth=2, alpha=0.3, zorder=0)
            else:
                # Standard rendering for other feature types
                highway = row['highway'][0] if row['highway'] is not None and len(row['highway']) > 0 else 'unknown'
                
                # Categorize
                carway_types = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 
                               'residential', 'service', 'unclassified', 'road', 'living_street',
                               'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
                footway_types = {'footway', 'pedestrian', 'steps', 'path', 'bridleway'}
                cycleway_types = {'cycleway'}
                
                if highway in carway_types:
                    color = 'gray'
                    linewidth = 2.5
                    alpha = 0.8
                    features_present['carway'] = True
                elif highway in footway_types:
                    color = 'blue'
                    linewidth = 2
                    alpha = 0.9
                    features_present['footway'] = True
                elif highway in cycleway_types:
                    color = 'green'
                    linewidth = 2
                    alpha = 0.9
                    features_present['cycleway'] = True
                else:
                    color = 'black'
                    linewidth = 1.5
                    alpha = 0.6
                
                # Plot centerline
                if geom.geom_type == 'LineString':
                    x, y = geom.xy
                    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=2)
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        x, y = line.xy
                        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=2)
            
            # Plot sidewalk geometries (buffered - if present)
            for side in ['left', 'right']:
                sidewalk_geom = row[f'sidewalk_{side}_geometry']
                if sidewalk_geom is not None:
                    try:
                        sw_geom = shapely_wkb.loads(bytes(sidewalk_geom))
                        # Highlight buffered sidewalks
                        is_buffered = row.get(f'sidewalk_{side}_buffered', False)
                        sw_color = 'cyan' if is_buffered else 'lightblue'
                        sw_width = 2.5 if is_buffered else 1.5
                        sw_style = '--' if is_buffered else ':'
                        
                        if is_buffered:
                            features_present['sidewalk_buffered'] = True
                        else:
                            features_present['sidewalk_separate'] = True
                        
                        if sw_geom.geom_type == 'LineString':
                            x, y = sw_geom.xy
                            ax.plot(x, y, color=sw_color, linewidth=sw_width, alpha=0.9, 
                                   linestyle=sw_style, zorder=3)
                        elif sw_geom.geom_type == 'MultiLineString':
                            for line in sw_geom.geoms:
                                x, y = line.xy
                                ax.plot(x, y, color=sw_color, linewidth=sw_width, alpha=0.9, 
                                       linestyle=sw_style, zorder=3)
                    except:
                        pass
            
            # Plot bikeway geometries (buffered - if present)
            for side in ['left', 'right']:
                for num in [1, 2]:
                    bikeway_geom = row[f'bikeway_{side}_{num}_geometry']
                    if bikeway_geom is not None:
                        try:
                            bw_geom = shapely_wkb.loads(bytes(bikeway_geom))
                            # Highlight buffered bikeways
                            is_buffered = row.get(f'bikeway_{side}_buffered', False)
                            bw_color = 'lime' if is_buffered else 'lightgreen'
                            bw_width = 2.5 if is_buffered else 1.5
                            bw_style = ':' if is_buffered else '-.'
                            
                            if is_buffered:
                                features_present['bikeway_buffered'] = True
                            else:
                                features_present['bikeway_separate'] = True
                            
                            if bw_geom.geom_type == 'LineString':
                                x, y = bw_geom.xy
                                ax.plot(x, y, color=bw_color, linewidth=bw_width, alpha=0.9, 
                                       linestyle=bw_style, zorder=3)
                            elif bw_geom.geom_type == 'MultiLineString':
                                for line in bw_geom.geoms:
                                    x, y = line.xy
                                    ax.plot(x, y, color=bw_color, linewidth=bw_width, alpha=0.9, 
                                           linestyle=bw_style, zorder=3)
                        except:
                            pass
            
            # Plot curb ramp slots (all slots shown, populated ones in blue)
            for side in ['left', 'right']:
                for slot in ['start', 'end']:
                    # Get the endpoint of the street for this slot
                    if slot == 'start':
                        slot_point = Point(geom.coords[0])
                    else:  # end
                        slot_point = Point(geom.coords[-1])
                    
                    # Offset the slot point slightly to the left or right of the centerline
                    # to show which side the slot is on
                    offset_distance = 0.00002  # ~2 meters in degrees
                    if len(geom.coords) >= 2:
                        if slot == 'start':
                            dx = geom.coords[1][0] - geom.coords[0][0]
                            dy = geom.coords[1][1] - geom.coords[0][1]
                        else:
                            dx = geom.coords[-1][0] - geom.coords[-2][0]
                            dy = geom.coords[-1][1] - geom.coords[-2][1]
                        
                        # Perpendicular vector (rotated 90 degrees)
                        length = (dx**2 + dy**2)**0.5
                        if length > 0:
                            if side == 'left':
                                perp_x = -dy / length * offset_distance
                                perp_y = dx / length * offset_distance
                            else:  # right
                                perp_x = dy / length * offset_distance
                                perp_y = -dx / length * offset_distance
                            
                            slot_point = Point(slot_point.x + perp_x, slot_point.y + perp_y)
                    
                    # Check all 3 positions for this slot
                    for position in [1, 2, 3]:
                        ramp_geom = row.get(f'sidewalk_{side}_curbramp_{slot}_{position}_geometry')
                        ramp_id = row.get(f'sidewalk_{side}_curbramp_{slot}_{position}_ID')
                        public_id = row.get(f'public_data_id_sidewalk_{side}_curbramp_{slot}_{position}')
                        
                        # Check if this slot is populated
                        is_populated = ramp_geom is not None
                        
                        if is_populated:
                            features_present['curb_ramp_populated'] = True
                            try:
                                ramp_point = shapely_wkb.loads(bytes(ramp_geom))
                                
                                # Blue dot for populated slots
                                marker_color = 'blue'
                                edge_color = 'darkblue'
                                marker_size = 10
                                alpha_val = 0.9
                                
                                # Determine label based on data source
                                if public_id is not None and not pd.isna(public_id):
                                    # Government data
                                    label_text = f'G{position}'
                                else:
                                    # Default/inferred data
                                    label_text = f'D{position}'
                                
                                # Plot the populated slot
                                ax.plot(ramp_point.x, ramp_point.y, 'o', color=marker_color, 
                                       markersize=marker_size, alpha=alpha_val, zorder=5,
                                       markeredgecolor=edge_color, markeredgewidth=1.5)
                                
                                # Add text label
                                ax.text(ramp_point.x, ramp_point.y, f' {label_text}', 
                                       fontsize=7, color=edge_color, weight='bold',
                                       ha='left', va='bottom', zorder=6,
                                       bbox=dict(boxstyle='round,pad=0.2', 
                                                facecolor='white', alpha=0.7, 
                                                edgecolor='none'))
                            except:
                                pass
                        else:
                            # Empty slot - show as red dot at the slot location
                            # Only show position 1 for empty slots to avoid clutter
                            if position == 1:
                                features_present['curb_ramp_empty'] = True
                                ax.plot(slot_point.x, slot_point.y, 'o', color='red', 
                                       markersize=10, alpha=0.7, zorder=4,
                                       markeredgecolor='darkred', markeredgewidth=1.5)
            
            # Plot curb return geometry (if present)
            curb_return_geom = row.get('curb_return_geometry')
            if curb_return_geom is not None:
                try:
                    curb_return = shapely_wkb.loads(bytes(curb_return_geom))
                    features_present['curb_return'] = True
                    if curb_return.geom_type == 'LineString':
                        x, y = curb_return.xy
                        ax.plot(x, y, color='orange', linewidth=3, alpha=0.8, 
                               linestyle='--', zorder=4, label='_nolegend_')
                except:
                    pass
        except Exception as e:
            pass
    
    # Add block ID labels for block_assignment feature
    if feature_type == 'block_assignment' and block_segments:
        from shapely.geometry import MultiPoint
        for block_id, segments in block_segments.items():
            if segments:
                # Calculate centroid of all segments for this block
                all_coords = []
                for seg in segments:
                    if seg.geom_type == 'LineString':
                        all_coords.extend(list(seg.coords))
                
                if all_coords:
                    centroid = MultiPoint(all_coords).centroid
                    _, color = block_colors[block_id]
                    
                    # Draw label with contrasting background
                    ax.text(centroid.x, centroid.y, str(block_id), 
                           fontsize=14, color='white', weight='bold',
                           ha='center', va='center', zorder=10,
                           bbox=dict(boxstyle='round,pad=0.6', 
                                    facecolor=color, alpha=0.95, 
                                    edgecolor='black', linewidth=2.5))
    
    # Plot target point
    ax.plot(target_lon, target_lat, 'r*', markersize=30, label='Feature Location', zorder=10,
           markeredgecolor='darkred', markeredgewidth=2)
    
    # Set bounds
    ax.set_xlim(bbox['minx'], bbox['maxx'])
    ax.set_ylim(bbox['miny'], bbox['maxy'])
    
    # Labels and formatting
    ax.set_xlabel('Longitude', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=13, fontweight='bold')
    ax.set_title(f'{label} - {city_name}\n{target_lat:.6f}°N, {abs(target_lon):.6f}°W', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal')
    
    # Legend - only show items that are actually present
    from matplotlib.lines import Line2D
    legend_elements = []
    
    if features_present['carway']:
        legend_elements.append(mpatches.Patch(color='gray', label='Carway'))
    if features_present['footway']:
        legend_elements.append(mpatches.Patch(color='blue', label='Footway (separate)'))
    if features_present['sidewalk_buffered']:
        legend_elements.append(Line2D([0], [0], color='cyan', linewidth=2.5, linestyle='--', label='Sidewalk (buffered)'))
    if features_present['sidewalk_separate']:
        legend_elements.append(Line2D([0], [0], color='lightblue', linewidth=1.5, linestyle=':', label='Sidewalk (separate)'))
    if features_present['cycleway']:
        legend_elements.append(mpatches.Patch(color='green', label='Cycleway (separate)'))
    if features_present['bikeway_buffered']:
        legend_elements.append(Line2D([0], [0], color='lime', linewidth=2.5, linestyle=':', label='Bikeway (buffered)'))
    if features_present['bikeway_separate']:
        legend_elements.append(Line2D([0], [0], color='lightgreen', linewidth=1.5, linestyle='-.', label='Bikeway (separate)'))
    if features_present['curb_ramp_empty']:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markeredgecolor='darkred', markersize=10, label='Curb Ramp Slot (Empty)'))
    if features_present['curb_ramp_populated']:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markeredgecolor='darkblue', markersize=10, label='Curb Ramp Slot (Populated)'))
    if features_present['curb_return']:
        legend_elements.append(Line2D([0], [0], color='orange', linewidth=3, linestyle='--', label='Curb Return'))
    
    # Always show the target marker
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                                  markeredgecolor='darkred', markersize=15, label='Feature Location'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    # Save map
    output_dir.mkdir(parents=True, exist_ok=True)
    map_filename = f"{city_name.lower().replace(' ', '_')}_{feature_type}.png"
    output_path = output_dir / map_filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"    ✓ Map saved to: {output_path}")
    print(f"      Image size: {output_path.stat().st_size / 1024:.1f} KB")
    
    plt.close()


def generate_all_diagnostic_maps(df, city_name):
    """Generate all diagnostic maps for a city."""
    print("\n" + "=" * 80)
    print(f"GENERATING DIAGNOSTIC MAPS - {city_name}")
    print("=" * 80)
    
    # Find feature examples
    examples = find_feature_examples(df, city_name)
    
    if len(examples) == 0:
        print(f"  ⚠ No feature examples found for {city_name}")
        return
    
    # Output directory
    output_dir = Path("Notebooks/Karna/Proximity Model/Output/Diagnostic Maps")
    
    # Generate maps for each feature type
    for feature_type, feature_info in examples.items():
        generate_diagnostic_map(df, city_name, feature_type, feature_info, output_dir)


def generate_map(df, config):
    """Generate a map visualization for a specific location."""
    city_name = config['name']
    map_config = config['map_location']
    
    print("\n" + "=" * 80)
    print(f"MAP VISUALIZATION - {map_config['label']}")
    print("=" * 80)
    
    target_lat = map_config['lat']
    target_lon = map_config['lon']
    
    print(f"\nTarget location: {target_lat:.6f}°N, {target_lon:.6f}°W")
    
    # Define block-level bounding box (approximately 150m x 150m)
    buffer_deg = 0.0015  # ~150 meters
    bbox = {
        'minx': target_lon - buffer_deg,
        'maxx': target_lon + buffer_deg,
        'miny': target_lat - buffer_deg,
        'maxy': target_lat + buffer_deg
    }
    
    print(f"Bounding box: {bbox}")
    
    # Filter streets within bounding box
    print("\nFiltering streets within bounding box...")
    
    df['in_bbox'] = df['street_geometry'].apply(lambda x: geometry_intersects_bbox(x, bbox))
    filtered_df = df[df['in_bbox']].copy()
    
    print(f"Found {len(filtered_df)} streets in bounding box")
    
    if len(filtered_df) == 0:
        print("⚠ No streets found in bounding box. Map cannot be generated.")
        return
    
    # Create map
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot each street
    for idx, row in filtered_df.iterrows():
        try:
            geom = shapely_wkb.loads(bytes(row['street_geometry']))
            
            # Get street name and highway type for labeling
            name = row['name'][0] if row['name'] is not None and len(row['name']) > 0 else 'Unnamed'
            highway = row['highway'][0] if row['highway'] is not None and len(row['highway']) > 0 else 'unknown'
            
            # Categorize highway types into simplified groups
            carway_types = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 
                           'residential', 'service', 'unclassified', 'road', 'living_street',
                           'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
            footway_types = {'footway', 'pedestrian', 'steps', 'path', 'bridleway'}
            cycleway_types = {'cycleway'}
            
            # Determine category and color
            if highway in carway_types:
                category = 'carway'
                color = 'gray'
                linewidth = 2
            elif highway in footway_types:
                category = 'footway'
                color = 'blue'
                linewidth = 1.5
            elif highway in cycleway_types:
                category = 'cycleway'
                color = 'green'
                linewidth = 1.5
            else:
                category = 'other'
                color = 'black'
                linewidth = 1
            
            # Plot geometry
            if geom.geom_type == 'LineString':
                x, y = geom.xy
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.7)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    x, y = line.xy
                    ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.7)
            
            # Plot curb ramps (if present) - without labels for overview
            for side in ['left', 'right']:
                for slot in ['start', 'end']:
                    for position in [1, 2, 3]:
                        ramp_geom = row[f'sidewalk_{side}_curbramp_{slot}_{position}_geometry']
                        if ramp_geom is not None:
                            try:
                                ramp_point = shapely_wkb.loads(bytes(ramp_geom))
                                ax.plot(ramp_point.x, ramp_point.y, 'o', color='red', 
                                       markersize=4, alpha=0.8, zorder=5,
                                       markeredgecolor='darkred', markeredgewidth=0.5)
                            except:
                                pass
            
            # Plot curb return geometry (if present)
            curb_return_geom = row.get('curb_return_geometry')
            if curb_return_geom is not None:
                try:
                    curb_return = shapely_wkb.loads(bytes(curb_return_geom))
                    if curb_return.geom_type == 'LineString':
                        x, y = curb_return.xy
                        ax.plot(x, y, color='orange', linewidth=2, alpha=0.7, 
                               linestyle='--', zorder=4)
                except:
                    pass
        except Exception as e:
            print(f"  Warning: Could not plot row {idx}: {e}")
    
    # Plot target point
    ax.plot(target_lon, target_lat, 'r*', markersize=20, label='Target Location', zorder=10)
    
    # Set bounds
    ax.set_xlim(bbox['minx'], bbox['maxx'])
    ax.set_ylim(bbox['miny'], bbox['maxy'])
    
    # Labels and formatting
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Street Centerlines - {map_config["label"]}\n{target_lat:.6f}°N, {abs(target_lon):.6f}°W', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(color='gray', label='Carway'),
        mpatches.Patch(color='blue', label='Footway (separate)'),
        mpatches.Patch(color='green', label='Cycleway (separate)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markeredgecolor='darkred', markersize=6, label='Curb Ramp'),
        Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Curb Return'),
        mpatches.Patch(color='black', label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Save map
    # Create unique filename based on location label
    location_slug = map_config['label'].lower().replace(' ', '_').replace('/', '_')
    map_filename = f"{city_name.lower().replace(' ', '_')}_{location_slug}_map.png"
    output_path = Path("Notebooks/Karna/Proximity Model/Output/Diagnostic Maps") / map_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Map saved to: {output_path}")
    print(f"  Image size: {output_path.stat().st_size / 1024:.1f} KB")
    
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PARQUET FILE VERIFICATION")
    print("=" * 80)
    print(f"\nVerifying {len(PARQUET_FILES)} parquet file(s)...\n")
    
    # Process each parquet file
    for config in PARQUET_FILES:
        df = verify_parquet_file(config)
        
        if df is not None:
            # Generate main location map
            generate_map(df, config)
            
            # Generate diagnostic maps
            generate_all_diagnostic_maps(df, config['name'])
            
            # Generate additional location maps
            if 'additional_locations' in config and len(config['additional_locations']) > 0:
                print(f"\n{'='*80}")
                print(f"GENERATING ADDITIONAL LOCATION MAPS - {config['name']}")
                print(f"{'='*80}")
                
                for loc in config['additional_locations']:
                    loc_config = {
                        'name': config['name'],
                        'map_location': loc
                    }
                    generate_map(df, loc_config)
    
    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETE")
    print("=" * 80)
