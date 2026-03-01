## Street Network Parquet Schema:

For all columns, if a value is empty it is considered missing and should be available for public input. Exceptions are public_data_id columns, maxspeed, and geometry columns (for now).

Column titles are grouped by facility type and facility location relative to the bearing of the street centerline in the general format facility_side_n for segments and facility_start/end(_n) for point features associated with those segments. 

*Column Section Types:*
Street Centerline
Street Feature

Sidewalk Centerline
Sidewalk Feature

Bikelane Centerline
Bikelane Feature

Main Geometries

**Output Parquet Format (EPSG=4326)**

*Street Centerlines*
block_ids: struct{left: id, right: id}
street_id: osmid by default
block_sides: struct{left: block side, right: block side}
public_data_id_street: if populated, Government data is being used for street centerline geometry
start_node_id: osmid by default
start_node_is_block_node: False by default. True if the node is one of the vertices of the polygon described by the segments that make up a block. 
start_node_is_intersection_node: False by default. True if the node is shared by 3 or more block sides.  
end_node_id: osmid by default
end_node_is_block_node
end_node_is_intersection_node
public_data_id_start_end_nodes: Tuple(start node id, end node id). Only used for lookup and analysis. 
normalized_bearing: degrees
name
highway
maxspeed: Should be able to configure default in case of missing value in config
oneway
lanes
lane_width
surface

*Street Centerline Features*
street_feature_types: List[Str], parallel with street_feature_geometry multipoint.
public_data_id_street_feature: List[Str | null] parallel with street_feature_geometry multipoint. If populated, Government data is being used for street_feature_geometry entry.
street_feature_geometry: Multi-point
street_feature_geometry_projected: Multi-point. Closest coordinate along the streetsegment linestring that is in line with the reported feature location. Parallel with street_feature_geometry.


*Sidewalk Centerlines (Left)*
sidewalk_left_ID: Sequential id assigned during network generation if sidewalk present on left of street segment.
sidewalk_left_block_ID: Sequential ID
sidewalk_left_presence: Based on OSM sidewalk:left/right/separate, left and right should be normalized based on segment bearing and osm sidewalk input should be adjusted accordingly
public_data_id_sidewalk_left: if populated, Government data is being used for sidewalk_left_geometry
sidewalk_left_surface
sidewalk_left_quality: Populated with OSM smoothness by default
sidewalk_left_width
sidewalk_left_incline
sidewalk_left_buffered

*Curb Ramp (Left, Start, 1)*
sidewalk_left_curbramp_start_1_ID: Sequential id assigned during network generation. Left and start are determined by normalized bearing of street section
public_data_id_sidewalk_left_curbramp_start_1: if populated, Government data is being used for sidewalk_left_curbramp_start_1_geometry
sidewalk_left_curbramp_start_1_returnloc: 	
Direction of the curb return, [NW, N, NE, E, SE, S, SW, W]
sidewalk_left_curbramp_start_1_returnposition:	
Position of the curb ramp on the return, [Left, Center, Right]
sidewalk_left_curbramp_start_1_condition_score
sidewalk_left_curbramp_start_1_geometry


*Curb Ramp (Left, Start, 2)*
sidewalk_left_curbramp_start_2_ID: In case of multiple ramps with identical ID, curb return loc, and position on return
public_data_id_sidewalk_left_curbramp_start_2
sidewalk_left_curbramp_start_2_returnloc 	
sidewalk_left_curbramp_start_2_returnposition
sidewalk_left_curbramp_start_2_condition_score
sidewalk_left_curbramp_start_2_geometry

*Curb Ramp (Left, Start, 3)*
sidewalk_left_curbramp_start_3_ID
public_data_id_sidewalk_left_curbramp_start_3
sidewalk_left_curbramp_start_3_returnloc 	
sidewalk_left_curbramp_start_3_returnposition
sidewalk_left_curbramp_start_3_condition_score
sidewalk_left_curbramp_start_3_geometry

*Curb Ramp (Left, End, 1)*
sidewalk_left_curbramp_end_1_ID
public_data_id_sidewalk_left_curbramp_end_1
sidewalk_left_curbramp_end_1_returnloc 	
sidewalk_left_curbramp_end_1_returnposition
sidewalk_left_curbramp_end_1_condition_score
sidewalk_left_curbramp_end_1_geometry

*Curb Ramp (Left, End, 2)*
sidewalk_left_curbramp_end_2_ID
public_data_id_sidewalk_left_curbramp_end_2
sidewalk_left_curbramp_end_2_returnloc 	
sidewalk_left_curbramp_end_2_returnposition
sidewalk_left_curbramp_end_2_condition_score
sidewalk_left_curbramp_end_2_geometry

*Curb Ramp (Left, End, 3)*
sidewalk_left_curbramp_end_3_ID
public_data_id_sidewalk_left_curbramp_end_3
sidewalk_left_curbramp_end_3_returnloc 	
sidewalk_left_curbramp_end_3_returnposition
sidewalk_left_curbramp_end_3_condition_score
sidewalk_left_curbramp_end_3_geometry

*Sidewalk Centerline Features (Left)*
sidewalk_left_feature_ids: Sequentially assigned during network generation. List[Str], parallel with sidewalk_left_feature_geometry multipoint.
sidewalk_left_feature_types: List[Str], parallel with sidewalk_left_feature_geometry multipoint.
public_data_id_sidewalk_left_feature: List[Str | null] parallel with sidewalk_left_geometry. If populated, Government data is being used for sidewalk_left_feature_geometry
sidewalk_left_feature_geometry: Multi-point
sidewalk_left_feature_geometry_projected: Multi-point. Closest coordinate along the sidewalk_leftsegment linestring that is in line with the reported feature location. Parallel with sidewalk_left_feature_geometry.


*Sidewalk Centerlines (Right)*
sidewalk_right_ID: Sequential id assigned during network generation if sidewalk present on right of street segment.
sidewalk_right_block_ID
sidewalk_right_presence: Based on OSM sidewalk:right/right/separate, right and right should be normalized based on segment bearing
public_data_id_sidewalk_right: if populated, Government data is being used for sidewalk_right_geometry
sidewalk_right_surface
sidewalk_right_quality: Populated with OSM smoothness by default
sidewalk_right_width
sidewalk_right_incline
sidewalk_right_buffered

*Curb Ramp (Right, Start, 1)*
sidewalk_right_curbramp_start_1_ID: Sequential id assigned during network generation. right and start are determined by normalized bearing of street section
public_data_id_sidewalk_right_curbramp_start_1: if populated, Government data is being used for sidewalk_right_curbramp_start_1_geometry
sidewalk_right_curbramp_start_1_returnloc: 	
Direction of the curb return, [NW, N, NE, E, SE, S, SW, W]
sidewalk_right_curbramp_start_1_returnposition:	
Position of the curb ramp on the return, [right, Center, Right]
sidewalk_right_curbramp_start_1_condition_score
sidewalk_right_curbramp_start_1_geometry

*Curb Ramp (Right, Start, 2)*
sidewalk_right_curbramp_start_2_ID: In case of multiple ramps with identical CNN, curb return loc, and position on return
public_data_id_sidewalk_right_curbramp_start_2
sidewalk_right_curbramp_start_2_returnloc 	
sidewalk_right_curbramp_start_2_returnposition
sidewalk_right_curbramp_start_2_condition_score
sidewalk_right_curbramp_start_2_geometry

*Curb Ramp (Right, Start, 3)*
sidewalk_right_curbramp_start_3_ID
public_data_id_sidewalk_right_curbramp_start_3
sidewalk_right_curbramp_start_3_returnloc 	
sidewalk_right_curbramp_start_3_returnposition
sidewalk_right_curbramp_start_3_condition_score
sidewalk_right_curbramp_start_3_geometry

*Curb Ramp (Right, End, 1)*
sidewalk_right_curbramp_end_1_ID
public_data_id_sidewalk_right_curbramp_end_1
sidewalk_right_curbramp_end_1_returnloc 	
sidewalk_right_curbramp_end_1_returnposition
sidewalk_right_curbramp_end_1_condition_score
sidewalk_right_curbramp_end_1_geometry

*Curb Ramp (Right, End, 2)*
sidewalk_right_curbramp_end_2_ID
public_data_id_sidewalk_right_curbramp_end_2
sidewalk_right_curbramp_end_2_returnloc 	
sidewalk_right_curbramp_end_2_returnposition
sidewalk_right_curbramp_end_2_condition_score
sidewalk_right_curbramp_end_2_geometry

*Curb Ramp (Right, End, 3)*
sidewalk_right_curbramp_end_3_ID
public_data_id_sidewalk_right_curbramp_end_3
sidewalk_right_curbramp_end_3_returnloc 	
sidewalk_right_curbramp_end_3_returnposition
sidewalk_right_curbramp_end_3_condition_score
sidewalk_right_curbramp_end_3_geometry

*Sidewalk Centerline Features (Right)*
sidewalk_right_feature_ids: Sequentially assigned during network generation. List[Str], parallel with sidewalk_right_feature_geometry multipoint.
sidewalk_right_feature_types: List[Str], parallel with sidewalk_right_feature_geometry multipoint.
public_data_id_sidewalk_right_feature: List[Str | null] parallel with sidewalk_right_geometry. If populated, Government data is being used for sidewalk_right_feature_geometry
sidewalk_right_feature_geometry: Multi-point
sidewalk_right_feature_geometry_projected: Multi-point. Closest coordinate along the sidewalk_rightsegment linestring that is in line with the reported feature location. Parallel with sidewalk_right_feature_geometry.

*Crosswalk (Start)*
crosswalk_start_id: Sequentially assigned during network generation. 
crosswalk_start_block_ids: Tuple(sidewalk_right_block_id, sidewalk_left_block_id)
crosswalk_start_type
public_data_id_crosswalk_start: If populated, Government data is being used for crosswalk_start_geometry
crosswalk_start_controlled
crosswalk_start_marked
crosswalk_start_markings
crosswalk_start_signals: List[yes/no, button yes/no, sound yes/no, vibration yes/no, flashing_lights yes/button/sensor] (select one from each choice)
crosswalk_start_island=yes/no
crosswalk_start_kerb
crosswalk_start_tactile_paving
crosswalk_start_traffic_calming
crosswalk_start_continuous
crosswalk_start_condition
crosswalk_start_geometry
crosswalk_start_island_geometry: Multipoint list of crossing islands. 

*Crosswalk (End)*
crosswalk_end_id
crosswalk_end_block_ids
crosswalk_end_type
public_data_id_crosswalk_end: If populated, Government data is being used for crosswalk_end_geometry
crosswalk_end_controlled
crosswalk_end_marked
crosswalk_end_markings
crosswalk_end_signals: List[yes/no, button yes/no, sound yes/no, vibration yes/no, flashing_lights yes/button/sensor] (select one from each choice)
crosswalk_end_island=yes/no
crosswalk_end_kerb
crosswalk_end_tactile_paving
crosswalk_end_traffic_calming
crosswalk_end_continuous
crosswalk_end_condition
crosswalk_end_geometry
crosswalk_end_island_geometry

*Bikeway Centerline (Left, 1)*
bikeway_left_1_id: Sequentially assigned during network generation if bikeway is present.
bikeway_left_1_block_id: Should also be called in place of bikeway_left_2_block_id
public_data_id_bikeway_left_1: If populated, Government data is being used for bikeway_left_1_geometry
bikeway_left_1_type: Based on OSM cycleway
bikeway_left_1_surface
bikeway_left_1_quality: Populated with OSM smoothness by default
bikeway_left_1_permitted: Based on OSM bicycle
bikeway_left_1_width
bikeway_left_1_incline
bikeway_left_buffered

*Bikeway Centerline (Left, 2)*
bikeway_left_2_id: In case of cycleway:left:2
public_data_id_bikeway_left_2
bikeway_left_2_type
bikeway_left_2_surface
bikeway_left_2_quality: Populated with OSM smoothness by default
bikeway_left_2_permitted
bikeway_left_2_width
bikeway_left_2_incline

*Bikeway Centerline Features (Left, 1)*
bikeway_left_1_feature_ids: List[Str], parallel with bikeway_left_1_feature_geometry multipoint.
bikeway_left_1_feature_types: List[Str], parallel with bikeway_left_1_feature_geometry multipoint.
public_data_id_bikeway_left_1_features: List[Str | null] parallel with sidewalk_right_geometry. If populated, Government data is being used for bikeway_left_1_feature_geometry
bikeway_left_1_feature_geometry: Multi-point
bikeway_left_1_feature_geometry_projected: Multi-point. Closest coordinate along the bikeway_left_1segment linestring that is in line with the reported feature location. Parallel with bikeway_left_1_feature_geometry.

*Bikeway Centerline Features (Left, 2)*
bikeway_left_2_feature_types: List[Str], parallel with bikeway_left_2_feature_geometry multipoint.
public_data_id_bikeway_left_2_features
bikeway_left_2_feature_geometry: Multi-point
bikeway_left_2_feature_geometry_projected: Multi-point. Closest coordinate along the bikeway_left_2segment linestring that is in line with the reported feature location. Parallel with bikeway_left_2_feature_geometry.

*Bikeway Centerline (Right, 1)*
bikeway_right_1_id: Sequentially assigned during network generation if bikeway is present. 
bikeway_right_1_block_id: Should also be called in place of bikeway_right_2_block_id
public_data_id_bikeway_right_1: If populated, Government data is being used for bikeway_right_1_geometry
bikeway_right_1_type: Based on OSM cycleway
bikeway_right_1_surface
bikeway_right_1_quality
bikeway_right_1_permitted: Based on OSM bicycle
bikeway_right_1_width
bikeway_right_1_incline
bikeway_right_buffered

*Bikeway Centerline (Right, 2)*
bikeway_right_2_id: In case of cycleway:right:2
public_data_id_bikeway_right_2
bikeway_right_2_type
bikeway_right_2_surface
bikeway_right_2_quality
bikeway_right_2_permitted
bikeway_right_2_width
bikeway_right_2_incline

*Bikeway Centerline Features (Right, 1)*
bikeway_right_1_feature_ids: List[Str], parallel with bikeway_right_1_feature_geometry multipoint.
bikeway_right_1_feature_types: List[Str], parallel with bikeway_right_1_feature_geometry multipoint.
public_data_id_bikeway_right_1_features: List[Str | null] parallel with sidewalk_right_geometry. If populated, Government data is being used for bikeway_right_1_feature_geometry
bikeway_right_1_feature_geometry: Multi-point
bikeway_right_1_feature_geometry_projected: Multi-point. Closest coordinate along the bikeway_right_1segment linestring that is in line with the reported feature location. Parallel with bikeway_right_1_feature_geometry.

*Bikeway Centerline Features (Right, 2)*
bikeway_right_2_feature_types: List[Str], parallel with bikeway_right_2_feature_geometry multipoint.
public_data_id_bikeway_right_2_features: 
bikeway_right_2_feature_geometry: Multi-point
bikeway_right_2_feature_geometry_projected: Multi-point. Closest coordinate along the bikeway_right_2segment linestring that is in line with the reported feature location. Parallel with bikeway_right_2_feature_geometry.

*Main Geometry Columns (wkb)*
street_geometry
start_node_geometry
end_node_geometry
sidewalk_left_geometry
sidewalk_right_geometry
curb_return_geometry
bikeway_left_1_geometry
bikeway_left_2_geometry
bikeway_right_1_geometry
bikeway_right_2_geometry