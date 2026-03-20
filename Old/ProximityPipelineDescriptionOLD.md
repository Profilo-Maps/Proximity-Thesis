Output should be 2 parquet files per city, one for sanity map and one for the detailed street network.

Columns should be populated with OSM data by default, with the columns provided in the column configs corresponding to data in the default slots that should be updated.

config should have options for: global config{export path,default max speed for streets, curb ramp trustworthiness outer buffer, curb ramp trustworthiness inner buffer}, cities_config{City 1[government data paths, street columns, sidewalk columns, bikelane columns, curb ramp columns, feature columns], City2 City 1[government data paths, street columns, sidewalk columns, bikelane columns, curb ramp columns, curb ramp trustworthiness (bool), feature columns]}



https://data.sfgov.org/City-Infrastructure/Curb-Ramps/ch9w-7kih/about_data

## Data Pipeline
**Sanity Buffer Map**:
1. Use parcel data to create aggregated building footprint boundaries that it does not make sense for sidewalks to intersect with. 
2. If parcel data is not available, preload a map with street highway categories and assign maximum off-set values for each segment based on the highway category, number of lanes, and the table below
3. For both methods, export sanity buffer map of osm street segments with related osmids and max_offset_widths as a parquet to the specified output directory with file name [city]_sanity.parquet
+------------------+------------+-----------+------------+------------+--------------------+
| OSM highway      | Lane (m)   | Ref (L)   | Buffer (m) | Ref (B)    | Offset Formula     |
+------------------+------------+-----------+------------+------------+--------------------+
| motorway / trunk | 3.7 (12')  | [1]       | 3.0 (10')  | [2]        | (Lanes * 1.85)+3.0 |
| primary          | 3.4 (11')  | [3]       | 2.5 (8')   | [4]        | (Lanes * 1.7) +2.5 |
| secondary        | 3.3 (11')  | [5]       | 2.0 (6')   | [6]        | (Lanes * 1.65)+2.0 |
| tertiary         | 3.0 (10')  | [7]       | 1.5 (5')   | [8]        | (Lanes * 1.5) +1.5 |
| residential      | 3.0 (10')  | [9]       | 1.2 (4')   | [1]        | (Lanes * 1.5) +1.2 |
| service / alley  | 2.7 (9')   | [10]      | 0.5 (1.5') | [11]       | (Lanes * 1.35)+0.5 |
+------------------+------------+-----------+------------+------------+--------------------+

+-----------------------+---------------------------------------------------------------------------------------------------------+-------------+---------+
| Source Documentation  | URL                                                                                                     | Table Row   |Table Col|
+-----------------------+---------------------------------------------------------------------------------------------------------+-------------+---------+
| AASHTO Green Book     | https://store.transportation.org/Item/PublicationDetail?ID=4653                                         | Motorway    | Lane    |
| FHWA Shoulder Design  | https://ops.fhwa.dot.gov/publications/fhwahop15023/ch8.htm                                              | Motorway    | Buffer  |
| NACTO Lane Guide      | https://nacto.org/publication/urban-street-design-guide/street-design-elements/lane-width/              | Primary     | Lane    |
| Caltrans HDM          | https://dot.ca.gov/-/media/dot-media/programs/design/documents/chp0300-dec-2020--changesa11y.pdf        | Primary     | Buffer  |
| NACTO Secondary St.   | https://nacto.org/publication/urban-street-design-guide/street-design-elements/lane-width/              | Secondary   | Lane    |
| OSM Wiki: Width       | https://wiki.openstreetmap.org/wiki/Key:shoulder                                                        | Secondary   | Buffer  |
| ITE Walkable Comm.    | https://pdhacademy.com/wp-content/uploads/2014/04/Designing-Walkable-Urban-Thoroughfares-Part-1.pdf     | Tertiary    | Lane    |
| ADA / PROWAG          | https://www.access-board.gov/prowag/proposed/planning-and-design-for-alterations/chapter5/              | Tertiary    | Buffer  |
| OSM Wiki: Lanes       | https://wiki.openstreetmap.org/wiki/Key:width                                                           | Residential | Lane    |
| AASHTO Ped. Guide     | https://downloads.transportation.org/GPF-2-Errata.pdf                                                   | Residential | Buffer  |
| SF Better Streets     | https://sfbetterstreets.org/design-guidelines/street-types/index.html                                   | Service     | Lane    |
| OSM Wiki: Service     | https://wiki.openstreetmap.org/wiki/Tag:highway%3Dsecondary_link                                        | Service     | Buffer  |
+-----------------------+---------------------------------------------------------------------------------------------------------+-------------+---------+


**Build Denormalized Street Network**
The big picture of this table is that a street segment is given slots on either side of the street where it can have data stored for contiguous street facilities (ie bikelanes and sidewalks). Each street segment is defined by a start node and end node, pulled initially from OSM's shape nodes. OSM intersection nodes should be loaded for analysis but not explicitly stored in the table. Instead, street segments who have a shape node that is also an intersection node should have their start/end_node_is_block_node booleans set to true. Street segments are organized into block sides, which contain all of the street segments that line one side of a block, and block sides are organized into blocks, which are bounded of a street network (either cyclically bounded a loop or terminally bounded a dead end). Sidewalks and bikelanes are also assigned start and end nodes. For bikelanes these nodes are typically the same as a street segment, though additional nodes can be assigned where a bikefacility is not associated with a roadway. For sidewalks, these start and end nodes are made up of curb ramps. This structure allows for more granular navigation of transitions between street facilities. Curb ramps are connected across streets by crosswalks. Because this table is flat and stores multiple geometry columns, technically any of these facilities could exist independent of the others if explicitly marked in OSM or government data. 

A node is any point where one or more street segments begin or end. Every segment has a start node and an end node.

A block node is a node that is a vertex of at least one detected block polygon. All block nodes have is_block_node = true. Block nodes are identified during Phase 2 as a direct output of block detection. A node that sits mid-segment (a degree-2 pass-through) is not a block node even if it's a vertex of the underlying linestring geometry.

An intersection node is a block node where 3 or more block faces meet. Intersection nodes are where streets cross, where block sides change, and where Phase 3 analysis (curb ramps, crosswalks, facility gap correction) happens. 

Denormalization of OSM data is split into 3 phases:

**1. Load Street Network**
1. Pull Street, Cycleway, footway, and crossing  Data and Geometry from OSM. Cache the crossing data separately.

*Data can be sourced from the following OSM tags*
*OSM Sidewalk Data Format*
sidewalk:both/left/right=yes/no/separate/none
sidewalk:*:surface or sidewalk:surface
sidewalk:*:width
sidewalk:*:incline

OR (if sidewalk=separate)

footway=sidewalk
surface=*
smoothness=*
incline=*
width=*

*OSM Cycleway Data Format*
cycleway:left/right=lane/track/opposite_lane/shared_lane/share_busway/separate/no
cycleway:*:width
cycleway:*:surface
oneway:bicycle=*
bicycle=yes/no/designated/use_sidepath/dismount/private/destination

OR (if cycleway=separate)

highway=cycleway
surface=*
smoothness=*
incline=*
width=*

*OSM Crosswalk Data Format*
highway=crossing or footway=crossing
traffic_signals/uncontrolled
marked/unmarked
crossing:markings=*
crossing:signals=yes/no
button_operated=yes/no
traffic_signals:sound=yes/no
traffic_signals:vibration=yes/no
flashing_lights=yes/button/sensor
crossing:island=yes/no
kerb=*
tactile_paving=yes/no
traffic_calming=table
crossing:continuous=yes/no

2. If Government provided centerlines and/or intersection nodes are provided, join government centerline geometries with OSM street segment geometries and/or nodes, preserving the government geometries and merging the data per the user specified column configs. 

3. Check all street segments for associated bikeways. For streets that have associated bikeways but no separate bikeway geometries stored, 
generate parallel offset bikeway geometry based on street lane width and lane number data. Use the "Sanity Buffer" to impose limits on maximum offset based off of precalculated map or highway class. Max_offset = distance(street_geometry, nearest parcel edge)

4. Check all street segments for associated sidewalks. For streets that have associated sidewalks but no separate sidewalk geometries stored, 
generate parallel offset sidewalk geometry based on street lane number and width as well as bikelane number and width data. Use the "Sanity Buffer" to impose limits on maximum offset based off of precalculated map or highway class. Max_offset = distance(street_geometry, nearest parcel edge)

5. Scan street network for segments that have an inter vertex deflection angle of more than 45degrees anywhere along their length. If a vertex deflection greater than the threshold is detected, split the segment into 2 segments which will have independently calculated bearings. For the vertex where they are split, store the end node of the start side segment and the start node of the end side segment as a negative, sequentially assigned id. So for the 1st split, one segment would have -1 as its start node and the other would have -1 as its end node, for the second split the program does it would be -2 etc. 

6. Conduct layer aware planarization based on OSM layers to ensure that all crossing street, independent bikeway, and independent footway segments are assigned node geometries within each layer. Assigning node geometries should mean that the start/end_node_is_block_node boolean is set to true. 

**2. Assign Block Structure**
In many cases, one block side will have multiple segments that describe the same street surface between two other perpendicular streets. Sometimes this will be because of midblock changes in footway or cycleway facilities, other times it will be because of errors. These segments need to be aggregated. 

A block is a series of contiguous street segments that form a cycle or a that lack an intersection node at one or more ends. A block side is a series of street segments that are part of a block and have start or end nodes at orthogonal streets. Block sides can be identified with matching block ids and normalized bearings (with a buffer range). Blocks will be shaped differently depending on the street surface (carway, footway, cycleway) segments they are measured with. Street segments will be associated with 2 blocks (left and right) as they have no side of the street. Sidewalks and bikeways will be associated with one of the block values in the street segment's block id depending on their left/right position. 

1. Place a bounding box around the city's boundary polygon with a square grid assigned within it. The x-axis of the grid should increase from left to right, and the y-axis of the grid should increase from South to North. This box should be cached for later reference. Block IDs will be made up of the grid column number, then the grid row number, then the sequential position in that grid square that they were detected. Starting in the SouthWestern corner of that grid and working north until the top of the column, search for blocks. When the northern end of the grid is reached, move one unit east and then restart from the southern end of the grid. Block detection should be parallelized between grid chunks. When a block spans multiple grid cells, ownership is assigned to the cell with the lowest row index (furthest south), with ties broken by lowest column index (furthest west).

2. Detect blocks, assign bearings, and classify nodes in a single left-turn traversal pass. For every directed edge that has not yet been assigned to a face, follow the leftmost turn at each node until the traversal returns to the starting edge (cyclically bounded block) or reaches a degree-1 terminal node (terminally bounded block). At each step, record the bearing of the current edge (already computed for turn-angle selection) and accumulate the shoelace cross-product. When a face closes, the sign of the accumulated sum determines winding order: positive (CCW) indicates an interior face (a real block), negative (CW) indicates the unbounded exterior face (discard), zero-area indicates a degenerate dead-end block. Use the winding order to assign the block ID to the left or right slot of each edge in the face: if directed edge u→v was traversed for this face, the face is to the left of u→v. The opposite directed edge v→u belongs to the adjacent face. A terminally bounded block consists of all contiguous segments between a degree-3+ node and a degree-1 terminal node. Minimum segment count is 1. Cul-de-sac bulbs are their own cyclically bounded blocks; the approach street is a separate terminally bounded block. During traversal, maintain a map of node ID → set of block face IDs. Each time a node is encountered, add the current face's block ID to that node's set.

4. Serial stitch gaps between blocks on the edges of grid sections. 

5. Mark nodes and re-assign facility data. For each segment whose start or end node is a vertex of a detected block polygon, set start/end_node_is_block_node to true. Using the node → block face set from step 2, set start/end_node_is_intersection_node to true for any node referenced by 3 or more distinct block faces. Dead-end terminal nodes are block nodes but not intersection nodes. Using the normalized bearings from step 2, compare each segment's block-relative bearing against its original OSM-tagged left/right orientation. Where they differ, swap sidewalk and bikelane data to the correct left/right slot.

6. Label block sides according to the circular mean of the bearings of all street segments that fall between two intersection nodes on one face of a block. Street segments that make up dead-end blocks should have their left block side slot filled by default.

**3. Intersection-based analysis**

At nodes where multiple blocks meet, transitions will need to be handled for pathfinding. For bike facilities, these transitions can follow street segments. For sidewalks, transitions involve leaving one sidewalk, entering a crosswalk, exiting a crosswalk, and entering a second sidewalk. These transition points are handled by using curb ramps as start and end nodes for sidewalk segments. By default, two directional curb ramps are generated per corner (one per crosswalk approach), aligned with each crossing direction.

All facility corrections at an intersection node are resolved in a single constraint-satisfaction pass rather than sequentially by facility type. This prevents ordering dependencies where a bikelane correction invalidates a sidewalk placement or vice versa.

Facility meeting points and curb ramp positions are computed using angle-bisector placement. For each corner, the angular bisector of the two bounding street centerlines determines the direction along which candidate points are placed. The distance along the bisector is the average of the two bounding segments' offset widths (half-street-width + facility offset), clamped to the lesser of the two segments' sanity buffer max_offset values. This adapts naturally to skewed intersections and asymmetric street widths without requiring street polygon data.

Corner angle edge cases:
- Interior angle near 180° (flat corners): handled as T-intersection flat corners (see step 0).
- Interior angle below 15° (near-parallel merge/fork): skip facility meeting-point generation for this corner. These are typically highway ramps or merge geometries where sidewalk continuity does not apply.

Iterate through intersection nodes. These should be found by looking for rows in the parquet with `start/end_node_is_intersection_node` set to true. The corresponding start/end point geometry of those segments is the intersection node. Look first for street segment data, and then if that is missing default to bikeway segment data, then to footway segment data. Parallelize between grid boxes using the block bounding box.

**Corner enumeration and T-intersection detection**

0. For each intersection node, enumerate all approaching street segments and sort them by normalized bearing. Identify corners as the angular sectors between each consecutive pair of approaching segments. Count the number of approaching segments:
   - 4+ segments: standard intersection, one corner between each consecutive pair.
   - 3 segments (T-intersection): three corners exist. Identify the stem segment (the one whose opposite bearing has no corresponding approach). The two corners flanking the stem are normal corners. The third corner, spanning the through-street across from the stem, is a "flat corner" — it has no opposing crosswalk. For the flat corner, generate curb ramps only if sidewalks are present on the through-street segments, but do not generate a crosswalk across the stem approach. Sidewalk geometry on the flat corner should be connected continuously (no gap for a crossing) unless OSM or government data explicitly tags a crosswalk there.
   - 2 segments: not a true intersection (degree-2 pass-through). Skip unless the node is flagged as a block node for other reasons.

**Intersection boundary and geometry normalization**

Before the constraint satisfaction pass, all facility geometry entering the intersection must be normalized to consistent endpoints at the intersection boundary. This handles overshooting buffered geometry, undershooting separate footways, and continuous footways that pass through the intersection without breaking.

1. Define the intersection boundary as the convex hull of the street centerline endpoints at this node, expanded outward by the maximum offset width among all approaching segments. If fewer than 3 distinct centerline endpoints exist at the node (e.g., a T-intersection where endpoints are nearly collinear), fall back to a circle centered on the intersection node with radius equal to the maximum offset width. This boundary defines the zone within which facility geometry belongs to the intersection rather than to a block side.

2. For each sidewalk and bikelane geometry that approaches this intersection node, classify it:
   - Overshooting: the facility linestring crosses the intersection boundary and continues past the intersection node. Trim the linestring back to its intersection point with the boundary. The trim point becomes the new segment endpoint.
   - Undershooting: the facility linestring terminates before reaching the intersection boundary. Flag for extension (handled in the constraint satisfaction pass, step 7).
   - Continuous through-intersection: a single facility linestring (typically a `footway=sidewalk` or `highway=cycleway` with `separate` geometry) passes through the intersection node without terminating. Split the linestring at its two intersection points with the boundary, creating two segments (one per block side) and discarding the interior portion that falls within the boundary. Each new segment now has an endpoint at the boundary. If the facility linestring is tangent to or barely clips the boundary (intersection segment length < 1m), treat it as a single undershooting segment on the side with the longer remaining geometry.
   - No intersection: the facility linestring does not reach or cross the boundary. Leave it unchanged — it may belong to a different intersection or be an orphaned segment.

3. For any continuous-through-intersection facility that was split in step 2, create new rows in the parquet for the newly created segments. Assign them the same attribute data as the original segment. Assign the original segment's start node to the first new segment and the original segment's end node to the second new segment. Assign new negative sequential IDs to the split endpoints at the intersection boundary (following the same convention as Phase 1 step 5 deflection splits). Update block side assignments for the new segments based on which side of the intersection boundary they fall on.

**Single-pass constraint satisfaction per corner**

For each corner at the intersection node, resolve all facility geometry in one pass:

4. Collect the two street segments that bound this corner. For each segment, gather its bikelane slots and sidewalk slots on the side facing this corner. Compute the angular bisector of the two bounding street centerlines at this corner.

5. Check bikelane contiguity. If both segments have bike facilities on their corner-facing side and one or more was generated through buffering, check whether the geometries are contiguous (their endpoints at the intersection boundary are within a configurable tolerance, default 2m). If not, flag for correction.

6. Check sidewalk contiguity. If both segments have sidewalks on their corner-facing side, check whether the geometries are contiguous (same tolerance). If not, flag for correction.

7. Build a corner exclusion zone: the union of all street centerline buffers at this node (centerline buffered by half-street-width using lane count and lane width), the sanity buffer boundary, and any bike/sidewalk facilities with separately stored (non-buffered) geometry. This zone defines where no new facility geometry may be placed.

8. Resolve all flagged facilities for this corner simultaneously:

   *Bikelanes*
   - For flagged bike facilities where both are buffered, compute a candidate meeting point along the corner's angular bisector at a distance equal to the average of the two segments' bikelane offset widths, clamped to the sanity buffer. If the candidate point falls inside the exclusion zone, walk it outward along the bisector until it clears the zone or hits the sanity buffer clamp. If no valid point exists, flag the corner for manual review. Otherwise, extend both bikelane geometries to meet at that point. Add the meeting point and extended geometries to the exclusion zone.
   - For flagged bike facilities where one is buffered and one has separately stored geometry, extend the buffered segment to meet the stored geometry without crossing the exclusion zone. Add the extended geometry to the exclusion zone.

   *Sidewalks*
   - For flagged sidewalks where both are buffered, compute a candidate meeting point along the corner's angular bisector at a distance equal to the average of the two segments' sidewalk offset widths (accounting for any bikelane width between street and sidewalk), clamped to the sanity buffer. If the candidate point falls inside the updated exclusion zone (which now includes any corrected bikelane geometry), walk it outward along the bisector until it clears the zone or hits the clamp. If no valid point exists, flag the corner for manual review. Otherwise, extend both sidewalk geometries to meet at that point.
   - For flagged sidewalks where one is buffered, extend the buffered segment to meet the stored geometry without crossing the exclusion zone.
   - For corners where two footways with geometries marked `separate` are adjacent or where one footway wraps around the corner, compute the meeting point along the angular bisector constrained by the exclusion zone.
   - For undershooting sidewalks (endpoints that were classified as undershooting in step 2 but not flagged as non-contiguous because only one sidewalk exists on this corner), extend the sidewalk to the intersection boundary along its existing bearing, constrained by the exclusion zone.

9. If the exclusion zone is so tight that no valid placement point exists within the corner's angular sector (e.g., very narrow right-of-way), flag the corner for manual review rather than forcing a degenerate placement.

**Default curb ramp generation (two per corner)**

10. For each corner, generate two directional curb ramps — one aligned with each crosswalk approach direction. Curb ramp "a" is placed at the endpoint of the sidewalk segment arriving from the first bounding street. Curb ramp "b" is placed at the endpoint of the sidewalk segment arriving from the second bounding street. Each ramp's geometry defaults to the start/end point of its corresponding sidewalk centerline (now consistently positioned at or near the intersection boundary after normalization). Each ramp is the start/end node for exactly one sidewalk segment and one crosswalk.

   To generate the two directional ramps from the corner meeting point (computed via bisector in step 8), project each ramp position outward from the meeting point along the respective crosswalk approach direction by a configurable offset distance (default: half the sidewalk width or 0.75m, whichever is greater). This splits the single meeting point into two ramp points, one per crosswalk direction.

   When this split creates a gap between the two new ramp points along the corner's curb line, the geometry of the curb face between them is the curb return. Compute this as the arc or line segment connecting ramp "a" to ramp "b" along the corner, following the curb edge (approximated as the arc of a circle centered on the intersection node that passes through both ramp points, or as a straight segment if the corner angle is near 180°). Store this geometry in `curb_return_geometry` for the corresponding street segment's corner. If the computed curb return length is below a minimum threshold (configurable, default 0.3m), collapse the two ramps back into a single apex ramp and leave `curb_return_geometry` empty.

   For T-intersection flat corners: if sidewalks are present on both through-street segments, generate a single connecting curb ramp (or none if the sidewalk is continuous) rather than two directional ramps, since there is no crossing to serve on the stem side. No curb return geometry is generated for flat corners with a single ramp.

**Multi-face conflict resolution**

11. After all corners at a node have been processed independently, run a conflict check across corners that share the same node. For each pair of adjacent corners:
   - Check whether any corrected facility geometry from one corner overlaps or conflicts with corrected geometry from an adjacent corner (e.g., two extended bikelane segments from neighboring corners that cross each other).
   - If conflicts exist, resolve by pulling both conflicting geometries back to the midpoint of their overlap region, re-constrained by the exclusion zone. If no valid resolution exists, flag both corners for manual review.
   - Check that curb ramp placements from adjacent corners maintain a minimum separation distance (configurable, default 1.0m). If two ramps from neighboring corners are closer than the threshold, merge them into a single apex ramp and update the sidewalk segment endpoints accordingly. Clear the `curb_return_geometry` for any merged ramps.

**Government curb ramp integration**

*Curbramps*
Curb ramp slots refer to the curb ramp data columns in the parquet that are associated with the start and end of a sidewalk segment. Curb Ramp Slot b in the diagram below would be the right side end ramp on STR1. It should be joined by crosswalk to Slot a. 

                | [STR1]|
                | (End) |
                |       |
  ______________|[a]_[b]|______________
     (Start)  [h]       [c]       (End)
  ---[STR 2]--    Node        ---[STR 2]---
  ____________[g]_______[d]____________
                |[f] [e]|
                |       |
                |       |
                |(Start)|
                |[STR1] |

With two default ramps per corner, slots [a] and [b] are both populated by default (one per crosswalk direction) rather than sharing a single apex ramp.

If government curb ramp data is being added and ramps are trustworthy:

12. If a government ramp is within the inner trustworthiness buffer of a default ramp, record the public data id and any relevant attributes specified in config in that ramp's curb slot but retain the default geometry.

13. If a government curb point is outside the inner trustworthiness buffer but inside the outer buffer, flag the corresponding default ramp for replacement.

14. When replacing a default ramp from government data, replace the ramp point geometry with the government data and snap the related sidewalk segment endpoint to the new point. Recompute `curb_return_geometry` for the affected corner using the updated ramp positions.

15. If more than two government curb ramps fall within the outer buffer of a single corner, assign the two closest to the existing directional ramp slots (replacing their geometry per step 14). Assign additional ramps in order of proximity to the slot 2 and slot 3 curb ramp columns for the segments at that corner.

16. If government data provides only a single apex ramp where the pipeline generated two directional ramps, merge the two default ramps into one: update both sidewalk segment endpoints to meet at the government ramp point, and clear the second ramp slot. Store the connecting geometry between the two original sidewalk endpoints as `curb_return_geometry`.


**4. Crosswalks**
1. Check the OSM crosswalk data that was previously cached (or government crosswalk data) for all crosswalks that fall within the intersection bounding box. For each of those points, store the crosswalk data in the appropriate start/end crosswalk slots for the street segment. For each crosswalk, draw a line between the left and right curb ramp slots as well as any crosswalk island points which the crosswalk is associated with that crosses through the crosswalk point to populate the geometry. 

2. If a crosswalk is not called out in either OSM or government datasets but a street has both left and right curb ramp slots populated at its start or end with attribute data outside of geometry, then crosswalk geometry should still be drawn for that street but crosswalk attributes should not be populated. 







