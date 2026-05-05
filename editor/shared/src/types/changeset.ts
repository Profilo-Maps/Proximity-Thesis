// Changeset types for the county editor save flow.
// Field names use snake_case to match the FastAPI Pydantic models exactly.

export interface BBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface MovedEndpoint {
  node_id: string;
  new_x: number;
  new_y: number;
  rubber_band_segments: string[];
}

export interface AddedNode {
  x: number;
  y: number;
}

export interface ToggledCurbRamp {
  segment_id: string;
  side: 'left' | 'right';
  position: 'start' | 'end';
  index: 1 | 2 | 3;
  enabled: boolean;
}

export interface RampRef {
  segment_id: string;
  side: string;
  position: string;
  index: number;
}

export interface DrawnCrosswalk {
  ramp_a: RampRef;
  ramp_b: RampRef;
}

export interface EditedHull {
  node_id: string;
  geometry: { type: 'Polygon'; coordinates: [number, number][][] };
  utm_anchor?: { x: number; y: number } | null;
}

export interface AttrEdit {
  street_grid_id: string;
  col: string;
  old_value: string | null;
  new_value: string | null;
}

export interface ConsolidatedHull {
  node_keys: [number, number][];
  buffer_m: number | null;
}

export interface DeletedPoint {
  node_id: string;
  x: number;
  y: number;
}

export interface DeletedHull {
  node_id: string;
}

export interface MergedSegments {
  surviving_seg_id: string;
  consumed_seg_id: string;
  shared_node_id: string;
}

export interface DrawnSegment {
  coordinates: [number, number][];
}

export interface Edits {
  moved_endpoints: MovedEndpoint[];
  added_nodes: AddedNode[];
  toggled_curb_ramps: ToggledCurbRamp[];
  drawn_crosswalks: DrawnCrosswalk[];
  edited_hulls: EditedHull[];
  attr_edits: AttrEdit[];
  consolidated_hulls: ConsolidatedHull[];
  deleted_points: DeletedPoint[];
  deleted_hulls: DeletedHull[];
  merged_segments: MergedSegments[];
  drawn_segments: DrawnSegment[];
}

/** Pipeline stage indices */
export enum PipelineStage {
  SNAP_ENDPOINTS = 0,
  BUILD_HULLS = 1,
  ASSIGN_RAMPS = 2,
  CREATE_XWALKS = 3,
  ATTR_ONLY = 99,
}

/** The full changeset sent to POST /save */
export interface SaveRequest {
  parquet: string;
  dirty_from_stage: PipelineStage;
  bbox: BBox;
  edits: Edits;
}

export interface SaveResponse {
  status: string;
  stages_run: number[];
  rows_affected: number;
  bbox_used: BBox;
}

/** Edit types and their pipeline stage mapping */
export const EDIT_STAGE_MAP: Record<string, PipelineStage> = {
  move_point: PipelineStage.SNAP_ENDPOINTS,
  snap_point: PipelineStage.SNAP_ENDPOINTS,
  merge_segments: PipelineStage.SNAP_ENDPOINTS,
  split_segment: PipelineStage.SNAP_ENDPOINTS,
  draw_segment: PipelineStage.SNAP_ENDPOINTS,
  add_point: PipelineStage.BUILD_HULLS,
  delete_point: PipelineStage.BUILD_HULLS,
  add_polygon: PipelineStage.BUILD_HULLS,
  delete_polygon: PipelineStage.BUILD_HULLS,
  consolidate_hulls: PipelineStage.BUILD_HULLS,
  toggle_curb_ramp: PipelineStage.ASSIGN_RAMPS,
  edit_polygon_face: PipelineStage.ASSIGN_RAMPS,
  draw_crosswalk: PipelineStage.CREATE_XWALKS,
  attr_edit: PipelineStage.ATTR_ONLY,
};
