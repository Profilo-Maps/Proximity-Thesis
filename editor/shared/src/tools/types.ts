/**
 * Platform-agnostic types for tool interactions.
 * Both web (MapLibre) and mobile (RN Mapbox) adapters convert their
 * native events into these shapes before passing to tool logic.
 */

import type { Feature, FeatureCollection, Geometry, Point, LineString } from 'geojson';

/** Describes a mutation to the main GeoJSON feature source */
export type SourceMutation =
  | { type: 'delete'; matchId: string; matchKey: '_node_id' | '_seg_id' | '_fid' }
  | { type: 'add'; feature: Feature }
  | { type: 'updateCoords'; matchId: string; matchKey: '_seg_id' | '_fid'; vertexIndex: number; coords: [number, number] };

/** A map tap/click in geographic coordinates */
export interface MapTapEvent {
  /** Longitude */
  lng: number;
  /** Latitude */
  lat: number;
}

/** A feature returned from a spatial query near a tap point */
export interface QueriedFeature {
  geometry: Geometry;
  properties: Record<string, unknown>;
}

/** Result of projecting a tap onto the nearest line segment */
export interface NearestSegmentResult {
  lng: number;
  lat: number;
  segId: string;
}

/**
 * Platform adapter interface.
 * Each platform (web MapLibre, RN Mapbox) implements this to bridge
 * native map interactions into the shared tool logic.
 */
/** Result of querying two segments to find their shared node */
export interface SegmentPairInfo {
  seg1Id: string;
  seg2Id: string;
  sharedNodeId: string | null;
  seg1Coords: [number, number][];
  seg2Coords: [number, number][];
}

export interface MapAdapter {
  /** Query rendered features near a tap point, filtered by feature type */
  queryFeatures(tap: MapTapEvent, featureTypes: string[]): QueriedFeature | null;

  /** Query rendered hull polygons near a tap point */
  queryHulls(tap: MapTapEvent): QueriedFeature | null;

  /** Find the nearest point on any visible street segment to the tap */
  findNearestSegmentPoint(tap: MapTapEvent): NearestSegmentResult | null;

  /** Get node/coordinate info for two segments to check if they share a node */
  getSegmentPairInfo(seg1Id: string, seg2Id: string): SegmentPairInfo | null;

  /** Get hull polygon vertices for vertex editing */
  getHullVertices(nodeId: string): [number, number][] | null;
}

/**
 * Callbacks the tool logic invokes to update app state.
 * Platform-agnostic — both web and mobile implement these.
 */
export interface ToolCallbacks {
  setStatusText(text: string): void;
  setSelectedFeatureId(id: string | null): void;
  addEditPreview(geometry: Geometry, editType: string, props?: Record<string, unknown>): void;
  replaceEditPreviews(filter: (et: string) => boolean, newFeature: { geometry: Geometry; editType: string } | null): void;

  /** Mutate the main GeoJSON feature source in real-time (delete, add, update features) */
  mutateSource(mutation: SourceMutation): void;

  // Changeset recorders
  recordAddedNode(edit: { x: number; y: number }): void;
  recordMovedEndpoint(edit: { node_id: string; new_x: number; new_y: number; rubber_band_segments: string[] }): void;
  recordDrawnCrosswalk(edit: {
    ramp_a: { segment_id: string; side: string; position: string; index: number };
    ramp_b: { segment_id: string; side: string; position: string; index: number };
  }): void;
  recordEditedHull(edit: { node_id: string; geometry: { type: 'Polygon'; coordinates: [number, number][][] } }): void;
  recordDeletedPoint(edit: { node_id: string; x: number; y: number }): void;
  recordDeletedHull(edit: { node_id: string }): void;
  recordMergedSegments(edit: { surviving_seg_id: string; consumed_seg_id: string; shared_node_id: string }): void;
  recordDrawnSegment(edit: { coordinates: [number, number][] }): void;
}

/** All tool IDs */
export type ToolId =
  | 'move_point' | 'add_node' | 'delete_point'
  | 'highlight_point' | 'merge_segments' | 'split_segment' | 'draw_segment'
  | 'add_polygon' | 'delete_polygon' | 'edit_polygon_face';

/** Subtypes available for tools that create new features */
export type PointSubtype = 'node' | 'ramp' | 'calm';
export type SegmentSubtype = 'street' | 'bikeway' | 'sidewalk' | 'crosswalk' | 'curb_return';
export type PolygonSubtype = 'hull' | 'slot';
export type PolygonEditAction = 'add_ramp' | 'move_vertex' | 'delete_vertex';
export type MoveSubtype = 'move' | 'snap';
export type ToolSubtype = PointSubtype | SegmentSubtype | PolygonSubtype | PolygonEditAction | MoveSubtype;

/** Which tools support subtypes, and their options */
export const TOOL_SUBTYPES: Partial<Record<ToolId, { options: { id: ToolSubtype; label: string }[]; default: ToolSubtype }>> = {
  move_point: {
    options: [
      { id: 'move', label: 'Move' },
      { id: 'snap', label: 'Snap to segment' },
    ],
    default: 'move',
  },
  add_node: {
    options: [
      { id: 'node', label: 'Intersection node' },
      { id: 'ramp', label: 'Curb ramp' },
      { id: 'calm', label: 'Traffic calming' },
    ],
    default: 'node',
  },
  draw_segment: {
    options: [
      { id: 'street', label: 'Street' },
      { id: 'bikeway', label: 'Bikeway' },
      { id: 'sidewalk', label: 'Sidewalk' },
      { id: 'crosswalk', label: 'Crosswalk' },
      { id: 'curb_return', label: 'Curb return' },
    ],
    default: 'street',
  },
  add_polygon: {
    options: [
      { id: 'hull', label: 'Intersection hull' },
      { id: 'slot', label: 'Crosswalk slot' },
    ],
    default: 'hull',
  },
  edit_polygon_face: {
    options: [
      { id: 'move_vertex',   label: 'Move vertex' },
      { id: 'add_ramp',      label: 'Add curb ramp' },
      { id: 'delete_vertex', label: 'Delete vertex' },
    ],
    default: 'move_vertex',
  },
};

/** Multi-click tool state that persists between taps */
export interface ToolSessionState {
  moveTarget: string | null;
  drawPoints: [number, number][];
  mergeFirst: string | null;
  crosswalkFirst: { segId: string; side: string; position: string; index: number } | null;
  /** edit_polygon_face: node ID of hull being edited */
  editingHullId: string | null;
  /** edit_polygon_face: current vertex coordinates (mutable during drag) */
  editingHullVertices: [number, number][] | null;
  /** edit_polygon_face: index of vertex being dragged, or null */
  draggingVertexIndex: number | null;
  /** add_node (ramp subtype): selected sidewalk to attach ramp to */
  rampTarget: { segId: string; side: string; position: string } | null;
}

export function createToolSession(): ToolSessionState {
  return {
    moveTarget: null,
    drawPoints: [],
    mergeFirst: null,
    crosswalkFirst: null,
    editingHullId: null,
    editingHullVertices: null,
    draggingVertexIndex: null,
    rampTarget: null,
  };
}
