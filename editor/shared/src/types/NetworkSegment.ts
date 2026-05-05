// NETWORK SEGMENT TYPES
// TypeScript interfaces for the Proximity graph network.
// Each parquet row maps 1:1 to a NetworkSegment.
//
// Editability rule: all attributes are editable except
// public_data_id_* columns, maxspeed, and geometry columns.
// "seperator" spelling is intentional — matches parquet column names.

import type { LineString, MultiLineString, MultiPoint, Point, Geometry } from 'geojson';

export interface StreetFeatureSet {
  types: string[] | null;
  geometry: MultiPoint | null;
  geometryProjected: MultiPoint | null;
  attributes: Record<string, unknown>[] | null;
}

export interface SegmentFeatureSet {
  ids: string[] | null;
  types: string[] | null;
  geometry: MultiPoint | null;
  geometryProjected: MultiPoint | null;
}

export interface CurbRampSlot {
  id: string | null;
  position: 'start' | 'end';
  slotNumber: 1 | 2 | 3;
  returnloc: string | null;
  returnposition: string | null;
  conditionScore: number | null;
  geometry: Point | null;
}

export interface SidewalkData {
  id: string;
  gridId: string;
  presence: string | null;
  surface: string | null;
  quality: string | null;
  width: number | null;
  incline: number | null;
  seperator: string | null;
  buffered: string;
  geometry: LineString | MultiLineString | null;
  curbRamps: CurbRampSlot[];
  features: SegmentFeatureSet | null;
}

export interface CrosswalkData {
  id: string | null;
  gridIds: string | null;
  type: string | null;
  controlled: string | null;
  marked: string | null;
  markings: string | null;
  signals: string | null;
  island: string | null;
  kerb: string | null;
  tactilePaving: string | null;
  trafficCalming: string | null;
  continuous: string | null;
  condition: string | null;
  geometry: LineString | null;
  islandGeometry: MultiPoint | null;
}

export interface BikewayData {
  id: string | null;
  gridId: string | null;
  type: string | null;
  surface: string | null;
  quality: string | null;
  permitted: string | null;
  width: number | null;
  incline: number | null;
  seperator: string | null;
  buffered: string;
  geometry: LineString | MultiLineString | null;
  features: SegmentFeatureSet | null;
}

export interface NetworkSegment {
  streetId: string;
  streetGridId: string;
  intersectionReviewFlag: boolean;
  startNodeId: number;
  startNodeIsIntersection: boolean;
  endNodeId: number;
  endNodeIsIntersection: boolean;
  normalizedBearing: number;

  street: {
    name: string | null;
    highway: string | null;
    maxspeed: number | null;
    oneway: string | null;
    lanes: number | null;
    laneWidth: number | null;
    surface: string | null;
    incline: number | null;
    geometry: LineString;
    features: StreetFeatureSet | null;
  };

  sidewalkLeft: SidewalkData | null;
  sidewalkRight: SidewalkData | null;
  crosswalkStart: CrosswalkData | null;
  crosswalkEnd: CrosswalkData | null;
  bikewayLeft1: BikewayData | null;
  bikewayLeft2: BikewayData | null;
  bikewayRight1: BikewayData | null;
  bikewayRight2: BikewayData | null;

  startNodeGeometry: Point | null;
  endNodeGeometry: Point | null;
  curbReturnGeometry: Geometry | null;
  publicDataIds: Record<string, string | null>;
}

export type FacilityType =
  | 'street'
  | 'sidewalk_left'
  | 'sidewalk_right'
  | 'crosswalk_start'
  | 'crosswalk_end'
  | 'bikeway_left_1'
  | 'bikeway_left_2'
  | 'bikeway_right_1'
  | 'bikeway_right_2';

export interface SegmentEdit {
  id?: string;
  tripId: string;
  userId: string;
  streetGridId: string;
  facilityType: FacilityType;
  fieldName: string;
  oldValue: string | null;
  newValue: string;
  createdAt?: string;
}
