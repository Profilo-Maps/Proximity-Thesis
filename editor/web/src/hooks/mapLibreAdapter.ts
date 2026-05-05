/**
 * Web-specific MapLibre adapter.
 * Implements MapAdapter by delegating to maplibregl.Map APIs.
 * This is the ONLY file in the tool pipeline that imports maplibre-gl.
 */

import type maplibregl from 'maplibre-gl';
import type { MapAdapter, MapTapEvent, QueriedFeature, NearestSegmentResult, SegmentPairInfo } from '@proximity/shared/tools';
import { nearestPointOnLine } from '@proximity/shared/tools';
import { LAYER_DEFS } from './useMapLayers';

export function createMapLibreAdapter(map: maplibregl.Map): MapAdapter {
  const featureLayerIds = LAYER_DEFS.map((l) => l.id);

  return {
    queryFeatures(tap: MapTapEvent, featureTypes: string[]): QueriedFeature | null {
      const point = map.project([tap.lng, tap.lat]);
      const bbox: [maplibregl.PointLike, maplibregl.PointLike] = [
        [point.x - 8, point.y - 8],
        [point.x + 8, point.y + 8],
      ];
      const features = map.queryRenderedFeatures(bbox, { layers: featureLayerIds });
      const match = features.find((f) => featureTypes.includes(f.properties?._t as string));
      if (!match) return null;
      return { geometry: match.geometry, properties: match.properties as Record<string, unknown> };
    },

    queryHulls(tap: MapTapEvent): QueriedFeature | null {
      const point = map.project([tap.lng, tap.lat]);
      const features = map.queryRenderedFeatures(point, { layers: ['px-hull-fill'] });
      if (features.length === 0) return null;
      return { geometry: features[0].geometry, properties: features[0].properties as Record<string, unknown> };
    },

    findNearestSegmentPoint(tap: MapTapEvent): NearestSegmentResult | null {
      const point = map.project([tap.lng, tap.lat]);
      const bbox: [maplibregl.PointLike, maplibregl.PointLike] = [
        [point.x - 20, point.y - 20],
        [point.x + 20, point.y + 20],
      ];
      const features = map.queryRenderedFeatures(bbox, { layers: ['px-streets'] });
      if (features.length === 0) return null;

      let bestDist = Infinity;
      let bestResult: NearestSegmentResult | null = null;

      for (const f of features) {
        const geom = f.geometry;
        if (geom.type !== 'LineString' && geom.type !== 'MultiLineString') continue;
        const isMulti = geom.type === 'MultiLineString';
        const result = nearestPointOnLine(tap.lng, tap.lat, geom.coordinates, isMulti);
        if (result && result.dist < bestDist) {
          bestDist = result.dist;
          bestResult = {
            lng: result.x,
            lat: result.y,
            segId: (f.properties?._seg_id ?? '') as string,
          };
        }
      }
      return bestResult;
    },

    getSegmentPairInfo(seg1Id: string, seg2Id: string): SegmentPairInfo | null {
      // Query all visible street features to find our two segments
      const allStreets = map.queryRenderedFeatures(undefined as any, { layers: ['px-streets'] });
      const f1 = allStreets.find(f => f.properties?._seg_id === seg1Id);
      const f2 = allStreets.find(f => f.properties?._seg_id === seg2Id);
      if (!f1 || !f2) return null;
      if (f1.geometry.type !== 'LineString' || f2.geometry.type !== 'LineString') return null;

      const c1 = f1.geometry.coordinates as [number, number][];
      const c2 = f2.geometry.coordinates as [number, number][];

      const eps = 0.00001; // ~1m tolerance
      const close = (a: [number, number], b: [number, number]) =>
        Math.abs(a[0] - b[0]) < eps && Math.abs(a[1] - b[1]) < eps;

      // Check all 4 endpoint combinations
      const s1Start = c1[0];
      const s1End = c1[c1.length - 1];
      const s2Start = c2[0];
      const s2End = c2[c2.length - 1];

      let sharedPoint: [number, number] | null = null;
      if (close(s1Start, s2Start) || close(s1Start, s2End)) sharedPoint = s1Start;
      else if (close(s1End, s2Start) || close(s1End, s2End)) sharedPoint = s1End;

      // Find node at shared point
      let sharedNodeId: string | null = null;
      if (sharedPoint) {
        const pt = map.project(sharedPoint);
        const bbox: [maplibregl.PointLike, maplibregl.PointLike] = [
          [pt.x - 10, pt.y - 10],
          [pt.x + 10, pt.y + 10],
        ];
        const nodes = map.queryRenderedFeatures(bbox, { layers: featureLayerIds });
        const node = nodes.find(n => n.properties?._t === 'node');
        if (node) sharedNodeId = (node.properties?._node_id ?? '') as string;
      }

      return {
        seg1Id,
        seg2Id,
        sharedNodeId,
        seg1Coords: c1,
        seg2Coords: c2,
      };
    },

    getHullVertices(nodeId: string): [number, number][] | null {
      const hulls = map.queryRenderedFeatures(undefined as any, { layers: ['px-hull-fill'] });
      const hull = hulls.find(f => String(f.properties?.node_id) === nodeId);
      if (!hull || hull.geometry.type !== 'Polygon') return null;
      const ring = hull.geometry.coordinates[0] as [number, number][];
      // Remove closing vertex (same as first)
      return ring.slice(0, -1);
    },
  };
}
