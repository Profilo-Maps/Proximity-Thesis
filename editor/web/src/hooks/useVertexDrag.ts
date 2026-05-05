/**
 * Hook for drag-to-move vertex editing on MapLibre.
 * Extracts vertices from all visible line features (streets, bikeways, sidewalks)
 * into a separate GeoJSON source, and handles mousedown → drag → mouseup.
 *
 * On commit, mutates the original feature's coordinates in-place on the
 * `proximity-features` source so changes are visible immediately.
 */

import { useEffect, useRef, useCallback } from 'react';
import type maplibregl from 'maplibre-gl';
import type { Feature, FeatureCollection, Point } from 'geojson';
import { useEditorStore } from '../store/editorStore';
import { useChangesetStore } from '../store/changesetStore';

/** Line feature types whose vertices should be draggable */
const DRAGGABLE_LINE_TYPES = ['street', 'bikeway', 'sidewalk'];
/** Point feature types that can be dragged directly */
const DRAGGABLE_POINT_TYPES = ['node', 'ramp', 'calm'];

/** Vertex layers that receive pointer events */
const VERTEX_LAYERS = ['px-vertices-end', 'px-vertices-mid'];

interface DragState {
  /** Source feature _seg_id or _fid */
  featureId: string;
  /** Feature type (_t) */
  featureType: string;
  /** Index of vertex within the LineString coordinates array (-1 for standalone points) */
  vertexIndex: number;
  /** Original coordinates before drag started */
  originalCoords: [number, number];
  /** Whether this is an endpoint (first/last vertex of a line, or a standalone point) */
  isEndpoint: boolean;
}

/**
 * Extract vertices from line features into a FeatureCollection of Points.
 * Endpoints get `_endpoint: true`, mid-vertices `_endpoint: false`.
 * When a selectedSegmentId is provided, only extract vertices from that segment.
 */
export function extractVertices(fc: FeatureCollection | null, selectedSegmentId?: string | null): FeatureCollection {
  const features: Feature<Point>[] = [];
  if (!fc) return { type: 'FeatureCollection', features };

  for (const f of fc.features) {
    const t = f.properties?._t;

    // Standalone point features (node, ramp, calm)
    if (DRAGGABLE_POINT_TYPES.includes(t) && f.geometry.type === 'Point') {
      const fid = f.properties?._fid ?? f.properties?._node_id ?? f.properties?._seg_id ?? '';
      features.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [f.geometry.coordinates[0], f.geometry.coordinates[1]] },
        properties: {
          _src_id: fid,
          _src_type: t,
          _vi: -1,
          _endpoint: true,
        },
      });
      continue;
    }

    // Line vertices
    if (!DRAGGABLE_LINE_TYPES.includes(t)) continue;
    if (f.geometry.type !== 'LineString') continue;

    const fid = f.properties?._seg_id ?? f.properties?._fid ?? '';

    // If a segment is selected, only show vertices for that segment
    if (selectedSegmentId && fid !== selectedSegmentId) continue;

    const coords = f.geometry.coordinates;
    for (let i = 0; i < coords.length; i++) {
      const isEndpoint = i === 0 || i === coords.length - 1;
      features.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [coords[i][0], coords[i][1]] },
        properties: {
          _src_id: fid,
          _src_type: t,
          _vi: i,
          _endpoint: isEndpoint,
        },
      });
    }
  }

  return { type: 'FeatureCollection', features };
}

export function useVertexDrag(mapRef: React.RefObject<maplibregl.Map | null>) {
  const dragRef = useRef<DragState | null>(null);
  const activeTool = useEditorStore((s) => s.activeTool);
  const features = useEditorStore((s) => s.features);
  const selectedFeatureId = useEditorStore((s) => s.selectedFeatureId);

  const isVertexTool = activeTool === 'move_point';

  // Show/hide vertex layers — only when move_point tool is active
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const vis = isVertexTool ? 'visible' : 'none';
    for (const id of VERTEX_LAYERS) {
      try { map.setLayoutProperty(id, 'visibility', vis); } catch { /* layer may not exist yet */ }
    }
  }, [isVertexTool, mapRef]);

  // Rebuild vertex source when features change (only relevant when tool is active)
  // Only show vertices for the selected segment to avoid cluttering the map
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    const src = map.getSource('proximity-vertices') as maplibregl.GeoJSONSource | undefined;
    if (!src) return;
    src.setData(extractVertices(isVertexTool ? features : null, isVertexTool ? selectedFeatureId : null));
  }, [features, isVertexTool, selectedFeatureId, mapRef]);

  /** Find the vertex point under the cursor */
  const hitTestVertex = useCallback((e: maplibregl.MapMouseEvent): DragState | null => {
    const map = mapRef.current;
    if (!map) return null;

    const bbox: [maplibregl.PointLike, maplibregl.PointLike] = [
      [e.point.x - 10, e.point.y - 10],
      [e.point.x + 10, e.point.y + 10],
    ];

    const hits = map.queryRenderedFeatures(bbox, { layers: VERTEX_LAYERS });
    if (hits.length === 0) return null;

    const h = hits[0];
    return {
      featureId: (h.properties?._src_id ?? '') as string,
      featureType: (h.properties?._src_type ?? '') as string,
      vertexIndex: h.properties?._vi as number,
      originalCoords: (h.geometry as Point).coordinates as [number, number],
      isEndpoint: h.properties?._endpoint as boolean,
    };
  }, [mapRef]);

  /** Update the vertex position in the features source in real-time */
  const moveVertexInSource = useCallback((drag: DragState, lngLat: [number, number]) => {
    const map = mapRef.current;
    if (!map) return;

    const store = useEditorStore.getState();
    const fc = store.features;
    if (!fc) return;

    // Find the source feature and update its coordinate
    const isStandalonePoint = drag.vertexIndex === -1;

    if (isStandalonePoint) {
      // Standalone point feature (node, ramp, calm)
      const feature = fc.features.find((f) => {
        const id = f.properties?._fid ?? f.properties?._node_id ?? f.properties?._seg_id;
        return id === drag.featureId && f.geometry.type === 'Point';
      });
      if (!feature || feature.geometry.type !== 'Point') return;
      feature.geometry.coordinates = [lngLat[0], lngLat[1]];
    } else {
      // Line vertex
      const feature = fc.features.find((f) => {
        const id = f.properties?._seg_id ?? f.properties?._fid;
        return id === drag.featureId && f.geometry.type === 'LineString';
      });
      if (!feature || feature.geometry.type !== 'LineString') return;
      feature.geometry.coordinates[drag.vertexIndex] = lngLat;
    }

    // Push to map source
    const src = map.getSource('proximity-features') as maplibregl.GeoJSONSource | undefined;
    if (src) src.setData(fc);

    // Also update the vertex points for the selected segment
    const editorState = useEditorStore.getState();
    const vertSrc = map.getSource('proximity-vertices') as maplibregl.GeoJSONSource | undefined;
    if (vertSrc) vertSrc.setData(extractVertices(fc, editorState.selectedFeatureId));
  }, [mapRef]);

  /** Commit the drag as a changeset entry */
  const commitDrag = useCallback((drag: DragState, finalLngLat: [number, number]) => {
    const changeset = useChangesetStore.getState();
    const isStandalonePoint = drag.vertexIndex === -1;

    if (isStandalonePoint) {
      // Standalone point (node, ramp, calm) — record as moved endpoint using the feature ID
      changeset.recordMovedEndpoint({
        node_id: drag.featureId,
        new_x: finalLngLat[0],
        new_y: finalLngLat[1],
        rubber_band_segments: [],
      });
      useEditorStore.getState().setStatusText(
        `${drag.featureType} ${drag.featureId} moved to ${finalLngLat[0].toFixed(5)}, ${finalLngLat[1].toFixed(5)}`
      );
    } else {
      // Line vertex
      changeset.recordMovedEndpoint({
        node_id: `${drag.featureId}:v${drag.vertexIndex}`,
        new_x: finalLngLat[0],
        new_y: finalLngLat[1],
        rubber_band_segments: [drag.featureId],
      });
      useEditorStore.getState().setStatusText(
        `Vertex ${drag.vertexIndex} of ${drag.featureId} moved to ${finalLngLat[0].toFixed(5)}, ${finalLngLat[1].toFixed(5)}`
      );
    }
  }, []);

  // Wire mousedown/mousemove/mouseup for drag
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const onMouseDown = (e: maplibregl.MapMouseEvent) => {
      if (!isVertexTool) return;
      const drag = hitTestVertex(e);
      if (!drag) return;

      e.preventDefault();
      // Snapshot BEFORE drag mutates geometry so undo can restore the pre-drag state
      useEditorStore.getState().pushSnapshot();
      dragRef.current = drag;
      map.getCanvas().style.cursor = 'grabbing';
      // Disable map panning during drag
      map.dragPan.disable();
    };

    const onMouseMove = (e: maplibregl.MapMouseEvent) => {
      if (!dragRef.current) {
        // Hover cursor
        if (isVertexTool) {
          const hit = hitTestVertex(e);
          map.getCanvas().style.cursor = hit ? 'grab' : 'crosshair';
        }
        return;
      }
      const lngLat: [number, number] = [e.lngLat.lng, e.lngLat.lat];
      moveVertexInSource(dragRef.current, lngLat);
    };

    const onMouseUp = (e: maplibregl.MapMouseEvent) => {
      if (!dragRef.current) return;
      const lngLat: [number, number] = [e.lngLat.lng, e.lngLat.lat];
      moveVertexInSource(dragRef.current, lngLat);
      commitDrag(dragRef.current, lngLat);
      dragRef.current = null;
      map.dragPan.enable();
      map.getCanvas().style.cursor = isVertexTool ? 'crosshair' : '';
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dragRef.current) {
        // Cancel drag — revert to original position
        moveVertexInSource(dragRef.current, dragRef.current.originalCoords);
        dragRef.current = null;
        map.dragPan.enable();
        map.getCanvas().style.cursor = isVertexTool ? 'crosshair' : '';
      }
    };

    map.on('mousedown', onMouseDown);
    map.on('mousemove', onMouseMove);
    map.on('mouseup', onMouseUp);
    window.addEventListener('keydown', onKeyDown);

    return () => {
      map.off('mousedown', onMouseDown);
      map.off('mousemove', onMouseMove);
      map.off('mouseup', onMouseUp);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [mapRef, isVertexTool, hitTestVertex, moveVertexInSource, commitDrag]);
}
