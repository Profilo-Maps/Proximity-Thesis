/**
 * Web-specific hook that wires MapLibre map events to the shared tool logic.
 * Platform-specific concerns (DOM cursor, keyboard, MapLibre events) live here.
 * All tool logic lives in @proximity/shared/tools.
 */

import { useEffect, useCallback, useRef } from 'react';
import type maplibregl from 'maplibre-gl';
import { useEditorStore } from '../store/editorStore';
import { useChangesetStore } from '../store/changesetStore';
import { createMapLibreAdapter } from './mapLibreAdapter';
import {
  handleToolTap,
  handleToolFinish,
  TOOL_STATUS,
  createToolSession,
  type ToolCallbacks,
  type MapAdapter,
  type ToolSessionState,
  type SourceMutation,
} from '@proximity/shared/tools';
import { extractVertices, VERTEX_LAYERS } from './useVertexDrag';

export function useToolHandler(
  mapRef: React.RefObject<maplibregl.Map | null>
) {
  const activeTool = useEditorStore((s) => s.activeTool);
  const setStatusText = useEditorStore((s) => s.setStatusText);

  // Shared tool session state (persists across renders, reset on tool change)
  const sessionRef = useRef<ToolSessionState>(createToolSession());

  // Adapter ref — recreated when map changes
  const adapterRef = useRef<MapAdapter | null>(null);

  // Reset session when tool changes
  useEffect(() => {
    sessionRef.current = createToolSession();
    setStatusText(activeTool ? TOOL_STATUS[activeTool] ?? '' : '');
  }, [activeTool, setStatusText]);

  // === Platform-specific: cursor ===
  const probeActive = useEditorStore((s) => s.probeActive);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const canvas = map.getCanvas();
    canvas.style.cursor = activeTool ? 'crosshair' : probeActive ? 'cell' : '';
    return () => { canvas.style.cursor = ''; };
  }, [activeTool, probeActive, mapRef]);

  // Build platform-agnostic callbacks that bridge to Zustand stores
  const buildCallbacks = useCallback((): ToolCallbacks => {
    const store = useEditorStore.getState();
    const changeset = useChangesetStore.getState();
    return {
      setStatusText: (t) => store.setStatusText(t),
      setSelectedFeatureId: (id) => store.setSelectedFeatureId(id),
      addEditPreview: (geometry, editType, props) => {
        const s = useEditorStore.getState();
        const next = {
          type: 'FeatureCollection' as const,
          features: [
            ...s.editFeatures.features,
            { type: 'Feature' as const, geometry, properties: { _et: editType, ...(props ?? {}) } },
          ],
        };
        s.setEditFeatures(next);
        const editSrc = mapRef.current?.getSource('proximity-edits') as import('maplibre-gl').GeoJSONSource | undefined;
        editSrc?.setData(next);
      },
      replaceEditPreviews: (filter, newFeature) => {
        const s = useEditorStore.getState();
        const kept = s.editFeatures.features.filter(
          (f) => !filter((f.properties?._et ?? '') as string)
        );
        if (newFeature) {
          kept.push({ type: 'Feature', geometry: newFeature.geometry, properties: { _et: newFeature.editType } });
        }
        const next = { type: 'FeatureCollection' as const, features: kept };
        s.setEditFeatures(next);
        const editSrc = mapRef.current?.getSource('proximity-edits') as import('maplibre-gl').GeoJSONSource | undefined;
        editSrc?.setData(next);
      },
      recordAddedNode: (edit) => changeset.recordAddedNode(edit),
      recordMovedEndpoint: (edit) => changeset.recordMovedEndpoint(edit),
      recordDrawnCrosswalk: (edit) => changeset.recordDrawnCrosswalk(edit),
      recordEditedHull: (edit) => changeset.recordEditedHull(edit),
      mutateSource: (mutation: SourceMutation) => {
        const map = mapRef.current;
        if (!map) return;
        const store = useEditorStore.getState();
        const fc = store.features;
        if (!fc) return;

        // Snapshot for undo before mutating
        store.pushSnapshot();

        if (mutation.type === 'delete') {
          fc.features = fc.features.filter((f) => {
            const val = f.properties?.[mutation.matchKey];
            return val !== mutation.matchId;
          });
        } else if (mutation.type === 'add') {
          fc.features.push(mutation.feature);
        } else if (mutation.type === 'updateCoords') {
          const feat = fc.features.find((f) => {
            const id = f.properties?.[mutation.matchKey];
            return id === mutation.matchId && f.geometry.type === 'LineString';
          });
          if (feat && feat.geometry.type === 'LineString') {
            feat.geometry.coordinates[mutation.vertexIndex] = mutation.coords;
          }
        }

        // Push to map sources
        const src = map.getSource('proximity-features') as import('maplibre-gl').GeoJSONSource | undefined;
        if (src) src.setData(fc);
        const vertSrc = map.getSource('proximity-vertices') as import('maplibre-gl').GeoJSONSource | undefined;
        if (vertSrc) vertSrc.setData(extractVertices(fc, store.selectedFeatureId));
      },
      recordDeletedPoint: (edit) => changeset.recordDeletedPoint(edit),
      recordDeletedHull: (edit) => changeset.recordDeletedHull(edit),
      recordMergedSegments: (edit) => changeset.recordMergedSegments(edit),
      recordDrawnSegment: (edit) => changeset.recordDrawnSegment(edit),
    };
  }, []);

  // === Platform-specific: Escape + Enter keys ===
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        const store = useEditorStore.getState();
        if (store.activeTool) {
          store.setActiveTool(null);
          store.setStatusText('');
          store.setEditFeatures({ type: 'FeatureCollection', features: [] });
        }
      }
      if (e.key === 'Enter') {
        const tool = useEditorStore.getState().activeTool;
        if (!tool) return;
        const cb = buildCallbacks();
        const st = useEditorStore.getState().activeSubtype;
        handleToolFinish(tool, cb, sessionRef.current, st);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [buildCallbacks]);

  // Main click handler — delegates to shared tool logic
  const handleToolClick = useCallback(
    (e: maplibregl.MapMouseEvent) => {
      const tool = useEditorStore.getState().activeTool;
      if (!tool) return;

      // Skip click handling for drag-based tools (handled by useVertexDrag)
      if (tool === 'move_point') return;

      const map = mapRef.current;
      if (!map) return;

      // ── Vertex-aware tools: delete_point and add_node ──────────────────
      const pad = 10;
      const ptBbox: [maplibregl.PointLike, maplibregl.PointLike] = [
        [e.point.x - pad, e.point.y - pad],
        [e.point.x + pad, e.point.y + pad],
      ];

      if (tool === 'delete_point') {
        // Check if click hit a visible line vertex
        const vertHits = map.queryRenderedFeatures(ptBbox, { layers: VERTEX_LAYERS });
        const lineVert = vertHits.find((h) => (h.properties?._vi as number) >= 0);
        if (lineVert) {
          const vi: number = lineVert.properties?._vi as number;
          const srcId: string = lineVert.properties?._src_id as string;
          const store = useEditorStore.getState();
          const fc = store.features;
          if (!fc) return;
          // Find the line feature and remove the vertex
          const feat = fc.features.find((f) => {
            const id = f.properties?._seg_id ?? f.properties?._fid;
            return id === srcId && f.geometry.type === 'LineString';
          });
          if (feat && feat.geometry.type === 'LineString' && feat.geometry.coordinates.length > 2) {
            store.pushSnapshot();
            feat.geometry.coordinates.splice(vi, 1);
            const featuresSrc = map.getSource('proximity-features') as import('maplibre-gl').GeoJSONSource | undefined;
            featuresSrc?.setData(fc);
            const vertSrc = map.getSource('proximity-vertices') as import('maplibre-gl').GeoJSONSource | undefined;
            vertSrc?.setData(extractVertices(fc, store.selectedFeatureId));
            store.setStatusText(`Removed vertex ${vi} from ${srcId}`);
            useChangesetStore.getState().recordMovedEndpoint({
              node_id: `${srcId}:deleted_v${vi}`,
              new_x: 0, new_y: 0,
              rubber_band_segments: [srcId],
            });
          }
          return;
        }
        // Fall through to normal delete_point logic (intersection nodes, points)
      }

      if (tool === 'add_node') {
        const store = useEditorStore.getState();
        const selectedId = store.selectedFeatureId;
        if (selectedId) {
          const fc = store.features;
          const feat = fc?.features.find((f) => {
            const id = f.properties?._seg_id ?? f.properties?._fid;
            return id === selectedId && f.geometry.type === 'LineString';
          });
          if (feat && feat.geometry.type === 'LineString') {
            // Find the closest segment of the line and insert vertex there
            const coords = feat.geometry.coordinates as [number, number][];
            const px = e.lngLat.lng;
            const py = e.lngLat.lat;
            let bestSeg = 0;
            let bestDist = Infinity;
            for (let i = 0; i < coords.length - 1; i++) {
              const [ax, ay] = coords[i];
              const [bx, by] = coords[i + 1];
              const dx = bx - ax; const dy = by - ay;
              const len2 = dx * dx + dy * dy;
              if (len2 === 0) continue;
              const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
              const nx = ax + t * dx; const ny = ay + t * dy;
              const d = (px - nx) ** 2 + (py - ny) ** 2;
              if (d < bestDist) { bestDist = d; bestSeg = i; }
            }
            store.pushSnapshot();
            coords.splice(bestSeg + 1, 0, [px, py]);
            const featuresSrc = map.getSource('proximity-features') as import('maplibre-gl').GeoJSONSource | undefined;
            featuresSrc?.setData(fc!);
            const vertSrc = map.getSource('proximity-vertices') as import('maplibre-gl').GeoJSONSource | undefined;
            vertSrc?.setData(extractVertices(fc!, selectedId));
            store.setStatusText(`Added vertex at position ${bestSeg + 1} on ${selectedId}`);
            return;
          }
        }
        // Fall through to normal add_node logic
      }

      // Lazily create/update adapter
      if (!adapterRef.current) {
        adapterRef.current = createMapLibreAdapter(map);
      }

      const tap = { lng: e.lngLat.lng, lat: e.lngLat.lat };
      const cb = buildCallbacks();

      const st = useEditorStore.getState().activeSubtype;
      handleToolTap(tool, tap, adapterRef.current, cb, sessionRef.current, st);
    },
    [mapRef, buildCallbacks]
  );

  // Invalidate adapter when map instance changes
  useEffect(() => {
    adapterRef.current = null;
  }, [mapRef.current]);

  // === Hull vertex drag for edit_polygon_face ===
  // Runs independently of the generic vertex drag system (useVertexDrag), because hull
  // vertices live in the edit-preview source (proximity-edits), not proximity-features.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || activeTool !== 'edit_polygon_face') return;

    let draggingVi: number | null = null;

    const hitTestHandle = (e: maplibregl.MapMouseEvent): number | null => {
      const bbox: [maplibregl.PointLike, maplibregl.PointLike] = [
        [e.point.x - 10, e.point.y - 10],
        [e.point.x + 10, e.point.y + 10],
      ];
      const hits = map.queryRenderedFeatures(bbox, { layers: ['px-edits-vertex-handle'] });
      if (hits.length === 0) return null;
      const vi = hits[0].properties?._vi;
      return typeof vi === 'number' ? vi : null;
    };

    const redrawHullPreview = (verts: [number, number][]) => {
      const s = useEditorStore.getState();
      const kept = s.editFeatures.features.filter(
        (f) => f.properties?._et !== 'vertex-handle' && f.properties?._et !== 'polygon-preview'
      );
      for (let i = 0; i < verts.length; i++) {
        kept.push({ type: 'Feature', geometry: { type: 'Point', coordinates: verts[i] }, properties: { _et: 'vertex-handle', _vi: i } });
      }
      const ring = [...verts, verts[0]];
      kept.push({ type: 'Feature', geometry: { type: 'Polygon', coordinates: [ring] }, properties: { _et: 'polygon-preview' } });
      const next = { type: 'FeatureCollection' as const, features: kept };
      s.setEditFeatures(next);
      const editSrc = map.getSource('proximity-edits') as import('maplibre-gl').GeoJSONSource | undefined;
      editSrc?.setData(next);
    };

    const onMouseDown = (e: maplibregl.MapMouseEvent) => {
      const session = sessionRef.current;
      if (!session.editingHullId || !session.editingHullVertices) return;
      const vi = hitTestHandle(e);
      if (vi === null) return;
      e.preventDefault();
      draggingVi = vi;
      map.getCanvas().style.cursor = 'grabbing';
      map.dragPan.disable();
    };

    const onMouseMove = (e: maplibregl.MapMouseEvent) => {
      const session = sessionRef.current;
      if (draggingVi === null) {
        if (session.editingHullId) {
          const vi = hitTestHandle(e);
          map.getCanvas().style.cursor = vi !== null ? 'grab' : 'crosshair';
        }
        return;
      }
      if (!session.editingHullVertices) return;
      session.editingHullVertices[draggingVi] = [e.lngLat.lng, e.lngLat.lat];
      redrawHullPreview(session.editingHullVertices);
    };

    const onMouseUp = () => {
      if (draggingVi === null) return;
      draggingVi = null;
      map.dragPan.enable();
      map.getCanvas().style.cursor = 'crosshair';
    };

    map.on('mousedown', onMouseDown);
    map.on('mousemove', onMouseMove);
    map.on('mouseup', onMouseUp);

    return () => {
      map.off('mousedown', onMouseDown);
      map.off('mousemove', onMouseMove);
      map.off('mouseup', onMouseUp);
    };
  }, [activeTool, mapRef]); // eslint-disable-line react-hooks/exhaustive-deps

  return { handleToolClick };
}
