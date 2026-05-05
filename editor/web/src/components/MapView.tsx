import { useEffect, useRef, useCallback } from 'react';
import maplibregl from 'maplibre-gl';
import { useEditorStore, type LayerVisibility } from '../store/editorStore';
import { fetchFeatures, fetchHulls } from '../api/editorApi';
import { LAYER_DEFS, EDIT_LAYER_DEFS, HIGHLIGHT_LAYER_DEFS, VERTEX_LAYER_DEFS } from '../hooks/useMapLayers';
import { useToolHandler } from '../hooks/useToolHandler';
import { useVertexDrag } from '../hooks/useVertexDrag';
import type { MapMouseEvent } from 'maplibre-gl';

// Map layer IDs to store layer keys for visibility sync
const LAYER_KEY_MAP: Record<string, string> = {
  'px-streets': 'streets',
  'px-bk-sep': 'bikesSep',
  'px-bk-off': 'bikesOff',
  'px-sw-sep': 'sidewalksSep',
  'px-sw-off': 'sidewalksOff',
  'px-crosswalks': 'crosswalks',
  'px-curb-returns': 'curbReturns',
  'px-hull-fill': 'hulls',
  'px-hull-line': 'hulls',
  'px-slot-fill': 'slots',
  'px-slot-line': 'slots',
  'px-calming': 'calming',
  'px-nodes': 'nodes',
  'px-ramps': 'ramps',
};

export function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const debounceRef = useRef<number>(0);

  const parquet = useEditorStore((s) => s.parquet);
  const setZoom = useEditorStore((s) => s.setZoom);
  const setCenter = useEditorStore((s) => s.setCenter);
  const setFeatures = useEditorStore((s) => s.setFeatures);
  const setHullsAndSlots = useEditorStore((s) => s.setHullsAndSlots);
  const setSelectedFeatureId = useEditorStore((s) => s.setSelectedFeatureId);
  const statusText = useEditorStore((s) => s.statusText);
  const layers = useEditorStore((s) => s.layers);

  // Wire tool handler — use ref to avoid stale closure in map event listener
  const { handleToolClick } = useToolHandler(mapRef);
  const toolClickRef = useRef<(e: MapMouseEvent) => void>(handleToolClick);
  useEffect(() => { toolClickRef.current = handleToolClick; }, [handleToolClick]);

  // Wire vertex drag system
  useVertexDrag(mapRef);

  const refreshData = useCallback(async (map: maplibregl.Map) => {
    if (!parquet) return;
    const bounds = map.getBounds();
    const bbox: [number, number, number, number] = [
      bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth(),
    ];
    const zoom = Math.floor(map.getZoom());

    try {
      const fc = await fetchFeatures(parquet, bbox, zoom);
      setFeatures(fc);

      const source = map.getSource('proximity-features') as maplibregl.GeoJSONSource | undefined;
      if (source) source.setData(fc);

      if (zoom >= 17) {
        const { hulls, slots } = await fetchHulls(parquet, bbox);
        setHullsAndSlots(hulls, slots);

        const hullSource = map.getSource('proximity-hulls') as maplibregl.GeoJSONSource | undefined;
        if (hullSource) hullSource.setData({ type: 'FeatureCollection', features: hulls });

        const slotSource = map.getSource('proximity-slots') as maplibregl.GeoJSONSource | undefined;
        if (slotSource) slotSource.setData({ type: 'FeatureCollection', features: slots });
      }
    } catch (err) {
      console.error('[MapView] Failed to refresh data:', err);
    }
  }, [parquet, setFeatures, setHullsAndSlots]);

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          carto: {
            type: 'raster',
            tiles: [
              'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
              'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
              'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
            ],
            tileSize: 256,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
          },
        },
        layers: [{ id: 'carto', type: 'raster', source: 'carto' }],
      },
      center: [-122.42, 37.77],
      zoom: 15,
      maxZoom: 23,
    });

    map.on('load', () => {
      // Add empty sources
      map.addSource('proximity-features', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.addSource('proximity-hulls', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.addSource('proximity-slots', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.addSource('proximity-edits', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.addSource('proximity-highlight', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.addSource('proximity-vertices', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });

      // Add feature layers
      for (const layerDef of LAYER_DEFS) {
        map.addLayer(layerDef);
      }

      // Add edit overlay layers
      for (const layerDef of EDIT_LAYER_DEFS) {
        map.addLayer(layerDef);
      }

      // Add highlight layers (rendered on top of edits)
      for (const layerDef of HIGHLIGHT_LAYER_DEFS) {
        map.addLayer(layerDef);
      }

      // Add vertex layers (hidden by default, shown when vertex tool active)
      for (const layerDef of VERTEX_LAYER_DEFS) {
        map.addLayer(layerDef);
        map.setLayoutProperty(layerDef.id, 'visibility', 'none');
      }

      // Initial data load
      refreshData(map);
    });

    map.on('moveend', () => {
      setZoom(Math.floor(map.getZoom()));
      setCenter([map.getCenter().lng, map.getCenter().lat]);

      clearTimeout(debounceRef.current);
      debounceRef.current = window.setTimeout(() => refreshData(map), 300);
    });

    map.on('click', (e) => {
      const state = useEditorStore.getState();

      // Tool takes priority
      if (state.activeTool) {
        toolClickRef.current(e);
        return;
      }

      // Probe mode: record click origin for proximity sorting
      if (state.probeActive) {
        state.setProbeOrigin([e.lngLat.lng, e.lngLat.lat]);
        return;
      }

      // Default: select feature — use a 6px bbox so thin lines are reliably hit
      const pad = 6;
      const bbox: [maplibregl.PointLike, maplibregl.PointLike] = [
        [e.point.x - pad, e.point.y - pad],
        [e.point.x + pad, e.point.y + pad],
      ];
      const hits = map.queryRenderedFeatures(bbox, {
        layers: LAYER_DEFS.map((l) => l.id),
      }).filter((f) => f.properties?._t != null);

      const highlightSource = map.getSource('proximity-highlight') as maplibregl.GeoJSONSource | undefined;
      if (hits.length > 0) {
        const p = hits[0].properties!;
        const fid = (p._fid || p._seg_id || p._node_id || p.street_grid_id || null) as string | null;
        setSelectedFeatureId(fid);
        // Set highlight immediately from the rendered feature geometry
        if (highlightSource) {
          highlightSource.setData({ type: 'FeatureCollection', features: [hits[0]] });
        }
      } else {
        setSelectedFeatureId(null);
        if (highlightSource) {
          highlightSource.setData({ type: 'FeatureCollection', features: [] });
        }
      }
    });

    mapRef.current = map;

    return () => { map.remove(); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync layer visibility from store → map
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    for (const [layerId, storeKey] of Object.entries(LAYER_KEY_MAP)) {
      const visible = layers[storeKey as keyof LayerVisibility];
      try {
        map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
      } catch {
        // Layer may not exist yet during initial render
      }
    }
  }, [layers]);

  // Highlight selected feature on map
  // Resize MapLibre canvas when right panel collapses/expands
  const probeMinimized = useEditorStore((s) => s.probeMinimized);
  const historyMinimized = useEditorStore((s) => s.historyMinimized);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const t = setTimeout(() => map.resize(), 60);
    return () => clearTimeout(t);
  }, [probeMinimized, historyMinimized]);

  const selectedFeatureId = useEditorStore((s) => s.selectedFeatureId);
  const allFeatures = useEditorStore((s) => s.features);
  // Clear highlight when selection is cleared (set is handled directly in click handler)
  useEffect(() => {
    if (selectedFeatureId) return;
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    const source = map.getSource('proximity-highlight') as maplibregl.GeoJSONSource | undefined;
    if (source) source.setData({ type: 'FeatureCollection', features: [] });
  }, [selectedFeatureId]);

  // Sync features store → map source (for undo/real-time mutations)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded() || !allFeatures) return;
    const src = map.getSource('proximity-features') as maplibregl.GeoJSONSource | undefined;
    if (src) src.setData(allFeatures);
  }, [allFeatures]);

  // Update edit overlay when edits change
  const editFeatures = useEditorStore((s) => s.editFeatures);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    const source = map.getSource('proximity-edits') as maplibregl.GeoJSONSource | undefined;
    if (source) source.setData(editFeatures);
  }, [editFeatures]);

  // Re-fetch data when parquet changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !parquet) return;
    if (map.isStyleLoaded()) {
      refreshData(map);
    } else {
      map.once('load', () => refreshData(map));
    }
  }, [parquet, refreshData]);

  return (
    <>
      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />
      {statusText && <div className="px-status">{statusText}</div>}
    </>
  );
}
