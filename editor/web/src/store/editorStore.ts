import { create } from 'zustand';
import type { Feature, FeatureCollection } from 'geojson';
import type { ToolSubtype } from '@proximity/shared/tools';
import { TOOL_SUBTYPES } from '@proximity/shared/tools';

export type ToolId =
  | 'move_point' | 'add_node' | 'delete_point'
  | 'highlight_point' | 'merge_segments' | 'split_segment' | 'draw_segment'
  | 'add_polygon' | 'delete_polygon' | 'edit_polygon_face' | null;

export type ProbeMode = 'inspect' | 'attribute-insert';

export interface LayerVisibility {
  streets: boolean;
  bikesSep: boolean;
  bikesOff: boolean;
  sidewalksSep: boolean;
  sidewalksOff: boolean;
  crosswalks: boolean;
  curbReturns: boolean;
  hulls: boolean;
  slots: boolean;
  nodes: boolean;
  ramps: boolean;
  calming: boolean;
}

interface EditorState {
  // Data
  parquet: string;
  setParquet: (p: string) => void;

  // Tool state
  activeTool: ToolId;
  setActiveTool: (t: ToolId) => void;
  activeSubtype: ToolSubtype | null;
  setActiveSubtype: (s: ToolSubtype) => void;

  // Probe mode — mutually exclusive with activeTool
  probeActive: boolean;
  setProbeActive: (active: boolean) => void;
  probeOrigin: [number, number] | null;
  setProbeOrigin: (origin: [number, number] | null) => void;

  // Selection
  selectedFeatureId: string | null;
  setSelectedFeatureId: (id: string | null) => void;

  // Probe
  probeMode: ProbeMode;
  setProbeMode: (mode: ProbeMode) => void;

  // Map viewport
  zoom: number;
  setZoom: (z: number) => void;
  center: [number, number];
  setCenter: (c: [number, number]) => void;

  // Layer visibility
  layers: LayerVisibility;
  toggleLayer: (layer: keyof LayerVisibility) => void;

  // Features from server
  features: FeatureCollection | null;
  setFeatures: (fc: FeatureCollection | null) => void;

  // Feature snapshots for undo (stack of JSON strings)
  featureSnapshots: string[];
  pushSnapshot: () => void;
  popSnapshot: () => FeatureCollection | null;

  // Hulls and slots
  hulls: Feature[];
  slots: Feature[];
  setHullsAndSlots: (hulls: Feature[], slots: Feature[]) => void;

  // Edit overlay
  editFeatures: FeatureCollection;
  setEditFeatures: (fc: FeatureCollection) => void;

  // Slide panels
  searchOpen: boolean;
  toggleSearch: () => void;
  configOpen: boolean;
  toggleConfig: () => void;

  // Status text
  statusText: string;
  setStatusText: (t: string) => void;

  // Hidden features
  hiddenFeatures: Set<string>;
  toggleFeatureVisibility: (id: string) => void;

  // Bottom attribute table
  bottomExpanded: boolean;
  setBottomExpanded: (v: boolean) => void;

  // Right panel collapse state
  probeMinimized: boolean;
  setProbeMinimized: (v: boolean) => void;
  historyMinimized: boolean;
  setHistoryMinimized: (v: boolean) => void;
}

const DEFAULT_LAYERS: LayerVisibility = {
  streets: true,
  bikesSep: true,
  bikesOff: true,
  sidewalksSep: true,
  sidewalksOff: true,
  crosswalks: true,
  curbReturns: true,
  hulls: true,
  slots: true,
  nodes: true,
  ramps: true,
  calming: true,
};

export const useEditorStore = create<EditorState>((set, get) => ({
  parquet: '',
  setParquet: (p) => set({ parquet: p }),

  activeTool: null,
  setActiveTool: (t) => set((s) => {
    if (s.activeTool === t) return { activeTool: null, activeSubtype: null };
    const subtypeDef = t ? TOOL_SUBTYPES[t] : undefined;
    // Activating a tool disables probe mode
    return { activeTool: t, activeSubtype: subtypeDef?.default ?? null, probeActive: false, probeOrigin: null };
  }),
  activeSubtype: null,
  setActiveSubtype: (s) => set({ activeSubtype: s }),

  probeActive: false,
  setProbeActive: (active) => set((s) => {
    if (active === s.probeActive) return s;
    if (active) {
      // Activating probe disables any active tool
      return { probeActive: true, activeTool: null, activeSubtype: null, probeOrigin: null };
    }
    return { probeActive: false, probeOrigin: null };
  }),
  probeOrigin: null,
  setProbeOrigin: (origin) => set({ probeOrigin: origin }),

  selectedFeatureId: null,
  setSelectedFeatureId: (id) => set({ selectedFeatureId: id }),

  probeMode: 'inspect',
  setProbeMode: (mode) => set({ probeMode: mode }),

  zoom: 15,
  setZoom: (z) => set({ zoom: z }),
  center: [-122.42, 37.77],
  setCenter: (c) => set({ center: c }),

  layers: { ...DEFAULT_LAYERS },
  toggleLayer: (layer) =>
    set((s) => ({ layers: { ...s.layers, [layer]: !s.layers[layer] } })),

  features: null,
  setFeatures: (fc) => set({ features: fc }),

  featureSnapshots: [],
  pushSnapshot: () => set((s) => {
    if (!s.features) return s;
    const snap = JSON.stringify(s.features);
    // Keep max 50 snapshots
    const snaps = [...s.featureSnapshots, snap].slice(-50);
    return { featureSnapshots: snaps };
  }),
  popSnapshot: (): FeatureCollection | null => {
    const s = get();
    if (s.featureSnapshots.length === 0) return null;
    const last = s.featureSnapshots[s.featureSnapshots.length - 1];
    set({ featureSnapshots: s.featureSnapshots.slice(0, -1) });
    return JSON.parse(last) as FeatureCollection;
  },

  hulls: [],
  slots: [],
  setHullsAndSlots: (hulls, slots) => set({ hulls, slots }),

  editFeatures: { type: 'FeatureCollection', features: [] },
  setEditFeatures: (fc) => set({ editFeatures: fc }),

  searchOpen: false,
  toggleSearch: () => set((s) => ({ searchOpen: !s.searchOpen, configOpen: false })),
  configOpen: false,
  toggleConfig: () => set((s) => ({ configOpen: !s.configOpen, searchOpen: false })),

  statusText: '',
  setStatusText: (t) => set({ statusText: t }),

  hiddenFeatures: new Set(),
  toggleFeatureVisibility: (id) =>
    set((s) => {
      const next = new Set(s.hiddenFeatures);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { hiddenFeatures: next };
    }),

  bottomExpanded: false,
  setBottomExpanded: (v) => set({ bottomExpanded: v }),

  probeMinimized: false,
  setProbeMinimized: (v) => set({ probeMinimized: v }),
  historyMinimized: false,
  setHistoryMinimized: (v) => set({ historyMinimized: v }),
}));
