import type { LayerSpecification } from 'maplibre-gl';

// Layer colors — must match server _FC dict in editor_server.py
const COLORS = {
  street: '#c0392b',
  bikewaySep: '#006400',
  bikewayOff: '#90ee90',
  sidewalkSep: '#00008b',
  sidewalkOff: '#add8e6',
  crosswalk: '#ff6b81',
  curbReturn: '#a55eea',
  hull: 'rgba(0, 212, 255, 0.15)',
  hullLine: '#00d4ff',
  slot: 'rgba(255, 107, 129, 0.2)',
  slotLine: '#ff6b81',
  node: '#e0e0e0',
  ramp: '#ff4757',
  calming: '#8b5cf6',
  editAdded: '#2ed573',
  editMoved: '#ffa502',
  editDeleted: '#ff4757',
  editGhost: 'rgba(255, 255, 255, 0.3)',
  highlight: '#00d4ff',
};

export const LAYER_DEFS: LayerSpecification[] = [
  // Streets
  {
    id: 'px-streets',
    type: 'line',
    source: 'proximity-features',
    filter: ['==', ['get', '_t'], 'street'],
    paint: {
      'line-color': ['coalesce', ['get', '_color'], COLORS.street],
      'line-width': ['interpolate', ['linear'], ['zoom'], 12, 1.5, 18, 4],
      'line-opacity': 0.9,
    },
  },
  // Bikeways — separate geometry
  {
    id: 'px-bk-sep',
    type: 'line',
    source: 'proximity-features',
    filter: ['all', ['==', ['get', '_t'], 'bikeway'], ['!=', ['get', '_off'], 'yes']],
    minzoom: 13,
    paint: {
      'line-color': ['coalesce', ['get', '_color'], COLORS.bikewaySep],
      'line-width': 2,
    },
  },
  // Bikeways — buffered (offset) geometry
  {
    id: 'px-bk-off',
    type: 'line',
    source: 'proximity-features',
    filter: ['all', ['==', ['get', '_t'], 'bikeway'], ['==', ['get', '_off'], 'yes']],
    minzoom: 13,
    paint: {
      'line-color': ['coalesce', ['get', '_color'], COLORS.bikewayOff],
      'line-width': 2,
    },
  },
  // Sidewalks — separate geometry
  {
    id: 'px-sw-sep',
    type: 'line',
    source: 'proximity-features',
    filter: ['all', ['==', ['get', '_t'], 'sidewalk'], ['!=', ['get', '_off'], 'yes']],
    minzoom: 15,
    paint: {
      'line-color': ['coalesce', ['get', '_color'], COLORS.sidewalkSep],
      'line-width': 1.5,
    },
  },
  // Sidewalks — buffered (offset) geometry
  {
    id: 'px-sw-off',
    type: 'line',
    source: 'proximity-features',
    filter: ['all', ['==', ['get', '_t'], 'sidewalk'], ['==', ['get', '_off'], 'yes']],
    minzoom: 15,
    paint: {
      'line-color': ['coalesce', ['get', '_color'], COLORS.sidewalkOff],
      'line-width': 1.5,
    },
  },
  // Crosswalks
  {
    id: 'px-crosswalks',
    type: 'line',
    source: 'proximity-features',
    filter: ['==', ['get', '_t'], 'crosswalk'],
    minzoom: 15,
    paint: {
      'line-color': COLORS.crosswalk,
      'line-width': 3,
      'line-dasharray': [1, 1],
    },
  },
  // Curb returns
  {
    id: 'px-curb-returns',
    type: 'line',
    source: 'proximity-features',
    filter: ['==', ['get', '_t'], 'cret'],
    minzoom: 15,
    paint: {
      'line-color': COLORS.curbReturn,
      'line-width': 2,
    },
  },
  // Hull fill
  {
    id: 'px-hull-fill',
    type: 'fill',
    source: 'proximity-hulls',
    minzoom: 17,
    paint: {
      'fill-color': COLORS.hull,
    },
  },
  // Hull line
  {
    id: 'px-hull-line',
    type: 'line',
    source: 'proximity-hulls',
    minzoom: 17,
    paint: {
      'line-color': COLORS.hullLine,
      'line-width': 1,
    },
  },
  // Slot fill
  {
    id: 'px-slot-fill',
    type: 'fill',
    source: 'proximity-slots',
    minzoom: 17,
    paint: {
      'fill-color': COLORS.slot,
    },
  },
  // Slot line
  {
    id: 'px-slot-line',
    type: 'line',
    source: 'proximity-slots',
    minzoom: 17,
    paint: {
      'line-color': COLORS.slotLine,
      'line-width': 1,
    },
  },
  // Traffic calming
  {
    id: 'px-calming',
    type: 'circle',
    source: 'proximity-features',
    filter: ['==', ['get', '_t'], 'calm'],
    minzoom: 15,
    paint: {
      'circle-radius': 4,
      'circle-color': COLORS.calming,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#fff',
    },
  },
  // Nodes
  {
    id: 'px-nodes',
    type: 'circle',
    source: 'proximity-features',
    filter: ['==', ['get', '_t'], 'node'],
    minzoom: 17,
    paint: {
      'circle-radius': 5,
      'circle-color': COLORS.node,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#333',
    },
  },
  // Curb ramps
  {
    id: 'px-ramps',
    type: 'circle',
    source: 'proximity-features',
    filter: ['==', ['get', '_t'], 'ramp'],
    minzoom: 18,
    paint: {
      'circle-radius': 4,
      'circle-color': COLORS.ramp,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#fff',
    },
  },
];

// Highlight layers — rendered on top of edit layers
export const HIGHLIGHT_LAYER_DEFS: LayerSpecification[] = [
  // Highlight (lines)
  {
    id: 'px-highlight',
    type: 'line',
    source: 'proximity-highlight',
    filter: ['!=', ['geometry-type'], 'Point'],
    paint: {
      'line-color': COLORS.highlight,
      'line-width': 6,
      'line-opacity': 0.6,
    },
  },
  // Highlight (points)
  {
    id: 'px-highlight-point',
    type: 'circle',
    source: 'proximity-highlight',
    filter: ['==', ['geometry-type'], 'Point'],
    paint: {
      'circle-radius': 10,
      'circle-color': 'transparent',
      'circle-stroke-width': 3,
      'circle-stroke-color': COLORS.highlight,
      'circle-opacity': 0.8,
    },
  },
];

// Vertex layer — all line vertices as draggable points
export const VERTEX_LAYER_DEFS: LayerSpecification[] = [
  // Mid-vertices (smaller, dimmer)
  {
    id: 'px-vertices-mid',
    type: 'circle',
    source: 'proximity-vertices',
    filter: ['==', ['get', '_endpoint'], 0],
    paint: {
      'circle-radius': 4,
      'circle-color': '#aaa',
      'circle-stroke-width': 1,
      'circle-stroke-color': '#333',
      'circle-opacity': 0.7,
    },
  },
  // Endpoints (larger, brighter)
  {
    id: 'px-vertices-end',
    type: 'circle',
    source: 'proximity-vertices',
    filter: ['==', ['get', '_endpoint'], 1],
    paint: {
      'circle-radius': 6,
      'circle-color': '#fff',
      'circle-stroke-width': 2,
      'circle-stroke-color': '#000',
    },
  },
];

export const EDIT_LAYER_DEFS: LayerSpecification[] = [
  // Added geometry (green)
  {
    id: 'px-edits-added-line',
    type: 'line',
    source: 'proximity-edits',
    filter: ['==', ['get', '_et'], 'added'],
    paint: { 'line-color': COLORS.editAdded, 'line-width': 3 },
  },
  {
    id: 'px-edits-added-point',
    type: 'circle',
    source: 'proximity-edits',
    filter: ['all', ['==', ['get', '_et'], 'node'], ['==', ['geometry-type'], 'Point']],
    paint: { 'circle-radius': 7, 'circle-color': COLORS.editAdded, 'circle-stroke-width': 2, 'circle-stroke-color': '#fff' },
  },
  // Moved geometry (yellow)
  {
    id: 'px-edits-moved',
    type: 'circle',
    source: 'proximity-edits',
    filter: ['==', ['get', '_et'], 'moved'],
    paint: { 'circle-radius': 7, 'circle-color': COLORS.editMoved, 'circle-stroke-width': 2, 'circle-stroke-color': '#fff' },
  },
  // Ghost (original position)
  {
    id: 'px-edits-ghost',
    type: 'circle',
    source: 'proximity-edits',
    filter: ['==', ['get', '_et'], 'ghost'],
    paint: { 'circle-radius': 5, 'circle-color': COLORS.editGhost, 'circle-stroke-width': 1, 'circle-stroke-color': '#666' },
  },
  // Deleted (red) — points
  {
    id: 'px-edits-deleted',
    type: 'circle',
    source: 'proximity-edits',
    filter: ['all', ['==', ['get', '_et'], 'deleted'], ['==', ['geometry-type'], 'Point']],
    paint: { 'circle-radius': 6, 'circle-color': COLORS.editDeleted, 'circle-stroke-width': 2, 'circle-stroke-color': '#fff' },
  },
  // Deleted (red) — lines
  {
    id: 'px-edits-deleted-line',
    type: 'line',
    source: 'proximity-edits',
    filter: ['all', ['==', ['get', '_et'], 'deleted'], ['!=', ['geometry-type'], 'Point']],
    paint: { 'line-color': COLORS.editDeleted, 'line-width': 3, 'line-dasharray': [2, 2] },
  },
  // Line preview (draw_segment)
  {
    id: 'px-edits-line-preview',
    type: 'line',
    source: 'proximity-edits',
    filter: ['==', ['get', '_et'], 'line-preview'],
    paint: { 'line-color': COLORS.editAdded, 'line-width': 3, 'line-dasharray': [4, 2] },
  },
  // Polygon preview fill (add_polygon / edit_polygon_face)
  {
    id: 'px-edits-polygon-preview',
    type: 'fill',
    source: 'proximity-edits',
    filter: ['==', ['get', '_et'], 'polygon-preview'],
    paint: { 'fill-color': 'rgba(0, 212, 255, 0.2)' },
  },
  // Polygon preview outline
  {
    id: 'px-edits-polygon-preview-line',
    type: 'line',
    source: 'proximity-edits',
    filter: ['==', ['get', '_et'], 'polygon-preview'],
    paint: { 'line-color': '#00d4ff', 'line-width': 2, 'line-dasharray': [4, 2] },
  },
  // Vertex handles for polygon editing
  {
    id: 'px-edits-vertex-handle',
    type: 'circle',
    source: 'proximity-edits',
    filter: ['==', ['get', '_et'], 'vertex-handle'],
    paint: { 'circle-radius': 6, 'circle-color': '#00d4ff', 'circle-stroke-width': 2, 'circle-stroke-color': '#fff' },
  },
];
