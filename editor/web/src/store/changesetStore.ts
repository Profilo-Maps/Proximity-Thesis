import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  Edits,
  MovedEndpoint,
  AddedNode,
  ToggledCurbRamp,
  DrawnCrosswalk,
  EditedHull,
  AttrEdit,
  ConsolidatedHull,
  DeletedPoint,
  DeletedHull,
  MergedSegments,
  DrawnSegment,
  PipelineStage,
  BBox,
} from '@proximity/shared/types/changeset';
import { EDIT_STAGE_MAP } from '@proximity/shared/types/changeset';

interface EditEntry {
  type: string;
  label: string;
  timestamp: number;
  /** Key in Edits object this entry was appended to */
  editsKey: keyof Edits;
}

interface ChangesetState {
  parquet: string;
  dirtyFromStage: PipelineStage | null;
  edits: Edits;
  history: EditEntry[];

  setParquet: (p: string) => void;
  recordMovedEndpoint: (edit: MovedEndpoint) => void;
  recordAddedNode: (edit: AddedNode) => void;
  recordToggledRamp: (edit: ToggledCurbRamp) => void;
  recordDrawnCrosswalk: (edit: DrawnCrosswalk) => void;
  recordEditedHull: (edit: EditedHull) => void;
  recordAttrEdit: (edit: AttrEdit) => void;
  recordConsolidatedHull: (edit: ConsolidatedHull) => void;
  recordDeletedPoint: (edit: DeletedPoint) => void;
  recordDeletedHull: (edit: DeletedHull) => void;
  recordMergedSegments: (edit: MergedSegments) => void;
  recordDrawnSegment: (edit: DrawnSegment) => void;
  undo: () => void;
  hasEdits: () => boolean;
  clear: () => void;
  computeBBox: () => BBox | null;
}

const EMPTY_EDITS: Edits = {
  moved_endpoints: [],
  added_nodes: [],
  toggled_curb_ramps: [],
  drawn_crosswalks: [],
  edited_hulls: [],
  attr_edits: [],
  consolidated_hulls: [],
  deleted_points: [],
  deleted_hulls: [],
  merged_segments: [],
  drawn_segments: [],
};

function updateStage(current: PipelineStage | null, editType: string): PipelineStage {
  const stage = EDIT_STAGE_MAP[editType] ?? 99;
  if (current === null) return stage as PipelineStage;
  return Math.min(current, stage) as PipelineStage;
}

export const useChangesetStore = create<ChangesetState>()(
  persist(
    (set, get) => ({
      parquet: '',
      dirtyFromStage: null,
      edits: { ...EMPTY_EDITS },
      history: [],

      setParquet: (p) => set({ parquet: p }),

      recordMovedEndpoint: (edit) =>
        set((s) => ({
          edits: { ...s.edits, moved_endpoints: [...s.edits.moved_endpoints, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'move_point'),
          history: [...s.history, { type: 'move_point', label: `Move node ${edit.node_id}`, timestamp: Date.now(), editsKey: 'moved_endpoints' }],
        })),

      recordAddedNode: (edit) =>
        set((s) => ({
          edits: { ...s.edits, added_nodes: [...s.edits.added_nodes, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'add_point'),
          history: [...s.history, { type: 'add_point', label: `Add node at ${edit.x.toFixed(5)}, ${edit.y.toFixed(5)}`, timestamp: Date.now(), editsKey: 'added_nodes' }],
        })),

      recordToggledRamp: (edit) =>
        set((s) => ({
          edits: { ...s.edits, toggled_curb_ramps: [...s.edits.toggled_curb_ramps, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'toggle_curb_ramp'),
          history: [...s.history, { type: 'toggle_curb_ramp', label: `${edit.enabled ? 'Enable' : 'Disable'} ramp ${edit.side}/${edit.position}/${edit.index}`, timestamp: Date.now(), editsKey: 'toggled_curb_ramps' }],
        })),

      recordDrawnCrosswalk: (edit) =>
        set((s) => ({
          edits: { ...s.edits, drawn_crosswalks: [...s.edits.drawn_crosswalks, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'draw_crosswalk'),
          history: [...s.history, { type: 'draw_crosswalk', label: `Crosswalk ${edit.ramp_a.segment_id} → ${edit.ramp_b.segment_id}`, timestamp: Date.now(), editsKey: 'drawn_crosswalks' }],
        })),

      recordEditedHull: (edit) =>
        set((s) => ({
          edits: { ...s.edits, edited_hulls: [...s.edits.edited_hulls, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'edit_polygon_face'),
          history: [...s.history, { type: 'edit_polygon_face', label: `Edit hull ${edit.node_id}`, timestamp: Date.now(), editsKey: 'edited_hulls' }],
        })),

      recordAttrEdit: (edit) =>
        set((s) => ({
          edits: { ...s.edits, attr_edits: [...s.edits.attr_edits, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'attr_edit'),
          history: [...s.history, { type: 'attr_edit', label: `${edit.col} = ${edit.new_value}`, timestamp: Date.now(), editsKey: 'attr_edits' }],
        })),

      recordConsolidatedHull: (edit) =>
        set((s) => ({
          edits: { ...s.edits, consolidated_hulls: [...s.edits.consolidated_hulls, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'consolidate_hulls'),
          history: [...s.history, { type: 'consolidate_hulls', label: `Consolidate ${edit.node_keys.length} nodes`, timestamp: Date.now(), editsKey: 'consolidated_hulls' }],
        })),

      recordDeletedPoint: (edit) =>
        set((s) => ({
          edits: { ...s.edits, deleted_points: [...s.edits.deleted_points, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'delete_point'),
          history: [...s.history, { type: 'delete_point', label: `Delete node ${edit.node_id}`, timestamp: Date.now(), editsKey: 'deleted_points' }],
        })),

      recordDeletedHull: (edit) =>
        set((s) => ({
          edits: { ...s.edits, deleted_hulls: [...s.edits.deleted_hulls, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'delete_polygon'),
          history: [...s.history, { type: 'delete_polygon', label: `Delete hull ${edit.node_id}`, timestamp: Date.now(), editsKey: 'deleted_hulls' }],
        })),

      recordMergedSegments: (edit) =>
        set((s) => ({
          edits: { ...s.edits, merged_segments: [...s.edits.merged_segments, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'merge_segments'),
          history: [...s.history, { type: 'merge_segments', label: `Merge ${edit.consumed_seg_id} into ${edit.surviving_seg_id}`, timestamp: Date.now(), editsKey: 'merged_segments' }],
        })),

      recordDrawnSegment: (edit) =>
        set((s) => ({
          edits: { ...s.edits, drawn_segments: [...s.edits.drawn_segments, edit] },
          dirtyFromStage: updateStage(s.dirtyFromStage, 'draw_segment'),
          history: [...s.history, { type: 'draw_segment', label: `Draw segment (${edit.coordinates.length} pts)`, timestamp: Date.now(), editsKey: 'drawn_segments' }],
        })),

      undo: () =>
        set((s) => {
          if (s.history.length === 0) return s;
          const last = s.history[s.history.length - 1];
          const key = last.editsKey;
          const arr = s.edits[key] as unknown[];
          return {
            edits: { ...s.edits, [key]: arr.slice(0, -1) },
            history: s.history.slice(0, -1),
          };
        }),

      hasEdits: () => {
        const e = get().edits;
        return Object.values(e).some((arr) => arr.length > 0);
      },

      clear: () =>
        set({
          dirtyFromStage: null,
          edits: { ...EMPTY_EDITS },
          history: [],
        }),

      computeBBox: () => {
        const e = get().edits;
        const coords: [number, number][] = [];
        for (const m of e.moved_endpoints) { coords.push([m.new_x, m.new_y]); }
        for (const n of e.added_nodes) { coords.push([n.x, n.y]); }
        for (const h of e.edited_hulls) {
          for (const ring of h.geometry.coordinates) {
            for (const v of ring) coords.push(v);
          }
        }
        for (const c of e.consolidated_hulls) {
          for (const k of c.node_keys) coords.push(k);
        }
        for (const d of e.deleted_points) { coords.push([d.x, d.y]); }
        for (const ds of e.drawn_segments) {
          for (const c of ds.coordinates) coords.push(c);
        }
        if (coords.length === 0) return null;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const [x, y] of coords) {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
        return { minX, minY, maxX, maxY };
      },
    }),
    { name: 'px-editor-changeset' }
  )
);
