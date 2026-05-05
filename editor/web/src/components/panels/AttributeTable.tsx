import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useEditorStore } from '../../store/editorStore';
import { useChangesetStore } from '../../store/changesetStore';
import { fetchRows } from '../../api/editorApi';

interface AttributeTableProps {
  expanded: boolean;
  onToggle: () => void;
}

/** All columns shown in the full table view */
const ALL_COLUMNS = [
  'street_grid_id', '_node_id', '_seg_id', '_fid', '_t',
  'name', 'highway', 'osmid', 'surface',
  'lanes', 'lane_width', 'maxspeed', 'oneway', 'street_incline', 'length', 'access',
  'normalized_bearing',
  'start_node_id', 'start_node_is_intersection_node',
  'end_node_id', 'end_node_is_intersection_node',
  'crosswalk_start_type', 'crosswalk_end_type', 'crosswalk_start_marked', 'crosswalk_end_marked',
  '_side', '_position',
  'is_intersection', 'node_type',
  'accessible',
];

const READ_ONLY = new Set([
  'street_grid_id', '_node_id', '_seg_id', '_fid', '_t',
  'osmid', 'maxspeed', 'length', 'normalized_bearing', 'street_incline',
  'start_node_id', 'start_node_is_intersection_node',
  'end_node_id', 'end_node_is_intersection_node',
]);

const TYPE_HIGHLIGHT: Record<string, Set<string>> = {
  street:    new Set(['street_grid_id', 'name', 'highway', 'lanes', 'lane_width', 'surface', 'maxspeed', 'oneway', 'street_incline']),
  bikeway:   new Set(['_seg_id', 'name', 'highway', 'surface']),
  sidewalk:  new Set(['_seg_id', '_side', 'surface']),
  crosswalk: new Set(['_seg_id', 'crosswalk_start_type', 'crosswalk_end_type', 'crosswalk_start_marked', 'crosswalk_end_marked']),
  cret:      new Set(['_seg_id']),
  node:      new Set(['_node_id', 'is_intersection', 'node_type']),
  ramp:      new Set(['_fid', '_side', '_position', 'accessible']),
  calm:      new Set(['_fid', '_t']),
};

const TYPE_ANCHOR: Record<string, string> = {
  street: 'street_grid_id', bikeway: '_seg_id', sidewalk: '_seg_id',
  crosswalk: '_seg_id', cret: '_seg_id', node: '_node_id', ramp: '_fid', calm: '_fid',
};

function rowId(row: Record<string, unknown>): string {
  return String(row.street_grid_id ?? row._node_id ?? row._seg_id ?? row._fid ?? '');
}

export function AttributeTable({ expanded, onToggle }: AttributeTableProps) {
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const parquet = useEditorStore((s) => s.parquet);
  const center = useEditorStore((s) => s.center);
  const selectedId = useEditorStore((s) => s.selectedFeatureId);
  const setSelectedFeatureId = useEditorStore((s) => s.setSelectedFeatureId);
  const features = useEditorStore((s) => s.features);
  const recordAttrEdit = useChangesetStore((s) => s.recordAttrEdit);

  const rowRefs = useRef<Map<string, HTMLTableRowElement>>(new Map());
  const thRefs = useRef<Map<string, HTMLTableCellElement>>(new Map());
  const bodyRef = useRef<HTMLDivElement>(null);

  // Fetch server rows when expanded
  useEffect(() => {
    if (!expanded || !parquet) return;
    const delta = 0.005;
    const bbox: [number, number, number, number] = [
      center[0] - delta, center[1] - delta,
      center[0] + delta, center[1] + delta,
    ];
    fetchRows(parquet, bbox)
      .then(setRows)
      .catch((err) => console.error('[AttributeTable] fetch failed:', err));
  }, [expanded, parquet, center]);

  // Augment server rows with non-street GeoJSON features from store
  const allRows = useMemo(() => {
    const nonStreet: Record<string, unknown>[] = [];
    if (features) {
      for (const f of features.features) {
        const t = f.properties?._t;
        if (!t || t === 'street') continue;
        nonStreet.push({ ...f.properties });
      }
    }
    const seen = new Set<string>();
    const merged: Record<string, unknown>[] = [];
    for (const r of [...rows, ...nonStreet]) {
      const id = rowId(r);
      if (!seen.has(id)) { seen.add(id); merged.push(r); }
    }
    return merged;
  }, [rows, features]);

  // Scroll selected row + anchor column into view
  useEffect(() => {
    if (!selectedId || !expanded) return;
    const tr = rowRefs.current.get(selectedId);
    if (!tr) return;
    tr.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    const row = allRows.find((r) => rowId(r) === selectedId);
    const featureType = String(row?._t ?? 'street');
    const anchor = TYPE_ANCHOR[featureType] ?? ALL_COLUMNS[0];
    const th = thRefs.current.get(anchor);
    if (th && bodyRef.current) bodyRef.current.scrollLeft = th.offsetLeft;
  }, [selectedId, expanded, allRows]);

  const handleCellEdit = useCallback((id: string, col: string, oldValue: unknown, newValue: string) => {
    if (String(oldValue) === newValue) return;
    recordAttrEdit({
      street_grid_id: id,
      col,
      old_value: oldValue == null ? null : String(oldValue),
      new_value: newValue,
    });
  }, [recordAttrEdit]);

  return (
    <div className="px-bottom-strip">
      <div className="px-bottom-strip-header" onClick={onToggle}>
        <span>{allRows.length} rows{selectedId ? ` · ${selectedId}` : ''}</span>
        <span>{expanded ? '\u25BC' : '\u25B2'}</span>
      </div>
      <div className="px-bottom-strip-body" ref={bodyRef}>
        <table>
          <thead>
            <tr>
              {ALL_COLUMNS.map((col) => (
                <th key={col} ref={(el) => { if (el) thRefs.current.set(col, el); else thRefs.current.delete(col); }}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {allRows.map((row, i) => {
              const id = rowId(row);
              const featureType = String(row._t ?? 'street');
              const highlight = TYPE_HIGHLIGHT[featureType] ?? new Set<string>();
              const isSelected = selectedId === id;
              return (
                <tr
                  key={id || i}
                  ref={(el) => { if (el) rowRefs.current.set(id, el); else rowRefs.current.delete(id); }}
                  className={isSelected ? 'selected' : ''}
                  onClick={() => setSelectedFeatureId(id)}
                >
                  {ALL_COLUMNS.map((col) => {
                    const val = row[col];
                    const isHighlighted = isSelected && highlight.has(col);
                    const isReadOnly = READ_ONLY.has(col);
                    return (
                      <td
                        key={col}
                        style={isHighlighted ? { background: 'rgba(0,212,255,0.15)', outline: '1px solid rgba(0,212,255,0.4)' } : undefined}
                      >
                        {isReadOnly ? (
                          <span>{String(val ?? '')}</span>
                        ) : (
                          <input
                            defaultValue={String(val ?? '')}
                            onBlur={(e) => handleCellEdit(id, col, val, e.target.value)}
                          />
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
