import { useState, useMemo, useEffect, useCallback } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { useEditorStore } from '../../store/editorStore';
import { useChangesetStore } from '../../store/changesetStore';
import { nearestPointOnLine } from '@proximity/shared/tools/geometry';
import { fetchRows } from '../../api/editorApi';
import type { Feature } from 'geojson';

// ── Focus view helpers ─────────────────────────────────────────────────────

interface FocusAttr {
  label: string;
  value: unknown;
  readOnly: boolean;
  editKey?: { col: string; streetGridId: string };
}

function fmt(v: unknown): string {
  if (v == null || String(v).trim() === '' || String(v) === 'nan' || String(v) === 'None') return '—';
  return String(v);
}
function ro(label: string, value: unknown): FocusAttr {
  return { label, value, readOnly: true };
}
function rw(label: string, value: unknown, col: string, streetGridId: string): FocusAttr {
  return { label, value, readOnly: false, editKey: { col, streetGridId } };
}
function parseBkFid(fid: string): { side: string; n: string; segId: string } | null {
  const parts = fid.split('_');
  if (parts[0] !== 'bk' || parts.length < 4) return null;
  return { side: parts[1], n: parts[2], segId: parts.slice(3).join('_') };
}
function parseRampFid(fid: string): { side: string; pos: string; n: string; segId: string } | null {
  const parts = fid.split('_');
  if (parts.length < 4) return null;
  return { side: parts[0], pos: parts[1], n: parts[2], segId: parts.slice(3).join('_') };
}

function buildFocusView(
  feature: Feature,
  parentRow: Record<string, unknown> | null,
): { title: string; attrs: FocusAttr[] } {
  const p = feature.properties ?? {};
  const t = String(p._t ?? '');
  const segId = String(p._seg_id ?? '');
  const row = parentRow ?? {};

  if (t === 'street') {
    const name = fmt(row.name ?? p.name);
    return {
      title: `Street: ${name}`,
      attrs: [
        ro('Name',         row.name ?? p.name),
        ro('Highway',      row.highway ?? p.highway),
        ro('OSM ID',       row.osmid),
        ro('Grid ID',      row.street_grid_id ?? segId),
        ro('Bearing (°)',  row.normalized_bearing),
        rw('Lanes',        row.lanes, 'lanes', segId),
        rw('Lane width',   row.lane_width, 'lane_width', segId),
        ro('Max speed',    row.maxspeed),
        rw('Oneway',       row.oneway, 'oneway', segId),
        ro('Length (m)',   row.length),
        rw('Surface',      row.surface, 'surface', segId),
        ro('Incline',      row.street_incline),
        ro('Start node',   row.start_node_id),
        ro('End node',     row.end_node_id),
      ],
    };
  }
  if (t === 'sidewalk') {
    const s = String(p._side ?? 'left');
    const k = (col: string) => `sidewalk_${s}_${col}`;
    return {
      title: `Sidewalk (${s}) · ${fmt(row.name)}`,
      attrs: [
        ro('Side',      s),
        ro('Presence',  row[k('presence')]),
        rw('Width (m)', row[k('width')],   k('width'),   segId),
        rw('Surface',   row[k('surface')], k('surface'), segId),
        ro('Quality',   row[k('quality')]),
        ro('Incline',   row[k('incline')]),
        ro('Separator', row[k('seperator')]),
        ro('Kerb',      row[k('kerb')]),
        ro('Street',    row.name),
      ],
    };
  }
  if (t === 'bikeway') {
    const parsed = parseBkFid(String(p._fid ?? ''));
    const s = parsed?.side ?? String(p._side ?? 'left');
    const n = parsed?.n ?? '1';
    const k = (col: string) => `bikeway_${s}_${n}_${col}`;
    const kind = String(p._off ?? '') === 'yes' ? 'buffered' : 'separate';
    return {
      title: `Bikeway (${s}-${n}) [${kind}]`,
      attrs: [
        ro('Side',       s),
        ro('Lane #',     n),
        ro('Kind',       kind),
        rw('Type',       row[k('type')],    k('type'),    segId),
        rw('Width (m)',  row[k('width')],   k('width'),   segId),
        rw('Surface',    row[k('surface')], k('surface'), segId),
        ro('Quality',    row[k('quality')]),
        ro('Permitted',  row[k('permitted')]),
        ro('Street',     row.name),
      ],
    };
  }
  if (t === 'node') {
    return {
      title: `Intersection Node`,
      attrs: [
        ro('Node ID',       p._node_id),
        ro('Is intersection', p.is_intersection),
        ro('Node type',     p.node_type),
      ],
    };
  }
  if (t === 'ramp') {
    const parsed = parseRampFid(String(p._fid ?? ''));
    const s   = parsed?.side ?? String(p._side ?? '');
    const pos = parsed?.pos  ?? String(p._position ?? '');
    const n   = parsed?.n    ?? '1';
    const base = `sidewalk_${s}_curbramp_${pos}_${n}`;
    return {
      title: `Curb Ramp (${s} ${pos} #${n})`,
      attrs: [
        ro('Side',      s),
        ro('Position',  pos),
        ro('Condition', row[`${base}_condition_score`]),
        ro('Street',    row.name),
      ],
    };
  }
  if (t === 'crosswalk') {
    const pos = String(p._xw_pos ?? 'start');
    const k = (col: string) => `crosswalk_${pos}_${col}`;
    return {
      title: `Crosswalk (${pos})`,
      attrs: [
        ro('Position',  pos),
        rw('Type',      row[k('type')], k('type'), segId),
        ro('Marked',    row[k('marked')]),
        ro('Signals',   row[k('signals')]),
        ro('Street',    row.name),
      ],
    };
  }
  if (t === 'calm') {
    return {
      title: `Traffic Calming`,
      attrs: [ro('Street', row.name ?? p.name), ro('Type', p._t)],
    };
  }
  if (t === 'cret') {
    return {
      title: `Curb Return`,
      attrs: [ro('Street', row.name ?? p.name), ro('Grid ID', segId)],
    };
  }
  return { title: String(t), attrs: [] };
}


const SEGMENT_SYMBOL: Record<string, string> = {
  street: '\u2501',
  bikeway: '\u2504',
  sidewalk: '\u2505',
  crosswalk: '\u2550',
  cret: '\u2502',
};

const POINT_SYMBOL: Record<string, string> = {
  node: '\u25CF',
  ramp: '\u25B2',
  calm: '\u25C6',
};

/** Compute shortest distance from [px, py] to a feature's geometry (degrees). */
function distToFeature(px: number, py: number, f: Feature): number {
  const g = f.geometry;
  if (!g) return Infinity;
  if (g.type === 'Point') {
    const dx = g.coordinates[0] - px;
    const dy = g.coordinates[1] - py;
    return Math.sqrt(dx * dx + dy * dy);
  }
  if (g.type === 'LineString') {
    const r = nearestPointOnLine(px, py, g.coordinates as number[][], false);
    return r?.dist ?? Infinity;
  }
  if (g.type === 'MultiLineString') {
    const r = nearestPointOnLine(px, py, g.coordinates as number[][][], true);
    return r?.dist ?? Infinity;
  }
  return Infinity;
}

function SelectionCard({
  feature,
  parentRow,
  onAttrEdit,
}: {
  feature: Feature | null;
  parentRow: Record<string, unknown> | null;
  onAttrEdit: (streetGridId: string, col: string, oldVal: unknown, newVal: string) => void;
}) {
  const setSelectedFeatureId = useEditorStore((s) => s.setSelectedFeatureId);

  if (!feature) return null;

  const props = feature.properties ?? {};
  const ftype = props._t ?? 'unknown';
  const fid = props._fid || props._seg_id || props._node_id || '';
  const focusView = buildFocusView(feature, parentRow);

  return (
    <div className="px-selection-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <div>
          <div className="sel-type">{ftype}</div>
          <div className="sel-id">{String(fid)}</div>
        </div>
        <button
          onClick={() => setSelectedFeatureId(null)}
          title="Deselect"
          style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 14, lineHeight: 1 }}
        >×</button>
      </div>
      {/* Focus view — attribute list matching test_maps popups */}
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
        <tbody>
          {focusView.attrs.map((attr, i) => {
            const displayVal = fmt(attr.value);
            let cell: ReactNode;
            if (!attr.readOnly && attr.editKey) {
              const { col, streetGridId } = attr.editKey;
              cell = (
                <input
                  defaultValue={displayVal === '—' ? '' : displayVal}
                  onBlur={(e) => onAttrEdit(streetGridId, col, attr.value, e.target.value)}
                  style={{ background: 'transparent', border: 'none', borderBottom: '1px solid #444', color: 'var(--text)', fontSize: 11, width: '100%', outline: 'none' }}
                />
              );
            } else {
              cell = (
                <span style={{ color: displayVal === '—' ? 'var(--text-dim)' : 'var(--text)', fontStyle: displayVal === '—' ? 'italic' : undefined }}>
                  {displayVal}
                </span>
              );
            }
            return (
              <tr key={i}>
                <td style={{ color: 'var(--text-dim)', fontSize: 10, fontWeight: 600, paddingRight: 8, whiteSpace: 'nowrap', paddingBottom: 2 }}>
                  {attr.label}
                </td>
                <td style={{ fontSize: 11, paddingBottom: 2 }}>{cell}</td>
              </tr>
            );
          })}
          {focusView.attrs.length === 0 && (
            <tr><td colSpan={2} style={{ color: 'var(--text-dim)', fontStyle: 'italic', fontSize: 11 }}>No data</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

type ProbeTab = 'segments' | 'nodes';

export function ProbePanel() {
  const minimized = useEditorStore((s) => s.probeMinimized);
  const setMinimized = useEditorStore((s) => s.setProbeMinimized);
  const [tab, setTab] = useState<ProbeTab>('segments');
  const features = useEditorStore((s) => s.features);
  const selectedId = useEditorStore((s) => s.selectedFeatureId);
  const setSelectedFeatureId = useEditorStore((s) => s.setSelectedFeatureId);
  const toggleFeatureVisibility = useEditorStore((s) => s.toggleFeatureVisibility);
  const hiddenFeatures = useEditorStore((s) => s.hiddenFeatures);
  const probeActive = useEditorStore((s) => s.probeActive);
  const setProbeActive = useEditorStore((s) => s.setProbeActive);
  const probeOrigin = useEditorStore((s) => s.probeOrigin);
  const parquet = useEditorStore((s) => s.parquet);
  const center = useEditorStore((s) => s.center);
  const recordAttrEdit = useChangesetStore((s) => s.recordAttrEdit);

  const [parentRow, setParentRow] = useState<Record<string, unknown> | null>(null);

  const selectedFeature = useMemo(() => {
    if (!selectedId || !features) return null;
    return features.features.find((f) => {
      const p = f.properties;
      if (!p) return false;
      return (p._fid || p._seg_id || p._node_id || p.street_grid_id || null) === selectedId;
    }) ?? null;
  }, [selectedId, features]);

  // Fetch parent street row when a sub-feature is selected
  useEffect(() => {
    if (!selectedFeature || !parquet) { setParentRow(null); return; }
    const p = selectedFeature.properties ?? {};
    const t = String(p._t ?? '');
    // Streets: use their own row (seg_id = street_grid_id)
    const segId = String(p._seg_id || p.street_grid_id || '');
    if (!segId) { setParentRow(null); return; }
    const delta = 0.005;
    const bbox: [number, number, number, number] = [
      center[0] - delta, center[1] - delta,
      center[0] + delta, center[1] + delta,
    ];
    fetchRows(parquet, bbox)
      .then((rows) => {
        const match = rows.find((r) => r.street_grid_id === segId);
        if (t === 'street') {
          // For streets, use the feature's own properties merged with the row
          setParentRow(match ?? { ...p });
        } else {
          setParentRow(match ?? null);
        }
      })
      .catch(() => setParentRow(null));
  }, [selectedFeature, parquet, center]);

  const handleAttrEdit = useCallback((streetGridId: string, col: string, oldVal: unknown, newVal: string) => {
    if (String(oldVal) === newVal) return;
    recordAttrEdit({
      street_grid_id: streetGridId,
      col,
      old_value: oldVal == null ? null : String(oldVal),
      new_value: newVal,
    });
  }, [recordAttrEdit]);

  const allSegments = useMemo(() => {
    if (!features) return [];
    return features.features.filter((f) =>
      ['street', 'bikeway', 'sidewalk', 'crosswalk', 'cret'].includes(f.properties?._t)
    );
  }, [features]);

  const allPoints = useMemo(() => {
    if (!features) return [];
    return features.features.filter((f) =>
      ['node', 'ramp', 'calm'].includes(f.properties?._t)
    );
  }, [features]);

  // Lists are empty when probe is inactive; sorted by proximity when a click origin is set
  const segments = useMemo(() => {
    if (!probeActive) return [];
    if (!probeOrigin) return [];
    const [px, py] = probeOrigin;
    return [...allSegments]
      .map((f) => ({ f, d: distToFeature(px, py, f) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 50)
      .map(({ f }) => f);
  }, [allSegments, probeOrigin, probeActive]);

  const points = useMemo(() => {
    if (!probeActive) return [];
    if (!probeOrigin) return [];
    const [px, py] = probeOrigin;
    return [...allPoints]
      .map((f) => ({ f, d: distToFeature(px, py, f) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 50)
      .map(({ f }) => f);
  }, [allPoints, probeOrigin, probeActive]);

  const btnBase: CSSProperties = {
    background: 'none', border: 'none', fontSize: 11, fontWeight: 600,
    padding: '6px 0', cursor: 'pointer', fontFamily: 'monospace', textTransform: 'uppercase',
  };

  if (minimized) {
    return (
      <div
        onClick={() => setMinimized(false)}
        title="Expand Probe"
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '8px 12px', cursor: 'pointer', flexShrink: 0,
          borderBottom: '1px solid var(--border)',
          fontSize: 12, fontWeight: 700, color: 'var(--text-dim)',
          textTransform: 'uppercase', letterSpacing: '0.5px',
        }}
      >
        <span style={{ fontSize: 14 }}>{'\u22B3'}</span>
        <span>Probe</span>
      </div>
    );
  }

  return (
    <div className="px-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div className="px-panel-header">
        <span>Probe</span>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        {/* Probe mode toggle */}
        <button
          title={probeActive ? 'Probe active — click map to sort by proximity' : 'Activate probe mode'}
          onClick={() => setProbeActive(!probeActive)}
          style={{
            background: probeActive ? '#fff' : 'none',
            color: probeActive ? '#000' : 'var(--text-dim)',
            border: '1px solid',
            borderColor: probeActive ? '#fff' : '#555',
            borderRadius: 3,
            fontSize: 13,
            padding: '1px 6px',
            cursor: 'pointer',
            fontFamily: 'monospace',
          }}
        >
          {'\u22B3'} {/* ⊳ */}
        </button>
        <button
          title="Collapse"
          onClick={() => setMinimized(true)}
          style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 14, padding: '0 2px' }}
        >
          {'\u2212'}
        </button>
        </div>
      </div>

      {/* Status line when probe active */}
      {probeActive && (
        <div style={{ padding: '4px 8px', fontSize: 10, color: 'var(--text-dim)', fontFamily: 'monospace', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
          {probeOrigin
            ? `Origin: ${probeOrigin[0].toFixed(5)}, ${probeOrigin[1].toFixed(5)}`
            : 'Click map to sort by proximity'}
        </div>
      )}

      {/* Selection detail card */}
      <SelectionCard feature={selectedFeature} parentRow={parentRow} onAttrEdit={handleAttrEdit} />

      {/* Tab bar */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', padding: '0 4px', flexShrink: 0 }}>
        <button
          onClick={() => setTab('segments')}
          style={{ ...btnBase, flex: 1, borderBottom: tab === 'segments' ? '2px solid #fff' : '2px solid transparent', color: tab === 'segments' ? 'var(--text)' : 'var(--text-dim)' }}
        >
          Segs ({segments.length})
        </button>
        <button
          onClick={() => setTab('nodes')}
          style={{ ...btnBase, flex: 1, borderBottom: tab === 'nodes' ? '2px solid #fff' : '2px solid transparent', color: tab === 'nodes' ? 'var(--text)' : 'var(--text-dim)' }}
        >
          Nodes ({points.length})
        </button>
      </div>

      {/* Tab content */}
      <div className="px-panel-body" style={{ flex: 1, overflowY: 'auto' }}>
        {tab === 'segments' && (
          <ul className="px-probe-list">
            {segments.map((f) => {
              const id = f.properties?._fid ?? f.properties?._seg_id;
              const t = f.properties?._t as string;
              const isHidden = id && hiddenFeatures.has(id);
              return (
                <li
                  key={id}
                  className={`px-probe-item${selectedId === id ? ' selected' : ''}`}
                  onClick={() => setSelectedFeatureId(id)}
                  style={{ opacity: isHidden ? 0.4 : 1 }}
                >
                  <span style={{ color: 'var(--text-dim)', width: 14, textAlign: 'center', flexShrink: 0 }}>
                    {SEGMENT_SYMBOL[t] ?? '\u2501'}
                  </span>
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {f.properties?.street_grid_id ?? f.properties?._seg_id ?? id}
                  </span>
                  <button
                    title="Toggle visibility"
                    onClick={(ev) => { ev.stopPropagation(); if (id) toggleFeatureVisibility(id); }}
                    style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 12 }}
                  >
                    {isHidden ? '\u25CB' : '\u25CF'}
                  </button>
                </li>
              );
            })}
            {segments.length === 0 && (
              <li style={{ color: 'var(--text-dim)', fontSize: 12, padding: '8px 0' }}>No segments in view</li>
            )}
          </ul>
        )}

        {tab === 'nodes' && (
          <ul className="px-probe-list">
            {points.map((f, i) => {
              const id = f.properties?._fid ?? f.properties?._node_id ?? `pt-${i}`;
              const t = f.properties?._t as string;
              const isHidden = id && hiddenFeatures.has(id);
              return (
                <li
                  key={id}
                  className={`px-probe-item${selectedId === id ? ' selected' : ''}`}
                  onClick={() => setSelectedFeatureId(id)}
                  style={{ opacity: isHidden ? 0.4 : 1 }}
                >
                  <span style={{ color: 'var(--text-dim)', width: 14, textAlign: 'center', flexShrink: 0 }}>
                    {POINT_SYMBOL[t] ?? '\u25CF'}
                  </span>
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {id}
                  </span>
                  <button
                    title="Toggle visibility"
                    onClick={(ev) => { ev.stopPropagation(); if (id) toggleFeatureVisibility(id); }}
                    style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 12 }}
                  >
                    {isHidden ? '\u25CB' : '\u25CF'}
                  </button>
                </li>
              );
            })}
            {points.length === 0 && (
              <li style={{ color: 'var(--text-dim)', fontSize: 12, padding: '8px 0' }}>No points in view</li>
            )}
          </ul>
        )}
      </div>
    </div>
  );
}
