import { useChangesetStore } from '../../store/changesetStore';
import { useEditorStore } from '../../store/editorStore';
import { saveChangesetStream } from '../../api/editorApi';
import { useState, useEffect } from 'react';
import type { PipelineStage } from '@proximity/shared/types/changeset';

const EDIT_ICONS: Record<string, string> = {
  move_point: '\u2197',
  snap_point: '\u2295',
  add_point: '+',
  delete_point: '\u00D7',
  merge_segments: '\u2295',
  split_segment: '\u2702',
  draw_segment: '\u270F',
  add_polygon: '\u25A3',
  delete_polygon: '\u25A2',
  edit_polygon_face: '\u25C7',
  consolidate_hulls: '\u229E',
  toggle_curb_ramp: '\u267F',
  draw_crosswalk: '\u2550',
  attr_edit: '\u270E',
};

export function HistoryPanel() {
  const minimized = useEditorStore((s) => s.historyMinimized);
  const setMinimized = useEditorStore((s) => s.setHistoryMinimized);
  const history = useChangesetStore((s) => s.history);
  const edits = useChangesetStore((s) => s.edits);
  const dirtyFromStage = useChangesetStore((s) => s.dirtyFromStage);
  const parquet = useChangesetStore((s) => s.parquet);
  const clear = useChangesetStore((s) => s.clear);
  const undo = useChangesetStore((s) => s.undo);
  const computeBBox = useChangesetStore((s) => s.computeBBox);
  const hasAnyEdits = history.length > 0;
  const [saveProgress, setSaveProgress] = useState<{ pct: number; stage: string } | null>(null);
  const saving = saveProgress !== null;

  const popSnapshot = useEditorStore((s) => s.popSnapshot);
  const setFeatures = useEditorStore((s) => s.setFeatures);

  const handleUndo = () => {
    undo();
    const restored = popSnapshot();
    if (restored) setFeatures(restored);
  };

  // Ctrl+Z for undo
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        handleUndo();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [undo]);

  const handleSave = async () => {
    if (!hasAnyEdits) return;
    const bbox = computeBBox();
    // If no coordinate bbox (e.g. merge/crosswalk/hull-delete edits), fall back to current viewport
    const { center } = useEditorStore.getState();
    const delta = 0.05; // ~5 km viewport fallback
    const effectiveBbox = bbox ?? {
      minX: center[0] - delta, minY: center[1] - delta,
      maxX: center[0] + delta, maxY: center[1] + delta,
    };

    setSaveProgress({ pct: 0, stage: 'starting' });
    try {
      await saveChangesetStream(
        {
          parquet,
          dirty_from_stage: (dirtyFromStage ?? 99) as PipelineStage,
          bbox: effectiveBbox,
          edits,
        },
        (p) => setSaveProgress({ pct: p.pct, stage: p.stage }),
      );
      clear();
    } catch (err) {
      console.error('[HistoryPanel] Save failed:', err);
      alert('Save failed. Check console for details.');
    } finally {
      setSaveProgress(null);
    }
  };

  const handleDiscard = () => {
    if (!hasAnyEdits) return;
    if (window.confirm('Discard all pending edits?')) {
      clear();
    }
  };

  if (minimized) {
    return (
      <div
        onClick={() => setMinimized(false)}
        title="Expand History"
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '8px 12px', cursor: 'pointer', flexShrink: 0,
          borderTop: '1px solid var(--border)',
          fontSize: 12, fontWeight: 700, color: 'var(--text-dim)',
          textTransform: 'uppercase', letterSpacing: '0.5px',
        }}
      >
        <span style={{ fontSize: 14, color: 'var(--success)' }}>{'\u2713'}</span>
        <span>History ({history.length})</span>
      </div>
    );
  }

  const STAGE_LABELS: Record<string, string> = {
    starting: 'Starting…',
    applying_edits: 'Applying edits…',
    snap_endpoints: 'Snapping endpoints…',
    build_hulls: 'Building hulls…',
    assign_ramps: 'Assigning ramps…',
    create_crosswalks: 'Creating crosswalks…',
    writing_parquet: 'Writing parquet…',
    done: 'Done',
  };

  return (
    <div className="px-panel" style={{ maxHeight: '40%', display: 'flex', flexDirection: 'column' }}>
      <div className="px-panel-header">
        <span>History ({history.length})</span>
        {saving ? (
          <div style={{ flex: 1, marginLeft: 8, display: 'flex', flexDirection: 'column', gap: 3, justifyContent: 'center' }}>
            <div style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'monospace' }}>
              {STAGE_LABELS[saveProgress!.stage] ?? saveProgress!.stage}
            </div>
            <div style={{ background: '#333', borderRadius: 2, height: 4, overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${saveProgress!.pct}%`,
                  background: 'var(--success)',
                  borderRadius: 2,
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
          </div>
        ) : (
        <div style={{ display: 'flex', gap: 4 }}>
          <button
            disabled={!hasAnyEdits}
            onClick={handleUndo}
            title="Undo last edit (Ctrl+Z)"
            style={{
              background: 'none', border: 'none',
              color: hasAnyEdits ? 'var(--text)' : 'var(--text-dim)',
              cursor: hasAnyEdits ? 'pointer' : 'default',
              fontSize: 14, padding: '2px 4px',
            }}
          >
            {'\u21A9'}
          </button>
          <button
            className="save-btn"
            disabled={!hasAnyEdits}
            onClick={handleSave}
            title="Save all edits"
            style={{
              color: hasAnyEdits ? 'var(--success)' : undefined,
              cursor: hasAnyEdits ? 'pointer' : 'default',
            }}
          >
            {'\u2713'}
          </button>
          <button
            className="discard-btn"
            disabled={!hasAnyEdits}
            onClick={handleDiscard}
            title="Discard all edits"
            style={{
              color: hasAnyEdits ? 'var(--danger)' : undefined,
              cursor: hasAnyEdits ? 'pointer' : 'default',
            }}
          >
            {'\u2717'}
          </button>
          <button
            title="Collapse"
            onClick={() => setMinimized(true)}
            style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 14, padding: '2px 4px' }}
          >
            {'\u2212'}
          </button>
        </div>
        )}
      </div>
      <div className="px-panel-body">
        {history.length === 0 ? (
          <p style={{ color: 'var(--text-dim)', fontSize: 12 }}>No edits yet</p>
        ) : (
          history.slice().reverse().map((entry, i) => (
            <div key={i} className="px-history-item">
              <span className="icon">{EDIT_ICONS[entry.type] ?? '?'}</span>
              <div style={{ flex: 1 }}>
                <div>{entry.label}</div>
                <div className="timestamp">
                  {new Date(entry.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
