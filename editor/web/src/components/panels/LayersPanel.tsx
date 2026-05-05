import { useState } from 'react';
import { useEditorStore, type LayerVisibility } from '../../store/editorStore';

const LAYER_CONFIG: { key: keyof LayerVisibility; label: string; color: string; circle?: boolean; line?: boolean }[] = [
  { key: 'streets',       label: 'Streets',              color: '#c0392b', line: true },
  { key: 'bikesSep',      label: 'Bikeways – separate',  color: '#006400', line: true },
  { key: 'bikesOff',      label: 'Bikeways – buffered',  color: '#90ee90', line: true },
  { key: 'sidewalksSep',  label: 'Sidewalks – separate', color: '#00008b', line: true },
  { key: 'sidewalksOff',  label: 'Sidewalks – buffered', color: '#add8e6', line: true },
  { key: 'crosswalks',    label: 'Crosswalks',           color: '#ff6b81', line: true },
  { key: 'curbReturns',   label: 'Curb Returns',         color: '#a55eea', line: true },
  { key: 'hulls',      label: 'Hulls',            color: '#00d4ff' },
  { key: 'slots',      label: 'Slots',            color: '#ff6b81' },
  { key: 'nodes',      label: 'Nodes',            color: '#e0e0e0', circle: true },
  { key: 'ramps',      label: 'Curb Ramps',       color: '#ff4757', circle: true },
  { key: 'calming',    label: 'Traffic Calming',   color: '#8b5cf6', circle: true },
];

export function LayersPanel() {
  const [minimized, setMinimized] = useState(false);
  const layers = useEditorStore((s) => s.layers);
  const toggleLayer = useEditorStore((s) => s.toggleLayer);

  if (minimized) {
    return (
      <div
        className="px-layers-panel"
        style={{ minWidth: 'auto', cursor: 'pointer' }}
        onClick={() => setMinimized(false)}
      >
        Layers
      </div>
    );
  }

  return (
    <div className="px-layers-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontWeight: 600, fontSize: 11, textTransform: 'uppercase', color: 'var(--text-dim)' }}>
          Layers
        </span>
        <button
          onClick={() => setMinimized(true)}
          style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 12 }}
        >
          {'\u2212'}
        </button>
      </div>
      {LAYER_CONFIG.map(({ key, label, color, circle, line }) => (
        <div
          key={key}
          className={`layer-row${!layers[key] ? ' hidden' : ''}`}
          onClick={() => toggleLayer(key)}
        >
          <div
            className="layer-swatch"
            style={line ? {
              background: layers[key] ? color : '#333',
              width: 20,
              height: 4,
              borderRadius: 2,
              alignSelf: 'center',
            } : {
              background: layers[key] ? color : '#333',
              borderRadius: circle ? '50%' : '2px',
            }}
          />
          <span className="layer-label">{label}</span>
        </div>
      ))}
    </div>
  );
}
