import { useState } from 'react';
import { useEditorStore } from '../../store/editorStore';

export function SearchPanel() {
  const searchOpen = useEditorStore((s) => s.searchOpen);
  const toggleSearch = useEditorStore((s) => s.toggleSearch);
  const setSelectedFeatureId = useEditorStore((s) => s.setSelectedFeatureId);
  const features = useEditorStore((s) => s.features);

  const [mode, setMode] = useState<'segment' | 'node'>('segment');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<string[]>([]);

  const handleSearch = () => {
    if (!features || !query.trim()) return;
    const q = query.trim().toLowerCase();
    const matches = features.features
      .filter((f) => {
        if (mode === 'segment') {
          return f.properties?._seg_id?.toLowerCase().includes(q) ||
                 f.properties?.name?.toLowerCase().includes(q);
        }
        return String(f.properties?._fid ?? '').toLowerCase().includes(q);
      })
      .map((f) => f.properties?._fid ?? f.properties?._seg_id ?? '')
      .filter(Boolean)
      .slice(0, 20);
    setResults(matches);
  };

  return (
    <div className={`px-slide-panel${searchOpen ? ' open' : ''}`}>
      <button className="panel-close" onClick={toggleSearch}>{'\u00D7'}</button>
      <div style={{ padding: '40px 16px 16px' }}>
        <h3 style={{ fontSize: 14, marginBottom: 12 }}>Search</h3>
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <button
            className={`px-btn${mode === 'segment' ? ' px-btn-primary' : ''}`}
            onClick={() => setMode('segment')}
            style={{ flex: 1, fontSize: 12 }}
          >
            Segment
          </button>
          <button
            className={`px-btn${mode === 'node' ? ' px-btn-primary' : ''}`}
            onClick={() => setMode('node')}
            style={{ flex: 1, fontSize: 12 }}
          >
            Node
          </button>
        </div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <input
            className="px-input"
            placeholder={mode === 'segment' ? 'Segment ID or name...' : 'Node ID...'}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button className="px-btn px-btn-primary" onClick={handleSearch}>Go</button>
        </div>
        <ul className="px-probe-list">
          {results.map((id) => (
            <li
              key={id}
              className="px-probe-item"
              onClick={() => { setSelectedFeatureId(id); toggleSearch(); }}
            >
              <span style={{ flex: 1 }}>{id}</span>
            </li>
          ))}
          {results.length === 0 && query && (
            <li style={{ color: 'var(--text-dim)', fontSize: 12, padding: 8 }}>
              No results
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}
