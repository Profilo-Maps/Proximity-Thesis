import { useState, useEffect } from 'react';
import { MapView } from './components/MapView';
import { ToolBar } from './components/ToolBar';
import { ProbePanel } from './components/panels/ProbePanel';
import { HistoryPanel } from './components/panels/HistoryPanel';
import { AttributeTable } from './components/panels/AttributeTable';
import { LayersPanel } from './components/panels/LayersPanel';
import { SearchPanel } from './components/panels/SearchPanel';
import { ConfigPanel } from './components/panels/ConfigPanel';
import { SubtypeSelector } from './components/SubtypeSelector';
import { useEditorStore } from './store/editorStore';
import { useChangesetStore } from './store/changesetStore';
import { fetchParquetList } from './api/editorApi';

export function App() {
  const bottomExpanded = useEditorStore((s) => s.bottomExpanded);
  const setBottomExpanded = useEditorStore((s) => s.setBottomExpanded);
  const probeMinimized = useEditorStore((s) => s.probeMinimized);
  const historyMinimized = useEditorStore((s) => s.historyMinimized);
  const parquet = useEditorStore((s) => s.parquet);
  const setParquet = useEditorStore((s) => s.setParquet);
  const setChangesetParquet = useChangesetStore((s) => s.setParquet);
  const searchOpen = useEditorStore((s) => s.searchOpen);
  const configOpen = useEditorStore((s) => s.configOpen);
  const toggleSearch = useEditorStore((s) => s.toggleSearch);
  const toggleConfig = useEditorStore((s) => s.toggleConfig);
  const [parquetList, setParquetList] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  // Initialize parquet from URL query param or auto-detect
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get('parquet');

    if (fromUrl) {
      setParquet(fromUrl);
      setChangesetParquet(fromUrl);
      setLoading(false);
      return;
    }

    fetchParquetList()
      .then((files) => {
        setParquetList(files);
        if (files.length === 1) {
          setParquet(files[0]);
          setChangesetParquet(files[0]);
        }
      })
      .catch((err) => console.error('[App] Failed to list parquets:', err))
      .finally(() => setLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelectParquet = (name: string) => {
    setParquet(name);
    setChangesetParquet(name);
    const url = new URL(window.location.href);
    url.searchParams.set('parquet', name);
    window.history.replaceState({}, '', url.toString());
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', color: '#fff' }}>
        Loading...
      </div>
    );
  }

  if (!parquet && parquetList.length > 1) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', gap: 16, color: '#fff' }}>
        <h2>Select a parquet file</h2>
        {parquetList.map((name) => (
          <button
            key={name}
            className="px-btn px-btn-primary"
            style={{ padding: '12px 24px', fontSize: 14 }}
            onClick={() => handleSelectParquet(name)}
          >
            {name}
          </button>
        ))}
      </div>
    );
  }

  if (!parquet) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', color: '#fff' }}>
        No parquet files found in Output/. Run the pipeline first.
      </div>
    );
  }

  const slideOpen = searchOpen || configOpen;
  const bothCollapsed = probeMinimized && historyMinimized;

  return (
    <div className={`px-root${bottomExpanded ? ' bottom-expanded' : ''}${bothCollapsed ? ' right-collapsed' : ''}`}>
      <ToolBar />
      <SubtypeSelector />

      <div className="px-map-area">
        <MapView />
        <LayersPanel />
      </div>

      <div className={`px-right-col${bothCollapsed ? ' both-collapsed' : ''}`}>
        <ProbePanel />
        <HistoryPanel />
      </div>

      <AttributeTable
        expanded={bottomExpanded}
        onToggle={() => setBottomExpanded(!bottomExpanded)}
      />

      {/* Overlays — outside grid flow */}
      <div className="px-overlays">
        <div
          className={`px-slide-backdrop${slideOpen ? ' open' : ''}`}
          onClick={() => { if (searchOpen) toggleSearch(); if (configOpen) toggleConfig(); }}
        />
        <SearchPanel />
        <ConfigPanel />
      </div>
    </div>
  );
}
