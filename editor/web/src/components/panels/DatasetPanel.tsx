import { useState, useEffect } from 'react';
import { useEditorStore } from '../../store/editorStore';
import { useChangesetStore } from '../../store/changesetStore';
import { fetchParquetList } from '../../api/editorApi';

export function DatasetPanel() {
  const datasetOpen = useEditorStore((s) => s.datasetOpen);
  const toggleDataset = useEditorStore((s) => s.toggleDataset);
  const parquet = useEditorStore((s) => s.parquet);
  const setParquet = useEditorStore((s) => s.setParquet);
  const setChangesetParquet = useChangesetStore((s) => s.setParquet);

  const [files, setFiles] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!datasetOpen) return;
    setLoading(true);
    fetchParquetList()
      .then(setFiles)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [datasetOpen]);

  const handleSelect = (name: string) => {
    setParquet(name);
    setChangesetParquet(name);
    const url = new URL(window.location.href);
    url.searchParams.set('parquet', name);
    window.history.replaceState({}, '', url.toString());
    toggleDataset();
  };

  return (
    <div className={`px-slide-panel${datasetOpen ? ' open' : ''}`}>
      <button className="panel-close" onClick={toggleDataset}>{'\u00D7'}</button>
      <div style={{ padding: '40px 16px 16px', display: 'flex', flexDirection: 'column', height: '100%' }}>
        <h3 style={{ fontSize: 14, marginBottom: 12 }}>Select Dataset</h3>
        {loading && <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>Loading...</div>}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {files.map((name) => (
            <button
              key={name}
              onClick={() => handleSelect(name)}
              style={{
                background: name === parquet ? '#fff' : 'var(--surface)',
                color: name === parquet ? '#000' : 'var(--text)',
                border: '1px solid var(--border)',
                borderRadius: 4,
                padding: '8px 12px',
                textAlign: 'left',
                cursor: 'pointer',
                fontSize: 13,
                fontFamily: 'monospace',
              }}
            >
              {name === parquet ? '● ' : '○ '}{name}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
