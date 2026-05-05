import { useState, useEffect } from 'react';
import { useEditorStore } from '../../store/editorStore';
import { fetchConfig, saveConfig } from '../../api/editorApi';

interface ConfigField {
  name: string;
  type: string;
  default: unknown;
}

export function ConfigPanel() {
  const configOpen = useEditorStore((s) => s.configOpen);
  const toggleConfig = useEditorStore((s) => s.toggleConfig);
  const parquet = useEditorStore((s) => s.parquet);

  const [tab, setTab] = useState<'global' | 'city'>('global');
  const [fields, setFields] = useState<ConfigField[]>([]);
  const [globalValues, setGlobalValues] = useState<Record<string, unknown>>({});
  const [cityValues, setCityValues] = useState<Record<string, unknown>>({});
  const [staged, setStaged] = useState<Record<string, unknown>>({});

  useEffect(() => {
    if (!configOpen || !parquet) return;
    fetchConfig(parquet).then((data) => {
      setFields(data.fields);
      setGlobalValues(data.global);
      setCityValues(data.city);
      setStaged({});
    }).catch(console.error);
  }, [configOpen, parquet]);

  const currentValues = tab === 'global' ? globalValues : cityValues;

  const handleChange = (key: string, value: string) => {
    setStaged((s) => ({ ...s, [key]: value }));
  };

  const handleSave = async () => {
    if (Object.keys(staged).length === 0) return;
    try {
      await saveConfig(parquet, tab, staged);
      if (tab === 'global') setGlobalValues((v) => ({ ...v, ...staged }));
      else setCityValues((v) => ({ ...v, ...staged }));
      setStaged({});
    } catch (err) {
      console.error('[ConfigPanel] Save failed:', err);
      alert('Config save failed.');
    }
  };

  return (
    <div className={`px-slide-panel${configOpen ? ' open' : ''}`}>
      <button className="panel-close" onClick={toggleConfig}>{'\u00D7'}</button>
      <div style={{ padding: '40px 16px 16px', display: 'flex', flexDirection: 'column', height: '100%' }}>
        <h3 style={{ fontSize: 14, marginBottom: 12 }}>Pipeline Config</h3>
        <div className="px-config-tabs">
          <button
            className={`px-config-tab${tab === 'global' ? ' active' : ''}`}
            onClick={() => { setTab('global'); setStaged({}); }}
          >
            Global
          </button>
          <button
            className={`px-config-tab${tab === 'city' ? ' active' : ''}`}
            onClick={() => { setTab('city'); setStaged({}); }}
          >
            City
          </button>
        </div>
        <div style={{ flex: 1, overflow: 'auto', paddingTop: 8 }}>
          {fields.map((field) => (
            <div key={field.name} className="px-config-field">
              <label>{field.name}</label>
              {field.type === 'bool' ? (
                <input
                  type="checkbox"
                  checked={Boolean(staged[field.name] ?? currentValues[field.name] ?? field.default)}
                  onChange={(e) => handleChange(field.name, String(e.target.checked))}
                />
              ) : (
                <input
                  type={field.type === 'float' || field.type === 'int' ? 'number' : 'text'}
                  step={field.type === 'float' ? '0.1' : '1'}
                  value={String(staged[field.name] ?? currentValues[field.name] ?? field.default ?? '')}
                  onChange={(e) => handleChange(field.name, e.target.value)}
                />
              )}
            </div>
          ))}
        </div>
        {Object.keys(staged).length > 0 && (
          <div style={{ padding: '12px 0', borderTop: '1px solid var(--border)' }}>
            <button className="px-btn px-btn-primary" style={{ width: '100%' }} onClick={handleSave}>
              Save {tab} config ({Object.keys(staged).length} changes)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
