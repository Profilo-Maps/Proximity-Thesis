import { useEditorStore } from '../store/editorStore';
import { TOOL_SUBTYPES } from '@proximity/shared/tools';
import type { ToolSubtype } from '@proximity/shared/tools';

const S = {
  container: {
    position: 'absolute' as const,
    left: '62px',
    top: '60px',
    background: '#111',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '4px',
    zIndex: 15,
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '2px',
    minWidth: '120px',
  },
  label: {
    fontSize: '9px',
    fontWeight: 700 as const,
    color: '#666',
    letterSpacing: '1px',
    padding: '2px 6px',
    fontFamily: 'monospace',
  },
  option: {
    background: 'transparent',
    border: 'none',
    borderRadius: '3px',
    color: '#ccc',
    fontSize: '11px',
    fontFamily: 'monospace',
    padding: '4px 8px',
    cursor: 'pointer',
    textAlign: 'left' as const,
  },
  optionActive: {
    background: '#fff',
    color: '#000',
  },
};

export function SubtypeSelector() {
  const activeTool = useEditorStore((s) => s.activeTool);
  const activeSubtype = useEditorStore((s) => s.activeSubtype);
  const setActiveSubtype = useEditorStore((s) => s.setActiveSubtype);

  if (!activeTool) return null;
  const def = TOOL_SUBTYPES[activeTool];
  if (!def) return null;

  return (
    <div style={S.container}>
      <div style={S.label}>TYPE</div>
      {def.options.map((opt) => {
        const isActive = activeSubtype === opt.id;
        return (
          <button
            key={opt.id}
            style={isActive ? { ...S.option, ...S.optionActive } : S.option}
            onClick={() => setActiveSubtype(opt.id as ToolSubtype)}
            onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = '#333'; }}
            onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent'; }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
