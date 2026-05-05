import { useEditorStore } from '../store/editorStore';
import type { ToolId } from '../store/editorStore';

const SECTIONS = [
  {
    label: 'POINT',
    tools: [
      { id: 'move_point',   label: '\u2195' },
      { id: 'add_node',     label: '\u002B' },
      { id: 'delete_point', label: '\u00D7' },
    ],
  },
  {
    label: 'LINE',
    tools: [
      { id: 'merge_segments', label: '\u2AFA' },
      { id: 'split_segment',  label: '\u2702' },
      { id: 'draw_segment',   label: '\u2571' },
    ],
  },
  {
    label: 'POLY',
    tools: [
      { id: 'add_polygon',       label: '\u2B21' },
      { id: 'delete_polygon',    label: '\u2B22' },
      { id: 'edit_polygon_face', label: '\u2B23' },
    ],
  },
];

const S = {
  nav: {
    gridRow: '1 / 3',
    gridColumn: '1 / 2',
    background: '#000',
    borderRight: '1px solid #333',
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'stretch' as const,
    padding: 0,
    zIndex: 10,
    overflowY: 'auto' as const,
    overflowX: 'hidden' as const,
    width: '56px',
  },
  section: {
    borderBottom: '1px solid #333',
    padding: '6px 0',
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center' as const,
    gap: '2px',
  },
  sectionLabel: {
    fontSize: '9px',
    fontWeight: 700 as const,
    letterSpacing: '1px',
    color: '#666',
    textAlign: 'center' as const,
    marginBottom: '2px',
    fontFamily: 'monospace',
  },
  btn: {
    width: '46px',
    height: '32px',
    border: 'none',
    borderRadius: '3px',
    background: 'transparent',
    color: '#fff',
    fontSize: '18px',
    fontWeight: 400 as const,
    fontFamily: 'Arial, Helvetica, sans-serif',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    letterSpacing: '0',
  },
  btnActive: {
    background: '#fff',
    color: '#000',
  },
  topRow: {
    display: 'flex',
    flexDirection: 'row' as const,
    justifyContent: 'center' as const,
    gap: '2px',
    padding: '6px 0',
    borderBottom: '1px solid #333',
  },
  bottomRow: {
    marginTop: 'auto' as const,
    display: 'flex',
    flexDirection: 'row' as const,
    justifyContent: 'center' as const,
    gap: '2px',
    padding: '6px 0',
    borderTop: '1px solid #333',
  },
  topBtn: {
    width: '24px',
    height: '28px',
    border: 'none',
    borderRadius: '3px',
    background: 'transparent',
    color: '#999',
    fontSize: '16px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  },
};

export function ToolBar() {
  const activeTool = useEditorStore((s) => s.activeTool);
  const setActiveTool = useEditorStore((s) => s.setActiveTool);
  const toggleSearch = useEditorStore((s) => s.toggleSearch);
  const toggleConfig = useEditorStore((s) => s.toggleConfig);

  return (
    <nav style={S.nav}>
      <div style={S.topRow}>
        <button
          style={S.topBtn}
          title="Search"
          onClick={toggleSearch}
          onMouseEnter={(e) => { e.currentTarget.style.background = '#333'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
        >
          {'\u2315'}
        </button>
      </div>
      {SECTIONS.map((section) => (
        <div key={section.label} style={S.section}>
          <div style={S.sectionLabel}>{section.label}</div>
          {section.tools.map((tool) => {
            const isActive = activeTool === tool.id;
            return (
              <button
                key={tool.id}
                style={isActive ? { ...S.btn, ...S.btnActive } : S.btn}
                title={tool.id.replace(/_/g, ' ')}
                onClick={() => setActiveTool(tool.id as ToolId)}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.background = '#333';
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.background = 'transparent';
                }}
              >
                {tool.label}
              </button>
            );
          })}
        </div>
      ))}
      <div style={S.bottomRow}>
        <button
          style={S.topBtn}
          title="Settings"
          onClick={toggleConfig}
          onMouseEnter={(e) => { e.currentTarget.style.background = '#333'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
        >
          {'\u2699'}
        </button>
      </div>
    </nav>
  );
}
