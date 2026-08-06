import { useState, useEffect, useRef, useCallback } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism"

const API_BASE = "http://localhost:8000"

/* ==================== 设计系统 / Design Tokens ==================== */
const tokens = {
  primary: '#39c5bb',
  primaryLight: '#5eead4',
  primaryDark: '#0d9488',
  primaryGlow: 'rgba(57, 197, 187, 0.4)',
  bgBase: '#020617',
  bgSurface: 'rgba(15, 23, 42, 0.6)',
  bgElevated: 'rgba(30, 41, 59, 0.7)',
  bgHover: 'rgba(51, 65, 85, 0.5)',
  textPrimary: '#f1f5f9',
  textSecondary: '#94a3b8',
  textMuted: '#64748b',
  border: 'rgba(148, 163, 184, 0.1)',
  borderActive: 'rgba(57, 197, 187, 0.3)',
  shadowSm: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
  shadowMd: '0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.4)',
  shadowGlow: '0 0 20px rgba(57, 197, 187, 0.15)',
  radiusSm: '8px',
  radiusMd: '12px',
  radiusLg: '16px',
  radiusXl: '24px',
}

/* ==================== 全局样式注入 ==================== */
const GlobalStyles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;600;700&display=swap');
    * { scrollbar-width: thin; scrollbar-color: ${tokens.primary}33 transparent; }
    *::-webkit-scrollbar { width: 6px; height: 6px; }
    *::-webkit-scrollbar-track { background: transparent; }
    *::-webkit-scrollbar-thumb { background: ${tokens.primary}33; border-radius: 3px; }
    *::-webkit-scrollbar-thumb:hover { background: ${tokens.primary}66; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeInScale { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
    @keyframes pulse-glow { 0%, 100% { box-shadow: 0 0 5px ${tokens.primaryGlow}; } 50% { box-shadow: 0 0 20px ${tokens.primaryGlow}, 0 0 40px ${tokens.primaryGlow}; } }
    @keyframes typing { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
    @keyframes slideInLeft { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes shimmer { 0% { background-position: -1000px 0; } 100% { background-position: 1000px 0; } }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    @keyframes dropPulse { 0% { box-shadow: 0 0 0 0 ${tokens.primaryGlow}; } 70% { box-shadow: 0 0 0 10px transparent; } 100% { box-shadow: 0 0 0 0 transparent; } }
    .msg-enter { animation: fadeIn 0.35s cubic-bezier(0.4, 0, 0.2, 1) forwards; }
    .sidebar-item { animation: slideInLeft 0.3s ease forwards; }
    .cursor-blink { animation: typing 1s step-end infinite; }
    .glass { background: ${tokens.bgElevated}; backdrop-filter: blur(20px) saturate(180%); -webkit-backdrop-filter: blur(20px) saturate(180%); border: 1px solid ${tokens.border}; }
    .glass-strong { background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(30px) saturate(200%); -webkit-backdrop-filter: blur(30px) saturate(200%); border: 1px solid ${tokens.border}; }
    .card-hover { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
    .card-hover:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4), 0 0 0 1px ${tokens.primary}30; }
  `}</style>
)

/* ==================== 图标组件 ==================== */
const Icon = ({ children, size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}>
    {children}
  </svg>
)

const Icons = {
  plus: <><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></>,
  trash: <><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></>,
  send: <><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></>,
  search: <><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></>,
  cpu: <><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></>,
  file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></>,
  chart: <><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></>,
  close: <><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></>,
  chevronDown: <><polyline points="6 9 12 15 18 9"/></>,
  message: <><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></>,
  sparkles: <><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></>,
  bot: <><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></>,
  user: <><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></>,
  copy: <><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></>,
  check: <><polyline points="20 6 9 17 4 12"/></>,
  refresh: <><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></>,
  image: <><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></>,
  settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></>,
  heart: <><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></>,
  zap: <><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></>,
  code: <><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></>,
  pen: <><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></>,
  activity: <><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></>,
  circle: <><circle cx="12" cy="12" r="10"/></>,
}

/* ==================== 自定义 Hook：后端会话持久化 ==================== */
function useBackendSessions() {
  const [sessions, setSessions] = useState({})
  const [currentId, setCurrentId] = useState(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE}/sessions`)
      .then(r => r.json())
      .then(data => {
        if (data.sessions && Object.keys(data.sessions).length > 0) {
          setSessions(data.sessions)
          setCurrentId(data.current_id || Object.keys(data.sessions)[0])
        } else {
          createSessionOnBackend()
        }
        setLoaded(true)
      })
      .catch(() => {
        const id = 'session_' + Date.now()
        setSessions({ [id]: { id, title: '新对话', messages: [] } })
        setCurrentId(id)
        setLoaded(true)
      })
  }, [])

  const syncToBackend = useCallback((newSessions, newCurrentId) => {
    fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessions: newSessions, current_id: newCurrentId }),
    }).catch(() => {})
  }, [])

  const current = sessions[currentId] || { id: currentId, title: '新对话', messages: [] }

  const addMessage = (role, content, images = []) => {
    setSessions(prev => {
      const s = { ...prev }
      s[currentId] = { ...s[currentId], messages: [...s[currentId].messages, { role, content, images }] }
      if (role === 'user' && s[currentId].messages.length === 1) {
        s[currentId].title = content.length > 20 ? content.slice(0, 20) + '...' : content || '新对话'
      }
      syncToBackend(s, currentId)
      return s
    })
  }

  const updateLastAssistant = (fullContent) => {
    setSessions(prev => {
      const s = { ...prev }
      const msgs = [...s[currentId].messages]
      if (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') {
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: fullContent }
      } else { msgs.push({ role: 'assistant', content: fullContent }) }
      s[currentId] = { ...s[currentId], messages: msgs }
      syncToBackend(s, currentId)
      return s
    })
  }

  const deleteMessage = (msgIndex) => {
    setSessions(prev => {
      const s = { ...prev }
      const msgs = s[currentId].messages.filter((_, i) => i !== msgIndex)
      s[currentId] = { ...s[currentId], messages: msgs }
      syncToBackend(s, currentId)
      return s
    })
  }

  const createSessionOnBackend = () => {
    const id = 'session_' + Date.now()
    const newSessions = { ...sessions, [id]: { id, title: '新对话', messages: [] } }
    setSessions(newSessions)
    setCurrentId(id)
    syncToBackend(newSessions, id)
  }

  const deleteSession = (id) => {
    setSessions(prev => {
      const newSessions = { ...prev }
      delete newSessions[id]
      const keys = Object.keys(newSessions)
      let nextCurrent = currentId
      if (keys.length === 0) {
        const newId = 'session_' + Date.now()
        newSessions[newId] = { id: newId, title: '新对话', messages: [] }
        nextCurrent = newId
      } else if (currentId === id) {
        nextCurrent = keys[0]
      }
      setCurrentId(nextCurrent)
      syncToBackend(newSessions, nextCurrent)
      return newSessions
    })
  }

  const switchSession = (id) => {
    setCurrentId(id)
    syncToBackend(sessions, id)
  }

  return { sessions, currentId, current, loaded, addMessage, updateLastAssistant, deleteMessage, createSession: createSessionOnBackend, deleteSession, switchSession }
}

/* ==================== 健康状态 Hook ==================== */
function useHealthCheck() {
  const [health, setHealth] = useState({ status: 'checking', latency: null })

  useEffect(() => {
    const check = async () => {
      const start = Date.now()
      try {
        const res = await fetch(`${API_BASE}/health`, { method: 'GET', signal: AbortSignal.timeout(5000) })
        const latency = Date.now() - start
        if (res.ok) {
          setHealth({ status: 'online', latency })
        } else {
          setHealth({ status: 'error', latency })
        }
      } catch {
        setHealth({ status: 'offline', latency: null })
      }
    }
    check()
    const interval = setInterval(check, 15000)
    return () => clearInterval(interval)
  }, [])

  return health
}

/* ==================== 子组件 ==================== */

const ModeBadge = ({ mode }) => {
  const configs = {
    '4b': { label: 'Qwen3.5 · 4B', color: tokens.primary, bg: `${tokens.primary}15` },
    '9b': { label: 'Qwen3.5 · 9B', color: '#a78bfa', bg: 'rgba(167, 139, 250, 0.15)' },
    'review': { label: '三模型审查', color: '#f472b6', bg: 'rgba(244, 114, 182, 0.15)' },
    'agent': { label: 'Agent', color: '#60a5fa', bg: 'rgba(96, 165, 250, 0.15)' },
    'hyde': { label: 'HyDE', color: '#fbbf24', bg: 'rgba(251, 191, 36, 0.15)' },
    'selfrag': { label: 'Self-RAG', color: '#34d399', bg: 'rgba(52, 211, 153, 0.15)' },
  }
  const c = configs[mode] || configs['4b']
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, letterSpacing: '0.05em', color: c.color,
      background: c.bg, padding: '3px 10px', borderRadius: 20,
      border: `1px solid ${c.color}30`, display: 'inline-flex', alignItems: 'center', gap: 4,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: c.color, display: 'inline-block' }} />
      {c.label}
    </span>
  )
}

const HealthIndicator = ({ health }) => {
  const config = {
    online: { color: '#34d399', label: '在线', pulse: true },
    checking: { color: tokens.primary, label: '检测中...', pulse: true },
    error: { color: '#fbbf24', label: '异常', pulse: false },
    offline: { color: '#f87171', label: '离线', pulse: false },
  }
  const c = config[health.status] || config.checking
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: tokens.textMuted }}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%', background: c.color,
        boxShadow: c.pulse ? `0 0 8px ${c.color}` : 'none',
        animation: c.pulse ? 'pulse-glow 2s infinite' : 'none',
      }} />
      <span>{c.label}{health.latency ? ` · ${health.latency}ms` : ''}</span>
    </div>
  )
}

const CopyButton = ({ text }) => {
  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }
  return (
    <button onClick={handleCopy} style={{
      background: 'rgba(255,255,255,0.1)', border: 'none', color: tokens.textSecondary,
      cursor: 'pointer', padding: '4px 8px', borderRadius: 6, fontSize: 11,
      display: 'flex', alignItems: 'center', gap: 4, transition: 'all 0.2s',
    }} onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.2)'; e.currentTarget.style.color = tokens.textPrimary }}
      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; e.currentTarget.style.color = tokens.textSecondary }}>
      <Icon size={12}>{copied ? Icons.check : Icons.copy}</Icon>
      {copied ? '已复制' : '复制'}
    </button>
  )
}

const MessageActions = ({ content, onRegenerate, onDelete, index }) => {
  const [visible, setVisible] = useState(false)
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, opacity: visible ? 1 : 0,
      transition: 'opacity 0.2s', pointerEvents: visible ? 'auto' : 'none',
    }}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}>
      <CopyButton text={content} />
      <button onClick={onRegenerate} style={{
        background: 'rgba(255,255,255,0.1)', border: 'none', color: tokens.textSecondary,
        cursor: 'pointer', padding: '4px 8px', borderRadius: 6, fontSize: 11,
        display: 'flex', alignItems: 'center', gap: 4, transition: 'all 0.2s',
      }} onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.2)'; e.currentTarget.style.color = tokens.textPrimary }}
        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; e.currentTarget.style.color = tokens.textSecondary }}>
        <Icon size={12}>{Icons.refresh}</Icon>
        重新生成
      </button>
      <button onClick={() => onDelete(index)} style={{
        background: 'rgba(255,255,255,0.1)', border: 'none', color: tokens.textSecondary,
        cursor: 'pointer', padding: '4px 8px', borderRadius: 6, fontSize: 11,
        display: 'flex', alignItems: 'center', gap: 4, transition: 'all 0.2s',
      }} onMouseEnter={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.2)'; e.currentTarget.style.color = '#f87171' }}
        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; e.currentTarget.style.color = tokens.textSecondary }}>
        <Icon size={12}>{Icons.trash}</Icon>
        删除
      </button>
    </div>
  )
}

const MarkdownRenderer = ({ content }) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children, ...props }) {
          const match = /language-(\\w+)/.exec(className || '')
          const codeString = String(children).replace(/\\n$/, '')
          return !inline && match ? (
            <div style={{
              borderRadius: tokens.radiusMd, overflow: 'hidden', margin: '12px 0',
              border: `1px solid ${tokens.border}`, boxShadow: tokens.shadowMd, position: 'relative',
            }}>
              <div style={{
                background: 'rgba(0,0,0,0.3)', padding: '8px 16px', fontSize: 12, color: tokens.textSecondary,
                borderBottom: `1px solid ${tokens.border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#ff5f57' }} />
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#febc2e' }} />
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#28c840' }} />
                  <span style={{ marginLeft: 8, fontFamily: 'monospace' }}>{match[1]}</span>
                </div>
                <CopyButton text={codeString} />
              </div>
              <SyntaxHighlighter
                style={oneDark} language={match[1]} PreTag="div"
                customStyle={{ margin: 0, borderRadius: 0, fontSize: 13, lineHeight: 1.6, background: '#0f172a' }}
                {...props}
              >
                {codeString}
              </SyntaxHighlighter>
            </div>
          ) : (
            <code style={{
              background: 'rgba(57, 197, 187, 0.1)', color: tokens.primaryLight,
              padding: '2px 6px', borderRadius: 4, fontSize: '0.9em',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
              border: `1px solid ${tokens.primary}20`,
            }} {...props}>
              {children}
            </code>
          )
        },
        p({ children }) { return <p style={{ margin: '0.6em 0', lineHeight: 1.7 }}>{children}</p> },
        h1({ children }) { return <h1 style={{ fontSize: 20, fontWeight: 700, margin: '1em 0 0.5em', color: tokens.textPrimary, borderBottom: `1px solid ${tokens.border}`, paddingBottom: 8 }}>{children}</h1> },
        h2({ children }) { return <h2 style={{ fontSize: 17, fontWeight: 600, margin: '0.8em 0 0.4em', color: tokens.textPrimary }}>{children}</h2> },
        h3({ children }) { return <h3 style={{ fontSize: 15, fontWeight: 600, margin: '0.6em 0 0.3em', color: tokens.primaryLight }}>{children}</h3> },
        ul({ children }) { return <ul style={{ paddingLeft: 20, margin: '0.5em 0' }}>{children}</ul> },
        ol({ children }) { return <ol style={{ paddingLeft: 20, margin: '0.5em 0' }}>{children}</ol> },
        li({ children }) { return <li style={{ margin: '0.3em 0', lineHeight: 1.7 }}>{children}</li> },
        blockquote({ children }) {
          return <blockquote style={{
            borderLeft: `3px solid ${tokens.primary}`, margin: '1em 0', padding: '0.5em 1em',
            background: 'rgba(57, 197, 187, 0.05)', borderRadius: `0 ${tokens.radiusSm} ${tokens.radiusSm} 0`, color: tokens.textSecondary,
          }}>{children}</blockquote>
        },
        table({ children }) {
          return <div style={{ overflowX: 'auto', margin: '1em 0' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>{children}</table>
          </div>
        },
        th({ children }) { return <th style={{ border: `1px solid ${tokens.border}`, padding: '8px 12px', background: tokens.bgElevated, textAlign: 'left', fontWeight: 600 }}>{children}</th> },
        td({ children }) { return <td style={{ border: `1px solid ${tokens.border}`, padding: '8px 12px' }}>{children}</td> },
        hr() { return <hr style={{ border: 'none', borderTop: `1px solid ${tokens.border}`, margin: '1.5em 0' }} /> },
        a({ href, children }) { return <a href={href} target="_blank" rel="noreferrer" style={{ color: tokens.primaryLight, textDecoration: 'none', borderBottom: `1px solid ${tokens.primary}40` }}>{children}</a> },
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

/* ==================== 设置面板 ==================== */
const SettingsPanel = ({ open, onClose, settings, onChange }) => {
  if (!open) return null
  return (
    <div style={{
      position: 'fixed', inset: 0, backgroundColor: 'rgba(2, 6, 23, 0.8)',
      backdropFilter: 'blur(8px)', display: 'flex', justifyContent: 'center', alignItems: 'center',
      zIndex: 1000, animation: 'fadeIn 0.2s ease',
    }}>
      <div className="glass-strong" style={{
        borderRadius: tokens.radiusLg, padding: 32, maxWidth: 480, width: '90%',
        boxShadow: tokens.shadowMd, border: `1px solid ${tokens.border}`,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 24 }}>⚙️</span>
            设置
          </h2>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', color: tokens.textMuted,
            cursor: 'pointer', padding: 4, borderRadius: 6, display: 'flex', alignItems: 'center',
          }}>
            <Icon size={20}>{Icons.close}</Icon>
          </button>
        </div>

        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: 13, color: tokens.textSecondary, marginBottom: 8, display: 'block' }}>
            流式输出
          </label>
          <button onClick={() => onChange('stream', !settings.stream)} style={{
            width: 44, height: 24, borderRadius: 12, border: 'none', cursor: 'pointer',
            background: settings.stream ? tokens.primary : tokens.bgHover,
            position: 'relative', transition: 'all 0.3s',
          }}>
            <span style={{
              position: 'absolute', top: 2, left: settings.stream ? 22 : 2,
              width: 20, height: 20, borderRadius: '50%', background: '#fff',
              transition: 'all 0.3s', boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
            }} />
          </button>
        </div>

        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: 13, color: tokens.textSecondary, marginBottom: 8, display: 'block' }}>
            自动滚动
          </label>
          <button onClick={() => onChange('autoScroll', !settings.autoScroll)} style={{
            width: 44, height: 24, borderRadius: 12, border: 'none', cursor: 'pointer',
            background: settings.autoScroll ? tokens.primary : tokens.bgHover,
            position: 'relative', transition: 'all 0.3s',
          }}>
            <span style={{
              position: 'absolute', top: 2, left: settings.autoScroll ? 22 : 2,
              width: 20, height: 20, borderRadius: '50%', background: '#fff',
              transition: 'all 0.3s', boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
            }} />
          </button>
        </div>

        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: 13, color: tokens.textSecondary, marginBottom: 8, display: 'block' }}>
            历史消息数限制
          </label>
          <input
            type="range" min="4" max="40" value={settings.historyLimit}
            onChange={e => onChange('historyLimit', parseInt(e.target.value))}
            style={{ width: '100%', accentColor: tokens.primary }}
          />
          <div style={{ fontSize: 12, color: tokens.textMuted, marginTop: 4, textAlign: 'right' }}>
            保留最近 {settings.historyLimit} 条
          </div>
        </div>

        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: 13, color: tokens.textSecondary, marginBottom: 8, display: 'block' }}>
            主题色
          </label>
          <div style={{ display: 'flex', gap: 8 }}>
            {['#39c5bb', '#a78bfa', '#f472b6', '#60a5fa', '#fbbf24'].map(color => (
              <button key={color} onClick={() => onChange('themeColor', color)} style={{
                width: 32, height: 32, borderRadius: '50%', background: color,
                border: settings.themeColor === color ? '2px solid #fff' : '2px solid transparent',
                cursor: 'pointer', boxShadow: settings.themeColor === color ? `0 0 12px ${color}` : 'none',
              }} />
            ))}
          </div>
        </div>

        <button onClick={onClose} style={{
          marginTop: 8, width: '100%', padding: '10px 24px',
          background: `linear-gradient(135deg, ${tokens.primary}, ${tokens.primaryDark})`,
          color: '#fff', border: 'none', borderRadius: tokens.radiusMd,
          cursor: 'pointer', fontSize: 14, fontWeight: 500, boxShadow: tokens.shadowGlow,
        }}>
          完成
        </button>
      </div>
    </div>
  )
}

/* ==================== 快捷功能卡片 ==================== */
const QuickCards = ({ onSelect }) => {
  const cards = [
    { icon: '⚡', title: '代码生成', desc: '编写、解释或优化代码', color: tokens.primary },
    { icon: '📄', title: '文档总结', desc: '快速提炼长文核心要点', color: '#a78bfa' },
    { icon: '🔍', title: '知识问答', desc: '基于知识库检索回答', color: '#60a5fa' },
    { icon: '✍️', title: '创意写作', desc: '撰写文章、故事或文案', color: '#f472b6' },
    { icon: '📊', title: '数据分析', desc: '解读数据并生成图表', color: '#34d399' },
    { icon: '🌐', title: '翻译助手', desc: '多语言精准翻译', color: '#fbbf24' },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12, marginTop: 32, maxWidth: 600 }}>
      {cards.map((card, i) => (
        <button key={i} onClick={() => onSelect(card.title)} className="card-hover glass" style={{
          padding: '16px', borderRadius: tokens.radiusMd, textAlign: 'left',
          border: `1px solid ${tokens.border}`, cursor: 'pointer', background: tokens.bgSurface,
          display: 'flex', flexDirection: 'column', gap: 8, animation: `fadeInScale 0.4s ease ${i * 0.05}s both`,
        }}>
          <span style={{ fontSize: 24 }}>{card.icon}</span>
          <div style={{ fontSize: 14, fontWeight: 600, color: tokens.textPrimary }}>{card.title}</div>
          <div style={{ fontSize: 12, color: tokens.textMuted, lineHeight: 1.5 }}>{card.desc}</div>
        </button>
      ))}
    </div>
  )
}
/* ==================== 审查日志组件（新增） ==================== */
const ReviewLog = ({ text }) => {
  const parts = text.split('[三角形协作日志]')
  const mainContent = parts[0]
  const logText = parts.length > 1 ? parts[1] : ''
  const isPassed = logText.includes('✓ 通过')
  const isRevised = logText.includes('未通过')

  return (
    <div>
      <MarkdownRenderer content={mainContent} />
      <details style={{
        marginTop: 12,
        background: 'rgba(57,197,187,0.05)',
        border: `1px solid ${tokens.border}`,
        borderRadius: tokens.radiusMd,
        padding: 12,
      }}>
        <summary style={{
          cursor: 'pointer',
          fontWeight: 600,
          color: isPassed ? '#34d399' : isRevised ? '#fbbf24' : tokens.textSecondary,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <span style={{ fontSize: 14 }}>🔍</span>
          三角形协作审查报告
          <span style={{
            fontSize: 11,
            marginLeft: 'auto',
            background: isPassed ? 'rgba(52,211,153,0.2)' : isRevised ? 'rgba(251,191,36,0.2)' : 'rgba(148,163,184,0.2)',
            padding: '2px 10px',
            borderRadius: 12,
          }}>
            {isPassed ? '✅ 通过' : isRevised ? '⚠️ 修订' : '📋 详情'}
          </span>
        </summary>
        <div style={{
          marginTop: 8,
          fontSize: 13,
          color: tokens.textSecondary,
          whiteSpace: 'pre-wrap',
          fontFamily: 'monospace',
        }}>
          {logText}
        </div>
      </details>
    </div>
  )
}

/* ==================== App 主组件 ==================== */
function App() {
  const { sessions, currentId, current, loaded, addMessage, updateLastAssistant, deleteMessage, createSession, deleteSession, switchSession } = useBackendSessions()
  const health = useHealthCheck()
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [filePath, setFilePath] = useState('')
  const [uploadStatus, setUploadStatus] = useState('')
  const [evalData, setEvalData] = useState(null)
  const [showEval, setShowEval] = useState(false)
  const [mode, setMode] = useState('4b')
  const [showModeMenu, setShowModeMenu] = useState(false)
  const [hoveredSessionId, setHoveredSessionId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState({ stream: true, autoScroll: true, historyLimit: 20, themeColor: tokens.primary })
  const [pendingImages, setPendingImages] = useState([])
  const [isDragging, setIsDragging] = useState(false)

  const messagesEndRef = useRef(null)
  const modeRef = useRef(null)
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => { if (settings.autoScroll) messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [current.messages])

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (modeRef.current && !modeRef.current.contains(event.target)) setShowModeMenu(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // 图片粘贴
  useEffect(() => {
    const handlePaste = (e) => {
      const items = e.clipboardData?.items
      if (!items) return
      const images = []
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (file) images.push(file)
        }
      }
      if (images.length > 0) {
        e.preventDefault()
        handleImageFiles(images)
      }
    }
    document.addEventListener('paste', handlePaste)
    return () => document.removeEventListener('paste', handlePaste)
  }, [])

  const handleImageFiles = (files) => {
    const newImages = []
    files.forEach(file => {
      const reader = new FileReader()
      reader.onload = (e) => {
        newImages.push(e.target.result)
        if (newImages.length === files.length) {
          setPendingImages(prev => [...prev, ...newImages])
        }
      }
      reader.readAsDataURL(file)
    })
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
    if (files.length > 0) handleImageFiles(files)
  }

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true) }
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false) }

  const removePendingImage = (idx) => {
    setPendingImages(prev => prev.filter((_, i) => i !== idx))
  }

  const getModelName = () => mode === '4b' ? 'qwen3.5:4b' : 'qwen3.5:9b'

  const sendMessage = async () => {
    const text = input.trim()
    if ((!text && pendingImages.length === 0) || loading) return

    addMessage('user', text, pendingImages)
    setInput('')
    const imagesToSend = [...pendingImages]
    setPendingImages([])
    setLoading(true)

    const history = current.messages.slice(-settings.historyLimit).map(m => ({ role: m.role, content: m.content }))

    const doRequest = async (url, body) => {
      try {
        const response = await fetch(url, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) {
        updateLastAssistant('❌ 请求失败：' + err.message)
      } finally { setLoading(false) }
    }

    if (mode === 'selfrag') {
      await doRequest(`${API_BASE}/chat/selfrag`, { message: text, history, model: 'qwen3.5:9b', images: imagesToSend })
      return
    }
    if (mode === 'hyde') {
      await doRequest(`${API_BASE}/chat/hyde`, { message: text, history, model: 'qwen3.5:9b', images: imagesToSend })
      return
    }
    if (mode === 'agent') {
      await doRequest(`${API_BASE}/agent`, { message: text, history, images: imagesToSend })
      return
    }
    if (mode === 'review') {
      await doRequest(`${API_BASE}/chat/review`, { message: text, sessionId: currentId, history, models: ['qwen3.5:9b', 'qwen3.5:9b', 'qwen3.5:9b'], images: imagesToSend })
      return
    }

    const model = getModelName()
    if (!settings.stream) {
      await doRequest(`${API_BASE}/chat`, { message: text, sessionId: currentId, history, model, images: imagesToSend })
      return
    }

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, sessionId: currentId, history, model, images: imagesToSend }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let full = ''
      let streamEnded = false
      while (!streamEnded) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n').filter(l => l.startsWith('data: '))
        for (const line of lines) {
          const data = line.slice(6)
          if (data === '[DONE]') { streamEnded = true; break }
          try {
            const parsed = JSON.parse(data)
            if (parsed.token) { full += parsed.token; updateLastAssistant(full) }
          } catch (e) {}
        }
      }
      if (!full) updateLastAssistant('（助手没有返回内容）')
    } catch (err) {
      updateLastAssistant('❌ 请求失败：' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRegenerate = async (msgIndex) => {
    const msgs = current.messages
    let userMsg = null
    for (let i = msgIndex - 1; i >= 0; i--) {
      if (msgs[i].role === 'user') { userMsg = msgs[i]; break }
    }
    if (!userMsg) return
    setInput(userMsg.content)
    setPendingImages(userMsg.images || [])
    deleteMessage(msgIndex)
    setTimeout(() => sendMessage(), 100)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  const uploadDoc = async () => {
    if (!filePath.trim()) return
    setUploadStatus('上传中...')
    try {
      const res = await fetch(`${API_BASE}/add_docs`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([filePath.trim()]),
      })
      const data = await res.json()
      setUploadStatus(data.status === 'success' ? '✅ 上传成功' : '❌ 失败：' + data.message)
    } catch (e) { setUploadStatus('❌ 上传失败：' + e.message) }
  }

  const fetchEvalReport = async (e) => {
    if (e) e.stopPropagation()
    try {
      const res = await fetch(`${API_BASE}/eval/report`)
      const data = await res.json()
      if (data.message) { alert(data.message); return }
      setEvalData(data)
      setShowEval(true)
    } catch (err) { alert('获取评估报告失败: ' + err.message) }
  }

  const modes = [
    { value: '4b', label: '轻量对话', sub: 'Qwen3.5 4B', icon: '⚡' },
    { value: '9b', label: '深度推理', sub: 'Qwen3.5 9B', icon: '🧠' },
    { value: 'review', label: '三模型审查', sub: 'Review', icon: '🔍' },
    { value: 'agent', label: 'Agent 模式', sub: 'ReAct', icon: '🤖' },
    { value: 'hyde', label: 'HyDE 检索', sub: 'Hypothetical', icon: '🔮' },
    { value: 'selfrag', label: 'Self-RAG', sub: 'Adaptive', icon: '📚' },
  ]

  const activeMode = modes.find(m => m.value === mode)

  const handleQuickSelect = (title) => {
    const hints = {
      '代码生成': '请帮我写一段 Python 代码，实现一个快速排序算法',
      '文档总结': '请总结以下文档的核心观点：\n\n',
      '知识问答': '什么是 Self-RAG 技术？请详细解释其工作原理',
      '创意写作': '帮我写一篇关于未来 AI 发展的短故事',
      '数据分析': '请分析这组数据并给出可视化建议',
      '翻译助手': '请将以下内容翻译成英文：',
    }
    setInput(hints[title] || '')
    textareaRef.current?.focus()
  }

  if (!loaded) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: tokens.bgBase, color: tokens.textSecondary,
        fontFamily: "'Inter', 'Noto Sans SC', sans-serif",
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            width: 48, height: 48, borderRadius: 16,
            background: `linear-gradient(135deg, ${tokens.primary}, ${tokens.primaryLight})`,
            margin: '0 auto 16px', animation: 'pulse-glow 2s infinite',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24,
          }}>🎵</div>
          <div>正在加载会话...</div>
        </div>
      </div>
    )
  }

  return (
    <div style={{
      display: 'flex', height: '100vh',
      fontFamily: "'Inter', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif",
      background: tokens.bgBase, color: tokens.textPrimary, overflow: 'hidden',
    }}>
      <GlobalStyles />
      <div style={{
        position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none',
        background: `
          radial-gradient(ellipse 80% 50% at 50% -20%, rgba(57, 197, 187, 0.15), transparent),
          radial-gradient(ellipse 60% 40% at 80% 80%, rgba(57, 197, 187, 0.08), transparent),
          radial-gradient(ellipse 50% 30% at 20% 60%, rgba(102, 204, 255, 0.06), transparent)
        `,
      }} />
      {/* 侧边栏 */}
      <aside className="glass-strong" style={{
        width: sidebarOpen ? 280 : 0, minWidth: sidebarOpen ? 280 : 0, height: '100vh',
        display: 'flex', flexDirection: 'column', position: 'relative', zIndex: 10,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', overflow: 'hidden',
        borderRight: `1px solid ${tokens.border}`,
      }}>
        <div style={{ padding: '24px 20px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 12,
            background: `linear-gradient(135deg, ${tokens.primary}, ${tokens.primaryLight})`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: `0 0 16px ${tokens.primaryGlow}`, fontSize: 18,
          }}>🎵</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: '-0.02em' }}>Miku AI</div>
            <div style={{ fontSize: 11, color: tokens.textMuted, marginTop: 2 }}>初音未来 · 智能助手</div>
          </div>
        </div>

        <div style={{ padding: '0 16px 12px' }}>
          <button onClick={createSession} style={{
            width: '100%', padding: '10px 14px', borderRadius: tokens.radiusMd,
            background: 'rgba(57, 197, 187, 0.1)', border: `1px solid ${tokens.primary}30`,
            color: tokens.primaryLight, cursor: 'pointer', fontSize: 13, fontWeight: 500,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, transition: 'all 0.2s',
          }} onMouseEnter={e => { e.currentTarget.style.background = 'rgba(57, 197, 187, 0.2)'; e.currentTarget.style.boxShadow = tokens.shadowGlow }}
            onMouseLeave={e => { e.currentTarget.style.background = 'rgba(57, 197, 187, 0.1)'; e.currentTarget.style.boxShadow = 'none' }}>
            <Icon size={16}>{Icons.plus}</Icon> 新建对话
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '4px 12px' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: tokens.textMuted, padding: '8px 8px 4px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>最近对话</div>
          {Object.values(sessions).map((s, idx) => (
            <div key={s.id} className="sidebar-item" onClick={() => switchSession(s.id)}
              onMouseEnter={() => setHoveredSessionId(s.id)} onMouseLeave={() => setHoveredSessionId(null)}
              style={{
                padding: '10px 12px', borderRadius: tokens.radiusMd, cursor: 'pointer', fontSize: 13,
                color: s.id === currentId ? tokens.textPrimary : tokens.textSecondary,
                background: s.id === currentId ? 'rgba(57, 197, 187, 0.15)' : 'transparent',
                border: s.id === currentId ? `1px solid ${tokens.primary}30` : '1px solid transparent',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4,
                transition: 'all 0.15s ease', animationDelay: `${idx * 0.03}s`,
              }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, overflow: 'hidden' }}>
                <Icon size={14} style={{ flexShrink: 0, opacity: 0.6 }}>{Icons.message}</Icon>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.title || '新对话'}</span>
              </div>
              <button onClick={(e) => { e.stopPropagation(); deleteSession(s.id) }} style={{
                background: 'none', border: 'none', color: tokens.textMuted, cursor: 'pointer',
                padding: 4, borderRadius: 6, display: hoveredSessionId === s.id ? 'flex' : 'none', alignItems: 'center', transition: 'all 0.15s',
              }} onMouseEnter={e => e.currentTarget.style.color = '#ef4444'} onMouseLeave={e => e.currentTarget.style.color = tokens.textMuted}>
                <Icon size={14}>{Icons.trash}</Icon>
              </button>
            </div>
          ))}
        </div>

        <div style={{ padding: '12px 16px', borderTop: `1px solid ${tokens.border}` }} ref={modeRef}>
          <div style={{ fontSize: 11, fontWeight: 600, color: tokens.textMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>问答模式</div>
          <button onClick={() => setShowModeMenu(!showModeMenu)} style={{
            width: '100%', padding: '10px 12px', borderRadius: tokens.radiusMd,
            background: tokens.bgHover, border: `1px solid ${tokens.border}`, color: tokens.textPrimary,
            cursor: 'pointer', fontSize: 13, display: 'flex', alignItems: 'center', justifyContent: 'space-between', transition: 'all 0.2s',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 16 }}>{activeMode?.icon}</span>
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontWeight: 500 }}>{activeMode?.label}</div>
                <div style={{ fontSize: 11, color: tokens.textMuted }}>{activeMode?.sub}</div>
              </div>
            </div>
            <Icon size={16} style={{ color: tokens.textMuted, transform: showModeMenu ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>{Icons.chevronDown}</Icon>
          </button>
          {showModeMenu && (
            <div style={{ marginTop: 8, borderRadius: tokens.radiusMd, background: tokens.bgElevated, border: `1px solid ${tokens.border}`, overflow: 'hidden', boxShadow: tokens.shadowMd }}>
              {modes.map(m => (
                <div key={m.value} onClick={() => { setMode(m.value); setShowModeMenu(false) }} style={{
                  padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer',
                  fontSize: 13, color: mode === m.value ? tokens.primaryLight : tokens.textSecondary,
                  background: mode === m.value ? 'rgba(57, 197, 187, 0.1)' : 'transparent',
                  borderLeft: mode === m.value ? `3px solid ${tokens.primary}` : '3px solid transparent', transition: 'all 0.15s',
                }}>
                  <span style={{ fontSize: 16, width: 20, textAlign: 'center' }}>{m.icon}</span>
                  <div><div style={{ fontWeight: 500 }}>{m.label}</div><div style={{ fontSize: 11, color: tokens.textMuted }}>{m.sub}</div></div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ padding: '12px 16px', borderTop: `1px solid ${tokens.border}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: tokens.textMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>知识库</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <input type="text" value={filePath} onChange={e => setFilePath(e.target.value)} placeholder="/docs/manual.pdf"
              style={{ flex: 1, padding: '8px 12px', borderRadius: tokens.radiusSm, border: `1px solid ${tokens.border}`, background: tokens.bgHover, color: tokens.textPrimary, fontSize: 12, outline: 'none', fontFamily: 'inherit' }} />
            <button onClick={uploadDoc} style={{ padding: '8px 12px', borderRadius: tokens.radiusSm, background: tokens.primary, border: 'none', color: '#fff', cursor: 'pointer', fontSize: 12, fontWeight: 500 }} onMouseEnter={e => e.currentTarget.style.background = tokens.primaryLight}>
              <Icon size={16}>{Icons.file}</Icon>
            </button>
          </div>
          {uploadStatus && <div style={{ fontSize: 11, color: uploadStatus.includes('✅') ? tokens.primaryLight : '#f87171', marginTop: 6 }}>{uploadStatus}</div>}
        </div>

        <div style={{ padding: '12px 16px', borderTop: `1px solid ${tokens.border}` }}>
          <button onClick={fetchEvalReport} style={{
            width: '100%', padding: '10px 14px', borderRadius: tokens.radiusMd, background: 'transparent',
            border: `1px solid ${tokens.border}`, color: tokens.textSecondary, cursor: 'pointer', fontSize: 13,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, transition: 'all 0.2s',
          }} onMouseEnter={e => { e.currentTarget.style.borderColor = tokens.primary; e.currentTarget.style.color = tokens.primaryLight }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = tokens.border; e.currentTarget.style.color = tokens.textSecondary }}>
            <Icon size={16}>{Icons.chart}</Icon> 查看评估报告
          </button>
        </div>
        <div style={{ padding: '12px 16px', borderTop: `1px solid ${tokens.border}`, fontSize: 11, color: tokens.textMuted, textAlign: 'center' }}>{Object.keys(sessions).length} 个会话 · Miku AI</div>
      </aside>

      {/* 主聊天区 */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', position: 'relative', zIndex: 5 }}>
        <header className="glass" style={{ padding: '14px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <button onClick={() => setSidebarOpen(!sidebarOpen)} style={{ background: 'none', border: 'none', color: tokens.textSecondary, cursor: 'pointer', padding: 6, borderRadius: 8, display: 'flex', alignItems: 'center' }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
            </button>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, color: tokens.textPrimary }}>{current.title}</div>
              <div style={{ fontSize: 12, color: tokens.textMuted, marginTop: 2 }}>{current.messages.length} 条消息</div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <HealthIndicator health={health} />
            <ModeBadge mode={mode} />
            <button onClick={() => setShowSettings(true)} style={{ background: 'none', border: 'none', color: tokens.textSecondary, cursor: 'pointer', padding: 6, borderRadius: 8, display: 'flex', alignItems: 'center' }} onMouseEnter={e => e.currentTarget.style.color = tokens.textPrimary} onMouseLeave={e => e.currentTarget.style.color = tokens.textSecondary}>
              <Icon size={18}>{Icons.settings}</Icon>
            </button>
          </div>
        </header>

        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 0', position: 'relative' }} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
          {isDragging && (
            <div style={{ position: 'absolute', inset: 0, zIndex: 50, border: `2px dashed ${tokens.primary}`, borderRadius: tokens.radiusLg, background: 'rgba(57, 197, 187, 0.1)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', animation: 'dropPulse 1.5s infinite', pointerEvents: 'none' }}>
              <div style={{ textAlign: 'center', color: tokens.primaryLight }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>📸</div>
                <div style={{ fontSize: 18, fontWeight: 600 }}>释放以上传图片</div>
              </div>
            </div>
          )}

          {current.messages.length === 0 && (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: tokens.textSecondary, animation: 'fadeIn 0.6s ease' }}>
              <div style={{ width: 80, height: 80, borderRadius: 24, background: `linear-gradient(135deg, ${tokens.primary}20, ${tokens.primaryLight}10)`, border: `1px solid ${tokens.primary}20`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 40, marginBottom: 24, boxShadow: `0 0 40px ${tokens.primaryGlow}` }}>🎵</div>
              <div style={{ fontSize: 24, fontWeight: 700, marginBottom: 8, color: tokens.textPrimary }}>Miku AI</div>
              <div style={{ fontSize: 14, color: tokens.textMuted }}>基于 Qwen3.5 的智能助手，随时为你效劳</div>
              <QuickCards onSelect={handleQuickSelect} />
            </div>
          )}

          <div style={{ maxWidth: 800, margin: '0 auto', padding: '0 24px' }}>
            {current.messages.map((msg, i) => (
              <div key={i} className="msg-enter" style={{ display: 'flex', gap: 16, marginBottom: 24, justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
                <div style={{ width: 32, height: 32, borderRadius: 10, flexShrink: 0, background: msg.role === 'user' ? `linear-gradient(135deg, ${tokens.primary}, ${tokens.primaryLight})` : tokens.bgElevated, display: 'flex', alignItems: 'center', justifyContent: 'center', border: msg.role === 'user' ? 'none' : `1px solid ${tokens.border}`, boxShadow: msg.role === 'user' ? `0 0 12px ${tokens.primaryGlow}` : 'none', fontSize: 14 }}>{msg.role === 'user' ? '👤' : '🎵'}</div>
                <div style={{ maxWidth: 'calc(100% - 80px)', padding: msg.role === 'user' ? '12px 18px' : '0', fontSize: 14, lineHeight: 1.7, wordBreak: 'break-word', ...(msg.role === 'user' ? { background: `linear-gradient(135deg, ${tokens.primary}, ${tokens.primaryDark})`, color: '#fff', borderRadius: '18px 18px 4px 18px', boxShadow: `0 4px 16px ${tokens.primaryGlow}` } : { background: 'transparent', color: tokens.textPrimary }) }}>
                  {msg.role === 'assistant' ? (
                    <div className="glass" style={{ padding: '16px 20px', borderRadius: '4px 18px 18px 18px', border: `1px solid ${tokens.border}` }}>
                      {msg.content.includes('[三角形协作日志]') ? (
                        <ReviewLog text={msg.content} />
                      ) : (
                        <MarkdownRenderer content={msg.content} />
                      )}
                      <MessageActions content={msg.content} onRegenerate={() => handleRegenerate(i)} onDelete={deleteMessage} index={i} />
                      {i === current.messages.length - 1 && loading && <span className="cursor-blink" style={{ display: 'inline-block', width: 2, height: 18, background: tokens.primary, verticalAlign: 'text-bottom', marginLeft: 4, borderRadius: 1 }} />}
                    </div>
                  ) : (
                    <div>{msg.content}{msg.images && msg.images.length > 0 && <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>{msg.images.map((img, idx) => <img key={idx} src={img} alt="" style={{ maxWidth: 200, maxHeight: 200, borderRadius: tokens.radiusSm, border: `1px solid ${tokens.border}` }} />)}</div>}</div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
        {/* 输入区 */}
        <div className="glass-strong" style={{ padding: '16px 24px 24px', flexShrink: 0 }}>
          <div style={{ maxWidth: 800, margin: '0 auto', position: 'relative' }}>
            {pendingImages.length > 0 && (
              <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                {pendingImages.map((img, idx) => (
                  <div key={idx} style={{ position: 'relative' }}>
                    <img src={img} alt="" style={{ width: 64, height: 64, objectFit: 'cover', borderRadius: tokens.radiusSm, border: `1px solid ${tokens.border}` }} />
                    <button onClick={() => removePendingImage(idx)} style={{ position: 'absolute', top: -6, right: -6, width: 18, height: 18, borderRadius: '50%', background: '#ef4444', border: 'none', color: '#fff', cursor: 'pointer', fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>×</button>
                  </div>
                ))}
              </div>
            )}
            <div style={{ position: 'relative' }}>
              <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="输入消息，Shift+Enter 换行，Ctrl+V 粘贴图片..." rows={1}
                style={{ width: '100%', padding: '14px 96px 14px 18px', borderRadius: tokens.radiusXl, border: `1px solid ${tokens.border}`, background: tokens.bgBase, color: tokens.textPrimary, fontSize: 14, fontFamily: 'inherit', resize: 'none', outline: 'none', lineHeight: 1.6, minHeight: 52, maxHeight: 200, boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.2)', transition: 'all 0.2s' }}
                onFocus={e => { e.currentTarget.style.borderColor = tokens.primary; e.currentTarget.style.boxShadow = `inset 0 2px 4px rgba(0,0,0,0.2), 0 0 0 3px ${tokens.primary}15` }}
                onBlur={e => { e.currentTarget.style.borderColor = tokens.border; e.currentTarget.style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,0.2)' }} />
              <button onClick={() => fileInputRef.current?.click()} style={{ position: 'absolute', right: 52, bottom: 8, width: 36, height: 36, borderRadius: '50%', background: 'transparent', border: 'none', color: tokens.textSecondary, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }} onMouseEnter={e => e.currentTarget.style.color = tokens.primaryLight} onMouseLeave={e => e.currentTarget.style.color = tokens.textSecondary}>
                <Icon size={18}>{Icons.image}</Icon>
              </button>
              <input ref={fileInputRef} type="file" accept="image/*" multiple style={{ display: 'none' }} onChange={e => { const files = Array.from(e.target.files || []); if (files.length > 0) handleImageFiles(files); e.target.value = '' }} />
              <button onClick={sendMessage} disabled={loading || (!input.trim() && pendingImages.length === 0)} style={{ position: 'absolute', right: 8, bottom: 8, width: 36, height: 36, borderRadius: '50%', background: (loading || (!input.trim() && pendingImages.length === 0)) ? tokens.bgHover : tokens.primary, border: 'none', color: '#fff', cursor: (loading || (!input.trim() && pendingImages.length === 0)) ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: (!loading && (input.trim() || pendingImages.length > 0)) ? `0 0 12px ${tokens.primaryGlow}` : 'none' }}>
                <Icon size={18}>{Icons.send}</Icon>
              </button>
            </div>
          </div>
          <div style={{ textAlign: 'center', marginTop: 8, fontSize: 11, color: tokens.textMuted }}>Miku AI 可能会生成不准确的信息，请核实重要信息 · 支持 Ctrl+V 粘贴图片</div>
        </div>
      </main>

      <SettingsPanel open={showSettings} onClose={() => setShowSettings(false)} settings={settings} onChange={(key, value) => setSettings(prev => ({ ...prev, [key]: value }))} />

      {showEval && evalData && (
        <div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(2, 6, 23, 0.8)', backdropFilter: 'blur(8px)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000, animation: 'fadeIn 0.2s ease' }}>
          <div className="glass-strong" style={{ borderRadius: tokens.radiusLg, padding: 32, maxWidth: 480, width: '90%', boxShadow: tokens.shadowMd, border: `1px solid ${tokens.border}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
              <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 10 }}><span style={{ fontSize: 24 }}>📊</span> RAG 系统质量报告</h2>
              <button onClick={() => setShowEval(false)} style={{ background: 'none', border: 'none', color: tokens.textMuted, cursor: 'pointer', padding: 4, borderRadius: 6, display: 'flex', alignItems: 'center' }}><Icon size={20}>{Icons.close}</Icon></button>
            </div>
            <p style={{ color: tokens.textSecondary, fontSize: 13, marginBottom: 20 }}>评估时间：{evalData.timestamp}</p>
            {[
              { key: 'avg_faithfulness', label: '答案忠实度', icon: '🎯' },
              { key: 'avg_context_recall', label: '上下文召回率', icon: '🔁' },
              { key: 'avg_context_precision', label: '上下文精确度', icon: '🎯' },
            ].map(item => {
              const val = evalData[item.key]
              const pct = typeof val === 'number' ? (val * 100).toFixed(1) + '%' : 'N/A'
              const num = typeof val === 'number' ? val : 0
              return (
                <div key={item.key} style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontSize: 13, color: tokens.textSecondary }}>{item.icon} {item.label}</span>
                    <span style={{ fontSize: 15, fontWeight: 700, color: tokens.primaryLight }}>{pct}</span>
                  </div>
                  <div style={{ height: 6, borderRadius: 3, background: tokens.bgHover, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${num * 100}%`, borderRadius: 3, background: `linear-gradient(90deg, ${tokens.primary}, ${tokens.primaryLight})`, transition: 'width 1s ease', boxShadow: `0 0 8px ${tokens.primaryGlow}` }} />
                  </div>
                </div>
              )
            })}
            <button onClick={() => setShowEval(false)} style={{ marginTop: 8, width: '100%', padding: '10px 24px', background: `linear-gradient(135deg, ${tokens.primary}, ${tokens.primaryDark})`, color: '#fff', border: 'none', borderRadius: tokens.radiusMd, cursor: 'pointer', fontSize: 14, fontWeight: 500, boxShadow: tokens.shadowGlow }}>关闭</button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
