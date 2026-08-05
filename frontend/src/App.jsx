import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

const STORAGE_KEY = 'chat_sessions'

// ==================== 自定义 Hook：会话管理（不变） ====================
function useSessions() {
  const [sessions, setSessions] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const data = JSON.parse(saved)
        if (Object.keys(data).length > 0) return data
      }
    } catch (e) {}
    const id = 'session_' + Date.now()
    return { [id]: { id, title: '新对话', messages: [] } }
  })

  const [currentId, setCurrentId] = useState(() => Object.keys(sessions)[0])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
  }, [sessions])

  const current = sessions[currentId] || { id: currentId, title: '新对话', messages: [] }

  const addMessage = (role, content) => {
    setSessions(prev => {
      const s = { ...prev }
      s[currentId] = { ...s[currentId], messages: [...s[currentId].messages, { role, content }] }
      if (role === 'user' && s[currentId].messages.length === 1) {
        s[currentId].title = content.length > 20 ? content.slice(0, 20) + '...' : content
      }
      return s
    })
  }
  const updateLastAssistant = (fullContent) => {
    setSessions(prev => {
      const s = { ...prev }
      const msgs = [...s[currentId].messages]
      if (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') {
        msgs[msgs.length - 1] = { role: 'assistant', content: fullContent }
      } else { msgs.push({ role: 'assistant', content: fullContent }) }
      s[currentId] = { ...s[currentId], messages: msgs }
      return s
    })
  }
  const createSession = () => {
    const id = 'session_' + Date.now()
    setSessions(prev => ({ ...prev, [id]: { id, title: '新对话', messages: [] } }))
    setCurrentId(id)
  }
  const deleteSession = (id) => {
    setSessions(prev => {
      const newSessions = { ...prev }
      delete newSessions[id]
      const keys = Object.keys(newSessions)
      if (keys.length === 0) {
        const newId = 'session_' + Date.now()
        return { [newId]: { id: newId, title: '新对话', messages: [] } }
      }
      if (currentId === id) setCurrentId(keys[0])
      return newSessions
    })
  }
  const switchSession = (id) => setCurrentId(id)
  return { sessions, currentId, current, addMessage, updateLastAssistant, createSession, deleteSession, switchSession }
}

// ==================== App 主组件 ====================
function App() {
  const { sessions, currentId, current, addMessage, updateLastAssistant, createSession, deleteSession, switchSession } = useSessions()
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [filePath, setFilePath] = useState('')
  const [uploadStatus, setUploadStatus] = useState('')
  const [evalData, setEvalData] = useState(null)
  const [showEval, setShowEval] = useState(false)
  const [mode, setMode] = useState('4b')
  const [showModeMenu, setShowModeMenu] = useState(false)
  const [hoveredSessionId, setHoveredSessionId] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const messagesEndRef = useRef(null)
  const modeRef = useRef(null)

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [current.messages])

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (modeRef.current && !modeRef.current.contains(event.target)) {
        setShowModeMenu(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // ========== 模型映射 ==========
  const getModelName = () => {
    if (mode === '4b') return 'qwen3.5:4b'
    return 'qwen3.5:9b'
  }

  const displayModelName = () => {
    const m = getModelName()
    if (m === '4b') return '4B · 轻量'
    if (m === '9b') return '9B · 深度'
    if (m === 'review') return '审查模式'
    if (m === 'agent') return 'Agent'
    if (m === 'hyde') return 'HyDE'
    if (m === 'selfrag') return 'Self-RAG'
    return '未知'
  }

  // ========== 发送消息（保持不变） ==========
  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    addMessage('user', text)
    setInput('')
    setLoading(true)

    const history = current.messages.slice(-20).map(m => ({
      role: m.role,
      content: m.content
    }))

    if (mode === 'selfrag') {
      try {
        const response = await fetch('http://localhost:8000/chat/selfrag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, history: history, model: 'qwen3.5:9b' })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) {
        updateLastAssistant('❌ Self-RAG 请求失败：' + err.message)
      } finally { setLoading(false) }
      return
    }

    if (mode === 'hyde') {
      try {
        const response = await fetch('http://localhost:8000/chat/hyde', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, history: history, model: 'qwen3.5:9b' })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) {
        updateLastAssistant('❌ HyDE 请求失败：' + err.message)
      } finally { setLoading(false) }
      return
    }

    if (mode === 'agent') {
      try {
        const response = await fetch('http://localhost:8000/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, history: history })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) {
        updateLastAssistant('❌ Agent 请求失败：' + err.message)
      } finally { setLoading(false) }
      return
    }

    if (mode === 'review') {
      try {
        const response = await fetch('http://localhost:8000/chat/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            history: history,
            models: ['qwen3.5:9b', 'qwen3.5:9b', 'qwen3.5:9b']
          })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) {
        updateLastAssistant('❌ 审查请求失败：' + err.message)
      } finally { setLoading(false) }
      return
    }

    // 流式
    const model = getModelName()
    try {
      const response = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: history, model: model })
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let full = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n').filter(l => l.startsWith('data: '))
        for (const line of lines) {
          const data = line.slice(6)
          if (data === '[DONE]') break
          try {
            const parsed = JSON.parse(data)
            if (parsed.token) {
              full += parsed.token
              updateLastAssistant(full)
            }
          } catch (e) {}
        }
      }
      if (!full) full = '（助手没有返回内容）'
    } catch (err) {
      updateLastAssistant('❌ 请求失败：' + err.message)
    } finally { setLoading(false) }
  }

  const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }

  const uploadDoc = async () => {
    if (!filePath.trim()) return
    setUploadStatus('上传中...')
    try {
      const res = await fetch('http://localhost:8000/add_docs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([filePath.trim()]) })
      const data = await res.json()
      setUploadStatus(data.status === 'success' ? '✅ 上传成功' : '❌ 失败：' + data.message)
    } catch (e) { setUploadStatus('❌ 上传失败：' + e.message) }
  }

  const fetchEvalReport = async () => {
    try {
      const res = await fetch('http://localhost:8000/eval/report')
      const data = await res.json()
      if (data.message) { alert(data.message); return }
      setEvalData(data)
      setShowEval(true)
    } catch (err) { alert('获取评估报告失败: ' + err.message) }
  }

  const mikuCyan = '#39c5bb'
  const mikuLightCyan = '#66ccff'

  const iconSmall = { width: 16, height: 16, borderRadius: '50%', verticalAlign: 'middle', marginRight: 6, objectFit: 'cover' }
  const iconMedium = { width: 22, height: 22, borderRadius: '50%', marginRight: 8, objectFit: 'cover' }

  const modes = [
    { value: '4b', label: '轻量对话 (4B)', img: 'miku-light.jpg' },
    { value: '9b', label: '深度推理 (9B)', img: 'miku-deep.jpg' },
    { value: 'review', label: '三模型审查', img: 'miku-review.jpg' },
    { value: 'agent', label: 'Agent 模式', img: 'miku-agent.jpg' },
    { value: 'hyde', label: 'HyDE 检索', img: 'miku-hyde.jpg' },
    { value: 'selfrag', label: 'Self-RAG', img: 'miku-selfrag.jpg' }
  ]

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: "'Inter', 'Noto Sans SC', system-ui, sans-serif", background: '#0f172a', backgroundImage: 'radial-gradient(circle at 20% 20%, rgba(57,197,187,0.15) 0%, transparent 40%), radial-gradient(circle at 80% 60%, rgba(102,204,255,0.1) 0%, transparent 40%)', color: '#e2e8f0', overflow: 'hidden' }}>
      {/* ===== 侧边栏 ===== */}
      <div style={{
        width: sidebarCollapsed ? 0 : 280,
        transition: 'width 0.3s ease',
        background: 'rgba(15, 23, 42, 0.85)',
        backdropFilter: 'blur(15px)',
        borderRight: '1px solid rgba(57,197,187,0.2)',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        height: '100vh',
        overflow: 'hidden'
      }}>
        {!sidebarCollapsed && (
          <>
            <div style={{ padding: '20px 16px', textAlign: 'center', borderBottom: '1px solid rgba(57,197,187,0.2)' }}>
              <img src="/miku-logo.jpg" alt="Logo" style={{ width: 44, height: 44, borderRadius: '50%', objectFit: 'cover', boxShadow: '0 0 15px rgba(57,197,187,0.4)' }} />
              <div style={{ fontSize: 16, fontWeight: 700, color: '#fff', marginTop: 8 }}>Miku AI</div>
              <div style={{ fontSize: 12, color: mikuCyan, opacity: 0.8 }}>初音未来 智能助手</div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px' }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: '#cbd5e1' }}>对话列表</span>
              <button onClick={createSession} style={{ background: 'rgba(57,197,187,0.15)', border: `1px solid ${mikuCyan}`, color: '#fff', padding: '3px 12px', borderRadius: 20, cursor: 'pointer', fontSize: 13, transition: 'all 0.2s' }}>+ 新建</button>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '0 12px' }}>
              {Object.values(sessions).map(s => (
                <div
                  key={s.id}
                  onClick={() => switchSession(s.id)}
                  onMouseEnter={() => setHoveredSessionId(s.id)}
                  onMouseLeave={() => setHoveredSessionId(null)}
                  style={{
                    padding: '10px 14px',
                    borderRadius: 12,
                    cursor: 'pointer',
                    fontSize: 14,
                    color: s.id === currentId ? '#fff' : '#94a3b8',
                    background: s.id === currentId ? `linear-gradient(135deg, ${mikuCyan}40, ${mikuLightCyan}40)` : 'transparent',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 2,
                    transition: 'all 0.2s ease',
                    transform: s.id === currentId ? 'translateX(4px)' : 'none',
                  }}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{s.title || '新对话'}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); deleteSession(s.id) }}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#ef4444',
                      cursor: 'pointer',
                      fontSize: 16,
                      padding: '0 4px',
                      opacity: hoveredSessionId === s.id ? 1 : 0,
                      transition: 'opacity 0.2s',
                    }}
                  >✕</button>
                </div>
              ))}
            </div>

            <div style={{ padding: '12px 16px', borderTop: '1px solid rgba(57,197,187,0.2)' }} ref={modeRef}>
              <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>问答模式</div>
              <div style={{ position: 'relative' }}>
                <button onClick={() => setShowModeMenu(!showModeMenu)} style={{ width: '100%', padding: '8px 12px', borderRadius: 10, background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(57,197,187,0.3)', color: '#e2e8f0', cursor: 'pointer', textAlign: 'left', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13 }}>
                  <span>
                    <img src={modes.find(m => m.value === mode).img} alt="" style={iconSmall} />
                    {modes.find(m => m.value === mode).label}
                  </span>
                  <span style={{ fontSize: 10 }}>▼</span>
                </button>
                {showModeMenu && (
                  <div style={{ position: 'absolute', bottom: '100%', left: 0, right: 0, marginBottom: 4, background: 'rgba(15,23,42,0.95)', borderRadius: 10, backdropFilter: 'blur(15px)', border: '1px solid rgba(57,197,187,0.3)', padding: 4, zIndex: 20, animation: 'fadeIn 0.2s ease' }}>
                    {modes.map(m => (
                      <div key={m.value} onClick={() => { setMode(m.value); setShowModeMenu(false) }} style={{ padding: '8px 10px', display: 'flex', alignItems: 'center', cursor: 'pointer', borderRadius: 8, fontSize: 13, color: mode === m.value ? '#fff' : '#94a3b8', background: mode === m.value ? 'rgba(57,197,187,0.2)' : 'transparent', transition: 'background 0.15s' }}>
                        <img src={m.img} alt="" style={iconSmall} />
                        {m.label}
                        {mode === m.value && <span style={{ marginLeft: 'auto', color: mikuCyan }}>✓</span>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div style={{ padding: '12px 16px', borderTop: '1px solid rgba(57,197,187,0.2)' }}>
              <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>知识库文档</div>
              <input value={filePath} onChange={e => setFilePath(e.target.value)} placeholder="文件路径，如 D:/doc.pdf" style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid rgba(57,197,187,0.3)', background: 'rgba(255,255,255,0.05)', color: '#e2e8f0', fontSize: 13, outline: 'none', marginBottom: 8, boxSizing: 'border-box' }} />
              <button onClick={uploadDoc} style={{ width: '100%', padding: '8px', background: `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})`, border: 'none', borderRadius: 8, color: '#0f172a', fontWeight: 600, fontSize: 13, cursor: 'pointer', transition: 'opacity 0.2s' }}>上传文档</button>
              {uploadStatus && <div style={{ fontSize: 12, marginTop: 6, color: uploadStatus.includes('成功') ? '#4ade80' : '#f87171' }}>{uploadStatus}</div>}
            </div>

            <div style={{ padding: '12px 16px', borderTop: '1px solid rgba(57,197,187,0.2)' }}>
              <button onClick={fetchEvalReport} style={{ width: '100%', padding: '8px', background: 'rgba(57,197,187,0.1)', border: '1px solid rgba(57,197,187,0.4)', borderRadius: 8, color: '#e2e8f0', fontSize: 13, cursor: 'pointer', transition: 'background 0.2s', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                <span>📊</span> 评估报告
              </button>
            </div>
          </>
        )}
      </div>

      {/* 侧边栏折叠按钮 */}
      <button onClick={() => setSidebarCollapsed(!sidebarCollapsed)} style={{ position: 'absolute', left: sidebarCollapsed ? 10 : 270, top: 20, zIndex: 30, background: 'rgba(15,23,42,0.8)', backdropFilter: 'blur(10px)', border: '1px solid rgba(57,197,187,0.3)', borderRadius: '50%', width: 32, height: 32, color: '#e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: 'left 0.3s', fontSize: 16 }}>
        {sidebarCollapsed ? '☰' : '◀'}
      </button>

      {/* ===== 主聊天区 ===== */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(10px)', position: 'relative' }}>
        <div style={{ padding: '14px 24px', borderBottom: '1px solid rgba(57,197,187,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(15,23,42,0.9)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 16, fontWeight: 600, color: '#f1f5f9' }}>{current.title}</span>
            <span style={{ fontSize: 12, background: 'rgba(57,197,187,0.15)', color: mikuCyan, padding: '2px 10px', borderRadius: 20, border: '1px solid rgba(57,197,187,0.3)' }}>{displayModelName()}</span>
          </div>
          <span style={{ fontSize: 12, color: '#94a3b8' }}>{current.messages.length} 条消息</span>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 24px 0', scrollBehavior: 'smooth' }}>
          {current.messages.length === 0 && (
            <div style={{ textAlign: 'center', marginTop: 80, color: '#94a3b8' }}>
              <img src="/miku-logo.jpg" alt="Welcome" style={{ width: 64, height: 64, borderRadius: '50%', marginBottom: 12, boxShadow: '0 0 20px rgba(57,197,187,0.3)' }} />
              <div style={{ fontSize: 20, fontWeight: 500, marginBottom: 8, color: '#e2e8f0' }}>开始对话</div>
              <div style={{ fontSize: 14 }}>你的初音未来智能助手已上线</div>
            </div>
          )}

          {current.messages.map((msg, i) => (
            <div key={i} style={{
              display: 'flex',
              gap: 12,
              marginBottom: 24,
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              animation: 'messageIn 0.4s ease',
            }}>
              <div style={{
                width: 36,
                height: 36,
                borderRadius: '50%',
                background: msg.role === 'user' ? `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})` : 'rgba(57,197,187,0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                border: msg.role === 'user' ? 'none' : `1px solid ${mikuCyan}`,
                overflow: 'hidden',
                boxShadow: msg.role === 'user' ? '0 0 12px rgba(57,197,187,0.4)' : 'none',
              }}>
                {msg.role === 'user' ? (
                  <img src="/miku-user.jpg" alt="User" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { e.target.style.display = 'none'; e.target.parentNode.innerText = '我'; }} />
                ) : (
                  <img src="/miku-ai.jpg" alt="AI" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                )}
              </div>
              <div style={{
                maxWidth: '75%',
                padding: '12px 18px',
                fontSize: 15,
                lineHeight: 1.7,
                wordBreak: 'break-word',
                ...(msg.role === 'user' ? {
                  background: `linear-gradient(135deg, ${mikuCyan}dd, ${mikuLightCyan}dd)`,
                  color: '#0f172a',
                  borderRadius: '18px 18px 4px 18px',
                  boxShadow: '0 4px 14px rgba(57,197,187,0.3)',
                } : {
                  background: 'rgba(255,255,255,0.08)',
                  color: '#e2e8f0',
                  borderRadius: '18px 18px 18px 4px',
                  border: '1px solid rgba(57,197,187,0.2)',
                  boxShadow: '0 4px 14px rgba(0,0,0,0.2)',
                })
              }}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '')
                      const codeString = String(children).replace(/\n$/, '')
                      return !inline && match ? (
                        <SyntaxHighlighter style={oneDark} language={match[1]} PreTag="div" {...props} customStyle={{ borderRadius: 12, padding: 16, fontSize: 14, background: 'rgba(0,0,0,0.6)' }}>
                          {codeString}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props} style={{ background: 'rgba(57,197,187,0.15)', padding: '2px 6px', borderRadius: 6, fontSize: '0.9em' }}>{children}</code>
                      )
                    }
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
                {i === current.messages.length - 1 && msg.role === 'assistant' && loading && (
                  <span className="typing-cursor" />
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div style={{ background: 'rgba(15,23,42,0.9)', borderTop: '1px solid rgba(57,197,187,0.2)', padding: '16px 24px 24px', display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="给 Miku 发送消息..."
            rows={2}
            style={{
              flex: 1,
              padding: '12px 16px',
              border: '1px solid rgba(57,197,187,0.4)',
              borderRadius: 16,
              fontSize: 15,
              fontFamily: 'inherit',
              resize: 'none',
              outline: 'none',
              backgroundColor: 'rgba(255,255,255,0.05)',
              color: '#e2e8f0',
              lineHeight: 1.5,
              maxHeight: 150,
            }}
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            style={{
              padding: '12px 24px',
              background: loading ? '#334155' : `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})`,
              color: loading ? '#94a3b8' : '#0f172a',
              border: 'none',
              borderRadius: 16,
              fontSize: 15,
              fontWeight: 600,
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: loading ? 'none' : '0 4px 14px rgba(57,197,187,0.3)',
            }}
          >
            {loading ? '...' : '发送'}
          </button>
        </div>
      </div>

      {/* 评估弹窗 */}
      {showEval && evalData && (
        <div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000, backdropFilter: 'blur(5px)' }}>
          <div style={{ background: '#1e293b', borderRadius: 20, padding: 30, maxWidth: 480, width: '80%', border: `1px solid ${mikuCyan}`, boxShadow: '0 20px 50px rgba(0,0,0,0.5)' }}>
            <h2 style={{ marginTop: 0, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
              <span>📊</span> RAG 系统质量报告
            </h2>
            <p style={{ color: '#94a3b8', fontSize: 13 }}>评估时间：{evalData.timestamp}</p>
            <table style={{ width: '100%', borderCollapse: 'collapse', color: '#cbd5e1' }}>
              <tbody>
                {[
                  { label: '答案忠实度', key: 'avg_faithfulness' },
                  { label: '上下文召回率', key: 'avg_context_recall' },
                  { label: '上下文精确度', key: 'avg_context_precision' }
                ].map(item => (
                  <tr key={item.key}>
                    <td style={{ padding: '10px 0', borderBottom: '1px solid rgba(57,197,187,0.2)' }}>{item.label}</td>
                    <td style={{ padding: '10px 0', borderBottom: '1px solid rgba(57,197,187,0.2)', fontWeight: 'bold', color: mikuCyan }}>
                      {typeof evalData[item.key] === 'number' ? (evalData[item.key] * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button onClick={() => setShowEval(false)} style={{ marginTop: 20, padding: '8px 24px', background: `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})`, color: '#0f172a', border: 'none', borderRadius: 12, cursor: 'pointer', fontWeight: 600 }}>关闭</button>
          </div>
        </div>
      )}

      {/* 全局动画样式 */}
      <style>{`
        @keyframes messageIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        .typing-cursor {
          display: inline-block;
          width: 2px;
          height: 18px;
          background: ${mikuCyan};
          vertical-align: text-bottom;
          margin-left: 2px;
          animation: blink 0.8s infinite;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(57,197,187,0.3); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(57,197,187,0.5); }
      `}</style>
    </div>
  )
}

export default App