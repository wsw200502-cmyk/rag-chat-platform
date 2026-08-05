import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

const STORAGE_KEY = 'chat_sessions'

// ==================== 自定义 Hook：会话管理 ====================
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
      if (role === 'user' && s[currentId].messages.length === 0) {
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
  const [mode, setMode] = useState('review')
  const [showModeMenu, setShowModeMenu] = useState(false)
  const [hoveredSessionId, setHoveredSessionId] = useState(null)

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

  const getModelName = () => {
    return mode === '14b' ? 'qwen2.5:14b' : 'qwen2.5:7b'
  }

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    addMessage('user', text)
    setInput('')
    setLoading(true)

    const history = current.messages.slice(-20).map(m => ({ role: m.role, content: m.content }))

    // Self-RAG 模式
    if (mode === 'selfrag') {
      try {
        const response = await fetch('http://localhost:8000/chat/selfrag', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, history: history, model: 'qwen2.5:7b' })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) { updateLastAssistant('? Self-RAG 请求失败：' + err.message) } finally { setLoading(false) }
      return
    }

    // HyDE 模式
    if (mode === 'hyde') {
      try {
        const response = await fetch('http://localhost:8000/chat/hyde', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, history: history, model: 'qwen2.5:14b' })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) { updateLastAssistant('? HyDE 请求失败：' + err.message) } finally { setLoading(false) }
      return
    }

    // Agent 模式
    if (mode === 'agent') {
      try {
        const response = await fetch('http://localhost:8000/agent', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, history: history })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) { updateLastAssistant('? Agent 请求失败：' + err.message) } finally { setLoading(false) }
      return
    }

    // 审查模式
    if (mode === 'review') {
      try {
        const response = await fetch('http://localhost:8000/chat/review', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, sessionId: currentId, history: history, models: ['qwen2.5:14b', 'qwen2.5:14b', 'qwen2.5:14b'] })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        updateLastAssistant(data.response || '（未收到回答）')
      } catch (err) { updateLastAssistant('? 请求失败：' + err.message) } finally { setLoading(false) }
      return
    }

    // 流式模式（7B/14B）
    const model = getModelName()
    try {
      const response = await fetch(`http://localhost:8000/chat/stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, sessionId: currentId, history: history, model: model })
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
      if (!full) full = '（助手没有返回内容）'
    } catch (err) { updateLastAssistant('? 请求失败：' + err.message) } finally { setLoading(false) }
  }

  const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }

  const uploadDoc = async () => {
    if (!filePath.trim()) return
    setUploadStatus('上传中...')
    try {
      const res = await fetch('http://localhost:8000/add_docs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([filePath.trim()]) })
      const data = await res.json()
      setUploadStatus(data.status === 'success' ? '? 上传成功' : '? 失败：' + data.message)
    } catch (e) { setUploadStatus('? 上传失败：' + e.message) }
  }

  const fetchEvalReport = async (e) => {
    if (e) e.stopPropagation()
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

  const modes = [
    { value: '7b', label: '轻量对话 (7B)', emoji: '??' },
    { value: '14b', label: '深度推理 (14B)', emoji: '??' },
    { value: 'review', label: '三模型审查', emoji: '??' },
    { value: 'agent', label: 'Agent 模式', emoji: '??' },
    { value: 'hyde', label: 'HyDE 检索', emoji: '??' },
    { value: 'selfrag', label: 'Self-RAG', emoji: '??' }
  ]

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: "'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif", background: '#86cecb' }}>
      {/* 侧边栏 */}
      <div style={{ width: 260, background: 'rgba(255,255,255,0.75)', backdropFilter: 'blur(10px)', color: '#1a3c34', display: 'flex', flexDirection: 'column', flexShrink: 0, height: '100vh', borderRight: '1px solid rgba(57,197,187,0.3)', boxShadow: '2px 0 10px rgba(57,197,187,0.1)' }}>
        <div style={{ padding: '20px', textAlign: 'center', borderBottom: '1px solid rgba(57,197,187,0.2)' }}>
          <div style={{ fontSize: 48 }}>??</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: mikuCyan, marginTop: 8 }}>Miku AI</div>
          <div style={{ fontSize: 11, color: '#86cecb' }}>初音未来 智能助手</div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 20px' }}>
          <span style={{ fontSize: 16, fontWeight: 600 }}>对话</span>
          <button onClick={createSession} style={{ background: 'rgba(57,197,187,0.15)', border: '1px solid #39c5bb', color: '#1a3c34', padding: '4px 10px', borderRadius: 20, cursor: 'pointer', fontSize: 14 }}>+ 新建</button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '8px 12px' }}>
          {Object.values(sessions).map(s => (
            <div
              key={s.id}
              onClick={() => switchSession(s.id)}
              onMouseEnter={() => setHoveredSessionId(s.id)}
              onMouseLeave={() => setHoveredSessionId(null)}
              style={{
                padding: '10px 14px', borderRadius: 20, cursor: 'pointer', fontSize: 14,
                color: s.id === currentId ? '#fff' : '#1a3c34',
                background: s.id === currentId ? `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})` : 'transparent',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2, transition: 'all 0.15s'
              }}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{s.title || '新对话'}</span>
              <button onClick={(e) => { e.stopPropagation(); deleteSession(s.id) }} style={{ background: 'none', border: 'none', color: s.id === currentId ? '#fff' : '#39c5bb', cursor: 'pointer', fontSize: 16, padding: '0 4px', display: hoveredSessionId === s.id ? 'inline-block' : 'none' }}>?</button>
            </div>
          ))}
        </div>

        <div style={{ padding: '12px 20px', borderTop: '1px solid rgba(57,197,187,0.2)' }} ref={modeRef}>
          <div style={{ color: '#1a3c34', fontSize: 13, marginBottom: 6 }}>? 问答模式</div>
          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowModeMenu(!showModeMenu)} style={{ width: '100%', padding: 8, borderRadius: 20, background: 'rgba(255,255,255,0.8)', color: '#1a3c34', border: `1px solid ${mikuCyan}`, outline: 'none', textAlign: 'left', display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
              <span style={{ display: 'flex', alignItems: 'center' }}>
                <span style={{ marginRight: 6 }}>{modes.find(m => m.value === mode).emoji}</span>
                {modes.find(m => m.value === mode).label}
              </span>
              <span>▼</span>
            </button>
            {showModeMenu && (
              <ul style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: '#fff', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.15)', listStyle: 'none', padding: 4, marginTop: 4, zIndex: 10 }}>
                {modes.map(m => (
                  <li key={m.value} onClick={() => { setMode(m.value); setShowModeMenu(false) }} style={{ padding: '6px 12px', display: 'flex', alignItems: 'center', cursor: 'pointer', borderRadius: 8, background: mode === m.value ? 'rgba(57,197,187,0.1)' : 'transparent', color: '#1a3c34', fontSize: 14 }}>
                    <span style={{ marginRight: 6 }}>{m.emoji}</span>
                    {m.label}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <div style={{ padding: '12px 20px', borderTop: '1px solid rgba(57,197,187,0.2)' }}>
          <div style={{ color: '#1a3c34', fontSize: 13, marginBottom: 6 }}>?? 添加文档到知识库</div>
          <input type="text" value={filePath} onChange={e => setFilePath(e.target.value)} placeholder="D:/docs/手册.pdf" style={{ width: '100%', padding: 8, borderRadius: 20, border: `1px solid ${mikuCyan}`, background: 'rgba(255,255,255,0.8)', color: '#1a3c34', marginBottom: 8, boxSizing: 'border-box', outline: 'none' }} />
          <button onClick={uploadDoc} style={{ width: '100%', padding: 8, background: `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})`, border: 'none', borderRadius: 20, color: '#fff', cursor: 'pointer', fontSize: 14 }}>上传</button>
          {uploadStatus && <div style={{ fontSize: 12, color: '#1a3c34', marginTop: 6 }}>{uploadStatus}</div>}
        </div>

        <div style={{ padding: '12px 20px', borderTop: '1px solid rgba(57,197,187,0.2)' }}>
          <button onClick={(e) => { e.stopPropagation(); fetchEvalReport(e) }} style={{ width: '100%', padding: 8, background: `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})`, border: 'none', borderRadius: 20, color: '#fff', cursor: 'pointer', fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            ?? 查看评估报告
          </button>
        </div>

        <div style={{ padding: '12px 20px', borderTop: '1px solid rgba(57,197,187,0.2)', fontSize: 12, color: '#1a3c34' }}>共 {Object.keys(sessions).length} 个会话</div>
      </div>

      {/* 主聊天区 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', background: 'rgba(255,255,255,0.6)', backdropFilter: 'blur(5px)' }}>
        <div style={{ padding: '12px 24px', borderBottom: `1px solid rgba(57,197,187,0.3)`, background: 'rgba(255,255,255,0.8)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 16, fontWeight: 600, color: '#1a3c34' }}>{current.title}</span>
            <span style={{ fontSize: 12, background: 'rgba(57,197,187,0.15)', color: '#1a3c34', padding: '2px 10px', borderRadius: 20 }}>Qwen2.5 · RAG</span>
          </div>
          <span style={{ fontSize: 13, color: '#1a3c34' }}>{current.messages.length} 条消息</span>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 24px 0' }}>
          {current.messages.length === 0 && (
            <div style={{ textAlign: 'center', color: '#1a3c34', marginTop: 80 }}>
              <div style={{ fontSize: 64 }}>??</div>
              <div style={{ fontSize: 18, fontWeight: 500, marginBottom: 8, color: '#1a3c34' }}>开始对话</div>
              <div style={{ fontSize: 14, color: '#86cecb' }}>你的初音未来智能助手已上线</div>
            </div>
          )}
          {current.messages.map((msg, i) => (
            <div key={i} style={{ display: 'flex', gap: 12, marginBottom: 20, justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
              <div style={{ width: 34, height: 34, borderRadius: '50%', background: msg.role === 'user' ? `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})` : 'transparent', color: msg.role === 'user' ? '#fff' : mikuCyan, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18, flexShrink: 0, border: msg.role === 'user' ? 'none' : `1px solid ${mikuCyan}`, overflow: 'hidden' }}>
                {msg.role === 'user' ? '??' : '??'}
              </div>
              <div style={{ maxWidth: '75%', padding: '10px 16px', fontSize: 15, lineHeight: 1.6, wordBreak: 'break-word', ...(msg.role === 'user' ? { background: `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})`, color: '#fff', borderRadius: '20px 20px 4px 20px', boxShadow: '0 2px 8px rgba(57,197,187,0.3)' } : { background: 'rgba(255,255,255,0.9)', color: '#1a3c34', borderRadius: '20px 20px 20px 4px', border: `1px solid ${mikuCyan}`, boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }) }}>
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ code({ node, inline, className, children, ...props }) { const match = /language-(\w+)/.exec(className || ''); const codeString = String(children).replace(/\n$/, ''); return !inline && match ? (<SyntaxHighlighter style={oneDark} language={match[1]} PreTag="div" wrapLines {...props}>{codeString}</SyntaxHighlighter>) : (<code className={className} {...props}>{children}</code>) } }}>{msg.content}</ReactMarkdown>
                {i === current.messages.length - 1 && msg.role === 'assistant' && loading && (<span style={{ display: 'inline-block', width: 2, height: 18, background: mikuCyan, verticalAlign: 'text-bottom', marginLeft: 2 }} />)}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div style={{ background: 'rgba(255,255,255,0.8)', borderTop: `1px solid rgba(57,197,187,0.3)`, padding: '12px 24px 20px', display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="输入消息..." rows={2} style={{ flex: 1, padding: '12px 16px', border: `1px solid ${mikuCyan}`, borderRadius: 20, fontSize: 15, fontFamily: 'inherit', resize: 'none', outline: 'none', backgroundColor: 'rgba(255,255,255,0.9)', maxHeight: 150, lineHeight: 1.5 }} />
          <button onClick={sendMessage} disabled={loading} style={{ padding: '10px 24px', background: loading ? '#ccc' : `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})`, color: '#fff', border: 'none', borderRadius: 20, fontSize: 15, fontWeight: 500, cursor: loading ? 'not-allowed' : 'pointer', height: 44, boxShadow: `0 2px 8px rgba(57,197,187,0.4)` }}>{loading ? '思考中...' : '发送'}</button>
        </div>
      </div>

      {/* 评估报告弹窗 */}
      {showEval && evalData && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
          <div style={{ background: '#fff', borderRadius: 20, padding: 30, maxWidth: 500, width: '80%', boxShadow: '0 10px 40px rgba(0,0,0,0.3)', border: `1px solid ${mikuCyan}` }}>
            <h2 style={{ marginTop: 0, color: mikuCyan, display: 'flex', alignItems: 'center' }}>
              ?? RAG 系统质量报告
            </h2>
            <p style={{ color: '#1a3c34' }}>评估时间：{evalData.timestamp}</p>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                <tr>
                  <td style={{ padding: 8, borderBottom: '1px solid #eee', color: '#1a3c34' }}>?? 答案忠实度</td>
                  <td style={{ padding: 8, borderBottom: '1px solid #eee', fontWeight: 'bold', color: mikuCyan }}>
                    {typeof evalData.avg_faithfulness === 'number' ? (evalData.avg_faithfulness * 100).toFixed(1) + '%' : 'N/A'}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: 8, borderBottom: '1px solid #eee', color: '#1a3c34' }}>?? 上下文召回率</td>
                  <td style={{ padding: 8, borderBottom: '1px solid #eee', fontWeight: 'bold', color: mikuCyan }}>
                    {typeof evalData.avg_context_recall === 'number' ? (evalData.avg_context_recall * 100).toFixed(1) + '%' : 'N/A'}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: 8, borderBottom: '1px solid #eee', color: '#1a3c34' }}>?? 上下文精确度</td>
                  <td style={{ padding: 8, borderBottom: '1px solid #eee', fontWeight: 'bold', color: mikuCyan }}>
                    {typeof evalData.avg_context_precision === 'number' ? (evalData.avg_context_precision * 100).toFixed(1) + '%' : 'N/A'}
                  </td>
                </tr>
              </tbody>
            </table>
            <button onClick={() => setShowEval(false)} style={{ marginTop: 20, padding: '8px 24px', background: `linear-gradient(135deg, ${mikuCyan}, ${mikuLightCyan})`, color: '#fff', border: 'none', borderRadius: 20, cursor: 'pointer' }}>关闭</button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App