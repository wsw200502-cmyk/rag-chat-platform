# 本地化多模型协作 RAG 智能对话平台

基于 **FastAPI + Ollama + LangChain + React** 构建的本地 RAG 智能对话系统，支持 **14B×3 三模型链式审查**、**ReAct Agent**、**HyDE 检索增强**、**Self-RAG（自我反思检索）**、**混合检索 + Reranker 重排序**、**SSE 流式对话**、**RAGAS 自动评估**，前端采用初音未来主题。

## 🖥️ 技术栈

- **AI 核心**：Python · FastAPI · LangChain · Chroma · Ollama (Qwen2.5 7B/14B)  
- **检索增强**：BM25 · Cross-Encoder Reranker (BGE-Reranker) · Tavily Search  
- **RAG 进阶**：HyDE（假设文档嵌入）· Self-RAG（7B 决策 + 14B 生成）  
- **Agent 框架**：ReAct 循环 · 工具注册 · 子任务规划 · 反思模块  
- **前端**：React 18 · Vite · ReactMarkdown · SyntaxHighlighter  
- **评估**：RAGAS (忠实度、上下文召回率、精确度)  
- **模型调度**：Ollama 本地部署 · GPU 加速 · 动态 keep_alive 管理  

## ⚡ 快速启动

### 1. 启动 Ollama
ash
ollama serve


### 2. 安装 Python 依赖
ash
pip install -r requirements.txt


### 3. 启动 AI 后端
ash
python agent_api.py

服务运行在 http://localhost:8000，终端输出 🚀 智能助手已启动... 表示成功。

### 4. 启动前端
ash
cd frontend
npm install
npm run dev

前端运行在 http://localhost:5173，打开浏览器即可使用。

## 📋 环境变量

在运行后端前，可设置以下环境变量（非必须）：
ash
export TAVILY_API_KEY="你的Tavily Key"   # 用于网络搜索，不设置则跳过搜索
export OLLAMA_KEEP_ALIVE="30m"           # 模型常驻时间，默认与代码保持一致
export OLLAMA_MAX_LOADED_MODELS=1        # 单模型模式，避免显存溢出

如果不想使用搜索功能，可完全忽略 TAVILY_API_KEY，系统会自动降级。

## ✨ 核心功能

| 模式 | 说明 | 调用接口 |
|------|------|----------|
| 轻量对话 (7B) | 7B 模型流式对话，适合日常问答 | /chat/stream |
| 深度推理 (14B) | 14B 模型流式对话，复杂推理 | /chat/stream |
| 三模型审查 | 14B×3 解耦→生成→审查，高可靠性 | /chat/review |
| Agent 模式 | ReAct 循环，自主调用搜索/计算/知识库 | /agent |
| HyDE 检索 | 生成假设答案后检索，提升召回 | /chat/hyde |
| Self-RAG | 模型自主判断检索时机，7B 决策 + 14B 生成 | /chat/selfrag |

## 🧩 项目亮点

- **多模型协作与 Agent**：从简单对话到三模型审查，再到 ReAct Agent 和 Self-RAG，展示了完整的 Agent 演进路线。
- **混合检索 + Reranker**：向量检索 (Chroma) + BM25 关键词检索 + Cross-Encoder 重排序，显著提升 RAG 回答质量。
- **RAG 深度优化**：集成 HyDE 和 Self-RAG，前者用假设答案提升检索命中率，后者让模型自主决策检索时机，避免冗余检索。
- **显存优化与常驻策略**：通过 keep_alive 和 OLLAMA_MAX_LOADED_MODELS 精细控制显存占用，8GB 显卡即可流畅运行 14B 模型。
- **生产级可靠性**：502 自动重试、嵌入模型预热、工具调用防重复、最终答案兜底策略等，保障系统稳定。
- **完整评估体系**：RAGAS 忠实度、召回率、精确度自动评估，支持前端查看报告。

## 📊 系统接口概览

| 接口 | 说明 |
|------|------|
| /chat/stream | 流式对话（支持 7B/14B） |
| /chat/review | 三模型审查（默认全 14B） |
| /agent | ReAct Agent 智能体 |
| /chat/hyde | HyDE 检索增强生成 |
| /chat/selfrag | Self-RAG（7B 决策 + 14B 生成） |
| /add_docs | 上传文档到知识库 |
| /retrieve | 测试混合检索 |
| /eval/report | 获取 RAGAS 评估报告 |

## 📂 项目结构

ag-chat-platform/
├── agent_api.py            # 后端主入口
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.jsx         # 前端主组件
│       └── ...
├── chroma_db/              # 向量数据库
├── requirements.txt
└── README.md


## 🤝 贡献与许可

欢迎提交 Issue 和 PR！本项目基于 MIT 许可证开源。
