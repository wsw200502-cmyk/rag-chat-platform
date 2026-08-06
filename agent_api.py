import asyncio
import base64
import inspect
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List

import requests
import uvicorn
from config import get_settings
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

settings = get_settings()

# 路径自动创建
Path(settings.sessions_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path("./uploaded_docs").mkdir(parents=True, exist_ok=True)

# ========== ① 禁用系统代理 ==========
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

# 兼容 .env 手动解析
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)
if not os.getenv("TAVILY_API_KEY"):
    try:
        with open(env_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    except Exception:
        pass

print(f"[DEBUG] .env 路径: {env_path}")
print(f"[DEBUG] TAVILY_API_KEY 是否加载: {os.getenv('TAVILY_API_KEY') is not None}")

TAVILY_API_KEY = settings.tavily_api_key or os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    from langchain_tavily import TavilySearch

# ==================== 结构化日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_api")

if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY 未设置，网络搜索功能将不可用。")


# ==================== 模型名称规范化 ====================
def normalize_model_name(name: str) -> str:
    mapping = {
        "qwen2.5:14b": "qwen3.5:9b",
        "qwen3:14b": "qwen3.5:9b",
    }
    return mapping.get(name, name)

# ==================== 模型缓存 & 显存管理 ====================
llm_cache = {}
_active_model_key = None

def get_llm(model_name: str, keep_alive: int = -1):
    """获取LLM，支持显存管理：8G显存一次只常驻一个模型"""
    global _active_model_key
    model_name = normalize_model_name(model_name)
    cache_key = (model_name, keep_alive)
    
    if keep_alive == -1 and _active_model_key != cache_key and _active_model_key is not None:
        logger.info(f"显存管理: 释放之前常驻模型 {_active_model_key[0]}")
        try:
            requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={"model": _active_model_key[0], "keep_alive": 0},
                proxies={"http": None, "https": None},
                timeout=10,
            )
        except Exception as e:
            logger.debug(f"释放模型显存时出错: {e}")
        if _active_model_key in llm_cache:
            del llm_cache[_active_model_key]
    
    if cache_key not in llm_cache:
        logger.info(f"初始化模型: {model_name} (keep_alive={keep_alive})")
        llm_cache[cache_key] = __import__(
            "langchain_ollama", fromlist=["ChatOllama"]
        ).ChatOllama(
            model=model_name,
            temperature=0,
            base_url=settings.ollama_base_url,
            num_predict=settings.ollama_num_predict,
            num_ctx=settings.ollama_context_length,
            keep_alive=keep_alive,
        )
    
    if keep_alive == -1:
        _active_model_key = cache_key
    return llm_cache[cache_key]

# ==================== 自定义 Embedding ====================
class OllamaEmbeddingDirect:
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or settings.ollama_embed_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def _call(self, text: str) -> List[float]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            proxies={"http": None, "https": None},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embeddings") or data.get("embedding")
        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
            return emb[0]
        return emb

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._call(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._call(text)

embedding = OllamaEmbeddingDirect()

# ==================== 向量数据库 ====================
vectorstore = Chroma(
    persist_directory=settings.chroma_persist_dir,
    embedding_function=embedding,
)

# ==================== 网络搜索 ====================
if TAVILY_API_KEY:
    search = TavilySearch(api_key=TAVILY_API_KEY, max_results=3)
else:
    search = None

# ==================== 混合检索 ====================
global_documents_text: list[str] = []
bm25_index = None

def rebuild_bm25():
    global bm25_index
    if global_documents_text:
        tokenized_corpus = [doc.split() for doc in global_documents_text]
        bm25_index = BM25Okapi(tokenized_corpus)
    else:
        bm25_index = None

# ==================== Reranker ====================
reranker = None

def get_reranker():
    global reranker
    if reranker is None:
        reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            local_files_only=True,
            trust_remote_code=True,
        )
    return reranker

# ==================== 文档导入 ====================
def add_documents_to_store(file_paths: list[str]) -> int:
    global global_documents_text
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    global_documents_text.extend([chunk.page_content for chunk in chunks])
    rebuild_bm25()
    return len(chunks)

# ==================== 混合检索 + Rerank ====================
def retrieve_with_hybrid_and_rerank(
    query: str,
    k_vector: int = 10,
    k_bm25: int = 10,
    final_k: int = 3,
    max_retries: int = 5,
) -> list[Document]:
    vec_docs = []
    for attempt in range(max_retries):
        try:
            vec_docs = vectorstore.similarity_search(query, k=k_vector)
            break
        except Exception as e:
            if "502" in str(e) and attempt < max_retries - 1:
                logger.warning(f"向量检索失败，重试中: {e}")
                time.sleep(2**attempt + 3)
                continue
            else:
                logger.error(f"向量检索最终失败: {e}")
                vec_docs = []
                break

    bm25_docs = []
    if bm25_index is not None:
        tokenized_query = query.split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_bm25]
        for idx in top_indices:
            bm25_docs.append(Document(page_content=global_documents_text[idx]))

    all_docs = {doc.page_content: doc for doc in vec_docs + bm25_docs}.values()
    if not all_docs:
        return []

    model = get_reranker()
    doc_texts = [doc.page_content for doc in all_docs]
    pairs = [[query, text] for text in doc_texts]
    scores = model.predict(pairs)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:final_k]
    return [list(all_docs)[i] for i in sorted_indices]

# ==================== 构建消息列表 ====================
def build_messages(
    question: str,
    history: list[dict[str, str]],
    extra_context: str = "",
) -> list:
    local_docs = retrieve_with_hybrid_and_rerank(question)
    local_context = "\n".join([doc.page_content for doc in local_docs]) if local_docs else "无相关本地知识"
    web_context = "网络搜索不可用（未配置 API Key）"
    if search is not None:
        try:
            web_results = search.invoke(question)
            web_context = str(web_results)
        except Exception as e:
            logger.warning(f"网络搜索调用失败: {e}")

    context = f"[本地知识库]\n{local_context}\n\n[网络搜索结果]\n{web_context}"
    if extra_context:
        context += f"\n\n[额外检索结果]\n{extra_context}"

    messages = [
        SystemMessage(
            content=(
                "你是一个智能助手。请基于提供的资料和自身知识回答问题。\n"
                "规则：\n"
                "1. 如果是闲聊，自然回复，无需代码。\n"
                "2. 只在明确要求写代码时才输出代码块。\n"
                "3. 代码需完整可运行，置于 Markdown 代码块中。\n"
                "4. 如果用户要求“只输出代码”，仅输出代码块。"
            )
        )
    ]
    for h in history[-10:]:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=f"{context}\n\n用户问题：{question}"))
    return messages

# ==================== 后处理 ====================
def post_process(content: str, question: str) -> str:
    content = re.sub(r"^ython\b", r"```python", content, flags=re.MULTILINE)
    content = re.sub(r"^thon\b", r"```python", content, flags=re.MULTILINE)
    content = re.sub(r"^sh\b", r"```sh", content, flags=re.MULTILINE)
    content = re.sub(r"^bash\b", r"```bash", content, flags=re.MULTILINE)
    content = re.sub(r"^shell\b", r"```shell", content, flags=re.MULTILINE)
    content = re.sub(r"^markdown\b", r"```markdown", content, flags=re.MULTILINE)
    content = re.sub(r"^``([a-zA-Z]+)", r"```\1", content, flags=re.MULTILINE)
    content = re.sub(r"(import\s+\S+)([a-zA-Z])", r"\1\n\2", content)
    content = re.sub(r"(from\s+\S+\s+import\s+\S+)([a-zA-Z])", r"\1\n\2", content)
    content = re.sub(r"([a-zA-Z_]\s*=\s*)(def |class |import )", r"\1\n\2", content)

    if any(kw in content for kw in ["import ", "def ", "print(", "class "]) and "```" not in content:
        content = f"```python\n{content.strip()}\n```"

    if any(kw in question for kw in ["只输出代码", "只给我代码", "只要代码", "去掉所有文本", "我只要代码"]):
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        else:
            return re.sub(r"```.*?```", "", content, flags=re.DOTALL).strip()
    return content

# ==================== 非流式接口 ====================
def generate_answer(
    question: str,
    history: list[dict[str, str]],
    model_name: str = None,
) -> str:
    model_name = normalize_model_name(model_name or settings.ollama_default_model)
    messages = build_messages(question, history)
    keep_alive = 0 if "4b" in model_name.lower() else -1
    llm = get_llm(model_name, keep_alive=keep_alive)
    response = llm.invoke(messages)
    return post_process(response.content, question)

# ==================== 多模型协作 ====================
def generate_collaborative(question: str, history: list[dict[str, str]]) -> str:
    planner = get_llm(settings.ollama_deep_model, keep_alive=-1)
    planner_prompt = (
        "你是一个任务规划专家。将用户需求分解为清晰的步骤，并生成优化的提示词。"
        "只输出优化后的提示词，不要解释。\n"
        f"用户需求：{question}"
    )
    plan_messages = [HumanMessage(content=planner_prompt)]
    plan_response = planner.invoke(plan_messages)
    optimized_query = plan_response.content.strip()

    executor = get_llm(settings.ollama_deep_model, keep_alive=-1)
    messages = build_messages(optimized_query, history)
    messages[-1] = HumanMessage(
        content=(f"根据资料回答用户问题。\n用户问题：{question}\n优化提示词：{optimized_query}")
    )
    final_response = executor.invoke(messages)
    return final_response.content

# ==================== FastAPI ====================
app = FastAPI(title="AI Assistant (RAG + Agent + HyDE + Self-RAG + Vision)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model: str = None
    history: list[dict[str, str]] = []
    models: list[str] | None = None

class ChatResponse(BaseModel):
    response: str

# ==================== 原有路由 ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model = normalize_model_name(req.model or settings.ollama_default_model)
    answer = generate_answer(req.message, req.history, model)
    return ChatResponse(response=answer)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_stream():
        model = normalize_model_name(req.model or settings.ollama_default_model)
        keep_alive = 0 if "4b" in model.lower() else -1
        llm = get_llm(model, keep_alive=keep_alive)
        messages = build_messages(req.message, req.history)
        try:
            async for chunk in llm.astream(messages):
                if chunk.content:
                    data = json.dumps({"token": chunk.content})
                    yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式生成出错: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8")

@app.post("/chat/multi")
async def chat_multi(req: ChatRequest):
    answer = generate_collaborative(req.message, req.history)
    return ChatResponse(response=answer)

# ==================== 三角形多Agent协作架构（内切圆共享核心） ====================
class SharedCore:
    def __init__(self):
        self.memory: dict[str, Any] = {}
        self.retrieval_log: list[dict] = []
        self.tool_log: list[dict] = []

    async def retrieve(self, query: str, k_vector: int = 10, k_bm25: int = 10, final_k: int = 5) -> list[Document]:
        docs = retrieve_with_hybrid_and_rerank(query, k_vector, k_bm25, final_k)
        self.retrieval_log.append({"query": query, "count": len(docs)})
        return docs

    def web_search(self, query: str) -> str:
        if search is None:
            return "网络搜索不可用（未配置 API Key）"
        try:
            result = str(search.invoke(query))
            return result
        except Exception as e:
            return f"网络搜索失败: {e}"

    def use_tool(self, tool_name: str, tool_input: str) -> str:
        tool = tool_registry.get_tool(tool_name) if 'tool_registry' in globals() else None
        if not tool:
            return f"错误：未找到工具 '{tool_name}'"
        try:
            if inspect.iscoroutinefunction(tool):
                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(tool(tool_input), loop)
                    result = future.result(timeout=30)
                except RuntimeError:
                    result = asyncio.run(tool(tool_input))
            else:
                result = tool(tool_input)
            self.tool_log.append({"tool": tool_name, "input": tool_input, "output": str(result)[:200]})
            return result
        except Exception as e:
            return f"工具执行错误: {e}"

    def store(self, key: str, value: Any):
        self.memory[key] = value

    def get(self, key: str, default=None):
        return self.memory.get(key, default)

    def get_tool_descriptions(self) -> str:
        return tool_registry.get_all_descriptions() if 'tool_registry' in globals() else ""

class TriangleAgent:
    def __init__(self, name: str, model_name: str, role_desc: str, core: SharedCore):
        self.name = name
        self.model_name = normalize_model_name(model_name)
        self.role_desc = role_desc
        self.core = core

    async def think(self, instruction: str, history: list[dict] = None, max_retries: int = 3) -> str:
        llm = get_llm(self.model_name, keep_alive=-1)
        messages = [
            SystemMessage(content=(
                f"你是 {self.name}。\n"
                f"角色定位：{self.role_desc}\n"
                f"当前日期：{datetime.now().strftime('%Y年%m月%d日')}"
            ))
        ]
        if history:
            for h in history[-6:]:
                role = h.get("role")
                content = h.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=instruction))

        for attempt in range(max_retries):
            try:
                resp = await asyncio.to_thread(llm.invoke, messages)
                return resp.content.strip()
            except Exception as e:
                if "502" in str(e) and attempt < max_retries - 1:
                    wait_time = 2**attempt + 10
                    logger.warning(f"[{self.name}] LLM 502，等待 {wait_time}s 重试 {attempt+1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise e

    async def analyze(self, question: str, history: list[dict] = None) -> str:
        docs = await self.core.retrieve(question)
        self.core.store("phase1_docs", docs)
        local_ctx = "\n".join([d.page_content for d in docs]) if docs else "无相关本地知识"
        web_ctx = self.core.web_search(question)

        prompt = (
            f"请分析以下问题，指出回答该问题需要确认的关键事实。\n"
            f"问题：{question}\n"
            f"参考资料：{local_ctx}\n"
            f"网络搜索摘要：{web_ctx}\n\n"
            f"只需列出关键点，不要展开。"
        )
        return await self.think(prompt, history)

    async def generate(self, question: str, analysis: str, history: list[dict] = None) -> str:
        docs = self.core.get("phase1_docs") or await self.core.retrieve(question)
        local_ctx = "\n".join([d.page_content for d in docs]) if docs else "无相关本地知识"
        web_ctx = self.core.web_search(question)

        prompt = (
            f"基于分析结果和资料，直接回答问题。\n"
            f"问题：{question}\n"
            f"分析：{analysis}\n"
            f"本地资料：{local_ctx}\n"
            f"网络搜索：{web_ctx}\n\n"
            f"请给出准确、简洁的答案。"
        )
        return await self.think(prompt, history)

    async def verify(self, question: str, analysis: str, draft: str, history: list[dict] = None) -> str:
        verify_docs = await self.core.retrieve(f"验证：{question}", final_k=5)
        verify_ctx = "\n".join([d.page_content for d in verify_docs]) if verify_docs else "无相关验证资料"

        prompt = (
            f"请审查以下答案是否准确、完整。\n"
            f"问题：{question}\n"
            f"答案：{draft}\n"
            f"验证资料：{verify_ctx}\n\n"
            f"如果准确无误，回复“VERIFIED”；否则回复“REVISION: 具体问题”。"
        )
        return await self.think(prompt, history)

# ==================== ToolRegistry ====================
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, description: str):
        def decorator(func: Callable):
            self._tools[name] = func
            self._descriptions[name] = description
            return func
        return decorator

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)

    def get_all_descriptions(self) -> str:
        return "\n".join([f"- {name}: {desc}" for name, desc in self._descriptions.items()])

tool_registry = ToolRegistry()

# ==================== /chat/review 端点（含容错） ====================
@app.post("/chat/review")
async def chat_review(req: ChatRequest):
    for i in range(5):
        try:
            _ = embedding.embed_query("warmup")
            break
        except Exception as e:
            if i < 4:
                logger.warning(f"嵌入预热失败，重试 {i+1}/5: {e}")
                time.sleep(2)
            else:
                logger.error(f"嵌入预热最终失败: {e}")

    models = req.models or [settings.ollama_deep_model] * 3
    models = [normalize_model_name(m) for m in models]
    if len(models) < 3:
        return ChatResponse(response="三角形协作审查模式需要至少三个模型")

    core = SharedCore()
    agent_a = TriangleAgent("Agent-A·分析", models[0], "需求分析专家", core)
    agent_b = TriangleAgent("Agent-B·生成", models[1], "答案生成专家", core)
    agent_c = TriangleAgent("Agent-C·审查", models[2], "事实核查专家", core)

    # ----- 阶段1：分析 -----
    try:
        analysis = await agent_a.analyze(req.message, req.history)
        core.store("analysis", analysis)
        logger.info("[Phase 1] 分析完成")
    except Exception as e:
        logger.error(f"Agent-A 分析失败: {e}")
        analysis = None

    # ----- 阶段2：生成（含容错接管） -----
    draft = None
    if analysis:
        try:
            draft = await agent_b.generate(req.message, analysis, req.history)
            core.store("draft", draft)
            logger.info("[Phase 2] 生成完成")
        except Exception as e:
            logger.error(f"Agent-B 生成失败: {e}")
            draft = None

    if not draft:
        logger.warning("Agent-B 故障，启用 Agent-C 接管生成")
        try:
            takeover_prompt = (
                f"Agent-A 和 Agent-B 均未产出有效草案。请直接根据以下信息回答用户问题。\n"
                f"问题：{req.message}\n"
                f"分析：{analysis or '无'}\n"
                f"请给出简洁准确的答案。"
            )
            draft = await agent_c.think(takeover_prompt, req.history)
            core.store("draft", draft)
            logger.info("Agent-C 接管生成完成")
        except Exception as e:
            logger.error(f"Agent-C 接管也失败: {e}")
            return ChatResponse(response=f"三角形协作完全失败：{e!s}")

    # ----- 阶段3：审查 -----
    try:
        review = await agent_c.verify(req.message, analysis or "", draft, req.history)
        core.store("review", review)
        logger.info("[Phase 3] 审查完成")
    except Exception as e:
        logger.error(f"Agent-C 审查失败: {e}")
        review = "审查不可用"

    # ----- 综合仲裁 -----
    needs_revision = any(kw in review.upper() for kw in ["REVISION", "FAIL", "修改", "错误", "不通过"])
    if needs_revision:
        docs = core.get("phase1_docs") or await core.retrieve(req.message)
        local_ctx = "\n".join([d.page_content for d in docs]) if docs else "无相关本地知识"
        revision_prompt = (
            f"你的答案草案未通过事实核查。\n\n"
            f"审查意见：{review}\n\n"
            f"原始问题：{req.message}\n"
            f"Agent-A 分析：{analysis or '无'}\n\n"
            f"共享检索资料：\n{local_ctx}\n\n"
            f"请根据审查意见和共享资料修订答案，确保事实准确："
        )
        try:
            final_answer = await agent_b.think(revision_prompt, req.history) if analysis else await agent_c.think(revision_prompt, req.history)
        except:
            final_answer = draft  # 修订失败则返回原草案
        status_tag = (
            f"\n\n[三角形协作日志]\n"
            f"- Agent-A 分析: {'✓' if analysis else '✗ 失败'}\n"
            f"- Agent-B 生成: {'✓' if analysis and draft else '✗ 失败 (已接管)'}\n"
            f"- Agent-C 审查: ✗ 未通过\n"
            f"- 修订: ✓ 已执行\n"
            f"- 审查详情: {review[:200]}"
        )
    else:
        final_answer = draft
        status_tag = (
            f"\n\n[三角形协作日志]\n"
            f"- Agent-A 分析: {'✓' if analysis else '✗ 失败'}\n"
            f"- Agent-B 生成: {'✓' if analysis and draft else '✗ 失败 (已接管)'}\n"
            f"- Agent-C 审查: ✓ 通过\n"
            f"- 审查详情: {review[:200]}"
        )

    return ChatResponse(response=final_answer + status_tag)

# ==================== HyDE ====================
def generate_hypothetical_answer(question: str, model_name: str = None) -> str:
    model_name = normalize_model_name(model_name or settings.ollama_deep_model)
    prompt = f"""请根据你的知识，为以下问题生成一段假设性的回答（哪怕不知道确切答案，也请给出合理的推测性描述）。只需输出回答内容，不需要解释。

问题：{question}
假设答案："""
    llm = get_llm(model_name, keep_alive=-1)
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()

def hyde_retrieve_and_answer(question: str, history: list[dict[str, str]], model_name: str = None) -> str:
    model_name = normalize_model_name(model_name or settings.ollama_deep_model)
    hypothetical = generate_hypothetical_answer(question)
    local_docs = retrieve_with_hybrid_and_rerank(hypothetical)
    local_context = "\n".join([doc.page_content for doc in local_docs]) if local_docs else "无相关本地知识"

    messages = [
        SystemMessage(
            content=(
                "你是一个智能助手。请结合提供的资料和自身知识回答问题。\n"
                "规则：\n"
                "1. 如果是闲聊，自然回复，无需代码。\n"
                "2. 只在明确要求写代码时才输出代码块。\n"
                "3. 代码需完整可运行，置于 Markdown 代码块中。\n"
                "4. 如果用户要求“只输出代码”，仅输出代码块。"
            )
        )
    ]
    for h in history[-10:]:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=f"[本地知识库（基于假设答案检索）]\n{local_context}\n\n用户问题：{question}"))

    llm = get_llm(model_name, keep_alive=-1)  # 复用 9B 常驻
    resp = llm.invoke(messages)
    return resp.content.strip()

@app.post("/chat/hyde")
async def chat_hyde(req: ChatRequest):
    try:
        model = normalize_model_name(req.model or settings.ollama_deep_model)
        answer = await asyncio.to_thread(hyde_retrieve_and_answer, req.message, req.history, model)
        return ChatResponse(response=answer)
    except Exception as e:
        logger.error(f"HyDE 请求失败: {e!s}")
        return ChatResponse(response=f"HyDE 处理失败: {e!s}")

# ==================== Self-RAG ====================
def should_retrieve(question: str) -> bool:
    chat_keywords = [
        "你好",
        "嗨",
        "hello",
        "hi",
        "谢谢",
        "再见",
        "拜拜",
        "在吗",
        "早上好",
        "晚上好",
    ]
    lowered = question.lower().strip()
    return not any(kw in lowered for kw in chat_keywords)

SELF_RAG_SYSTEM_PROMPT = """你是一个能够自我反思的智能助手。在回答问题时，你可以决定是否需要检索外部知识。
请遵循以下规则：
1. 如果不需要检索，直接回答问题。
2. 如果需要检索，输出：<RETRIEVE> 查询关键词 </RETRIEVE>，然后停止。系统会为你提供检索结果。
3. 收到检索结果后，你可以再次决定是否需要进一步检索（输出另一个 <RETRIEVE>），或者直接给出最终答案。
4. 当你认为信息足够时，输出最终答案。
5. 如果你认为检索到的信息无助于回答，输出 <REJECT> 并用自己的知识回答。
"""

def self_rag_answer(
    question: str,
    history: list[dict[str, str]],
    decision_model: str = None,
    generation_model: str = None,
    max_iterations: int = 3,
) -> str:
    decision_model = normalize_model_name(decision_model or settings.ollama_deep_model)
    generation_model = normalize_model_name(generation_model or settings.ollama_deep_model)
    decision_llm = get_llm(decision_model, keep_alive=-1)
    generation_llm = get_llm(generation_model, keep_alive=-1)

    messages = [SystemMessage(content=SELF_RAG_SYSTEM_PROMPT)]
    for h in history[-10:]:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=f"用户问题：{question}"))

    for i in range(max_iterations):
        response = decision_llm.invoke(messages)
        content = response.content.strip()
        messages.append(AIMessage(content=content))

        retrieve_match = re.search(r"<RETRIEVE>\s*(.*?)\s*</RETRIEVE>", content, re.DOTALL)
        reject_match = re.search(r"<REJECT>", content)

        if retrieve_match:
            query = retrieve_match.group(1).strip()
            logger.info(f"Self-RAG 检索: {query}")
            docs = retrieve_with_hybrid_and_rerank(query)
            context = "\n".join([doc.page_content for doc in docs]) if docs else "无相关结果"
            if search is not None:
                try:
                    web_results = search.invoke(query)
                    context += "\n\n网络搜索结果：\n" + str(web_results)
                except Exception as e:
                    logger.warning(f"网络搜索失败: {e}")
            messages.append(HumanMessage(content=f"检索结果：\n{context}\n\n请根据检索结果继续回答，或继续检索。"))
        elif reject_match or not retrieve_match:
            break
        else:
            break

    final_prompt = "请根据以上所有信息和检索结果，给出最终答案。"
    messages.append(HumanMessage(content=final_prompt))
    final_response = generation_llm.invoke(messages)
    return final_response.content.strip()

@app.post("/chat/selfrag")
async def chat_selfrag(req: ChatRequest):
    try:
        answer = await asyncio.to_thread(
            self_rag_answer,
            req.message,
            req.history,
            decision_model=settings.ollama_deep_model,
            generation_model=settings.ollama_deep_model,
        )
        return ChatResponse(response=answer)
    except Exception as e:
        logger.error(f"Self-RAG 请求失败: {e!s}")
        return ChatResponse(response=f"Self-RAG 处理失败: {e!s}")

# ==================== 视觉识别 ====================
@app.post("/chat/vision")
async def chat_vision(
    message: str = Form(""),
    sessionId: str = Form(""),
    model: str = Form("qwen3.5"),
    history: str = Form("[]"),
    files: list[UploadFile] = File(default=[]),
):
    try:
        hist = json.loads(history)
    except Exception:
        hist = []

    images_b64 = []
    for file in files:
        content = await file.read()
        images_b64.append(base64.b64encode(content).decode("utf-8"))

    if not images_b64:
        return ChatResponse(response="没有收到图片")

    ollama_messages = []
    for h in hist[-10:]:
        role = h.get("role")
        content = h.get("content", "")
        if role == "system":
            ollama_messages.append({"role": "system", "content": content})
        elif role == "user":
            ollama_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            ollama_messages.append({"role": "assistant", "content": content})

    ollama_messages.append({
        "role": "user",
        "content": message or "请描述这张图片",
        "images": images_b64,
    })

    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": model,
                "messages": ollama_messages,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            proxies={"http": None, "https": None},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("message", {}).get("content", "") or "（模型无返回）"
    except Exception as e:
        logger.error(f"视觉模型调用失败: {e}")
        return ChatResponse(response=f"❌ 视觉模型请求失败: {e}")

    return ChatResponse(response=answer)

# ==================== ReAct Agent 模块（增强版） ====================
@tool_registry.register("search", "搜索互联网获取实时信息，输入查询字符串")
async def search_tool(query: str) -> str:
    if search is None:
        return "网络搜索不可用，请设置 TAVILY_API_KEY"
    try:
        results = search.invoke(query)
        return str(results)
    except Exception as e:
        return f"搜索失败: {e!s}"

@tool_registry.register("calculator", "执行数学计算，输入数学表达式字符串")
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "不允许的表达式"
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e!s}"

@tool_registry.register("retrieve_knowledge", "检索本地知识库，输入查询字符串")
async def retrieve_knowledge(query: str) -> str:
    docs = retrieve_with_hybrid_and_rerank(query)
    if not docs:
        return "未找到相关知识"
    return "\n".join([doc.page_content for doc in docs])

# ==================== 记忆系统 ====================
@dataclass
class WorkingMemory:
    current_goal: str = ""
    subtasks: list[dict[str, Any]] = field(default_factory=list)
    intermediate_results: dict[str, str] = field(default_factory=dict)
    last_action: str | None = None
    last_input: str | None = None
    failure_count: int = 0

@dataclass
class EpisodicMemory:
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: int, action: str, input_str: str, output: str, reflection: str = ""):
        self.events.append({
            "step": step,
            "action": action,
            "input": input_str,
            "output": output,
            "reflection": reflection,
            "timestamp": time.time(),
        })

@dataclass
class LongTermMemory:
    user_preferences: dict[str, Any] = field(default_factory=dict)
    tool_effectiveness: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_tool_result(self, tool_name: str, success: bool):
        if tool_name not in self.tool_effectiveness:
            self.tool_effectiveness[tool_name] = {"success": 0, "fail": 0}
        self.tool_effectiveness[tool_name]["success" if success else "fail"] += 1

    def get_tool_reliability(self, tool_name: str) -> float:
        stats = self.tool_effectiveness.get(tool_name, {"success": 0, "fail": 0})
        total = stats["success"] + stats["fail"]
        return stats["success"] / total if total > 0 else 0.5

# ==================== 反思模块 ====================
class ReflexionModule:
    def __init__(self, model_name: str = None):
        self.model_name = normalize_model_name(model_name or settings.ollama_deep_model)

    def reflect(self, goal: str, action: str, input_str: str, observation: str, history: str) -> str:
        prompt = (
            f"你刚刚执行了一个动作但结果不理想（工具抛出异常或执行失败）。\n"
            f"目标：{goal}\n"
            f"动作：{action}\n"
            f"输入：{input_str}\n"
            f"结果：{observation}\n\n"
            f"请用一句话总结：1) 为什么失败；2) 下一步应该调整什么策略。\n"
            f"反思："
        )
        llm = get_llm(self.model_name, keep_alive=-1)  # 复用 9B 常驻
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()

# ==================== 增强版 Agent ====================
class RobustAgentExecutor:
    def __init__(
        self,
        model_name: str = None,
        planner_model: str = None,
        max_steps: int = 12,
        timeout: int = 300,
        max_failures: int = 3,
    ):
        self.model_name = normalize_model_name(model_name or settings.ollama_deep_model)
        self.planner_model = normalize_model_name(planner_model or settings.ollama_deep_model)
        self.max_steps = max_steps
        self.timeout = timeout
        self.max_failures = max_failures
        self.semaphore = asyncio.Semaphore(2)
        self.reflexion = ReflexionModule(self.model_name)
        self.long_term = LongTermMemory()
        self._tool_result_cache: dict[tuple[str, str], tuple[str, bool]] = {}
        self._failed_tool_calls: set[tuple[str, str]] = set()
        self._dynamic_max_steps = self.max_steps

    def _cache_key(self, tool_name: str, tool_input: str) -> tuple[str, str]:
        return (tool_name.strip().lower(), tool_input.strip().lower()[:200])

    def _get_cached_tool_result(self, tool_name: str, tool_input: str) -> tuple[str, bool] | None:
        key = self._cache_key(tool_name, tool_input)
        if key in self._tool_result_cache:
            result, success = self._tool_result_cache[key]
            logger.info(f"工具缓存命中: {tool_name} -> {'成功' if success else '失败'}")
            return result, success
        return None

    def _set_cached_tool_result(self, tool_name: str, tool_input: str, result: str, success: bool):
        key = self._cache_key(tool_name, tool_input)
        self._tool_result_cache[key] = (result, success)

    async def _plan_subtasks(self, user_message: str, history: list[dict]) -> list[dict[str, Any]]:
        history_text = ""
        if history:
            for msg in history[-4:]:
                role = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content']}\n"

        assess_prompt = (
            f"你是一个任务复杂度评估专家。请判断以下用户需求的复杂程度，只输出一个数字 1-5：\n"
            f"1 = 极简单（单步直接回答，如问候、简单事实）\n"
            f"2 = 简单（1-2 个步骤，如查一个知识点）\n"
            f"3 = 中等（3-4 个步骤，需要查询+推理+计算）\n"
            f"4 = 复杂（5-6 个步骤，多工具协作、多源信息整合）\n"
            f"5 = 极复杂（7+ 个步骤，需要深度推理、多轮验证、代码生成调试）\n"
            f"只输出单个数字，不要任何解释。\n\n"
            f"用户需求：{user_message}"
        )
        llm = get_llm(self.planner_model, keep_alive=-1)
        assess_resp = await asyncio.to_thread(llm.invoke, [HumanMessage(content=assess_prompt)])
        try:
            complexity = int(''.join(filter(str.isdigit, assess_resp.content.strip()))[:1])
            complexity = max(1, min(5, complexity))
        except Exception:
            complexity = 2

        max_tasks = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}[complexity]
        self._dynamic_max_steps = {1: 3, 2: 5, 3: 8, 4: 10, 5: 12}[complexity]
        logger.info(f"任务复杂度评级: {complexity}/5，计划子任务数: ≤{max_tasks}，单任务步数: ≤{self._dynamic_max_steps}")

        plan_prompt = (
            f"你是任务规划专家。将用户需求拆解为 **{max_tasks} 个以内** 的原子化子任务。\n"
            f"要求：\n"
            f"1. 每个子任务只做一件事（查询 / 计算 / 分析 / 验证）\n"
            f"2. 子任务之间通过 depends_on 表达依赖关系\n"
            f"3. 输出必须是严格的 JSON 数组，每个元素包含 id, description, depends_on(列表)\n"
            f"4. 不要输出任何其他文字\n\n"
            f"用户需求：{user_message}\n"
            f"历史：{history_text}\n\n"
            f"JSON 输出："
        )

        resp = await asyncio.to_thread(llm.invoke, [HumanMessage(content=plan_prompt)])
        try:
            content = resp.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            subtasks = json.loads(content.strip())
            if not subtasks or len(subtasks) == 0:
                raise ValueError("空列表")
            for i, st in enumerate(subtasks):
                st.setdefault("id", i)
                st.setdefault("depends_on", [])
                st.setdefault("status", "pending")
                st.setdefault("result", "")
            return self._validate_plan(subtasks)
        except Exception as e:
            logger.warning(f"子任务解析失败或为空，回退到单任务模式: {e}")
            return [{"id": 0, "description": user_message, "depends_on": [], "status": "pending", "result": ""}]

    def _validate_plan(self, subtasks: list[dict]) -> list[dict]:
        graph = {st["id"]: [] for st in subtasks}
        in_degree = {st["id"]: 0 for st in subtasks}
        for st in subtasks:
            for dep in st.get("depends_on", []):
                if dep in graph:
                    graph[dep].append(st["id"])
                    in_degree[st["id"]] += 1

        queue = [k for k, v in in_degree.items() if v == 0]
        topo = []
        while queue:
            node = queue.pop(0)
            topo.append(node)
            for nei in graph[node]:
                in_degree[nei] -= 1
                if in_degree[nei] == 0:
                    queue.append(nei)

        if len(topo) != len(subtasks):
            logger.warning("检测到子任务依赖存在环路，回退到单任务")
            return [{
                "id": 0,
                "description": subtasks[0]["description"] if subtasks else "执行任务",
                "depends_on": [],
                "status": "pending",
                "result": "",
            }]

        seen = set()
        filtered = []
        for st in subtasks:
            desc = st.get("description", "").strip().lower()[:40]
            if desc not in seen:
                seen.add(desc)
                filtered.append(st)
        if len(filtered) < len(subtasks):
            logger.info(f"子任务去重：{len(subtasks)} -> {len(filtered)}")
        return filtered

    def _parse_action(self, content: str) -> dict[str, str] | None:
        content = content.strip()
        final_match = re.search(
            r'(?:final|最终答案|答案)[:\s]*["\']?(.*?)["\']?\s*$',
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if final_match:
            return {"final": final_match.group(1).strip()}
        try:
            if content.startswith("{") and content.endswith("}"):
                data = json.loads(content)
                if "final" in data:
                    return {"final": str(data["final"])}
                if "tool" in data and "input" in data:
                    return {"tool": data["tool"], "input": str(data["input"])}
        except Exception:
            pass
        action_match = re.search(r"Action:\s*(.*)", content, re.IGNORECASE)
        input_match = re.search(r"Action Input:\s*(.*)", content, re.IGNORECASE)
        if action_match:
            return {
                "tool": action_match.group(1).strip().strip('"'),
                "input": input_match.group(1).strip().strip('"') if input_match else "",
            }
        return None

    async def run(self, user_message: str, history: list[dict] = None) -> str:
        async with self.semaphore:
            logger.info(f"启动增强 Agent: {user_message[:50]}...")
            working = WorkingMemory(current_goal=user_message)
            episodic = EpisodicMemory()
            working.subtasks = await self._plan_subtasks(user_message, history)
            logger.info(f"规划子任务: {len(working.subtasks)} 个")
            all_results = []
            for task in working.subtasks:
                if task["status"] == "completed":
                    continue
                deps_satisfied = all(
                    working.subtasks[d]["status"] == "completed"
                    for d in task.get("depends_on", [])
                    if isinstance(d, int) and 0 <= d < len(working.subtasks)
                )
                if not deps_satisfied:
                    task["status"] = "skipped"
                    continue
                result = await self._execute_single_task(task, working, episodic, history)
                task["result"] = result
                task["status"] = "completed" if not result.startswith("失败") else "failed"
                all_results.append(f"[{task['description']}] {result}")
            return await self._synthesize(user_message, working, episodic, all_results)

    async def _execute_single_task(self, task, working, episodic, history):
        history_text = ""
        if history:
            for msg in history[-4:]:
                role = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content']}\n"

        tool_desc = tool_registry.get_all_descriptions()
        memory_hint = ""
        if working.intermediate_results:
            memory_hint = "已获得的中间结果：\n" + "\n".join(
                [f"- {k}: {v[:100]}" for k, v in working.intermediate_results.items()]
            )

        current_date = datetime.now().strftime("%Y年%m月%d日")
        system_prompt = (
            f"你是一个智能 Agent，当前子任务：{task['description']}\n"
            f"当前日期：{current_date}\n\n"
            f"可用工具：\n{tool_desc}\n\n"
            f"{memory_hint}\n\n"
            f"规则：\n"
            f'1. 必须使用 JSON 格式调用工具：{{"tool": "工具名", "input": "输入"}}\\n'
            f'2. 获得足够信息后，输出：{{"final": "最终答案"}} 或直接以 \'Final Answer:\' 开头输出文本。\\n'
            f"3. 不要重复调用已成功的工具，直接基于结果作答。\n"
            f"4. 如果连续两次无法提取有效信息，请立即输出 final 答案。\n\n"
            f"历史对话：\n{history_text}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"开始执行子任务：{task['description']}"),
        ]

        llm = get_llm(self.model_name, keep_alive=-1)
        step = 0
        start_time = time.time()
        used_tools = []
        last_tool_result = None
        dynamic_limit = getattr(self, '_dynamic_max_steps', self.max_steps)

        while step < dynamic_limit and (time.time() - start_time) < self.timeout:
            step += 1
            response = await asyncio.to_thread(llm.invoke, messages)
            content = response.content.strip()
            messages.append(AIMessage(content=content))

            action = self._parse_action(content)
            if action and "final" in action:
                logger.info(f"子任务完成: {task['description'][:30]}...")
                return action["final"]

            if action and "tool" in action:
                tool_name = action["tool"].strip()
                tool_input = action["input"]

                tool = tool_registry.get_tool(tool_name)
                if not tool:
                    observation = f"错误：未找到工具 '{tool_name}'。可用工具：{list(tool_registry._tools.keys())}"
                    success = False
                    messages.append(HumanMessage(content=f"Observation: {observation}"))
                    working.failure_count += 1
                    if working.failure_count >= self.max_failures:
                        return f"失败：连续调用不存在工具 {working.failure_count} 次"
                    continue

                cache_key = self._cache_key(tool_name, tool_input)
                if cache_key in self._failed_tool_calls:
                    cached = self._get_cached_tool_result(tool_name, tool_input)
                    observation = f"[此前已失败，不再重试] {cached[0] if cached else '未知错误'}"
                    messages.append(HumanMessage(content=f"Observation: {observation}"))
                    working.failure_count += 1
                    if working.failure_count >= self.max_failures:
                        return f"失败：多次调用已确认失败的工具"
                    continue

                cached = self._get_cached_tool_result(tool_name, tool_input)
                if cached is not None:
                    observation, success = cached
                    if not success:
                        messages.append(HumanMessage(content=f"Observation: [此前已失败，跳过] {observation}"))
                        working.failure_count += 1
                        if working.failure_count >= self.max_failures:
                            return f"失败：连续 {working.failure_count} 次工具调用失败（含缓存失败）"
                        continue
                else:
                    call_sig = f"{tool_name}:{tool_input}"
                    repeat_count = sum(1 for u in used_tools[-6:] if u == call_sig)
                    if repeat_count >= 2:
                        messages.append(HumanMessage(content="你已经重复调用同一工具多次，请立即输出 final 答案。"))
                        continue
                    used_tools.append(call_sig)

                    try:
                        if inspect.iscoroutinefunction(tool):
                            observation = await tool(tool_input)
                        else:
                            observation = await asyncio.to_thread(tool, tool_input)
                        success = True
                        last_tool_result = observation
                    except Exception as e:
                        observation = f"工具执行出错: {e!s}"
                        success = False
                        self._failed_tool_calls.add(cache_key)

                    self._set_cached_tool_result(tool_name, tool_input, observation, success)

                self.long_term.record_tool_result(tool_name, success)
                logger.info(f"[{task['description'][:20]}] 工具 {tool_name} -> {'成功' if success else '失败'}")

                reflection = ""
                if not success:
                    working.failure_count += 1
                    if working.failure_count <= self.max_failures:
                        reflection = self.reflexion.reflect(
                            task["description"], tool_name, tool_input, observation, history_text
                        )
                        logger.info(f"反思: {reflection}")
                    else:
                        return f"失败：连续 {working.failure_count} 次工具调用失败，最后错误：{observation}"

                episodic.add(step, tool_name, tool_input, observation, reflection)
                if reflection:
                    observation += f"\n[反思] {reflection}"

                messages.append(HumanMessage(content=f"Observation: {observation}"))
                working.intermediate_results[f"{tool_name}_{step}"] = observation
            else:
                messages.append(
                    HumanMessage(content='请立即输出 final 答案，格式：{"final": "..."} 或 Final Answer: ...')
                )

        if last_tool_result:
            logger.info("循环结束，使用最后一次工具结果生成答案")
            prompt = f"根据以下信息回答问题：\n{last_tool_result}\n\n问题：{task['description']}\n答案："
            llm = get_llm(self.model_name, keep_alive=-1)
            resp = await asyncio.to_thread(llm.invoke, [HumanMessage(content=prompt)])
            return resp.content.strip()
        return "失败：达到最大步数或超时，且无可用工具结果。"

    async def _synthesize(self, user_message, working, episodic, results):
        context = "\n".join(results)
        prompt = (
            f"基于以下子任务执行结果，回答用户的原始问题。简洁、准确。\n\n"
            f"用户问题：{user_message}\n"
            f"子任务结果：\n{context}\n\n"
            f"最终回答："
        )
        llm = get_llm(self.model_name, keep_alive=-1)
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()

agent_executor = RobustAgentExecutor()

@app.post("/agent")
async def agent_endpoint(req: ChatRequest):
    try:
        answer = await agent_executor.run(req.message, req.history)
        return ChatResponse(response=answer)
    except Exception as e:
        logger.error(f"Agent 运行失败: {e!s}")
        return ChatResponse(response=f"Agent 运行失败: {e!s}")

# ==================== 文档与评估 ====================
@app.post("/add_docs")
async def add_docs(files: list[str]):
    count = add_documents_to_store(files)
    return {"status": "success", "chunks_added": count}

@app.post("/upload_docs")
async def upload_docs(files: list[UploadFile] = File(...)):
    saved_paths = []
    upload_dir = Path("./uploaded_docs")
    upload_dir.mkdir(exist_ok=True)
    for file in files:
        file_path = upload_dir / f"{__import__('uuid').uuid4().hex}_{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(file_path))
    count = add_documents_to_store(saved_paths)
    return {"status": "success", "chunks_added": count, "files": [Path(p).name for p in saved_paths]}

@app.post("/retrieve")
async def retrieve_contexts(query: str):
    docs = retrieve_with_hybrid_and_rerank(query)
    return {"contexts": [doc.page_content for doc in docs]}

@app.get("/eval/report")
async def get_eval_report():
    eval_path = Path(settings.eval_file_path)
    if not eval_path.exists():
        return {"message": "暂无评估报告"}
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ==================== 健康检查 ====================
@app.get("/health")
async def health():
    status = {"api": "ok", "timestamp": datetime.now().isoformat()}
    try:
        requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5, proxies={"http": None, "https": None})
        status["ollama"] = "ok"
    except Exception as e:
        status["ollama"] = f"error: {e}"
    try:
        _ = embedding.embed_query("ping")
        status["chroma"] = "ok"
    except Exception as e:
        status["chroma"] = f"error: {e}"
    status["tavily"] = "configured" if TAVILY_API_KEY else "not_configured"
    overall = "ok" if status["ollama"] == "ok" and status["chroma"] == "ok" else "degraded"
    return {"status": overall, "services": status}

# ==================== 会话持久化 ====================
def _session_path(sid: str) -> Path:
    return Path(settings.sessions_dir) / f"{sid}.json"

@app.get("/sessions")
async def list_sessions():
    sessions = []
    for p in Path(settings.sessions_dir).glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "id": data.get("id", p.stem),
                    "title": data.get("title", "新对话"),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", []))
                })
        except Exception:
            continue
    return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

@app.get("/sessions/{sid}")
async def get_session(sid: str):
    path = _session_path(sid)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/sessions/{sid}")
async def save_session(sid: str, body: dict):
    path = _session_path(sid)
    body["updated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)
    return {"status": "saved"}

@app.delete("/sessions/{sid}")
async def delete_session_api(sid: str):
    path = _session_path(sid)
    if path.exists():
        path.unlink()
    return {"status": "deleted"}

# ==================== 启动 ====================
if __name__ == "__main__":
    print("正在预热 Reranker 模型...")
    _ = get_reranker()

    print("正在预热嵌入模型 (nomic-embed-text)...")
    for i in range(5):
        try:
            _ = embedding.embed_query("warmup")
            print("嵌入模型预热成功")
            break
        except Exception as e:
            print(f"嵌入模型预热失败，重试 {i + 1}/5: {e}")
            time.sleep(3)
    else:
        print("嵌入模型预热最终失败，将在第一次请求时自动重试")

    print(f"正在预热 LLM ({settings.ollama_deep_model}) 常驻...")
    try:
        warmup_llm = get_llm(settings.ollama_deep_model, keep_alive=-1)
        _ = warmup_llm.invoke([HumanMessage(content="你好")])
        print(f"LLM {settings.ollama_deep_model} 预热成功，已常驻显存")
    except Exception as e:
        print(f"LLM {settings.ollama_deep_model} 预热失败: {e}")

    print("🚀 智能助手已启动，支持全 GPU 4096 上下文 + 三角形审查 + ReAct Agent + HyDE + Self-RAG + Vision")
    print(f"   - 上下文长度: {settings.ollama_context_length}")
    uvicorn.run(app, host=settings.host, port=settings.port)