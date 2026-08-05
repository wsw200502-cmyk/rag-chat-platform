import asyncio
import inspect
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List

import requests

# ========== ① 禁用系统代理，防止 localhost 被劫持导致 502 ==========
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

from dotenv import load_dotenv

# 显式指定 .env 路径（确保和 agent_api.py 同目录）
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)

# 如果 load_dotenv 因 BOM 头解析失败，手动解析兜底
if not os.getenv("TAVILY_API_KEY"):
    try:
        with open(env_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig 自动去掉 BOM 头
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        print("[DEBUG] 已手动解析 .env（load_dotenv 未生效）")
    except Exception as e:
        print(f"[DEBUG] 手动解析 .env 失败: {e}")

# 调试用：确认 Key 是否加载成功
print(f"[DEBUG] .env 路径: {env_path}")
print(f"[DEBUG] TAVILY_API_KEY 是否加载: {os.getenv('TAVILY_API_KEY') is not None}")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    from langchain_tavily import TavilySearch

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ==================== 结构化日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_api")

if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY 未设置，网络搜索功能将不可用。若需使用，请复制 .env.example 为 .env 并填入 Key。")

# ==================== 模型名称规范化 ====================
def normalize_model_name(name: str) -> str:
    """将旧版或别名模型名称映射到实际可用名称"""
    mapping = {
        "qwen2.5:14b": "qwen3.5:9b",   # 旧名称映射到深度推理模型
        "qwen3:14b": "qwen3.5:9b",     # 兼容前端旧 14B 标签
    }
    return mapping.get(name, name)

# ==================== 模型缓存（支持 keep_alive 参数） ====================
llm_cache = {}


def get_llm(model_name: str, keep_alive: int = -1):
    """
    获取或创建 LLM 实例。
    keep_alive=-1 表示常驻显存（适合 qwen3.5:9b 深度任务）。
    keep_alive=0 表示用后即焚（适合 qwen3.5:4b 轻度对话）。
    """
    model_name = normalize_model_name(model_name)  # 规范化
    cache_key = (model_name, keep_alive)
    if cache_key not in llm_cache:
        logger.info(f"初始化模型: {model_name} (keep_alive={keep_alive})")
        llm_cache[cache_key] = ChatOllama(
            model=model_name,
            temperature=0,
            base_url="http://127.0.0.1:11434",  # ← 强制 IPv4
            num_predict=8192,
            keep_alive=keep_alive,
        )
    return llm_cache[cache_key]


# ==================== 自定义 Embedding（绕过 langchain-ollama 502 问题） ====================
class OllamaEmbeddingDirect:
    """直接请求 Ollama /api/embed，禁用系统代理，避免 502。"""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://127.0.0.1:11434"):  # ← 强制 IPv4
        self.model = model
        self.base_url = base_url.rstrip("/")

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

# ==================== 嵌入模型 ====================
embedding = OllamaEmbeddingDirect(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",  # ← 强制 IPv4
)

# ==================== 向量数据库（内嵌模式） ====================
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
)

# ==================== 网络搜索（容错降级） ====================
if TAVILY_API_KEY:
    search = TavilySearch(api_key=TAVILY_API_KEY, max_results=3)
else:
    search = None

# ==================== 混合检索全局变量 ====================
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
        # 优先使用本地缓存，避免网络下载失败
        reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            local_files_only=True,
            trust_remote_code=True
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
    """执行混合检索 + Cross-Encoder Reranker 重排序。

    先分别进行向量检索和 BM25 检索，合并去重后
    使用 Cross-Encoder 对结果重新打分排序。

    Args:
        query: 用户查询字符串。
        k_vector: 向量检索返回的文档数量。
        k_bm25: BM25 检索返回的文档数量。
        final_k: Reranker 最终返回的文档数量。
        max_retries: 向量检索失败时的最大重试次数。

    Returns:
        按相关性排序后的文档列表。
    """
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
    """构建发送给 LLM 的消息列表。

    整合本地知识库、网络搜索结果、历史对话和系统提示。

    Args:
        question: 当前用户问题。
        history: 历史对话记录，每项为 {"role": "user"|"assistant", "content": "..."}。
        extra_context: 额外的上下文信息（如 HyDE 检索结果）。

    Returns:
        符合 LangChain 格式的消息对象列表。
    """
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
    model_name: str = "qwen3.5:4b",  # ← 轻度对话默认 4B
) -> str:
    """生成非流式回答。

    调用指定模型，基于检索到的上下文和历史对话生成回答，
    并对输出进行后处理（如自动补全代码块）。

    Args:
        question: 用户问题。
        history: 历史对话记录。
        model_name: 使用的 Ollama 模型名称。

    Returns:
        经过后处理的回答字符串。
    """
    model_name = normalize_model_name(model_name)  # 规范化
    messages = build_messages(question, history)
    # 4B 轻量用完即焚，9B 深度常驻
    keep_alive = 0 if "4b" in model_name.lower() else -1
    llm = get_llm(model_name, keep_alive=keep_alive)
    response = llm.invoke(messages)
    return post_process(response.content, question)


# ==================== 多模型协作 ====================
def generate_collaborative(question: str, history: list[dict[str, str]]) -> str:
    # ----- 规划：深度推理 9B 常驻 -----
    planner = get_llm("qwen3.5:9b", keep_alive=-1)
    planner_prompt = (
        "你是一个任务规划专家。将用户需求分解为清晰的步骤，并生成优化的提示词。"
        "只输出优化后的提示词，不要解释。\n"
        f"用户需求：{question}"
    )
    plan_messages = [HumanMessage(content=planner_prompt)]
    plan_response = planner.invoke(plan_messages)
    optimized_query = plan_response.content.strip()

    # ----- 执行：深度推理 9B 常驻（共享同一实例）-----
    executor = get_llm("qwen3.5:9b", keep_alive=-1)
    messages = build_messages(optimized_query, history)
    messages[-1] = HumanMessage(
        content=(f"根据资料回答用户问题。\n用户问题：{question}\n优化提示词：{optimized_query}")
    )
    final_response = executor.invoke(messages)
    return final_response.content


# ==================== FastAPI 应用 ====================
app = FastAPI(title="AI Assistant (RAG + Agent + HyDE + Self-RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    model: str = "qwen3.5:4b"  # ← 默认轻度对话 4B
    history: list[dict[str, str]] = []
    models: list[str] | None = None


class ChatResponse(BaseModel):
    response: str


# ==================== 原有路由 ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model = normalize_model_name(req.model)
    answer = generate_answer(req.message, req.history, model)
    return ChatResponse(response=answer)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_stream():
        model = normalize_model_name(req.model)
        # 4B 轻量用完即焚，9B 深度常驻
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


# ==================== 三模型链式审查 ====================
@app.post("/chat/review")
async def chat_review(req: ChatRequest):
    for i in range(5):
        try:
            _ = embedding.embed_query("warmup")
            break
        except Exception as e:
            if i < 4:
                logger.warning(f"嵌入预热失败，重试 {i + 1}/5: {e}")
                time.sleep(2)
            else:
                logger.error(f"嵌入预热最终失败: {e}")

    # 三模型审查：解耦 + 执行 + 审查，全部使用 9B 深度推理
    models = req.models or ["qwen3.5:9b", "qwen3.5:9b", "qwen3.5:9b"]
    models = [normalize_model_name(m) for m in models]
    if len(models) < 3:
        return ChatResponse(response="审查模式需要至少三个模型")

    def robust_llm_call(llm, messages, max_retries=5):
        for attempt in range(max_retries):
            try:
                return llm.invoke(messages)
            except Exception as e:
                if "502" in str(e) and attempt < max_retries - 1:
                    wait_time = 2**attempt + 15
                    logger.warning(f"LLM调用失败(502)，等待 {wait_time}s 重试 {attempt + 1}/{max_retries - 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e

    # 解耦：9B 常驻（共享同一实例）
    decomposer = get_llm(models[0], keep_alive=-1)
    decomp_prompt = (
        f"你是需求分析专家。将用户需求分解为清晰步骤并生成优化提示词。只输出优化提示词。\n用户需求：{req.message}"
    )
    try:
        decomp_resp = await asyncio.to_thread(robust_llm_call, decomposer, [HumanMessage(content=decomp_prompt)])
        optimized_query = decomp_resp.content.strip()
    except Exception as e:
        return ChatResponse(response=f"需求解耦失败: {e!s}")

    # 执行生成：9B 常驻（共享同一实例）
    executor_model = models[1] if len(models) > 1 else "qwen3.5:9b"
    executor = get_llm(executor_model, keep_alive=-1)
    messages = build_messages(optimized_query, req.history)
    try:
        exec_resp = await asyncio.to_thread(robust_llm_call, executor, messages)
        draft_answer = exec_resp.content
    except Exception as e:
        return ChatResponse(response=f"生成回答失败: {e!s}")

    # 审查：9B 常驻（共享同一实例）
    reviewer = get_llm(models[2], keep_alive=-1)
    review_prompt = (
        f"你是严格的事实核查员。审查回答是否忠实于知识库和用户问题。\n"
        f"如果准确输出'PASS'，否则输出'FAIL: 具体问题'。\n\n"
        f"原始问题：{req.message}\n"
        f"优化提示词：{optimized_query}\n"
        f"回答草案：{draft_answer}\n\n"
        f"审查结果（PASS 或 FAIL: ...）："
    )
    try:
        review_resp = await asyncio.to_thread(robust_llm_call, reviewer, [HumanMessage(content=review_prompt)])
        review_result = review_resp.content.strip()
    except Exception as e:
        review_result = f"审查出错: {e!s}"

    final_answer = f"{draft_answer}\n\n[审查意见: {review_result}]"
    return ChatResponse(response=final_answer)


# ==================== 文档与评估 ====================
@app.post("/add_docs")
async def add_docs(files: list[str]):
    count = add_documents_to_store(files)
    return {"status": "success", "chunks_added": count}


@app.post("/retrieve")
async def retrieve_contexts(query: str):
    docs = retrieve_with_hybrid_and_rerank(query)
    return {"contexts": [doc.page_content for doc in docs]}


EVAL_FILE_PATH = "D:/ragas_eval/eval_results.json"


@app.get("/eval/report")
async def get_eval_report():
    if not os.path.exists(EVAL_FILE_PATH):
        return {"message": "暂无评估报告"}
    with open(EVAL_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ==================== HyDE 检索增强生成 ====================
def generate_hypothetical_answer(question: str, model_name: str = "qwen3.5:9b") -> str:
    model_name = normalize_model_name(model_name)
    prompt = f"""请根据你的知识，为以下问题生成一段假设性的回答（哪怕不知道确切答案，也请给出合理的推测性描述）。只需输出回答内容，不需要解释。

问题：{question}
假设答案："""
    llm = get_llm(model_name, keep_alive=-1)  # 深度推理 9B 常驻
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


def hyde_retrieve_and_answer(question: str, history: list[dict[str, str]], model_name: str = "qwen3.5:9b") -> str:
    model_name = normalize_model_name(model_name)
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

    llm = get_llm(model_name, keep_alive=-1)  # 深度推理 9B 常驻
    resp = llm.invoke(messages)
    return resp.content.strip()


@app.post("/chat/hyde")
async def chat_hyde(req: ChatRequest):
    try:
        model = normalize_model_name(req.model)
        answer = await asyncio.to_thread(hyde_retrieve_and_answer, req.message, req.history, model)
        return ChatResponse(response=answer)
    except Exception as e:
        logger.error(f"HyDE 请求失败: {e!s}")
        return ChatResponse(response=f"HyDE 处理失败: {e!s}")


# ==================== Self-RAG 实现 ====================


def should_retrieve(question: str) -> bool:
    """判断问题是否需要检索外部知识。

    基于关键词规则判断：闲聊问候类问题不需要检索，
    知识性、事实性问题需要检索。

    Args:
        question: 用户输入的问题字符串。

    Returns:
        True 表示需要检索，False 表示可直接回答。
    """
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
    decision_model: str = "qwen3.5:9b",
    generation_model: str = "qwen3.5:9b",
    max_iterations: int = 3,
) -> str:
    decision_model = normalize_model_name(decision_model)
    generation_model = normalize_model_name(generation_model)
    # 决策与生成均使用 9B 深度推理，共享同一常驻实例
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
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
            else:
                context = "无相关结果"
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
            decision_model="qwen3.5:9b",
            generation_model="qwen3.5:9b",
        )
        return ChatResponse(response=answer)
    except Exception as e:
        logger.error(f"Self-RAG 请求失败: {e!s}")
        return ChatResponse(response=f"Self-RAG 处理失败: {e!s}")


# ==================== ReAct Agent 模块（增强版） ====================


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

    def get_tool(self, name: str) -> Callable:
        return self._tools.get(name)

    def get_all_descriptions(self) -> str:
        return "\n".join([f"- {name}: {desc}" for name, desc in self._descriptions.items()])


tool_registry = ToolRegistry()


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
        self.events.append(
            {
                "step": step,
                "action": action,
                "input": input_str,
                "output": output,
                "reflection": reflection,
                "timestamp": time.time(),
            }
        )


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
    def __init__(self, model_name: str = "qwen3.5:9b"):
        self.model_name = normalize_model_name(model_name)

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
        llm = get_llm(self.model_name, keep_alive=-1)  # 共享 9B 常驻实例
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()


# ==================== 增强版 Agent ====================


class RobustAgentExecutor:
    def __init__(
        self,
        model_name: str = "qwen3.5:9b",
        planner_model: str = "qwen3.5:9b",
        max_steps: int = 8,
        timeout: int = 180,
        max_failures: int = 2,
    ):
        self.model_name = normalize_model_name(model_name)
        self.planner_model = normalize_model_name(planner_model)
        self.max_steps = max_steps
        self.timeout = timeout
        self.max_failures = max_failures
        self.semaphore = asyncio.Semaphore(2)
        self.reflexion = ReflexionModule(model_name)
        self.long_term = LongTermMemory()

    async def _plan_subtasks(self, user_message: str, history: list[dict]) -> list[dict[str, Any]]:
        history_text = ""
        if history:
            for msg in history[-4:]:
                role = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content']}\n"

        prompt = (
            f"你是任务规划专家。将用户需求拆解为 1–3 个核心子任务，只包含最必要步骤。\n"
            f"输出必须是严格的 JSON 数组，每个元素包含 id, description, depends_on(依赖的前置id列表)。\n"
            f"不要输出任何其他文字。\n\n"
            f"用户需求：{user_message}\n"
            f"历史：{history_text}\n\n"
            f"JSON 输出："
        )

        llm = get_llm(self.planner_model, keep_alive=-1)  # 9B 常驻
        resp = llm.invoke([HumanMessage(content=prompt)])
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
            return subtasks
        except Exception as e:
            logger.warning(f"子任务解析失败或为空，回退到单任务模式: {e}")
            return [
                {
                    "id": 0,
                    "description": user_message,
                    "depends_on": [],
                    "status": "pending",
                    "result": "",
                }
            ]

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
        """执行增强版 ReAct Agent 流程。

        包括任务规划、子任务执行、工具调用、反思和结果综合。
        支持记忆系统和防重复调用机制。

        Args:
            user_message: 用户的原始输入。
            history: 历史对话记录。

        Returns:
            Agent 的最终回答字符串。
        """
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

        # 核心执行模型：9B 深度推理常驻
        llm = get_llm(self.model_name, keep_alive=-1)
        step = 0
        start_time = time.time()
        used_tools = []
        last_tool_result = None

        while step < self.max_steps and (time.time() - start_time) < self.timeout:
            step += 1
            response = await asyncio.to_thread(llm.invoke, messages)
            content = response.content.strip()
            messages.append(AIMessage(content=content))

            action = self._parse_action(content)
            if action and "final" in action:
                logger.info(f"子任务完成: {task['description'][:30]}...")
                return action["final"]

            if action and "tool" in action:
                tool_name = action["tool"]
                tool_input = action["input"]

                call_sig = f"{tool_name}:{tool_input}"
                repeat_count = sum(1 for u in used_tools[-6:] if u == call_sig)
                if repeat_count >= 2:
                    messages.append(HumanMessage(content="你已经重复调用同一工具多次，请立即输出 final 答案。"))
                    continue
                used_tools.append(call_sig)

                tool = tool_registry.get_tool(tool_name)
                if not tool:
                    observation = f"错误：未找到工具 '{tool_name}'"
                    success = False
                else:
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

                self.long_term.record_tool_result(tool_name, success)
                logger.info(f"[{task['description'][:20]}] 工具 {tool_name} -> {'成功' if success else '失败'}")

                reflection = ""
                if not success:
                    working.failure_count += 1
                    if working.failure_count <= self.max_failures:
                        reflection = self.reflexion.reflect(
                            task["description"],
                            tool_name,
                            tool_input,
                            observation,
                            history_text,
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

    # 预热 qwen3.5:9b（深度推理常驻）
    print("正在预热 LLM (qwen3.5:9b) 常驻...")
    try:
        warmup_llm = get_llm("qwen3.5:9b", keep_alive=-1)
        _ = warmup_llm.invoke([HumanMessage(content="你好")])
        print("LLM qwen3.5:9b 预热成功，已常驻显存")
    except Exception as e:
        print(f"LLM qwen3.5:9b 预热失败: {e}")

    print("🚀 智能助手已启动，支持混合检索+Reranker+多模型协作+链式审查+ReAct Agent+HyDE+Self-RAG")
    print("   - 所有 Ollama 请求使用 127.0.0.1 (IPv4)，避免 localhost 解析为 ::1 导致 404")
    print("   - 轻度对话 (/chat)：qwen3.5:4b，keep_alive=0（用完即焚，不占显存）")
    print("   - 深度推理（审查/Agent/HyDE/Self-RAG）：qwen3.5:9b，keep_alive=-1（常驻显存）")
    print("   - 模型名称自动规范化：qwen3:14b / qwen2.5:14b -> qwen3.5:9b")
    uvicorn.run(app, host="0.0.0.0", port=8000)