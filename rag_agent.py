import os
os.environ["TAVILY_API_KEY"] = "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"   
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

# ========== 模型初始化 ==========
llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://localhost:11434")
embedding = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# ========== 向量数据库（持久化到D盘）==========
vectorstore = Chroma(persist_directory="D:/chroma_db", embedding_function=embedding)

# ========== 文档加载工具 ==========
def load_documents_to_store(file_paths):
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
    return len(chunks)

# ========== 搜索工具（可选）==========
search = TavilySearch(api_key=os.environ["TAVILY_API_KEY"], max_results=3)

class RAGAgent:
    def answer(self, question: str) -> str:
        # 本地知识库检索
        local_docs = vectorstore.similarity_search(question, k=3)
        local_context = "\n".join([doc.page_content for doc in local_docs])
        # 网络搜索（可注释掉以禁用）
        try:
            web_results = search.invoke(question)
            web_context = str(web_results)
        except:
            web_context = "网络搜索不可用"
        # 提示词
        system = SystemMessage(content="你是一个专业助手，请严格根据提供的资料回答。若资料不足，请如实说明。用中文回复。")
        user = HumanMessage(content=f"【本地知识库】\n{local_context}\n\n【网络搜索结果】\n{web_context}\n\n用户问题：{question}")
        return llm.invoke([system, user]).content

agent = RAGAgent()

# ========== API接口 ==========
app = FastAPI()
class AskRequest(BaseModel): question: str
class AskResponse(BaseModel): answer: str

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    return AskResponse(answer=agent.answer(req.question))

@app.post("/load_docs")
async def load_docs(files: list[str]):
    count = load_documents_to_store(files)
    return {"status": "success", "chunks_added": count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)