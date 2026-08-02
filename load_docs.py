from rag_agent import load_documents_to_store

# 注意：文件路径必须与 docs 文件夹内的实际文件名完全一致（包括扩展名）
files = [
    "D:/my-langchain-agent/docs/产品手册.txt",
    "D:/my-langchain-agent/docs/常见问题.txt"
]

count = load_documents_to_store(files)
print(f"成功导入 {count} 个文本块")