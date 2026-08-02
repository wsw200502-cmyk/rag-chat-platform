import time, ollama, psutil

client = ollama.Client(host="http://localhost:11434")  # 显式指定地址

model = "qwen2.5:14b"
context_sizes = [2048, 4096, 8192]
prompt = "请用至少500字详细介绍Python异步编程的核心概念"

for ctx in context_sizes:
    try:
        print(f"\n===== 上下文长度: {ctx} =====")
        mem_before = psutil.virtual_memory().used / (1024**3)
        start = time.time()
        resp = client.chat(
            model=model,
            messages=[{"role":"user","content":prompt}],
            options={"num_ctx": ctx}
        )
        end = time.time()
        mem_after = psutil.virtual_memory().used / (1024**3)
        print(f"耗时: {end-start:.2f}s | 系统内存变化: {mem_after-mem_before:.2f}GB")
    except Exception as e:
        print(f"上下文长度 {ctx} 测试失败: {e}")