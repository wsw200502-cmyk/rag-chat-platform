import time, ollama, psutil

client = ollama.Client(host="http://localhost:11434")  # 显式指定地址

models = ["qwen2.5:7b", "qwen2.5:14b"]
prompt = "请用中文解释什么是量子纠缠，不少于200字"

for model in models:
    try:
        print(f"\n===== 测试模型: {model} =====")
        mem_before = psutil.virtual_memory().used / (1024**3)
        start = time.time()
        resp = client.chat(model=model, messages=[{"role":"user","content":prompt}])
        end = time.time()
        mem_after = psutil.virtual_memory().used / (1024**3)
        print(f"耗时: {end-start:.2f}s | 系统内存变化: {mem_after-mem_before:.2f}GB")
        print(f"回答长度: {len(resp['message']['content'])} 字符")
    except Exception as e:
        print(f"模型 {model} 测试失败: {e}")