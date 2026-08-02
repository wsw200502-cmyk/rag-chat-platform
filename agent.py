import os
import warnings
warnings.filterwarnings("ignore")

# ========== 1. 密钥（必须替换成你的真实密钥）==========
os.environ["DEEPSEEK_API_KEY"] = "sk-sk-6826f4094744485e92f70f60d4d6371a"
os.environ["TAVILY_API_KEY"] = "tvly-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"

# ========== 2. 导入依赖 ==========
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch          # 修改这里

# ========== 3. 配置 LLM（大脑）==========
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ["DEEPSEEK_API_KEY"]
)

# ========== 4. 定义工具 ==========
search = TavilySearch(
    api_key=os.environ["TAVILY_API_KEY"],
    max_results=2
)

@tool
def get_current_weather(city: str) -> str:
    """查询指定城市的实时天气。参数 city 是城市名称（中文或英文）。"""
    return f"{city} 现在晴朗，气温 26°C，湿度 60%。"

tools = [search, get_current_weather]

# ========== 5. 记忆（多轮对话支持）==========
memory = MemorySaver()

# ========== 6. 创建 Agent ==========
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt="你是一个聪明、谨慎的助手。优先用工具获取实时信息，不要编造。"
)

# ========== 7. 对话循环 ==========
print("✅ DeepSeek Agent 启动！输入 'quit' 退出。")

config = {"configurable": {"thread_id": "user-1"}}

while True:
    user_input = input("\n你：")
    if user_input.lower() in ["quit", "exit"]:
        break

    result = agent.invoke(
        {"messages": [("user", user_input)]},
        config=config
    )

    answer = result["messages"][-1].content
    print(f"\nAgent：{answer}")