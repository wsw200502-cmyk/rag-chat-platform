import requests
import json
import sys

# 您的 Agent 服务地址
API_URL = "http://localhost:8000/chat"

def chat_with_agent(message: str) -> str:
    """发送消息给 Agent，并返回回复"""
    try:
        resp = requests.post(
            API_URL,
            json={"message": message},
            timeout=30  # 30秒超时
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "（空回复）")
        else:
            return f"❌ 服务器返回错误状态码：{resp.status_code}\n{resp.text}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到 Agent 服务，请确保服务已启动（python agent_api.py）"
    except requests.exceptions.Timeout:
        return "⏰ 请求超时，请稍后重试"
    except Exception as e:
        return f"❌ 请求异常：{e}"

def main():
    print("🤖 欢迎使用您的智能助手 Agent")
    print("💡 输入消息后按回车发送，输入 'exit' 或 'quit' 退出")
    print("-" * 50)
    
    while True:
        # 获取用户输入
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 再见！")
            break
        
        # 调用 Agent
        print("AI：", end="", flush=True)
        reply = chat_with_agent(user_input)
        print(reply)

if __name__ == "__main__":
    main()
