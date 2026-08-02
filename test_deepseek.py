import requests
import json

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-dbce6511ae7144b08d5e9460a56c58c1",
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "用中文回复：你好"}]
}
response = requests.post(url, headers=headers, json=data)
print("状态码:", response.status_code)
print("回复:", response.json())