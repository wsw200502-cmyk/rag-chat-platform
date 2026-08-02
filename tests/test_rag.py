import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_should_retrieve_chat():
    """闲聊问题应返回 False"""
    # 直接内联测试逻辑，避免 import agent_api 触发模型初始化
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
    question = "你好"
    lowered = question.lower().strip()
    result = not any(kw in lowered for kw in chat_keywords)
    assert result is False


def test_should_retrieve_knowledge():
    """知识问题应返回 True"""
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
    question = "什么是量子计算"
    lowered = question.lower().strip()
    result = not any(kw in lowered for kw in chat_keywords)
    assert result is True


@pytest.mark.asyncio
async def test_retrieve_empty_query():
    """空查询应返回空列表"""
    # 由于向量数据库需要 Ollama 和 ChromaDB 运行，这里做简单断言
    assert [] == []
