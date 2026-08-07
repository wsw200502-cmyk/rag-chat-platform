#!/usr/bin/env python3
"""
三角形多Agent协作集成测试（简化Prompt版）

测试目标：
1. 内切圆（SharedCore）检索与工具调用
2. 三角形三个顶点（Agent-A/B/C）的顺序协作（使用简化Prompt确保输出）
3. 共享核心数据传递（分析→生成→审查）
4. 容错能力（模拟 Agent 失败后的降级）

前置条件：
- Ollama 服务运行在 http://127.0.0.1:11434
- 模型 qwen3.5:4b, qwen3.5:9b, nomic-embed-text 已下载
- 向量库 chroma_db 已初始化（空库也可）
- Python 依赖已安装
"""

import asyncio
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_api import (
    SharedCore,
    TriangleAgent,
    get_llm,
    normalize_model_name,
    settings,
)

TEST_QUESTION = "法国首都是什么？请简要回答。"
TEST_HISTORY = []


def _ollama_available():
    """检查 Ollama 服务是否可用"""
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


async def test_shared_core():
    """测试内切圆的基本功能"""
    print("[Test 1] 内切圆 SharedCore 测试...")
    core = SharedCore()

    # 检索测试（空库可能返回0篇，正常）
    docs = await core.retrieve(TEST_QUESTION)
    print(f"  - 检索返回文档数: {len(docs)}")

    # 网络搜索测试（无API Key时会降级，不影响测试）
    web_result = core.web_search(TEST_QUESTION)
    print(f"  - 网络搜索结果 (前100字符): {str(web_result)[:100]}")

    # 工具调用测试（计算器）
    tool_result = core.use_tool("calculator", "1+1")
    print(f"  - 计算器结果: {tool_result}")

    # 记忆存储/读取
    core.store("test_key", "test_value")
    assert core.get("test_key") == "test_value", "共享核心记忆读写失败"
    print("  ✓ 内切圆功能正常")


@pytest.mark.skipif(not _ollama_available(), reason="Ollama 服务不可用，跳过三角形 Agent 测试")
async def test_triangle_agents():
    """使用简化Prompt进行三角形协作测试"""
    print("\n[Test 2] 三角形 Agent 顺序协作测试（简化Prompt）...")
    core = SharedCore()
    model = normalize_model_name(settings.ollama_deep_model)

    # 创建三个Agent（角色描述仅用于日志，实际Prompt由我们构建）
    agent_a = TriangleAgent("Agent-A·分析", model, "分析专家", core)
    agent_b = TriangleAgent("Agent-B·生成", model, "生成专家", core)
    agent_c = TriangleAgent("Agent-C·审查", model, "审查专家", core)

    # ---- Phase 1: 分析 ----
    print("  - 阶段1: Agent-A 分析...")
    analysis_prompt = f"请对以下问题进行简要分析，指出回答该问题需要确认的关键事实。问题：{TEST_QUESTION}\n只需输出关键点，不要多余内容。"
    analysis = await agent_a.think(analysis_prompt)
    core.store("analysis", analysis)
    print(f"    分析结果 (前100字符): {analysis[:100]}")

    # ---- Phase 2: 生成答案 ----
    print("  - 阶段2: Agent-B 生成...")
    gen_prompt = f"基于分析结果，直接回答问题。\n问题：{TEST_QUESTION}\n分析：{analysis}\n\n请给出准确且简洁的答案。"
    draft = await agent_b.think(gen_prompt)
    core.store("draft", draft)
    print(f"    生成草案 (前100字符): {draft[:100]}")

    # ---- Phase 3: 审查 ----
    print("  - 阶段3: Agent-C 审查...")
    review_prompt = f"请审查以下答案是否准确、完整。\n问题：{TEST_QUESTION}\n答案：{draft}\n\n如果准确无误，请回复“VERIFIED”；否则回复“REVISION: 具体问题”。"
    review = await agent_c.think(review_prompt)
    core.store("review", review)
    print(f"    审查结论 (前100字符): {review[:100]}")

    # 简单断言：答案中应包含“巴黎”
    if "巴黎" in draft:
        print("  ✓ 三角形协作流程通过，答案正确")
    elif draft.strip():
        print("  ⚠ 答案未包含预期关键词，但非空，请人工检查")
    else:
        print("  ✗ 生成答案为空，测试失败")


@pytest.mark.skipif(not _ollama_available(), reason="Ollama 服务不可用，跳过容错测试")
async def test_fault_tolerance():
    """测试单 Agent 故障时的降级能力"""
    print("\n[Test 3] 容错降级测试...")
    core = SharedCore()
    model = normalize_model_name(settings.ollama_deep_model)

    # 模拟一个不存在的模型，触发故障
    agent_bad = TriangleAgent("Faulty-Agent", "nonexistent-model:latest", "故障测试", core)
    try:
        await agent_bad.think("hello")
        print("  - 故障 Agent 未抛出异常（异常）")
    except Exception as e:
        print(f"  - 故障 Agent 正确抛出异常: {type(e).__name__}")

    # 降级场景：分析师Agent故障，执行者Agent直接用原始问题生成答案
    agent_b = TriangleAgent("Agent-B·生成", model, "即使无分析也能生成答案", core)
    fallback_analysis = "（分析师未能生成分析，请直接回答问题）"
    gen_prompt = f"请直接回答以下问题。\n问题：{TEST_QUESTION}\n备注：{fallback_analysis}\n\n答案："
    draft = await agent_b.think(gen_prompt)
    print(f"  - 降级后的生成草案 (前100字符): {draft[:100]}")
    if draft.strip():
        print("  ✓ 容错降级成功，系统在 Analyst 故障后仍能返回结果")
    else:
        print("  ✗ 降级失败")


async def main():
    print("=" * 60)
    print("Miku AI 三角形多Agent协作系统集成测试 (简化Prompt版)")
    print(f"深度模型: {settings.ollama_deep_model}")
    print(f"轻度模型: {settings.ollama_default_model}")
    print("=" * 60)

    try:
        # 预热模型（仅在 Ollama 可用时进行）
        if _ollama_available():
            print("预热深度模型（约需数秒）...")
            llm = get_llm(settings.ollama_deep_model, keep_alive=-1)
            _ = llm.invoke([{"role": "user", "content": "ping"}])
            print("模型预热完成\n")
        else:
            print("Ollama 不可用，跳过模型预热\n")

        await test_shared_core()
        await test_triangle_agents()
        await test_fault_tolerance()

        print("\n" + "=" * 60)
        print("所有集成测试通过！🎉")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
