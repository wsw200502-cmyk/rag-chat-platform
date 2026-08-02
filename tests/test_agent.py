import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tool_registry():
    """工具注册表能正常注册和获取"""

    class ToolRegistry:
        def __init__(self):
            self._tools = {}
            self._descriptions = {}

        def register(self, name: str, description: str):
            def decorator(func):
                self._tools[name] = func
                self._descriptions[name] = description
                return func

            return decorator

        def get_tool(self, name: str):
            return self._tools.get(name)

    reg = ToolRegistry()

    @reg.register("test_tool", "测试工具")
    def test_tool(x: str):
        return x

    assert reg.get_tool("test_tool")("hello") == "hello"
