"""工具注册表：模型只能看到和调用预先注册的工具。"""

from __future__ import annotations

from typing import Any, Iterable


class ToolRegistry:
    def __init__(self, tools: Iterable[Any] = ()):
        self._tools: dict[str, Any] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Any) -> None:
        name = getattr(tool, "name", "")
        if not name:
            raise ValueError("工具必须有非空 name。")
        if name in self._tools:
            raise ValueError(f"工具已存在：{name}")
        self._tools[name] = tool

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def all(self) -> list[Any]:
        return list(self._tools.values())

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)
