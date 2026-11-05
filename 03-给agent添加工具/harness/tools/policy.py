"""工具调用权限和资源限制。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    require_confirmation: frozenset[str] = field(default_factory=frozenset)
    max_tool_calls: int = 20
    max_calls_per_turn: int = 3
    max_output_chars: int = 8_000

    def allows(self, name: str) -> bool:
        return name in self.allowed_tools
