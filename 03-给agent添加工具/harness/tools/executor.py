"""统一执行工具并生成 ToolMessage。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import ToolMessage

from .policy import ToolPolicy
from .registry import ToolRegistry


@dataclass(frozen=True)
class ToolExecutionResult:
    message: ToolMessage
    cancelled: bool = False
    exit_requested: bool = False


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, policy: ToolPolicy):
        self.registry = registry
        self.policy = policy

    def execute(self, tool_call: dict[str, Any], *, input_fn: Callable[[str], str] | None = None, confirm_fn: Callable[[str], str] | None = None) -> ToolExecutionResult:
        name = tool_call.get("name")
        call_id = str(tool_call.get("id", ""))
        args = tool_call.get("args", {})
        if not isinstance(name, str) or not name:
            return self._error(call_id, "工具调用缺少名称。")
        if not isinstance(args, dict):
            return self._error(call_id, "工具参数必须是对象。")
        if not self.policy.allows(name):
            return self._error(call_id, f"工具未授权：{name}")
        tool = self.registry.get(name)
        if tool is None:
            return self._error(call_id, f"工具不存在：{name}")

        if name == "ask_user":
            return self._ask_user(call_id, args, input_fn)
        if name in self.policy.require_confirmation:
            if confirm_fn is None:
                return self._error(call_id, "该工具需要用户确认，但当前没有确认入口。")
            answer = confirm_fn(f"工具 {name} 将修改工作区，确认执行吗？输入 yes 确认：").strip().lower()
            if answer in ("/cancel", "/exit", "/quit"):
                return ToolExecutionResult(ToolMessage(content="用户取消了工具执行。", tool_call_id=call_id, status="error"), True, answer in ("/exit", "/quit"))
            if answer not in ("y", "yes"):
                return self._error(call_id, "用户未确认，工具没有执行。")
        try:
            output = tool.invoke(args)
            return ToolExecutionResult(ToolMessage(content=self._limit(str(output)), tool_call_id=call_id, name=name))
        except Exception as exc:
            return self._error(call_id, f"工具执行失败：{type(exc).__name__}。")

    def _ask_user(self, call_id: str, args: dict[str, Any], input_fn: Callable[[str], str] | None) -> ToolExecutionResult:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip() or input_fn is None:
            return self._error(call_id, "ask_user 缺少 question 参数或当前没有用户输入入口。")
        try:
            answer = input_fn(f"助手：{question}\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            return ToolExecutionResult(ToolMessage(content="用户中断了回答。", tool_call_id=call_id, status="error"), True)
        if answer.lower() in ("/cancel", "/exit", "/quit"):
            return ToolExecutionResult(ToolMessage(content="用户取消了任务。", tool_call_id=call_id, status="error"), True, answer.lower() in ("/exit", "/quit"))
        if not answer:
            return self._error(call_id, "用户没有提供有效回答。")
        return ToolExecutionResult(ToolMessage(content=answer, tool_call_id=call_id, name="ask_user"))

    def _error(self, call_id: str, content: str) -> ToolExecutionResult:
        return ToolExecutionResult(ToolMessage(content=content, tool_call_id=call_id, status="error"))

    def _limit(self, content: str) -> str:
        if len(content) <= self.policy.max_output_chars:
            return content
        return content[: self.policy.max_output_chars] + "\n[工具结果过长，后续内容已截断]"
