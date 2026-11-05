"""支持 AIMessage 工具调用的工作空间任务循环。"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage

from .model import ModelClient, ModelError, message_to_text
from .prompt_loader import load_system_prompt
from .state import TaskResult, TaskState
from .tools.executor import ToolExecutor


class WorkspaceHarness:
    def __init__(self, model: ModelClient, executor: ToolExecutor, *, max_turns: int = 8, max_tool_calls: int | None = None, system_prompt: str | None = None):
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0。")
        self.model = model
        self.executor = executor
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls or executor.policy.max_tool_calls
        self.system_prompt = system_prompt if system_prompt is not None else load_system_prompt()

    def run_task(self, goal: str, *, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> TaskResult:
        state = TaskState.start(goal.strip())
        if not state.goal:
            return TaskResult("error", "任务不能为空。", 0, "empty_goal")

        while state.status == "running":
            if state.turns >= self.max_turns:
                state.status, state.stop_reason = "max_turns", "达到模型调用轮数上限"
                state.answer = f"达到最大轮数 {self.max_turns}，本次任务已停止。"
                break
            state.turns += 1
            try:
                output_fn(f"[第 {state.turns}/{self.max_turns} 轮] 正在请求模型……")
                response = self.model.generate(tuple(state.messages), self.system_prompt)
            except ModelError as exc:
                state.status, state.stop_reason, state.answer = "error", "model_error", str(exc)
                break
            except KeyboardInterrupt:
                state.status, state.stop_reason, state.answer = "cancelled", "interrupted", "任务已取消。"
                break

            if not isinstance(response, AIMessage):
                state.status, state.stop_reason, state.answer = "error", "invalid_model_response", "模型未返回 AIMessage。"
                break
            state.messages.append(response)
            tool_calls = response.tool_calls or []
            if not tool_calls:
                answer = message_to_text(response.content)
                if not answer:
                    state.status, state.stop_reason, state.answer = "error", "empty_model_response", "模型没有返回文本或工具调用。"
                else:
                    state.status, state.stop_reason, state.answer = "completed", "final_answer", answer
                break

            if len(tool_calls) > self.executor.policy.max_calls_per_turn:
                state.status, state.stop_reason, state.answer = "max_tool_calls", "超过单轮工具调用上限", "本轮工具调用数量超过上限，任务已停止。"
                break
            for tool_call in tool_calls:
                if state.tool_calls >= self.max_tool_calls:
                    state.status, state.stop_reason, state.answer = "max_tool_calls", "达到工具调用总次数上限", f"达到工具调用上限 {self.max_tool_calls}，本次任务已停止。"
                    break
                state.tool_calls += 1
                result = self.executor.execute(tool_call, input_fn=input_fn, confirm_fn=input_fn)
                state.messages.append(result.message)
                if result.cancelled:
                    state.status, state.stop_reason, state.answer = "cancelled", "user_cancelled", "任务已取消。"
                    state.exit_requested = result.exit_requested
                    break
            if state.status != "running":
                break

        return TaskResult(state.status, state.answer, state.turns, state.stop_reason, state.tool_calls, state.exit_requested)
