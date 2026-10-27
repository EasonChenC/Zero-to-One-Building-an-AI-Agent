"""单个任务的 while 循环；不绑定终端输入，也不直接操作文件。"""

from __future__ import annotations

from typing import Callable

from .model import ModelClient, ModelError
from .prompt_loader import load_system_prompt
from .state import Message, TaskResult, TaskState


class WorkspaceHarness:
    def __init__(self, model: ModelClient, *, max_turns: int = 8, system_prompt: str | None = None):
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0。")
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt if system_prompt is not None else load_system_prompt()

    def run_task(
        self, goal: str, *, input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> TaskResult:
        state = TaskState.start(goal.strip())
        if not state.goal:
            return TaskResult("error", "任务不能为空。", 0, "empty_goal")

        while state.status == "running":
            if state.turns >= self.max_turns:
                state.status, state.stop_reason = "max_turns", "达到模型调用轮数上限"
                state.answer = f"达到最大轮数 {self.max_turns}，本次任务已停止，尚未获得最终回答。"
                break

            # 一轮就是一次模型请求；失败的请求也计数，不做隐式自动重试。
            state.turns += 1
            try:
                output_fn(f"[第 {state.turns}/{self.max_turns} 轮] 正在请求模型……")
                action = self.model.generate(tuple(state.messages), self.system_prompt)
            except ModelError as exc:
                state.status, state.stop_reason, state.answer = "error", "model_error", str(exc)
                break
            except KeyboardInterrupt:
                state.status, state.stop_reason, state.answer = "cancelled", "interrupted", "任务已取消。"
                break

            # 保留提问和最终回答，让状态完整；不是跨任务的长期记忆。
            state.messages.append(Message("assistant", action.to_json()))
            if action.kind == "final":
                state.status, state.stop_reason, state.answer = "completed", "final_answer", action.message
                break

            if action.kind != "ask_user":
                state.status, state.stop_reason = "error", "unsupported_action"
                state.answer = "收到尚未实现的动作，已停止；不会执行任意模型指令。"
                break

            output_fn(f"\n助手：{action.message}")
            if state.turns >= self.max_turns:
                # 无下一次调用额度时，不收集一段已经无法处理的用户回答。
                continue

            # 空输入只重新读取，不触发新的模型调用，也不丢失模型的问题。
            while True:
                try:
                    answer = input_fn("你（/cancel 取消任务，/exit 退出程序）：").strip()
                except (EOFError, KeyboardInterrupt):
                    state.status, state.stop_reason, state.answer = "cancelled", "input_interrupted", "任务已取消。"
                    break
                command = answer.lower()
                if command in ("/cancel", "/exit", "/quit"):
                    state.status, state.stop_reason, state.answer = "cancelled", "user_cancelled", "任务已取消。"
                    state.exit_requested = command in ("/exit", "/quit")
                    break
                if answer:
                    state.messages.append(Message("user", answer))
                    break
                output_fn("补充信息不能为空，请继续回答；或输入 /cancel。")

        return TaskResult(state.status, state.answer, state.turns, state.stop_reason, state.exit_requested)
