"""终端入口：创建文件工具、注册表和支持工具调用的 Harness。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from langchain_core.tools import StructuredTool

from harness.config import ConfigurationError, Settings
from harness.loop import WorkspaceHarness
from harness.model import ClaudeClient
from harness.tools import ToolExecutor, ToolPolicy, ToolRegistry, build_file_tools


PROJECT_DIR = Path(__file__).resolve().parent


def build_harness(settings: Settings) -> WorkspaceHarness:
    file_tools = build_file_tools(PROJECT_DIR)

    def ask_user(question: str) -> str:
        """当缺少完成任务所需的信息时向用户提问。由 Harness 接管。"""
        return question

    ask_user = StructuredTool.from_function(
        ask_user,
        name="ask_user",
        description="当缺少完成任务所需的信息时向用户提问。由 Harness 接管并等待用户回答。",
    )
    registry = ToolRegistry([ask_user, *file_tools])
    policy = ToolPolicy(allowed_tools=registry.names(), require_confirmation=frozenset({"write_file"}))
    executor = ToolExecutor(registry, policy)
    return WorkspaceHarness(ClaudeClient(settings, tools=registry.all()), executor, max_turns=settings.max_turns)


def run_cli(harness: WorkspaceHarness, *, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> int:
    output_fn("工作空间任务助手已启动。当前可读取、搜索和写入本章工作区内的文本文件。")
    output_fn("/exit 退出程序；任务内 /cancel 取消。每个新任务使用独立的历史记录。")
    while True:
        try:
            goal = input_fn("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\n已退出。")
            return 0
        if goal.lower() in ("/exit", "/quit"):
            output_fn("已退出。")
            return 0
        if not goal:
            continue
        if goal.lower() == "/cancel":
            output_fn("当前没有正在执行的任务，请输入任务或 /exit。")
            continue
        result = harness.run_task(goal, input_fn=input_fn, output_fn=output_fn)
        output_fn(f"\n助手：{result.answer}")
        output_fn(f"[状态：{result.status} | 模型请求：{result.turns} 次 | 工具调用：{result.tool_calls} 次]")
        if result.exit_requested:
            output_fn("已退出。")
            return 0


def main() -> int:
    try:
        settings = Settings.from_env()
    except (ConfigurationError, OSError, UnicodeError) as exc:
        print(f"配置加载失败：{exc}")
        print("请填写本章目录下的 .env；配置字段可参见 .env.example。")
        return 1
    return run_cli(build_harness(settings))


if __name__ == "__main__":
    sys.exit(main())
