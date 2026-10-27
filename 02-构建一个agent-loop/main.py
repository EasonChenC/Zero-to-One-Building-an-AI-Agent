"""终端入口。换成 Web/API 时，可复用 harness 中的任务循环。"""

from __future__ import annotations

import sys
from typing import Callable

from harness.config import ConfigurationError, Settings
from harness.loop import WorkspaceHarness
from harness.model import ClaudeClient


def run_cli(
    harness: WorkspaceHarness, *, input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    output_fn("工作空间任务助手已启动。当前仅分析对话内容，不会读取或修改本地文件。")
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
        output_fn(f"[状态：{result.status} | 模型请求：{result.turns} 次]")
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
    harness = WorkspaceHarness(ClaudeClient(settings), max_turns=settings.max_turns)
    return run_cli(harness)


if __name__ == "__main__":
    sys.exit(main())
