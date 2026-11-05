"""第三章任务状态：直接保存 LangChain 消息历史。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage


TaskStatus = Literal["running", "completed", "cancelled", "max_turns", "max_tool_calls", "error"]


@dataclass
class TaskState:
    goal: str
    messages: list[BaseMessage] = field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    status: TaskStatus = "running"
    answer: str = ""
    stop_reason: str = ""
    exit_requested: bool = False

    @classmethod
    def start(cls, goal: str) -> "TaskState":
        return cls(goal=goal, messages=[HumanMessage(content=goal)])


@dataclass(frozen=True)
class TaskResult:
    status: TaskStatus
    answer: str
    turns: int
    stop_reason: str
    tool_calls: int = 0
    exit_requested: bool = False
