"""任务状态"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal


ActionKind = Literal["ask_user", "final"]
TaskStatus = Literal["running", "completed", "cancelled", "max_turns", "error"]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ModelAction:
    kind: ActionKind
    message: str

    def to_json(self) -> str:
        return json.dumps({"action": self.kind, "message": self.message}, ensure_ascii=False)


@dataclass
class TaskState:
    goal: str
    messages: list[Message] = field(default_factory=list)
    turns: int = 0
    status: TaskStatus = "running"
    answer: str = ""
    stop_reason: str = ""
    exit_requested: bool = False

    @classmethod
    def start(cls, goal: str) -> TaskState:
        return cls(goal=goal, messages=[Message("user", goal)])


@dataclass(frozen=True)
class TaskResult:
    status: TaskStatus
    answer: str
    turns: int
    stop_reason: str
    exit_requested: bool = False
