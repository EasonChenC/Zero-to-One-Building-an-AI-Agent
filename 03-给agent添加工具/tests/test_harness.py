"""第三章离线测试：模拟 AIMessage，不发送真实模型请求。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from harness.config import ConfigurationError, Settings
from harness.loop import WorkspaceHarness
from harness.model import ModelError, message_to_text
from harness.prompt_loader import PromptLoadError, load_system_prompt
from harness.state import TaskResult
from harness.tools import ToolExecutor, ToolPolicy, ToolRegistry, build_file_tools
from main import run_cli


def call(name: str, args: dict, call_id: str = "call-1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class FakeModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, system_prompt):
        self.calls.append(tuple(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def make_harness(model, root: Path, *, allowed=None, confirm=None, max_turns=8, max_tools=20):
    tools = build_file_tools(root)
    ask_user = StructuredTool.from_function(lambda question: question, name="ask_user", description="向用户提问")
    registry = ToolRegistry([ask_user, *tools])
    policy = ToolPolicy(
        allowed_tools=frozenset(allowed or registry.names()),
        require_confirmation=frozenset(confirm or set()),
        max_tool_calls=max_tools,
    )
    return WorkspaceHarness(model, ToolExecutor(registry, policy), max_turns=max_turns)


class LoopTests(unittest.TestCase):
    def test_direct_answer_uses_message_content(self):
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel(AIMessage(content="直接回答"))
            result = make_harness(model, Path(d)).run_task("任务", output_fn=lambda _: None)
        self.assertEqual((result.status, result.answer, result.turns), ("completed", "直接回答", 1))
        self.assertIsInstance(model.calls[0][0], HumanMessage)

    def test_tool_result_is_returned_to_model(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.py").write_text("print('ok')", encoding="utf-8")
            model = FakeModel(
                AIMessage(content="", tool_calls=[call("search_files", {"pattern": "*.py"})]),
                AIMessage(content="找到了 main.py"),
            )
            result = make_harness(model, Path(d)).run_task("找 Python 文件", output_fn=lambda _: None)
        self.assertEqual(result.answer, "找到了 main.py")
        self.assertEqual(result.tool_calls, 1)
        self.assertIsInstance(model.calls[1][-1], ToolMessage)
        self.assertIn("main.py", model.calls[1][-1].content)

    def test_ask_user_is_a_tool(self):
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel(
                AIMessage(content="", tool_calls=[call("ask_user", {"question": "项目路径？"})]),
                AIMessage(content="收到路径"),
            )
            inputs = iter(["/tmp/project"])
            result = make_harness(model, Path(d)).run_task("检查项目", input_fn=lambda _: next(inputs), output_fn=lambda _: None)
        self.assertEqual(result.answer, "收到路径")
        self.assertEqual(model.calls[1][-1].content, "/tmp/project")

    def test_write_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel(
                AIMessage(content="", tool_calls=[call("write_file", {"path": "note.txt", "content": "hello"})]),
                AIMessage(content="已处理"),
            )
            answers = iter(["no"])
            result = make_harness(model, Path(d), confirm={"write_file"}).run_task("写文件", input_fn=lambda _: next(answers), output_fn=lambda _: None)
            self.assertFalse(Path(d, "note.txt").exists())
        self.assertEqual(result.answer, "已处理")
        self.assertIn("未确认", model.calls[1][-1].content)

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tools = build_file_tools(Path(d))
            read_tool = next(tool for tool in tools if tool.name == "read_file")
            with self.assertRaises(Exception):
                read_tool.invoke({"path": "../outside.txt"})

    def test_unknown_tool_is_not_executed(self):
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel(
                AIMessage(content="", tool_calls=[call("delete_everything", {})]),
                AIMessage(content="工具不可用"),
            )
            result = make_harness(model, Path(d)).run_task("删除", output_fn=lambda _: None)
        self.assertEqual(result.status, "completed")
        self.assertIn("未授权", model.calls[1][-1].content)

    def test_tool_call_limit(self):
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel(AIMessage(content="", tool_calls=[call("search_files", {})]))
            result = make_harness(model, Path(d), max_tools=0).run_task("搜索", output_fn=lambda _: None)
        self.assertEqual(result.status, "max_tool_calls")

    def test_cancel_from_ask_user(self):
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel(AIMessage(content="", tool_calls=[call("ask_user", {"question": "需要什么？"})]))
            result = make_harness(model, Path(d)).run_task("任务", input_fn=lambda _: "/cancel", output_fn=lambda _: None)
        self.assertEqual(result.status, "cancelled")

    def test_cli_exits_from_nested_tool_question(self):
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel(AIMessage(content="", tool_calls=[call("ask_user", {"question": "环境？"})]))
            harness = make_harness(model, Path(d))
            inputs = iter(["任务", "/exit"])
            code = run_cli(harness, input_fn=lambda _: next(inputs), output_fn=lambda _: None)
        self.assertEqual(code, 0)


class ToolTests(unittest.TestCase):
    def test_registry_rejects_duplicate_names(self):
        first = StructuredTool.from_function(lambda: "1", name="same", description="")
        second = StructuredTool.from_function(lambda: "2", name="same", description="")
        registry = ToolRegistry([first])
        with self.assertRaises(ValueError):
            registry.register(second)

    def test_output_is_truncated(self):
        tool = StructuredTool.from_function(lambda: "x" * 20, name="long", description="")
        executor = ToolExecutor(ToolRegistry([tool]), ToolPolicy(allowed_tools=frozenset({"long"}), max_output_chars=10))
        result = executor.execute(call("long", {}))
        self.assertIn("截断", result.message.content)


class ConfigAndPromptTests(unittest.TestCase):
    def test_model_id_alias(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".env"
            path.write_text("ANTHROPIC_API_KEY=test\nMODEL_ID=claude-relay\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(path)
        self.assertEqual(settings.model, "claude-relay")

    def test_prompt_uses_native_tool_protocol(self):
        prompt = load_system_prompt()
        self.assertIn("tool_calls", prompt)
        self.assertNotIn('每次回复只返回一个 JSON 对象', prompt)

    def test_message_to_text(self):
        self.assertEqual(message_to_text(" hello "), "hello")
        self.assertEqual(message_to_text([{"type": "text", "text": "hi"}]), "hi")


if __name__ == "__main__":
    unittest.main()
