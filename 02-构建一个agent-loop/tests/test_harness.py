"""离线验证：模拟 LangChain 模型，不发送真实请求。"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness.config import ConfigurationError, Settings
from harness.loop import WorkspaceHarness
from harness.model import ClaudeClient, ModelError, parse_action
from harness.prompt_loader import PromptLoadError, load_system_prompt
from harness.state import Message, ModelAction
from main import run_cli


class FakeModel:
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.calls = []

    def generate(self, messages, system_prompt):
        self.calls.append(tuple(messages))
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        return action


class LoopTests(unittest.TestCase):
    def run_task(self, model, goal="整理一份上线清单", answers=(), max_turns=8):
        user_input = Mock(side_effect=answers)
        result = WorkspaceHarness(model, max_turns=max_turns).run_task(
            goal, input_fn=user_input, output_fn=lambda _: None,
        )
        return result, user_input

    def test_direct_final(self):
        model = FakeModel(ModelAction("final", "清单"))
        result, user_input = self.run_task(model)
        self.assertEqual((result.status, result.answer, result.turns), ("completed", "清单", 1))
        user_input.assert_not_called()

    def test_question_and_answer_are_kept(self):
        model = FakeModel(ModelAction("ask_user", "部署在哪里？"), ModelAction("final", "Linux 上线清单"))
        result, _ = self.run_task(model, answers=["Linux"])
        self.assertEqual(result.turns, 2)
        self.assertEqual(len(model.calls[1]), 3)
        self.assertEqual(model.calls[1][1].role, "assistant")
        self.assertEqual(json.loads(model.calls[1][1].content)["action"], "ask_user")
        self.assertEqual(model.calls[1][2], Message("user", "Linux"))

    def test_empty_answer_is_reprompted_without_model_call(self):
        model = FakeModel(ModelAction("ask_user", "环境？"), ModelAction("final", "结果"))
        result, user_input = self.run_task(model, answers=["", "  ", "Linux"])
        self.assertEqual(user_input.call_count, 3)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(model.calls), 2)

    def test_cancel(self):
        result, _ = self.run_task(FakeModel(ModelAction("ask_user", "环境？")), answers=["/cancel"])
        self.assertEqual(result.status, "cancelled")
        self.assertFalse(result.exit_requested)

    def test_exit_during_task(self):
        result, _ = self.run_task(FakeModel(ModelAction("ask_user", "环境？")), answers=["/exit"])
        self.assertEqual(result.status, "cancelled")
        self.assertTrue(result.exit_requested)

    def test_eof_during_question(self):
        result, _ = self.run_task(FakeModel(ModelAction("ask_user", "环境？")), answers=[EOFError()])
        self.assertEqual(result.status, "cancelled")

    def test_last_question_does_not_collect_unusable_answer(self):
        result, user_input = self.run_task(FakeModel(ModelAction("ask_user", "环境？")), max_turns=1)
        self.assertEqual(result.status, "max_turns")
        self.assertEqual(result.turns, 1)
        user_input.assert_not_called()

    def test_model_failure_is_not_completion_and_counts_as_attempt(self):
        result, _ = self.run_task(FakeModel(ModelError("网络错误")))
        self.assertEqual((result.status, result.turns), ("error", 1))

    def test_interrupt_during_request(self):
        model = Mock()
        model.generate.side_effect = KeyboardInterrupt
        result, _ = self.run_task(model)
        self.assertEqual(result.status, "cancelled")

    def test_empty_goal_does_not_call_model(self):
        model = FakeModel()
        result, _ = self.run_task(model, goal=" ")
        self.assertEqual((result.status, result.turns), ("error", 0))
        self.assertEqual(model.calls, [])

    def test_invalid_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            WorkspaceHarness(FakeModel(), max_turns=0)

    def test_new_tasks_have_independent_history(self):
        model = FakeModel(ModelAction("final", "一"), ModelAction("final", "二"))
        harness = WorkspaceHarness(model)
        harness.run_task("任务一", output_fn=lambda _: None)
        harness.run_task("任务二", output_fn=lambda _: None)
        self.assertEqual(model.calls[1], (Message("user", "任务二"),))

    def test_outer_cli_accepts_two_tasks_then_exits(self):
        model = FakeModel(ModelAction("final", "一"), ModelAction("final", "二"))
        inputs = iter(["", "任务一", "任务二", "/exit"])
        code = run_cli(WorkspaceHarness(model), input_fn=lambda _: next(inputs), output_fn=lambda _: None)
        self.assertEqual(code, 0)
        self.assertEqual(len(model.calls), 2)

    def test_cli_exits_from_nested_question(self):
        inputs = iter(["任务", "/exit"])
        model = FakeModel(ModelAction("ask_user", "环境？"))
        self.assertEqual(run_cli(WorkspaceHarness(model), input_fn=lambda _: next(inputs), output_fn=lambda _: None), 0)


class ActionTests(unittest.TestCase):
    def test_valid_action(self):
        self.assertEqual(parse_action('{"action":"final","message":"答案"}'), ModelAction("final", "答案"))

    def test_whole_code_fence_is_accepted(self):
        text = "```json\n{\"action\":\"ask_user\",\"message\":\"环境？\"}\n```"
        self.assertEqual(parse_action(text).kind, "ask_user")

    def test_invalid_outputs_are_rejected(self):
        for text in (
            "普通文本", "[]", "null", '{"action":"tool_call","message":"命令"}',
            '{"action":"final","message":""}', '{"action":"final","message":9}',
            '前言 {"action":"final","message":"答案"}', '{"action":[],"message":"内容"}',
        ):
            with self.subTest(text=text), self.assertRaises(ModelError):
                parse_action(text)


class ConfigTests(unittest.TestCase):
    def settings(self, content, environ=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(content, encoding="utf-8")
            with patch.dict(os.environ, environ or {}, clear=True):
                return Settings.from_env(path)

    def test_env_loads_and_system_variables_override(self):
        settings = self.settings("ANTHROPIC_API_KEY='file-key'\nANTHROPIC_MODEL=claude-test\n", {"ANTHROPIC_API_KEY": "system-key"})
        self.assertEqual(settings.api_key, "system-key")
        self.assertEqual(settings.endpoint, "https://api.anthropic.com/v1/messages")
        self.assertNotIn("system-key", repr(settings))

    def test_model_id_alias_is_supported_for_relay_configs(self):
        settings = self.settings(
            "ANTHROPIC_API_KEY=test\nMODEL_ID=claude-relay\nANTHROPIC_BASE_URL=https://relay.example.com\n"
        )
        self.assertEqual(settings.model, "claude-relay")
        self.assertEqual(settings.endpoint, "https://relay.example.com/v1/messages")

    def test_endpoint_paths(self):
        for base in ("https://example.com", "https://example.com/v1", "https://example.com/v1/messages/"):
            settings = self.settings(f"ANTHROPIC_API_KEY=test\nANTHROPIC_MODEL=test\nANTHROPIC_BASE_URL={base}\n")
            self.assertEqual(settings.endpoint.rstrip("/"), "https://example.com/v1/messages")

    def test_full_endpoint_takes_precedence(self):
        settings = self.settings("ANTHROPIC_API_KEY=test\nANTHROPIC_MODEL=test\nANTHROPIC_API_URL=https://example.com/custom/messages\n")
        self.assertEqual(settings.endpoint, "https://example.com/custom/messages")

    def test_missing_credentials(self):
        with self.assertRaises(ConfigurationError):
            self.settings("ANTHROPIC_MODEL=test\n")

    def test_invalid_values(self):
        for extra in ("HARNESS_MAX_TURNS=0", "ANTHROPIC_MAX_TOKENS=wrong", "ANTHROPIC_API_URL=http://example.com/messages",
                      "ANTHROPIC_API_URL=https://user:pass@example.com/messages", "ANTHROPIC_API_URL=https://example.com/messages?key=secret"):
            with self.subTest(extra=extra), self.assertRaises(ConfigurationError):
                self.settings("ANTHROPIC_API_KEY=test\nANTHROPIC_MODEL=test\n" + extra)


class PromptTests(unittest.TestCase):
    def test_project_prompt_is_loaded_from_markdown(self):
        prompt = load_system_prompt()
        self.assertIn("工作空间任务助手", prompt)
        self.assertIn('"ask_user"', prompt)

    def test_missing_or_empty_prompt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.md"
            with self.assertRaises(PromptLoadError):
                load_system_prompt(path)
            path.write_text("  ", encoding="utf-8")
            with self.assertRaises(PromptLoadError):
                load_system_prompt(path)


@unittest.skipUnless(importlib.util.find_spec("langchain_anthropic"), "安装 langchain-anthropic 后运行 LangChain 适配器测试")
class ApiTests(unittest.TestCase):
    def client(self):
        return ClaudeClient(Settings(api_key="test-secret", model="claude-test", endpoint="https://example.com/v1/messages"))

    def response(self, action="final", message="答案", stop_reason="end_turn"):
        payload = {"stop_reason": stop_reason, "content": [{"type": "text", "text": json.dumps({"action": action, "message": message})}]}
        return io.BytesIO(json.dumps(payload).encode())

    def test_request_and_response(self):
        client = self.client()
        client._model = Mock()
        client._model.invoke.return_value = type("Response", (), {"content": json.dumps({"action": "final", "message": "答案"})})()
        result = client.generate([Message("user", "任务")], "系统规则")
        self.assertEqual(result.kind, "final")
        client._model.invoke.assert_called_once()
        sent = client._model.invoke.call_args.args[0]
        self.assertEqual(sent[0].type, "system")
        self.assertEqual(sent[1].type, "human")

    def test_content_blocks_are_supported(self):
        client = self.client()
        client._model = Mock()
        client._model.invoke.return_value = type(
            "Response", (), {"content": [{"type": "text", "text": '{"action":"final","message":"答案"}'}]}
        )()
        self.assertEqual(client.generate([Message("user", "任务")], "规则").message, "答案")

    def test_model_exception_is_wrapped_without_echoing_secret(self):
        client = self.client()
        client._model = Mock()
        client._model.invoke.side_effect = RuntimeError("test-secret-body")
        with self.assertRaisesRegex(ModelError, "LangChain Claude 请求失败") as caught:
            client.generate([Message("user", "任务")], "规则")
        self.assertNotIn("test-secret-body", str(caught.exception))

    def test_truncated_answer_is_rejected(self):
        client = self.client()
        client._model = Mock()
        client._model.invoke.return_value = type("Response", (), {"content": ""})()
        with self.assertRaises(ModelError):
            client.generate([Message("user", "任务")], "规则")

    def test_malformed_or_unsupported_responses(self):
        for content in ("普通文本", "[]", '{"action":"tool_call","message":"命令"}', ""):
            client = self.client()
            client._model = Mock()
            client._model.invoke.return_value = type("Response", (), {"content": content})()
            with self.subTest(content=content), self.assertRaises(ModelError):
                client.generate([Message("user", "任务")], "规则")


if __name__ == "__main__":
    unittest.main()
