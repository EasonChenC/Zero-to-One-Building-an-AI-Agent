"""LangChain 模型接口与 Claude 适配器。"""

from __future__ import annotations

import json
from typing import Any, Protocol, Sequence

from .config import Settings
from .state import Message, ModelAction


class ModelError(RuntimeError):
    """模型通信、响应或动作格式不符合要求。"""


class ModelClient(Protocol):
    """Loop 只依赖这个接口，可替换模型或测试客户端。"""

    def generate(self, messages: Sequence[Message], system_prompt: str) -> ModelAction:
        ...


def parse_action(text: str) -> ModelAction:
    """解析模型返回的 JSON 动作。"""
    text = text.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip() in ("```", "```json") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1])
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelError("模型未返回合法 JSON 动作；本次任务已停止。") from exc
    if not isinstance(payload, dict):
        raise ModelError("模型动作必须是 JSON 对象。")
    kind, message = payload.get("action"), payload.get("message")
    if kind not in ("ask_user", "final"):
        raise ModelError("当前版本只支持 ask_user 和 final 动作。")
    if not isinstance(message, str) or not message.strip():
        raise ModelError("模型动作缺少非空的 message 字段。")
    return ModelAction(kind=kind, message=message.strip())


def _message_content(response: Any) -> str:
    """兼容 AIMessage 的字符串和内容块列表。"""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
    raise ModelError("LangChain 返回了无法识别的消息内容。")


class ClaudeClient:
    """使用 langchain-anthropic 调用 Claude Messages API。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ModelError("缺少 LangChain 依赖，请运行：pip install -e .") from exc

        kwargs: dict[str, Any] = {
            "model_name": settings.model,
            "max_tokens_to_sample": settings.max_tokens,
            "timeout": settings.timeout_seconds,
            "max_retries": 0,
            "api_key": settings.api_key,
        }
        # ChatAnthropic 接收基础地址，不接收完整 /v1/messages 地址。
        base_url = settings.endpoint
        for suffix in ("/v1/messages", "/messages"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        kwargs["base_url"] = base_url
        self._model = ChatAnthropic(**kwargs)

    def generate(self, messages: Sequence[Message], system_prompt: str) -> ModelAction:
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            langchain_messages = [SystemMessage(content=system_prompt)]
            for message in messages:
                if message.role == "user":
                    langchain_messages.append(HumanMessage(content=message.content))
                elif message.role == "assistant":
                    langchain_messages.append(AIMessage(content=message.content))
                else:
                    raise ModelError(f"不支持的消息角色：{message.role}")
            response = self._model.invoke(langchain_messages)
            return parse_action(_message_content(response))
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"LangChain Claude 请求失败：{type(exc).__name__}") from exc
