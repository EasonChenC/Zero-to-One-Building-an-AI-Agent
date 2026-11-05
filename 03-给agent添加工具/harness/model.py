"""基于 LangChain AIMessage 的 Claude 模型适配器。"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from .config import Settings


class ModelError(RuntimeError):
    """模型通信或响应格式不符合要求。"""


class ModelClient(Protocol):
    """任务循环只依赖这个接口，便于替换真实模型和测试模型。"""

    def generate(self, messages: Sequence[BaseMessage], system_prompt: str) -> AIMessage:
        ...


def message_to_text(content: Any) -> str:
    """把 AIMessage 的字符串或内容块转换为可展示文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text").strip()
    return ""


class ClaudeClient:
    """使用 langchain-anthropic 调用 Claude，并原样返回 AIMessage。"""

    def __init__(self, settings: Settings, tools: Sequence[Any] = ()):
        self.settings = settings
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ModelError("缺少 LangChain 依赖，请运行：uv sync") from exc

        kwargs: dict[str, Any] = {
            "model_name": settings.model,
            "max_tokens_to_sample": settings.max_tokens,
            "timeout": settings.timeout_seconds,
            "max_retries": 0,
            "api_key": settings.api_key,
        }
        base_url = settings.endpoint
        for suffix in ("/v1/messages", "/messages"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        kwargs["base_url"] = base_url
        model = ChatAnthropic(**kwargs)
        self._model = model.bind_tools(list(tools)) if tools else model

    def generate(self, messages: Sequence[BaseMessage], system_prompt: str) -> AIMessage:
        try:
            response = self._model.invoke([SystemMessage(content=system_prompt), *messages])
            if not isinstance(response, AIMessage):
                raise ModelError("模型未返回 AIMessage。")
            return response
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"LangChain Claude 请求失败：{type(exc).__name__}") from exc
