"""加载版本化的提示词资源；提示词正文不写在 Python 源码中。"""

from __future__ import annotations

from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "workspace-assistant.md"
MAX_PROMPT_BYTES = 128 * 1024


class PromptLoadError(RuntimeError):
    """提示词资源不存在、过大或无法读取。"""


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    """读取固定项目资源，不执行文件内容，也不做变量展开。"""
    try:
        if not path.is_file():
            raise PromptLoadError(f"提示词文件不存在：{path}")
        if path.stat().st_size > MAX_PROMPT_BYTES:
            raise PromptLoadError("提示词文件过大，拒绝加载。")
        prompt = path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise PromptLoadError(f"无法读取提示词文件：{path}") from exc
    if not prompt:
        raise PromptLoadError(f"提示词文件为空：{path}")
    return prompt
