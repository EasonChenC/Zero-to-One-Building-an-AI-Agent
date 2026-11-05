"""本章内置文件工具，所有路径都必须位于工作区目录内。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool


MAX_READ_CHARS = 20_000
MAX_SEARCH_RESULTS = 100


class WorkspaceFileTools:
    def __init__(self, workspace_root: Path):
        self.root = workspace_root.resolve()
        if not self.root.is_dir():
            raise ValueError(f"工作区目录不存在：{self.root}")

    def _safe_path(self, path: str) -> Path:
        candidate = Path(path)
        resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("路径必须位于工作区目录内。") from exc
        return resolved

    def read_file(self, path: str) -> str:
        """读取工作区内的 UTF-8 文本文件。"""
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        text = target.read_text(encoding="utf-8")
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + "\n[文件内容过长，后续内容已截断]"
        return text

    def write_file(self, path: str, content: str) -> str:
        """写入工作区内的 UTF-8 文件。"""
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入文件：{target.relative_to(self.root)}"

    def search_files(self, pattern: str = "*", max_results: int = 50) -> str:
        """在工作区内按文件名模式搜索文件，不读取文件内容。"""
        if not 1 <= max_results <= MAX_SEARCH_RESULTS:
            raise ValueError(f"max_results 必须在 1 到 {MAX_SEARCH_RESULTS} 之间。")
        matches = [path.relative_to(self.root).as_posix() for path in self.root.glob(pattern) if path.is_file()][:max_results]
        return "\n".join(matches) if matches else "未找到匹配文件。"

    def tools(self) -> list[Any]:
        return [
            StructuredTool.from_function(self.read_file, name="read_file", description="读取工作区内的 UTF-8 文本文件。只能读取工作区目录内的路径。"),
            StructuredTool.from_function(self.write_file, name="write_file", description="将文本写入工作区内的 UTF-8 文件。执行前需要用户确认。"),
            StructuredTool.from_function(self.search_files, name="search_files", description="在工作区内按文件名模式搜索文件，不读取文件内容。"),
        ]


def build_file_tools(workspace_root: Path) -> list[Any]:
    return WorkspaceFileTools(workspace_root).tools()
