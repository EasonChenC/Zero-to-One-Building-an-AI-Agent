"""从项目目录的 .env 和系统环境变量加载配置，不依赖第三方库。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_DIR = Path(__file__).resolve().parent.parent


class ConfigurationError(ValueError):
    """配置缺失或格式不正确。"""


def read_env_file(path: Path) -> dict[str, str]:
    """读取 KEY=VALUE；支持整行注释、成对引号，不执行变量或命令展开。"""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f".env 第 {number} 行应为 KEY=VALUE。")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key.isidentifier():
            raise ConfigurationError(f".env 第 {number} 行的变量名不合法。")
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ConfigurationError(f".env 第 {number} 行的引号没有配对。")
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str
    endpoint: str
    max_turns: int = 8
    max_tokens: int = 2048
    timeout_seconds: int = 120
    api_version: str = "2023-06-01"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        # 不修改 os.environ；系统变量优先于 .env，方便以后部署为服务。
        values = read_env_file(env_file if env_file is not None else PROJECT_DIR / ".env")
        values.update(os.environ)

        def required(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise ConfigurationError(f"请在 .env 中填写 {name}。")
            return value

        def positive_int(name: str, default: int) -> int:
            try:
                value = int(values.get(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} 必须是正整数。") from exc
            if value <= 0:
                raise ConfigurationError(f"{name} 必须是正整数。")
            return value

        api_key = required("ANTHROPIC_API_KEY")
        # 中转站配置常用 MODEL_ID；标准 Anthropic 配置使用 ANTHROPIC_MODEL。
        # 优先采用项目标准名称，同时兼容中转站提供的变量名。
        model = values.get("ANTHROPIC_MODEL", "").strip() or values.get("MODEL_ID", "").strip()
        if not model:
            raise ConfigurationError("请在 .env 中填写 ANTHROPIC_MODEL（或中转站的 MODEL_ID）。")
        endpoint = values.get("ANTHROPIC_API_URL", "").strip()
        if not endpoint:
            base = values.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").strip().rstrip("/")
            if base.endswith("/messages"):
                endpoint = base
            elif base.endswith("/v1"):
                endpoint = base + "/messages"
            else:
                endpoint = base + "/v1/messages"

        # 密钥只放在请求头。禁止明文 HTTP、URL 用户信息及查询参数。
        try:
            url = urlsplit(endpoint)
            valid_url = (
                url.scheme == "https" and bool(url.hostname) and url.port != 0
                and not url.username and not url.password and not url.query and not url.fragment
            )
        except ValueError as exc:
            raise ConfigurationError("Claude API 地址格式不正确。") from exc
        if not valid_url:
            raise ConfigurationError("API 地址必须是 HTTPS 地址，且不能包含账号、密码、查询参数或片段。")

        api_version = values.get("ANTHROPIC_VERSION", "2023-06-01").strip()
        if not api_version or any("\r" in value or "\n" in value for value in (api_key, api_version)):
            raise ConfigurationError("密钥或 API 版本格式不正确。")

        return cls(
            api_key=api_key, model=model, endpoint=endpoint, api_version=api_version,
            max_turns=positive_int("HARNESS_MAX_TURNS", 8),
            max_tokens=positive_int("ANTHROPIC_MAX_TOKENS", 2048),
            timeout_seconds=positive_int("ANTHROPIC_TIMEOUT_SECONDS", 120),
        )
