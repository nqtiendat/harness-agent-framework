"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_HARNESS_", env_file=".env", extra="ignore")

    environment: str = "dev"
    storage_dir: Path = Path("storage")
    default_model: str = "mock"
    max_iterations: int = 8
    max_tool_calls: int = 16
    max_seconds: int = 120
    max_model_calls: int = 16
    max_estimated_tokens: int = 64_000
    extra: dict[str, Any] = Field(default_factory=dict)


def load_yaml(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {candidate}")
    return data


def load_settings(path: str | Path = "configs/default.yaml") -> HarnessSettings:
    data = load_yaml(path)
    return HarnessSettings(**data)

