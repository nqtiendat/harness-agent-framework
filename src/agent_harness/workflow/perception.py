"""Agent-Computer Interface perception pipeline."""

from __future__ import annotations

from pathlib import Path

from agent_harness.core.types import Observation
from agent_harness.workflow.environment_perception import EnvironmentPerceptionEngine, PerceptionConfig


class PerceptionPipeline:
    def __init__(self, raw_directory: str | Path = "storage/workspaces/raw", config: PerceptionConfig | None = None) -> None:
        self.raw_directory = Path(raw_directory)
        self.raw_directory.mkdir(parents=True, exist_ok=True)
        self.engine = EnvironmentPerceptionEngine(self.raw_directory, config=config)

    def perceive(self, raw: str | dict | list, source: str = "environment", content_type: str | None = None) -> Observation:
        return self.engine.perceive(raw, source=source, content_type=content_type)
