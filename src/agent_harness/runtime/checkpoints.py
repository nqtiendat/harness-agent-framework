"""Checkpoint persistence for resumable workflows."""

from __future__ import annotations

from pathlib import Path

from agent_harness.core.types import AgentState


class CheckpointStore:
    def __init__(self, directory: str | Path = "storage/workspaces") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState) -> Path:
        path = self.directory / f"{state.session_id}.checkpoint.json"
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, session_id: str) -> AgentState:
        path = self.directory / f"{session_id}.checkpoint.json"
        return AgentState.model_validate_json(path.read_text(encoding="utf-8"))

