"""Audit logging for traceability."""

from __future__ import annotations

import json
from pathlib import Path

from agent_harness.core.events import HarnessEvent


class AuditLogger:
    def __init__(self, directory: str | Path = "storage/audit") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, event: HarnessEvent) -> None:
        path = self.directory / f"{event.session_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def read_session(self, session_id: str) -> list[dict]:
        path = self.directory / f"{session_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

