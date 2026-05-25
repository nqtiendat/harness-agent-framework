"""Sandbox boundary for tool calls."""

from __future__ import annotations

from pathlib import Path

from agent_harness.tools.tool_sandbox import ToolSandbox


class SandboxPolicy(ToolSandbox):
    """Backward-compatible alias for the richer ToolSandbox."""

    def __init__(self, workspace: str | Path = ".") -> None:
        super().__init__(workspace=workspace)

    def assert_within_workspace(self, path: str | Path) -> Path:
        return self.assert_path_within_workspace(path)
