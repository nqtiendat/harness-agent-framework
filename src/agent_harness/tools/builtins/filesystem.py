"""Read-only filesystem tools."""

from __future__ import annotations

from agent_harness.core.types import ToolResult
from agent_harness.tools.sandbox import SandboxPolicy
from agent_harness.tools.schemas import ToolDefinition


def read_file_definition() -> ToolDefinition:
    return ToolDefinition(
        name="filesystem.read_file",
        description="Read a UTF-8 text file within the workspace.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        permission_resource="filesystem.read",
        risk_level="low",
    )


def make_read_file_handler(sandbox: SandboxPolicy):
    async def handler(arguments: dict) -> ToolResult:
        path = sandbox.assert_within_workspace(arguments["path"])
        return ToolResult(call_id=arguments.get("call_id", ""), name="filesystem.read_file", ok=True, output=path.read_text(encoding="utf-8"))

    return handler

