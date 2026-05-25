"""Shell tool placeholder."""

from __future__ import annotations

from agent_harness.core.exceptions import PermissionDeniedError
from agent_harness.tools.schemas import ToolDefinition


def shell_command_definition() -> ToolDefinition:
    return ToolDefinition(
        name="shell.command",
        description="Run a shell command. Disabled by default and intended for sandboxed deployments.",
        input_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        permission_resource="shell.command",
        risk_level="high",
    )


async def disabled_shell_handler(arguments: dict):
    raise PermissionDeniedError("Shell execution is disabled by the default builtin handler.")

