"""HTTP tools."""

from __future__ import annotations

import httpx

from agent_harness.core.types import ToolResult
from agent_harness.tools.schemas import ToolDefinition


def http_get_definition() -> ToolDefinition:
    return ToolDefinition(
        name="http.get",
        description="Fetch a URL with HTTP GET.",
        input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        permission_resource="http.request",
        risk_level="medium",
    )


async def http_get_handler(arguments: dict) -> ToolResult:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(arguments["url"])
        return ToolResult(call_id=arguments.get("call_id", ""), name="http.get", ok=True, output=response.text[:5000])

