"""Tool schemas and definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.types import ToolResult


ToolHandler = Callable[[dict[str, Any]], Awaitable[ToolResult]]


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    permission_resource: str
    risk_level: str = "low"

