"""Tool registry."""

from __future__ import annotations

from agent_harness.tools.schemas import ToolDefinition, ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_definition(self, name: str) -> ToolDefinition:
        return self._definitions[name]

    def get_handler(self, name: str) -> ToolHandler:
        return self._handlers[name]

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

