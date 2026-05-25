"""Feasibility filtering and sandbox simulation for tool invocation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from agent_harness.core.exceptions import PermissionDeniedError, ToolExecutionError
from agent_harness.core.types import ToolCall
from agent_harness.tools.schemas import ToolDefinition


class SandboxVerdict(BaseModel):
    allowed: bool
    reason: str
    normalized_arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "low"


class ToolFeasibilityFilter:
    """Reject malformed or impossible actions before governance spends budget."""

    def validate(self, call: ToolCall, definition: ToolDefinition) -> SandboxVerdict:
        if call.name != definition.name:
            return SandboxVerdict(allowed=False, reason=f"Tool name mismatch: {call.name} != {definition.name}")
        missing = self._missing_required(call.arguments, definition.input_schema)
        if missing:
            return SandboxVerdict(allowed=False, reason=f"Missing required arguments: {', '.join(missing)}")
        type_error = self._first_type_error(call.arguments, definition.input_schema)
        if type_error:
            return SandboxVerdict(allowed=False, reason=type_error)
        normalized = self._normalize_arguments(call.arguments)
        return SandboxVerdict(
            allowed=True,
            reason="Action is structurally feasible.",
            normalized_arguments=normalized,
            risk_level=max(call.risk_level, definition.risk_level, key=self._risk_weight),
        )

    def _missing_required(self, arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        required = schema.get("required", []) if isinstance(schema, dict) else []
        return [key for key in required if key not in arguments or arguments[key] in {None, ""}]

    def _first_type_error(self, arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        for key, spec in properties.items():
            if key not in arguments:
                continue
            expected = spec.get("type")
            value = arguments[key]
            if expected == "string" and not isinstance(value, str):
                return f"Argument {key} must be a string."
            if expected == "object" and not isinstance(value, dict):
                return f"Argument {key} must be an object."
            if expected == "array" and not isinstance(value, list):
                return f"Argument {key} must be an array."
            if expected == "integer" and not isinstance(value, int):
                return f"Argument {key} must be an integer."
            if expected == "number" and not isinstance(value, (int, float)):
                return f"Argument {key} must be a number."
        return None

    def _normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        for key, value in list(normalized.items()):
            if isinstance(value, str):
                normalized[key] = value.strip()
        return normalized

    def _risk_weight(self, risk: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(risk, 0)


class ToolSandbox:
    """Policy boundary for tool calls before handler execution."""

    dangerous_shell_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\brm\s+-rf\b",
            r"\bdel\s+/[sq]\b",
            r"\bformat\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bgit\s+reset\s+--hard\b",
            r">\s*/dev/sd[a-z]",
        ]
    ]

    def __init__(
        self,
        workspace: str | Path = ".",
        allowed_url_schemes: set[str] | None = None,
        feasibility_filter: ToolFeasibilityFilter | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.allowed_url_schemes = allowed_url_schemes or {"http", "https"}
        self.feasibility_filter = feasibility_filter or ToolFeasibilityFilter()

    def evaluate(self, call: ToolCall, definition: ToolDefinition) -> SandboxVerdict:
        verdict = self.feasibility_filter.validate(call, definition)
        if not verdict.allowed:
            return verdict

        arguments = verdict.normalized_arguments
        path_error = self._validate_path_arguments(arguments)
        if path_error:
            return SandboxVerdict(allowed=False, reason=path_error, normalized_arguments=arguments, risk_level=verdict.risk_level)

        url_error = self._validate_url_arguments(arguments)
        if url_error:
            return SandboxVerdict(allowed=False, reason=url_error, normalized_arguments=arguments, risk_level=verdict.risk_level)

        shell_error = self._validate_shell_arguments(call.name, arguments)
        if shell_error:
            return SandboxVerdict(allowed=False, reason=shell_error, normalized_arguments=arguments, risk_level="high")

        return verdict

    def assert_allowed(self, call: ToolCall, definition: ToolDefinition) -> dict[str, Any]:
        verdict = self.evaluate(call, definition)
        if not verdict.allowed:
            raise ToolExecutionError(f"Tool action rejected before execution: {verdict.reason}")
        return verdict.normalized_arguments

    def assert_path_within_workspace(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.workspace):
            raise PermissionDeniedError(f"Path escapes workspace: {resolved}")
        return resolved

    def _validate_path_arguments(self, arguments: dict[str, Any]) -> str | None:
        for key in ("path", "file", "directory", "target_path", "output_path"):
            value = arguments.get(key)
            if not isinstance(value, str) or not value:
                continue
            try:
                self.assert_path_within_workspace(value)
            except PermissionDeniedError as exc:
                return str(exc)
        return None

    def _validate_url_arguments(self, arguments: dict[str, Any]) -> str | None:
        for key in ("url", "uri", "endpoint"):
            value = arguments.get(key)
            if not isinstance(value, str) or not value:
                continue
            parsed = urlparse(value)
            if parsed.scheme not in self.allowed_url_schemes:
                return f"URL scheme is not allowed: {parsed.scheme or '<missing>'}"
            if not parsed.netloc:
                return f"URL must include a host: {value}"
        return None

    def _validate_shell_arguments(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if "shell" not in tool_name and "command" not in tool_name:
            return None
        command = arguments.get("command")
        if not isinstance(command, str):
            return None
        for pattern in self.dangerous_shell_patterns:
            if pattern.search(command):
                return f"Shell command rejected by sandbox pattern: {pattern.pattern}"
        return None

