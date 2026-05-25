"""Layered permission controls for runtime actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_harness.core.exceptions import PermissionDeniedError


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    resource: str
    action: str
    risk_level: str = "low"
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PermissionResult:
    decision: PermissionDecision
    reason: str = ""


class PermissionController:
    def __init__(self, default_decision: PermissionDecision = PermissionDecision.ASK) -> None:
        self.default_decision = default_decision
        self._rules: dict[str, PermissionDecision] = {}

    def add_rule(self, resource: str, decision: PermissionDecision | str) -> None:
        self._rules[resource] = PermissionDecision(decision)

    def check(self, request: PermissionRequest) -> PermissionResult:
        decision = self._rules.get(request.resource, self.default_decision)
        if decision is PermissionDecision.DENY:
            raise PermissionDeniedError(f"Permission denied for {request.resource}:{request.action}")
        return PermissionResult(decision=decision, reason=f"Matched policy for {request.resource}")

