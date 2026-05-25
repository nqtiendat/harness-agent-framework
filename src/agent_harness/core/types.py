"""Shared typed models for agent harness components."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    AGENT = "agent"


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    risk_level: Literal["low", "medium", "high"] = "low"


class ToolResult(BaseModel):
    call_id: str
    name: str
    ok: bool
    output: str = ""
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    summary: str
    signals: list[str] = Field(default_factory=list)
    raw_ref: str | None = None
    source: str = "environment"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    status: Literal["pending", "in_progress", "completed", "failed", "skipped"] = "pending"
    tool_call: ToolCall | None = None
    calibration_required: bool = True
    expected_outcome: str | None = None
    expected_signals: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    rationale: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def pending_steps(self) -> list[PlanStep]:
        return [step for step in self.steps if step.status == "pending"]


class AgentState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    messages: list[Message] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    plan: Plan | None = None
    iteration: int = 0
    done: bool = False
    outcome: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(BaseModel):
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

