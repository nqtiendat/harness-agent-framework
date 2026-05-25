"""Runtime events emitted by the harness."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_harness.core.types import utc_now


EventType = Literal[
    "session.started",
    "workflow.perceived",
    "workflow.planned",
    "workflow.acted",
    "workflow.calibrated",
    "permission.checked",
    "budget.checked",
    "memory.written",
    "skill.loaded",
    "session.completed",
]


class HarnessEvent(BaseModel):
    type: EventType
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)

