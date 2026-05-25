"""API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class RunRequest(BaseModel):
    goal: str


class RunResponse(BaseModel):
    session_id: str
    outcome: str | None

