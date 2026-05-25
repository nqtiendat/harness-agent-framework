"""Episodic memory primitives."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_harness.core.types import utc_now


class Episode(BaseModel):
    title: str
    summary: str
    session_id: str
    created_at: object = Field(default_factory=utc_now)

