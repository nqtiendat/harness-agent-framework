"""Procedural memory primitives."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Procedure(BaseModel):
    name: str
    steps: list[str] = Field(default_factory=list)
    source_skill: str | None = None

