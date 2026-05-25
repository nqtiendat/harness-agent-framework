"""Semantic memory primitives."""

from __future__ import annotations

from pydantic import BaseModel


class SemanticFact(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0

