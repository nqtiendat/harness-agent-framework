"""Debate coordination."""

from __future__ import annotations

from pydantic import BaseModel


class DebatePosition(BaseModel):
    agent: str
    claim: str
    evidence: str = ""


class DebateCoordinator:
    def decide(self, positions: list[DebatePosition]) -> DebatePosition | None:
        if not positions:
            return None
        return max(positions, key=lambda position: len(position.evidence))

