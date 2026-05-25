"""Collaboration helpers over shared state."""

from __future__ import annotations

from agent_harness.orchestration.shared_state import SharedStateItem, SharedStateSpace


class CollaborationCoordinator:
    def __init__(self, state: SharedStateSpace) -> None:
        self.state = state

    def publish_finding(self, agent: str, key: str, value: str) -> None:
        self.state.write(SharedStateItem(key=key, value=value, author=agent))

