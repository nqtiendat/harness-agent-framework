"""Agent runtime session."""

from __future__ import annotations

from agent_harness.core.types import AgentState, Message, Role


class AgentSession:
    def __init__(self, goal: str) -> None:
        self.state = AgentState(goal=goal)
        self.state.messages.append(Message(role=Role.USER, content=goal))

