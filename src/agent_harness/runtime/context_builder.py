"""Build compact context for model calls."""

from __future__ import annotations

from agent_harness.core.types import AgentState, Message, Role


class ContextBuilder:
    def build(self, state: AgentState) -> list[Message]:
        messages = [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are an agent operating inside a governed harness. "
                    "Use compact reasoning, request tools through structured actions, "
                    "and respect calibration points."
                ),
            )
        ]
        messages.extend(state.messages[-8:])
        if state.plan:
            pending = [step.description for step in state.plan.pending_steps()]
            messages.append(Message(role=Role.SYSTEM, content=f"Current pending plan steps: {pending}"))
        if state.observations:
            signals = state.observations[-1].signals
            messages.append(Message(role=Role.SYSTEM, content=f"Latest environment signals: {signals}"))
        return messages

