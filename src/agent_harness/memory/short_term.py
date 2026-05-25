"""Short-term memory lifecycle management."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorkingMemory(BaseModel):
    decisions: list[str] = Field(default_factory=list)
    calibration_points: list[str] = Field(default_factory=list)
    core_messages: list[str] = Field(default_factory=list)
    tool_result_refs: list[str] = Field(default_factory=list)
    conversation_summary: str = ""

    def record_decision(self, decision: str) -> None:
        self.decisions.append(decision)

    def record_calibration_point(self, point: str) -> None:
        self.calibration_points.append(point)

    def record_core_message(self, message: str) -> None:
        self.core_messages.append(message)

    def compress(self) -> str:
        important = [*self.decisions[-20:], *self.calibration_points[-10:], *self.core_messages[-8:]]
        self.conversation_summary = "\n".join(important)
        return self.conversation_summary

    def clear_rehydratable_tool_results(self) -> None:
        self.tool_result_refs.clear()
