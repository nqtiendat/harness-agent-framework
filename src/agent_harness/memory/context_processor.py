"""Lifecycle-aware context processing and compression."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.types import Message, Observation, Role


class ContextCategory(str, Enum):
    DECISION = "decision"
    CALIBRATION_POINT = "calibration_point"
    CORE_CONVERSATION = "core_conversation"
    REHYDRATABLE_TOOL_OUTPUT = "rehydratable_tool_output"
    EPHEMERAL = "ephemeral"


class ContextItem(BaseModel):
    category: ContextCategory
    content: str
    importance: int = 1
    ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextProcessingResult(BaseModel):
    compressed_summary: str
    retained_items: list[ContextItem] = Field(default_factory=list)
    cleared_refs: list[str] = Field(default_factory=list)
    estimated_chars_saved: int = 0


class ContextProcessor:
    """Keep decision-critical state and evict state that can be rehydrated."""

    def __init__(
        self,
        max_summary_chars: int = 5000,
        max_decisions: int = 30,
        max_calibration_points: int = 20,
        max_core_messages: int = 12,
        compaction_watermark_chars: int | None = None,
    ) -> None:
        self.max_summary_chars = max_summary_chars
        self.max_decisions = max_decisions
        self.max_calibration_points = max_calibration_points
        self.max_core_messages = max_core_messages
        # Trigger an in-iteration compaction once raw working-memory size
        # crosses this threshold. Defaults to 80 % of the final summary budget.
        self.compaction_watermark_chars = compaction_watermark_chars or int(max_summary_chars * 0.8)

    def should_compact(
        self,
        *,
        decisions: list[str],
        calibration_points: list[str],
        core_messages: list[str],
    ) -> bool:
        """Return True when raw buffers exceed the watermark and need compaction."""
        raw_chars = sum(len(item) for item in (*decisions, *calibration_points, *core_messages))
        return raw_chars >= self.compaction_watermark_chars

    def process(
        self,
        *,
        decisions: list[str],
        calibration_points: list[str],
        core_messages: list[str],
        tool_result_refs: list[str],
        observations: list[Observation] | None = None,
        messages: list[Message] | None = None,
    ) -> ContextProcessingResult:
        retained: list[ContextItem] = []
        retained.extend(self._decision_items(decisions))
        retained.extend(self._calibration_items(calibration_points, observations or []))
        retained.extend(self._conversation_items(core_messages, messages or []))

        retained = sorted(retained, key=lambda item: item.importance, reverse=True)
        summary = self._render_summary(retained)
        if len(summary) > self.max_summary_chars:
            summary = self._render_summary(self._fit_items(retained))

        saved = sum(len(ref) for ref in tool_result_refs)
        return ContextProcessingResult(
            compressed_summary=summary,
            retained_items=retained,
            cleared_refs=list(tool_result_refs),
            estimated_chars_saved=saved,
        )

    def classify_message(self, message: Message) -> ContextItem:
        content = message.content.strip()
        lower = content.lower()
        if message.role is Role.TOOL:
            return ContextItem(
                category=ContextCategory.REHYDRATABLE_TOOL_OUTPUT,
                content="tool output stored externally",
                importance=0,
                ref=message.metadata.get("raw_ref") or message.metadata.get("artifact"),
            )
        if any(word in lower for word in ("decided", "decision", "must", "constraint", "approved", "denied")):
            return ContextItem(category=ContextCategory.DECISION, content=content, importance=5)
        if any(word in lower for word in ("calibration", "verified", "confirmed", "failed", "completed")):
            return ContextItem(category=ContextCategory.CALIBRATION_POINT, content=content, importance=4)
        if message.role in {Role.USER, Role.SYSTEM}:
            return ContextItem(category=ContextCategory.CORE_CONVERSATION, content=content, importance=3)
        return ContextItem(category=ContextCategory.EPHEMERAL, content=content, importance=1)

    def classify_observation(self, observation: Observation) -> ContextItem:
        signal_text = "; ".join(observation.signals[:3])
        content = observation.summary if not signal_text else f"{observation.summary} | signals: {signal_text}"
        importance = 4 if observation.metadata.get("calibration") else 3
        if observation.raw_ref:
            importance += 1
        return ContextItem(
            category=ContextCategory.CALIBRATION_POINT,
            content=content,
            importance=importance,
            ref=observation.raw_ref,
            metadata={"source": observation.source, "content_type": observation.metadata.get("content_type")},
        )

    def _decision_items(self, decisions: list[str]) -> list[ContextItem]:
        return [
            ContextItem(category=ContextCategory.DECISION, content=item, importance=5)
            for item in decisions[-self.max_decisions :]
            if item.strip()
        ]

    def _calibration_items(self, calibration_points: list[str], observations: list[Observation]) -> list[ContextItem]:
        items = [
            ContextItem(category=ContextCategory.CALIBRATION_POINT, content=item, importance=4)
            for item in calibration_points[-self.max_calibration_points :]
            if item.strip()
        ]
        items.extend(self.classify_observation(item) for item in observations[-self.max_calibration_points :])
        return items[-self.max_calibration_points :]

    def _conversation_items(self, core_messages: list[str], messages: list[Message]) -> list[ContextItem]:
        items = [
            ContextItem(category=ContextCategory.CORE_CONVERSATION, content=item, importance=3)
            for item in core_messages[-self.max_core_messages :]
            if item.strip()
        ]
        for message in messages[-self.max_core_messages :]:
            classified = self.classify_message(message)
            if classified.category in {
                ContextCategory.DECISION,
                ContextCategory.CALIBRATION_POINT,
                ContextCategory.CORE_CONVERSATION,
            }:
                items.append(classified)
        return items[-self.max_core_messages :]

    def _fit_items(self, items: list[ContextItem]) -> list[ContextItem]:
        selected: list[ContextItem] = []
        total = 0
        for item in items:
            projected = total + len(item.content) + 24
            if projected > self.max_summary_chars:
                continue
            selected.append(item)
            total = projected
        return selected

    def _render_summary(self, items: list[ContextItem]) -> str:
        grouped: dict[ContextCategory, list[ContextItem]] = {
            ContextCategory.DECISION: [],
            ContextCategory.CALIBRATION_POINT: [],
            ContextCategory.CORE_CONVERSATION: [],
        }
        for item in items:
            if item.category in grouped:
                grouped[item.category].append(item)

        sections: list[str] = []
        labels = {
            ContextCategory.DECISION: "Decisions",
            ContextCategory.CALIBRATION_POINT: "Calibration Points",
            ContextCategory.CORE_CONVERSATION: "Core Conversation",
        }
        for category, label in labels.items():
            values = grouped[category]
            if values:
                lines = "\n".join(f"- {self._trim(item.content, 500)}" for item in values)
                sections.append(f"{label}:\n{lines}")
        return "\n\n".join(sections)

    def _trim(self, value: str, limit: int) -> str:
        value = " ".join(value.split())
        return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."

