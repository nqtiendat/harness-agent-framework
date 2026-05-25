"""Memory manager facade."""

from __future__ import annotations

from agent_harness.core.types import Message, Observation
from agent_harness.memory.context_processor import ContextProcessingResult, ContextProcessor
from agent_harness.memory.long_term_memory import LongTermMemoryService
from agent_harness.memory.short_term import WorkingMemory


class MemoryManager:
    def __init__(
        self,
        short_term: WorkingMemory | None = None,
        long_term: LongTermMemoryService | None = None,
        context_processor: ContextProcessor | None = None,
    ) -> None:
        self.short_term = short_term or WorkingMemory()
        self.long_term = long_term or LongTermMemoryService()
        self.context_processor = context_processor or ContextProcessor()

    def end_iteration(
        self,
        observations: list[Observation] | None = None,
        messages: list[Message] | None = None,
    ) -> ContextProcessingResult:
        result = self.context_processor.process(
            decisions=self.short_term.decisions,
            calibration_points=self.short_term.calibration_points,
            core_messages=self.short_term.core_messages,
            tool_result_refs=self.short_term.tool_result_refs,
            observations=observations,
            messages=messages,
        )
        self.short_term.conversation_summary = result.compressed_summary
        self.short_term.clear_rehydratable_tool_results()
        return result

    def maybe_compact(
        self,
        observations: list[Observation] | None = None,
        messages: list[Message] | None = None,
    ) -> ContextProcessingResult | None:
        """Run mid-iteration compaction if the watermark is breached."""
        if not self.context_processor.should_compact(
            decisions=self.short_term.decisions,
            calibration_points=self.short_term.calibration_points,
            core_messages=self.short_term.core_messages,
        ):
            return None
        return self.end_iteration(observations=observations, messages=messages)
