"""Budget-capped execution controls."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from agent_harness.core.exceptions import BudgetExceededError


@dataclass(slots=True)
class BudgetLimits:
    max_iterations: int = 8
    max_tool_calls: int = 16
    max_seconds: int = 120
    max_model_calls: int = 16
    max_estimated_tokens: int = 64_000


class BudgetController:
    def __init__(self, limits: BudgetLimits | None = None) -> None:
        self.limits = limits or BudgetLimits()
        self.started_at = monotonic()
        self.tool_calls = 0
        self.model_calls = 0
        self.estimated_tokens = 0

    def check_iteration(self, iteration: int) -> None:
        self._check_time()
        if iteration >= self.limits.max_iterations:
            raise BudgetExceededError(f"Max iterations exceeded: {iteration}/{self.limits.max_iterations}")

    def record_tool_call(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.limits.max_tool_calls:
            raise BudgetExceededError(f"Max tool calls exceeded: {self.tool_calls}/{self.limits.max_tool_calls}")

    def record_model_call(self, estimated_tokens: int = 0) -> None:
        self.model_calls += 1
        self.estimated_tokens += estimated_tokens
        if self.model_calls > self.limits.max_model_calls:
            raise BudgetExceededError(f"Max model calls exceeded: {self.model_calls}/{self.limits.max_model_calls}")
        if self.estimated_tokens > self.limits.max_estimated_tokens:
            raise BudgetExceededError(
                f"Max estimated tokens exceeded: {self.estimated_tokens}/{self.limits.max_estimated_tokens}"
            )

    def _check_time(self) -> None:
        elapsed = monotonic() - self.started_at
        if elapsed > self.limits.max_seconds:
            raise BudgetExceededError(f"Max runtime exceeded: {elapsed:.1f}s/{self.limits.max_seconds}s")

