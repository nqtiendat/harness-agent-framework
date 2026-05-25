"""Action execution bridge."""

from __future__ import annotations

from agent_harness.core.types import PlanStep, ToolResult
from agent_harness.tools.executor import ToolExecutor


class ActionRunner:
    def __init__(self, executor: ToolExecutor) -> None:
        self.executor = executor

    async def act(self, step: PlanStep) -> ToolResult | None:
        if not step.tool_call:
            return None
        return await self.executor.execute(step.tool_call)

