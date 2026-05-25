"""Task planning, decomposition, and revision."""

from __future__ import annotations

from agent_harness.core.types import Observation, Plan, PlanStep, ToolCall


class Planner:
    MAX_REVISION_ATTEMPTS = 2

    def create_plan(self, goal: str, observations: list[Observation] | None = None) -> Plan:
        return Plan(
            goal=goal,
            rationale="Initial conservative decomposition with calibration after each step.",
            steps=[
                PlanStep(
                    description="Inspect current context and constraints",
                    calibration_required=True,
                    expected_outcome="context inspected; constraints identified",
                ),
                PlanStep(
                    description="Execute the smallest safe next action",
                    calibration_required=True,
                    expected_outcome="action executed without error",
                ),
                PlanStep(
                    description="Summarize outcome and persist important decisions",
                    calibration_required=True,
                    expected_outcome="outcome summarized; decisions recorded",
                ),
            ],
        )

    def next_step(self, plan: Plan) -> PlanStep | None:
        pending = plan.pending_steps()
        return pending[0] if pending else None

    def attach_tool_call(self, step: PlanStep, tool_name: str, arguments: dict) -> PlanStep:
        step.tool_call = ToolCall(name=tool_name, arguments=arguments)
        return step

    def revise(self, plan: Plan, failed_step: PlanStep, failure_reason: str) -> PlanStep | None:
        """Insert a recovery step after a failure, capped at MAX_REVISION_ATTEMPTS.

        Returns the inserted step, or None if the budget is exhausted (caller
        should treat that as terminal). Recovery is appended right after the
        failed step so it runs next.
        """
        attempts = int(failed_step.metadata.get("revision_attempts", 0))
        if attempts >= self.MAX_REVISION_ATTEMPTS:
            return None
        failed_step.metadata["revision_attempts"] = attempts + 1

        recovery = PlanStep(
            description=f"Re-attempt: {failed_step.description}",
            calibration_required=True,
            expected_outcome=failed_step.expected_outcome,
            expected_signals=list(failed_step.expected_signals),
            metadata={
                "revision_of": failed_step.id,
                "failure_reason": failure_reason,
                "revision_attempts": attempts + 1,
            },
        )
        try:
            index = plan.steps.index(failed_step)
        except ValueError:
            plan.steps.append(recovery)
        else:
            plan.steps.insert(index + 1, recovery)
        return recovery
