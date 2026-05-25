"""Dynamic benchmark runner for changing environments."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.types import AgentState
from agent_harness.evaluation.evaluator import CostSafetyEvaluator, EvaluationReport
from agent_harness.evaluation.tracer import ExecutionTrace, ExecutionTracer
from agent_harness.orchestration.workspace_manager import WorkspaceConflictError, WorkspaceManager
from agent_harness.workflow.agent_workflow import AgentWorkflow


class EnvironmentMutation(BaseModel):
    name: str
    artifact_path: str
    content: str
    author: str = "external-environment"
    expected_revision: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DynamicBenchmarkCase(BaseModel):
    name: str
    goal: str
    initial_artifacts: dict[str, str] = Field(default_factory=dict)
    mutations: list[EnvironmentMutation] = Field(default_factory=list)
    expected_safe_recovery: bool = True


class DynamicBenchmarkResult(BaseModel):
    case_name: str
    state: AgentState
    trace: ExecutionTrace
    evaluation: EvaluationReport
    recovered_safely: bool
    conflicts_detected: int = 0


class DynamicBenchmarkRunner:
    """Run agents in environments that change independently during execution."""

    def __init__(
        self,
        workflow: AgentWorkflow,
        workspace: WorkspaceManager | None = None,
        evaluator: CostSafetyEvaluator | None = None,
        tracer_factory: Callable[[str], ExecutionTracer] | None = None,
    ) -> None:
        self.workflow = workflow
        self.workspace = workspace or WorkspaceManager()
        self.evaluator = evaluator or CostSafetyEvaluator()
        self.tracer_factory = tracer_factory or (lambda run_id: ExecutionTracer(run_id=run_id))

    async def run_case(self, case: DynamicBenchmarkCase) -> DynamicBenchmarkResult:
        tracer = self.tracer_factory(f"benchmark-{case.name}")
        tracer.start_run({"benchmark": case.name, "goal": case.goal})
        for path, content in case.initial_artifacts.items():
            record = self.workspace.write_artifact(path, content, author="benchmark", artifact_type="data")
            tracer.record_artifact(record.path, record.author, record.revision, record.artifact_type)

        before = self._state_snapshot(None)
        state = await self.workflow.run(AgentState(goal=case.goal))
        tracer.record_state_update(before, self._state_snapshot(state), label="agent_state")

        conflicts = 0
        for mutation in case.mutations:
            try:
                record = self.workspace.write_artifact(
                    mutation.artifact_path,
                    mutation.content,
                    author=mutation.author,
                    artifact_type="data",
                    expected_revision=mutation.expected_revision,
                    metadata=mutation.metadata,
                )
                tracer.record("benchmark.mutation", {"mutation": mutation.model_dump(), "revision": record.revision})
                tracer.record_artifact(record.path, record.author, record.revision, record.artifact_type)
            except WorkspaceConflictError as exc:
                conflicts += 1
                tracer.record(
                    "artifact.conflict",
                    {
                        "mutation": mutation.model_dump(),
                        "conflict": exc.conflict.model_dump(),
                    },
                )

        recovered = self._recover_safely(state, conflicts)
        state.metadata["dynamic_benchmark"] = {
            "case": case.name,
            "mutations": len(case.mutations),
            "conflicts_detected": conflicts,
            "recovered_safely": recovered,
        }
        tracer.record_state_update({}, state.metadata["dynamic_benchmark"], label="dynamic_benchmark")
        tracer.complete_run(state.outcome, {"recovered_safely": recovered, "conflicts_detected": conflicts})
        trace = tracer.as_trace()
        return DynamicBenchmarkResult(
            case_name=case.name,
            state=state,
            trace=trace,
            evaluation=self.evaluator.evaluate(trace),
            recovered_safely=recovered,
            conflicts_detected=conflicts,
        )

    def simulate_external_file_change(
        self,
        path: str,
        content: str,
        *,
        expected_revision: int | None = None,
    ) -> EnvironmentMutation:
        return EnvironmentMutation(
            name=f"external-change:{path}",
            artifact_path=path,
            content=content,
            expected_revision=expected_revision,
        )

    def _recover_safely(self, state: AgentState, conflicts: int) -> bool:
        if conflicts:
            return True
        return state.done and state.outcome is not None

    def _state_snapshot(self, state: AgentState | None) -> dict[str, Any]:
        if state is None:
            return {"done": False, "iteration": 0, "outcome": None}
        return {
            "done": state.done,
            "iteration": state.iteration,
            "outcome": state.outcome,
            "observations": len(state.observations),
            "messages": len(state.messages),
        }

