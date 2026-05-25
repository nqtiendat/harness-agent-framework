"""Environment construction for agentic trajectory generation."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agent_harness.core.types import AgentState
from agent_harness.evaluation.dynamic_benchmark import DynamicBenchmarkCase, DynamicBenchmarkRunner, EnvironmentMutation
from agent_harness.evaluation.evaluator import CostSafetyEvaluator, EvaluationReport
from agent_harness.evaluation.tracer import ExecutionTrace, ExecutionTracer
from agent_harness.orchestration.orchestrator import OrchestrationResult, VerificationOrchestrator
from agent_harness.orchestration.workspace_manager import WorkspaceManager
from agent_harness.workflow.agent_workflow import AgentWorkflow


def _deterministic_task_id(seed: int) -> str:
    """Stable, UUID-shaped task id derived from a seed."""
    rng = random.Random(seed)
    return str(UUID(int=rng.getrandbits(128)))


class SimulationTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    initial_artifacts: dict[str, str] = Field(default_factory=dict)
    mutations: list[EnvironmentMutation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None

    @classmethod
    def seeded(cls, *, seed: int, **kwargs: Any) -> "SimulationTask":
        """Construct a reproducible task whose id is derived from `seed`."""
        return cls(id=_deterministic_task_id(seed), seed=seed, **kwargs)


class SimulationResult(BaseModel):
    task: SimulationTask
    state: AgentState
    trace: ExecutionTrace
    evaluation: EvaluationReport
    trace_path: Path | None = None
    verification_ok: bool | None = None


class EnvironmentSimulator:
    """Feed tasks into the harness and collect traceable trajectories."""

    def __init__(
        self,
        workflow: AgentWorkflow | None = None,
        orchestrator: VerificationOrchestrator | None = None,
        benchmark_runner: DynamicBenchmarkRunner | None = None,
        workspace_factory: Callable[[SimulationTask, ExecutionTracer], WorkspaceManager] | None = None,
        evaluator: CostSafetyEvaluator | None = None,
        trace_directory: str | Path = "storage/traces",
    ) -> None:
        self.workflow = workflow
        self.orchestrator = orchestrator
        self.benchmark_runner = benchmark_runner
        self.workspace_factory = workspace_factory
        self.evaluator = evaluator or CostSafetyEvaluator()
        self.trace_directory = Path(trace_directory)
        self.trace_directory.mkdir(parents=True, exist_ok=True)

    async def run_task(self, task: SimulationTask) -> SimulationResult:
        if task.seed is not None:
            # Seed stdlib random so any sim-side stochastic choices (mutation
            # ordering, sampling) are reproducible. Note: this does not seed
            # the model, which controls its own determinism.
            random.seed(task.seed)
        tracer = ExecutionTracer(run_id=f"sim-{task.id}", directory=self.trace_directory)
        tracer.start_run({"task_id": task.id, "goal": task.goal, "seed": task.seed, "metadata": task.metadata})

        if self.benchmark_runner or task.mutations:
            result = await self._run_dynamic(task, tracer)
            return SimulationResult(
                task=task,
                state=result.state,
                trace=result.trace,
                evaluation=result.evaluation,
                trace_path=self.trace_directory / f"{tracer.run_id}.jsonl",
                verification_ok=result.state.metadata.get("verification", {}).get("ok"),
            )

        if self.orchestrator:
            orchestration = await self._run_orchestrated(task, tracer)
            trace = tracer.as_trace()
            return SimulationResult(
                task=task,
                state=orchestration.primary_state,
                trace=trace,
                evaluation=self.evaluator.evaluate(trace),
                trace_path=self.trace_directory / f"{tracer.run_id}.jsonl",
                verification_ok=orchestration.verification.ok,
            )

        if not self.workflow:
            raise ValueError("EnvironmentSimulator requires workflow, orchestrator, or benchmark_runner")

        previous_tracer = self.workflow.tracer
        self.workflow.tracer = tracer
        state = await self.workflow.run(AgentState(goal=task.goal))
        self.workflow.tracer = previous_tracer
        trace = tracer.as_trace()
        return SimulationResult(
            task=task,
            state=state,
            trace=trace,
            evaluation=self.evaluator.evaluate(trace),
            trace_path=self.trace_directory / f"{tracer.run_id}.jsonl",
            verification_ok=state.metadata.get("verification", {}).get("ok"),
        )

    async def run_batch(self, tasks: list[SimulationTask]) -> list[SimulationResult]:
        results: list[SimulationResult] = []
        for task in tasks:
            results.append(await self.run_task(task))
        return results

    def generate_curriculum(self, goals: list[str], *, base_seed: int | None = None) -> list[SimulationTask]:
        tasks: list[SimulationTask] = []
        for index, goal in enumerate(goals, start=1):
            kwargs: dict[str, Any] = {
                "goal": goal,
                "initial_artifacts": {"README.md": f"# Task {index}\n\n{goal}\n"},
                "metadata": {"curriculum_index": index},
            }
            if base_seed is not None:
                tasks.append(SimulationTask.seeded(seed=base_seed + index, **kwargs))
            else:
                tasks.append(SimulationTask(**kwargs))
        return tasks

    async def _run_orchestrated(self, task: SimulationTask, tracer: ExecutionTracer) -> OrchestrationResult:
        if self.orchestrator is None:
            raise ValueError("Missing orchestrator")
        previous_workflow_tracer = self.orchestrator.primary_workflow.tracer
        self.orchestrator.primary_workflow.tracer = tracer
        if self.workspace_factory:
            self.orchestrator.workspace = self.workspace_factory(task, tracer)
        else:
            self.orchestrator.workspace.tracer = tracer
        for path, content in task.initial_artifacts.items():
            record = self.orchestrator.workspace.write_artifact(path, content, author="environment", artifact_type="data")
            tracer.record_artifact(record.path, record.author, record.revision, record.artifact_type)
        result = await self.orchestrator.run_with_verification(task.goal)
        self.orchestrator.primary_workflow.tracer = previous_workflow_tracer
        tracer.record_state_update({}, self._state_snapshot(result.primary_state), label="orchestrated_state")
        tracer.record("run.completed", {"outcome": result.primary_state.outcome, "verification_ok": result.verification.ok})
        return result

    async def _run_dynamic(self, task: SimulationTask, tracer: ExecutionTracer):
        previous_tracer = self.workflow.tracer if self.workflow else None
        if self.benchmark_runner is None:
            if not self.workflow:
                raise ValueError("Dynamic simulation requires benchmark_runner or workflow")
            self.workflow.tracer = tracer
            workspace = self.workspace_factory(task, tracer) if self.workspace_factory else WorkspaceManager(tracer=tracer)
            self.benchmark_runner = DynamicBenchmarkRunner(
                self.workflow,
                workspace=workspace,
                evaluator=self.evaluator,
                tracer_factory=lambda run_id: tracer,
            )
        case = DynamicBenchmarkCase(
            name=task.id,
            goal=task.goal,
            initial_artifacts=task.initial_artifacts,
            mutations=task.mutations,
        )
        result = await self.benchmark_runner.run_case(case)
        if self.workflow:
            self.workflow.tracer = previous_tracer
        return result

    def _state_snapshot(self, state: AgentState) -> dict[str, Any]:
        return {
            "goal": state.goal,
            "done": state.done,
            "outcome": state.outcome,
            "iteration": state.iteration,
            "observations": len(state.observations),
            "messages": len(state.messages),
            "verification": state.metadata.get("verification"),
        }
