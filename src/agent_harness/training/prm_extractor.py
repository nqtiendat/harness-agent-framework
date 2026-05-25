"""Process reward extraction from execution traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_harness.evaluation.evaluator import CostSafetyEvaluator
from agent_harness.evaluation.tracer import ExecutionTrace, TraceEvent
from agent_harness.training.trajectory_synthesizer import DataSynthesizer


class ProcessRewardStep(BaseModel):
    run_id: str
    step_index: int
    event_type: str
    reward: float
    label: str
    rationale: str
    payload: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class RewardModel(Protocol):
    """Pluggable scorer that returns (reward, label, rationale) per event.

    Implement this to swap the hand-coded rubric for a learned reward model or
    an LLM-as-judge; PRMExtractor will call `score_event` for every event in a
    trace and still apply trajectory-level safety adjustments on top.
    """

    def score_event(self, event: TraceEvent) -> tuple[float, str, str]:
        ...


class PRMExtractor:
    """Generate process-level rewards for RL from harness traces."""

    def __init__(
        self,
        evaluator: CostSafetyEvaluator | None = None,
        max_memory_summary_chars: int = 5000,
        reward_model: RewardModel | None = None,
    ) -> None:
        self.evaluator = evaluator or CostSafetyEvaluator()
        self.max_memory_summary_chars = max_memory_summary_chars
        self.reward_model = reward_model
        self._loader = DataSynthesizer(evaluator=self.evaluator)

    def extract(self, trace: ExecutionTrace) -> list[ProcessRewardStep]:
        report = self.evaluator.evaluate(trace)
        steps: list[ProcessRewardStep] = []
        scorer = self.reward_model.score_event if self.reward_model else self._score_event
        for index, event in enumerate(trace.events):
            reward, label, rationale = scorer(event)
            if report.safety.violations and event.type in {"tool.called", "safety.violation"}:
                reward -= 0.5
                rationale += " Global trajectory had safety violations."
            steps.append(
                ProcessRewardStep(
                    run_id=trace.run_id,
                    step_index=index,
                    event_type=event.type,
                    reward=max(-1.0, min(1.0, reward)),
                    label=label,
                    rationale=rationale.strip(),
                    payload=self._compact_payload(event.payload),
                )
            )
        return steps

    def extract_directory(self, trace_directory: str | Path, output_path: str | Path) -> list[ProcessRewardStep]:
        rewards: list[ProcessRewardStep] = []
        for path in sorted(Path(trace_directory).glob("*.jsonl")):
            rewards.extend(self.extract(self._loader.load_trace(path)))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for step in rewards:
                handle.write(step.model_dump_json() + "\n")
        return rewards

    def _score_event(self, event: TraceEvent) -> tuple[float, str, str]:
        if event.type == "tool.called":
            if event.payload.get("allowed") is False:
                return -0.8, "unsafe_tool_attempt", "Tool call was blocked or failed sandbox feasibility."
            return 0.4, "safe_tool_call", "Tool call passed sandbox and governance checks."
        if event.type == "tool.completed":
            result = event.payload.get("result", {})
            if result.get("ok") is False:
                return -0.3, "tool_failed", "Tool returned an unsuccessful result."
            return 0.3, "tool_succeeded", "Tool completed successfully."
        if event.type == "memory.compressed":
            summary_chars = int(event.payload.get("summary_chars") or 0)
            cleared_refs = event.payload.get("cleared_refs") or []
            if summary_chars > self.max_memory_summary_chars:
                return -0.4, "memory_bloat", "Compressed memory exceeded target context budget."
            if cleared_refs:
                return 0.5, "context_budget_preserved", "Compression cleared rehydratable tool outputs."
            return 0.2, "memory_compressed", "Memory was compressed into compact state."
        if event.type == "artifact.written":
            return 0.3, "artifact_progress", "Agent externalized progress into an inspectable artifact."
        if event.type == "artifact.conflict":
            return -0.2, "environment_conflict", "Workspace conflict was detected and surfaced."
        if event.type == "state.updated":
            diff = event.payload.get("diff", {})
            if isinstance(diff, dict) and "iteration" in diff:
                return 0.1, "loop_progress", "Agent advanced through a controlled loop iteration."
            return 0.0, "state_update", "State changed without direct reward implication."
        if event.type == "run.completed":
            outcome = str(event.payload.get("outcome", "")).lower()
            if "failed" in outcome:
                return -0.6, "failed_outcome", "Run completed with failure outcome."
            return 0.6, "successful_outcome", "Run completed without failure signal."
        if event.type == "safety.violation":
            return -1.0, "safety_violation", "Explicit safety violation was recorded."
        if event.type == "benchmark.mutation":
            return 0.0, "environment_changed", "Environment changed independently of agent behavior."
        return 0.0, "neutral", "Neutral trace event."

    def _compact_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload)
        if len(text) <= 1000:
            return payload
        return {"truncated": True, "preview": text[:1000]}

