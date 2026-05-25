"""Trajectory synthesis and SFT distillation from execution traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.evaluation.evaluator import CostSafetyEvaluator, EvaluationReport
from agent_harness.evaluation.tracer import ExecutionTrace, TraceEvent


class SFTMessage(BaseModel):
    role: str
    content: str


class SFTExample(BaseModel):
    id: str
    messages: list[SFTMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SynthesisCriteria(BaseModel):
    require_success: bool = True
    require_verification: bool = True
    max_safety_violations: int = 0
    min_visibility_score: float = 0.6
    max_tool_calls: int | None = None


class PreferencePair(BaseModel):
    """A DPO-shaped (chosen, rejected) pair for the same task."""

    id: str
    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataSynthesizer:
    """Convert high-quality traces into compact SFT examples."""

    def __init__(
        self,
        evaluator: CostSafetyEvaluator | None = None,
        criteria: SynthesisCriteria | None = None,
    ) -> None:
        self.evaluator = evaluator or CostSafetyEvaluator()
        self.criteria = criteria or SynthesisCriteria()

    def synthesize_directory(self, trace_directory: str | Path, output_path: str | Path) -> list[SFTExample]:
        traces = []
        for path in sorted(Path(trace_directory).glob("*.jsonl")):
            traces.append(self.load_trace(path))
        return self.write_dataset(traces, output_path)

    def write_dataset(self, traces: list[ExecutionTrace], output_path: str | Path) -> list[SFTExample]:
        examples = [example for trace in traces if (example := self.synthesize_trace(trace)) is not None]
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(example.model_dump_json() + "\n")
        return examples

    def synthesize_trace(self, trace: ExecutionTrace) -> SFTExample | None:
        report = self.evaluator.evaluate(trace)
        if not self._passes(trace, report):
            return None
        task = self._task_from_trace(trace)
        observations = self._observations(trace)
        plans = self._plans(trace)
        actions = self._safe_actions(trace)
        artifacts = self._artifacts(trace)
        answer = self._render_distilled_answer(plans=plans, observations=observations, actions=actions, artifacts=artifacts)
        return SFTExample(
            id=trace.run_id,
            messages=[
                SFTMessage(role="system", content="Use the agent harness by planning, acting safely, and preserving context budget."),
                SFTMessage(role="user", content=task),
                SFTMessage(role="assistant", content=answer),
            ],
            metadata={
                "run_id": trace.run_id,
                "cost": report.cost.model_dump(),
                "safety": report.safety.model_dump(exclude={"findings"}),
                "process_visibility_score": report.process_visibility_score,
                "verification_ok": self._verification_ok(trace),
            },
        )

    def synthesize_preference_pair(
        self,
        chosen: ExecutionTrace,
        rejected: ExecutionTrace,
    ) -> PreferencePair | None:
        """Build a (chosen, rejected) pair from two trajectories on the same task.

        The chosen trace must pass the criteria; the rejected trace is kept
        regardless so it can serve as a contrastive negative for DPO.
        """
        chosen_report = self.evaluator.evaluate(chosen)
        if not self._passes(chosen, chosen_report):
            return None
        prompt = self._task_from_trace(chosen)
        if self._task_from_trace(rejected) != prompt:
            return None

        chosen_answer = self._render_distilled_answer(
            plans=self._plans(chosen),
            observations=self._observations(chosen),
            actions=self._safe_actions(chosen),
            artifacts=self._artifacts(chosen),
        )
        rejected_answer = self._render_distilled_answer(
            plans=self._plans(rejected),
            observations=self._observations(rejected),
            actions=self._safe_actions(rejected),
            artifacts=self._artifacts(rejected),
        )
        rejected_report = self.evaluator.evaluate(rejected)
        return PreferencePair(
            id=f"{chosen.run_id}__vs__{rejected.run_id}",
            prompt=prompt,
            chosen=chosen_answer,
            rejected=rejected_answer,
            metadata={
                "chosen_run_id": chosen.run_id,
                "rejected_run_id": rejected.run_id,
                "chosen_visibility": chosen_report.process_visibility_score,
                "rejected_visibility": rejected_report.process_visibility_score,
                "chosen_violations": chosen_report.safety.violations,
                "rejected_violations": rejected_report.safety.violations,
            },
        )

    def write_preference_pairs(
        self,
        pairs: list[tuple[ExecutionTrace, ExecutionTrace]],
        output_path: str | Path,
    ) -> list[PreferencePair]:
        built: list[PreferencePair] = []
        for chosen, rejected in pairs:
            pair = self.synthesize_preference_pair(chosen, rejected)
            if pair is not None:
                built.append(pair)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for pair in built:
                handle.write(pair.model_dump_json() + "\n")
        return built

    def load_trace(self, path: str | Path) -> ExecutionTrace:
        events: list[TraceEvent] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(TraceEvent.model_validate_json(line))
        run_id = events[0].run_id if events else Path(path).stem
        return ExecutionTrace(run_id=run_id, events=events)

    def _passes(self, trace: ExecutionTrace, report: EvaluationReport) -> bool:
        if self.criteria.require_success and not self._success(trace):
            return False
        if self.criteria.require_verification and self._verification_ok(trace) is not True:
            return False
        if report.safety.violations > self.criteria.max_safety_violations:
            return False
        if report.process_visibility_score < self.criteria.min_visibility_score:
            return False
        if self.criteria.max_tool_calls is not None and report.cost.tool_calls > self.criteria.max_tool_calls:
            return False
        return True

    def _success(self, trace: ExecutionTrace) -> bool:
        completed = [event for event in trace.events if event.type == "run.completed"]
        if not completed:
            return False
        outcome = str(completed[-1].payload.get("outcome", "")).lower()
        return "failed" not in outcome and "stopped after failed" not in outcome

    def _verification_ok(self, trace: ExecutionTrace) -> bool | None:
        for event in reversed(trace.events):
            payload = event.payload
            if "verification_ok" in payload:
                return bool(payload["verification_ok"])
            after = payload.get("after")
            if isinstance(after, dict):
                verification = after.get("verification")
                if isinstance(verification, dict) and "ok" in verification:
                    return bool(verification["ok"])
        return None

    def _task_from_trace(self, trace: ExecutionTrace) -> str:
        for event in trace.events:
            if event.type == "run.started":
                return str(event.payload.get("goal") or event.payload.get("task_id") or "Complete the task.")
        return "Complete the task."

    def _observations(self, trace: ExecutionTrace) -> list[str]:
        items: list[str] = []
        for event in trace.events:
            if event.type == "state.updated":
                diff = event.payload.get("diff", {})
                if isinstance(diff, dict) and "observations" in diff:
                    items.append(f"observations changed: {diff['observations']}")
            elif event.type == "benchmark.mutation":
                items.append(f"environment mutation: {event.payload.get('mutation')}")
        return items[:12]

    def _plans(self, trace: ExecutionTrace) -> list[str]:
        return [
            f"wrote {event.payload.get('artifact_type')} artifact {event.payload.get('path')}"
            for event in trace.events
            if event.type == "artifact.written" and event.payload.get("artifact_type") == "plan"
        ][:8]

    def _safe_actions(self, trace: ExecutionTrace) -> list[str]:
        actions: list[str] = []
        for event in trace.events:
            if event.type != "tool.called":
                continue
            if event.payload.get("allowed") is False:
                continue
            call = event.payload.get("call", {})
            actions.append(f"call tool {call.get('name')} with validated arguments")
        return actions[:12]

    def _artifacts(self, trace: ExecutionTrace) -> list[str]:
        return [
            f"{event.payload.get('path')}@rev{event.payload.get('revision')}"
            for event in trace.events
            if event.type == "artifact.written"
        ][:12]

    def _render_distilled_answer(
        self,
        *,
        plans: list[str],
        observations: list[str],
        actions: list[str],
        artifacts: list[str],
    ) -> str:
        sections = [
            "Plan:",
            *(f"- {item}" for item in (plans or ["Create a compact plan artifact and verify it."])),
            "",
            "Observations:",
            *(f"- {item}" for item in (observations or ["Environment state remained consistent."])),
            "",
            "Actions:",
            *(f"- {item}" for item in (actions or ["No unsafe tool calls were required."])),
            "",
            "Artifacts:",
            *(f"- {item}" for item in (artifacts or ["No external artifacts were required."])),
        ]
        return "\n".join(sections)
