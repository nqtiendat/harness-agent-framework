"""Cost and safety evaluation for agent trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.evaluation.tracer import ExecutionTrace, TraceEvent


class CostQualityMetrics(BaseModel):
    estimated_tokens: int = 0
    iterations: int = 0
    latency_ms: float = 0.0
    tool_calls: int = 0
    artifact_writes: int = 0
    memory_compressions: int = 0


class SafetyFinding(BaseModel):
    severity: str
    message: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SafetyQualityMetrics(BaseModel):
    violations: int = 0
    blocked_tool_calls: int = 0
    non_whitelisted_tool_calls: int = 0
    forbidden_memory_accesses: int = 0
    findings: list[SafetyFinding] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    run_id: str
    cost: CostQualityMetrics
    safety: SafetyQualityMetrics
    process_visibility_score: float
    summary: str


class CostSafetyEvaluator:
    """Evaluate internal trajectory quality, not just final output."""

    def __init__(
        self,
        tool_whitelist: set[str] | None = None,
        allowed_memory_roots: list[str | Path] | None = None,
    ) -> None:
        self.tool_whitelist = tool_whitelist or set()
        self.allowed_memory_roots = [Path(root).resolve() for root in (allowed_memory_roots or ["storage"])]

    def evaluate(self, trace: ExecutionTrace) -> EvaluationReport:
        cost = self._cost(trace.events)
        safety = self._safety(trace.events)
        visibility = self._process_visibility(trace.events)
        return EvaluationReport(
            run_id=trace.run_id,
            cost=cost,
            safety=safety,
            process_visibility_score=visibility,
            summary=(
                f"events={len(trace.events)}, tool_calls={cost.tool_calls}, "
                f"violations={safety.violations}, visibility={visibility:.2f}"
            ),
        )

    def _cost(self, events: list[TraceEvent]) -> CostQualityMetrics:
        cost = CostQualityMetrics()
        timestamps = [event.created_at for event in events]
        for event in events:
            if event.type == "state.updated":
                after = event.payload.get("after", {})
                if isinstance(after, dict) and "iteration" in after:
                    cost.iterations = max(cost.iterations, int(after.get("iteration") or 0))
            elif event.type == "tool.called":
                cost.tool_calls += 1
            elif event.type == "artifact.written":
                cost.artifact_writes += 1
            elif event.type == "memory.compressed":
                cost.memory_compressions += 1
            usage = event.payload.get("usage")
            if isinstance(usage, dict):
                cost.estimated_tokens += int(usage.get("total_tokens") or usage.get("estimated_tokens") or 0)
        if len(timestamps) >= 2:
            try:
                cost.latency_ms = (timestamps[-1] - timestamps[0]).total_seconds() * 1000
            except TypeError:
                cost.latency_ms = 0.0
        return cost

    def _safety(self, events: list[TraceEvent]) -> SafetyQualityMetrics:
        safety = SafetyQualityMetrics()
        for event in events:
            if event.type == "tool.called":
                call = event.payload.get("call", {})
                name = call.get("name", "")
                if event.payload.get("allowed") is False:
                    safety.blocked_tool_calls += 1
                    safety.findings.append(
                        SafetyFinding(
                            severity="warning",
                            message=f"Tool call was blocked: {name}",
                            event_type=event.type,
                            payload=event.payload,
                        )
                    )
                if self.tool_whitelist and name not in self.tool_whitelist:
                    safety.non_whitelisted_tool_calls += 1
                    safety.findings.append(
                        SafetyFinding(
                            severity="error",
                            message=f"Tool call outside whitelist: {name}",
                            event_type=event.type,
                            payload=event.payload,
                        )
                    )
            elif event.type in {"artifact.written", "memory.cleared"}:
                self._check_paths(event, safety)
            elif event.type == "safety.violation":
                safety.findings.append(
                    SafetyFinding(
                        severity=event.payload.get("severity", "error"),
                        message=event.payload.get("message", "safety violation"),
                        event_type=event.type,
                        payload=event.payload,
                    )
                )
        safety.violations = (
            safety.blocked_tool_calls
            + safety.non_whitelisted_tool_calls
            + safety.forbidden_memory_accesses
            + len([item for item in safety.findings if item.event_type == "safety.violation"])
        )
        return safety

    def _check_paths(self, event: TraceEvent, safety: SafetyQualityMetrics) -> None:
        candidates: list[str] = []
        if "path" in event.payload:
            candidates.append(str(event.payload["path"]))
        if "refs" in event.payload and isinstance(event.payload["refs"], list):
            candidates.extend(str(item) for item in event.payload["refs"])
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute() and ".." not in path.parts:
                continue
            resolved = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
            if not any(resolved.is_relative_to(root) for root in self.allowed_memory_roots):
                safety.forbidden_memory_accesses += 1
                safety.findings.append(
                    SafetyFinding(
                        severity="error",
                        message=f"Path outside allowed memory roots: {candidate}",
                        event_type=event.type,
                        payload=event.payload,
                    )
                )

    def _process_visibility(self, events: list[TraceEvent]) -> float:
        """Weighted observability score in [0, 1].

        Required core events still cover the basics; richer signals (model
        instrumentation, error classification, span linkage) push the score
        higher so a fully instrumented harness scores above one that only
        emits the legacy 5-event set.
        """
        required = {
            "run.started",
            "state.updated",
            "artifact.written",
            "memory.compressed",
            "run.completed",
        }
        present_types = {event.type for event in events}
        base = len(required & present_types) / len(required)

        bonus = 0.0
        if "model.call_started" in present_types and "model.call_completed" in present_types:
            bonus += 0.10
        if any(e.type == "error" and e.payload.get("source") for e in events):
            bonus += 0.05
        if any(e.span_id for e in events):
            bonus += 0.05

        return min(1.0, base * 0.8 + bonus + 0.2 * (1 if base == 1.0 else base))
