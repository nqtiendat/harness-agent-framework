"""Calibration points for verifying plan progress."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_harness.core.types import Observation, PlanStep

VerifierFn = Callable[[PlanStep, Observation], bool]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    ok: bool
    rationale: str
    matched_signals: tuple[str, ...] = ()


class CalibrationManager:
    """Score an observation against a plan step's expected outcome.

    Three layers, in order of strength:
      1. Custom verifier callable (e.g. LLM-as-judge) if supplied.
      2. Expected-signal / expected-outcome keyword matching from the step.
      3. Fallback existence check on the observation (signals or summary).
    """

    def __init__(self, verifier: VerifierFn | None = None) -> None:
        self.verifier = verifier

    def calibrate(self, step: PlanStep, observation: Observation) -> bool:
        return self.evaluate(step, observation).ok

    def evaluate(
        self,
        step: PlanStep,
        observation: Observation,
        *,
        apply_rubric: bool = True,
    ) -> CalibrationResult:
        if self.verifier is not None:
            ok = bool(self.verifier(step, observation))
            return CalibrationResult(ok=ok, rationale="custom verifier")

        if not apply_rubric:
            if observation.signals or observation.summary:
                return CalibrationResult(ok=True, rationale="observation present (rubric skipped)")
            return CalibrationResult(ok=False, rationale="observation empty")

        haystack = (observation.summary + " " + " ".join(observation.signals)).lower()
        matched: list[str] = []
        if step.expected_signals:
            for needle in step.expected_signals:
                if needle.lower() in haystack:
                    matched.append(needle)
            if matched:
                return CalibrationResult(ok=True, rationale="expected signals matched", matched_signals=tuple(matched))
            return CalibrationResult(ok=False, rationale="expected signals absent", matched_signals=())

        if step.expected_outcome:
            keywords = [w for w in step.expected_outcome.lower().split() if len(w) > 3]
            hits = [w for w in keywords if w in haystack]
            if len(hits) >= max(1, len(keywords) // 2):
                return CalibrationResult(ok=True, rationale="expected outcome keywords matched", matched_signals=tuple(hits))
            return CalibrationResult(ok=False, rationale="expected outcome keywords missing", matched_signals=tuple(hits))

        if observation.signals or observation.summary:
            return CalibrationResult(ok=True, rationale="observation present (no rubric supplied)")
        return CalibrationResult(ok=False, rationale="observation empty")
