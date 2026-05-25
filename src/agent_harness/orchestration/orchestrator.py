"""Multi-agent orchestration with verification subagent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from agent_harness.core.types import AgentState, Message, Role
from agent_harness.orchestration.workspace_manager import WorkspaceManager
from agent_harness.skills.skill_manager import SkillManager
from agent_harness.workflow.agent_workflow import AgentWorkflow


class Orchestrator(ABC):
    @abstractmethod
    async def run(self, goal: str) -> AgentState:
        """Run one or more agents toward a goal."""


class VerificationFinding(BaseModel):
    artifact: str
    ok: bool
    message: str
    severity: str = "info"


class VerificationReport(BaseModel):
    ok: bool
    verifier: str = "verification-subagent"
    findings: list[VerificationFinding] = Field(default_factory=list)
    checked_artifacts: list[str] = Field(default_factory=list)


class VerificationStrategy(Protocol):
    def verify(self, goal: str, workspace: WorkspaceManager, artifact_paths: list[str]) -> VerificationReport:
        """Verify primary output from a black-box workspace snapshot."""


class ReadOnlyVerificationSubagent:
    """Verifier that sees artifacts, not the primary agent's reasoning trace.

    Beyond existence + non-empty, it applies per-path checklist rules so a plan
    artifact has to look like a plan and a report has to look like a report.
    Pass a `rules` mapping `{path_or_suffix: [predicate(content) -> ok, message]}`
    to extend; sensible defaults cover `plan.md`, `primary-report.md`, and any
    `*-report.md` file.
    """

    def __init__(
        self,
        required_artifacts: list[str] | None = None,
        rules: dict[str, list[tuple]] | None = None,
    ) -> None:
        self.required_artifacts = required_artifacts or []
        self.rules = rules or self._default_rules()

    @staticmethod
    def _default_rules() -> dict[str, list[tuple]]:
        def _plan_has_numbered_steps(content: str) -> bool:
            return sum(1 for line in content.splitlines() if line.strip()[:2].rstrip(".").isdigit()) >= 1

        def _report_mentions_outcome(content: str) -> bool:
            lower = content.lower()
            return "outcome" in lower or "result" in lower or "summary" in lower

        return {
            "plan.md": [(_plan_has_numbered_steps, "plan must contain at least one numbered step")],
            "primary-report.md": [(_report_mentions_outcome, "report must reference an outcome/result/summary")],
            "*-report.md": [(_report_mentions_outcome, "report must reference an outcome/result/summary")],
        }

    def _rules_for(self, path: str) -> list[tuple]:
        if path in self.rules:
            return self.rules[path]
        for pattern, rules in self.rules.items():
            if pattern.startswith("*") and path.endswith(pattern[1:]):
                return rules
        return []

    def verify(self, goal: str, workspace: WorkspaceManager, artifact_paths: list[str]) -> VerificationReport:
        findings: list[VerificationFinding] = []
        paths = sorted(set([*artifact_paths, *self.required_artifacts]))
        for path in paths:
            if not workspace.exists(path):
                findings.append(
                    VerificationFinding(
                        artifact=path,
                        ok=False,
                        message="Required artifact is missing.",
                        severity="error",
                    )
                )
                continue
            content = workspace.read_artifact(path)
            if not content.strip():
                findings.append(
                    VerificationFinding(
                        artifact=path,
                        ok=False,
                        message="Artifact is empty.",
                        severity="error",
                    )
                )
                continue
            rule_failures = [msg for predicate, msg in self._rules_for(path) if not predicate(content)]
            if rule_failures:
                findings.append(
                    VerificationFinding(
                        artifact=path,
                        ok=False,
                        message="; ".join(rule_failures),
                        severity="error",
                    )
                )
                continue
            findings.append(
                VerificationFinding(
                    artifact=path,
                    ok=True,
                    message=f"Checklist passed for goal: {goal[:120]}",
                )
            )
        if not findings:
            findings.append(
                VerificationFinding(
                    artifact="workspace",
                    ok=False,
                    message="No artifacts were available for verification.",
                    severity="error",
                )
            )
        return VerificationReport(
            ok=all(finding.ok for finding in findings),
            findings=findings,
            checked_artifacts=paths,
        )


@dataclass(slots=True)
class OrchestrationResult:
    primary_state: AgentState
    verification: VerificationReport
    workspace_snapshot: dict


class VerificationOrchestrator(Orchestrator):
    """Centralized pattern: primary agent executes, verifier black-box tests artifacts."""

    def __init__(
        self,
        primary_workflow: AgentWorkflow,
        workspace: WorkspaceManager | None = None,
        verifier: VerificationStrategy | None = None,
        skill_manager: SkillManager | None = None,
    ) -> None:
        self.primary_workflow = primary_workflow
        self.workspace = workspace or WorkspaceManager()
        self.verifier = verifier or ReadOnlyVerificationSubagent(required_artifacts=["plan.md"])
        self.skill_manager = skill_manager
        self.last_result: OrchestrationResult | None = None

    async def run(self, goal: str) -> AgentState:
        result = await self.run_with_verification(goal)
        return result.primary_state

    async def run_with_verification(self, goal: str) -> OrchestrationResult:
        primary_state = await self.primary_workflow.run(AgentState(goal=goal))
        produced = self._publish_primary_artifacts(primary_state)
        verification = self.verifier.verify(goal, self.workspace, produced)
        self._publish_verification_report(verification)

        primary_state.metadata["verification"] = verification.model_dump()
        primary_state.metadata["workspace"] = self.workspace.snapshot_for_agent()
        if verification.ok:
            primary_state.outcome = f"{primary_state.outcome} Verification passed."
            if self.skill_manager is not None:
                primary_state.metadata["acquired_skill"] = self.skill_manager.acquire_from_state(
                    primary_state,
                    name=f"verified-{goal}",
                    description=f"Verified workflow for: {goal}",
                    tags=["verified", "orchestrated"],
                ).package.name
        else:
            primary_state.outcome = f"{primary_state.outcome} Verification failed."

        self.last_result = OrchestrationResult(
            primary_state=primary_state,
            verification=verification,
            workspace_snapshot=self.workspace.snapshot_for_agent(),
        )
        return self.last_result

    def _publish_primary_artifacts(self, state: AgentState) -> list[str]:
        plan_text = self._render_plan_artifact(state)
        report_text = self._render_primary_report(state)
        self.workspace.write_artifact("plan.md", plan_text, author="primary-agent", artifact_type="plan")
        self.workspace.write_artifact("primary-report.md", report_text, author="primary-agent", artifact_type="report")
        return ["plan.md", "primary-report.md"]

    def _publish_verification_report(self, report: VerificationReport) -> None:
        lines = [f"# Verification Report", "", f"ok: {report.ok}", ""]
        for finding in report.findings:
            lines.append(f"- [{finding.severity}] {finding.artifact}: {finding.message}")
        self.workspace.write_artifact(
            "verification-report.md",
            "\n".join(lines),
            author="verification-subagent",
            artifact_type="report",
        )

    def _render_plan_artifact(self, state: AgentState) -> str:
        lines = [f"# Plan for {state.goal}", ""]
        if state.plan:
            for index, step in enumerate(state.plan.steps, start=1):
                lines.append(f"{index}. [{step.status}] {step.description}")
        else:
            lines.append("No plan was created.")
        return "\n".join(lines)

    MAX_REPORT_OBSERVATIONS = 8
    MAX_REPORT_ACTIONS = 8

    def _render_primary_report(self, state: AgentState) -> str:
        # Summarize rather than dump — the workspace is artifact-mediated, but a
        # report that copies every assistant message reintroduces token bloat.
        recent_obs = state.observations[-self.MAX_REPORT_OBSERVATIONS :]
        observations = "\n".join(f"- {observation.summary}" for observation in recent_obs)
        assistant_messages = [m for m in state.messages if m.role is Role.ASSISTANT]
        recent_actions = assistant_messages[-self.MAX_REPORT_ACTIONS :]
        actions = "\n".join(f"- {message.content}" for message in recent_actions)
        elided_obs = max(0, len(state.observations) - len(recent_obs))
        elided_actions = max(0, len(assistant_messages) - len(recent_actions))
        return (
            f"# Primary Agent Report\n\n"
            f"Goal: {state.goal}\n\n"
            f"Outcome: {state.outcome}\n\n"
            f"Iterations: {state.iteration}\n\n"
            f"Observations (last {len(recent_obs)} of {len(state.observations)}):\n"
            f"{observations or '- none'}"
            + (f"\n- ... {elided_obs} earlier observations elided" if elided_obs else "")
            + "\n\n"
            f"Recent actions (last {len(recent_actions)} of {len(assistant_messages)}):\n"
            f"{actions or '- none'}"
            + (f"\n- ... {elided_actions} earlier actions elided" if elided_actions else "")
            + "\n"
        )
