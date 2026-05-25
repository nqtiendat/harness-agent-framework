"""Centralized coordinator pattern."""

from __future__ import annotations

from agent_harness.core.types import AgentState
from agent_harness.orchestration.orchestrator import (
    OrchestrationResult,
    ReadOnlyVerificationSubagent,
    VerificationOrchestrator,
    VerificationStrategy,
)
from agent_harness.orchestration.workspace_manager import WorkspaceManager
from agent_harness.skills.skill_manager import SkillManager
from agent_harness.workflow.agent_workflow import AgentWorkflow


class CentralizedOrchestrator(VerificationOrchestrator):
    def __init__(
        self,
        workflow: AgentWorkflow,
        workspace: WorkspaceManager | None = None,
        verifier: VerificationStrategy | None = None,
        skill_manager: SkillManager | None = None,
    ) -> None:
        super().__init__(
            primary_workflow=workflow,
            workspace=workspace,
            verifier=verifier or ReadOnlyVerificationSubagent(required_artifacts=["plan.md", "primary-report.md"]),
            skill_manager=skill_manager,
        )

    async def run(self, goal: str) -> AgentState:
        return await super().run(goal)

    async def run_with_verification(self, goal: str) -> OrchestrationResult:
        return await super().run_with_verification(goal)
