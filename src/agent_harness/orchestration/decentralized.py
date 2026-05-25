"""Decentralized shared-state pattern."""

from __future__ import annotations

from agent_harness.core.types import AgentState
from agent_harness.orchestration.orchestrator import Orchestrator
from agent_harness.orchestration.shared_state import SharedStateItem, SharedStateSpace
from agent_harness.orchestration.workspace_manager import WorkspaceConflictError, WorkspaceManager


class DecentralizedOrchestrator(Orchestrator):
    def __init__(
        self,
        shared_state: SharedStateSpace | None = None,
        workspace: WorkspaceManager | None = None,
    ) -> None:
        self.shared_state = shared_state or SharedStateSpace()
        self.workspace = workspace or WorkspaceManager()

    async def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal, done=True, outcome="Recorded goal in shared state for decentralized agents.")
        self.shared_state.write(SharedStateItem(key=f"goal:{state.session_id}", value=goal, author="orchestrator"))
        self.workspace.write_artifact(
            f"goals/{state.session_id}.md",
            f"# Goal\n\n{goal}\n",
            author="orchestrator",
            artifact_type="plan",
        )
        state.metadata["workspace"] = self.workspace.snapshot_for_agent()
        return state

    def publish_plan(self, agent_id: str, path: str, content: str, expected_revision: int | None = None) -> str:
        try:
            record = self.workspace.write_artifact(
                path,
                content,
                author=agent_id,
                artifact_type="plan",
                expected_revision=expected_revision,
            )
            return f"published:{record.path}@rev{record.revision}"
        except WorkspaceConflictError as exc:
            return f"conflict:{exc.conflict.conflict_path}"
