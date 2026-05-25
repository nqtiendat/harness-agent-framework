"""Command-line entrypoint."""

from __future__ import annotations

import anyio
import typer
from rich import print

from agent_harness import __version__
from agent_harness.core.config import load_settings
from agent_harness.core.types import AgentState
from agent_harness.governance.audit import AuditLogger
from agent_harness.governance.budgets import BudgetController, BudgetLimits
from agent_harness.memory.manager import MemoryManager
from agent_harness.runtime.loop_controller import LoopController
from agent_harness.workflow.agent_workflow import AgentWorkflow
from agent_harness.workflow.calibration import CalibrationManager
from agent_harness.workflow.perception import PerceptionPipeline
from agent_harness.workflow.planning import Planner

app = typer.Typer(help="Agent Harness Framework CLI")


@app.command("run")
def run(goal: str) -> None:
    settings = load_settings()
    budget = BudgetController(
        BudgetLimits(
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
            max_seconds=settings.max_seconds,
            max_model_calls=settings.max_model_calls,
            max_estimated_tokens=settings.max_estimated_tokens,
        )
    )
    workflow = AgentWorkflow(
        loop_controller=LoopController(budget),
        perception=PerceptionPipeline(settings.storage_dir / "workspaces" / "raw"),
        planner=Planner(),
        calibration=CalibrationManager(),
        memory=MemoryManager(),
        audit=AuditLogger(settings.storage_dir / "audit"),
    )
    state = anyio.run(workflow.run, AgentState(goal=goal))
    print({"session_id": state.session_id, "outcome": state.outcome, "iterations": state.iteration})


@app.command("version")
def version() -> None:
    print(__version__)


if __name__ == "__main__":
    app()
