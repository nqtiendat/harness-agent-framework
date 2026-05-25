"""Loop control for threshold-triggered execution."""

from __future__ import annotations

from agent_harness.core.types import AgentState
from agent_harness.governance.budgets import BudgetController


class LoopController:
    def __init__(self, budget: BudgetController) -> None:
        self.budget = budget

    def should_continue(self, state: AgentState) -> bool:
        self.budget.check_iteration(state.iteration)
        return not state.done

