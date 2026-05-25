"""Governed tool execution."""

from __future__ import annotations

from agent_harness.core.exceptions import PermissionDeniedError, ToolExecutionError
from agent_harness.core.types import ToolCall, ToolResult
from agent_harness.evaluation.tracer import ErrorSource, ExecutionTracer
from agent_harness.governance.budgets import BudgetController
from agent_harness.governance.permissions import PermissionController, PermissionRequest
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tools.tool_sandbox import ToolSandbox


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionController,
        budget: BudgetController,
        sandbox: ToolSandbox | None = None,
        tracer: ExecutionTracer | None = None,
    ) -> None:
        self.registry = registry
        self.permissions = permissions
        self.budget = budget
        self.sandbox = sandbox or ToolSandbox()
        self.tracer = tracer

    async def execute(self, call: ToolCall) -> ToolResult:
        definition = self.registry.get_definition(call.name)
        try:
            normalized_arguments = self.sandbox.assert_allowed(call, definition)
        except Exception as exc:
            if self.tracer:
                self.tracer.record_tool_call(call, allowed=False, reason=str(exc))
                self.tracer.record_error(
                    source=ErrorSource.HARNESS_GOVERNANCE,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    context={"tool": call.name, "stage": "sandbox"},
                    span_id=call.call_id,
                )
            raise
        if self.tracer:
            self.tracer.record_tool_call(call, allowed=True)
        try:
            self.permissions.check(
                PermissionRequest(
                    resource=definition.permission_resource,
                    action=call.name,
                    risk_level=max(call.risk_level, definition.risk_level, key={"low": 0, "medium": 1, "high": 2}.get),
                    metadata=normalized_arguments,
                )
            )
        except PermissionDeniedError as exc:
            if self.tracer:
                self.tracer.record_error(
                    source=ErrorSource.HARNESS_GOVERNANCE,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    context={"tool": call.name, "stage": "permissions"},
                    span_id=call.call_id,
                )
            raise
        self.budget.record_tool_call()
        try:
            result = await self.registry.get_handler(call.name)(normalized_arguments)
            if self.tracer:
                self.tracer.record_tool_result(result)
            return result
        except Exception as exc:
            if self.tracer:
                self.tracer.record_error(
                    source=ErrorSource.TOOL_RUNTIME,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    context={"tool": call.name, "stage": "handler"},
                    span_id=call.call_id,
                )
            raise ToolExecutionError(f"Tool failed: {call.name}") from exc
