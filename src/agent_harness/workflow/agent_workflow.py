"""Perceive-Plan-Act workflow with post-action re-perception and re-planning."""

from __future__ import annotations

from agent_harness.core.events import HarnessEvent
from agent_harness.core.types import AgentState, Message, Observation, Role, ToolResult
from agent_harness.evaluation.tracer import ExecutionTracer
from agent_harness.governance.audit import AuditLogger
from agent_harness.memory.manager import MemoryManager
from agent_harness.runtime.loop_controller import LoopController
from agent_harness.workflow.action import ActionRunner
from agent_harness.workflow.calibration import CalibrationManager
from agent_harness.workflow.perception import PerceptionPipeline
from agent_harness.workflow.planning import Planner


class AgentWorkflow:
    def __init__(
        self,
        loop_controller: LoopController,
        perception: PerceptionPipeline,
        planner: Planner,
        calibration: CalibrationManager,
        memory: MemoryManager,
        audit: AuditLogger | None = None,
        tracer: ExecutionTracer | None = None,
        action_runner: ActionRunner | None = None,
    ) -> None:
        self.loop_controller = loop_controller
        self.perception = perception
        self.planner = planner
        self.calibration = calibration
        self.memory = memory
        self.audit = audit
        self.tracer = tracer
        self.action_runner = action_runner

    async def run(self, state: AgentState) -> AgentState:
        if self.tracer:
            self.tracer.start_run({"goal": state.goal, "session_id": state.session_id})
        self._emit(state, "session.started", {"goal": state.goal})
        while self.loop_controller.should_continue(state):
            before_iteration = self._state_snapshot(state)
            if state.plan is None:
                state.plan = self.planner.create_plan(state.goal, state.observations)
                self._emit(state, "workflow.planned", {"steps": [s.description for s in state.plan.steps]})

            step = self.planner.next_step(state.plan)
            if step is None:
                state.done = True
                state.outcome = "Plan completed."
                break

            step.status = "in_progress"

            # 1. Pre-action perception (synthetic — describes what we're about to do).
            pre_observation = self.perception.perceive(
                f"Executing step: {step.description}", source="workflow"
            )
            state.observations.append(pre_observation)
            self._emit(state, "workflow.perceived", pre_observation.model_dump())

            # 2. Action (if a tool is attached and we have a runner).
            tool_result = await self._maybe_act(step, state)

            # 3. Post-action re-perception — close the feedback loop so calibration
            # judges the *real* effect, not the description of intent.
            observation = self._post_action_observation(step, tool_result, pre_observation)
            if observation is not pre_observation:
                state.observations.append(observation)
                self._emit(state, "workflow.perceived", observation.model_dump() | {"phase": "post_action"})

            # 4. Calibrate against the rubric on the step. We only apply the
            # strict rubric when we actually have post-action evidence (a tool
            # result or external observation); otherwise the synthetic
            # pre-action observation would over-constrain calibration.
            has_real_evidence = tool_result is not None or observation is not pre_observation
            result = self.calibration.evaluate(step, observation, apply_rubric=has_real_evidence)
            ok = result.ok
            step.status = "completed" if ok else "failed"
            self._emit(
                state,
                "workflow.calibrated",
                {"step_id": step.id, "ok": ok, "rationale": result.rationale},
            )

            self.memory.short_term.record_decision(f"{step.description}: {step.status}")
            self.memory.short_term.record_calibration_point(
                f"{step.description}: calibration={'passed' if ok else 'failed'}"
                f" ({result.rationale}); observation={observation.summary}"
            )
            state.messages.append(Message(role=Role.ASSISTANT, content=f"{step.description}: {step.status}"))
            compression = self.memory.end_iteration(observations=state.observations, messages=state.messages)
            if self.tracer:
                self.tracer.record_memory_compression(
                    summary_chars=len(compression.compressed_summary),
                    cleared_refs=compression.cleared_refs,
                    retained_items=len(compression.retained_items),
                )
            state.iteration += 1
            if self.tracer:
                self.tracer.record_state_update(before_iteration, self._state_snapshot(state), label="workflow_iteration")

            # 5. On failure, try to revise the plan instead of halting outright.
            if not ok:
                recovery = self.planner.revise(state.plan, step, result.rationale)
                if recovery is None:
                    state.done = True
                    state.outcome = (
                        f"Stopped after failed calibration for step {step.id}: {result.rationale}"
                    )
                else:
                    self._emit(
                        state,
                        "workflow.planned",
                        {"revised": True, "added_step": recovery.description, "after": step.id},
                    )

        if state.outcome is None:
            state.outcome = "Stopped by loop controller."
        self._emit(state, "session.completed", {"outcome": state.outcome})
        if self.tracer:
            self.tracer.complete_run(state.outcome, {"iterations": state.iteration})
        return state

    async def _maybe_act(self, step, state: AgentState) -> ToolResult | None:
        if self.action_runner is None or step.tool_call is None:
            return None
        result = await self.action_runner.act(step)
        if result is not None:
            self._emit(state, "workflow.acted", {"step_id": step.id, "tool": step.tool_call.name, "ok": result.ok})
        return result

    def _post_action_observation(
        self,
        step,
        tool_result: ToolResult | None,
        fallback: Observation,
    ) -> Observation:
        if tool_result is None:
            return fallback
        signals: list[str] = []
        if tool_result.error:
            signals.append(f"error: {tool_result.error}")
        if tool_result.artifacts:
            signals.extend(f"artifact: {ref}" for ref in tool_result.artifacts[:4])
        summary = (
            f"Tool {tool_result.name} {'succeeded' if tool_result.ok else 'failed'}: "
            f"{tool_result.output[:200]}"
        )
        return Observation(
            summary=summary.strip(),
            signals=signals,
            source="tool_result",
            metadata={"step_id": step.id, "call_id": tool_result.call_id, "ok": tool_result.ok},
        )

    def _emit(self, state: AgentState, event_type: str, payload: dict) -> None:
        if self.audit:
            self.audit.write(HarnessEvent(type=event_type, session_id=state.session_id, payload=payload))

    def _state_snapshot(self, state: AgentState) -> dict:
        return {
            "session_id": state.session_id,
            "goal": state.goal,
            "iteration": state.iteration,
            "done": state.done,
            "outcome": state.outcome,
            "observations": len(state.observations),
            "messages": len(state.messages),
            "plan_steps": len(state.plan.steps) if state.plan else 0,
        }
