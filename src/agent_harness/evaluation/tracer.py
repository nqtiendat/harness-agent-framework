"""Non-blocking execution tracer for agent runs."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_harness.core.types import ToolCall, ToolResult, utc_now


TraceEventType = Literal[
    "run.started",
    "run.completed",
    "state.updated",
    "tool.called",
    "tool.completed",
    "artifact.written",
    "artifact.conflict",
    "memory.compressed",
    "memory.cleared",
    "safety.violation",
    "benchmark.mutation",
    "model.call_started",
    "model.call_completed",
    "skill.loaded",
    "error",
]


class ErrorSource(str, Enum):
    """Classifies the origin of a traced failure so model vs harness errors can be separated."""

    MODEL = "model"
    TOOL_RUNTIME = "tool_runtime"
    HARNESS_GOVERNANCE = "harness_governance"
    WORKSPACE = "workspace"
    UNKNOWN = "unknown"


class TraceEvent(BaseModel):
    run_id: str
    type: TraceEventType
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)
    span_id: str | None = None
    parent_span_id: str | None = None


class ExecutionTrace(BaseModel):
    run_id: str
    events: list[TraceEvent] = Field(default_factory=list)


class ExecutionTracer:
    """Append-only trace collector with async file flushing."""

    def __init__(
        self,
        run_id: str | None = None,
        directory: str | Path = "storage/traces",
        flush_to_disk: bool = True,
    ) -> None:
        self.run_id = run_id or str(uuid4())
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush_to_disk = flush_to_disk
        self._events: list[TraceEvent] = []
        self._queue: Queue[TraceEvent | None] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None
        # Span stack: each entry is the span_id that became the implicit parent
        # of newly recorded events while it is on top.
        self._span_stack: list[str] = []
        if flush_to_disk:
            self._thread = Thread(target=self._writer_loop, name=f"trace-writer-{self.run_id}", daemon=True)
            self._thread.start()

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    # ----- span helpers -----------------------------------------------------
    def new_span(self) -> str:
        return uuid4().hex

    def push_span(self, span_id: str | None = None) -> str:
        span = span_id or self.new_span()
        self._span_stack.append(span)
        return span

    def pop_span(self) -> str | None:
        return self._span_stack.pop() if self._span_stack else None

    def current_span(self) -> str | None:
        return self._span_stack[-1] if self._span_stack else None

    # ----- recording --------------------------------------------------------
    def start_run(self, metadata: dict[str, Any] | None = None) -> None:
        self.record("run.started", metadata or {})

    def complete_run(self, outcome: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        payload = {"outcome": outcome}
        payload.update(metadata or {})
        self.record("run.completed", payload)
        self.close()

    def record(
        self,
        event_type: TraceEventType,
        payload: dict[str, Any] | None = None,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            type=event_type,
            payload=payload or {},
            span_id=span_id,
            parent_span_id=parent_span_id if parent_span_id is not None else self.current_span(),
        )
        self._events.append(event)
        if self.flush_to_disk:
            if self._thread and self._thread.is_alive():
                self._queue.put(event)
            else:
                self._write_event_sync(event)
        return event

    def record_state_update(self, before: dict[str, Any], after: dict[str, Any], label: str = "state") -> None:
        self.record(
            "state.updated",
            {
                "label": label,
                "before": self._compact(before),
                "after": self._compact(after),
                "diff": self._dict_diff(before, after),
            },
        )

    def record_tool_call(self, call: ToolCall, allowed: bool | None = None, reason: str = "") -> None:
        self.record(
            "tool.called",
            {"call": call.model_dump(), "allowed": allowed, "reason": reason},
            span_id=call.call_id,
        )

    def record_tool_result(self, result: ToolResult) -> None:
        self.record(
            "tool.completed",
            {"result": result.model_dump()},
            span_id=result.call_id,
        )

    def record_artifact(self, path: str, author: str, revision: int | None = None, artifact_type: str = "generic") -> None:
        self.record(
            "artifact.written",
            {"path": path, "author": author, "revision": revision, "artifact_type": artifact_type},
        )

    def record_memory_compression(self, summary_chars: int, cleared_refs: list[str], retained_items: int) -> None:
        self.record(
            "memory.compressed",
            {"summary_chars": summary_chars, "cleared_refs": cleared_refs, "retained_items": retained_items},
        )
        if cleared_refs:
            self.record("memory.cleared", {"refs": cleared_refs})

    def record_model_call_started(
        self,
        *,
        model: str,
        request_summary: dict[str, Any],
        span_id: str | None = None,
    ) -> str:
        span = span_id or self.new_span()
        self.record(
            "model.call_started",
            {"model": model, "request": request_summary},
            span_id=span,
        )
        return span

    def record_model_call_completed(
        self,
        *,
        model: str,
        span_id: str,
        usage: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        finish_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "model": model,
            "usage": usage or {},
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
        }
        if error:
            payload["error"] = error
        self.record("model.call_completed", payload, span_id=span_id)

    def record_skill_loaded(self, *, skill: str, command: str | None, script_sha256: str | None) -> None:
        self.record(
            "skill.loaded",
            {"skill": skill, "command": command, "script_sha256": script_sha256},
        )

    def record_error(
        self,
        *,
        source: ErrorSource | str,
        message: str,
        exception_type: str | None = None,
        context: dict[str, Any] | None = None,
        span_id: str | None = None,
    ) -> None:
        self.record(
            "error",
            {
                "source": source.value if isinstance(source, ErrorSource) else str(source),
                "message": message,
                "exception_type": exception_type,
                "context": context or {},
            },
            span_id=span_id,
        )

    def as_trace(self) -> ExecutionTrace:
        return ExecutionTrace(run_id=self.run_id, events=self.events)

    def close(self) -> None:
        if self.flush_to_disk and self._thread:
            self._queue.put(None)
            self._thread.join(timeout=2)

    def _writer_loop(self) -> None:
        path = self.directory / f"{self.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                try:
                    event = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                if event is None:
                    break
                handle.write(event.model_dump_json() + "\n")
                handle.flush()

    def _write_event_sync(self, event: TraceEvent) -> None:
        path = self.directory / f"{self.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def _dict_diff(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        keys = sorted(set(before) | set(after))
        diff = {}
        for key in keys:
            if before.get(key) != after.get(key):
                diff[key] = {"before": before.get(key), "after": after.get(key)}
        return self._compact(diff)

    def _compact(self, value: Any, limit: int = 4000) -> Any:
        encoded = json.dumps(value, default=str, ensure_ascii=True)
        if len(encoded) <= limit:
            return value
        return {"truncated": True, "preview": encoded[:limit]}
