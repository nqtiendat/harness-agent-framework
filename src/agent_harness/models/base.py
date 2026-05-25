"""Foundation model abstraction."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from agent_harness.core.types import ModelRequest, ModelResponse
from agent_harness.evaluation.tracer import ErrorSource, ExecutionTracer


class ModelClient(ABC):
    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a model response for the supplied conversation state."""


class TracedModelClient(ModelClient):
    """Wraps any ModelClient and emits model.* events + ErrorSource.MODEL on failure.

    Use this so the tracer can separate model-side failures (timeouts,
    hallucinated tool calls, API errors) from harness-side failures.
    """

    def __init__(self, inner: ModelClient, tracer: ExecutionTracer, model_name: str | None = None) -> None:
        self.inner = inner
        self.tracer = tracer
        self.model_name = model_name or inner.__class__.__name__

    async def complete(self, request: ModelRequest) -> ModelResponse:
        span = self.tracer.record_model_call_started(
            model=self.model_name,
            request_summary={
                "messages": len(request.messages),
                "tools": len(request.tools),
                "last_role": request.messages[-1].role.value if request.messages else None,
            },
        )
        started = time.perf_counter()
        try:
            response = await self.inner.complete(request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            self.tracer.record_model_call_completed(
                model=self.model_name,
                span_id=span,
                latency_ms=latency_ms,
                error=str(exc),
            )
            self.tracer.record_error(
                source=ErrorSource.MODEL,
                message=str(exc),
                exception_type=type(exc).__name__,
                span_id=span,
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        self.tracer.record_model_call_completed(
            model=self.model_name,
            span_id=span,
            usage=response.usage,
            latency_ms=latency_ms,
            finish_reason=response.metadata.get("finish_reason") if response.metadata else None,
        )
        return response
