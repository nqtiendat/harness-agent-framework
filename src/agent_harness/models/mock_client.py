"""Deterministic model client for tests and local smoke runs."""

from __future__ import annotations

from agent_harness.core.types import ModelRequest, ModelResponse
from agent_harness.models.base import ModelClient


class MockModelClient(ModelClient):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        last = request.messages[-1].content if request.messages else ""
        return ModelResponse(content=f"Mock response for: {last[:120]}")

