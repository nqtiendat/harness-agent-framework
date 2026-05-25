"""FastAPI app factory."""

from __future__ import annotations

from agent_harness.service.schemas import RunRequest, RunResponse


def create_app():
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install agent-harness-framework[service] to use the API service") from exc

    app = FastAPI(title="Agent Harness Framework")

    @app.post("/runs", response_model=RunResponse)
    async def run_agent(request: RunRequest) -> RunResponse:
        return RunResponse(session_id="not-started", outcome=f"Accepted goal: {request.goal}")

    return app

