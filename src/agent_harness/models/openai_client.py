"""Optional OpenAI model client."""

from __future__ import annotations

from agent_harness.core.types import ModelRequest, ModelResponse
from agent_harness.models.base import ModelClient


class OpenAIModelClient(ModelClient):
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model

    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install agent-harness-framework[llm] to use OpenAIModelClient") from exc

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role.value, "content": m.content} for m in request.messages],
        )
        content = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else {}
        return ModelResponse(content=content, usage=usage)

