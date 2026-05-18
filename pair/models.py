from abc import ABC, abstractmethod

import anthropic


class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], system: str = "") -> str: ...


class AnthropicClient(LLMClient):
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = anthropic.AsyncAnthropic()

    async def chat(self, messages: list[dict], system: str = "") -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": messages,
        }
        if system:
            # Cache the system prompt — it's static per run and reused across all streams.
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        response = await self._client.messages.create(**kwargs)
        return response.content[0].text


def make_client(provider: str, model: str) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model)
    raise ValueError(f"Unsupported provider: {provider!r}")
