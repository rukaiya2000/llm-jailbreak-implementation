import os
from abc import ABC, abstractmethod

import openai


class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], system: str = "") -> str: ...


class OpenAIClient(LLMClient):
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = openai.AsyncOpenAI()

    async def chat(self, messages: list[dict], system: str = "") -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_completion_tokens=1024,
        )
        return response.choices[0].message.content


class NavigatorClient(LLMClient):
    """LiteLLM proxy at UF AI Navigator — OpenAI-compatible endpoint."""

    _BASE_URL = "https://api.ai.it.ufl.edu"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = openai.AsyncOpenAI(
            api_key=os.environ.get("NAVIGATOR_API_KEY", ""),
            base_url=self._BASE_URL,
        )

    async def chat(self, messages: list[dict], system: str = "") -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=1024,
        )
        return response.choices[0].message.content


def make_client(provider: str, model: str) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(model)
    if provider == "navigator":
        return NavigatorClient(model)
    raise ValueError(f"Unsupported provider: {provider!r}")
