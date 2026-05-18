import json
import re

from .models import LLMClient
from .prompts import ATTACKER_SYSTEM_PROMPT


def _parse_attacker_response(text: str) -> tuple[str, str]:
    # Direct JSON parse
    try:
        data = json.loads(text.strip())
        return data.get("improvement", ""), data.get("prompt", text)
    except (json.JSONDecodeError, AttributeError):
        pass

    # JSON inside a markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("improvement", ""), data.get("prompt", text)
        except json.JSONDecodeError:
            pass

    # Any JSON object in the text
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("improvement", ""), data.get("prompt", text)
        except json.JSONDecodeError:
            pass

    return "", text


class Attacker:
    def __init__(self, client: LLMClient, goal: str) -> None:
        self._client = client
        self._goal = goal
        self._system = ATTACKER_SYSTEM_PROMPT.format(goal=goal)

    async def generate(self, messages: list[dict]) -> tuple[str, str]:
        """Send attacker conversation to the LLM. Returns (improvement, prompt)."""
        raw = await self._client.chat(messages, self._system)
        return _parse_attacker_response(raw)
