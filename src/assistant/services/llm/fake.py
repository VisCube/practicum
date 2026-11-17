"""LLM-заглушка: отвечает по сценарию или фиксированной строкой"""

from collections.abc import Callable


class FakeLlm:
    """Заглушка LLM - отвечает фиксированной строкой или вычисляет ответ через callable"""
    def __init__(self, reply: str | Callable[[str, str], str] = "{}") -> None:
        self._reply = reply
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append((system, user))
        return self._reply(system, user) if callable(self._reply) else self._reply