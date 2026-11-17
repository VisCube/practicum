"""Порт LLM. Используется для фолбэка классификации, разбора слотов и ответов по базе знаний"""

from typing import Protocol


class LlmError(Exception):
    pass


class LlmClient(Protocol):
    """Контракт для LLM-адаптеров: один запрос - один ответ"""
    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Один запрос-ответ. json_mode — просим модель вернуть валидный JSON-объект"""
        ...
