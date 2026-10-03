"""Порт базы знаний"""

from typing import Protocol

from assistant.core.models import Passage


class KnowledgeBase(Protocol):
    """Контракт для адаптеров базы знаний"""
    async def search(self, query: str, *, sources: list[str] | None = None,
                     limit: int = 3) -> list[Passage]:
        """Релевантные фрагменты. sources - имена файлов без расширения для сужения поиска"""
        ...