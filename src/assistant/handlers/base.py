"""Контракт сценария. Оркестратор выбирает handler по виду темы и отдаёт ему ход"""

from typing import Protocol

from assistant.core.models import CallSession, Reply


class Handler(Protocol):
    async def handle(self, session: CallSession, text: str) -> Reply:
        """Один ход внутри сценария. Сессию мутирует, реплику возвращает

        Первый вход в сценарий — когда session.state == TOPIC (тему только что определили,
        text — та самая реплика, из которой её определили)
        """
        ...