"""Порт управления звонком. В request/response-модели платформа сама читает флаги из ответа,
клиент нужен для действий «вне хода»: перевод на оператора, завершение"""

from typing import Protocol


class TelephonyClient(Protocol):
    """"Контракт для управления звонком вне хода диалога: перевод и завершение"""

    async def transfer_to_operator(self, call_id: str, reason: str) -> None: ...
    """Переводит звонок на живого оператора с указанием причины"""

    async def hangup(self, call_id: str) -> None: ...
    """Завершает звонок со стороны платформы"""