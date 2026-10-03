"""Телефония-заглушка: журналирует действия"""


class FakeTelephony:
    """Заглушка телефонии - записывает переводы и завершения для проверки в тестах"""
    def __init__(self) -> None:
        self.transfers: list[tuple[str, str]] = []
        self.hangups: list[str] = []

    async def transfer_to_operator(self, call_id: str, reason: str) -> None:
        self.transfers.append((call_id, reason))

    async def hangup(self, call_id: str) -> None:
        self.hangups.append(call_id)