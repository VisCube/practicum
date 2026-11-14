"""ASR-заглушка: возвращает заданный текст, аудио игнорирует."""

from assistant.core.models import Audio


class FakeAsr:
    """Заглушка ASR для тестов: возвращает заданный текст, сохраняет последние фрагменты."""

    def __init__(self, text: str = "", *, max_history: int = 200) -> None:
        self.text = text
        self.received: list[Audio] = []
        self._max_history = max_history

    async def transcribe(self, audio: Audio) -> str:
        self.received.append(audio)
        if self._max_history > 0 and len(self.received) > self._max_history:
            del self.received[: len(self.received) - self._max_history]
        return self.text
