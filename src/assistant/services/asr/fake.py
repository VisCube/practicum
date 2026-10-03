"""ASR-заглушка: возвращает заданный текст, аудио игнорирует"""

from assistant.core.models import Audio


class FakeAsr:
    """Заглушка ASR для тестов: возвращает заданный текст, сохраняет полученные фрагменты"""
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.received: list[Audio] = []

    async def transcribe(self, audio: Audio) -> str:
        self.received.append(audio)
        return self.text