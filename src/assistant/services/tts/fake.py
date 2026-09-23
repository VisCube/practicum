"""TTS-заглушка: аудио не делает, текст запоминает. Фронт озвучит через Web Speech API"""

from assistant.core.models import Audio


class FakeTts:
    """Заглушка TTS - аудио не синтезирует, текст запоминает для проверки в тестах"""
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> Audio:
        self.spoken.append(text)
        return Audio()