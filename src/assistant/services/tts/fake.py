"""TTS-заглушка: аудио не делает, текст запоминает. Фронт озвучит через Web Speech API."""

from assistant.core.models import Audio


class FakeTts:
    """Заглушка TTS — аудио не синтезирует, текст запоминает для проверки в тестах."""

    def __init__(self, *, max_history: int = 200) -> None:
        self.spoken: list[str] = []
        self._max_history = max_history

    async def synthesize(self, text: str, speaker: str | None = None) -> Audio:
        self.spoken.append(text)
        if self._max_history > 0 and len(self.spoken) > self._max_history:
            del self.spoken[: len(self.spoken) - self._max_history]
        return Audio()
