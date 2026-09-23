"""Порт синтеза речи"""

from typing import Protocol

from assistant.core.models import Audio


class TtsError(Exception):
    pass


class SpeechSynthesizer(Protocol):
    """Контракт для TTS-адаптеров: текст → аудио"""
    async def synthesize(self, text: str) -> Audio:
        """Текст → аудио. Может вернуть Audio.empty — тогда клиент озвучивает сам или показывает текст."""
        ...