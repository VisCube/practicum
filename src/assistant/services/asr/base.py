"""Порт распознавания речи (короткие реплики, не поток)"""

from typing import Protocol

from assistant.core.models import Audio


class AsrError(Exception):
    pass


class SpeechRecognizer(Protocol):
    """Контракт для ASR-адаптеров: принимает аудио одной реплики, возвращает текст"""
    async def transcribe(self, audio: Audio) -> str:
        """Аудио одной реплики → текст. Пустая строка — ничего не распознано"""
        ...