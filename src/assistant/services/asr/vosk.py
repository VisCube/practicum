"""Vosk — оффлайн ASR на CPU. Принимает ровно то, что шлёт стенд: lpcm 16 kHz mono"""

import asyncio
import json

from assistant.core.models import Audio
from assistant.services.asr.base import AsrError


class VoskAsr:
    """Оффлайн ASR на CPU через Vosk - принимает только lpcm 16 kHz mono"""
    def __init__(self, model_path: str, *, sample_rate: int = 16000) -> None:
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        self.model = Model(model_path)
        self.sample_rate = sample_rate

    async def transcribe(self, audio: Audio) -> str:
        if audio.empty:
            return ""
        mime_lower = audio.mime.lower()
        if not any(pcm_marker in mime_lower for pcm_marker in ("pcm", "l16", "raw")):
            raise AsrError(f"Vosk принимает только lpcm, получен {audio.mime}")
        return await asyncio.to_thread(self._recognize, audio.data)

    def _recognize(self, data: bytes) -> str:
        """Синхронное распознавание - вызывается через asyncio.to_thread"""
        from vosk import KaldiRecognizer
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.AcceptWaveform(data)
        return json.loads(recognizer.FinalResult()).get("text", "").strip()