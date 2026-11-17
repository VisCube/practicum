"""Vosk — оффлайн ASR на CPU. Принимает ровно то, что шлёт стенд: lpcm 16 kHz mono."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Executor

from assistant.core.models import Audio
from assistant.nlu.itn import numbers_to_digits
from assistant.services.asr.base import AsrError


class VoskAsr:
    """Оффлайн ASR на CPU через Vosk — принимает только lpcm 16 kHz mono."""

    def __init__(
        self,
        model_path: str,
        *,
        sample_rate: int = 16000,
        use_itn: bool = True,
        timeout: float | None = None,
        executor: Executor | None = None,
    ) -> None:
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.use_itn = use_itn
        self.timeout = timeout
        self._executor = executor

    async def transcribe(self, audio: Audio) -> str:
        if audio.empty:
            return ""
        mime_lower = audio.mime.lower()
        if not any(pcm_marker in mime_lower for pcm_marker in ("pcm", "l16", "raw")):
            raise AsrError(f"Vosk принимает только lpcm, получен {audio.mime}")
        try:
            loop = asyncio.get_running_loop()
            coro = loop.run_in_executor(self._executor, self._recognize, audio.data)
            text = (
                await asyncio.wait_for(coro, timeout=self.timeout) if self.timeout else await coro
            )
            return numbers_to_digits(text) if (text and self.use_itn) else text
        except AsrError:
            raise
        except Exception as exc:
            raise AsrError(f"vosk recognize failed: {exc}") from exc

    def _recognize(self, data: bytes) -> str:
        """Синхронное распознавание — вызывается через пул потоков."""
        from vosk import KaldiRecognizer

        aligned = data[: len(data) & ~1]
        if len(aligned) < 2:
            return ""
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.AcceptWaveform(aligned)
        return str(json.loads(recognizer.FinalResult()).get("text", "")).strip()
