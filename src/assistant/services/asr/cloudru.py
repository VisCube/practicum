"""Cloud.ru Whisper ASR поверх `_upstream/src/transcribe.py`.

Демо-стенд шлёт сырой поток `audio/pcm` (LPCM 16-bit mono, 16 kHz без заголовка),
а функция `transcribe(data: bytes)` во внешнем модуле передаёт байты как `audio.wav`.
Адаптер на лету оборачивает сырой PCM в корректный RIFF/WAVE буфер в памяти.
"""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import Callable

from assistant.core.models import Audio
from assistant.nlu.itn import numbers_to_digits
from assistant.services.asr.base import AsrError
from assistant.services.cloudru.bridge import load_transcribe


def pcm16_to_wav(pcm: bytes, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Оборачивает 16-bit signed little-endian PCM в WAV-контейнер в памяти."""
    aligned = pcm[: len(pcm) & ~1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(aligned)
    return buf.getvalue()


def _is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


class CloudRuAsr:
    """Адаптер `SpeechRecognizer` к облачной модели STT в Cloud.ru."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        timeout: float | None = 10.0,
        use_itn: bool = True,
        transcribe_fn: Callable[[bytes], str] | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.timeout = timeout
        self.use_itn = use_itn
        self._transcribe_fn = transcribe_fn or load_transcribe()

    async def transcribe(self, audio: Audio) -> str:
        if audio.empty:
            return ""

        wav_bytes = self._prepare_wav(audio)
        if not wav_bytes:
            return ""

        try:
            coro = asyncio.to_thread(self._transcribe_fn, wav_bytes)
            raw_text = (
                await asyncio.wait_for(coro, timeout=self.timeout)
                if self.timeout
                else await coro
            )
        except AsrError:
            raise
        except Exception as exc:
            raise AsrError(f"cloudru transcribe failed: {exc}") from exc

        text = (raw_text or "").strip()
        return numbers_to_digits(text) if (text and self.use_itn) else text

    def _prepare_wav(self, audio: Audio) -> bytes:
        mime_lower = audio.mime.lower()
        if _is_wav(audio.data):
            return audio.data
        if any(marker in mime_lower for marker in ("pcm", "l16", "raw")):
            if len(audio.data) < 2:
                return b""
            return pcm16_to_wav(audio.data, sample_rate=self.sample_rate)
        raise AsrError(f"CloudRuAsr accepts lpcm or wav, got {audio.mime}")
