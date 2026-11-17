"""Cloud.ru Whisper ASR с пулом соединений, выравниванием PCM и фильтром галлюцинаций.

Демо-стенд шлёт сырой поток `audio/pcm` (LPCM 16-bit mono, 16 kHz без заголовка),
который на лету оборачивается в RIFF/WAVE буфер в памяти.
В боевом режиме используется единый `AsyncOpenAI`-клиент с keep-alive пулом сокетов
без пересоздания TCP/TLS-соединения и без блокировки рабочих потоков на каждую фразу;
инъекция синхронного `transcribe_fn` сохранена для тестов и внешней совместимости.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import re
import wave
from collections.abc import Callable
from typing import Any

from assistant.core.models import Audio
from assistant.nlu.itn import numbers_to_digits
from assistant.services.asr.base import AsrError

# Типичные русские галлюцинации Whisper на тишине, вздохах и фоновом шуме
_WHISPER_HALLUCINATION_RE = re.compile(
    r"^(?:"
    r"субтитры\s+(?:сделал|создавал|подогнал|добавил|подготовил)\b.*|"
    r"редактор\s+субтитров\b.*|"
    r"корректор\s+[а-яёa-z.\s]+|"
    r"перевод\s+и\s+озвучка\b.*|"
    r"продолжение\s+следует\.{0,3}|"
    r"спасибо\s+за\s+(?:просмотр|внимание)[!.]*|"
    r"подписывайтесь\s+на\s+(?:наш\s+)?канал[!.]*|"
    r"ставьте\s+лайк[иа]?\b.*|"
    r".*\bamara\.org\b.*|"
    r".*\bdimatorzok\b.*"
    r")$",
    re.IGNORECASE,
)


def is_whisper_hallucination(text: str) -> bool:
    """Отсекает характерные субтитровые галлюцинации Whisper на пустом звуке/шуме."""
    cleaned = text.strip().strip("«»\"' ")
    if not cleaned:
        return False
    return _WHISPER_HALLUCINATION_RE.match(cleaned) is not None


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
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        transcribe_fn: Callable[[bytes], str] | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.timeout = timeout
        self.use_itn = use_itn
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._transcribe_fn = transcribe_fn
        self._async_client: Any = None

    def _get_async_client(self) -> tuple[Any, str]:
        api_key = self._api_key or os.environ.get("FM_API_KEY", "")
        base_url = self._base_url or os.environ.get(
            "FM_BASE_URL",
            "https://foundation-models.api.cloud.ru/v1",
        )
        model = self._model or os.environ.get("STT_MODEL", "openai/whisper-large-v3")
        if not api_key:
            raise AsrError("CloudRuAsr: не задан FM_API_KEY")
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout or 10.0,
                max_retries=1,
            )
        return self._async_client, model

    async def aclose(self) -> None:
        if self._async_client is not None:
            client, self._async_client = self._async_client, None
            await client.close()

    async def warm(self) -> None:
        """Готовит клиент и канал до провайдера.

        Клиент `AsyncOpenAI` создаётся лениво, а вместе с ним на первой реплике
        случаются DNS + TCP + TLS до Cloud.ru. Здесь же открываем и само
        соединение (без запроса распознавания — тратить квоту не нужно).
        """
        client, _ = self._get_async_client()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.models.list(), timeout=self.timeout or 10.0)

    async def transcribe(self, audio: Audio) -> str:
        if audio.empty:
            return ""

        wav_bytes = self._prepare_wav(audio)
        if not wav_bytes:
            return ""

        try:
            if self._transcribe_fn is not None:
                coro = asyncio.to_thread(self._transcribe_fn, wav_bytes)
                raw_text = (
                    await asyncio.wait_for(coro, timeout=self.timeout)
                    if self.timeout
                    else await coro
                )
            else:
                client, model = self._get_async_client()
                call = client.audio.transcriptions.create(
                    model=model,
                    file=("audio.wav", wav_bytes, "audio/wav"),
                    language="ru",
                )
                resp = (
                    await asyncio.wait_for(call, timeout=self.timeout)
                    if self.timeout
                    else await call
                )
                raw_text = resp.text
        except AsrError:
            raise
        except Exception as exc:
            raise AsrError(f"cloudru transcribe failed: {exc}") from exc

        text = (raw_text or "").strip()
        if not text or is_whisper_hallucination(text):
            return ""
        return numbers_to_digits(text) if self.use_itn else text

    def _prepare_wav(self, audio: Audio) -> bytes:
        mime_lower = audio.mime.lower()
        if _is_wav(audio.data):
            return audio.data
        if any(marker in mime_lower for marker in ("pcm", "l16", "raw")):
            if len(audio.data) < 2:
                return b""
            return pcm16_to_wav(audio.data, sample_rate=self.sample_rate)
        raise AsrError(f"CloudRuAsr accepts lpcm or wav, got {audio.mime}")
