"""Piper — оффлайн TTS на CPU. Отдаёт wav, браузер играет нативно.
Повторяющиеся фразы (приветствие, вопросы по слотам) отдаются из кэша."""

from __future__ import annotations

import asyncio
import io
import wave
from collections import OrderedDict
from concurrent.futures import Executor

from assistant.core.models import Audio
from assistant.services.tts.base import SpeechSynthesizer, TtsError
from assistant.services.tts.normalize import clean_for_tts


class PiperTts(SpeechSynthesizer):
    """Оффлайн TTS через Piper — синтезирует wav на CPU, повторные фразы отдаёт из LRU-кэша."""

    def __init__(
        self,
        model_path: str,
        speaker: str | None,
        *,
        normalize: bool = True,
        cache_size: int = 128,
        timeout: float | None = None,
        executor: Executor | None = None,
    ) -> None:
        from piper import PiperVoice

        self.voice = PiperVoice.load(model_path)
        # piper-tts >= 1.3 — synthesize_wav, старые — synthesize
        self._synthesize_fn = getattr(self.voice, "synthesize_wav", None) or self.voice.synthesize
        self.normalize = normalize
        self.timeout = timeout
        self._cache: OrderedDict[str, Audio] = OrderedDict()
        self._spoken_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size
        self._inflight: dict[str, asyncio.Future[tuple[Audio, str]]] = {}
        self._executor = executor

    def last_spoken(self, text: str) -> str:
        raw = text.strip()
        return self._spoken_cache.get(raw, raw)

    async def synthesize(self, text: str, *, speaker: str | None = None) -> Audio:
        raw = text.strip()
        if not raw:
            return Audio()
        if (cached_audio := self._cache.get(raw)) is not None:
            self._cache.move_to_end(raw)
            return cached_audio
        if (inflight := self._inflight.get(raw)) is not None:
            audio, _ = await asyncio.shield(inflight)
            return audio

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[Audio, str]] = loop.create_future()
        self._inflight[raw] = fut
        try:
            coro = loop.run_in_executor(self._executor, self._synth, raw)
            audio, spoken = (
                await asyncio.wait_for(coro, timeout=self.timeout) if self.timeout else await coro
            )
        except TtsError as exc:
            fut.set_exception(exc)
            fut.exception()
            raise
        except Exception as exc:
            tts_err = TtsError(f"piper synthesize failed: {exc}")
            fut.set_exception(tts_err)
            fut.exception()
            raise tts_err from exc
        else:
            self._cache[raw] = audio
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            self._spoken_cache[raw] = spoken
            if len(self._spoken_cache) > self._cache_size:
                self._spoken_cache.popitem(last=False)
            fut.set_result((audio, spoken))
            return audio
        finally:
            self._inflight.pop(raw, None)

    def _synth(self, raw_text: str) -> tuple[Audio, str]:
        """Синхронный синтез — вызывается через пул потоков."""
        prepared = clean_for_tts(raw_text, expand_digits=True) if self.normalize else raw_text
        if not prepared:
            return Audio(), ""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._synthesize_fn(prepared, wav_file)
        return Audio(data=wav_buffer.getvalue(), mime="audio/wav"), prepared
