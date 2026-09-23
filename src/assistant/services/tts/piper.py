"""Piper — оффлайн TTS на CPU. Отдаёт wav, браузер играет нативно
Повторяющиеся фразы (приветствие, вопросы по слотам) отдаются из кэша"""

import asyncio
import io
import wave
from collections import OrderedDict

from assistant.core.models import Audio


class PiperTts:
    """Оффлайн TTS через Piper - синтезирует wav на CPU, повторные фразы отдаёт из LRU-кэша"""
    def __init__(self, model_path: str, *, cache_size: int = 64) -> None:
        from piper import PiperVoice
        self.voice = PiperVoice.load(model_path)
        # piper-tts >= 1.3 — synthesize_wav, старые — synthesize
        self._synthesize_fn = getattr(self.voice, "synthesize_wav", None) or self.voice.synthesize
        self._cache: OrderedDict[str, Audio] = OrderedDict()
        self._cache_size = cache_size

    async def synthesize(self, text: str) -> Audio:
        cache_key = text.strip()
        if not cache_key:
            return Audio()
        if (cached_audio := self._cache.get(cache_key)) is not None:
            self._cache.move_to_end(cache_key)
            return cached_audio
        audio = await asyncio.to_thread(self._synth, cache_key)
        self._cache[cache_key] = audio
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return audio

    def _synth(self, text: str) -> Audio:
        """Синхронный синтез - вызывается через asyncio.to_thread"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._synthesize_fn(text, wav_file)
        return Audio(data=wav_buffer.getvalue(), mime="audio/wav")