"""Silero TTS.

Пайплайн внутри пула воркеров (без блокировки asyncio event loop):
  clean_for_tts → rutextnorm → expand_remaining_digits → silero-stress → SSML → apply_tts
Кэширование двухуровневое:
  1) LRU полных реплик по сырому ключу (хит за ~0 мс без нормализации и акцентора);
  2) потокобезопасный LRU отдельных предложений (общие хвосты вроде «Могу ещё чем-то помочь?»
     не синтезируются повторно даже рядом с динамическим номером заявки).
Под параллельные звонки поддерживается пул независимых инстансов (`workers >= 1`).
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import re
import threading
import wave
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from assistant.core.models import Audio
from assistant.services.tts.base import SpeechSynthesizer, TtsError
from assistant.services.tts.normalize import (
    clean_for_tts,
    expand_remaining_digits,
    split_tts_sentences,
)

logger = logging.getLogger(__name__)

_STRESS_MARK = re.compile(r"\+|[\u0301\u0341]")
_VALID_RATES = frozenset({"x-slow", "slow", "medium", "fast", "x-fast"})
_VALID_PITCHES = frozenset({"x-low", "low", "medium", "high", "x-high"})


@dataclass
class _SileroWorker:
    model: Any
    accentor: Any


def _discover_speakers(model: object) -> list[str]:
    for attr in ("speakers", "speaker", "spk_labels"):
        value = getattr(model, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            return sorted(str(k) for k in value)
        if isinstance(value, list | tuple | set):
            return sorted(str(s) for s in value)
    return []


def _pick_default_speaker(speakers: list[str]) -> str:
    if not speakers:
        raise TtsError("Silero model has no speakers list")
    for preferred in (
        "ru_eduard",
        "ru_xenia",
        "xenia",
        "ru_aidar",
        "aidar",
        "ru_ekaterina",
        "ru_dmitriy",
        "ru_roman",
    ):
        if preferred in speakers:
            return preferred
    ru = [s for s in speakers if s.startswith("ru_")]
    return sorted(ru)[0] if ru else speakers[0]


def _load_normalizer() -> Callable[[str], str] | None:
    try:
        from rutextnorm import normalize_russian

        fn: Callable[[str], str] = normalize_russian
        return fn
    except ImportError:
        logger.warning("rutextnorm not installed — using built-in TTS normalizer")
        return None


def _load_accentor(device: str) -> Any:
    try:
        from silero_stress import load_accentor

        accentor = load_accentor(lang="ru")
        if hasattr(accentor, "to"):
            accentor.to(device=device)
        return accentor
    except Exception as e:
        logger.warning("silero-stress unavailable (%s) — stress off", e)
        return None


def _load_silero_model(model_path: Path, device: Any) -> Any:
    try:
        importer = torch.package.PackageImporter(str(model_path))
        model = importer.load_pickle("tts_models", "model")
        if hasattr(model, "to"):
            model.to(device)
        return model
    except Exception as e:
        raise TtsError(f"failed to load Silero model: {e}") from e


class SileroTts(SpeechSynthesizer):
    """Оффлайн TTS. Контракт SpeechSynthesizer: text → wav."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        speaker: str | None = None,
        sample_rate: int = 24000,
        device: str = "cpu",
        put_accent: bool = True,
        put_yo: bool = True,
        normalize: bool = True,
        stress: bool = True,
        use_ssml: bool = True,
        rate: str = "fast",
        pitch: str = "medium",
        cache_size: int = 128,
        num_threads: int | None = 1,
        workers: int = 1,
        timeout: float | None = None,
        executor: Executor | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Silero model not found: {model_path}")

        if rate not in _VALID_RATES:
            raise ValueError(f"rate must be one of {sorted(_VALID_RATES)}, got {rate!r}")
        if pitch not in _VALID_PITCHES:
            raise ValueError(f"pitch must be one of {sorted(_VALID_PITCHES)}, got {pitch!r}")

        self.sample_rate = int(sample_rate)
        self.put_accent = put_accent
        self.put_yo = put_yo
        self.normalize = normalize
        self.use_ssml = use_ssml
        self.rate = rate
        self.pitch = pitch
        self.timeout = timeout
        self.device = torch.device(device)
        worker_count = max(1, int(workers))

        logger.info(
            "Loading Silero TTS from %s (sr=%d, device=%s, rate=%s, pitch=%s, workers=%d)",
            model_path,
            self.sample_rate,
            device,
            rate,
            pitch,
            worker_count,
        )

        self._workers: queue.SimpleQueue[_SileroWorker] = queue.SimpleQueue()
        first_model = _load_silero_model(model_path, self.device)
        self.model = first_model

        self.speakers = _discover_speakers(self.model)
        if self.speakers:
            logger.info(
                "Silero speakers (%d): %s%s",
                len(self.speakers),
                ", ".join(self.speakers[:12]),
                "..." if len(self.speakers) > 12 else "",
            )
        else:
            logger.warning("could not discover speakers; validation disabled")

        if speaker:
            if self.speakers and speaker not in self.speakers:
                raise ValueError(
                    f"speaker {speaker!r} not in model. Available: {', '.join(self.speakers)}"
                )
            self.speaker = speaker
        else:
            self.speaker = _pick_default_speaker(self.speakers) if self.speakers else "xenia"

        self._supports_accent = self._probe_apply_tts_kwargs()
        self._normalize_fn = _load_normalizer() if normalize else None
        first_accentor = _load_accentor(device) if stress else None
        self._accentor = first_accentor
        self._workers.put(_SileroWorker(model=first_model, accentor=first_accentor))

        for _ in range(worker_count - 1):
            extra_model = _load_silero_model(model_path, self.device)
            extra_accentor = _load_accentor(device) if stress else None
            self._workers.put(_SileroWorker(model=extra_model, accentor=extra_accentor))

        # Важно: silero_stress при импорте вызывает torch.set_num_threads(1),
        # поэтому выставляем потоки строго ПОСЛЕ загрузки всех акценторов.
        if device == "cpu" and num_threads is not None:
            torch.set_num_threads(num_threads)

        if self._normalize_fn:
            logger.info("TTS preprocess: domain + rutextnorm ON")
        elif self.normalize:
            logger.info("TTS preprocess: domain normalizer ON")
        if self._accentor is not None:
            logger.info("TTS preprocess: silero-stress ON")
        if self.use_ssml:
            logger.info("TTS SSML: rate=%s pitch=%s", self.rate, self.pitch)

        self._cache: OrderedDict[str, Audio] = OrderedDict()
        self._spoken_cache: OrderedDict[str, str] = OrderedDict()
        self._sentence_pcm_cache: OrderedDict[str, bytes] = OrderedDict()
        self._sentence_lock = threading.Lock()
        self._cache_size = cache_size
        self._inflight: dict[str, asyncio.Future[tuple[Audio, str]]] = {}

        self._own_executor: ThreadPoolExecutor | None = None
        if executor is not None:
            self._executor: Executor | None = executor
        else:
            self._own_executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="silero-tts",
            )
            self._executor = self._own_executor
        logger.info("Silero TTS ready (speaker=%s, workers=%d)", self.speaker, worker_count)

    async def aclose(self) -> None:
        if self._own_executor is not None:
            self._own_executor.shutdown(wait=False, cancel_futures=True)
            self._own_executor = None

    def _probe_apply_tts_kwargs(self) -> bool:
        import inspect

        try:
            params = inspect.signature(self.model.apply_tts).parameters
            return "put_accent" in params or "put_yo" in params
        except (TypeError, ValueError):
            return False

    def _prepare_text_with_accentor(self, text: str, accentor: Any) -> str:
        out = text.strip()
        if not out or out.lstrip().lower().startswith("<speak"):
            return out

        if self.normalize:
            out = clean_for_tts(out, expand_digits=False)
            if self._normalize_fn is not None:
                try:
                    out = self._normalize_fn(out)
                except Exception as e:
                    logger.warning("rutextnorm failed: %s", e)
            # Гарантируем, что ни одна цифра не дойдёт до словаря Silero сырой
            out = expand_remaining_digits(out)

        if accentor is not None and not _STRESS_MARK.search(out):
            try:
                out = accentor(
                    out,
                    put_stress=True,
                    put_yo=True,
                    put_stress_homo=True,
                    put_yo_homo=True,
                )
            except TypeError:
                try:
                    out = accentor(out)
                except Exception as e:
                    logger.warning("silero-stress failed: %s", e)
            except Exception as e:
                logger.warning("silero-stress failed: %s", e)

        return out

    def _prepare_text(self, text: str) -> str:
        return self._prepare_text_with_accentor(text, self._accentor)

    def last_spoken(self, text: str) -> str:
        """Возвращает нормализованный вид текста из кэша (для /debug/tts)."""
        raw = text.strip()
        return self._spoken_cache.get(raw, raw)

    def _wrap_ssml(self, text: str) -> str:
        t = text.strip()
        if not t or t.lstrip().lower().startswith("<speak"):
            return t
        esc = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<speak><prosody rate="{self.rate}" pitch="{self.pitch}">{esc}</prosody></speak>'

    async def synthesize(self, text: str, *, speaker: str | None = None) -> Audio:
        raw = text.strip()
        if not raw:
            return Audio()

        use_speaker = speaker or self.speaker
        if self.speakers and use_speaker not in self.speakers:
            raise TtsError(
                f"speaker {use_speaker!r} not in model. Available: {', '.join(self.speakers)}"
            )

        # Ключ по СЫРОМУ тексту: кэш-хит не тратит CPU ни на rutextnorm, ни на silero-stress
        cache_key = f"{use_speaker}::{self.rate}::{self.pitch}::{raw}"
        if (cached := self._cache.get(cache_key)) is not None:
            self._cache.move_to_end(cache_key)
            return cached

        # Singleflight: параллельные звонки с одинаковой репликой ждут один синтез
        if (inflight := self._inflight.get(cache_key)) is not None:
            audio, _ = await asyncio.shield(inflight)
            return audio

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[Audio, str]] = loop.create_future()
        self._inflight[cache_key] = fut
        try:
            coro = loop.run_in_executor(self._executor, self._synth_raw, raw, use_speaker)
            audio, spoken = (
                await asyncio.wait_for(coro, timeout=self.timeout) if self.timeout else await coro
            )
        except TtsError as exc:
            fut.set_exception(exc)
            fut.exception()
            raise
        except Exception as e:
            tts_err = TtsError(f"silero synthesize failed: {e}")
            fut.set_exception(tts_err)
            fut.exception()
            raise tts_err from e
        else:
            self._cache[cache_key] = audio
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            self._spoken_cache[raw] = spoken
            if len(self._spoken_cache) > self._cache_size:
                self._spoken_cache.popitem(last=False)
            fut.set_result((audio, spoken))
            return audio
        finally:
            self._inflight.pop(cache_key, None)

    def _get_cached_sentence(self, key: str) -> bytes | None:
        with self._sentence_lock:
            pcm = self._sentence_pcm_cache.get(key)
            if pcm is not None:
                self._sentence_pcm_cache.move_to_end(key)
            return pcm

    def _put_cached_sentence(self, key: str, pcm: bytes) -> None:
        limit = self._cache_size * 2
        with self._sentence_lock:
            self._sentence_pcm_cache[key] = pcm
            if len(self._sentence_pcm_cache) > limit:
                self._sentence_pcm_cache.popitem(last=False)

    def _synth_raw(self, raw_text: str, speaker: str) -> tuple[Audio, str]:
        worker = self._workers.get()
        try:
            sentences = split_tts_sentences(raw_text, max_chars=280)
            if not sentences:
                return Audio(), ""

            pcm_parts: list[bytes] = []
            spoken_parts: list[str] = []

            for sentence in sentences:
                sent_key = f"{speaker}::{self.rate}::{self.pitch}::{sentence}"
                prepared = self._prepare_text_with_accentor(sentence, worker.accentor)
                if not prepared:
                    continue
                spoken_parts.append(prepared)

                cached_pcm = self._get_cached_sentence(sent_key)
                if cached_pcm is not None:
                    pcm_parts.append(cached_pcm)
                    continue

                with torch.inference_mode():
                    audio_tensor = self._apply_tts_on(worker.model, prepared, speaker)
                wav = audio_tensor.detach().cpu().float().squeeze()
                if wav.ndim != 1:
                    wav = wav.reshape(-1)
                sent_pcm = (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()
                self._put_cached_sentence(sent_key, sent_pcm)
                pcm_parts.append(sent_pcm)

            if not pcm_parts:
                return Audio(), " ".join(spoken_parts)

            inter_pause = b"\x00\x00" * int(self.sample_rate * 0.10)
            tail_silence = b"\x00\x00" * int(self.sample_rate * 0.16)
            full_pcm = inter_pause.join(pcm_parts) + tail_silence

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(full_pcm)

            return Audio(data=buf.getvalue(), mime="audio/wav"), " ".join(spoken_parts)
        finally:
            self._workers.put(worker)

    def _apply_tts_on(self, model: Any, text: str, speaker: str) -> Any:
        """SSML first; fallback to plain text if model rejects ssml_text."""
        if self.use_ssml:
            try:
                return model.apply_tts(
                    ssml_text=self._wrap_ssml(text),
                    speaker=speaker,
                    sample_rate=self.sample_rate,
                )
            except TypeError:
                logger.warning("model has no ssml_text — SSML disabled")
                self.use_ssml = False

        kwargs: dict[str, Any] = {
            "text": text,
            "speaker": speaker,
            "sample_rate": self.sample_rate,
        }
        if self._supports_accent:
            kwargs["put_accent"] = self.put_accent
            kwargs["put_yo"] = self.put_yo
        try:
            return model.apply_tts(**kwargs)
        except TypeError:
            kwargs.pop("put_accent", None)
            kwargs.pop("put_yo", None)
            self._supports_accent = False
            return model.apply_tts(**kwargs)
