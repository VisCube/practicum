"""Silero TTS.
Пайплайн: rutextnorm → silero-stress → SSML(rate/pitch) → apply_tts
Повторы — LRU-кэш"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import wave
from collections import OrderedDict
from pathlib import Path

import torch

from assistant.core.models import Audio
from assistant.services.tts.base import TtsError, SpeechSynthesizer

logger = logging.getLogger(__name__)

_STRESS_MARK = re.compile(r"\+|[\u0301\u0341]")
_VALID_RATES = frozenset({"x-slow", "slow", "medium", "fast", "x-fast"})
_VALID_PITCHES = frozenset({"x-low", "low", "medium", "high", "x-high"})


def _discover_speakers(model: object) -> list[str]:
    for attr in ("speakers", "speaker", "spk_labels"):
        value = getattr(model, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            return sorted(str(k) for k in value.keys())
        if isinstance(value, (list, tuple, set)):
            return sorted(str(s) for s in value)
    return []


def _pick_default_speaker(speakers: list[str]) -> str:
    if not speakers:
        raise TtsError("Silero model has no speakers list")
    for preferred in (
            "ru_eduard", "ru_xenia", "xenia", "ru_aidar", "aidar",
            "ru_ekaterina", "ru_dmitriy", "ru_roman",
    ):
        if preferred in speakers:
            return preferred
    ru = [s for s in speakers if s.startswith("ru_")]
    return sorted(ru)[0] if ru else speakers[0]


def _load_normalizer():
    try:
        from rutextnorm import normalize_russian
        return normalize_russian
    except ImportError:
        logger.warning("rutextnorm not installed — normalize off")
        return None


def _load_accentor(device: str):
    try:
        from silero_stress import load_accentor
        accentor = load_accentor(lang="ru")
        if hasattr(accentor, "to"):
            accentor.to(device=device)
        return accentor
    except Exception as e:
        logger.warning("silero-stress unavailable (%s) — stress off", e)
        return None


class SileroTts(SpeechSynthesizer):
    """Оффлайн TTS. Контракт SpeechSynthesizer: text → wav"""

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
            cache_size: int = 64,
            num_threads: int | None = 4,
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
        self.use_ssml = use_ssml
        self.rate = rate
        self.pitch = pitch
        self.device = torch.device(device)

        if device == "cpu" and num_threads is not None:
            torch.set_num_threads(num_threads)

        logger.info(
            "Loading Silero TTS from %s (sr=%d, device=%s, rate=%s, pitch=%s)",
            model_path, self.sample_rate, device, rate, pitch,
        )
        try:
            importer = torch.package.PackageImporter(str(model_path))
            self.model = importer.load_pickle("tts_models", "model")
            if hasattr(self.model, "to"):
                self.model.to(self.device)
        except Exception as e:
            raise TtsError(f"failed to load Silero model: {e}") from e

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
        self._accentor = _load_accentor(device) if stress else None

        if self._normalize_fn:
            logger.info("TTS preprocess: rutextnorm ON")
        if self._accentor is not None:
            logger.info("TTS preprocess: silero-stress ON")
        if self.use_ssml:
            logger.info("TTS SSML: rate=%s pitch=%s", self.rate, self.pitch)

        self._cache: OrderedDict[str, Audio] = OrderedDict()
        self._cache_size = cache_size
        logger.info("Silero TTS ready (speaker=%s)", self.speaker)

    def _probe_apply_tts_kwargs(self) -> bool:
        import inspect
        try:
            params = inspect.signature(self.model.apply_tts).parameters
            return "put_accent" in params or "put_yo" in params
        except (TypeError, ValueError):
            return False

    def _prepare_text(self, text: str) -> str:
        out = text.strip()
        if not out:
            return out

        # уже полный SSML — не нормализуем/не stress'им целиком
        if out.lstrip().lower().startswith("<speak"):
            return out

        if self._normalize_fn is not None:
            try:
                out = self._normalize_fn(out)
            except Exception as e:
                logger.warning("rutextnorm failed: %s", e)

        if self._accentor is not None and not _STRESS_MARK.search(out):
            try:
                out = self._accentor(
                    out,
                    put_stress=True,
                    put_yo=True,
                    put_stress_homo=True,
                    put_yo_homo=True,
                )
            except TypeError:
                try:
                    out = self._accentor(out)
                except Exception as e:
                    logger.warning("silero-stress failed: %s", e)
            except Exception as e:
                logger.warning("silero-stress failed: %s", e)

        return out

    def _wrap_ssml(self, text: str) -> str:
        t = text.strip()
        if not t:
            return t
        if t.lstrip().lower().startswith("<speak"):
            return t
        esc = (
            t.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return (
            f'<speak><prosody rate="{self.rate}" pitch="{self.pitch}">'
            f"{esc}</prosody></speak>"
        )

    async def synthesize(self, text: str, *, speaker: str | None = None) -> Audio:
        raw = text.strip()
        if not raw:
            return Audio()

        use_speaker = speaker or self.speaker
        if self.speakers and use_speaker not in self.speakers:
            raise TtsError(
                f"speaker {use_speaker!r} not in model. "
                f"Available: {', '.join(self.speakers)}"
            )

        prepared = self._prepare_text(raw)
        cache_key = f"{use_speaker}::{self.rate}::{self.pitch}::{prepared}"
        if (cached := self._cache.get(cache_key)) is not None:
            self._cache.move_to_end(cache_key)
            return cached

        try:
            audio = await asyncio.to_thread(self._synth, prepared, use_speaker)
        except Exception as e:
            raise TtsError(f"silero synthesize failed: {e}") from e

        self._cache[cache_key] = audio
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return audio

    def _synth(self, text: str, speaker: str) -> Audio:
        with torch.inference_mode():
            audio_tensor = self._apply_tts(text, speaker)

        wav = audio_tensor.detach().cpu().float().squeeze()
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        pcm = (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)

        return Audio(data=buf.getvalue(), mime="audio/wav")

    def _apply_tts(self, text: str, speaker: str):
        """SSML first; fallback to plain text if model rejects ssml_text."""
        if self.use_ssml:
            try:
                return self.model.apply_tts(
                    ssml_text=self._wrap_ssml(text),
                    speaker=speaker,
                    sample_rate=self.sample_rate,
                )
            except TypeError:
                logger.warning("model has no ssml_text — SSML disabled")
                self.use_ssml = False

        kwargs: dict = {
            "text": text,
            "speaker": speaker,
            "sample_rate": self.sample_rate,
        }
        if self._supports_accent:
            kwargs["put_accent"] = self.put_accent
            kwargs["put_yo"] = self.put_yo
        try:
            return self.model.apply_tts(**kwargs)
        except TypeError:
            kwargs.pop("put_accent", None)
            kwargs.pop("put_yo", None)
            self._supports_accent = False
            return self.model.apply_tts(**kwargs)