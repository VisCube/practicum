"""T-one ASR через sherpa-onnx (offline). Вход как у стенда: lpcm 16 kHz mono → внутри 8 kHz"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

from assistant.core.models import Audio
from assistant.nlu.itn import numbers_to_digits
from assistant.services.asr.base import AsrError

logger = logging.getLogger(__name__)

_TONE_SR = 8000  # натив T-one


class ToneAsr:
    """Оффлайн ASR: sherpa-onnx streaming T-one CTC. Контракт SpeechRecognizer"""

    def __init__(
            self,
            model_dir: str | Path = "models/t-one",
            *,
            input_sample_rate: int = 16000,
            num_threads: int = 2,
    ) -> None:
        model_dir = Path(model_dir)
        model_path = model_dir / "model.onnx"
        tokens_path = model_dir / "tokens.txt"
        if not model_path.is_file():
            raise FileNotFoundError(f"T-one model.onnx not found: {model_path}")
        if not tokens_path.is_file():
            raise FileNotFoundError(f"T-one tokens.txt not found: {tokens_path}")

        try:
            import sherpa_onnx
        except ImportError as e:
            raise AsrError(
                "sherpa-onnx not installed. pip install sherpa-onnx"
            ) from e

        self.input_sample_rate = int(input_sample_rate)
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_t_one_ctc(
            tokens=str(tokens_path),
            model=str(model_path),
            num_threads=num_threads,
            sample_rate=int(_TONE_SR),
            feature_dim=80,
            debug=False,
        )
        logger.info(
            "T-one ASR ready (sherpa-onnx, model=%s, input_sr=%d → %d)",
            model_path, self.input_sample_rate, _TONE_SR,
        )

    async def transcribe(self, audio: Audio) -> str:
        if audio.empty:
            return ""
        mime_lower = audio.mime.lower()
        if not any(m in mime_lower for m in ("pcm", "l16", "raw")):
            raise AsrError(f"T-one accepts lpcm only, got {audio.mime}")
        try:
            return numbers_to_digits(await asyncio.to_thread(self._recognize, audio.data))
        except AsrError:
            raise
        except Exception as e:
            raise AsrError(f"tone recognize failed: {e}") from e

    def _recognize(self, data: bytes) -> str:
        if len(data) < 2:
            return ""

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return ""

        if self.input_sample_rate != _TONE_SR:
            samples = _resample_linear(samples, self.input_sample_rate, _TONE_SR)

        stream = self._recognizer.create_stream()
        # sherpa принимает float32 mono
        stream.accept_waveform(_TONE_SR, samples)
        tail = np.zeros(int(_TONE_SR * 0.8), dtype=np.float32)  # ~0.5 s silence flush
        stream.accept_waveform(_TONE_SR, tail)
        stream.input_finished()

        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        text = self._recognizer.get_result(stream)
        return (text or "").strip()


def _resample_linear(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or x.size == 0:
        return x
    n_out = max(1, int(round(x.size * sr_to / sr_from)))
    t_old = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)