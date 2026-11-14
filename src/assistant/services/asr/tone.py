"""T-one ASR через sherpa-onnx (offline). Вход как у стенда: lpcm 16 kHz mono → внутри 8 kHz."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from pathlib import Path

import numpy as np

from assistant.core.models import Audio
from assistant.nlu.itn import numbers_to_digits
from assistant.services.asr.base import AsrError

logger = logging.getLogger(__name__)

_TONE_SR = 8000  # натив T-one


class ToneAsr:
    """Оффлайн ASR: sherpa-onnx streaming T-one CTC. Контракт SpeechRecognizer."""

    def __init__(
        self,
        model_dir: str | Path = "models/t-one",
        *,
        input_sample_rate: int = 16000,
        num_threads: int = 1,
        use_itn: bool = True,
        timeout: float | None = None,
        executor: Executor | None = None,
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
            raise AsrError("sherpa-onnx not installed. pip install sherpa-onnx") from e

        self.input_sample_rate = int(input_sample_rate)
        self.use_itn = use_itn
        self.timeout = timeout
        self._executor = executor
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
            model_path,
            self.input_sample_rate,
            _TONE_SR,
        )

    async def transcribe(self, audio: Audio) -> str:
        if audio.empty:
            return ""
        mime_lower = audio.mime.lower()
        if not any(m in mime_lower for m in ("pcm", "l16", "raw")):
            raise AsrError(f"T-one accepts lpcm only, got {audio.mime}")
        try:
            loop = asyncio.get_running_loop()
            coro = loop.run_in_executor(self._executor, self._recognize, audio.data)
            text = (
                await asyncio.wait_for(coro, timeout=self.timeout) if self.timeout else await coro
            )
            return numbers_to_digits(text) if (text and self.use_itn) else text
        except AsrError:
            raise
        except Exception as e:
            raise AsrError(f"tone recognize failed: {e}") from e

    def _recognize(self, data: bytes) -> str:
        aligned = data[: len(data) & ~1]
        if len(aligned) < 2:
            return ""

        samples = np.frombuffer(aligned, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return ""

        if self.input_sample_rate != _TONE_SR:
            samples = _resample_antialiased(samples, self.input_sample_rate, _TONE_SR)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(_TONE_SR, samples)
        tail = np.zeros(int(_TONE_SR * 0.6), dtype=np.float32)  # flush хвоста CTC
        stream.accept_waveform(_TONE_SR, tail)
        stream.input_finished()

        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        text = self._recognizer.get_result(stream)
        return (text or "").strip()


def _resample_antialiased(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Децимация с низкочастотным усреднением окна (без алиасинга шипящих 4–8 кГц)."""
    if sr_from == sr_to or x.size == 0:
        return x
    # Типовой случай 16 кГц → 8 кГц: быстрый треугольный FIR [0.25, 0.5, 0.25] + децимация :2
    if sr_from == 2 * sr_to and x.size >= 4:
        even_len = x.size & ~1
        padded = np.empty(even_len + 2, dtype=np.float32)
        padded[0] = x[0]
        padded[1 : even_len + 1] = x[:even_len]
        padded[even_len + 1] = x[even_len - 1]
        return (
            0.25 * padded[0:even_len:2]
            + 0.5 * padded[1 : even_len + 1 : 2]
            + 0.25 * padded[2 : even_len + 2 : 2]
        ).astype(np.float32)

    if sr_from > sr_to:
        ratio = sr_from / sr_to
        n_out = max(1, int(x.size // ratio))
        edges = np.floor(np.arange(n_out + 1) * ratio).astype(np.int64)
        edges = np.clip(edges, 0, x.size)
        diffs = np.maximum(1, edges[1:] - edges[:-1])
        csum = np.empty(x.size + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(x, out=csum[1:])
        return ((csum[edges[1:]] - csum[edges[:-1]]) / diffs).astype(np.float32)

    n_out = max(1, round(x.size * sr_to / sr_from))
    t_old = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)
