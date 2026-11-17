"""Состояние голосового WS-звонка: буфер реплики, поколение хода, стейт.

Один экземпляр на одно соединение `WS /ws/call`. Здесь только состояние и
дешёвые операции над буфером — ASR, диалог и TTS живут в роуте, чтобы этот
модуль можно было тестировать без FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

DEFAULT_SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2          # lpcm16


class VoiceState(StrEnum):
    """Что происходит в звонке — клиент рисует по этому состоянию индикаторы."""

    listening = "listening"
    thinking = "thinking"
    speaking = "speaking"


@dataclass
class VoiceSession:
    """Буфер текущей реплики плюс счётчики ходов и перебиваний."""

    call_id: str
    phone: str
    sample_rate: int = DEFAULT_SAMPLE_RATE
    max_utterance_sec: float = 20.0
    state: VoiceState = VoiceState.listening
    turn: int = 0                 # номер хода диалога (растёт на каждой реплике)
    generation: int = 0           # растёт на перебивании — по нему отменяем озвучку
    truncated: bool = False       # реплику обрезали по лимитy длины

    _buffer: bytearray = field(default_factory=bytearray, repr=False)

    @property
    def max_bytes(self) -> int:
        return int(self.max_utterance_sec * self.sample_rate * BYTES_PER_SAMPLE)

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)

    @property
    def has_audio(self) -> bool:
        return len(self._buffer) > 0

    @property
    def duration_sec(self) -> float:
        return len(self._buffer) / (self.sample_rate * BYTES_PER_SAMPLE)

    def append(self, chunk: bytes) -> bool:
        """Добавляет аудио. False — кусок не влез в лимит реплики."""
        if not chunk:
            return True
        room = self.max_bytes - len(self._buffer)
        if room <= 0:
            self.truncated = True
            return False
        if len(chunk) > room:
            self._buffer.extend(chunk[: room - (room % BYTES_PER_SAMPLE)])
            self.truncated = True
            return False
        self._buffer.extend(chunk)
        return True

    def take_audio(self) -> bytes:
        """Забирает буфер реплики целиком и очищает его."""
        audio = bytes(self._buffer)
        self._buffer.clear()
        self.truncated = False
        return audio

    def drop_audio(self) -> None:
        self._buffer.clear()
        self.truncated = False

    def begin_turn(self) -> int:
        """Новый ход диалога: увеличивает счётчик ходов."""
        self.turn += 1
        return self.turn

    def barge(self) -> int:
        """Мягкое перебивание: новое поколение, всё прежнее считается отменённым."""
        self.generation += 1
        self.state = VoiceState.listening
        return self.generation

    def is_stale(self, generation: int) -> bool:
        """Поколение устарело — ответ этого хода уже не нужен."""
        return generation != self.generation
