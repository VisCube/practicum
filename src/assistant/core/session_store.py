"""Хранилище сессий звонков между ходами диалога.

Сессия живёт `ttl_sec` с последнего обращения — брошенные звонки и осиротевшие
блокировки не копятся в памяти. `lock(call_id)` сериализует ходы одного звонка.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Protocol

from assistant.core.models import CallSession

DEFAULT_TTL_SEC = 30 * 60
DEFAULT_MAX_SESSIONS = 5000


class SessionStore(Protocol):
    """Контракт хранилища сессий — get/save/delete и блокировка на звонок."""

    def get(self, call_id: str) -> CallSession | None: ...
    def save(self, session: CallSession) -> None: ...
    def delete(self, call_id: str) -> None: ...
    def lock(self, call_id: str) -> asyncio.Lock: ...
    def sweep(self) -> None: ...


class InMemorySessionStore:
    """Словарь в памяти с TTL и лимитом размера. Для прототипа и тестов; clock подменяется."""

    def __init__(
        self,
        ttl_sec: float = DEFAULT_TTL_SEC,
        clock: Callable[[], float] = time.monotonic,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self._data: dict[str, CallSession] = {}
        self._touched: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._ttl = ttl_sec
        self._clock = clock
        self._max_sessions = max_sessions
        self._sweep_every = min(60.0, ttl_sec / 2)
        self._last_sweep = clock()

    def get(self, call_id: str) -> CallSession | None:
        self._sweep()
        session = self._data.get(call_id)
        if session is not None:
            self._touched.pop(call_id, None)
            self._touched[call_id] = self._clock()
        return session

    def save(self, session: CallSession) -> None:
        self._sweep()
        cid = session.call_id
        self._data[cid] = session
        self._touched.pop(cid, None)
        self._touched[cid] = self._clock()
        self._evict_overflow()

    def delete(self, call_id: str) -> None:
        """Удаляет сессию — блокировку не трогает, пока она удерживается активным ходом."""
        self._data.pop(call_id, None)
        self._touched.pop(call_id, None)
        lock = self._locks.get(call_id)
        if lock is not None and not lock.locked():
            del self._locks[call_id]

    def lock(self, call_id: str) -> asyncio.Lock:
        return self._locks.setdefault(call_id, asyncio.Lock())

    def __len__(self) -> int:
        return len(self._data)

    def sweep(self) -> None:
        """Принудительная очистка просроченных сессий и осиротевших блокировок."""
        self._sweep(force=True)

    def _evict_overflow(self) -> None:
        if self._max_sessions <= 0 or len(self._data) <= self._max_sessions:
            return
        overflow = len(self._data) - self._max_sessions
        for cid in list(self._touched):
            if overflow <= 0:
                break
            lk = self._locks.get(cid)
            if lk is not None and lk.locked():
                continue
            self.delete(cid)
            overflow -= 1

    def _sweep(self, *, force: bool = False) -> None:
        """Удаляет просроченные сессии и свободные осиротевшие блокировки."""
        now = self._clock()
        if not force and now - self._last_sweep < self._sweep_every:
            return
        self._last_sweep = now
        for call_id, touched in list(self._touched.items()):
            if now - touched > self._ttl:
                self.delete(call_id)
        for call_id in list(self._locks):
            if call_id not in self._data and not self._locks[call_id].locked():
                del self._locks[call_id]
