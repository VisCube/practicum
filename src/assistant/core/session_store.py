"""Хранилище сессий звонков между ходами диалога

Сессия живёт ttl_sec с последнего обращения — брошенные звонки не копятся в памяти
lock(call_id) — блокировка на звонок: два параллельных хода (перебивание + реплика со стенда)
не должны затирать сессию друг другу; держит её граница API
"""

import asyncio
import time
from collections.abc import Callable
from typing import Protocol

from assistant.core.models import CallSession

DEFAULT_TTL_SEC = 30 * 60


class SessionStore(Protocol):
    """Контракт хранилища сессий — get/save/delete и блокировка на звонок"""
    def get(self, call_id: str) -> CallSession | None: ...
    def save(self, session: CallSession) -> None: ...
    def delete(self, call_id: str) -> None: ...
    def lock(self, call_id: str) -> asyncio.Lock: ...


class InMemorySessionStore:
    """Словарь в памяти с TTL. Для прототипа и тестов; clock подменяется в тестах"""

    def __init__(self, ttl_sec: float = DEFAULT_TTL_SEC, clock: Callable[[], float] = time.monotonic) -> None:
        self._data: dict[str, CallSession] = {}
        self._touched: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._ttl = ttl_sec
        self._clock = clock
        self._sweep_every = min(60.0, ttl_sec / 2)
        self._last_sweep = clock()

    def get(self, call_id: str) -> CallSession | None:
        self._sweep()
        session = self._data.get(call_id)
        if session is not None:
            self._touched[call_id] = self._clock()
        return session

    def save(self, session: CallSession) -> None:
        self._sweep()
        self._data[session.call_id] = session
        self._touched[session.call_id] = self._clock()

    def delete(self, call_id: str) -> None:
        """Удаляет сессию - блокировку не трогает, пока она удерживается активным ходом"""
        self._data.pop(call_id, None)
        self._touched.pop(call_id, None)
        lock = self._locks.get(call_id)
        if lock is not None and not lock.locked():
            del self._locks[call_id]

    def lock(self, call_id: str) -> asyncio.Lock:
        return self._locks.setdefault(call_id, asyncio.Lock())

    def __len__(self) -> int:
        return len(self._data)

    def _sweep(self) -> None:
        """Удаляет просроченные сессии - вызывается лениво, не чаще раза в _sweep_every секунд"""
        now = self._clock()
        if now - self._last_sweep < self._sweep_every:
            return
        self._last_sweep = now
        for call_id, touched in list(self._touched.items()):
            if now - touched > self._ttl:
                self.delete(call_id)