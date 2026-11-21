"""Single-process SQLite call journal; IO is performed off the event loop."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)

# Retry only lock contention. Other database errors may have uncertain outcomes
# or require intervention; do not blindly repeat a potentially committed write.
_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0)


def _is_lock_contention(error: BaseException) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    return isinstance(code, int) and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    journal_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL,
    phone TEXT NOT NULL,
    channel TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS call_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journal_id TEXT NOT NULL REFERENCES calls(journal_id) ON DELETE CASCADE,
    call_id TEXT NOT NULL,
    at REAL NOT NULL,
    kind TEXT NOT NULL,
    turn INTEGER,
    text TEXT,
    data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_call_id ON call_events(call_id, id);
CREATE INDEX IF NOT EXISTS events_journal_id ON call_events(journal_id, id);
CREATE INDEX IF NOT EXISTS calls_ended_at ON calls(ended_at);
CREATE INDEX IF NOT EXISTS calls_call_id ON calls(call_id, started_at);
"""


class CallJournal:
    def __init__(
        self,
        path: str,
        retention_days: int,
        *,
        queue_size: int = 10000,
        abandoned_sec: float = 1800,
        backup_dir: str = "",
        backup_retention_days: int = 30,
        backup_interval_sec: float = 86400,
        max_used_bytes: int = 512 * 1024 * 1024,
        min_free_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if max_used_bytes <= 0 or min_free_bytes < 0:
            raise ValueError("Некорректные лимиты места журнала")
        self.max_used_bytes = max_used_bytes
        self.min_free_bytes = min_free_bytes
        if retention_days < 1:
            raise ValueError("CALL_LOG_RETENTION_DAYS должен быть не меньше 1")
        if backup_retention_days < 1 or backup_interval_sec <= 0:
            raise ValueError("Срок хранения и интервал резервных копий должны быть положительными")
        self.path = Path(path)
        self.backup_dir = Path(backup_dir) if backup_dir.strip() else None
        if self.backup_dir and self.backup_dir.resolve() == self.path.resolve().parent:
            raise ValueError("Каталог резервных копий должен отличаться от каталога журнала")
        self.backup_retention_days = backup_retention_days
        self.backup_interval_sec = backup_interval_sec
        self.backup_stop = asyncio.Event()
        self.backup_task: asyncio.Task[None] | None = None
        self.retention_days = retention_days
        self.abandoned_sec = abandoned_sec
        self.lock_file: Any = None
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=queue_size)
        self.worker: asyncio.Task[None] | None = None
        self.accepting = False
        self._sessions: dict[str, str] = {}
        self._http_active: dict[str, str] = {}
        self._http_inflight: dict[str, int] = {}
        self._reap: dict[str, asyncio.TimerHandle] = {}

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        # Request durable WAL commits; SQLite and the storage device must honor fsync.
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        return db

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def _init_db(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # O_EXCL does not change existing database permissions.
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        os.chmod(self.path, 0o600)
        lock_file = open(str(self.path) + ".lock", "a+b")  # noqa: SIM115
        try:
            os.chmod(str(self.path) + ".lock", 0o600)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lock_file.close()
            raise RuntimeError("call journal: файл уже используется другим процессом") from None
        self.lock_file = lock_file
        try:
            with self._transaction() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(_SCHEMA)
                # Only one server process may use this journal. Unfinished sessions
                # from a previous process no longer have an in-memory session.
                db.execute(
                    "UPDATE calls SET status='unknown', ended_at=? WHERE ended_at IS NULL",
                    (time.time(),),
                )

        except BaseException:
            self.lock_file.close()
            self.lock_file = None
            raise

    async def start(self) -> None:
        await asyncio.to_thread(self._init_db)
        self.accepting = True
        self.worker = asyncio.create_task(self._run())
        if self.backup_dir is not None:
            self.backup_stop.clear()
            self.backup_task = asyncio.create_task(self._backup_loop())

    def _backup(self) -> None:
        if self.backup_dir is None:
            return
        self.backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.backup_dir.stat().st_mode & 0o077:
            raise PermissionError("backup directory must be private (mode 0700)")
        fd, tmp_name = tempfile.mkstemp(prefix=".call-log-", suffix=".tmp", dir=self.backup_dir)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with closing(self._connect()) as source, closing(sqlite3.connect(tmp)) as target:
                source.backup(target, pages=128, sleep=0.05)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            destination = self.backup_dir / f"call-log-{stamp}-{uuid4().hex}.sqlite3"
            os.replace(tmp, destination)
            cutoff = time.time() - self.backup_retention_days * 86400
            pattern = re.compile(r"^call-log-\d{8}T\d{6}Z-[0-9a-f]{32}\.sqlite3$")
            for entry in self.backup_dir.iterdir():
                if (
                    pattern.fullmatch(entry.name)
                    and not entry.is_symlink()
                    and entry.is_file()
                    and entry.stat().st_mtime < cutoff
                ):
                    entry.unlink()
        finally:
            tmp.unlink(missing_ok=True)

    async def _backup_loop(self) -> None:
        while not self.backup_stop.is_set():
            try:
                await asyncio.to_thread(self._backup)
                log.info("call journal backup completed")
            except Exception as error:
                log.error("call journal backup failed (%s)", type(error).__name__)
            with suppress(TimeoutError):
                await asyncio.wait_for(self.backup_stop.wait(), timeout=self.backup_interval_sec)

    def emit(
        self,
        call_id: str,
        kind: str,
        *,
        text: str | None = None,
        turn: int | None = None,
        **data: Any,
    ) -> None:
        if not self.accepting:
            return
        if kind in ("start", "resume"):
            timer = self._reap.pop(call_id, None)
            if timer:
                timer.cancel()
        if kind == "start":
            self._sessions[call_id] = uuid4().hex
            self._http_active.pop(call_id, None)
            if data.get("channel") == "http":
                self._http_active[call_id] = self._sessions[call_id]
        journal_id = self._sessions.get(call_id)
        if not journal_id:
            log.error("call journal missing start: event lost")
            return
        item = {
            "journal_id": journal_id,
            "call_id": call_id,
            "kind": kind,
            "at": time.time(),
            "text": text,
            "turn": turn,
            "data": data,
        }
        try:
            self.queue.put_nowait(item)
            if kind == "end" and self._http_active.get(call_id) == journal_id:
                self._http_active.pop(call_id, None)
            if kind in ("end", "disconnect"):
                timer = self._reap.pop(call_id, None)
                if timer:
                    timer.cancel()
                delay = 600 if kind == "end" else max(0, self.abandoned_sec) + 600
                self._reap[call_id] = asyncio.get_running_loop().call_later(
                    delay, self._forget, call_id, journal_id
                )
        except asyncio.QueueFull:
            log.error("call journal queue full: event lost")

    def http_request_begin(self, call_id: str) -> str | None:
        journal_id = self._http_active.get(call_id)
        if journal_id is not None:
            self._http_inflight[journal_id] = self._http_inflight.get(journal_id, 0) + 1
        return journal_id

    def http_request_end(self, journal_id: str | None) -> None:
        if journal_id is None:
            return
        remaining = self._http_inflight.get(journal_id, 0) - 1
        if remaining > 0:
            self._http_inflight[journal_id] = remaining
        else:
            self._http_inflight.pop(journal_id, None)

    def expire_http(self, contains_or_locked: Any) -> None:
        """Called on the event loop after the session-store sweep."""
        if not callable(contains_or_locked):
            return
        for call_id, journal_id in list(self._http_active.items()):
            if not self._http_inflight.get(journal_id) and not contains_or_locked(call_id):
                self.emit(call_id, "end", reason="unknown", detected="session_evicted")

    def _forget(self, call_id: str, journal_id: str) -> None:
        self._reap.pop(call_id, None)
        if self._sessions.get(call_id) == journal_id:
            self._sessions.pop(call_id, None)

    def _write(self, item: dict[str, Any]) -> None:
        cid, kind, at, data = item["call_id"], item["kind"], item["at"], item["data"]
        journal_id = item["journal_id"]
        with self._transaction() as db:
            if kind == "start":
                db.execute(
                    "INSERT INTO calls VALUES (?, ?, ?, ?, ?, NULL, 'active')",
                    (journal_id, cid, data.get("phone", ""), data.get("channel", "unknown"), at),
                )
            elif kind == "resume":
                db.execute(
                    "UPDATE calls SET status='active' WHERE journal_id=? AND ended_at IS NULL",
                    (journal_id,),
                )
            elif kind == "disconnect":
                db.execute(
                    "UPDATE calls SET status='disconnected' "
                    "WHERE journal_id=? AND ended_at IS NULL",
                    (journal_id,),
                )
            elif kind == "end":
                db.execute(
                    "UPDATE calls SET ended_at=?, status=? WHERE journal_id=? AND ended_at IS NULL",
                    (at, data.get("reason", "ended"), journal_id),
                )
            # A missing start must not turn into a silently invented record.
            exists = db.execute("SELECT 1 FROM calls WHERE journal_id=?", (journal_id,)).fetchone()
            if exists:
                details = {key: value for key, value in data.items() if key != "phone"}
                db.execute(
                    "INSERT INTO call_events(journal_id,call_id,at,kind,turn,text,data_json) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        journal_id,
                        cid,
                        at,
                        kind,
                        item["turn"],
                        item["text"],
                        json.dumps(details, ensure_ascii=False),
                    ),
                )
            else:
                log.error("call journal missing start: event lost")

    def _cleanup(self) -> None:
        cutoff = time.time() - self.retention_days * 86400
        with self._transaction() as db:
            now = time.time()
            db.execute(
                """UPDATE calls SET status='unknown', ended_at=? WHERE status='disconnected'
                AND ended_at IS NULL AND (SELECT MAX(at) FROM call_events
                WHERE journal_id=calls.journal_id) < ?""",
                (now, now - self.abandoned_sec),
            )
            db.execute("DELETE FROM calls WHERE ended_at IS NOT NULL AND ended_at < ?", (cutoff,))
            removed = 0
            # The estimate excludes SQLite free pages so they can be reused by future calls.
            # Bound each pass to avoid holding the SQLite write lock indefinitely.
            for _ in range(100):
                page_size = db.execute("PRAGMA page_size").fetchone()[0]
                pages = db.execute("PRAGMA page_count").fetchone()[0]
                free = db.execute("PRAGMA freelist_count").fetchone()[0]
                if (pages - free) * page_size <= self.max_used_bytes:
                    break
                oldest = db.execute(
                    "SELECT journal_id FROM calls WHERE ended_at IS NOT NULL "
                    "ORDER BY ended_at, journal_id LIMIT 1"
                ).fetchone()
                if oldest is None:
                    log.warning("call journal size limit exceeded: no completed calls to remove")
                    break
                db.execute("DELETE FROM calls WHERE journal_id=?", (oldest[0],))
                removed += 1
            if removed:
                log.warning("call journal size limit: removed %d oldest completed calls", removed)
        free_space = shutil.disk_usage(self.path.parent).free
        if free_space < self.min_free_bytes:
            log.warning("call journal filesystem low on free space: %d bytes", free_space)

    async def _write_with_retry(self, item: dict[str, Any]) -> None:
        """Keep the event at the head of the queue during transient lock contention."""
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                if item.get("kind") == "__cleanup":
                    await asyncio.to_thread(self._cleanup)
                else:
                    await asyncio.to_thread(self._write, item)
                return
            except sqlite3.Error as error:
                if not _is_lock_contention(error) or attempt == len(_RETRY_DELAYS):
                    raise
                log.warning("call journal database busy: retry %d", attempt + 1)
                await asyncio.sleep(_RETRY_DELAYS[attempt])

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                await self._write_with_retry(item)
            except Exception as error:
                # Never log event contents or exception text, which may contain personal data.
                log.error("call journal write failed: event lost (%s)", type(error).__name__)
            finally:
                self.queue.task_done()

    def sweep(self) -> None:
        if self.accepting:
            try:
                self.queue.put_nowait({"kind": "__cleanup"})
            except asyncio.QueueFull:
                log.error("call journal queue full: cleanup skipped")

    async def aclose(self) -> None:
        self.accepting = False
        for timer in self._reap.values():
            timer.cancel()
        self._reap.clear()
        self._sessions.clear()
        self._http_active.clear()
        self._http_inflight.clear()
        self.backup_stop.set()
        if self.backup_task:
            await self.backup_task
            self.backup_task = None
        if self.worker:
            await self.queue.put(None)
            await self.worker
            self.worker = None
        if self.lock_file:
            self.lock_file.close()
            self.lock_file = None
