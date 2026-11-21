import asyncio
import json
import sqlite3
import time
from pathlib import Path

from fastapi.testclient import TestClient

from assistant.api.app import create_app
from assistant.core.config import Settings
from assistant.services.call_log.journal import CallJournal

DATA = Path(__file__).resolve().parents[2] / "data"
PHONE = "+79990000001"


def test_ws_journal_resume_transcript_and_stop(tmp_path):
    path = tmp_path / "journal.sqlite3"
    settings = Settings(
        data_dir=DATA, debug_ui=True, call_log_enabled=True, call_log_path=str(path), _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        with client.websocket_connect("/ws/call") as ws:
            ws.send_json({"t": "start", "phone": PHONE})
            ready = ws.receive_json()
            cid = ready["call_id"]
            while True:
                e = ws.receive_json()
                if e.get("t") == "state" and e.get("value") == "listening":
                    break
            ws.send_json({"t": "text", "text": "течёт батарея"})
            while True:
                e = ws.receive_json()
                if e.get("t") == "state" and e.get("value") == "listening":
                    break
            ws.send_json({"t": "barge", "played_ms": 125})
            while ws.receive_json().get("t") != "barge_ack":
                pass
        with client.websocket_connect("/ws/call") as ws:
            ws.send_json({"t": "start", "phone": PHONE, "resume_call_id": cid})
            assert ws.receive_json()["resumed"] is True
            ws.send_json({"t": "stop"})
    with sqlite3.connect(path) as db:
        call = db.execute(
            "select phone,status,ended_at from calls where call_id=?", (cid,)
        ).fetchone()
        rows = db.execute(
            "select kind,text,data_json from call_events where call_id=? order by id", (cid,)
        ).fetchall()
    assert call[0] == PHONE and call[1] == "stop" and call[2] is not None
    kinds = [r[0] for r in rows]
    assert kinds.count("start") == 1
    assert "disconnect" in kinds and "resume" in kinds and "barge" in kinds and "end" in kinds
    assert ("user_text", "течёт батарея") in [(r[0], r[1]) for r in rows]
    assert any(r[0] == "bot_generated" and r[1] for r in rows)
    assert any(r[0] == "reply_sent" and "dialog" in json.loads(r[2])["timings"] for r in rows)
    assert not path.stat().st_mode & 0o077


def test_retention_and_crash_unknown(tmp_path):
    path = tmp_path / "calls.db"

    async def run():
        journal = CallJournal(str(path), 90)
        await journal.start()
        journal.emit("old", "start", phone="123", channel="ws")
        journal.emit("old", "end", reason="stop")
        journal.emit("live", "start", phone="456", channel="ws")
        await journal.queue.join()
        with sqlite3.connect(path) as db:
            db.execute(
                "UPDATE calls SET ended_at=? WHERE call_id=?", (time.time() - 91 * 86400, "old")
            )
        journal.sweep()
        await journal.queue.join()
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT call_id FROM calls").fetchall() == [("live",)]
            assert not db.execute("SELECT * FROM call_events WHERE call_id=?", ("old",)).fetchall()
        await journal.aclose()
        restarted = CallJournal(str(path), 90)
        await restarted.start()
        with sqlite3.connect(path) as db:
            assert (
                db.execute("SELECT status FROM calls WHERE call_id=?", ("live",)).fetchone()[0]
                == "unknown"
            )
        await restarted.aclose()

    asyncio.run(run())


def test_http_call_journal(tmp_path):
    path = tmp_path / "calls.db"
    settings = Settings(
        data_dir=DATA, debug_ui=True, call_log_enabled=True, call_log_path=str(path), _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/call/start", json={"phone": PHONE}, headers={"x-audio": "off"})
        assert response.status_code == 200
        cid = response.json()["call_id"]
        response = client.post(
            "/call/turn",
            json={"call_id": cid, "phone": PHONE, "text": "сломался лифт"},
            headers={"x-audio": "off"},
        )
        assert response.status_code == 200
        assert client.delete("/debug/call/" + cid).status_code == 204
    with sqlite3.connect(path) as db:
        events = db.execute("SELECT kind,text FROM call_events WHERE call_id=?", (cid,)).fetchall()
        assert ("user_text", "сломался лифт") in events
        assert (
            db.execute("SELECT status FROM calls WHERE call_id=?", (cid,)).fetchone()[0] == "stop"
        )


def test_concurrent_writer_denied(tmp_path):
    path = tmp_path / "journal.sqlite3"

    async def run():
        a, b = CallJournal(str(path), 90), CallJournal(str(path), 90)
        await a.start()
        try:
            try:
                await b.start()
                raise AssertionError("второй процесс с тем же журналом не должен стартовать")
            except RuntimeError as exc:
                assert "используется" in str(exc)
        finally:
            await a.aclose()
        await b.start()
        await b.aclose()

    asyncio.run(run())


def test_generated_reply_survives_interruption(tmp_path):
    from assistant.api.routes.voice_ws import VoiceConnection
    from assistant.core.models import CallState, Reply
    from assistant.core.voice_session import VoiceSession

    path = tmp_path / "journal.sqlite3"
    settings = Settings(
        data_dir=DATA, debug_ui=True, call_log_enabled=True, call_log_path=str(path), _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        c = client.app.state.container

        async def run():
            # This direct test uses the TestClient's event loop via its portal below.
            started, release = asyncio.Event(), asyncio.Event()

            class Ws:
                async def send_text(self, _payload):
                    return None

            async def delayed(*_args):
                started.set()
                await release.wait()
                return Reply(text="ответ, который был сгенерирован", state=CallState.TOPIC)

            c.orchestrator.handle_turn = delayed
            c.journal("test-call", "start", phone=PHONE, channel="ws")
            conn = VoiceConnection(Ws(), c, VoiceSession(call_id="test-call", phone=PHONE))
            conn.start_turn(conn.handle_utterance(text="вопрос"))
            await started.wait()
            await conn.handle_event({"t": "barge", "played_ms": 10})
            release.set()
            await conn.close()
            await c.call_journal.queue.join()

        client.portal.call(run)
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT kind,text FROM call_events WHERE call_id=?", ("test-call",)
        ).fetchall()
        assert ("bot_generated", "ответ, который был сгенерирован") in rows
        assert ("barge", None) in rows
        assert not any(kind == "reply_sent" for kind, _ in rows)


def test_reused_call_id_creates_separate_journal_records(tmp_path):
    path = tmp_path / "calls.db"

    async def run():
        journal = CallJournal(str(path), 90)
        await journal.start()
        journal.emit("same", "start", phone="111", channel="ws")
        journal.emit("same", "user_text", text="first")
        journal.emit("same", "end", reason="stop")
        journal.emit("same", "start", phone="222", channel="ws")
        journal.emit("same", "user_text", text="second")
        journal.emit("same", "end", reason="stop")
        await journal.aclose()

    asyncio.run(run())
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT journal_id,phone FROM calls WHERE call_id=? ORDER BY started_at", ("same",)
        ).fetchall()
        assert len(rows) == 2 and [row[1] for row in rows] == ["111", "222"]
        first = db.execute(
            "SELECT text FROM call_events WHERE journal_id=? AND kind='user_text'", (rows[0][0],)
        ).fetchall()
        second = db.execute(
            "SELECT text FROM call_events WHERE journal_id=? AND kind='user_text'", (rows[1][0],)
        ).fetchall()
        assert first == [("first",)] and second == [("second",)]


def test_retry_locked_preserves_order_and_no_duplicate(tmp_path, monkeypatch):
    import assistant.services.call_log.journal as module

    monkeypatch.setattr(module, "_RETRY_DELAYS", (0.001, 0.001))
    path = tmp_path / "retry.db"

    async def run():
        journal = CallJournal(str(path), 90)
        await journal.start()
        original = journal._write
        attempts = []

        def blocked_twice(item):
            attempts.append(item["kind"])
            if len(attempts) <= 2:
                err = sqlite3.OperationalError("locked")
                err.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise err
            original(item)

        monkeypatch.setattr(journal, "_write", blocked_twice)
        journal.emit("retry", "start", phone="123", channel="ws")
        journal.emit("retry", "user_text", text="hello")
        await journal.aclose()
        assert attempts == ["start", "start", "start", "user_text"]

    asyncio.run(run())
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT kind FROM call_events ORDER BY id").fetchall() == [
            ("start",),
            ("user_text",),
        ]


def test_non_lock_error_not_retried_or_logged_with_private_text(tmp_path, monkeypatch, caplog):
    path = tmp_path / "error.db"

    async def run():
        journal = CallJournal(str(path), 90)
        await journal.start()
        attempts = []

        def broken(item):
            attempts.append(item)
            raise sqlite3.OperationalError("private text and phone 123")

        monkeypatch.setattr(journal, "_write", broken)
        journal.emit("test", "start", phone="123", channel="ws")
        await journal.aclose()
        assert len(attempts) == 1

    asyncio.run(run())
    assert "event lost" in caplog.text
    assert "private text" not in caplog.text and "123" not in caplog.text


def test_lock_retry_exhausted_reports_loss(tmp_path, monkeypatch, caplog):
    import assistant.services.call_log.journal as module

    monkeypatch.setattr(module, "_RETRY_DELAYS", (0.001, 0.001))

    async def run():
        journal = CallJournal(str(tmp_path / "locked.db"), 90)
        await journal.start()
        attempts = []

        def locked(item):
            attempts.append(item)
            err = sqlite3.OperationalError("locked")
            err.sqlite_errorcode = sqlite3.SQLITE_LOCKED
            raise err

        monkeypatch.setattr(journal, "_write", locked)
        journal.emit("test", "start", phone="123", channel="ws")
        await journal.aclose()
        assert len(attempts) == 3

    asyncio.run(run())
    assert "event lost" in caplog.text


def test_actual_sqlite_lock_is_retried(tmp_path, monkeypatch, caplog):
    import assistant.services.call_log.journal as module

    monkeypatch.setattr(module, "_RETRY_DELAYS", (0.03, 0.03, 0.03, 0.03))
    path = tmp_path / "real-lock.db"

    async def run():
        journal = CallJournal(str(path), 90)
        await journal.start()
        original_connect = journal._connect

        def quick_timeout():
            db = original_connect()
            db.execute("PRAGMA busy_timeout=10")
            return db

        monkeypatch.setattr(journal, "_connect", quick_timeout)
        blocker = sqlite3.connect(path)
        blocker.execute("BEGIN IMMEDIATE")
        journal.emit("call", "start", phone="123", channel="ws")
        journal.emit("call", "user_text", text="hello")
        await asyncio.sleep(0.07)
        blocker.rollback()
        blocker.close()
        await journal.aclose()

    asyncio.run(run())
    assert "database busy: retry" in caplog.text
    assert "event lost" not in caplog.text
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT kind FROM call_events ORDER BY id").fetchall() == [
            ("start",),
            ("user_text",),
        ]


def test_asr_is_journaled_even_when_ws_send_fails(tmp_path):
    from assistant.api.routes.voice_ws import VoiceConnection
    from assistant.core.voice_session import VoiceSession

    path = tmp_path / "asr.db"
    settings = Settings(
        data_dir=DATA, debug_ui=True, call_log_enabled=True, call_log_path=str(path), _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        container = client.app.state.container

        async def run():
            class BrokenWs:
                async def send_text(self, _payload):
                    raise ConnectionError("closed")

            async def transcribe(_audio):
                return "распознано до обрыва"

            container.asr.transcribe = transcribe
            container.journal("asr-call", "start", phone=PHONE, channel="ws")
            session = VoiceSession(call_id="asr-call", phone=PHONE)
            session.append(b"\0" * 20000)
            conn = VoiceConnection(BrokenWs(), container, session)
            await conn.handle_utterance()
            await conn.close()
            await container.call_journal.queue.join()

        client.portal.call(run)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT text FROM call_events WHERE kind='user_asr'").fetchall() == [
            ("распознано до обрыва",)
        ]


def test_barge_links_last_sent_reply_as_candidate(tmp_path):
    from assistant.api.routes.voice_ws import VoiceConnection
    from assistant.core.voice_session import VoiceSession

    path = tmp_path / "barge.db"
    settings = Settings(
        data_dir=DATA, debug_ui=True, call_log_enabled=True, call_log_path=str(path), _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        container = client.app.state.container

        async def run():
            class Ws:
                async def send_text(self, _payload):
                    pass

            container.journal("barge-call", "start", phone=PHONE, channel="ws")
            conn = VoiceConnection(Ws(), container, VoiceSession(call_id="barge-call", phone=PHONE))
            await conn.send({"t": "reply", "turn": 2, "text": "ответ", "audio_b64": "x"})
            await conn.handle_event({"t": "barge", "played_ms": 125})
            await conn.close()
            await container.call_journal.queue.join()

        client.portal.call(run)
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT data_json FROM call_events WHERE kind='barge'").fetchone()[0]
        assert json.loads(raw)["candidate_reply_turn"] == 2
        assert json.loads(raw)["candidate_basis"] == "last_reply_sent_on_connection"


def test_precheck_asr_recorded_once_before_start_turn(tmp_path):
    from assistant.api.routes.voice_ws import VoiceConnection
    from assistant.core.voice_session import VoiceSession

    path = tmp_path / "precheck.db"
    settings = Settings(
        data_dir=DATA, debug_ui=True, call_log_enabled=True, call_log_path=str(path), _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        container = client.app.state.container

        async def run():
            class Ws:
                async def send_text(self, _payload):
                    pass

            async def transcribe(_audio):
                return "голосовая поправка"

            container.asr.transcribe = transcribe
            container.journal("precheck", "start", phone=PHONE, channel="ws")
            conn = VoiceConnection(Ws(), container, VoiceSession(call_id="precheck", phone=PHONE))
            await conn._precheck_concurrent_utterance(b"\0" * 20000)
            await conn.close()
            await container.call_journal.queue.join()

        client.portal.call(run)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT text FROM call_events WHERE kind='user_asr'").fetchall() == [
            ("голосовая поправка",)
        ]


def test_business_event_payload_is_saved(tmp_path):
    from assistant.api.routes.voice_ws import VoiceConnection
    from assistant.core.models import CallState, Event, Reply
    from assistant.core.voice_session import VoiceSession

    path = tmp_path / "events.db"
    settings = Settings(
        data_dir=DATA, debug_ui=True, call_log_enabled=True, call_log_path=str(path), _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        container = client.app.state.container

        async def run():
            class Ws:
                async def send_text(self, _payload):
                    pass

            async def dialog():
                return Reply(
                    text="ответ",
                    state=CallState.TOPIC,
                    events=[Event(type="transfer", payload={"reason": "тест"})],
                )

            container.journal("events", "start", phone=PHONE, channel="ws")
            conn = VoiceConnection(Ws(), container, VoiceSession(call_id="events", phone=PHONE))
            await conn._run_dialog(dialog())
            await conn.close()
            await container.call_journal.queue.join()

        client.portal.call(run)
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT data_json FROM call_events WHERE kind='bot_generated'").fetchone()[
            0
        ]
        assert json.loads(raw)["events"] == [{"type": "transfer", "payload": {"reason": "тест"}}]


def test_sqlite_backup_consistent_and_rotates_only_own_files(tmp_path):
    path = tmp_path / "data" / "calls.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o700)
    old_name = "call-log-20200101T000000Z-" + "a" * 32 + ".sqlite3"
    old = backup_dir / old_name
    old.write_text("old")
    other = backup_dir / "other.sqlite3"
    other.write_text("keep")
    old_time = time.time() - 31 * 86400
    import os

    os.utime(old, (old_time, old_time))
    os.utime(other, (old_time, old_time))

    async def run():
        journal = CallJournal(str(path), 90)
        await journal.start()
        journal.emit("call", "start", phone=PHONE, channel="ws")
        journal.emit("call", "user_text", text="проверка")
        await journal.queue.join()
        journal.backup_dir = backup_dir
        await asyncio.to_thread(journal._backup)
        await journal.aclose()

    asyncio.run(run())
    assert not old.exists() and other.exists()
    copies = list(backup_dir.glob("call-log-*.sqlite3"))
    assert len(copies) == 1
    assert copies[0].stat().st_mode & 0o077 == 0
    with sqlite3.connect(copies[0]) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT text FROM call_events WHERE kind='user_text'").fetchone() == (
            "проверка",
        )


def test_backup_scheduler_creates_copy_and_closes(tmp_path):
    path, copies = tmp_path / "data" / "calls.db", tmp_path / "backup"

    async def run():
        journal = CallJournal(str(path), 90, backup_dir=str(copies), backup_interval_sec=3600)
        await journal.start()
        # Wait for first backup, without relying on fixed sleeps.
        for _ in range(100):
            if list(copies.glob("call-log-*.sqlite3")):
                break
            await asyncio.sleep(0.01)
        assert list(copies.glob("call-log-*.sqlite3"))
        await journal.aclose()

    asyncio.run(run())


def test_backup_refuses_database_directory(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        CallJournal(str(tmp_path / "calls.db"), 90, backup_dir=str(tmp_path))


def test_backup_failure_does_not_rotate_existing_copies(tmp_path, monkeypatch):
    import pytest

    path = tmp_path / "data" / "calls.db"
    directory = tmp_path / "backups"
    directory.mkdir(mode=0o700)
    old = directory / ("call-log-20200101T000000Z-" + "a" * 32 + ".sqlite3")
    old.write_text("important")
    import os

    os.utime(old, (1, 1))

    async def run():
        journal = CallJournal(str(path), 90)
        await journal.start()
        journal.backup_dir = directory

        def broken():
            raise sqlite3.OperationalError("disk failure")

        monkeypatch.setattr(journal, "_connect", broken)
        with pytest.raises(sqlite3.OperationalError):
            journal._backup()
        assert old.exists()
        assert not list(directory.glob(".call-log-*.tmp"))
        await journal.aclose()

    asyncio.run(run())


def test_http_eviction_marks_unknown_without_touching_active_or_ws(tmp_path):
    from assistant.core.models import CallSession
    from assistant.core.session_store import InMemorySessionStore

    async def run():
        now = [1000.0]
        store = InMemorySessionStore(ttl_sec=10, clock=lambda: now[0])
        journal = CallJournal(str(tmp_path / "calls.db"), 90)
        await journal.start()
        for cid, channel in (("expired", "http"), ("inflight", "http"), ("ws", "ws")):
            journal.emit(cid, "start", phone=PHONE, channel=channel)
            store.save(CallSession(call_id=cid, phone=PHONE))
        token = journal.http_request_begin("inflight")
        now[0] += 11
        store.sweep()
        journal.expire_http(store.contains_or_locked)
        await journal.queue.join()
        with sqlite3.connect(journal.path) as db:
            statuses = dict(db.execute("SELECT call_id,status FROM calls"))
            assert statuses == {"expired": "unknown", "inflight": "active", "ws": "active"}
            assert (
                json.loads(
                    db.execute(
                        "SELECT data_json FROM call_events WHERE call_id='expired' AND kind='end'"
                    ).fetchone()[0]
                )["detected"]
                == "session_evicted"
            )
        journal.http_request_end(token)
        journal.expire_http(store.contains_or_locked)
        await journal.aclose()
        with sqlite3.connect(journal.path) as db:
            assert db.execute("SELECT status FROM calls WHERE call_id='inflight'").fetchone() == (
                "unknown",
            )

    asyncio.run(run())


def test_http_session_not_expired_by_journal_check(tmp_path):
    from assistant.core.models import CallSession
    from assistant.core.session_store import InMemorySessionStore

    async def run():
        now = [0.0]
        store = InMemorySessionStore(ttl_sec=10, clock=lambda: now[0])
        journal = CallJournal(str(tmp_path / "calls.db"), 90)
        await journal.start()
        journal.emit("call", "start", phone=PHONE, channel="http")
        store.save(CallSession(call_id="call", phone=PHONE))
        now[0] = 8
        store.sweep()
        journal.expire_http(store.contains_or_locked)
        now[0] = 11
        store.sweep()
        journal.expire_http(store.contains_or_locked)
        await journal.aclose()
        with sqlite3.connect(journal.path) as db:
            assert db.execute("SELECT status FROM calls").fetchone() == ("unknown",)

    asyncio.run(run())


def test_http_sweep_integrates_with_real_routes(tmp_path):
    path = tmp_path / "calls.db"
    settings = Settings(
        data_dir=DATA, debug_ui=True, call_log_enabled=True, call_log_path=str(path), _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/call/start", json={"phone": PHONE}, headers={"x-audio": "off"})
        assert response.status_code == 200
        cid = response.json()["call_id"]
        container = client.app.state.container

        async def sweep():
            container.orchestrator.store.delete(cid)
            container.sweep()
            container.call_journal.expire_http(container.orchestrator.store.contains_or_locked)
            await container.call_journal.queue.join()

        client.portal.call(sweep)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM calls WHERE call_id=?", (cid,)).fetchone() == (
            "unknown",
        )


def test_size_limit_deletes_oldest_completed_calls_not_active(tmp_path):
    path = tmp_path / "calls.db"

    async def run():
        journal = CallJournal(str(path), 90, max_used_bytes=110000)
        await journal.start()
        for cid in ("old", "new", "active"):
            journal.emit(cid, "start", phone=PHONE, channel="http")
            journal.emit(cid, "user_text", text=cid * 18000)
            if cid != "active":
                journal.emit(cid, "end", reason="stop")
        await journal.queue.join()
        with sqlite3.connect(path) as db:
            db.execute("UPDATE calls SET ended_at=100 WHERE call_id='old'")
            db.execute("UPDATE calls SET ended_at=200 WHERE call_id='new'")
        # Keep the synthetic old timestamps within retention.
        journal.retention_days = 100000
        journal.sweep()
        await journal.queue.join()
        with sqlite3.connect(path) as db:
            remaining = dict(db.execute("SELECT call_id,status FROM calls"))
            assert "old" not in remaining
            assert remaining.get("active") == "active"
            assert not db.execute("SELECT 1 FROM call_events WHERE call_id='old'").fetchone()
        await journal.aclose()

    asyncio.run(run())


def test_small_limit_never_deletes_active_call(tmp_path, caplog):
    async def run():
        journal = CallJournal(str(tmp_path / "calls.db"), 90, max_used_bytes=1)
        await journal.start()
        journal.emit("live", "start", phone=PHONE, channel="ws")
        journal.emit("live", "user_text", text="live data")
        journal.sweep()
        await journal.aclose()
        with sqlite3.connect(journal.path) as db:
            assert db.execute("SELECT status FROM calls").fetchone() == ("active",)
            assert db.execute("SELECT text FROM call_events WHERE kind='user_text'").fetchone() == (
                "live data",
            )

    asyncio.run(run())
    assert "no completed calls to remove" in caplog.text


def test_low_free_space_warning_does_not_delete_unexpired_calls(tmp_path, monkeypatch, caplog):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: usage(100, 99, 1))

    async def run():
        journal = CallJournal(str(tmp_path / "calls.db"), 90, min_free_bytes=10)
        await journal.start()
        journal.emit("done", "start", phone=PHONE, channel="http")
        journal.emit("done", "end", reason="stop")
        journal.sweep()
        await journal.aclose()
        with sqlite3.connect(journal.path) as db:
            assert db.execute("SELECT status FROM calls").fetchone() == ("stop",)

    asyncio.run(run())
    assert "filesystem low on free space" in caplog.text


def test_journal_uses_full_synchronous(tmp_path):
    async def run():
        journal = CallJournal(str(tmp_path / "calls.db"), 90)
        await journal.start()
        with journal._transaction() as db:
            assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        await journal.aclose()

    asyncio.run(run())


def test_committed_event_survives_abrupt_process_exit(tmp_path):
    import os
    import subprocess
    import sys

    path = tmp_path / "crash.db"
    script = """
import asyncio, os, sys
from assistant.services.call_log.journal import CallJournal

async def run():
    journal = CallJournal(sys.argv[1], 90)
    await journal.start()
    journal.emit("abrupt", "start", phone="123", channel="ws")
    journal.emit("abrupt", "user_text", text="committed")
    await journal.queue.join()
    os._exit(42)
asyncio.run(run())
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env=os.environ.copy(),
        check=False,
        timeout=15,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 42, proc.stderr

    async def restart():
        journal = CallJournal(str(path), 90)
        await journal.start()
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT status FROM calls").fetchone() == ("unknown",)
            assert db.execute("SELECT text FROM call_events WHERE kind='user_text'").fetchone() == (
                "committed",
            )
        await journal.aclose()

    asyncio.run(restart())
