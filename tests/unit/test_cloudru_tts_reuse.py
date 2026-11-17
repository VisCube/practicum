"""Переиспользование проверяется локальным WS-сервером, не реальным Cloud.ru."""

import asyncio
import base64
import json

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from assistant.services.tts.cloudru import CloudRuTts


def test_reuse_scope_voice_cancellation_expiry_and_shutdown(monkeypatch):
    import assistant.services.tts.cloudru as module

    monkeypatch.setattr(module, "_IDLE_TIMEOUT", 0.15)

    async def scenario():
        connections = []
        messages = []
        entered = asyncio.Event()
        peer_closed = asyncio.Event()

        async def handler(ws):
            number = len(connections)
            connections.append(number)
            try:
                await ws.send(json.dumps({"type": "session.created"}))
                update = json.loads(await ws.recv())
                voice = update["session"]["voice"]
                await ws.send(json.dumps({"type": "session.updated", "session": update["session"]}))
                async for raw in ws:
                    event = json.loads(raw)
                    text = event["response"]["instructions"]
                    messages.append((number, text, voice))
                    if text == "ждать":
                        entered.set()
                        await ws.wait_closed()
                        peer_closed.set()
                        return
                    await ws.send(
                        json.dumps(
                            {
                                "type": "response.audio.delta",
                                "delta": base64.b64encode(b"\x01\x00").decode(),
                            }
                        )
                    )
                    await ws.send(
                        json.dumps({"type": "response.done", "response": {"status": "completed"}})
                    )
            except ConnectionClosed:
                pass

        async with serve(handler, "127.0.0.1", 0, subprotocols=["realtime"]) as server:
            tts = CloudRuTts(api_key="key", base_url="https://example.com")
            tts._url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            await tts.synthesize_for_call("один", call_id="A")
            await tts.synthesize_for_call("два", call_id="A")
            assert messages[0][0] == messages[1][0]
            await tts.synthesize_for_call("три", call_id="B")
            assert messages[-1][0] != messages[0][0]
            await tts.synthesize_for_call("другой голос", call_id="A", speaker="other")
            assert messages[-1][0] not in (messages[0][0], messages[2][0])
            # Пробы без call_id никогда не используют диалоговое соединение.
            await tts.synthesize("проба")
            await tts.synthesize("проба два")
            assert messages[-1][0] != messages[-2][0]

            task = asyncio.create_task(tts.synthesize_for_call("ждать", call_id="A"))
            await asyncio.wait_for(entered.wait(), 2)
            interrupted_connection = messages[-1][0]
            task.cancel()
            try:
                await task
                raise AssertionError("отмена не дошла")
            except asyncio.CancelledError:
                pass
            await asyncio.wait_for(peer_closed.wait(), 2)
            await tts.synthesize_for_call("после отмены", call_id="A")
            assert messages[-1][0] != interrupted_connection
            tts.drop_call("A")
            assert not any(owner == "A" for owner, _, _ in tts._idle.values())
            await asyncio.sleep(0.25)
            assert not tts._idle
            await tts.synthesize_for_call("завершение", call_id="A")
            await tts.aclose()
            assert not tts._idle
            assert not tts._busy

    asyncio.run(scenario())


def test_parallel_same_call_does_not_share_socket():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        seen = []

        async def handler(ws):
            await ws.send(json.dumps({"type": "session.created"}))
            update = json.loads(await ws.recv())
            await ws.send(json.dumps({"type": "session.updated", "session": update["session"]}))
            try:
                async for raw in ws:
                    text = json.loads(raw)["response"]["instructions"]
                    seen.append((text, ws))
                    if len(seen) == 2:
                        started.set()
                    await release.wait()
                    await ws.send(json.dumps({"type": "response.audio.delta", "delta": "AQA="}))
                    await ws.send(
                        json.dumps({"type": "response.done", "response": {"status": "completed"}})
                    )
            except ConnectionClosed:
                pass

        async with serve(handler, "127.0.0.1", 0, subprotocols=["realtime"]) as server:
            tts = CloudRuTts(api_key="key", base_url="https://example.com")
            tts._url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            a = asyncio.create_task(tts.synthesize_for_call("первая", call_id="A"))
            b = asyncio.create_task(tts.synthesize_for_call("вторая", call_id="A"))
            await asyncio.wait_for(started.wait(), 2)
            assert seen[0][1] is not seen[1][1]
            release.set()
            await asyncio.gather(a, b)
            await tts.aclose()

    asyncio.run(scenario())


def test_failed_response_and_drop_busy_are_not_reused():
    from assistant.services.tts.base import TtsError

    async def scenario():
        seen = []
        entered = asyncio.Event()
        release = asyncio.Event()

        async def handler(ws):
            await ws.send(json.dumps({"type": "session.created"}))
            update = json.loads(await ws.recv())
            await ws.send(json.dumps({"type": "session.updated", "session": update["session"]}))
            try:
                async for raw in ws:
                    text = json.loads(raw)["response"]["instructions"]
                    seen.append((text, ws))
                    if text == "ожидание":
                        entered.set()
                        await release.wait()
                    await ws.send(json.dumps({"type": "response.audio.delta", "delta": "AQA="}))
                    await ws.send(
                        json.dumps(
                            {
                                "type": "response.done",
                                "response": {
                                    "status": "failed" if text == "ошибка" else "completed"
                                },
                            }
                        )
                    )
            except ConnectionClosed:
                pass

        async with serve(handler, "127.0.0.1", 0, subprotocols=["realtime"]) as server:
            tts = CloudRuTts(api_key="key", base_url="https://example.com")
            tts._url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            try:
                await tts.synthesize_for_call("ошибка", call_id="A")
                raise AssertionError("нельзя возвращать неполное аудио")
            except TtsError:
                pass
            await tts.synthesize_for_call("после ошибки", call_id="A")
            assert seen[0][1] is not seen[1][1]
            waiting = asyncio.create_task(tts.synthesize_for_call("ожидание", call_id="A"))
            await asyncio.wait_for(entered.wait(), 2)
            assert seen[1][1] is seen[2][1]
            tts.drop_call("A")
            release.set()
            await waiting
            assert not tts._idle
            await tts.synthesize_for_call("новый", call_id="A")
            assert seen[-1][1] is not seen[2][1]
            await tts.aclose()

    asyncio.run(scenario())


def test_drop_call_while_handshake_pending():
    from assistant.services.tts.base import TtsError

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()

        async def handler(ws):
            try:
                entered.set()
                await release.wait()
                await ws.send(json.dumps({"type": "session.created"}))
                update = json.loads(await ws.recv())
                await ws.send(json.dumps({"type": "session.updated", "session": update["session"]}))
                await ws.wait_closed()
            except ConnectionClosed:
                pass
            finally:
                closed.set()

        async with serve(handler, "127.0.0.1", 0, subprotocols=["realtime"]) as server:
            tts = CloudRuTts(api_key="key", base_url="https://example.com")
            tts._url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            task = asyncio.create_task(tts.synthesize_for_call("реплика", call_id="A"))
            await asyncio.wait_for(entered.wait(), 2)
            tts.drop_call("A")
            release.set()
            try:
                await task
                raise AssertionError("завершённый звонок не должен получить новый синтез")
            except TtsError:
                pass
            await asyncio.wait_for(closed.wait(), 2)
            assert not tts._idle and not tts._busy
            await tts.aclose()

    asyncio.run(scenario())
