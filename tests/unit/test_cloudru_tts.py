import asyncio
import base64
import io
import json
import wave

import pytest
from websockets.asyncio.server import serve

from assistant.api.dependencies import _build_tts
from assistant.core.config import Settings
from assistant.services.tts.base import TtsError
from assistant.services.tts.cloudru import CloudRuTts, _realtime_url


def test_url_and_config():
    assert (
        _realtime_url("https://foundation-models.api.cloud.ru/v1/", "GigaChat/SaluteSpeech")
        == "wss://foundation-models.api.cloud.ru/v1/realtime?model=GigaChat/SaluteSpeech"
    )
    with pytest.raises(ValueError):
        _realtime_url("http://example.com/v1", "x")
    with pytest.raises(ValueError, match="FM_API_KEY"):
        _build_tts(Settings(tts_adapter="cloudru", fm_api_key="", _env_file=None))
    tts = _build_tts(Settings(tts_adapter="cloudru", fm_api_key="secret", _env_file=None))
    assert isinstance(tts, CloudRuTts) and tts.speaker == "Bab-Freespeech"


def test_realtime_wav_and_parallel_isolation():
    async def scenario():
        requests = []

        async def handler(ws):
            assert ws.request.headers["Authorization"] == "Bearer secret"
            assert ws.request.headers["OpenAI-Beta"] == "realtime=v1"
            assert ws.subprotocol == "realtime"
            await ws.send(json.dumps({"type": "session.created"}))
            update = json.loads(await ws.recv())
            assert update["session"]["gigatts"]["content_type"] == "text"
            assert update["session"]["output_audio_format"] == "pcm16"
            await ws.send(json.dumps({"type": "session.updated", "session": update["session"]}))
            req = json.loads(await ws.recv())
            text = req["response"]["instructions"]
            requests.append((text, update["session"]["voice"]))
            data = b"\x01\x00" if text == "первый" else b"\x02\x00"
            for _ in range(2):
                await ws.send(
                    json.dumps(
                        {"type": "response.audio.delta", "delta": base64.b64encode(data).decode()}
                    )
                )
            await ws.send(
                json.dumps({"type": "response.done", "response": {"status": "completed"}})
            )

        async with serve(handler, "127.0.0.1", 0, subprotocols=["realtime"]) as server:
            tts = CloudRuTts(api_key="secret", base_url="https://example.com/v1")
            tts._url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime?model=x"
            a, b = await asyncio.gather(
                tts.synthesize("первый"), tts.synthesize("второй", speaker="other")
            )
            assert set(requests) == {("первый", "Bab-Freespeech"), ("второй", "other")}
            for audio, expected in [(a, b"\x01\x00" * 2), (b, b"\x02\x00" * 2)]:
                assert audio.mime == "audio/wav"
                with wave.open(io.BytesIO(audio.data)) as wav:
                    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
                        1,
                        2,
                        24000,
                    )
                    assert wav.readframes(2) == expected

    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["failed", "invalid", "empty", "error", "timeout", "cancel"])
def test_failure_and_cancellation(case):
    async def scenario():
        waiting = asyncio.Event()
        closed = asyncio.Event()

        async def handler(ws):
            try:
                await ws.send(json.dumps({"type": "session.created"}))
                await ws.recv()
                await ws.send(
                    json.dumps(
                        {"type": "session.updated", "session": {"output_audio_format": "pcm16"}}
                    )
                )
                await ws.recv()
                if case in ("timeout", "cancel"):
                    waiting.set()
                    await ws.wait_closed()
                elif case == "error":
                    await ws.send(json.dumps({"type": "error", "error": {"message": "private"}}))
                elif case == "invalid":
                    await ws.send(
                        json.dumps({"type": "response.audio.delta", "delta": "!invalid!"})
                    )
                else:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "response.done",
                                "response": {
                                    "status": "failed" if case == "failed" else "completed"
                                },
                            }
                        )
                    )
            finally:
                closed.set()

        async with serve(handler, "127.0.0.1", 0, subprotocols=["realtime"]) as server:
            tts = CloudRuTts(
                api_key="secret",
                base_url="https://example.com",
                timeout=0.15 if case == "timeout" else 3,
            )
            tts._url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            if case == "cancel":
                task = asyncio.create_task(tts.synthesize("text"))
                await asyncio.wait_for(waiting.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(TtsError) as error:
                    await tts.synthesize("text")
                assert "secret" not in str(error.value) and "private" not in str(error.value)
            await asyncio.wait_for(closed.wait(), 2)

    asyncio.run(scenario())


def test_cleanup_cloudru_only_and_preserves_content():
    from assistant.services.tts.cloudru import _clean_cloudru_text

    text = (
        "Поверка счётчика воды обязана проводиться раз в\u202f5\u202fлет "
        "(срок может быть иначе, если в паспорте указана другая дата). "
        "Узнать точную дату можно в паспорте прибора; уточните у менеджера УК."
    )
    result = _clean_cloudru_text(text)
    assert result == (
        "Поверка счётчика воды обязана проводиться раз в 5 лет, "
        "срок может быть иначе, если в паспорте указана другая дата. "
        "Узнать точную дату можно в паспорте прибора; уточните у менеджера УК."
    )
    assert _clean_cloudru_text("**БРИС** ЖКХ, +7 (912) 345-67-89") == (
        "БРИС ЖКХ, +7 (912) 345-67-89"
    )
    assert _clean_cloudru_text("условие (одно (другое)) верно") == ("условие (одно (другое)) верно")
    tts = CloudRuTts(api_key="key", base_url="https://example.com")
    assert tts._prepare("Тариф (при наличии договора).") == "Тариф, при наличии договора."


def test_synth_sends_cleaned_text_and_reports_it():
    from assistant.core.models import Audio

    async def scenario():
        sent = []
        tts = CloudRuTts(api_key="key", base_url="https://example.com")

        async def fake_synth(text, voice):
            sent.append((text, voice))
            return Audio(data=b"wav", mime="audio/wav")

        tts._synthesize = fake_synth
        original = "Пять лет (при другом сроке смотрите паспорт)."
        await tts.synthesize(original)
        assert sent == [("Пять лет, при другом сроке смотрите паспорт.", "Bab-Freespeech")]
        assert tts.last_spoken(original) == sent[0][0]

    asyncio.run(scenario())


def test_cloudru_ignores_global_normalization():
    tts = _build_tts(
        Settings(tts_adapter="cloudru", fm_api_key="key", tts_normalize=True, _env_file=None)
    )
    assert tts._prepare("Заявка № 42, ЖКХ. Тариф (по договору).") == (
        "Заявка № 42, ЖКХ. Тариф, по договору."
    )
