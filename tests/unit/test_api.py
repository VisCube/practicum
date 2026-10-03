"""Интеграционные тесты API через TestClient"""

from pathlib import Path

from fastapi.testclient import TestClient

from assistant.api.app import create_app
from assistant.core.config import Settings

DATA = Path(__file__).resolve().parents[2] / "data"
KNOWN = "+79990000001"


def _client(**kwargs) -> TestClient:
    """Собирает тестовый клиент с реальными данными из /data и заданными настройками"""
    settings = Settings(data_dir=DATA, debug_ui=True, _env_file=None, **kwargs)
    return TestClient(create_app(settings))


def _turn(client: TestClient, call_id: str, text: str, phone: str = KNOWN) -> dict:
    """Отправляет реплику через /call/turn и возвращает распакованный JSON-ответ"""
    return client.post("/call/turn", json={"call_id": call_id, "phone": phone, "text": text}).json()


def test_health_and_debug_config():
    with _client() as client:
        assert client.get("/health").json()["status"] == "ok"
        cfg = client.get("/debug/config").json()
        assert cfg["asr"] == "fake" and cfg["tts"] == "fake" and len(cfg["demo_phones"]) == 3
        assert "html" in client.get("/debug/").text.lower()


def test_telephony_contract_full_call():
    with _client() as client:
        response = client.post("/call/start", json={"phone": KNOWN}).json()
        call_id = response["call_id"]
        assert response["text"].startswith("Здравствуйте, Иван Иванович")
        assert response["events"][0]["type"] == "outage_notice"
        assert response["audio_b64"] is None  # FakeTts → браузер озвучит сам

        response = _turn(client, call_id, "течёт из батареи")
        assert response["state"] == "confirm"
        response = _turn(client, call_id, "да")
        assert response["state"] == "done" and response["events"][0]["type"] == "appeal_created"
        assert response["events"][0]["payload"]["escalated"] is False

        view = client.get(f"/debug/call/{call_id}").json()
        assert view["owner"] == "Иванов Иван Иванович" and len(view["appeals"]) == 1
        assert client.get("/debug/outbox").json()[0]["appeal"]["slots"]["leak_source"] == "из батареи"


def test_complaint_flags_and_operator_request_transfers():
    with _client() as client:
        call_id = client.post("/call/start", json={"phone": KNOWN}).json()["call_id"]

        response = _turn(client, call_id, "напишу в прокуратуру, течёт из батареи")
        assert response["transfer"] is False and response["state"] == "confirm"
        assert [e["type"] for e in response["events"]] == ["escalated"]
        assert response["events"][0]["payload"]["markers"] == ["прокуратура"]
        assert response["text"].startswith("Понимаю вас.")

        response = _turn(client, call_id, "дайте оператора")
        assert response["transfer"] is True and response["state"] == "transferred"
        transfer_event = response["events"][0]
        assert transfer_event["type"] == "transfer"
        assert transfer_event["payload"]["reason"] == "клиент просит оператора"
        assert transfer_event["payload"]["topic"].startswith("Протечка")
        assert transfer_event["payload"]["escalated"] is True


def test_debug_say_audio_with_fake_asr_returns_silence():
    with _client() as client:
        call_id = client.post("/call/start", json={"phone": "+70000000000"}).json()["call_id"]
        response = client.post(
            f"/debug/call/{call_id}/say-audio", content=b"\x00" * 3200,
            headers={"Content-Type": "audio/pcm", "X-Phone": "+70000000000"},
        ).json()
        assert response["heard"] == "" and response["text"] == ""


def test_debug_disabled_in_prod():
    settings = Settings(data_dir=DATA, debug_ui=False, _env_file=None)
    with TestClient(create_app(settings)) as client:
        assert client.get("/debug/config").status_code == 404
        assert client.get("/health").status_code == 200


def test_end_call_drops_session():
    with _client() as client:
        call_id = client.post("/call/start", json={"phone": KNOWN}).json()["call_id"]
        _turn(client, call_id, "как проходит поверка счётчика")
        response = _turn(client, call_id, "спасибо, всё")
        assert response["end_call"] is True
        assert client.get(f"/debug/call/{call_id}").status_code == 404


def test_debug_say_returns_heard():
    with _client() as client:
        call_id = client.post("/call/start", json={"phone": KNOWN}).json()["call_id"]
        response = client.post(
            f"/debug/call/{call_id}/say",
            json={"call_id": call_id, "phone": KNOWN, "text": "нет воды"},
        ).json()
        assert response["heard"] == "нет воды" and response["state"] == "confirm"