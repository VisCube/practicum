from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from assistant.api.dependencies import build_container
from assistant.core.config import Settings
from assistant.core.session_store import InMemorySessionStore
from assistant.services.cloudru.dialog import CloudRuDialogAdapter
from assistant.services.crm.fake import InMemoryCrm
from assistant.services.crm.outbox import InMemoryOutboxStore, Outbox
from assistant.services.crm.telegram import TelegramCrm
from assistant.services.telephony.fake import FakeTelephony

DATA = Path(__file__).resolve().parents[2] / 'data'


class Agent:
    def __init__(self, phone):
        self.phone = phone

    def handle_message(self, text):
        if text == 'Алло':
            return 'Здравствуйте', []
        result = {'id': 4422, 'category': 'Тест', 'phone': self.phone, 'description': 'тест'}
        call = SimpleNamespace(name='create_ticket', arguments=result, result=json.dumps(result))
        return 'Заявка создана', [call]


def adapter(crm, enabled):
    outbox = Outbox(crm, InMemoryOutboxStore())
    return CloudRuDialogAdapter(store=InMemorySessionStore(), crm=crm, outbox=outbox,
                               telephony=FakeTelephony(), assistant_factory=Agent,
                               outbox_enabled=enabled), outbox


@pytest.mark.asyncio
async def test_off_preserves_event_without_queue_or_delivery():
    crm = InMemoryCrm()
    dialog, outbox = adapter(crm, False)
    try:
        await dialog.start('off', '+79990000001')
        reply = await dialog.handle_turn('off', '+79990000001', 'создай')
        card = next(event for event in reply.events if event.type == 'appeal_created')
        assert card.payload['appeal_id'] == '4422'
        assert card.payload['delivery_mode'] == 'upstream'
        assert card.payload['slots']['description'] == 'тест'
        assert dialog.store.get('off').appeals == ['4422']
        assert outbox.store.all() == []
        assert await outbox.process_once() == 0
        assert crm.calls == []
    finally:
        await dialog.aclose()


@pytest.mark.asyncio
async def test_telegram_remains_a_crm_and_delivers_through_outbox():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'ok': True, 'result': {'message_id': 101}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    crm = TelegramCrm(InMemoryCrm(), 'test-token', 'test-chat', client=client)
    dialog, outbox = adapter(crm, True)
    try:
        await dialog.start('telegram', '+79990000001')
        reply = await dialog.handle_turn('telegram', '+79990000001', 'создай')
        card = next(event for event in reply.events if event.type == 'appeal_created')
        assert card.payload['delivery_mode'] == 'outbox'
        assert outbox.store.get('4422') is not None
        assert await outbox.process_once() == 1
        assert len(requests) == 1
        assert requests[0]['chat_id'] == 'test-chat'
        assert outbox.store.get('4422').status == 'sent'
    finally:
        await dialog.aclose()
        await crm.aclose()


def test_off_rejected_for_rules_and_telegram_requires_credentials(monkeypatch):
    monkeypatch.delenv('TELEGRAM_BOT_TOKEN', raising=False)
    monkeypatch.delenv('TELEGRAM_CHAT_ID', raising=False)
    with pytest.raises(RuntimeError, match='только с DIALOG_ADAPTER=cloudru'):
        build_container(Settings(data_dir=DATA, dialog_adapter='rules', crm_adapter='off',
                                 _env_file=None))
    with pytest.raises(RuntimeError, match='нужны TELEGRAM_BOT_TOKEN'):
        build_container(Settings(data_dir=DATA, dialog_adapter='cloudru', crm_adapter='telegram',
                                 telegram_bot_token='', telegram_chat_id='', _env_file=None))


@pytest.mark.parametrize(('mode', 'expected'), [('off', False), ('fake', True), ('telegram', True)])
def test_outbox_worker_obeys_explicit_switch(monkeypatch, mode, expected):
    from assistant.api import app as app_module

    started = threading.Event()

    class FakeOutbox:
        async def run(self, stop, interval=1):
            started.set()
            await stop.wait()

    class FakeContainer:
        def __init__(self):
            self.outbox = FakeOutbox()
            self.call_journal = None
            self.closeables = []

        async def warm(self):
            await asyncio.Event().wait()

        async def aclose(self):
            pass

        def sweep(self):
            pass

    monkeypatch.setattr(app_module, 'build_container', lambda _settings: FakeContainer())
    settings = Settings(dialog_adapter='cloudru', crm_adapter=mode, data_dir=DATA, _env_file=None)
    with TestClient(app_module.create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        assert started.wait(.2) is expected


def test_off_http_records_card_in_journal_without_outbox(tmp_path, monkeypatch):
    import sqlite3

    from assistant.api.app import create_app
    from assistant.api.dependencies import Container

    async def no_warm(_self):
        pass

    monkeypatch.setattr(Container, 'warm', no_warm)
    settings = Settings(
        data_dir=DATA, dialog_adapter='cloudru', crm_adapter='off',
        asr_adapter='fake', tts_adapter='fake', fm_api_key='test', llm_model='test',
        rag_key_id='id', rag_key_secret='secret', rag_base_url='https://example.invalid',
        call_log_enabled=True, call_log_path=str(tmp_path / 'calls.sqlite3'), _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        container = client.app.state.container
        container.orchestrator._factory = Agent
        start = client.post('/call/start', json={'phone': '+79990000001'}).json()
        call_id = start['call_id']
        result = client.post('/call/turn', json={
            'call_id': call_id, 'phone': '+79990000001', 'text': 'создай',
        }).json()
        card = next(event for event in result['events'] if event['type'] == 'appeal_created')
        assert card['payload']['appeal_id'] == '4422'
        assert card['payload']['delivery_mode'] == 'upstream'
        assert container.outbox.store.all() == []

    with sqlite3.connect(tmp_path / 'calls.sqlite3') as db:
        rows = db.execute("SELECT data_json FROM call_events WHERE kind='bot_generated'").fetchall()
    data = [json.loads(row[0]) for row in rows]
    assert any(any(event['type'] == 'appeal_created'
                   and event['payload']['appeal_id'] == '4422'
                   for event in item.get('events', [])) for item in data)


@pytest.mark.asyncio
async def test_off_does_not_dispatch_even_with_telegram_credentials(monkeypatch):
    def never_construct(*_args, **_kwargs):
        raise AssertionError('telegram client must not be constructed in off mode')

    monkeypatch.setattr('assistant.services.crm.telegram.TelegramCrm', never_construct)
    settings = Settings(
        data_dir=DATA, dialog_adapter='cloudru', crm_adapter='off',
        asr_adapter='fake', tts_adapter='fake', fm_api_key='test', llm_model='test',
        rag_key_id='id', rag_key_secret='secret', rag_base_url='https://example.invalid',
        telegram_bot_token='test-token', telegram_chat_id='test-chat', _env_file=None,
    )
    container = build_container(settings)
    container.orchestrator._factory = Agent
    try:
        await container.orchestrator.start('off-with-creds', '+79990000001')
        reply = await container.orchestrator.handle_turn('off-with-creds', '+79990000001', 'заявка')
        card = next(e for e in reply.events if e.type == 'appeal_created')
        assert card.payload['delivery_mode'] == 'upstream'
        assert container.outbox.store.all() == []
    finally:
        await container.aclose()
