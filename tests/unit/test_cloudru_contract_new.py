"""Независимые проверки новой границы вызова агента."""
import asyncio
import threading

import pytest

from assistant.core.session_store import InMemorySessionStore
from assistant.services.cloudru.dialog import CloudRuDialogAdapter
from assistant.services.crm.fake import InMemoryCrm
from assistant.services.crm.outbox import InMemoryOutboxStore, Outbox
from assistant.services.telephony.fake import FakeTelephony


def make_dialog(timeout=0.02):
    crm = InMemoryCrm()
    return CloudRuDialogAdapter(
        store=InMemorySessionStore(), crm=crm,
        outbox=Outbox(crm, InMemoryOutboxStore()), telephony=FakeTelephony(),
        agent_timeout=timeout, max_concurrency=4, assistant_factory=lambda phone: None,
    )


@pytest.mark.asyncio
async def test_timed_out_work_keeps_slot_until_thread_finishes():
    dialog = make_dialog()
    gate = threading.Event()
    started = threading.Event()

    class BlockingAgent:
        def handle_message(self, _text):
            started.set()
            gate.wait(2)
            return 'ok', []

    try:
        with pytest.raises(TimeoutError):
            await dialog._run_agent_call(BlockingAgent(), 'hello')
        assert started.is_set()
        assert dialog._semaphore._value == 3
        gate.set()
        for _ in range(100):
            if dialog._semaphore._value == 4:
                break
            await asyncio.sleep(0.01)
        assert dialog._semaphore._value == 4
    finally:
        gate.set()
        await dialog.aclose()


@pytest.mark.asyncio
async def test_stop_iteration_becomes_ordinary_agent_failure():
    dialog = make_dialog(1)

    class BrokenAgent:
        def handle_message(self, _text):
            raise StopIteration('unknown category')

    try:
        with pytest.raises(RuntimeError, match='upstream tool raised StopIteration'):
            await dialog._run_agent_call(BrokenAgent(), 'hello')
    finally:
        await dialog.aclose()


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_slot_until_thread_finishes():
    dialog = make_dialog(2)
    gate = threading.Event()
    started = threading.Event()

    class BlockingAgent:
        def handle_message(self, _text):
            started.set()
            gate.wait(2)
            return 'ok', []

    try:
        task = asyncio.create_task(dialog._run_agent_call(BlockingAgent(), 'hello'))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert dialog._semaphore._value == 3
        gate.set()
        for _ in range(100):
            if dialog._semaphore._value == 4:
                break
            await asyncio.sleep(0.01)
        assert dialog._semaphore._value == 4
    finally:
        gate.set()
        await dialog.aclose()


def test_ticket_result_must_be_explicitly_success_or_missing_fields():
    from assistant.services.cloudru.dialog import _is_recognized_ticket_result

    assert _is_recognized_ticket_result({'id': 1234})
    assert _is_recognized_ticket_result({'missing_fields': ['street']})
    for result in (None, '', {}, {'error': 'failed'}, {'id': True}, {'id': ''},
                   {'missing_fields': 'street'}):
        assert not _is_recognized_ticket_result(result)


@pytest.mark.asyncio
async def test_public_only_agent_handles_multiple_turns():
    seen = []

    class PublicOnly:
        def __init__(self, phone):
            self.phone = phone

        def handle_message(self, text):
            seen.append(text)
            return 'ответ', []

    dialog = make_dialog(1)
    dialog._factory = PublicOnly
    try:
        await dialog.start('public', '+79990000001')
        result = await dialog.handle_turn('public', '+79990000001', 'вопрос')
        assert result.text == 'ответ'
        assert seen == ['Алло', 'вопрос']
    finally:
        await dialog.aclose()


@pytest.mark.asyncio
async def test_barge_waits_for_inflight_agent_and_preserves_ticket_event():
    import json
    from types import SimpleNamespace

    gate = threading.Event()
    entered = threading.Event()
    calls = []

    class PublicOnly:
        def __init__(self, phone):
            self.phone = phone

        def handle_message(self, text):
            calls.append(text)
            if text == 'Алло':
                return 'здравствуйте', []
            if text == 'создать':
                entered.set()
                gate.wait(2)
                return 'создано', [SimpleNamespace(name='create_ticket',
                    arguments={'category': 'Тест', 'phone': self.phone, 'description': 'тест'},
                    result=json.dumps({'id': 4321, 'category': 'Тест'}))]
            return 'принято', []

    dialog = make_dialog(1)
    dialog._factory = PublicOnly
    try:
        await dialog.start('barge', '+79990000001')
        first = asyncio.create_task(dialog.handle_turn('barge', '+79990000001', 'создать'))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.01)
        assert entered.is_set()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(dialog.handle_turn('barge', '+79990000001', 'дополнение'))
        await asyncio.sleep(.03)
        assert calls == ['Алло', 'создать']
        gate.set()
        reply = await second
        assert calls == ['Алло', 'создать', 'дополнение']
        assert any(event.type == 'appeal_created' for event in reply.events)
        assert dialog.outbox.store.get('4321') is not None
    finally:
        gate.set()
        await dialog.aclose()


@pytest.mark.asyncio
async def test_timeout_transfers_instead_of_repeating_uncertain_action():
    gate = threading.Event()
    calls = []

    class SlowAgent:
        def __init__(self, phone):
            self.phone = phone

        def handle_message(self, text):
            calls.append(text)
            if text == 'Алло':
                return 'здравствуйте', []
            gate.wait(2)
            return 'поздно', []

    dialog = make_dialog(.02)
    dialog._factory = SlowAgent
    try:
        await dialog.start('uncertain', '+79990000001')
        first = await dialog.handle_turn('uncertain', '+79990000001', 'создать')
        assert first.transfer is True
        second = await dialog.handle_turn('uncertain', '+79990000001', 'ещё раз')
        assert second.transfer is True
        assert calls == ['Алло', 'создать']
    finally:
        gate.set()
        await dialog.aclose()


@pytest.mark.asyncio
async def test_cancelled_start_is_not_repeated():
    gate = threading.Event()
    entered = threading.Event()
    calls = []

    class PublicOnly:
        def __init__(self, phone):
            self.phone = phone

        def handle_message(self, text):
            calls.append(text)
            if text == 'Алло':
                entered.set()
                gate.wait(2)
            return 'ответ', []

    dialog = make_dialog(1)
    dialog._factory = PublicOnly
    try:
        start = asyncio.create_task(dialog.start('start-barge', '+79990000001'))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.01)
        assert entered.is_set()
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start
        next_turn = asyncio.create_task(dialog.handle_turn('start-barge', '+79990000001', 'вопрос'))
        await asyncio.sleep(.03)
        assert calls == ['Алло']
        gate.set()
        await next_turn
        assert calls == ['Алло', 'вопрос']
    finally:
        gate.set()
        await dialog.aclose()


def test_phone_contract():
    from assistant.services.cloudru.bridge import normalize_caller_phone

    assert normalize_caller_phone('+7 (999) 000-00-01') == '+79990000001'
    assert normalize_caller_phone('79990000001') == '+79990000001'
    for value in ('89990000001', '9990000001', '+799900000011', 'abc', ''):
        with pytest.raises(ValueError):
            normalize_caller_phone(value)


@pytest.mark.asyncio
async def test_cancelled_turn_does_not_restore_deleted_session():
    gate = threading.Event()
    entered = threading.Event()

    class PublicOnly:
        def __init__(self, phone):
            self.phone = phone

        def handle_message(self, text):
            if text != 'Алло':
                entered.set()
                gate.wait(2)
            return 'ответ', []

    dialog = make_dialog(1)
    dialog._factory = PublicOnly
    try:
        await dialog.start('closed', '+79990000001')
        turn = asyncio.create_task(dialog.handle_turn('closed', '+79990000001', 'вопрос'))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.01)
        assert entered.is_set()
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        dialog.store.delete('closed')
        dialog.drop_call('closed')
        gate.set()
        await asyncio.sleep(.05)
        assert dialog.store.get('closed') is None
        assert 'closed' not in dialog._agents
    finally:
        gate.set()
        await dialog.aclose()


@pytest.mark.asyncio
async def test_wait_for_worker_capacity_has_timeout():
    dialog = make_dialog(.03)
    for _ in range(4):
        await dialog._semaphore.acquire()

    class Agent:
        def handle_message(self, _text):
            raise AssertionError('must not run')

    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(dialog._run_agent_call(Agent(), 'hello'), timeout=.3)
    finally:
        for _ in range(4):
            dialog._semaphore.release()
        await dialog.aclose()


@pytest.mark.asyncio
async def test_dropped_call_id_can_be_safely_reused_after_pending_start():
    gate = threading.Event()
    entered = threading.Event()
    calls = []

    class Agent:
        def __init__(self, phone):
            self.phone = phone

        def handle_message(self, text):
            calls.append((self.phone, text))
            if self.phone == '+79990000001':
                entered.set()
                gate.wait(2)
            return self.phone, []

    dialog = make_dialog(1)
    dialog._factory = Agent
    try:
        old = asyncio.create_task(dialog.start('reused', '+79990000001'))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.01)
        assert entered.is_set()
        old.cancel()
        with pytest.raises(asyncio.CancelledError):
            await old
        dialog.drop_call('reused')
        new = asyncio.create_task(dialog.start('reused', '+79990000002'))
        gate.set()
        reply = await new
        assert reply.text == '+79990000002'
        assert calls == [('+79990000001', 'Алло'), ('+79990000002', 'Алло')]
    finally:
        gate.set()
        await dialog.aclose()


def test_missing_optional_category_metadata_does_not_prevent_labels(monkeypatch, tmp_path):
    import assistant.services.cloudru.bridge as bridge

    bridge.load_cloudru_categories.cache_clear()
    bridge.load_cloudru_slot_labels.cache_clear()
    monkeypatch.setattr(bridge, 'UPSTREAM_CATEGORIES', tmp_path / 'missing.json')
    try:
        assert bridge.load_cloudru_categories() == ()
        assert bridge.load_cloudru_slot_labels()['description'] == 'Описание'
    finally:
        bridge.load_cloudru_categories.cache_clear()
        bridge.load_cloudru_slot_labels.cache_clear()


def test_missing_demo_residents_does_not_prevent_startup(monkeypatch):
    import assistant.services.cloudru.bridge as bridge

    def missing():
        raise ModuleNotFoundError('demo CRM changed')

    monkeypatch.setattr(bridge, '_load_residents_module', missing)
    assert bridge.load_cloudru_demo_phones() == []


@pytest.mark.asyncio
async def test_real_upstream_assistant_with_mocked_llm(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from assistant.services.cloudru.bridge import load_assistant_cls

    values = {
        'FM_API_KEY': 'test', 'FM_BASE_URL': 'https://example.invalid',
        'LLM_MODEL': 'test', 'STT_MODEL': 'test', 'RAG_KEY_ID': 'test',
        'RAG_KEY_SECRET': 'test', 'RAG_BASE_URL': 'https://example.invalid',
        'IAM_TOKEN_URL': 'https://example.invalid',
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    cls = load_assistant_cls()
    module = importlib.import_module(cls.__module__)

    class MockClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **_kwargs):
            message = SimpleNamespace(
                tool_calls=None, content='Проверочный ответ',
                model_dump=lambda **_options: {'role': 'assistant', 'content': 'Проверочный ответ'},
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(module, 'OpenAI', MockClient)
    dialog = make_dialog(1)
    dialog._factory = cls
    try:
        first = await dialog.start('actual-upstream', '+79990000001')
        second = await dialog.handle_turn('actual-upstream', '+79990000001', 'проверка')
        assert first.text == second.text == 'Проверочный ответ'
    finally:
        await dialog.aclose()


@pytest.mark.asyncio
async def test_real_upstream_tool_call_becomes_outbox_event(monkeypatch):
    import importlib
    import json
    from types import SimpleNamespace

    from openai.types.chat import ChatCompletionMessageFunctionToolCall

    from assistant.services.cloudru.bridge import load_assistant_cls

    for key, value in {
        'FM_API_KEY': 'test', 'FM_BASE_URL': 'https://example.invalid',
        'LLM_MODEL': 'test', 'STT_MODEL': 'test', 'RAG_KEY_ID': 'test',
        'RAG_KEY_SECRET': 'test', 'RAG_BASE_URL': 'https://example.invalid',
        'IAM_TOKEN_URL': 'https://example.invalid',
    }.items():
        monkeypatch.setenv(key, value)
    cls = load_assistant_cls()
    module = importlib.import_module(cls.__module__)
    tickets = importlib.import_module('src.tools.crm.tickets')
    initial = len(tickets.TICKETS)
    monkeypatch.setattr(tickets, 'randint', lambda *_args: 99992)
    args = {'category': 'Лифты / Неисправность лифта', 'phone': '+79990000001',
            'description': 'Лифт не работает', 'street': 'Мира', 'house': '19',
            'resident': 'Иванов Иван Иванович', 'entry': 1, 'details': 'Застрял'}
    tool = ChatCompletionMessageFunctionToolCall.model_validate({
        'id': 'call-test', 'type': 'function',
        'function': {'name': 'create_ticket', 'arguments': json.dumps(args, ensure_ascii=False)},
    })

    class MockClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)
            self.count = 0

        def create(self, **_kwargs):
            self.count += 1
            if self.count == 2:
                payload = {'role': 'assistant', 'tool_calls': [tool.model_dump()]}
                msg = SimpleNamespace(tool_calls=[tool], content=None,
                                      model_dump=lambda **_options: payload)
            else:
                payload = {'role': 'assistant', 'content': 'Проверочный ответ'}
                msg = SimpleNamespace(tool_calls=None, content='Проверочный ответ',
                                      model_dump=lambda **_options: payload)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(module, 'OpenAI', MockClient)
    dialog = make_dialog(1)
    dialog._factory = cls
    try:
        await dialog.start('real-tool', '+79990000001')
        reply = await dialog.handle_turn('real-tool', '+79990000001', 'заявка')
        assert [event.type for event in reply.events] == ['tool_call', 'appeal_created']
        assert dialog.outbox.store.get('99992') is not None
    finally:
        del tickets.TICKETS[initial:]
        await dialog.aclose()


@pytest.mark.asyncio
async def test_unknown_category_is_not_invented_or_converted_to_ticket(monkeypatch):
    import importlib
    import json
    from types import SimpleNamespace

    from openai.types.chat import ChatCompletionMessageFunctionToolCall

    from assistant.services.cloudru.bridge import load_assistant_cls

    for key, value in {
        'FM_API_KEY': 'test', 'FM_BASE_URL': 'https://example.invalid',
        'LLM_MODEL': 'test', 'STT_MODEL': 'test', 'RAG_KEY_ID': 'test',
        'RAG_KEY_SECRET': 'test', 'RAG_BASE_URL': 'https://example.invalid',
        'IAM_TOKEN_URL': 'https://example.invalid',
    }.items():
        monkeypatch.setenv(key, value)
    cls = load_assistant_cls()
    module = importlib.import_module(cls.__module__)
    tool = ChatCompletionMessageFunctionToolCall.model_validate({
        'id': 'bad-cat', 'type': 'function',
        'function': {'name': 'get_category', 'arguments': json.dumps({'category': 'xyz nonsense'})},
    })

    class MockClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)
            self.count = 0

        def create(self, **_kwargs):
            self.count += 1
            if self.count == 2:
                payload = {'role': 'assistant', 'tool_calls': [tool.model_dump()]}
                msg = SimpleNamespace(tool_calls=[tool], content=None,
                                      model_dump=lambda **_options: payload)
            else:
                payload = {'role': 'assistant', 'content': 'Здравствуйте'}
                msg = SimpleNamespace(tool_calls=None, content='Здравствуйте',
                                      model_dump=lambda **_options: payload)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(module, 'OpenAI', MockClient)
    dialog = make_dialog(1)
    dialog._factory = cls
    try:
        await dialog.start('unknown-cat', '+79990000001')
        reply = await dialog.handle_turn('unknown-cat', '+79990000001', 'незнакомая тема')
        assert reply.transfer
        assert not dialog.outbox.store.all()
    finally:
        await dialog.aclose()


@pytest.mark.asyncio
async def test_failure_in_event_processing_fails_closed_without_replaying(monkeypatch):
    seen = []

    class Agent:
        def __init__(self, phone):
            self.phone = phone

        def handle_message(self, text):
            seen.append(text)
            return 'ок', []

    dialog = make_dialog(1)
    dialog._factory = Agent
    try:
        await dialog.start('processing-failure', '+79990000001')

        async def fail_processing(*_args):
            raise RuntimeError('failed processing tool event')

        monkeypatch.setattr(dialog, '_apply_tool_calls', fail_processing)
        result = await dialog.handle_turn('processing-failure', '+79990000001', 'создать')
        assert result.transfer
        again = await dialog.handle_turn('processing-failure', '+79990000001', 'повтори')
        assert again.transfer
        assert seen == ['Алло', 'создать']
    finally:
        await dialog.aclose()


@pytest.mark.asyncio
async def test_completed_call_cleanup_releases_bookkeeping():
    class Agent:
        def __init__(self, phone):
            self.phone = phone

        def handle_message(self, _text):
            return 'ответ', []

    dialog = make_dialog(1)
    dialog._factory = Agent
    try:
        await dialog.start('cleanup', '+79990000001')
        await dialog.handle_turn('cleanup', '+79990000001', 'вопрос')
        dialog.store.delete('cleanup')
        dialog.drop_call('cleanup')
        assert 'cleanup' not in dialog._start_locks
        assert 'cleanup' not in dialog._turn_locks
        assert 'cleanup' not in dialog._current_sessions
        assert 'cleanup' not in dialog._agents
    finally:
        await dialog.aclose()
