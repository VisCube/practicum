# Голосовой ассистент 1-й линии контакт-центра УК

Голосовой бот для контакт-центра управляющей компании. Принимает входящий звонок, выясняет причину обращения, собирает реквизиты и либо создаёт заявку в CRM, либо отвечает из базы знаний.

Два сценария:

- appeal — протечка, запах газа, перерасчёт, справка и т.д.: бот собирает ФИО / адрес / детали и кладёт заявку (Сделку) в CRM.
- consult — FAQ, тарифы, сроки: бот ищет ответ в Markdown-базе знаний, при наличии LLM — перефразирует его.

Если бот не понял — переспрашивает; если совсем не справился — переводит на живого оператора. Заявка уходит в CRM фоновым воркером, горячий путь звонка не ждёт внешние сервисы.

---

## Архитектура

```
Телефония / Дев-стенд (браузер)
        │  HTTP (text / audio)
        ▼
┌───────────────────────────────────────────────────────────┐
│  FastAPI (assistant.api.app:create_app)                   │
│                                                           │
│   routes/telephony  /call/start, /call/turn               │
│   routes/debug      /debug/* (только DEBUG_UI=true)       │
│                                                           │
│   dependencies.Container                                  │
│      ├── Orchestrator            ← ядро диалога           │
│      │     ├── SessionStore (in-memory, TTL)              │
│      │     ├── TopicClassifier (rule-based, YAML)         │
│      │     ├── EscalationDetector (YAML)                  │
│      │     ├── Handlers: Intake / Consultation            │
│      │     └── RuleSlotExtractor + (опц.) LlmTasks        │
│      ├── Asr (fake | vosk | t-one)                        │
│      ├── Tts (fake | piper | silero)                      │
│      ├── Crm (fake | telegram | bitrix*) + Outbox         │
│      └── Llm (fake | openai-совместимый)                  │
└───────────────────────────────────────────────────────────┘
                                          * в планах
```

Каждый внешний сервис живёт за интерфейсом (`services/*/base.py`) и имеет fake-двойника — проект запускается и тестируется без каких-либо внешних зависимостей.

---

## Стек

| Компонент | Технология                                                         |
|-----------|--------------------------------------------------------------------|
| Язык | Python ≥ 3.12                                                      |
| Веб-фреймворк | FastAPI + Uvicorn                                                  |
| Проверка/сериализация | Pydantic v2, pydantic-settings                                     |
| Конфигурация | Переменные окружения + `.env`                                      |
| Локальный ASR  | Vosk (опц.) / T-one (sherpa-onnx)                                   |
| Локальный TTS  | Piper  (опц.) / Silero                                                             |
| LLM  | Любой OpenAI-совместимый endpoint (YandexGPT, OpenRouter, Ollama…) |
| CRM | fake (in-memory) / Telegram ; Битрикс24 — в планах                 |
| Тесты | pytest + pytest-asyncio                                            |
| Упаковка | Docker (`Dockerfile`) / `pip install -e ".[local]"`                |

---

## Структура проекта

```
src/assistant/
  api/
    app.py                create_app() — фабрика приложения
    dependencies.py       build_container() — единственное место с импортами адаптеров
    routes/telephony.py   POST /call/start, POST /call/turn
    routes/debug.py       дев-стенд: /debug/*, HTML-фасад
    schemas/dto.py        StartRequest, TurnRequest, TurnResponse, SessionView…
    static/index.html     фронт дев-стенда
  core/
    config.py             Settings (env + .env)
    models.py             доменные модели: CallState, Topic, SlotSpec, Appeal…
    orchestrator.py       один ход диалога — от текста до реплики
    session_store.py      InMemorySessionStore с TTL
  handlers/
    base.py               интерфейс Handler
    intake.py             сценарий appeal: реквизиты → заявка → outbox
    consultation.py       сценарий consult: ответ из базы знаний
  nlu/
    classifier.py         keyword-классификатор тем (YAML)
    escalation.py         детектор «просит оператора / угроза жалобы»
    slots.py              RuleSlotExtractor + missing()
    yesno.py              распознавание да/нет
    text.py               нормализация, стемизация, числительные
    llm_tasks.py          классификация и слоты через LLM
  services/               каждый сервис: base.py + fake.py + реальный адаптер
    asr/                  FakeAsr, VoskAsr, ToneAsr
    tts/                  FakeTts, PiperTts, SileroTts
    llm/                  FakeLlm, OpenAICompatLlm
    crm/                  InMemoryCrm, TelegramCrm, Outbox
    telephony/            FakeTelephony
  templates/phrases.py    Все фразы робота в одном dict
  utils/                  Вспомогательные модули

---

## Конфигурация содержания

### `data/classifier.yaml`

Правится под реальное содержимое. Содержит:
* **`slots`** — описание каждого реквизита: `id`, `label`, `question`, `kind`
  (`number` / `text` / `address` / `name` / `choice`), `choices` (для `choice`),
  `cues` (слова-подсказки для распознавания).
* **`topics`** — таксономия обращений: `id`, `name` (читается вслух!), `kind`
  (`appeal` / `consult`), `bitrix_id` (значение в CRM), `risk_level`
  (`low` / `normal` / `critical`), `keywords` (корни слов), `required_slots`.

Классификация по числу совпавших keyword-корней; при ничьей — приоритет у `critical`.

### `data/escalation.yaml`

* **`transfer`** — клиент просит живого человека → перевод немедленно.
* **`complaint`** — угроза жалобы (прокуратура, ГЖИ, суд) → заявка с флагом, разговор не прерывается.

### `data/knowledge/*.md`

Markdown-файлы для сценария `consult`. Заголовок `#` или `##` — вопрос, текст ниже —
ответ. Запрашивается `MarkdownRetriever` по ключевым словам темы.

### `src/assistant/templates/phrases.py`

Все реплики робота — в одном `dict`. Формулировки править здесь, а не в коде сценариев.

---

## Сценарии диалога

### `appeal` (заявка) — `IntakeHandler`

```
TOPIC → [IDENTIFY →] SLOTS → CONFIRM → DONE
```

1. Определение темы по ключевым словам (правила → LLM-фолбэк).
2. Если собственник не опознан по телефону → идентификация (ФИО, адрес, квартира).
3. Сбор обязательных слотов темы.
4. Подтверждение сводки.
5. Заявка в outbox → фоновая отправка в CRM.

### `consult` (консультация) — `ConsultationHandler`

Ответ из Markdown-базы знаний по теме; LLM может дополнить/перефразировать.

### Машина состояний (`CallState`)

```
greeting → topic → [identify →] slots → confirm → done
                                              ↘ transferred
```
`transferred` — перевод (просьба клиента / угроза жалобы / техническая ошибка).
Сессия живёт в памяти 30 минут с момента последнего обращения.

---

## Тесты

```bash
pytest                              # все тесты
pytest tests/unit/test_dialog.py    # только сценарные
```

Тесты всегда работают на fake-адаптерах, без внешних зависимостей.

```bash
ruff check src tests
ruff format --check src tests
mypy src/assistant
```

---

## Ограничения и планы

* `CRM_ADAPTER=bitrix` — интерфейс есть, клиент не реализован.
* LLM — опциональный фолбэк; без неё всё работает на правилах.
* `InMemorySessionStore` — для горизонтального масштабирования нужен внешний store (Redis и т.п.).
