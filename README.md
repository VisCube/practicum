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
│      ├── Asr (fake | vosk)                                │
│      ├── Tts (fake | piper)                               │
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
| Локальный ASR  | Vosk (опц.)                                                        |
| Локальный TTS  | Piper  (опц.)                                                            |
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
    asr/                  FakeAsr, VoskAsr
    tts/                  FakeTts, PiperTts
    llm/                  FakeLlm, OpenAICompatLlm
    crm/                  InMemoryCrm, TelegramCrm, Outbox
    knowledge_base/       MarkdownRetriever, FakeKnowledgeBase
    telephony/            FakeTelephony
  templates/
    phrases.py            все реплики робота в одном месте
  main.py                 точка входа: uvicorn.run

data/
  classifier.yaml         таксономия тем и описания слотов
  escalation.yaml         маркеры эскалации
  knowledge/              Markdown-база знаний

tests/unit/               test_api, test_dialog, test_nlu, test_services
Dockerfile
pyproject.toml
.env.example
```

---

## Установка и запуск

### Требования

Python **3.12+**. Локальные модели (Vosk, Piper) — опционально, скачиваются отдельно
(см. ниже).

### Установка

```bash
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[dev]"      # + pytest / ruff / mypy
pip install -e ".[local]"    # + vosk + piper-tts (если нужны локальные модели)
```

Скопируйте .env.example в .env и заполните нужные переменные:

| Переменная | Назначение |
|------------|------------|
| `APP_ENV` | `dev` / `prod` |
| `DEBUG_UI` | `true` — монтирует `/debug/*` и HTML-фасад |
| `LOG_LEVEL` | `INFO`, `DEBUG` … |
| `ASR_ADAPTER` | `fake` \| `vosk` |
| `TTS_ADAPTER` | `fake` \| `piper` |
| `LLM_ADAPTER` | `fake` \| `openai` |
| `CRM_ADAPTER` | `fake` \| `telegram` (\| Битрикс24 — в планах) |
| `LLM_BASE_URL` | 	параметры LLM-endpoint |
| `LLM_API_KEY` | ключ LLM |
| `LLM_MODEL` | имя модели |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | только для `CRM_ADAPTER=telegram` |
| `TELEGRAM_CHANNEL_USERNAME` | опционально: username публичного канала — на него ссылается заявка |
| `VOSK_MODEL_PATH` | путь к модели Vosk (для `ASR_ADAPTER=vosk`) |
| `PIPER_MODEL_PATH` | путь к модели Piper (для `TTS_ADAPTER=piper`) |

### Запуск

```bash
# через скрипт (установлен при `pip install -e .`)
assistant

# или напрямую
python -m assistant.main

# порт — через PORT (по умолчанию 8000)
```

Сервис слушает `0.0.0.0:8000`.

### Docker

```bash
docker build -t assistant .
docker run -p 8000:8080 \
  -e APP_ENV=prod -e DEBUG_UI=true \
  -e ASR_ADAPTER=fake -e TTS_ADAPTER=piper \
  assistant
```

Модели Vosk и Piper должны лежать в models/ до сборки образа — Dockerfile копирует их внутрь.

---

## Локальные модели речи

| Модель | Кому нужна | Откуда |
|--------|------------|-------|
| Vosk small **ru** | `ASR_ADAPTER=vosk` | <https://alphacephei.com/vosk/models> |
| Piper **ru_RU ruslan-medium** | `TTS_ADAPTER=piper` | Piper TTS / HuggingFace |

Скачайте, положите в models/, укажите пути в .env.

---

## API

### Контракт телефона (`POST /call/*`)

Один и тот же формат используется реальной телефонной платформой и дев-стендом.

**`POST /call/start`** — начало звонка, возвращает приветствие.

**`POST /call/turn`** — реплика клиента, возвращает ответ робота.

| Поле (запрос) | Тип | Обяз. | Описание |
|---------------|-----|-------|----------|
| `phone` | str | да | Телефон звонящего |
| `call_id` | str | нет\* | ID звонка от платформы; если нет — генерируется |
| `text` | str | да | Распознанная реплика клиента |

**Ответ `TurnResponse`** (общий для обоих методов):

| Поле | Тип | Описание                                                      |
|------|-----|---------------------------------------------------------------|
| `call_id` | str | ID звонка                                                     |
| `text` | str | Реплика робота (транскрипт)                                   |
| `audio_b64` | str \| null | Аудио в base64, если TTS отработал                            |
| `audio_mime` | str | `audio/ogg`                                                   |
| `heard` | str \| null | Что распознал ASR (только для дев-стенда)                     |
| `transfer` | bool | Попросить платформу переключить на оператора                  |
| `end_call` | bool | Попросить платформу завершить звонок                          |
| `state` | CallState | Текущее состояние диалога                                     |
| `events` | list[Event] | transfer, appeal_created, escalated…                                                      |

> Любая необработанная ошибка на этом пути → робот говорит «техническая ошибка» и переводит на оператора

### Дев-стенд (`/debug/*`, только `DEBUG_UI=true`)

| Метод | Путь | Назначение |
|-------|------|------------|
| `GET` | `/debug/` | 	HTML-стенд — эмуляция звонка в браузере |
| `GET` | `/debug/config` | Какие адаптеры включены, demo-номера |
| `POST` | `/debug/call/{call_id}/say` | Реплика текстом, минуя ASR |
| `POST` | `/debug/call/{call_id}/say-audio` | Реплика аудио (lpcm/ogg, заголовок `X-Phone`) |
| `GET` | `/debug/call/{call_id}` | 	Снимок сессии: транскрипт, слоты, topic, escalated |
| `DELETE` | `/debug/call/{call_id}` | Завершить звонок и удалить сессию |
| `GET` | `/debug/outbox` | Outbox — очередь отправки заявок в CRM |
| `POST` | `/debug/tts` | 	Синтезировать произвольный текст для подбора голоса |

### Health

`GET /health` → `{"status":"ok","env":"…"}`

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