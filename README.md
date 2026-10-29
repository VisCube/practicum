# Голосовой ассистент 1-й линии контакт-центра УК

Голосовой бот для контакт-центра управляющей компании. Принимает входящий звонок, выясняет причину обращения, собирает реквизиты и либо создаёт заявку в CRM, либо отвечает из базы знаний.

Два сценария:

- **appeal** — протечка, запах газа, перерасчёт, справка и т.д.: бот собирает ФИО / адрес / детали и кладёт заявку (Сделку) в CRM.
- **consult** — FAQ, тарифы, сроки: бот ищет ответ в Markdown-базе знаний, при наличии LLM — перефразирует его.

Если бот не понял — переспрашивает; если совсем не справился — переводит на живого оператора. Авария (пожар, потоп, газ) — немедленный перевод со спец-фразой. Заявка уходит в CRM фоновым воркером, горячий путь звонка не ждёт внешние сервисы.

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
│      ├── DialogRunner (DIALOG_ADAPTER=rules | cloudru)    │
│      │     ├── Orchestrator (rules: YAML + ML + handlers) │
│      │     └── CloudRuDialogAdapter (Cloud.ru LLM + RAG)  │
│      ├── Asr (fake | vosk | t-one | cloudru)              │
│      ├── Tts (fake | piper | silero)                      │
│      ├── Crm (fake | telegram | bitrix*) + Outbox         │
│      └── Llm (fake | openai | cloudru)                    │
└───────────────────────────────────────────────────────────┘
                                          * в планах
```

Каждый внешний сервис живёт за интерфейсом (`services/*/base.py`) и имеет fake-двойника — проект запускается и тестируется без каких-либо внешних зависимостей.

---

## Стек

| Компонент | Технология |
|-----------|-----------|
| Язык | Python ≥ 3.12 |
| Веб-фреймворк | FastAPI + Uvicorn |
| Проверка/сериализация | Pydantic v2, pydantic-settings |
| Конфигурация | Переменные окружения + `.env` |
| Локальный ASR | Vosk (опц.) / T-one (sherpa-onnx) |
| Облачный ASR | Cloud.ru Foundation Models (`openai/whisper-large-v3`) |
| Локальный TTS | Piper (опц.) / Silero |
| Диалог и RAG | Правила (`Orchestrator` + `MarkdownRetriever`) или агент Cloud.ru (LLM tool calling + Managed RAG) |
| LLM | Любой OpenAI-совместимый endpoint / Cloud.ru Foundation Models |
| CRM | fake (in-memory) / Telegram ; Битрикс24 — в планах |
| Тесты | pytest + pytest-asyncio |
| Упаковка | Docker (`Dockerfile`, модульный `ARG EXTRAS`) / `pip install -e ".[...]"` |

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
    classifier.py         гибридный классификатор (ML + keywords, YAML)
    escalation.py         детектор emergency / transfer / complaint (YAML)
    slots.py              RuleSlotExtractor + missing()
    yesno.py              распознавание да/нет
    text.py               нормализация, стемизация, числительные
    llm_tasks.py          классификация и слоты через LLM
  services/               каждый сервис: base.py + fake.py + реальный адаптер
    asr/                  FakeAsr, VoskAsr, ToneAsr, CloudRuAsr
    tts/                  FakeTts, PiperTts, SileroTts
    llm/                  FakeLlm, OpenAICompatLlm
    crm/                  InMemoryCrm, TelegramCrm, Outbox
    cloudru/              мост и адаптер звонка к моделям и Managed RAG Cloud.ru
    telephony/            FakeTelephony
  templates/phrases.py    Все фразы робота в одном dict
  utils/                  Вспомогательные модули
```

---

## Конфигурация содержания

### `data/classifier.yaml`

Правится под реальное содержимое. Содержит:
* **`slots`** — описание каждого реквизита: `id`, `label`, `question`, `kind`
  (`number` / `text` / `address` / `name` / `choice`), `choices` (для `choice`),
  `cues` (слова-подсказки для распознавания чисел из свободной речи).
* **`topics`** — таксономия обращений: `id`, `name` (читается вслух!), `kind`
  (`appeal` / `consult`), `bitrix_id` (значение в CRM), `risk_level`
  (`low` / `normal` / `critical`), `keywords` (корни слов), `required_slots`.

Классификатор гибридный:
- **ML-модель** (если есть `model.pkl`) — sklearn pipeline на лемматизированном тексте.
- **Keyword fallback** — если модель не уверена (`conf < min_confidence`) или
  предсказанный slug не подтверждается keywords YAML-топика.
- При равенстве keywords — приоритет у `critical`, потом порядок в YAML.

YAML-топики сохраняют `kind`, `required_slots`, `bitrix_id` — модель возвращает slug,
классификатор подставляет полный YAML-топик по совпадению.

### `data/escalation.yaml`

* **`emergency`** — авария (пожар, потоп, прорыв, газ) → немедленный перевод с
  спец-фразой. Ключ `say` у каждого маркера ссылается на запись в `emergency_say`.
* **`emergency_say`** — тексты фраз для аварий: `fire` (112), `flood` (диспетчер),
  `gas` (104). Валидируется при загрузке: ключ из `say` должен существовать.
* **`transfer`** — клиент просит живого человека → перевод немедленно.
* **`complaint`** — угроза жалобы (прокуратура, ГЖИ, суд) → заявка с флагом,
  разговор не прерывается.

### `data/knowledge/*.md`

Markdown-файлы для сценария `consult`. Заголовок `#` или `##` — вопрос, текст ниже —
ответ. Запрашивается `MarkdownRetriever` по ключевым словам темы.

### `src/assistant/templates/phrases.py`

Все реплики робота — в одном `dict`. Формулировки править здесь, а не в коде сценариев.

---

## Приоритеты сигналов

На каждом ходе реплика проверяется в порядке:

1. **Emergency** — авария (пожар, потоп, прорыв, газ) → перевод немедленно со
   спец-фразой из `emergency_say`, выше оператора и жалобы.
2. **Transfer** — клиент просит оператора → перевод.
3. **Complaint** — угроза жалобы → флаг в заявке (или перевод, если
   `transfer_on_complaint=true`).
4. **Goodbye** — прощание (если сессия в `DONE` или `TOPIC` без темы и без попыток).
5. **Тема / обработчик** — классификация и вызов соответствующего handler'а.

---

## Сценарии диалога

### `appeal` (заявка) — `IntakeHandler`

```
TOPIC → [IDENTIFY →] SLOTS → CONFIRM → DONE
```

1. Определение темы (ML-модель → keyword fallback → LLM-фолбэк).
2. Если собственник не опознан по телефону → идентификация (ФИО, адрес, квартира).
3. Сбор обязательных слотов темы (правила → LLM → retry).
4. Подтверждение сводки.
5. Заявка в outbox → фоновая отправка в CRM.

### `consult` (консультация) — `ConsultationHandler`

Ответ из Markdown-базы знаний по теме; LLM может дополнить/перефразировать.
При двух промахах подряд — перевод на оператора.

### Машина состояний (`CallState`)

```
greeting → topic → [identify →] slots → confirm → done
                      ↘                            ↘
                      transferred ← emergency      transferred
```

- **Emergency** на любом шаге → `transferred` со спец-фразой из `emergency_say`.
- **Transfer** — перевод (emergency / просьба клиента / угроза жалобы / тех. ошибка).
- Сессия живёт в памяти 30 минут с момента последнего обращения.

---

## Модульная установка и сборка Docker

Локальные модели и облачные клиенты вынесены в независимые extras (`pyproject.toml` и `ARG EXTRAS` в `Dockerfile`):

| Модуль | Назначение | Модель в `./models/` |
|--------|------------|----------------------|
| `cloudru` | Cloud.ru: LLM-агент, Whisper STT, Managed RAG | — |
| `silero` | Локальный Silero TTS + нормализация/ударения | `models/v5_cis_base.pt` |
| `piper` | Локальный Piper TTS | `models/ru_RU-ruslan-medium.onnx*` |
| `tone` | Локальный T-one ASR (`sherpa-onnx`) | `models/t-one/` |
| `vosk` | Локальный Vosk ASR | `models/vosk-model-small-ru-0.22` |
| `clf` | Локальный ML-классификатор тем | `models/model.pkl` |
| `local` | Все локальные модули сразу (`vosk,tone,piper,silero,clf`) | все 5 моделей |
| `all` | `local` + `cloudru` | все 5 моделей |

Примеры:

```bash
# Облачный режим Cloud.ru + локальный синтез Silero TTS
pip install -e ".[cloudru,silero]"
docker build --build-arg EXTRAS="cloudru,silero" -t dispatch:cloudru .

# Полностью локальная сборка (по умолчанию)
docker build -t dispatch:local .
```

Для облачного режима в `.env` задаются `DIALOG_ADAPTER=cloudru`, `ASR_ADAPTER=cloudru`, `TTS_ADAPTER=silero`, а также ключи `FM_API_KEY`, `LLM_MODEL`, `STT_MODEL`, `RAG_KEY_ID`, `RAG_KEY_SECRET`, `RAG_BASE_URL` (см. `.env.example`).

---

## Диагностика дев-стенда

Стенд отдаётся с `Cache-Control: no-store`, а `/debug/config` возвращает `ui_build` —
отпечаток статики. В настройках стенда строка **`ui:`** показывает, какая сборка
реально открыта в браузере: если там старое значение — страница из кеша, нужен
жёсткий refresh.

Каждый ход приносит тайминги этапов (`timings`): `asr`, `dialog`, `tts` в
миллисекундах. В диалоге стенда они не показываются — смотрим в консоли
браузера (`[turn] этапы, мс: …`) или в ответе API напрямую.

Если серверное аудио не воспроизвелось (формат не поддержан, запрет автоплея,
воспроизведение встало) — стенд не оставляет пользователя в тишине: текст
читается голосом браузера, а причина пишется в консоль с префиксом `[tts]` и
один раз показывается в статусе. Проба синтеза показывает формат и размер
полученного аудио, а при сбое — точную причину.

Завершённый звонок не закрывается сам: экран с транскриптом остаётся, пока
пользователь не нажмёт «Выйти из звонка». AudioContext при этом не закрывается —
иначе следующий звонок мог стартовать молча и с медленным микрофоном.

`POST /debug/warmup` прогревает TTS/LLM (короткая фраза + один дешёвый запрос),
чтобы первый ход не платил за холодный старт. Приложение греет модели в фоне на
старте, стенд — при открытии и на старте звонка.

---

## Тесты

```bash
pytest                              # все тесты
pytest tests/unit/test_dialog.py    # только сценарные
pytest tests/unit/test_nlu.py       # NLU (классификатор, escalation, слоты)
pytest tests/unit/test_api.py       # интеграционные через TestClient
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
* LLM — опциональный фолбэк; без неё всё работает на правилах и keywords.
* ML-модель — опциональна; без неё классификатор работает на keywords.
* `InMemorySessionStore` — для горизонтального масштабирования нужен внешний store (Redis и т.п.).
