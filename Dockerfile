FROM python:3.12-slim

# Компоненты сборки через запятую:
#   cloudru  — облачные модели Cloud.ru (LLM, Whisper STT, Managed RAG)
#   silero   — локальный Silero TTS (models/v5_cis_base.pt)
#   piper    — локальный Piper TTS (models/ru_RU-ruslan-medium.onnx*)
#   tone     — локальный T-one ASR (models/t-one/)
#   vosk     — локальный Vosk ASR (models/vosk-model-small-ru-0.22)
#   clf      — локальный ML-классификатор (models/model.pkl)
#   local    — все локальные модели сразу (по умолчанию)
#   all      — local + cloudru
# Пример для Cloud.ru + Silero TTS:
#   docker build --build-arg EXTRAS="cloudru,silero" .
ARG EXTRAS="local"

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=0 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY data ./data
COPY src ./src

RUN --mount=type=bind,source=.,target=/ctx \
    set -eu; \
    EXTRAS_CLEAN="$(printf '%s' "${EXTRAS}" | tr -d ' ')"; \
    mkdir -p ./models; \
    has() { case ",${EXTRAS_CLEAN}," in *,"$1",*|*,local,*|*,all,*) return 0 ;; *) return 1 ;; esac; }; \
    need() { [ -e "/ctx/$1" ] || { echo "Missing required model for EXTRAS=${EXTRAS_CLEAN}: $1" >&2; exit 1; }; }; \
    if has vosk;   then need models/vosk-model-small-ru-0.22 && cp -r /ctx/models/vosk-model-small-ru-0.22 ./models/; fi; \
    if has piper;  then need models/ru_RU-ruslan-medium.onnx && need models/ru_RU-ruslan-medium.onnx.json && cp /ctx/models/ru_RU-ruslan-medium.onnx /ctx/models/ru_RU-ruslan-medium.onnx.json ./models/; fi; \
    if has clf;    then need models/model.pkl && cp /ctx/models/model.pkl ./models/model.pkl; fi; \
    if has silero; then need models/v5_cis_base.pt && cp /ctx/models/v5_cis_base.pt ./models/v5_cis_base.pt; fi; \
    if has tone;   then need models/t-one && cp -r /ctx/models/t-one ./models/t-one; fi

RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    EXTRAS_CLEAN="$(printf '%s' "${EXTRAS}" | tr -d ' ')"; \
    pip install -U pip && \
    if [ -n "${EXTRAS_CLEAN}" ]; then pip install ".[${EXTRAS_CLEAN}]"; else pip install .; fi

RUN useradd -m app && chown -R app:app /app
USER app

ENV APP_ENV=prod DEBUG_UI=false PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "assistant.main"]
