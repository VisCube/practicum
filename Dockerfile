FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install -U pip && pip install ".[local]"

COPY models/vosk-model-small-ru-0.22 ./models/vosk-model-small-ru-0.22
COPY models/ru_RU-ruslan-medium.onnx models/ru_RU-ruslan-medium.onnx.json ./models/
COPY data ./data

RUN useradd -m app && chown -R app:app /app
USER app

ENV APP_ENV=prod DEBUG_UI=false ASR_ADAPTER=vosk TTS_ADAPTER=piper PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "assistant.main"]