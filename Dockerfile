FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY src ./src
RUN pip install -U pip && pip install ".[local]"

COPY models/vosk-model-small-ru-0.22 ./models/vosk-model-small-ru-0.22
COPY models/ru_RU-ruslan-medium.onnx models/ru_RU-ruslan-medium.onnx.json ./models/
COPY models/t-one ./models/t-one/
COPY models/v5_cis_base.pt ./models/

COPY data ./data

ENV APP_ENV=prod DEBUG_UI=true ASR_ADAPTER=vosk TTS_ADAPTER=piper PORT=8080
EXPOSE 8080
CMD ["python", "-m", "assistant.main"]