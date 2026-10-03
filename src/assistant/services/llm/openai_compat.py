"""Любой OpenAI-совместимый chat/completions: YandexGPT, OpenRouter, OpenAI, Ollama, vLLM"""

import httpx

from assistant.services.llm.base import LlmError


class OpenAICompatLlm:
    """HTTP-клиент к любому OpenAI-совместимому chat/completions эндпоинту"""
    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str,
            *,
            timeout: float = 8.0,
            temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Отправляет запрос к LLM и возвращает текст ответа — при любой ошибке бросает LlmError"""
        request_body: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            request_body["response_format"] = {"type": "json_object"}
        try:
            response = await self._http.post("/chat/completions", json=request_body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as http_error:
            raise LlmError(f"LLM: {http_error}") from http_error

    async def aclose(self) -> None:
        """Закрывает HTTP-клиент"""
        await self._http.aclose()