"""Точка входа: uvicorn и ничего больше. Приложение собирает фабрика create_app."""
import os

import uvicorn

from assistant.core.config import Settings


def run() -> None:
    settings = Settings()
    uvicorn.run("assistant.api.app:create_app", factory=True,
                host="0.0.0.0", port=int(os.environ.get("PORT", "8000")),
                reload=settings.app_env == "dev", log_level=settings.log_level.lower())


if __name__ == "__main__":
    run()