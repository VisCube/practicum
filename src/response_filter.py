import re


STATUS_RE = re.compile(
    r"^\s*(?:"
    r"(?:we\s+)?(?:are\s+)?"
    r"(?:wait(?:ing)?\s+for|await(?:ing)?)"
    r"\s+(?:the\s+)?user(?:'s)?"
    r"(?:\s+(?:input|response))?"
    r"|"
    r"(?:ожидаю|ожидаем|ожидание|жду|ждём|ждем)"
    r"\s+(?:ответ(?:а)?|ввод(?:а)?)\s+пользователя"
    r"|ожидается\s+ответ\s+пользователя"
    r")\s*[.!?…]*\s*$",
    re.IGNORECASE,
)


def clean_response(content: str | None) -> str:
    if not content:
        return ""

    # Разбиваем ответ на непустые строки.
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    # Одна строка - всегда возвращаем без проверок
    if len(lines) == 1:
        return lines[0]

    # Берём последнюю строку
    last_line = lines[-1]

    # Проверяем, не является ли она служебной
    if (
        last_line.startswith(("(", "[")) # последняя строка в скобках
        or STATUS_RE.fullmatch(last_line) # последняя строка служебная
        or re.search(r"[a-zA-Z]", last_line) # последняя строка на английском
    ):
        return lines[-2]

    return last_line