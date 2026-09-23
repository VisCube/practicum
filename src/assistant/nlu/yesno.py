"""Распознавание подтверждения или отказа"""

from assistant.nlu.text import normalize

_YES = ("да", "верно", "правильно", "все так", "подтверждаю", "ага", "угу", "точно", "конечно")
_NO = ("нет", "неверно", "неправильно", "не так", "ошибка", "не то")


def is_yes(text: str) -> bool:
    """«Да нет, неправильно» и подобные конструкции считаются отказом, не согласием"""
    normalized = normalize(text)
    if normalized.startswith(("да нет", "да не ")):   # «да нет, неправильно» — это отказ
        return False
    return any(normalized == word or normalized.startswith(word + " ") or normalized.endswith(" " + word) for word in _YES)


def is_no(text: str) -> bool:
    """Проверяет вхождение маркера отказа - в начале, конце или как отдельное слово"""
    normalized = normalize(text)
    return any(normalized == word or normalized.startswith(word + " ") or f" {word} " in f" {normalized} " for word in _NO)