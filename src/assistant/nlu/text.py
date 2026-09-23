"""Нормализация текста для правил: регистр, ё→е, пунктуация, числительные словами."""

import re
from functools import lru_cache

_PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)


def normalize(text: str) -> str:
    """Приводит текст к нижнему регистру, заменяет ё→е и убирает пунктуацию"""
    text = text.lower().replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1024)
def _stem_pattern(stem: str) -> re.Pattern[str]:
    """«жилищн инспекц» → \\bжилищн\\S*\\s+инспекц ; «газ.» → \\bгаз\\b (слово целиком)"""
    pattern_parts = []
    for word in stem.split():
        is_exact = word.endswith(".")
        word = normalize(word)
        if not word:
            continue
        pattern_parts.append(re.escape(word) + (r"\b" if is_exact else r"\S*"))
    return re.compile(r"\b" + r"\s+".join(pattern_parts))


def contains_stem(text: str, stem: str) -> bool:
    """Каждое слово основы — начало слова в тексте; точка в конце слова - слово целиком

    «прокуратур»     найдёт «прокуратуру», «в прокуратуре»
    «жилищн инспекц» найдёт «жилищная инспекция», «жилищной инспекцией»
    «газ.»           найдёт «газ», но не «газон» и не «газета»"""
    return _stem_pattern(stem).search(normalize(text)) is not None


_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11,
    "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
_ORDINALS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7,
    "восьм": 8, "девят": 9, "десят": 10, "одиннадцат": 11, "двенадцат": 12,
    "тринадцат": 13, "четырнадцат": 14, "пятнадцат": 15, "шестнадцат": 16,
}
_ORD_END = re.compile(r"^(" + "|".join(_ORDINALS) + r")(ый|ой|ий|ом|ем|ая|ую|ье|ья|ого|ей)$")


def _value(word: str) -> int:
    return _HUNDREDS.get(word) or _TENS.get(word) or _UNITS[word]


def _rank(word: str) -> int:
    return 3 if word in _HUNDREDS else 2 if word in _TENS else 1 if word in _UNITS else 0


def words_to_digits(text: str) -> str:
    """«квартира сорок два» → «квартира 42», «на седьмом этаже» → «на 7 этаже», «два три» → «2 3»."""
    result: list[str] = []
    accumulator, last_rank = 0, 99

    def flush() -> None:
        nonlocal accumulator, last_rank
        if last_rank != 99:
            result.append(str(accumulator))
        accumulator, last_rank = 0, 99

    for word in text.split():
        rank = _rank(word)
        if rank and rank < last_rank:          # продолжение числа: сто → двадцать → пять
            accumulator += _value(word)
            last_rank = rank
            continue
        flush()
        if rank:                            # новое число началось
            accumulator, last_rank = _value(word), rank
        elif ordinal_match := _ORD_END.match(word):
            result.append(str(_ORDINALS[ordinal_match.group(1)]))
        else:
            result.append(word)
    flush()
    return " ".join(result)