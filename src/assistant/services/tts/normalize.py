"""Доменная нормализация текста перед синтезом речи (TTS).

В режиме Cloud.ru LLM возвращает текст без специальной разметки под TTS:
номера заявок («№ 5432»), даты («06.11.2026»), категории со слэшами и пометками
(«Освещение / Неисправность… (ДЛЯ КНОПКИ)»), сокращения ЖКХ («ул.», «д.», «кв.»,
«ГВС», «ХВС», «ИПУ»), телефоны, шальной Markdown и иногда латиницу.

Модели вроде Silero `v5_cis_base` не содержат в словаре цифр и латиницы —
любой неразвёрнутый токен проглатывается или роняет `apply_tts`. Этот модуль
приводит текст к чистой русской форме до передачи в `rutextnorm` / движок TTS.
"""

from __future__ import annotations

import re

# Служебные пометки из справочника категорий Cloud.ru
_SERVICE_TAG_RE = re.compile(r"\(\s*для\s+кнопки\s*\)", re.IGNORECASE)

# Markdown-артефакты в ответах LLM
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_CODE_RE = re.compile(r"`([^`]*)`")
_MD_EMPH_RE = re.compile(r"(\*\*|__|\*|_)")
_MD_HEADER_RE = re.compile(r"(?m)^\s*#{1,6}\s*")
_MD_BULLET_RE = re.compile(r"(?m)^\s*[-•]\s+")

# Даты вида 06.11.2026 (в т.ч. с предлогом родительного падежа: «до 06.11.2026»)
_DATE_GEN_PREP_RE = re.compile(
    r"\b(до|к|с|со|от|после|около|ранее|позднее)\s+"
    r"(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])\.(20\d{2})\b",
    re.IGNORECASE,
)
_DATE_NOM_RE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])\.(20\d{2})\b")

# Телефоны +7 / 8 (999) 000-00-01 → группы цифр через пробел, чтобы читались разборчиво
_PHONE_RE = re.compile(
    r"(?<!\d)(\+7|8)[\s\-()]*(\d{3})[\s\-()]*(\d{3})[\s\-]*(\d{2})[\s\-]*(\d{2})(?!\d)"
)
_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_DECIMAL_RE = re.compile(r"(?<!\d)(\d+)[,.](\d+)(?!\d|\.)")

# Сокращения, после точки которых НЕ начинается новое предложение
_NO_SPLIT_ABBR_RE = re.compile(
    r"(?:^|[\s(«\"])(?:"
    r"ул|пр-т|пр|пер|б-р|наб|ш|д|кв|под|эт|корп|стр|г|обл|р-н|руб|коп|тыс|млн|сек|мин|см|им|"
    r"т\.?\s*д|т\.?\s*п|т\.?\s*е|[А-ЯЁ]"
    r")\.$",
    re.IGNORECASE,
)

# Сокращения адресов и единиц ЖКХ
_ABBREV_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bкв\.\s*м\b", re.IGNORECASE), "квадратных метров"),
    (re.compile(r"\bул\.\s*(?=[А-ЯЁа-яё0-9])"), "улица "),
    (re.compile(r"\bпр-т\.?\s*(?=[А-ЯЁа-яё0-9])", re.IGNORECASE), "проспект "),
    (re.compile(r"\bпр\.\s*(?=[А-ЯЁа-яё])"), "проспект "),
    (re.compile(r"\bпер\.\s*(?=[А-ЯЁа-яё0-9])"), "переулок "),
    (re.compile(r"\bб-р\.?\s*(?=[А-ЯЁа-яё])", re.IGNORECASE), "бульвар "),
    (re.compile(r"\bнаб\.\s*(?=[А-ЯЁа-яё])"), "набережная "),
    (re.compile(r"\bд\.\s*(?=\d)"), "дом "),
    (re.compile(r"\bкв\.\s*(?=\d)"), "квартира "),
    (re.compile(r"\bпод\.\s*(?=\d)"), "подъезд "),
    (re.compile(r"\bэт\.\s*(?=\d)"), "этаж "),
    (re.compile(r"\bкорп\.\s*(?=\d)"), "корпус "),
    (re.compile(r"\bстр\.\s*(?=\d)"), "строение "),
    (re.compile(r"\bруб\.(?!\s*[А-ЯЁ])", re.IGNORECASE), "рублей"),
    (re.compile(r"\bкоп\.(?!\s*[А-ЯЁ])", re.IGNORECASE), "копеек"),
)

# Несклоняемые аббревиатуры ЖКХ и частые англицизмы → фонетическая запись (работает в любом падеже)
_ACRONYMS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bКР\s+на\s+СОИ\b"), "ка эр на сои"),
    (re.compile(r"\bОДПУ\b"), "о дэ пэ у"),
    (re.compile(r"\bИПУ\b"), "и пэ у"),
    (re.compile(r"\bГВС\b"), "гэ вэ эс"),
    (re.compile(r"\bХВС\b"), "хэ вэ эс"),
    (re.compile(r"\bОДН\b"), "о дэ эн"),
    (re.compile(r"\bЖКХ\b"), "жэ ка ха"),
    (re.compile(r"\bЖКУ\b"), "жэ ка у"),
    (re.compile(r"\bГЖИ\b"), "гэ жэ и"),
    (re.compile(r"\bФГИС\b"), "фэ гис"),
    (re.compile(r"\bМЧС\b"), "эм чэ эс"),
    (re.compile(r"\bФИО\b"), "фамилия имя отчество"),
    (re.compile(r"\bУК\b"), "у ка"),
    (re.compile(r"\bСМС\b", re.IGNORECASE), "эсэмэс"),
    (re.compile(r"\bSMS\b", re.IGNORECASE), "эсэмэс"),
    (re.compile(r"\be-?mail\b", re.IGNORECASE), "имейл"),
    (re.compile(r"\bwi-?fi\b", re.IGNORECASE), "вайфай"),
    (re.compile(r"\bpush\b", re.IGNORECASE), "пуш"),
    (re.compile(r"\bQR\b", re.IGNORECASE), "кью ар"),
    (re.compile(r"\bID\b", re.IGNORECASE), "айди"),
    (re.compile(r"\bOK\b", re.IGNORECASE), "окей"),
)

_LATIN_CHAR_MAP: dict[str, str] = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф",
    "g": "г", "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л",
    "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "и", "z": "з",
}
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")

_MONTHS_GEN: tuple[str, ...] = (
    "", "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)

_DAY_NOM: tuple[str, ...] = (
    "", "первое", "второе", "третье", "четвёртое", "пятое",
    "шестое", "седьмое", "восьмое", "девятое", "десятое",
    "одиннадцатое", "двенадцатое", "тринадцатое", "четырнадцатое", "пятнадцатое",
    "шестнадцатое", "семнадцатое", "восемнадцатое", "девятнадцатое", "двадцатое",
    "двадцать первое", "двадцать второе", "двадцать третье", "двадцать четвёртое",
    "двадцать пятое", "двадцать шестое", "двадцать седьмое", "двадцать восьмое",
    "двадцать девятое", "тридцатое", "тридцать первое",
)

_DAY_GEN: tuple[str, ...] = (
    "", "первого", "второго", "третьего", "четвёртого", "пятого",
    "шестого", "седьмого", "восьмого", "девятого", "десятого",
    "одиннадцатого", "двенадцатого", "тринадцатого", "четырнадцатого", "пятнадцатого",
    "шестнадцатого", "семнадцатого", "восемнадцатого", "девятнадцатого", "двадцатого",
    "двадцать первого", "двадцать второго", "двадцать третьего", "двадцать четвёртого",
    "двадцать пятого", "двадцать шестого", "двадцать седьмого", "двадцать восьмого",
    "двадцать девятого", "тридцатого", "тридцать первого",
)

_UNITS_M = (
    "", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
    "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_UNITS_F = (
    "", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
    "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_TENS = (
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
)
_HUNDREDS = (
    "", "сто", "двести", "триста", "четыреста", "пятьсот",
    "шестьсот", "семьсот", "восемьсот", "девятьсот",
)
_DIGITS_RE = re.compile(r"\d+")


def _under_thousand(n: int, *, feminine: bool = False) -> list[str]:
    parts: list[str] = []
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10])
        n %= 10
    if n > 0:
        parts.append((_UNITS_F if feminine else _UNITS_M)[n])
    return parts


def _thousand_form(n: int) -> str:
    rem100 = n % 100
    rem10 = n % 10
    if 11 <= rem100 <= 19:
        return "тысяч"
    if rem10 == 1:
        return "тысяча"
    if 2 <= rem10 <= 4:
        return "тысячи"
    return "тысяч"


def int_to_russian_words(n: int) -> str:
    """Переводит неотрицательное целое число (до миллионов) в русские слова."""
    if n == 0:
        return "ноль"
    if n < 0:
        return "минус " + int_to_russian_words(-n)
    if n >= 1_000_000:
        # Длинные числа (лицевые счета, ID) читаем по цифрам
        return " ".join(int_to_russian_words(int(ch)) for ch in str(n))

    parts: list[str] = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        parts.extend(_under_thousand(thousands, feminine=True))
        parts.append(_thousand_form(thousands))
    if rest:
        parts.extend(_under_thousand(rest, feminine=False))
    return " ".join(parts)


def _year_words(year: int) -> str:
    """«2026» → «две тысячи двадцать шестого года»."""
    rem = year % 100
    prefix = "две тысячи" if 2000 <= year < 2100 else int_to_russian_words(year - rem)
    if rem == 0:
        return f"{int_to_russian_words(year)} года"
    ord_tens = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят")
    ord_units_gen = (
        "", "первого", "второго", "третьего", "четвёртого", "пятого",
        "шестого", "седьмого", "восьмого", "девятого", "десятого",
        "одиннадцатого", "двенадцатого", "тринадцатого", "четырнадцатого", "пятнадцатого",
        "шестнадцатого", "семнадцатого", "восемнадцатого", "девятнадцатого", "двадцатого",
    )
    ord_decades_gen = ("", "десятого", "двадцатого", "тридцатого", "сорокового", "пятидесятого")
    if rem <= 20:
        tail = ord_units_gen[rem]
    elif rem % 10 == 0 and rem // 10 < len(ord_decades_gen):
        tail = ord_decades_gen[rem // 10]
    elif rem // 10 < len(ord_tens):
        tail = f"{ord_tens[rem // 10]} {ord_units_gen[rem % 10]}"
    else:
        return f"{int_to_russian_words(year)} года"
    return f"{prefix} {tail} года"


def _replace_date_gen(match: re.Match[str]) -> str:
    prep = match.group(1)
    day = int(match.group(2))
    month = int(match.group(3))
    year = int(match.group(4))
    return f"{prep} {_DAY_GEN[day]} {_MONTHS_GEN[month]} {_year_words(year)}"


def _replace_date_nom(match: re.Match[str]) -> str:
    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    return f"{_DAY_NOM[day]} {_MONTHS_GEN[month]} {_year_words(year)}"


def _replace_phone(match: re.Match[str]) -> str:
    lead = "плюс семь" if match.group(1) == "+7" else "восемь"
    groups = [int_to_russian_words(int(match.group(i))) for i in (2, 3, 4, 5)]
    # Сохраняем ведущие нули внутри двузначных/трёхзначных групп («00» → «ноль ноль»)
    rendered: list[str] = [lead]
    for idx, width in ((2, 3), (3, 3), (4, 2), (5, 2)):
        raw = match.group(idx)
        if len(raw) == width and raw.startswith("0"):
            rendered.append(" ".join(int_to_russian_words(int(ch)) for ch in raw))
        else:
            rendered.append(groups[idx - 2])
    return " ".join(rendered)


def _hour_word(h: int) -> str:
    rem100 = h % 100
    rem10 = h % 10
    if 11 <= rem100 <= 19:
        return "часов"
    if rem10 == 1:
        return "час"
    if 2 <= rem10 <= 4:
        return "часа"
    return "часов"


def _minute_word(m: int) -> str:
    rem100 = m % 100
    rem10 = m % 10
    if 11 <= rem100 <= 19:
        return "минут"
    if rem10 == 1:
        return "минута"
    if 2 <= rem10 <= 4:
        return "минуты"
    return "минут"


def _replace_time(match: re.Match[str]) -> str:
    hh = int(match.group(1))
    mm = int(match.group(2))
    h_words = int_to_russian_words(hh)
    if mm == 0:
        return f"{h_words} {_hour_word(hh)} ровно"
    # Минуты женского рода («одна минута», «две минуты»)
    m_words = (
        " ".join(_under_thousand(mm, feminine=True)) if mm > 0 else "ноль"
    )
    return f"{h_words} {_hour_word(hh)} {m_words} {_minute_word(mm)}"


def _replace_decimal(match: re.Match[str]) -> str:
    whole = int_to_russian_words(int(match.group(1)))
    frac_raw = match.group(2)
    if frac_raw.startswith("0"):
        frac = " ".join(int_to_russian_words(int(ch)) for ch in frac_raw)
    else:
        frac = int_to_russian_words(int(frac_raw))
    return f"{whole} и {frac}"


def _transliterate_latin(match: re.Match[str]) -> str:
    word = match.group(0).lower()
    return "".join(_LATIN_CHAR_MAP.get(ch, "") for ch in word)


def expand_remaining_digits(text: str) -> str:
    """Страховочная замена оставшихся цифр словами (когда rutextnorm недоступен или пропустил)."""
    def _repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        if len(raw) > 1 and raw.startswith("0"):
            return " ".join(int_to_russian_words(int(ch)) for ch in raw)
        return int_to_russian_words(int(raw))

    # «2-м», «4-й» — убираем буквенное наращение, чтобы не читалось «четыре м»
    cleaned = re.sub(r"\b(\d+)-[а-яё]{1,3}\b", r"\1", text, flags=re.IGNORECASE)
    return _DIGITS_RE.sub(_repl, cleaned)


def clean_for_tts(text: str, *, expand_digits: bool = False) -> str:
    """Очищает ответ LLM/шаблона от Markdown, слэшей, сокращений, дат и латиницы."""
    out = text.strip()
    if not out or out.lstrip().lower().startswith("<speak"):
        return out

    # 1. Markdown и технические теги категорий
    out = _SERVICE_TAG_RE.sub("", out)
    out = _MD_LINK_RE.sub(r"\1", out)
    out = _MD_CODE_RE.sub(r"\1", out)
    out = _MD_HEADER_RE.sub("", out)
    out = _MD_BULLET_RE.sub("", out)
    out = _MD_EMPH_RE.sub("", out)

    # 2. Символы и разделители категорий
    # («Лифты / Неисправность лифта» → «Лифты, Неисправность лифта»)
    out = re.sub(r"№\s*", "номер ", out)
    out = re.sub(r"(?<=\d)\s*%", " процентов", out)
    out = re.sub(r"\s*/\s*", ", ", out)

    # 3. Даты ДД.ММ.ГГГГ, время ЧЧ:ММ, дробные числа и телефоны
    out = _DATE_GEN_PREP_RE.sub(_replace_date_gen, out)
    out = _DATE_NOM_RE.sub(_replace_date_nom, out)
    out = _PHONE_RE.sub(_replace_phone, out)
    out = _TIME_RE.sub(_replace_time, out)
    out = _DECIMAL_RE.sub(_replace_decimal, out)

    # 4. Адресные сокращения и аббревиатуры ЖКХ
    for pattern, replacement in _ABBREV_RULES:
        out = pattern.sub(replacement, out)
    for pattern, replacement in _ACRONYMS:
        out = pattern.sub(replacement, out)

    # 5. Случайная латиница (Silero ru не читает латинские символы)
    out = _LATIN_WORD_RE.sub(_transliterate_latin, out)

    if expand_digits:
        out = expand_remaining_digits(out)

    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r",\s*,+", ",", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def split_tts_sentences(text: str, *, max_chars: int = 280) -> list[str]:
    """Режет длинную реплику на предложения ≤ max_chars для кэширования и безопасного apply_tts."""
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.lstrip().lower().startswith("<speak"):
        return [stripped]

    # Не рвём предложение после сокращений вида «ул. Мира», «д. 19», «кв. 42», «И. И.»
    raw_parts: list[str] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?…])\s+", stripped):
        prefix = stripped[start : match.start()].strip()
        if not prefix:
            start = match.end()
            continue
        if prefix.endswith(".") and _NO_SPLIT_ABBR_RE.search(prefix):
            continue
        raw_parts.append(prefix)
        start = match.end()
    tail = stripped[start:].strip()
    if tail:
        raw_parts.append(tail)
    chunks: list[str] = []
    for part in raw_parts:
        if len(part) <= max_chars:
            chunks.append(part)
            continue
        # Длинное предложение делим по запятым/тире, затем по словам
        clauses = [c.strip() for c in re.split(r"(?<=[,;—–])\s+", part) if c.strip()]
        buf = ""
        for clause in clauses:
            candidate = f"{buf} {clause}".strip() if buf else clause
            if len(candidate) <= max_chars:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            if len(clause) <= max_chars:
                buf = clause
            else:
                for word in clause.split():
                    cand_w = f"{buf} {word}".strip() if buf else word
                    if len(cand_w) <= max_chars:
                        buf = cand_w
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = word
        if buf:
            chunks.append(buf)
    return chunks or [stripped]
