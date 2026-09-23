"""Извлечение реквизитов правилами. LLM подключается сверху, в обработчиках"""

import re

from assistant.core.models import SlotSpec
from assistant.nlu.text import normalize, words_to_digits

_NUM = re.compile(r"\b(\d{1,4})([а-я])?\b")
_NUM_TOKEN = re.compile(r"^\d{1,4}[а-я]?$")
_QUESTION_WORDS = {"что", "какой", "какую", "зачем", "почему", "как", "откуда", "куда"}
MAX_CUE_DISTANCE = 2   # слов между подсказкой и числом: «квартира номер 42» — ок


class RuleSlotExtractor:
    """Простые правила: число для number, вариант для choice, короткая фраза для text"""

    def extract(self, spec: SlotSpec, text: str) -> str | None:
        """Реплика - прямой ответ на вопрос про этот слот"""
        normalized = normalize(text)
        if not normalized:
            return None
        match spec.kind:
            case "number":
                # берём последнее число: «сорок... то есть 42» → 42 (человек поправился)
                number_matches = list(_NUM.finditer(words_to_digits(normalized)))
                if not number_matches:
                    return None
                num_match = number_matches[-1]
                return num_match.group(1) + (num_match.group(2) or "")
            case "choice":
                for choice in spec.choices:
                    if normalize(choice) in normalized:
                        return choice
                return None
            case "name" | "address" | "text":
                if "?" in text or _QUESTION_WORDS & set(normalized.split()[:3]):
                    return None  # это вопрос, а не ответ
                if len(normalized.split()) > 12:
                    return None  # длинная речь — пусть разбирает LLM
                return text.strip().rstrip(".")
        return None

    def extract_in_context(self, spec: SlotSpec, text: str) -> str | None:
        """Число из свободной речи — только рядом с подсказкой из spec.cues

        «лифт застрял на 7 этаже» → floor=7, apartment=None: важен контекст
        При равном расстоянии предпочитаем число после подсказки («этаж 5»), не до"""
        if spec.kind != "number" or not spec.cues:
            return None
        tokens = words_to_digits(normalize(text)).split()
        cue_positions = [token_index for token_index, token in enumerate(tokens) if _is_cue(token, spec.cues)]
        number_positions = [token_index for token_index, w in enumerate(tokens) if _NUM_TOKEN.match(w)]
        if not cue_positions or not number_positions:
            return None

        def score(i: int) -> tuple[int, int]:
            return min((abs(i - cue_pos), 0 if i > cue_pos else 1) for cue_pos in cue_positions)

        best_index = min(number_positions, key=score)
        return tokens[best_index] if score(best_index)[0] <= MAX_CUE_DISTANCE else None


def _is_cue(word: str, cues: list[str]) -> bool:
    """Слово совпадает с подсказкой: точное совпадение или вхождение как префикс (cue без точки)"""
    for cue in cues:
        if cue.endswith("."):
            if word == cue[:-1]:
                return True
        elif word.startswith(cue):
            return True
    return False


def missing(required: list[str], filled: dict[str, str]) -> list[str]:
    """Возвращает слоты из required, которых ещё нет в filled"""
    return [slot_id for slot_id in required if not filled.get(slot_id)]