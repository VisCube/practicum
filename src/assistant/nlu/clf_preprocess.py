"""Лемматизация перед обращением к модели-классификатору"""

from __future__ import annotations

import re

import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
_CLEAN = re.compile(r"[^а-яёa-z0-9\s]", re.IGNORECASE)


def lemmatize_for_clf(text: str) -> str:
    text = text.lower()
    text = _CLEAN.sub(" ", text)
    words = text.split()
    return " ".join(_morph.parse(w)[0].normal_form for w in words)