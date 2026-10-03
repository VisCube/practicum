from __future__ import annotations

def numbers_to_digits(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        from text_to_num import alpha2digit
        return alpha2digit(text, "ru")
    except Exception:
        return text