import re
from typing import Any

_NULL_MARKERS = frozenset({"null", "nan"})
NORMALIZATION_VERSION = "whitespace-v1"


def clean_text(text: Any) -> str:
    if text is None:
        return ""

    normalized = str(text).strip()
    if normalized.casefold() in _NULL_MARKERS:
        return ""

    return re.sub(r"\s+", " ", normalized).strip()


def clean_ticket(ticket):
    ticket.content = clean_text(ticket.content)
    ticket.goal = clean_text(ticket.goal)
    return ticket
