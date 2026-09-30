from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

VERSION = "historical-retrieval-v1"
NULLS = {"", "null", "none", "nan", "n/a"}


def value(text: str | None) -> str:
    return "" if text is None or text.strip().lower() in NULLS else text.strip()


def content_key(text: str) -> str:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def stable_rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def write_json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(data, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def tokens(text: str) -> list[str]:
    """Deterministic Chinese overlapping bigrams plus Latin words; no learned vocabulary."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\[(?:phone|business_id|license_plate|id_card|name)\]", " ", text)
    result = []
    for term in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", text):
        if "\u3400" <= term[0] <= "\u9fff":
            result.extend(term[i:i + 2] for i in range(len(term) - 1))
            if len(term) == 1:
                result.append(term)
        else:
            result.append(term)
    return result
