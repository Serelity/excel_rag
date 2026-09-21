from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .quality import require_clean_case_content


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    case_content: str
    source_row: int


def _raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _load_tsv(path: Path) -> Iterator[SourceRecord]:
    _raise_csv_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t", strict=True)
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        duplicates = [name for name, count in Counter(reader.fieldnames).items() if count > 1]
        if duplicates:
            raise ValueError(f"TSV has duplicate columns: {duplicates}")
        missing = {"id", "case_content"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"TSV is missing required columns: {sorted(missing)}")
        try:
            for source_row, row in enumerate(reader, start=1):
                if None in row:
                    raise ValueError(
                        f"TSV record ending near physical line {reader.line_num} has extra fields"
                    )
                source_id = (row.get("id") or "").strip()
                if not source_id:
                    raise ValueError(f"TSV source row {source_row} has an empty id")
                content = row.get("case_content")
                if content is None:
                    raise ValueError(f"TSV source row {source_row} has no case_content field")
                if content.strip().casefold() in {"null", "nan"}:
                    content = ""
                require_clean_case_content(content, location=f"TSV source row {source_row}")
                yield SourceRecord(source_id, content, source_row)
        except csv.Error as exc:
            raise ValueError(f"invalid TSV near physical line {reader.line_num}: {exc}") from exc


def _load_jsonl(path: Path) -> Iterator[SourceRecord]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL value is not an object at {path}:{line_number}")
            source_id = value.get("source_id")
            content = value.get("case_content")
            source_row = value.get("source_row", line_number)
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"missing source_id at {path}:{line_number}")
            if not isinstance(content, str):
                raise ValueError(f"missing case_content at {path}:{line_number}")
            if type(source_row) is not int or source_row < 1:
                raise ValueError(f"invalid source_row at {path}:{line_number}")
            require_clean_case_content(content, location=f"{path}:{line_number}")
            yield SourceRecord(source_id.strip(), content, source_row)


def load_records(path: str | Path) -> Iterator[SourceRecord]:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.casefold() in {".jsonl", ".json"}:
        return _load_jsonl(source_path)
    return _load_tsv(source_path)
