import csv
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from schemas.ticket import Ticket

_COLUMN_MAP = {
    "ticket_id": "id",
    "content": "case_content",
    "goal": "case_goal",
    "category1": "case_accord_type_one_name",
    "category2": "case_accord_type_two_name",
    "category3": "case_accord_type_three_name",
    "city": "area_code_city",
    "district": "area_code_area",
    "create_time": "call_time",
}
_NULL_MARKERS = frozenset({"null", "nan"})


def safe_str(value: Any) -> str:
    """Normalize nullable TSV values without depending on pandas."""
    if value is None:
        return ""

    text = str(value).strip()
    if text.casefold() in _NULL_MARKERS:
        return ""
    return text


def _validate_header(path: Path, fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError(f"{path}: TSV is empty or has no header row")

    duplicate_columns = sorted(name for name, count in Counter(fieldnames).items() if count > 1)
    if duplicate_columns:
        columns = ", ".join(repr(name) for name in duplicate_columns)
        raise ValueError(f"{path}: TSV has duplicate column(s): {columns}")

    missing_columns = sorted(set(_COLUMN_MAP.values()) - set(fieldnames))
    if missing_columns:
        columns = ", ".join(repr(name) for name in missing_columns)
        raise ValueError(f"{path}: TSV is missing required column(s): {columns}")


def load_tickets(path: str | Path) -> Iterator[Ticket]:
    """Yield tickets from a TSV file without loading the full file into memory."""
    source_path = Path(path)

    with source_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t", strict=True)
        _validate_header(source_path, reader.fieldnames)

        try:
            for row in reader:
                if None in row:
                    raise ValueError(
                        f"{source_path}: TSV record ending at physical line "
                        f"{reader.line_num} has more fields than the header"
                    )

                short_rows = [column for column in _COLUMN_MAP.values() if row.get(column) is None]
                if short_rows:
                    columns = ", ".join(repr(name) for name in short_rows)
                    raise ValueError(
                        f"{source_path}: TSV record ending at physical line "
                        f"{reader.line_num} is missing field(s): {columns}"
                    )

                values = {field: safe_str(row[column]) for field, column in _COLUMN_MAP.items()}
                if not values["ticket_id"]:
                    raise ValueError(
                        f"{source_path}: TSV record ending at physical line "
                        f"{reader.line_num} has an empty required 'id'"
                    )

                yield Ticket(**values)
        except csv.Error as exc:
            raise ValueError(
                f"{source_path}: invalid TSV near physical line {reader.line_num}: {exc}"
            ) from exc
