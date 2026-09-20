#!/usr/bin/env python3
"""Build a privacy-aware, reproducible profile of the civic ticket TSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "civic-data-profile-v1"
NULL_TOKENS = {"null", "none", "nan", "n/a", "na"}
TEXT_FIELDS = {
    "case_content",
    "case_goal",
    "address_detail",
    "return_visit_reason",
    "custom_form_data_str",
}
SENSITIVE_FIELDS = {
    "id",
    "order_no",
    "order_id",
    "case_content",
    "case_goal",
    "address_detail",
    "knowledge_quote",
    "deptName",
    "return_visit_reason",
    "custom_form_data_str",
}
SAFE_TOP_VALUE_FIELDS = {
    "service_object_type",
    "case_public",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "order_type",
    "case_is_visit",
    "case_is_urgent",
    "info_protect",
    "hotspot",
    "case_labels",
    "order_source",
    "special_type",
    "case_accord_type_one_name",
    "case_accord_type_two_name",
    "case_accord_type_three_name",
    "case_accord_type_four_name",
    "case_accord_type_five_name",
    "order_status",
    "order_invalid_type",
    "delete_flag",
    "order_source_detail",
    "area_code",
    "is_accuracy",
    "belong_platform",
    "isOverTime",
    "isSignOverTime",
    "resultSatisfied",
    "visitCount",
    "visitResult",
    "firstVisitSatisfied",
    "appeal_status",
    "form_type",
}
DATETIME_FIELDS = {"call_time", "case_complete_time"}
TAXONOMY_FIELDS = tuple(
    f"case_accord_type_{name}_name" for name in ("one", "two", "three", "four", "five")
)
GEOGRAPHY_FIELDS = ("area_code_city", "area_code_area", "area_code_street")

PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
INTEGER_PATTERN = re.compile(r"[+-]?\d+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

DOMAIN_RULES: dict[str, set[str] | re.Pattern[str]] = {
    "case_public": {"是", "否"},
    "case_is_visit": {"是", "否"},
    "case_is_urgent": {"一般", "紧急", "非常紧急"},
    "info_protect": {"是", "否"},
    "hotspot": {"是", "否"},
    "delete_flag": re.compile(r"[01]"),
    "area_code": re.compile(r"\d{6}"),
    "order_status": re.compile(r"\d+"),
    "is_accuracy": re.compile(r"[01]"),
    "isOverTime": re.compile(r"[01]"),
    "isSignOverTime": re.compile(r"[01]"),
    "visitCount": re.compile(r"\d+"),
}


def semantic_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.casefold() in NULL_TOKENS:
        return None
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def short_hash(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=12).hexdigest()


def quantile_from_counts(counts: Counter[int], probability: float) -> int | None:
    total = sum(counts.values())
    if not total:
        return None
    target = max(1, math.ceil(total * probability))
    seen = 0
    for value, count in sorted(counts.items()):
        seen += count
        if seen >= target:
            return value
    return max(counts)


def safe_display_value(value: str) -> str:
    """Prevent malformed categorical values from leaking identifiers or PII."""
    if PHONE_PATTERN.search(value) or ID_CARD_PATTERN.search(value) or EMAIL_PATTERN.search(value):
        return "<REDACTED_PII_LIKE>"
    if re.search(r"\d{7,}", value):
        return "<REDACTED_LONG_NUMBER>"
    if len(value) > 120 or CONTROL_PATTERN.search(value):
        return "<REDACTED_UNSAFE_VALUE>"
    return value


def safe_counter_items(counter: Counter[str], top_k: int, key: str) -> list[dict[str, Any]]:
    displayed: Counter[str] = Counter()
    for value, count in counter.items():
        displayed[safe_display_value(value)] += count
    return [{key: value, "count": count} for value, count in displayed.most_common(top_k)]


class HyperLogLog:
    """Small deterministic approximate-distinct counter."""

    def __init__(self, precision: int = 12) -> None:
        self.precision = precision
        self.size = 1 << precision
        self.registers = bytearray(self.size)

    def add(self, value: str) -> None:
        hashed = int.from_bytes(
            hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big"
        )
        index = hashed & (self.size - 1)
        remainder = hashed >> self.precision
        bits = 64 - self.precision
        rank = bits - remainder.bit_length() + 1 if remainder else bits + 1
        if rank > self.registers[index]:
            self.registers[index] = rank

    def estimate(self) -> int:
        size = self.size
        alpha = 0.7213 / (1 + 1.079 / size)
        raw = alpha * size * size / sum(2.0 ** -register for register in self.registers)
        zeroes = self.registers.count(0)
        if raw <= 2.5 * size and zeroes:
            raw = size * math.log(size / zeroes)
        return max(0, round(raw))


@dataclass
class ColumnProfile:
    name: str
    total: int = 0
    blank: int = 0
    null_literal: int = 0
    filled: int = 0
    trimmed: int = 0
    multiline: int = 0
    control_character: int = 0
    integer_like: int = 0
    length_counts: Counter[int] = field(default_factory=Counter)
    values: Counter[str] | None = None
    approximate_distinct: HyperLogLog | None = None

    def __post_init__(self) -> None:
        if self.name in SAFE_TOP_VALUE_FIELDS:
            self.values = Counter()
        else:
            self.approximate_distinct = HyperLogLog()

    def update(self, raw: str | None) -> None:
        self.total += 1
        if raw is None or raw == "":
            self.blank += 1
            return
        stripped = raw.strip()
        if stripped != raw:
            self.trimmed += 1
        if not stripped:
            self.blank += 1
            return
        if stripped.casefold() in NULL_TOKENS:
            self.null_literal += 1
            return
        self.filled += 1
        self.length_counts[len(stripped)] += 1
        if "\n" in stripped or "\r" in stripped:
            self.multiline += 1
        if CONTROL_PATTERN.search(stripped):
            self.control_character += 1
        if INTEGER_PATTERN.fullmatch(stripped):
            self.integer_like += 1
        if self.values is not None:
            self.values[stripped] += 1
        elif self.approximate_distinct is not None:
            self.approximate_distinct.add(stripped)

    def result(self, top_k: int) -> dict[str, Any]:
        missing = self.blank + self.null_literal
        if self.values is not None:
            distinct = len(self.values)
            distinct_method = "exact"
            top_values = safe_counter_items(self.values, top_k, "value")
        else:
            distinct = self.approximate_distinct.estimate() if self.approximate_distinct else 0
            distinct_method = "hyperloglog_p12"
            top_values = None
        return {
            "name": self.name,
            "total": self.total,
            "filled": self.filled,
            "missing": missing,
            "missing_rate": missing / self.total if self.total else 0.0,
            "blank": self.blank,
            "null_literal": self.null_literal,
            "trimmed_rows": self.trimmed,
            "multiline_rows": self.multiline,
            "control_character_rows": self.control_character,
            "integer_like_rows": self.integer_like,
            "distinct": distinct,
            "distinct_method": distinct_method,
            "length": {
                "min": min(self.length_counts) if self.length_counts else None,
                "p50": quantile_from_counts(self.length_counts, 0.50),
                "p95": quantile_from_counts(self.length_counts, 0.95),
                "p99": quantile_from_counts(self.length_counts, 0.99),
                "max": max(self.length_counts) if self.length_counts else None,
            },
            "top_values": top_values,
            "values_suppressed": self.name in SENSITIVE_FIELDS,
        }


@dataclass
class DateProfile:
    filled: int = 0
    valid: int = 0
    invalid: int = 0
    minimum: datetime | None = None
    maximum: datetime | None = None
    months: Counter[str] = field(default_factory=Counter)

    def update(self, value: str | None) -> datetime | None:
        normalized = semantic_value(value)
        if normalized is None:
            return None
        self.filled += 1
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            self.invalid += 1
            return None
        self.valid += 1
        self.minimum = parsed if self.minimum is None else min(self.minimum, parsed)
        self.maximum = parsed if self.maximum is None else max(self.maximum, parsed)
        self.months[parsed.strftime("%Y-%m")] += 1
        return parsed

    def result(self) -> dict[str, Any]:
        return {
            "filled": self.filled,
            "valid": self.valid,
            "invalid": self.invalid,
            "min": self.minimum.isoformat(sep=" ") if self.minimum else None,
            "max": self.maximum.isoformat(sep=" ") if self.maximum else None,
            "monthly_counts": dict(sorted(self.months.items())),
        }


class ParentStore:
    def __init__(self, database_path: Path) -> None:
        self.connection = sqlite3.connect(database_path)
        self.connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE parents (
                order_id TEXT PRIMARY KEY,
                row_count INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                goal_hash TEXT NOT NULL,
                taxonomy_hash TEXT NOT NULL,
                geography_hash TEXT NOT NULL,
                knowledge_hash TEXT NOT NULL,
                content_conflict INTEGER NOT NULL DEFAULT 0,
                goal_conflict INTEGER NOT NULL DEFAULT 0,
                taxonomy_conflict INTEGER NOT NULL DEFAULT 0,
                geography_conflict INTEGER NOT NULL DEFAULT 0,
                knowledge_conflict INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE record_ids (
                record_id TEXT PRIMARY KEY,
                row_count INTEGER NOT NULL
            );
            """
        )
        self.missing_order_id = 0
        self.missing_record_id = 0

    def update(self, row: dict[str, str]) -> None:
        record_id = semantic_value(row.get("id"))
        if record_id is None:
            self.missing_record_id += 1
        else:
            self.connection.execute(
                """
                INSERT INTO record_ids(record_id, row_count) VALUES (?, 1)
                ON CONFLICT(record_id) DO UPDATE SET row_count = row_count + 1
                """,
                (record_id,),
            )
        order_id = semantic_value(row.get("order_id"))
        if order_id is None:
            self.missing_order_id += 1
            return
        values = (
            short_hash(semantic_value(row.get("case_content")) or "<MISSING>"),
            short_hash(semantic_value(row.get("case_goal")) or "<MISSING>"),
            short_hash(
                "\x1f".join(
                    semantic_value(row.get(name)) or "<MISSING>"
                    for name in TAXONOMY_FIELDS
                )
            ),
            short_hash(
                "\x1f".join(
                    semantic_value(row.get(name)) or "<MISSING>"
                    for name in GEOGRAPHY_FIELDS
                )
            ),
            short_hash(semantic_value(row.get("knowledge_quote")) or "<MISSING>"),
        )
        self.connection.execute(
            """
            INSERT INTO parents(
                order_id, row_count, content_hash, goal_hash, taxonomy_hash,
                geography_hash, knowledge_hash
            ) VALUES (?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                row_count = row_count + 1,
                content_conflict = content_conflict OR content_hash <> excluded.content_hash,
                goal_conflict = goal_conflict OR goal_hash <> excluded.goal_hash,
                taxonomy_conflict = taxonomy_conflict OR taxonomy_hash <> excluded.taxonomy_hash,
                geography_conflict = (
                    geography_conflict OR geography_hash <> excluded.geography_hash
                ),
                knowledge_conflict = knowledge_conflict OR knowledge_hash <> excluded.knowledge_hash
            """,
            (order_id, *values),
        )

    def result(self) -> dict[str, Any]:
        self.connection.commit()
        parent_count, row_count, duplicate_parents, rows_in_multi = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(row_count), 0),
                   COALESCE(SUM(row_count > 1), 0),
                   COALESCE(SUM(CASE WHEN row_count > 1 THEN row_count ELSE 0 END), 0)
            FROM parents
            """
        ).fetchone()
        group_sizes = {
            str(size): count
            for size, count in self.connection.execute(
                "SELECT row_count, COUNT(*) FROM parents GROUP BY row_count ORDER BY row_count"
            )
        }
        conflict_names = (
            "content_conflict",
            "goal_conflict",
            "taxonomy_conflict",
            "geography_conflict",
            "knowledge_conflict",
        )
        conflict_columns = ", ".join(
            f"COALESCE(SUM({name}), 0)" for name in conflict_names
        )
        conflict_values = self.connection.execute(
            f"SELECT {conflict_columns} FROM parents"
        ).fetchone()
        record_id_count, duplicate_record_ids, duplicate_record_rows = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(row_count > 1), 0),
                   COALESCE(SUM(CASE WHEN row_count > 1 THEN row_count ELSE 0 END), 0)
            FROM record_ids
            """
        ).fetchone()
        return {
            "parent_tickets": parent_count,
            "rows_with_parent_id": row_count,
            "missing_parent_id_rows": self.missing_order_id,
            "parents_with_multiple_rows": duplicate_parents,
            "rows_in_multirow_parents": rows_in_multi,
            "parent_row_count_histogram": group_sizes,
            "parent_conflicts": dict(zip(conflict_names, conflict_values)),
            "distinct_record_ids": record_id_count,
            "missing_record_id_rows": self.missing_record_id,
            "duplicate_record_ids": duplicate_record_ids,
            "rows_using_duplicate_record_ids": duplicate_record_rows,
        }

    def close(self) -> None:
        self.connection.close()


class DatasetProfiler:
    def __init__(self, header: list[str], top_k: int, parent_store: ParentStore) -> None:
        self.header = header
        self.top_k = top_k
        self.columns = {name: ColumnProfile(name) for name in header}
        self.parent_store = parent_store
        self.records = 0
        self.valid_width_records = 0
        self.invalid_width_records = 0
        self.widths: Counter[int] = Counter()
        self.full_row_distinct = HyperLogLog()
        self.pii: dict[str, Counter[str]] = {
            field: Counter() for field in TEXT_FIELDS if field in self.columns
        }
        self.domain_violations: dict[str, Counter[str]] = {
            field: Counter() for field in DOMAIN_RULES if field in self.columns
        }
        self.rows_with_domain_violation = 0
        self.rows_with_possible_pii = 0
        self.dates = {field: DateProfile() for field in DATETIME_FIELDS if field in self.columns}
        self.duration_seconds = Counter()
        self.duration_negative = 0
        self.text_relations = Counter()
        self.taxonomy_depth = Counter()
        self.taxonomy_gap_rows = 0
        self.taxonomy_paths = Counter()
        self.geography_depth = Counter()
        self.geography_gap_rows = 0
        self.geography_paths = Counter()
        self.knowledge = Counter()
        self.knowledge_types = Counter()
        self.knowledge_ids: set[str] = set()
        self.knowledge_items_per_row = Counter()
        self.custom_json = Counter()

    def mark_width(self, width: int) -> None:
        self.records += 1
        self.widths[width] += 1
        if width == len(self.header):
            self.valid_width_records += 1
        else:
            self.invalid_width_records += 1

    def update(self, values: list[str]) -> None:
        row = dict(zip(self.header, values))
        for name, value in row.items():
            self.columns[name].update(value)
        self.full_row_distinct.add("\x1e".join(values))
        self.parent_store.update(row)
        self._update_domains(row)
        self._update_text(row)
        self._update_dates(row)
        self._update_hierarchies(row)
        self._update_structured(row)

    def _update_domains(self, row: dict[str, str]) -> None:
        row_has_violation = False
        for name, rule in DOMAIN_RULES.items():
            if name not in row:
                continue
            value = semantic_value(row[name])
            if value is None:
                continue
            valid = value in rule if isinstance(rule, set) else rule.fullmatch(value) is not None
            if not valid:
                self.domain_violations[name][value] += 1
                row_has_violation = True
        if row_has_violation:
            self.rows_with_domain_violation += 1

    def _update_text(self, row: dict[str, str]) -> None:
        row_has_possible_pii = False
        for name, counters in self.pii.items():
            value = semantic_value(row.get(name))
            if value is None:
                continue
            if PHONE_PATTERN.search(value):
                counters["mainland_mobile"] += 1
                row_has_possible_pii = True
            if ID_CARD_PATTERN.search(value):
                counters["citizen_id"] += 1
                row_has_possible_pii = True
            if EMAIL_PATTERN.search(value):
                counters["email"] += 1
                row_has_possible_pii = True
        if row_has_possible_pii:
            self.rows_with_possible_pii += 1
        content = semantic_value(row.get("case_content"))
        goal = semantic_value(row.get("case_goal"))
        if content is None and goal is None:
            self.text_relations["both_missing"] += 1
        elif content is None:
            self.text_relations["content_missing_only"] += 1
        elif goal is None:
            self.text_relations["goal_missing_only"] += 1
        elif content == goal:
            self.text_relations["exactly_equal"] += 1
        elif goal in content:
            self.text_relations["goal_contained_in_content"] += 1
        elif content in goal:
            self.text_relations["content_contained_in_goal"] += 1
        else:
            self.text_relations["both_present_distinct"] += 1

    def _update_dates(self, row: dict[str, str]) -> None:
        parsed = {name: profile.update(row.get(name)) for name, profile in self.dates.items()}
        start = parsed.get("call_time")
        end = parsed.get("case_complete_time")
        if start is not None and end is not None:
            seconds = round((end - start).total_seconds())
            if seconds < 0:
                self.duration_negative += 1
            else:
                bucket = 0 if seconds == 0 else 2 ** int(math.log2(seconds))
                self.duration_seconds[bucket] += 1

    @staticmethod
    def _path(row: dict[str, str], fields: Iterable[str]) -> list[str | None]:
        return [semantic_value(row.get(name)) for name in fields]

    def _update_hierarchies(self, row: dict[str, str]) -> None:
        taxonomy = self._path(row, TAXONOMY_FIELDS)
        taxonomy_depth = max(
            (index + 1 for index, value in enumerate(taxonomy) if value), default=0
        )
        self.taxonomy_depth[str(taxonomy_depth)] += 1
        if any(
            value and any(previous is None for previous in taxonomy[:index])
            for index, value in enumerate(taxonomy)
        ):
            self.taxonomy_gap_rows += 1
        if taxonomy_depth:
            path = " > ".join(
                value or "<MISSING>" for value in taxonomy[:taxonomy_depth]
            )
            self.taxonomy_paths[path] += 1

        geography = self._path(row, GEOGRAPHY_FIELDS)
        geography_depth = max(
            (index + 1 for index, value in enumerate(geography) if value), default=0
        )
        self.geography_depth[str(geography_depth)] += 1
        if any(
            value and any(previous is None for previous in geography[:index])
            for index, value in enumerate(geography)
        ):
            self.geography_gap_rows += 1
        if geography_depth:
            path = " > ".join(
                value or "<MISSING>" for value in geography[:geography_depth]
            )
            self.geography_paths[path] += 1

    def _update_structured(self, row: dict[str, str]) -> None:
        knowledge_raw = semantic_value(row.get("knowledge_quote"))
        if knowledge_raw is None:
            self.knowledge["missing"] += 1
        else:
            try:
                parsed = json.loads(knowledge_raw)
            except (TypeError, ValueError):
                self.knowledge["invalid_json"] += 1
            else:
                self.knowledge["valid_json"] += 1
                if parsed is None:
                    self.knowledge["json_null"] += 1
                elif not isinstance(parsed, list):
                    self.knowledge["non_list"] += 1
                else:
                    self.knowledge["list_rows"] += 1
                    self.knowledge_items_per_row[len(parsed)] += 1
                    seen: set[str] = set()
                    for item in parsed:
                        if not isinstance(item, dict):
                            self.knowledge["malformed_items"] += 1
                            continue
                        value = semantic_value(str(item.get("value", "")))
                        item_type = semantic_value(str(item.get("type", "")))
                        label = semantic_value(str(item.get("label", "")))
                        if value is None or item_type is None:
                            self.knowledge["items_missing_identity"] += 1
                            continue
                        identity = f"{item_type}:{value}"
                        if identity in seen:
                            self.knowledge["duplicate_items_within_row"] += 1
                        seen.add(identity)
                        self.knowledge_ids.add(identity)
                        self.knowledge_types[item_type] += 1
                        self.knowledge["items"] += 1
                        if label is None:
                            self.knowledge["items_missing_label"] += 1
                    if seen:
                        self.knowledge["rows_with_items"] += 1

        custom_raw = semantic_value(row.get("custom_form_data_str"))
        if custom_raw is None:
            self.custom_json["missing"] += 1
        else:
            try:
                parsed_custom = json.loads(custom_raw)
            except (TypeError, ValueError):
                self.custom_json["invalid_json"] += 1
            else:
                self.custom_json["valid_json"] += 1
                self.custom_json[f"root_{type(parsed_custom).__name__}"] += 1

    def result(self) -> dict[str, Any]:
        parent_result = self.parent_store.result()
        column_results = [self.columns[name].result(self.top_k) for name in self.header]
        exact_distinct = {
            "id": parent_result["distinct_record_ids"],
            "order_id": parent_result["parent_tickets"],
        }
        for column in column_results:
            if column["name"] in exact_distinct:
                column["distinct"] = exact_distinct[column["name"]]
                column["distinct_method"] = "exact_sqlite"
        duration_count = sum(self.duration_seconds.values())
        return {
            "records": {
                "logical_records": self.records,
                "valid_width_records": self.valid_width_records,
                "invalid_width_records": self.invalid_width_records,
                "record_width_histogram": dict(sorted(self.widths.items())),
                "approximate_distinct_full_rows": self.full_row_distinct.estimate(),
            },
            "columns": column_results,
            "entities": parent_result,
            "domain_violations": {
                name: {
                    "count": sum(values.values()),
                    "values": safe_counter_items(values, self.top_k, "value"),
                }
                for name, values in self.domain_violations.items()
            },
            "quality_flags": {
                "rows_with_domain_violation": self.rows_with_domain_violation,
                "rows_with_possible_pii": self.rows_with_possible_pii,
            },
            "text": {
                "content_goal_relationship": dict(self.text_relations),
                "possible_pii_rows_by_field": {
                    name: dict(counters) for name, counters in self.pii.items()
                },
            },
            "dates": {
                "fields": {name: profile.result() for name, profile in self.dates.items()},
                "nonnegative_completion_duration_rows": duration_count,
                "negative_completion_duration_rows": self.duration_negative,
                "completion_duration_power_of_two_seconds_histogram": {
                    str(bucket): count for bucket, count in sorted(self.duration_seconds.items())
                },
            },
            "hierarchies": {
                "taxonomy": {
                    "depth_histogram": dict(self.taxonomy_depth),
                    "gap_rows": self.taxonomy_gap_rows,
                    "distinct_paths": len(self.taxonomy_paths),
                    "top_paths": safe_counter_items(
                        self.taxonomy_paths, self.top_k, "path"
                    ),
                },
                "geography": {
                    "depth_histogram": dict(self.geography_depth),
                    "gap_rows": self.geography_gap_rows,
                    "distinct_paths": len(self.geography_paths),
                    "top_paths": safe_counter_items(
                        self.geography_paths, self.top_k, "path"
                    ),
                },
            },
            "structured_fields": {
                "knowledge_quote": {
                    **dict(self.knowledge),
                    "distinct_knowledge_ids": len(self.knowledge_ids),
                    "knowledge_type_item_counts": {
                        safe_display_value(name): count
                        for name, count in self.knowledge_types.items()
                    },
                    "items_per_row_histogram": {
                        str(count): rows
                        for count, rows in sorted(self.knowledge_items_per_row.items())
                    },
                },
                "custom_form_data_str": dict(self.custom_json),
            },
        }


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        try:
            return next(csv.reader(source, delimiter="\t", strict=True))
        except StopIteration as exc:
            raise ValueError(f"empty TSV: {path}") from exc


def source_metadata(path: Path, *, include_hash: bool) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path) if include_hash else None,
        "sha256_skipped": not include_hash,
        "columns": len(read_header(path)),
        "header": read_header(path),
    }


def compare_sample_prefix(sample_path: Path, source_path: Path) -> dict[str, Any]:
    with (
        sample_path.open("r", encoding="utf-8-sig", newline="") as sample_file,
        source_path.open("r", encoding="utf-8-sig", newline="") as source_file,
    ):
        sample_reader = csv.reader(sample_file, delimiter="\t", strict=True)
        source_reader = csv.reader(source_file, delimiter="\t", strict=True)
        sample_header = next(sample_reader)
        source_header = next(source_reader)
        rows = 0
        mismatches = 0
        for sample_row in sample_reader:
            rows += 1
            source_row = next(source_reader, None)
            if source_row != sample_row:
                mismatches += 1
        return {
            "path": str(sample_path.resolve()),
            "size_bytes": sample_path.stat().st_size,
            "logical_records": rows,
            "header_matches": sample_header == source_header,
            "prefix_row_mismatches": mismatches,
            "is_exact_prefix": sample_header == source_header and mismatches == 0,
        }


def profile_sources(
    input_path: Path,
    raw_path: Path | None,
    *,
    top_k: int,
    database_path: Path,
    progress_every: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
    parent_store = ParentStore(database_path)
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as sanitized_file:
            sanitized_reader = csv.reader(sanitized_file, delimiter="\t", strict=True)
            sanitized_header = next(sanitized_reader)
            profiler = DatasetProfiler(sanitized_header, top_k, parent_store)
            raw_comparison: dict[str, Any] | None = None
            if raw_path is None:
                for values in sanitized_reader:
                    profiler.mark_width(len(values))
                    if len(values) == len(sanitized_header):
                        profiler.update(values)
                    if profiler.records % progress_every == 0:
                        print(f"profiled_records={profiler.records:,}", flush=True)
                physical_lines = sanitized_reader.line_num
            else:
                with raw_path.open("r", encoding="utf-8-sig", newline="") as raw_file:
                    # The historical export contains valid keys in rows with fewer
                    # trailing columns and quote sequences rejected by strict CSV.
                    # Keep the sanitized input strict; audit raw widths separately.
                    raw_reader = csv.reader(raw_file, delimiter="\t", strict=False)
                    raw_header = next(raw_reader)
                    shared = [name for name in sanitized_header if name in set(raw_header)]
                    raw_index = {name: raw_header.index(name) for name in shared}
                    sanitized_index = {name: sanitized_header.index(name) for name in shared}
                    mismatch_counts = Counter()
                    unavailable_counts = Counter()
                    key_mismatches = Counter()
                    raw_records = 0
                    paired_records = 0
                    missing_raw_records = 0
                    missing_sanitized_records = 0
                    invalid_raw_width_records = 0
                    for sanitized_values, raw_values in zip_longest(sanitized_reader, raw_reader):
                        if sanitized_values is None:
                            missing_sanitized_records += 1
                            raw_records += 1
                            continue
                        profiler.mark_width(len(sanitized_values))
                        if raw_values is None:
                            missing_raw_records += 1
                            if len(sanitized_values) == len(sanitized_header):
                                profiler.update(sanitized_values)
                            continue
                        raw_records += 1
                        paired_records += 1
                        if len(raw_values) != len(raw_header):
                            invalid_raw_width_records += 1
                        if len(sanitized_values) != len(sanitized_header):
                            continue
                        profiler.update(sanitized_values)
                        if profiler.records % progress_every == 0:
                            print(f"profiled_records={profiler.records:,}", flush=True)
                        for name in shared:
                            if raw_index[name] >= len(raw_values):
                                unavailable_counts[name] += 1
                            elif (
                                sanitized_values[sanitized_index[name]]
                                != raw_values[raw_index[name]]
                            ):
                                mismatch_counts[name] += 1
                        for name in ("id", "order_id"):
                            if name in raw_index and raw_index[name] < len(raw_values) and (
                                sanitized_values[sanitized_index[name]]
                                != raw_values[raw_index[name]]
                            ):
                                key_mismatches[name] += 1
                    physical_lines = sanitized_reader.line_num
                    raw_comparison = {
                        "raw_columns": len(raw_header),
                        "sanitized_columns": len(sanitized_header),
                        "shared_columns": len(shared),
                        "removed_columns": [
                            name for name in raw_header if name not in set(sanitized_header)
                        ],
                        "added_columns": [
                            name for name in sanitized_header if name not in set(raw_header)
                        ],
                        "raw_logical_records": raw_records,
                        "paired_records": paired_records,
                        "missing_raw_records": missing_raw_records,
                        "missing_sanitized_records": missing_sanitized_records,
                        "invalid_raw_width_records": invalid_raw_width_records,
                        "raw_physical_lines": raw_reader.line_num,
                        "raw_parser": "python_csv_tsv_permissive",
                        "positional_key_mismatches": dict(key_mismatches),
                        "shared_field_mismatch_counts": dict(sorted(mismatch_counts.items())),
                        "raw_field_unavailable_counts": dict(sorted(unavailable_counts.items())),
                        "raw_values_emitted": False,
                    }
            return profiler.result(), raw_comparison, physical_lines
    finally:
        parent_store.close()


def write_columns_csv(columns: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "total",
        "filled",
        "missing",
        "missing_rate",
        "blank",
        "null_literal",
        "distinct",
        "distinct_method",
        "length_min",
        "length_p50",
        "length_p95",
        "length_p99",
        "length_max",
        "trimmed_rows",
        "multiline_rows",
        "control_character_rows",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for column in columns:
            writer.writerow(
                {
                    **{name: column.get(name) for name in fields},
                    **{f"length_{name}": value for name, value in column["length"].items()},
                }
            )


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown(profile: dict[str, Any]) -> str:
    dataset = profile["dataset"]
    analysis = profile["analysis"]
    records = analysis["records"]
    entities = analysis["entities"]
    knowledge = analysis["structured_fields"]["knowledge_quote"]
    pii = analysis["text"]["possible_pii_rows_by_field"]
    domain_total = sum(item["count"] for item in analysis["domain_violations"].values())
    pii_total = sum(sum(types.values()) for types in pii.values())
    quality_flags = analysis.get("quality_flags", {})
    columns_by_name = {column["name"]: column for column in analysis["columns"]}
    content_column = columns_by_name.get("case_content", {})
    goal_column = columns_by_name.get("case_goal", {})
    content_counts = (
        f"{content_column.get('filled', 0):,} / {content_column.get('missing', 0):,}"
    )
    goal_counts = f"{goal_column.get('filled', 0):,} / {goal_column.get('missing', 0):,}"
    call_time = analysis["dates"]["fields"].get("call_time", {})
    call_time_missing = records["logical_records"] - call_time.get("valid", 0)
    call_time_counts = f"{call_time.get('valid', 0):,} / {call_time_missing:,}"
    taxonomy = analysis["hierarchies"]["taxonomy"]
    geography = analysis["hierarchies"]["geography"]
    column_lines = []
    for column in analysis["columns"]:
        length = column["length"]
        column_lines.append(
            f"| `{column['name']}` | {column['filled']:,} | {percent(column['missing_rate'])} | "
            f"{column['distinct']:,} ({column['distinct_method']}) | "
            f"{length['p50'] if length['p50'] is not None else '-'} / "
            f"{length['p95'] if length['p95'] is not None else '-'} / "
            f"{length['max'] if length['max'] is not None else '-'} |"
        )
    raw_comparison = profile.get("raw_to_sanitized_comparison")
    comparison_text = "未执行原始表对照。"
    if raw_comparison:
        comparison_text = (
            f"原始表 {raw_comparison['raw_columns']} 列，脱敏表 {raw_comparison['sanitized_columns']} 列，"
            f"保留 {raw_comparison['shared_columns']} 个同名字段，删除 "
            f"{len(raw_comparison['removed_columns'])} 个字段；成对记录 "
            f"{raw_comparison['paired_records']:,} 条，ID/父工单键位置不一致计数为 "
            f"{sum(raw_comparison['positional_key_mismatches'].values()):,}，原始表非标准列宽记录 "
            f"{raw_comparison['invalid_raw_width_records']:,} 条。"
        )
    conflict_total = sum(entities["parent_conflicts"].values())
    return f"""# 数据画像报告

生成时间：`{profile['generated_at']}`
画像协议：`{profile['schema_version']}`

## 1. 数据集身份

- 脱敏数据：`{dataset['sanitized']['path']}`
- 文件大小：{dataset['sanitized']['size_bytes']:,} bytes
- SHA256：`{dataset['sanitized']['sha256']}`
- 逻辑记录：{records['logical_records']:,}
- 物理行：{dataset['sanitized']['physical_lines']:,}
- 字段数：{dataset['sanitized']['columns']}
- 列宽异常记录：{records['invalid_width_records']:,}

物理行数不能作为记录数：正文中的合法换行会增加物理行，而 TSV 解析器识别的是逻辑记录。

## 2. 原始表与脱敏表关系

{comparison_text}

本报告不输出原始表字段值、工单正文、地址、ID、部门名、知识标题或 PII 命中内容。

## 3. 记录与父工单

- 唯一记录 ID：{entities['distinct_record_ids']:,}
- 重复记录 ID：{entities['duplicate_record_ids']:,}
- 父工单：{entities['parent_tickets']:,}
- 多行父工单：{entities['parents_with_multiple_rows']:,}
- 位于多行父工单中的记录：{entities['rows_in_multirow_parents']:,}
- 父工单字段冲突标记总计：{conflict_total:,}

父工单冲突详见 `profile.json` 的 `analysis.entities.parent_conflicts`。后续建模必须先按
`order_id` 聚合，不能把数据库物理行直接当成独立样本。

## 4. 文本与隐私风险

- 启发式 PII 命中记录：{quality_flags.get('rows_with_possible_pii', 0):,}（字段命中 {pii_total:,}）
- 受约束字段域异常记录：{quality_flags.get('rows_with_domain_violation', 0):,}（字段命中 {domain_total:,}）
- `case_content` 有效/缺失：{content_counts}
- `case_goal` 有效/缺失：{goal_counts}
- 正文/诉求关系：`{json.dumps(analysis['text']['content_goal_relationship'], ensure_ascii=False)}`

PII 检测是高召回正则筛查，既可能误报，也不能证明未命中的文本安全。命中记录在进入训练、
embedding 或外部服务前必须隔离复核。

## 5. 知识引用

- 有有效知识项的记录：{knowledge.get('rows_with_items', 0):,}
- 引用项总数：{knowledge.get('items', 0):,}
- 唯一知识 ID：{knowledge.get('distinct_knowledge_ids', 0):,}
- 非法 JSON：{knowledge.get('invalid_json', 0):,}
- JSON null：{knowledge.get('json_null', 0):,}
- 行内重复引用：{knowledge.get('duplicate_items_within_row', 0):,}

空引用表示“未观察到引用”，不是知识负例。后续召回评估只能把已引用知识作为 observed positive。

## 6. 时间与层级覆盖

- `call_time` 有效/缺失：{call_time_counts}
- `call_time` 范围：`{call_time.get('min')}` 至 `{call_time.get('max')}`
- 负处理时长：{analysis['dates']['negative_completion_duration_rows']:,}
- 分类路径：{taxonomy['distinct_paths']:,}，层级缺口记录：{taxonomy['gap_rows']:,}
- 地域路径：{geography['distinct_paths']:,}，层级缺口记录：{geography['gap_rows']:,}

月度量存在明显不连续区间，时间切分前必须区分真实业务波动、历史回灌和批次缺失，不能只按
随机比例划分。

## 7. 字段画像

| 字段 | 有效值 | 缺失率 | 基数 | 长度 P50 / P95 / Max |
|---|---:|---:|---:|---:|
{chr(10).join(column_lines)}

完整安全枚举、时间月分布、分类/地域路径、异常域值计数见 `profile.json`；平面字段表见
`columns.csv`。

## 8. 数据处理阶段建议

1. **冻结数据契约**：以文件 SHA256、45 列顺序和逻辑记录数作为输入门禁，拒绝静默换表。
2. **统一缺失值**：把空串及大小写不同的 `NULL/null/None/NaN/N/A` 统一为真正缺失值，保留原始缺失类型审计列。
3. **先隔离再清洗**：列宽异常、字段域异常、无效时间、负处理时长和 PII 命中进入 quarantine，不自动猜测修复。
4. **父工单聚合**：按 `order_id` 生成唯一建模单元；正文、诉求、分类、地域、知识集合冲突分别留审计标记。
5. **保留双文本视图**：`case_content` 与 `case_goal` 分开清洗，同时构造 joint 视图；不要用生成摘要覆盖原文。
6. **分类分层处理**：保留原始层级和规范化层级；层级缺口与罕见路径不直接回填。
7. **知识引用结构化**：解析为 `type:value` 稳定 ID，标签只作展示文本；空引用保持 unknown，不造负样本。
8. **时间切分防泄漏**：使用 `call_time` 做 train/dev/test，跨时间窗的相同父工单或相同文本指纹必须整体排除。
9. **处理版本化**：清洗结果写入新目录，附输入哈希、规则版本、记录计数和拒绝原因；绝不覆盖 `data/raw/`。
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/t_order_master.sanitized.v1_9.tsv"),
        help="Sanitized TSV used for all value-level profiling.",
    )
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=Path("data/raw/t_order_master.tsv"),
        help="Original TSV used only for schema/key/projection comparison.",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/raw/t_order_master_100.sanitized.v1_9.tsv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data_analysis/output"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--skip-raw-comparison", action="store_true")
    parser.add_argument("--skip-raw-hash", action="store_true")
    return parser


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    raw_path = None if args.skip_raw_comparison else args.raw_source.resolve()
    sample_path = args.sample.resolve() if args.sample else None
    for path in (input_path, raw_path, sample_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be greater than zero")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="civic-profile-") as temporary:
        analysis, comparison, physical_lines = profile_sources(
            input_path,
            raw_path,
            top_k=args.top_k,
            database_path=Path(temporary) / "entities.sqlite3",
            progress_every=args.progress_every,
        )
    sanitized_metadata = source_metadata(input_path, include_hash=True)
    sanitized_metadata["physical_lines"] = physical_lines
    profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "privacy": {
            "raw_values_emitted": False,
            "sensitive_top_values_suppressed": sorted(SENSITIVE_FIELDS),
            "pii_like_or_long_numeric_categorical_values_redacted": True,
            "pii_detection_is_heuristic": True,
        },
        "dataset": {
            "sanitized": sanitized_metadata,
            "raw_source": source_metadata(
                args.raw_source.resolve(), include_hash=not args.skip_raw_hash
            )
            if raw_path is not None
            else None,
            "sample": compare_sample_prefix(sample_path, input_path) if sample_path else None,
        },
        "raw_to_sanitized_comparison": comparison,
        "analysis": analysis,
    }
    (output / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_columns_csv(analysis["columns"], output / "columns.csv")
    (output / "profile.md").write_text(build_markdown(profile), encoding="utf-8")
    return profile


def main() -> int:
    args = build_parser().parse_args()
    profile = run_analysis(args)
    print(
        json.dumps(
            {
                "records": profile["analysis"]["records"]["logical_records"],
                "parents": profile["analysis"]["entities"]["parent_tickets"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
