"""Build a reproducible ticket-to-knowledge retrieval benchmark.

The source table is too large to group safely in memory.  This module uses a
temporary SQLite database to canonicalize parent tickets, expand observed
knowledge citations, and enforce exact-duplicate split isolation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TextIO


NULL_MARKERS = frozenset({"", "null", "nan"})
REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "order_id",
        "case_content",
        "case_goal",
        "area_code_city",
        "area_code_area",
        "case_accord_type_one_name",
        "case_accord_type_two_name",
        "case_accord_type_three_name",
        "call_time",
        "knowledge_quote",
        "delete_flag",
    }
)
_DATE_PREFIX = re.compile(r"^(\d{4})[-/](\d{2})[-/](\d{2})")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start: date
    end: date

    def contains(self, value: date) -> bool:
        return self.start <= value <= self.end


DEFAULT_SPLITS = (
    SplitWindow("train", date(2024, 5, 1), date(2025, 6, 30)),
    SplitWindow("dev", date(2025, 7, 1), date(2025, 7, 31)),
    SplitWindow("test", date(2025, 8, 1), date(2025, 8, 29)),
)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in NULL_MARKERS else text


def normalized_text(value: Any) -> str:
    """Normalize only the copy used for grouping; source text remains unchanged."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", safe_text(value))).strip()


def parse_source_date(value: Any) -> date | None:
    text = safe_text(value)
    match = _DATE_PREFIX.match(text)
    if match is None:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def content_goal_fingerprint(content: str, goal: str) -> str:
    payload = f"{normalized_text(content)}\0{normalized_text(goal)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _knowledge_signature(knowledge_ids: Iterable[str]) -> str:
    payload = "\0".join(sorted(set(knowledge_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_knowledge_quote(value: Any) -> list[dict[str, str]]:
    """Parse one knowledge_quote cell without treating missing citations as negatives."""

    text = safe_text(value)
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid knowledge_quote JSON: {exc.msg}") from exc
    if not isinstance(decoded, list):
        raise ValueError("knowledge_quote must be a JSON array")

    parsed: dict[str, dict[str, str]] = {}
    for position, item in enumerate(decoded):
        if not isinstance(item, Mapping):
            raise ValueError(f"knowledge_quote item {position} must be an object")
        item_type = safe_text(item.get("type"))
        value_id = safe_text(item.get("value"))
        if not item_type or not value_id:
            raise ValueError(
                f"knowledge_quote item {position} requires non-empty type and value"
            )
        knowledge_id = f"{item_type}:{value_id}"
        parsed[knowledge_id] = {
            "knowledge_id": knowledge_id,
            "title": safe_text(item.get("label")),
        }
    return [parsed[key] for key in sorted(parsed)]


def _open_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -131072;

        CREATE TABLE parents (
            group_id TEXT PRIMARY KEY,
            source_order_id TEXT NOT NULL,
            first_occurrence_id TEXT NOT NULL,
            content TEXT NOT NULL,
            content_norm TEXT NOT NULL,
            goal TEXT NOT NULL,
            goal_norm TEXT NOT NULL,
            category1 TEXT NOT NULL,
            category2 TEXT NOT NULL,
            category3 TEXT NOT NULL,
            city TEXT NOT NULL,
            district TEXT NOT NULL,
            call_date TEXT,
            content_goal_hash TEXT NOT NULL,
            knowledge_signature TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL DEFAULT 1,
            deleted INTEGER NOT NULL DEFAULT 0,
            content_conflict INTEGER NOT NULL DEFAULT 0,
            goal_conflict INTEGER NOT NULL DEFAULT 0,
            category_conflict INTEGER NOT NULL DEFAULT 0,
            location_conflict INTEGER NOT NULL DEFAULT 0,
            knowledge_conflict INTEGER NOT NULL DEFAULT 0,
            call_date_conflict INTEGER NOT NULL DEFAULT 0,
            natural_split TEXT,
            base_eligible INTEGER NOT NULL DEFAULT 0,
            cross_split_duplicate INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE edges (
            group_id TEXT NOT NULL,
            knowledge_id TEXT NOT NULL,
            PRIMARY KEY (group_id, knowledge_id)
        ) WITHOUT ROWID;

        CREATE TABLE knowledge_items (
            knowledge_id TEXT PRIMARY KEY,
            first_seen TEXT,
            last_seen TEXT
        );

        CREATE TABLE knowledge_titles (
            knowledge_id TEXT NOT NULL,
            title TEXT NOT NULL,
            observation_count INTEGER NOT NULL,
            first_seen TEXT,
            last_seen TEXT,
            PRIMARY KEY (knowledge_id, title)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _validate_header(path: Path, fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError(f"{path}: TSV has no header")
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def _json_line(output: TextIO, value: Mapping[str, Any]) -> None:
    output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _group_id(row: Mapping[str, Any]) -> tuple[str, str]:
    order_id = safe_text(row.get("order_id"))
    occurrence_id = safe_text(row.get("id"))
    if not occurrence_id:
        raise ValueError("source row has an empty id")
    if order_id:
        return f"order:{order_id}", order_id
    return f"occurrence:{occurrence_id}", ""


def _is_deleted(value: Any) -> bool:
    return safe_text(value).casefold() in {"1", "true", "yes", "y"}


def _upsert_parent(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    knowledge: list[dict[str, str]],
    call_date: date | None,
) -> None:
    group_id, order_id = _group_id(row)
    occurrence_id = safe_text(row.get("id"))
    content = safe_text(row.get("case_content"))
    goal = safe_text(row.get("case_goal"))
    content_norm = normalized_text(content)
    goal_norm = normalized_text(goal)
    category = tuple(
        safe_text(row.get(column))
        for column in (
            "case_accord_type_one_name",
            "case_accord_type_two_name",
            "case_accord_type_three_name",
        )
    )
    city = safe_text(row.get("area_code_city"))
    district = safe_text(row.get("area_code_area"))
    knowledge_ids = [item["knowledge_id"] for item in knowledge]
    knowledge_signature = _knowledge_signature(knowledge_ids)
    fingerprint = content_goal_fingerprint(content, goal)
    call_date_text = call_date.isoformat() if call_date else None

    existing = connection.execute(
        "SELECT * FROM parents WHERE group_id = ?", (group_id,)
    ).fetchone()
    if existing is None:
        connection.execute(
            """
            INSERT INTO parents (
                group_id, source_order_id, first_occurrence_id,
                content, content_norm, goal, goal_norm,
                category1, category2, category3, city, district,
                call_date, content_goal_hash, knowledge_signature, deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                order_id,
                occurrence_id,
                content,
                content_norm,
                goal,
                goal_norm,
                *category,
                city,
                district,
                call_date_text,
                fingerprint,
                knowledge_signature,
                int(_is_deleted(row.get("delete_flag"))),
            ),
        )
    else:
        known_date = existing["call_date"]
        earliest_date = min(
            value for value in (known_date, call_date_text) if value is not None
        ) if known_date is not None or call_date_text is not None else None
        connection.execute(
            """
            UPDATE parents SET
                occurrence_count = occurrence_count + 1,
                deleted = MAX(deleted, ?),
                call_date = ?,
                content_conflict = MAX(content_conflict, ?),
                goal_conflict = MAX(goal_conflict, ?),
                category_conflict = MAX(category_conflict, ?),
                location_conflict = MAX(location_conflict, ?),
                knowledge_conflict = MAX(knowledge_conflict, ?),
                call_date_conflict = MAX(call_date_conflict, ?)
            WHERE group_id = ?
            """,
            (
                int(_is_deleted(row.get("delete_flag"))),
                earliest_date,
                int(existing["content_norm"] != content_norm),
                int(existing["goal_norm"] != goal_norm),
                int(
                    (existing["category1"], existing["category2"], existing["category3"])
                    != category
                ),
                int((existing["city"], existing["district"]) != (city, district)),
                int(existing["knowledge_signature"] != knowledge_signature),
                int(
                    known_date is not None
                    and call_date_text is not None
                    and known_date != call_date_text
                ),
                group_id,
            ),
        )

    for item in knowledge:
        knowledge_id = item["knowledge_id"]
        title = item["title"]
        connection.execute(
            "INSERT OR IGNORE INTO edges (group_id, knowledge_id) VALUES (?, ?)",
            (group_id, knowledge_id),
        )
        connection.execute(
            """
            INSERT INTO knowledge_items (knowledge_id, first_seen, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT(knowledge_id) DO UPDATE SET
                first_seen = CASE
                    WHEN excluded.first_seen IS NULL THEN knowledge_items.first_seen
                    WHEN knowledge_items.first_seen IS NULL THEN excluded.first_seen
                    ELSE MIN(knowledge_items.first_seen, excluded.first_seen)
                END,
                last_seen = CASE
                    WHEN excluded.last_seen IS NULL THEN knowledge_items.last_seen
                    WHEN knowledge_items.last_seen IS NULL THEN excluded.last_seen
                    ELSE MAX(knowledge_items.last_seen, excluded.last_seen)
                END
            """,
            (knowledge_id, call_date_text, call_date_text),
        )
        if title:
            connection.execute(
                """
                INSERT INTO knowledge_titles (
                    knowledge_id, title, observation_count, first_seen, last_seen
                ) VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(knowledge_id, title) DO UPDATE SET
                    observation_count = observation_count + 1,
                    first_seen = CASE
                        WHEN excluded.first_seen IS NULL THEN knowledge_titles.first_seen
                        WHEN knowledge_titles.first_seen IS NULL THEN excluded.first_seen
                        ELSE MIN(knowledge_titles.first_seen, excluded.first_seen)
                    END,
                    last_seen = CASE
                        WHEN excluded.last_seen IS NULL THEN knowledge_titles.last_seen
                        WHEN knowledge_titles.last_seen IS NULL THEN excluded.last_seen
                        ELSE MAX(knowledge_titles.last_seen, excluded.last_seen)
                    END
                """,
                (knowledge_id, title, call_date_text, call_date_text),
            )


def _assign_splits(
    connection: sqlite3.Connection,
    splits: tuple[SplitWindow, ...],
) -> None:
    updates: list[tuple[str | None, int, str]] = []
    for row in connection.execute(
        """
        SELECT group_id, call_date, content_norm, goal_norm, deleted,
               content_conflict, goal_conflict, category_conflict,
               knowledge_conflict, call_date_conflict,
               EXISTS(SELECT 1 FROM edges WHERE edges.group_id = parents.group_id) AS has_edge
        FROM parents
        """
    ):
        parsed_date = date.fromisoformat(row["call_date"]) if row["call_date"] else None
        split_name = None
        if parsed_date is not None:
            split_name = next(
                (window.name for window in splits if window.contains(parsed_date)), None
            )
        eligible = int(
            split_name is not None
            and not row["deleted"]
            and bool(row["content_norm"] or row["goal_norm"])
            and bool(row["has_edge"])
            and not row["content_conflict"]
            and not row["goal_conflict"]
            and not row["category_conflict"]
            and not row["knowledge_conflict"]
            and not row["call_date_conflict"]
        )
        updates.append((split_name, eligible, row["group_id"]))
        if len(updates) >= 10_000:
            connection.executemany(
                "UPDATE parents SET natural_split = ?, base_eligible = ? WHERE group_id = ?",
                updates,
            )
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE parents SET natural_split = ?, base_eligible = ? WHERE group_id = ?",
            updates,
        )

    connection.executescript(
        """
        CREATE INDEX parents_fingerprint_split
            ON parents(content_goal_hash, natural_split);
        CREATE INDEX edges_knowledge_id
            ON edges(knowledge_id);
        CREATE TEMP TABLE cross_split_hashes AS
            SELECT content_goal_hash
            FROM parents
            WHERE base_eligible = 1
            GROUP BY content_goal_hash
            HAVING COUNT(DISTINCT natural_split) > 1;
        CREATE INDEX cross_split_hashes_idx
            ON cross_split_hashes(content_goal_hash);
        UPDATE parents
        SET cross_split_duplicate = 1
        WHERE base_eligible = 1
          AND content_goal_hash IN (SELECT content_goal_hash FROM cross_split_hashes);
        """
    )


def _query_record(row: sqlite3.Row, *, view: str) -> dict[str, Any]:
    if view == "content":
        text = row["content"]
    elif view == "goal":
        text = row["goal"]
    elif view == "joint":
        parts = []
        if row["content"]:
            parts.append(f"[事实] {row['content']}")
        if row["goal"]:
            parts.append(f"[诉求] {row['goal']}")
        text = "\n".join(parts)
    else:
        raise ValueError(f"unsupported query view: {view}")
    return {
        "_id": row["group_id"],
        "text": text,
        "metadata": {
            "split": row["natural_split"],
            "call_date": row["call_date"],
            "category": [
                value
                for value in (row["category1"], row["category2"], row["category3"])
                if value
            ],
            "city": row["city"],
            "district": row["district"],
            "content_goal_hash": row["content_goal_hash"],
            "occurrence_count": row["occurrence_count"],
        },
    }


def _write_outputs(
    connection: sqlite3.Connection,
    output_dir: Path,
    splits: tuple[SplitWindow, ...],
    source_path: Path,
    counters: Mapping[str, int],
) -> dict[str, Any]:
    qrels_dir = output_dir / "qrels"
    qrels_dir.mkdir(parents=True)

    with (
        (output_dir / "corpus.jsonl").open("w", encoding="utf-8") as corpus_output,
        (output_dir / "knowledge_title_conflicts.jsonl").open(
            "w", encoding="utf-8"
        ) as title_conflicts,
    ):
        for item in connection.execute(
            "SELECT * FROM knowledge_items ORDER BY knowledge_id"
        ):
            titles = connection.execute(
                """
                SELECT title, observation_count, first_seen, last_seen
                FROM knowledge_titles
                WHERE knowledge_id = ?
                ORDER BY observation_count DESC, title ASC
                """,
                (item["knowledge_id"],),
            ).fetchall()
            title = titles[0]["title"] if titles else item["knowledge_id"]
            citation_count = connection.execute(
                "SELECT COUNT(*) FROM edges WHERE knowledge_id = ?",
                (item["knowledge_id"],),
            ).fetchone()[0]
            _json_line(
                corpus_output,
                {
                    "_id": item["knowledge_id"],
                    "title": title,
                    "text": "",
                    "metadata": {
                        "source": "knowledge_quote_title",
                        "first_seen": item["first_seen"],
                        "last_seen": item["last_seen"],
                        "observed_citation_count": citation_count,
                        "title_variant_count": len(titles),
                    },
                },
            )
            if len(titles) > 1:
                _json_line(
                    title_conflicts,
                    {
                        "knowledge_id": item["knowledge_id"],
                        "selected_title": title,
                        "variants": [dict(row) for row in titles],
                    },
                )

    query_files = {
        "content": (output_dir / "queries.content.jsonl").open("w", encoding="utf-8"),
        "goal": (output_dir / "queries.goal.jsonl").open("w", encoding="utf-8"),
        "joint": (output_dir / "queries.joint.jsonl").open("w", encoding="utf-8"),
        "default": (output_dir / "queries.jsonl").open("w", encoding="utf-8"),
    }
    qrel_files = {
        window.name: (qrels_dir / f"{window.name}.tsv").open("w", encoding="utf-8", newline="")
        for window in splits
    }
    qrel_writers = {
        name: csv.writer(handle, delimiter="\t", lineterminator="\n")
        for name, handle in qrel_files.items()
    }
    for writer in qrel_writers.values():
        writer.writerow(["query-id", "corpus-id", "score"])

    try:
        eligible_rows = connection.execute(
            """
            SELECT * FROM parents
            WHERE base_eligible = 1 AND cross_split_duplicate = 0
            ORDER BY group_id
            """
        )
        for row in eligible_rows:
            for view in ("content", "goal", "joint"):
                record = _query_record(row, view=view)
                _json_line(query_files[view], record)
                if view == "joint":
                    _json_line(query_files["default"], record)
        for edge in connection.execute(
            """
            SELECT parents.natural_split, edges.group_id, edges.knowledge_id
            FROM edges
            JOIN parents USING(group_id)
            WHERE parents.base_eligible = 1
              AND parents.cross_split_duplicate = 0
            ORDER BY parents.natural_split, edges.group_id, edges.knowledge_id
            """
        ):
            qrel_writers[edge["natural_split"]].writerow(
                [edge["group_id"], edge["knowledge_id"], 1]
            )
    finally:
        for handle in query_files.values():
            handle.close()
        for handle in qrel_files.values():
            handle.close()

    with (output_dir / "parent_conflicts.jsonl").open("w", encoding="utf-8") as output:
        for row in connection.execute(
            """
            SELECT group_id, occurrence_count, content_conflict, goal_conflict,
                   category_conflict, location_conflict, knowledge_conflict,
                   call_date_conflict
            FROM parents
            WHERE content_conflict = 1 OR goal_conflict = 1 OR category_conflict = 1
               OR location_conflict = 1 OR knowledge_conflict = 1
               OR call_date_conflict = 1
            ORDER BY group_id
            """
        ):
            _json_line(output, dict(row))

    with (output_dir / "cross_split_duplicates.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for row in connection.execute(
            """
            SELECT content_goal_hash, COUNT(*) AS parent_count,
                   GROUP_CONCAT(DISTINCT natural_split) AS splits
            FROM parents
            WHERE cross_split_duplicate = 1
            GROUP BY content_goal_hash
            ORDER BY content_goal_hash
            """
        ):
            _json_line(output, dict(row))

    split_counts = {
        window.name: connection.execute(
            """
            SELECT COUNT(*) FROM parents
            WHERE base_eligible = 1 AND cross_split_duplicate = 0
              AND natural_split = ?
            """,
            (window.name,),
        ).fetchone()[0]
        for window in splits
    }
    qrel_counts = {
        window.name: connection.execute(
            """
            SELECT COUNT(*)
            FROM edges
            JOIN parents USING(group_id)
            WHERE parents.base_eligible = 1
              AND parents.cross_split_duplicate = 0
              AND parents.natural_split = ?
            """,
            (window.name,),
        ).fetchone()[0]
        for window in splits
    }
    conflict_counts = {
        column: connection.execute(
            f"SELECT COUNT(*) FROM parents WHERE {column} = 1"
        ).fetchone()[0]
        for column in (
            "content_conflict",
            "goal_conflict",
            "category_conflict",
            "location_conflict",
            "knowledge_conflict",
            "call_date_conflict",
        )
    }
    manifest = {
        "schema_version": "retrieval-dataset-v1",
        "source": {
            "path": str(source_path.resolve()),
            "size_bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        },
        "contract": {
            "task": "ticket-to-knowledge-title-retrieval",
            "query_unit": "canonical parent ticket",
            "knowledge_id": "type:value",
            "positive_label": "observed citation",
            "unobserved_label": "unknown",
            "corpus_scope": "static titles observed anywhere in the source snapshot",
            "exact_duplicate_policy": "exclude fingerprints spanning multiple splits",
        },
        "splits": [
            {"name": window.name, "start": window.start.isoformat(), "end": window.end.isoformat()}
            for window in splits
        ],
        "counts": {
            **dict(counters),
            "parent_tickets": connection.execute("SELECT COUNT(*) FROM parents").fetchone()[0],
            "knowledge_items": connection.execute(
                "SELECT COUNT(*) FROM knowledge_items"
            ).fetchone()[0],
            "observed_citation_edges": connection.execute(
                "SELECT COUNT(*) FROM edges"
            ).fetchone()[0],
            "cross_split_parent_tickets_excluded": connection.execute(
                "SELECT COUNT(*) FROM parents WHERE cross_split_duplicate = 1"
            ).fetchone()[0],
            "eligible_queries": split_counts,
            "qrels": qrel_counts,
            "parent_conflicts": conflict_counts,
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_retrieval_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    splits: tuple[SplitWindow, ...] = DEFAULT_SPLITS,
    overwrite: bool = False,
    limit: int | None = None,
    progress_every: int = 100_000,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build BEIR-compatible title retrieval data from the sanitized TSV."""

    source = Path(source_path)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"source TSV does not exist: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists: {destination}; pass overwrite=True explicitly"
        )
    resolved_destination = destination.resolve()
    protected_destinations = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        source.resolve().parent,
    }
    if resolved_destination in protected_destinations:
        raise ValueError(f"refusing unsafe output directory: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"output path exists and is not a directory: {destination}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")
    if progress_every <= 0:
        raise ValueError("progress_every must be greater than zero")
    if not splits:
        raise ValueError("at least one split window is required")
    split_names = [window.name for window in splits]
    if len(set(split_names)) != len(split_names):
        raise ValueError("split names must be unique")
    ordered_splits = sorted(splits, key=lambda item: item.start)
    for left, right in zip(ordered_splits, ordered_splits[1:]):
        if left.end >= right.start:
            raise ValueError("split windows must not overlap")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=destination.parent))
    database_directory = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    database_directory.mkdir(parents=True, exist_ok=True)
    database_handle = tempfile.NamedTemporaryFile(
        prefix="civic-rag-retrieval-",
        suffix=".sqlite3",
        dir=database_directory,
        delete=False,
    )
    database_path = Path(database_handle.name)
    database_handle.close()
    counters = {
        "source_rows_read": 0,
        "source_rows_rejected": 0,
        "source_rows_missing_call_date": 0,
    }

    connection: sqlite3.Connection | None = None
    try:
        connection = _open_sqlite(database_path)
        with (
            source.open("r", encoding="utf-8-sig", newline="") as source_file,
            (staging / "rejects.jsonl").open("w", encoding="utf-8") as rejects,
        ):
            reader = csv.DictReader(source_file, delimiter="\t", strict=True)
            _validate_header(source, reader.fieldnames)
            connection.execute("BEGIN")
            for row_number, row in enumerate(reader, start=2):
                if limit is not None and counters["source_rows_read"] >= limit:
                    break
                counters["source_rows_read"] += 1
                try:
                    if None in row:
                        raise ValueError("row has more fields than the header")
                    parsed_knowledge = parse_knowledge_quote(row.get("knowledge_quote"))
                    parsed_date = parse_source_date(row.get("call_time"))
                    if parsed_date is None:
                        counters["source_rows_missing_call_date"] += 1
                    _upsert_parent(connection, row, parsed_knowledge, parsed_date)
                except (TypeError, ValueError) as exc:
                    counters["source_rows_rejected"] += 1
                    _json_line(
                        rejects,
                        {
                            "row_number": row_number,
                            "occurrence_id": safe_text(row.get("id")),
                            "reason": str(exc),
                        },
                    )
                if counters["source_rows_read"] % progress_every == 0:
                    connection.commit()
                    connection.execute("BEGIN")
                    print(f"loaded_rows={counters['source_rows_read']}", flush=True)
            connection.commit()

        _assign_splits(connection, splits)
        connection.commit()
        manifest = _write_outputs(connection, staging, splits, source, counters)
        connection.close()
        connection = None

        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if connection is not None:
            connection.close()
        database_path.unlink(missing_ok=True)
