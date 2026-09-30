from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from .common import VERSION, content_key, file_hash, stable_rank, value, write_json, write_jsonl


class Groups:
    """Union-find closes transitive links across order_id AND normalized exact text."""

    def __init__(self):
        self.parent: list[int] = []
        self.size: list[int] = []
        self.periods: list[int] = []
        self.orders: dict[str, int] = {}
        self.texts: dict[str, int] = {}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def add(self, order: str, text_hash: str, period: int) -> int:
        i = len(self.parent)
        self.parent.append(i)
        self.size.append(1)
        self.periods.append(period)
        for table, key in ((self.orders, order), (self.texts, text_hash)):
            if not key:
                continue
            if key in table:
                a, b = self.find(i), self.find(table[key])
                if a != b:
                    if self.size[a] < self.size[b]:
                        a, b = b, a
                    self.parent[b] = a
                    self.size[a] += self.size[b]
                    self.periods[a] |= self.periods[b]
            else:
                table[key] = i
        return i


def references(raw: str) -> tuple[list[tuple[str, str]], str | None]:
    if not value(raw):
        return [], None
    try:
        items = json.loads(raw)
    except (TypeError, ValueError):
        return [], "invalid_reference_json"
    if not isinstance(items, list):
        return [], "invalid_reference_list"
    output = set()
    for item in items:
        if not isinstance(item, dict):
            return [], "invalid_reference_item"
        kind, key, label = item.get("type"), item.get("value"), item.get("label")
        if (
            type(kind) not in (str, int)
            or type(key) not in (str, int)
            or not isinstance(label, str)
            or not value(str(kind))
            or not value(str(key))
            or not value(label)
        ):
            return [], "invalid_reference_item"
        output.add((f"{str(kind).strip()}:{str(key).strip()}", label.strip()))
    return sorted(output), None


def period_for(text: str, dev_start: str, test_start: str) -> int:
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 8  # Missing time connects groups but cannot enter a chronological split.
    normalized = parsed.strftime("%Y-%m-%d")
    return 1 if normalized < dev_start else 2 if normalized < test_start else 4


def build_dataset(
    source: Path,
    output: Path,
    *,
    dev_start: str = "2026-01-01",
    test_start: str = "2026-02-01",
    queries_per_split: int = 200,
    seed: int = 42,
) -> dict:
    for date in (dev_start, test_start):
        if datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d") != date:
            raise ValueError("dates must use YYYY-MM-DD")
    if dev_start >= test_start or queries_per_split < 1:
        raise ValueError("require dev_start < test_start and positive queries_per_split")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=False)
    before = source.stat()
    input_hash = file_hash(source)
    counts = Counter()
    groups = Groups()
    db = sqlite3.connect(output / "dataset.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE records(
            rid INTEGER PRIMARY KEY, source_id TEXT UNIQUE NOT NULL, source_row INTEGER,
            order_id TEXT, text_hash TEXT, content TEXT, call_time TEXT, category TEXT,
            period INTEGER, group_id INTEGER, split TEXT);
        CREATE TABLE citations(rid INTEGER, knowledge_id TEXT, label TEXT,
            PRIMARY KEY(rid, knowledge_id, label));
    """)
    seen_ids = set()
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    try:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            required = {
                "id", "order_id", "case_content", "call_time", "knowledge_quote",
                "delete_flag", "order_invalid_type", "case_accord_type_one_name",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"required input fields: {sorted(required)}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("duplicate TSV header fields")
            for source_row, record in enumerate(reader, 1):
                counts["source_records"] += 1
                if None in record or any(v is None for v in record.values()):
                    raise ValueError(f"invalid TSV row width at logical record {source_row}")
                sid = value(record["id"])
                if not sid or sid in seen_ids:
                    raise ValueError(f"missing or duplicate ID at logical record {source_row}")
                seen_ids.add(sid)
                text = record["case_content"]
                reason = None
                if value(record["delete_flag"]) != "0":
                    reason = "deleted_or_unknown_delete_flag"
                elif value(record["order_invalid_type"]):
                    reason = "marked_invalid"
                elif not value(text):
                    reason = "empty_content"
                elif "\t" in text:
                    reason = "embedded_tabular_data"
                if reason:
                    counts["excluded_" + reason] += 1
                    continue
                refs, ref_error = references(record["knowledge_quote"])
                if ref_error:
                    # Keep usable text, but never turn damaged labels into negative targets.
                    counts[ref_error] += 1
                order = value(record["order_id"])
                time = value(record["call_time"])
                text_hash = content_key(text)
                period = period_for(time, dev_start, test_start)
                rid = groups.add(order, text_hash, period)
                db.execute(
                    "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,NULL,NULL)",
                    (rid, sid, source_row, order, text_hash, text, time,
                     value(record["case_accord_type_one_name"]), period),
                )
                db.executemany("INSERT INTO citations VALUES(?,?,?)",
                               ((rid, key, label) for key, label in refs))
                counts["accepted_records"] += 1
                if counts["accepted_records"] % 50000 == 0:
                    db.commit()
                    print(f"accepted_records={counts['accepted_records']}", flush=True)
        after = source.stat()
        if any(getattr(before, k) != getattr(after, k)
               for k in ("st_size", "st_mtime_ns", "st_ino")):
            raise RuntimeError("source changed during dataset construction")
        db.commit()
        splits = {1: "corpus", 2: "dev", 4: "test", 8: "missing_time"}
        group_counts = Counter()

        def assignments():
            for rid in range(len(groups.parent)):
                root = groups.find(rid)
                split = splits.get(groups.periods[root], "cross_period")
                counts["records_" + split] += 1
                if rid == root:
                    group_counts[split] += 1
                yield root, split, rid

        db.executemany("UPDATE records SET group_id=?,split=? WHERE rid=?", assignments())
        db.executescript("""
            CREATE INDEX record_split ON records(split);
            CREATE INDEX record_hash ON records(text_hash,split);
            CREATE INDEX record_group ON records(group_id);
            CREATE TABLE corpus AS
              SELECT text_hash,MIN(rid) AS rid FROM records WHERE split='corpus'
              GROUP BY text_hash;
            CREATE UNIQUE INDEX corpus_rid ON corpus(rid);
            CREATE TABLE corpus_links AS
              SELECT DISTINCT c.rid,q.knowledge_id FROM corpus c
              JOIN records r ON r.text_hash=c.text_hash AND r.split='corpus'
              JOIN citations q ON q.rid=r.rid;
            CREATE INDEX corpus_link_rid ON corpus_links(rid);
            CREATE TABLE catalog AS
              SELECT q.knowledge_id,q.label,COUNT(DISTINCT r.group_id) AS supporting_groups
              FROM citations q JOIN records r ON r.rid=q.rid
              WHERE r.split='corpus' GROUP BY q.knowledge_id,q.label;
        """)
        known = {r[0] for r in db.execute("SELECT DISTINCT knowledge_id FROM catalog")}
        counts["corpus_unique_texts"] = db.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
        counts["catalog_ids"] = len(known)
        counts["catalog_title_variants"] = db.execute("SELECT COUNT(*) FROM catalog").fetchone()[0]
        catalog_rows = [
            {"knowledge_id": key, "title": label, "supporting_groups": support}
            for key, label, support in db.execute(
                "SELECT * FROM catalog ORDER BY knowledge_id,supporting_groups DESC,label"
            )
        ]
        write_jsonl(output / "catalog.jsonl", catalog_rows)
        samples = {}
        for split in ("dev", "test"):
            # One deterministic representative per association/text component,
            # chosen WITHOUT requiring a label or a known target.
            representatives = db.execute(
                "SELECT r.rid,r.source_id FROM records r JOIN "
                "(SELECT MIN(rid) AS rid FROM records WHERE split=? GROUP BY group_id) s "
                "ON r.rid=s.rid", (split,)
            )
            pool = [(stable_rank(seed, sid), rid) for rid, sid in representatives]
            selected = sorted(pool)[:queries_per_split]
            queries, qrels = [], []
            stats = Counter(candidate_groups=len(pool), selected_queries=len(selected))
            for _, rid in selected:
                sid, source_row, text, time, category, group_id = db.execute(
                    "SELECT source_id,source_row,content,call_time,category,group_id "
                    "FROM records WHERE rid=?", (rid,)
                ).fetchone()
                ids = sorted({r[0] for r in db.execute(
                    "SELECT knowledge_id FROM citations WHERE rid=?", (rid,)
                )})
                queries.append({
                    "source_id": sid, "source_row": source_row, "case_content": text,
                    "call_time": time, "category1": category, "group_id": group_id,
                })
                qrels.append({"source_id": sid, "observed_knowledge_ids": ids})
                stats["labeled_queries" if ids else "unlabeled_queries"] += 1
                stats["observed_targets"] += len(ids)
                stats["targets_in_catalog"] += len(set(ids) & known)
                stats["unseen_targets"] += len(set(ids) - known)
            write_jsonl(output / f"queries.{split}.jsonl", queries)
            write_jsonl(output / f"qrels.{split}.jsonl", qrels)
            samples[split] = dict(stats)
        # Explicit, executable guarantees instead of assumptions about deduplication.
        for field in ("order_id", "text_hash", "group_id"):
            overlap = db.execute(
                f"SELECT COUNT(*) FROM (SELECT {field} FROM records "
                f"WHERE split IN ('corpus','dev','test') AND {field} != '' "
                f"GROUP BY {field} HAVING COUNT(DISTINCT split)>1)"
            ).fetchone()[0]
            if overlap:
                raise RuntimeError(f"split leakage detected for {field}")
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        manifest = {
            "version": VERSION, "input_sha256": input_hash, "input_bytes": before.st_size,
            "config": {"dev_start": dev_start, "test_start": test_start,
                       "queries_per_split": queries_per_split, "seed": seed},
            "counts": dict(counts), "association_groups": dict(group_counts), "samples": samples,
            "split_audit": {
                "order_id_overlap": 0, "normalized_text_overlap": 0, "group_overlap": 0,
            },
            "limitations": [
                "Export-time text and citations are not historical input snapshots.",
                "Normalized exact duplicates are isolated; paraphrased near-duplicates are not.",
                "Only corpus-period citation titles construct the observed catalog.",
                "Historical references are incomplete observed positives; empty means unknown.",
                "Representative-per-group sampling is not traffic-weighted.",
                "Records marked invalid/deleted are excluded using export-time flags.",
            ],
            "artifacts": {
                name: file_hash(output / name)
                for name in ("dataset.sqlite3", "catalog.jsonl", "queries.dev.jsonl",
                             "qrels.dev.jsonl", "queries.test.jsonl", "qrels.test.jsonl")
            },
        }
        write_json(output / "manifest.json", manifest)
        return manifest
    finally:
        db.close()
