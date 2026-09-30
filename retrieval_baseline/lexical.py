from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    VERSION,
    file_hash,
    read_jsonl,
    readonly,
    tokens,
    write_json,
    write_jsonl,
)

TOKENIZER = "nfkc-chinese-bigram-latin-v1"


def verified_manifest(dataset: Path, names: tuple[str, ...]) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("unsupported dataset version")
    for name in names:
        if file_hash(dataset / name) != manifest["artifacts"][name]:
            raise ValueError(f"dataset artifact changed: {name}")
    return manifest


def build_index(dataset: Path, output: Path) -> dict:
    manifest = verified_manifest(dataset, ("dataset.sqlite3",))
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = readonly(dataset / "dataset.sqlite3")
    target = sqlite3.connect(output / "index.sqlite3")
    try:
        target.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE VIRTUAL TABLE case_fts USING fts5(text,content='');
            CREATE VIRTUAL TABLE title_fts USING fts5(text,content='');
            CREATE VIRTUAL TABLE case_vocab USING fts5vocab(case_fts,'row');
            CREATE VIRTUAL TABLE title_vocab USING fts5vocab(title_fts,'row');
            CREATE TABLE cases(docid INTEGER PRIMARY KEY,source_id TEXT,source_row INTEGER,
                               group_id INTEGER);
            CREATE TABLE links(docid INTEGER,knowledge_id TEXT,PRIMARY KEY(docid,knowledge_id));
            CREATE TABLE titles(docid INTEGER PRIMARY KEY,knowledge_id TEXT UNIQUE);
        """)
        count = 0
        for rid, sid, row, group, content in source.execute(
            "SELECT r.rid,r.source_id,r.source_row,r.group_id,r.content FROM corpus c "
            "JOIN records r ON r.rid=c.rid ORDER BY r.rid"
        ):
            target.execute("INSERT INTO cases VALUES(?,?,?,?)", (rid, sid, row, group))
            target.execute(
                "INSERT INTO case_fts(rowid,text) VALUES(?,?)", (rid, " ".join(tokens(content)))
            )
            count += 1
            if count % 25000 == 0:
                target.commit()
                print(f"indexed_cases={count}", flush=True)
        target.executemany("INSERT INTO links VALUES(?,?)", source.execute(
            "SELECT rid,knowledge_id FROM corpus_links"
        ))
        titles = defaultdict(list)
        for key, label in source.execute("SELECT knowledge_id,label FROM catalog ORDER BY 1,2"):
            titles[key].append(label)
        for docid, (key, aliases) in enumerate(sorted(titles.items()), 1):
            target.execute("INSERT INTO titles VALUES(?,?)", (docid, key))
            target.execute(
                "INSERT INTO title_fts(rowid,text) VALUES(?,?)",
                (docid, " ".join(tokens("\n".join(aliases)))),
            )
        if not count or not titles:
            raise ValueError("historical corpus and knowledge catalog must both be nonempty")
        target.execute("INSERT INTO case_fts(case_fts) VALUES('optimize')")
        target.execute("INSERT INTO title_fts(title_fts) VALUES('optimize')")
        target.commit()
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        result = {
            "version": VERSION, "tokenizer": TOKENIZER,
            "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
            "source_input_sha256": manifest["input_sha256"],
            "index_sha256": file_hash(output / "index.sqlite3"),
            "indexed_cases": count, "indexed_knowledge_ids": len(titles),
            "seconds": time.perf_counter() - started, "sqlite_version": sqlite3.sqlite_version,
        }
        write_json(output / "manifest.json", result)
        return result
    finally:
        source.close()
        target.close()


def match_query(db: sqlite3.Connection, text: str, route: str, limit: int) -> str:
    frequencies = Counter(tokens(text))
    scored = []
    for term, frequency in frequencies.items():
        # Vocabulary comes exclusively from the historical index.
        found = db.execute(f"SELECT doc FROM {route}_vocab WHERE term=?", (term,)).fetchone()
        if found:
            score = min(frequency, 3) / math.log2(found[0] + 2)
            scored.append((-score, term))
    # Terms contain only CJK or ASCII alphanumerics; quoted OR prevents FTS syntax injection.
    return " OR ".join('"' + term + '"' for _, term in sorted(scored)[:limit])


def recommend(db: sqlite3.Connection, text: str, *, case_k: int, top_k: int, max_terms: int):
    query = match_query(db, text, "title", max_terms)
    title_ids = []
    if query:
        title_ids = [key for key, in db.execute(
            "SELECT t.knowledge_id FROM title_fts JOIN titles t ON t.docid=title_fts.rowid "
            "WHERE title_fts MATCH ? ORDER BY bm25(title_fts),t.knowledge_id LIMIT ?",
            (query, top_k),
        )]
    query = match_query(db, text, "case", max_terms)
    cases = []
    if query:
        cases = db.execute(
            "SELECT c.docid,c.source_id,c.source_row,c.group_id FROM case_fts "
            "JOIN cases c ON c.docid=case_fts.rowid WHERE case_fts MATCH ? "
            "ORDER BY bm25(case_fts),c.source_id LIMIT ?", (query, case_k),
        ).fetchall()
    case_ids, case_hits, supporting = vote_cases(db, cases, top_k=top_k)
    combined = defaultdict(float)
    for ranking in (title_ids, case_ids):
        for rank, key in enumerate(ranking, 1):
            combined[key] += 1 / (60 + rank)
    fused_ids = sorted(combined, key=lambda k: (-combined[k], k))[:top_k]
    return {
        "title_bm25": title_ids,
        "case_bm25_vote": case_ids,
        "title_case_rrf": fused_ids,
    }, case_hits, supporting


def vote_cases(db: sqlite3.Connection, cases: list[tuple], *, top_k: int):
    """Shared post-retrieval stage for lexical and dense candidates, in rank order."""
    # A correlated association group contributes at most once per knowledge ID.
    # Divide a case vote among its references, so long citation lists do not dominate.
    contributions: dict[tuple[int, str], float] = {}
    supporting = defaultdict(list)
    case_hits = []
    for rank, (docid, sid, source_row, group) in enumerate(cases, 1):
        ids = [r[0] for r in db.execute("SELECT knowledge_id FROM links WHERE docid=?", (docid,))]
        case_hits.append({"source_id": sid, "source_row": source_row, "rank": rank})
        for key in ids:
            vote = 1 / ((60 + rank) * len(ids))
            contributions[group, key] = max(vote, contributions.get((group, key), 0))
            supporting[key].append(sid)
    scores = defaultdict(float)
    for (_, key), vote in contributions.items():
        scores[key] += vote
    case_ids = sorted(scores, key=lambda k: (-scores[k], k))[:top_k]
    return case_ids, case_hits, {key: supporting[key] for key in case_ids}


def observed_metrics(
    targets: list[set[str]], predictions: list[list[str]], known: set[str],
) -> dict:
    if len(targets) != len(predictions):
        raise ValueError("predictions and targets have different lengths")
    labeled = [(gold, ranking) for gold, ranking in zip(targets, predictions, strict=True) if gold]
    denominator = len(labeled)
    result = {"labeled_queries": denominator, "unlabeled_queries": len(targets) - denominator}
    total = sum(len(g) for g, _ in labeled)
    available = sum(len(g & known) for g, _ in labeled)
    result["target_catalog_coverage"] = available / total if total else None
    result["unseen_target_count"] = total - available
    result["catalog_recall_ceiling"] = (
        sum(len(g & known) / len(g) for g, _ in labeled) / denominator if denominator else None
    )
    for k in (1, 5, 10):
        recalls, hits, conditional = [], [], []
        for gold, ranking in labeled:
            if len(ranking) != len(set(ranking)):
                raise ValueError("duplicate knowledge IDs in a ranking")
            retrieved = set(ranking[:k])
            recalls.append(len(gold & retrieved) / len(gold))
            hits.append(float(bool(gold & retrieved)))
            available_gold = gold & known
            if available_gold:
                conditional.append(len(available_gold & retrieved) / len(available_gold))
        result[f"observed_recall@{k}"] = sum(recalls) / denominator if denominator else None
        result[f"observed_hit@{k}"] = sum(hits) / denominator if denominator else None
        result[f"known_target_recall@{k}"] = (
            sum(conditional) / len(conditional) if conditional else None
        )
    result["known_target_query_count"] = sum(bool(g & known) for g, _ in labeled)
    return result


def evaluate(
    dataset: Path, index: Path, output: Path, *,
    split: str = "dev", case_k: int = 50, max_terms: int = 32,
) -> dict:
    if split not in {"dev", "test"} or case_k < 1 or max_terms < 1:
        raise ValueError("invalid split or retrieval configuration")
    verified_manifest(dataset, (f"queries.{split}.jsonl", f"qrels.{split}.jsonl"))
    index_manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    if (
        index_manifest["version"] != VERSION
        or index_manifest["tokenizer"] != TOKENIZER
        or index_manifest["dataset_manifest_sha256"] != file_hash(dataset / "manifest.json")
        or index_manifest["index_sha256"] != file_hash(index / "index.sqlite3")
    ):
        raise ValueError("index provenance or hash mismatch")
    queries = read_jsonl(dataset / f"queries.{split}.jsonl")
    qrels = read_jsonl(dataset / f"qrels.{split}.jsonl")
    if not queries or [q["source_id"] for q in queries] != [q["source_id"] for q in qrels]:
        raise ValueError("empty or misaligned queries and qrels")
    output.mkdir(parents=True, exist_ok=False)
    db = readonly(index / "index.sqlite3")
    try:
        known = {r[0] for r in db.execute("SELECT knowledge_id FROM titles")}
        results, latencies = [], []
        for number, query in enumerate(queries, 1):
            started = time.perf_counter()
            # Only case_content is given to retrieval; qrels never enter recommend().
            rankings, hits, support = recommend(
                db, query["case_content"], case_k=case_k, top_k=10, max_terms=max_terms
            )
            latencies.append(time.perf_counter() - started)
            results.append({"source_id": query["source_id"], "rankings": rankings,
                            "case_hits": hits, "supporting_cases": support})
            if number % 10 == 0:
                print(f"evaluated_queries={number}/{len(queries)}", flush=True)
        targets = [set(r["observed_knowledge_ids"]) for r in qrels]
        report = {
            "version": VERSION, "split": split, "query_count": len(queries),
            "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
            "index_manifest_sha256": file_hash(index / "manifest.json"),
            "config": {"case_k": case_k, "max_terms": max_terms, "top_k": 10, "rrf_k": 60},
            "metrics": {
                route: observed_metrics(targets, [r["rankings"][route] for r in results], known)
                for route in ("title_bm25", "case_bm25_vote", "title_case_rrf")
            },
            "latency_seconds_all_routes": {
                "mean": sum(latencies) / len(latencies),
                "p95": sorted(latencies)[math.ceil(len(latencies) * .95) - 1],
            },
            "limitations": [
                "Observed-reference metrics are not complete relevance or answer correctness.",
                "Case similarity is not evaluated by knowledge-reference overlap.",
                "Missing references are excluded from supervised metrics, not used as negatives.",
                "Unseen targets remain in overall recall denominators.",
                "CJK bigrams are a simple lexical baseline, not a tuned Chinese analyzer.",
                "Test evaluation should run only after development choices are frozen.",
            ],
        }
        write_jsonl(output / "rankings.jsonl", results)
        write_json(output / "report.json", report)
        return report
    finally:
        db.close()
