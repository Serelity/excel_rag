"""Replay case-level BM25/dense RRF with a fixed downstream candidate budget."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .common import VERSION, file_hash, read_jsonl, readonly, write_json, write_jsonl
from .compare import paired_delta
from .lexical import TOKENIZER, observed_metrics, verified_manifest, vote_cases

HYBRID_VERSION = "cached-case-rrf-v1"
ROUTES = ("case_bm25_vote", "dense_case_vote")
HYBRID_ROUTE = "hybrid_case_rrf_vote"


def _validate_hits(hits: list[dict], case_k: int) -> None:
    if not isinstance(hits, list) or len(hits) > case_k:
        raise ValueError("Invalid case candidate budget")
    seen = set()
    for rank, hit in enumerate(hits, 1):
        sid = hit.get("source_id")
        if (
            not isinstance(sid, str) or not sid or sid in seen
            or type(hit.get("rank")) is not int or hit["rank"] != rank
            or type(hit.get("source_row")) is not int or hit["source_row"] < 1
        ):
            raise ValueError("Invalid case candidate ID, source_row or sequential rank")
        seen.add(sid)


def fuse_cases(
    bm25_hits: list[dict], dense_hits: list[dict], *, case_k: int, rrf_k: int = 60,
) -> list[dict]:
    """Fuse only case ranks, never citation metadata or evaluation labels."""
    if type(case_k) is not int or case_k < 1 or rrf_k != 60:
        raise ValueError("Invalid candidate budget or RRF constant (must be 60)")
    combined: dict[str, dict] = {}
    for name, hits in zip(ROUTES, (bm25_hits, dense_hits), strict=True):
        _validate_hits(hits, case_k)
        for hit in hits:
            sid = hit["source_id"]
            row = combined.setdefault(sid, {
                "source_id": sid, "source_row": hit["source_row"],
                "rrf_score": 0.0, "route_ranks": {},
            })
            if row["source_row"] != hit["source_row"]:
                raise ValueError("Conflicting source_row across candidate routes")
            row["rrf_score"] += 1 / (rrf_k + hit["rank"])
            row["route_ranks"][name] = hit["rank"]
    ranked = sorted(combined.values(), key=lambda row: (-row["rrf_score"], row["source_id"]))
    return [dict(row, rank=rank) for rank, row in enumerate(ranked[:case_k], 1)]


def _metadata(
    source: sqlite3.Connection, index: sqlite3.Connection, sids: set[str],
) -> tuple[dict[str, tuple], dict[str, set[str]]]:
    if not sids:
        return {}, {}
    ordered = sorted(sids)
    marks = ",".join("?" for _ in ordered)
    rows = source.execute(
        "SELECT r.rid,r.source_id,r.source_row,r.group_id FROM records r "
        f"JOIN corpus c ON c.rid=r.rid WHERE r.source_id IN ({marks})", ordered,
    ).fetchall()
    metadata = {row[1]: row for row in rows}
    if set(metadata) != sids:
        raise ValueError("Candidates must belong to the frozen historical corpus")
    docids = [row[0] for row in rows]
    # Dataset source_id lookup is indexed; lexical cases lookup uses its docid primary key.
    indexed = index.execute(
        f"SELECT docid,source_id,source_row,group_id FROM cases WHERE docid IN ({marks})",
        docids,
    ).fetchall()
    if {row[1]: row for row in indexed} != metadata:
        raise ValueError("Candidate metadata differs from the historical dataset")
    references = {sid: set() for sid in sids}
    by_docid = {row[0]: row[1] for row in rows}
    links = source.execute(
        f"SELECT rid,knowledge_id FROM corpus_links WHERE rid IN ({marks})", docids,
    ).fetchall()
    indexed_links = index.execute(
        f"SELECT docid,knowledge_id FROM links WHERE docid IN ({marks})", docids,
    ).fetchall()
    if set(links) != set(indexed_links):
        raise ValueError("Candidate reference metadata differs from the historical dataset")
    for docid, key in links:
        references[by_docid[docid]].add(key)
    return metadata, references


def evaluate_hybrid(
    dataset: Path, index: Path, bm25: Path, dense: Path, output: Path, *, split: str = "dev",
) -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("Unknown evaluation split")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new output directory")
    verified_manifest(dataset, ("dataset.sqlite3", f"queries.{split}.jsonl",
                                f"qrels.{split}.jsonl"))
    paths = {
        "dataset_manifest": dataset / "manifest.json", "dataset": dataset / "dataset.sqlite3",
        "queries": dataset / f"queries.{split}.jsonl", "qrels": dataset / f"qrels.{split}.jsonl",
        "index_manifest": index / "manifest.json", "index": index / "index.sqlite3",
    }
    for name, directory in zip(ROUTES, (bm25, dense), strict=True):
        paths[f"{name}_report"] = directory / "report.json"
        paths[f"{name}_rankings"] = directory / "rankings.jsonl"
    hashes = {name: file_hash(path) for name, path in paths.items()}
    manifest = json.loads(paths["index_manifest"].read_text(encoding="utf-8"))
    if (
        manifest.get("version") != VERSION or manifest.get("tokenizer") != TOKENIZER
        or manifest.get("dataset_manifest_sha256") != hashes["dataset_manifest"]
        or manifest.get("index_sha256") != hashes["index"]
    ):
        raise ValueError("Index provenance or hash mismatch")
    reports = [json.loads(paths[f"{name}_report"].read_text(encoding="utf-8")) for name in ROUTES]
    case_k = reports[0]["config"]["case_k"]
    if type(case_k) is not int or case_k < 1:
        raise ValueError("Invalid case candidate budget")
    for name, report in zip(ROUTES, reports, strict=True):
        config = report["config"]
        index_key = ("index_manifest_sha256" if name == ROUTES[0]
                     else "metadata_index_manifest_sha256")
        if (
            report.get("version") != VERSION or report.get("split") != split
            or report.get("dataset_manifest_sha256") != hashes["dataset_manifest"]
            or report.get(index_key) != hashes["index_manifest"]
            or config.get("case_k") != case_k
            or config.get("top_k") != 10 or config.get("rrf_k") != 60
        ):
            raise ValueError("Runs must share dataset, split, metadata index and candidate budget")
        declared_hash = report.get("rankings_sha256")
        if name == ROUTES[1] and not declared_hash:
            raise ValueError("Dense report must declare rankings_sha256")
        if declared_hash is not None and declared_hash != hashes[f"{name}_rankings"]:
            raise ValueError("Rankings artifact changed")
    queries = read_jsonl(paths["queries"])
    expected = [row["source_id"] for row in queries]
    runs = [read_jsonl(paths[f"{name}_rankings"]) for name in ROUTES]
    if (
        not expected or len(expected) != len(set(expected))
        or any([row["source_id"] for row in run] != expected for run in runs)
        or any(report.get("query_count") != len(expected) for report in reports)
    ):
        raise ValueError("Runs must contain identical complete query IDs in dataset order")
    results, latencies, candidate_sets = [], [], []
    predictions = {name: [row["rankings"][name] for row in run]
                   for name, run in zip(ROUTES, runs, strict=True)}
    predictions[HYBRID_ROUTE] = []
    with closing(readonly(paths["dataset"])) as source, closing(readonly(paths["index"])) as db:
        known = {row[0] for row in source.execute("SELECT DISTINCT knowledge_id FROM catalog")}
        if known != {row[0] for row in db.execute("SELECT knowledge_id FROM titles")}:
            raise ValueError("Historical knowledge catalogs differ")
        for number, sid in enumerate(expected):
            route_hits = [run[number]["case_hits"] for run in runs]
            for hits in route_hits:
                _validate_hits(hits, case_k)
            union_sids = {hit["source_id"] for hits in route_hits for hit in hits}
            metadata, references = _metadata(source, db, union_sids)
            route_references = {}
            for name, hits in zip(ROUTES, route_hits, strict=True):
                if any(metadata[hit["source_id"]][2] != hit["source_row"] for hit in hits):
                    raise ValueError("Candidate source_row does not match historical corpus")
                ranking, _, _ = vote_cases(db, [metadata[h["source_id"]] for h in hits], top_k=10)
                if ranking != predictions[name][number]:
                    raise ValueError(f"Saved baseline vote differs from candidate replay: {name}")
                route_references[name] = set().union(*(references[h["source_id"]] for h in hits))
            started = time.perf_counter()
            fused = fuse_cases(*route_hits, case_k=case_k)
            ranking, _, support = vote_cases(
                db, [metadata[h["source_id"]] for h in fused], top_k=10,
            )
            latencies.append(time.perf_counter() - started)
            predictions[HYBRID_ROUTE].append(ranking)
            fused_references = set().union(*(references[h["source_id"]] for h in fused))
            union_references = set().union(*references.values())
            candidate_sets.append({**route_references, "route_union": union_references,
                                   HYBRID_ROUTE: fused_references})
            results.append({
                "source_id": sid, "rankings": {HYBRID_ROUTE: ranking}, "case_hits": fused,
                "supporting_cases": support, "candidate_knowledge_ids": sorted(fused_references),
                "union_candidate_count": len(union_sids),
                "route_candidate_counts": {name: len(hits) for name, hits in
                                           zip(ROUTES, route_hits, strict=True)},
            })
            if (number + 1) % 10 == 0:
                print(f"fused_queries={number + 1}/{len(expected)}", flush=True)
    # Labels enter only after every candidate list and prediction has been frozen.
    qrels = read_jsonl(paths["qrels"])
    if [row["source_id"] for row in qrels] != expected:
        raise ValueError("Misaligned queries and qrels")
    targets = [set(row["observed_knowledge_ids"]) for row in qrels]
    diagnosis, coverage = [], {name: [] for name in (*ROUTES, "route_union", HYBRID_ROUTE)}
    differences = {name: {k: [] for k in (1, 5, 10)} for name in ROUTES}
    stage_counts = {key: 0 for key in ("observed_targets", "targets_outside_catalog",
                                      "missed_in_union", "lost_in_fusion_cutoff",
                                      "lost_in_voting", "top10_target_hits")}
    for number, gold in enumerate(targets):
        if not gold:
            continue
        candidates = candidate_sets[number]
        union, fused = candidates["route_union"], candidates[HYBRID_ROUTE]
        top10 = set(predictions[HYBRID_ROUTE][number])
        row = {
            "source_id": expected[number], "observed_targets": sorted(gold),
            "targets_outside_catalog": sorted(gold - known),
            "union_candidate_target_hits": sorted(gold & union),
            "fused_candidate_target_hits": sorted(gold & fused),
            "missed_in_union": sorted((gold & known) - union),
            "lost_in_fusion_cutoff": sorted((gold & union) - fused),
            "lost_in_voting": sorted((gold & fused) - top10),
            "top10_target_hits": sorted(gold & top10), "routes": {},
        }
        for name in ROUTES:
            predicted = set(predictions[name][number])
            row["routes"][name] = {
                "candidate_target_hits": sorted(gold & candidates[name]),
                "top10_target_hits": sorted(gold & predicted),
                "missed_in_candidates": sorted((gold & known) - candidates[name]),
                "lost_in_voting": sorted((gold & candidates[name]) - predicted),
            }
            for k in (1, 5, 10):
                differences[name][k].append(
                    (len(gold & set(predictions[HYBRID_ROUTE][number][:k]))
                     - len(gold & set(predictions[name][number][:k]))) / len(gold)
                )
        for name, values in coverage.items():
            values.append(len(gold & candidates[name]) / len(gold))
        for key in stage_counts:
            stage_counts[key] += len(row[key])
        diagnosis.append(row)
    implementation = {name: file_hash(Path(__file__).with_name(name))
                      for name in ("hybrid.py", "lexical.py", "compare.py", "common.py")}
    report = {
        "version": VERSION, "hybrid_version": HYBRID_VERSION, "split": split,
        "query_count": len(expected), "dataset_manifest_sha256": hashes["dataset_manifest"],
        "metadata_index_manifest_sha256": hashes["index_manifest"],
        "config": {"input_field": "case_content", "case_k": case_k, "top_k": 10,
                   "rrf_k": 60, "route_weights": {name: 1 for name in ROUTES},
                   "input_candidate_limit_per_route": case_k,
                   "union_candidate_limit": 2 * case_k, "tie_break": "source_id"},
        "runs": {name: {"report_sha256": hashes[f"{name}_report"],
                        "rankings_sha256": hashes[f"{name}_rankings"],
                        "rankings_hash_declared": "rankings_sha256" in original}
                 for name, original in zip(ROUTES, reports, strict=True)},
        "metrics": {name: observed_metrics(targets, ranking, known)
                    for name, ranking in predictions.items()},
        "observed_reference_candidate_coverage": {
            name: sum(values) / len(values) if values else None for name, values in coverage.items()
        },
        "target_stage_counts": stage_counts,
        "hybrid_minus_baseline": {
            name: {f"recall@{k}": paired_delta(values) for k, values in diffs.items()}
            for name, diffs in differences.items()
        },
        "bootstrap": {"seed": 42, "resamples": 2000, "unit": "labeled query"},
        "cached_fusion_voting_latency_seconds": {
            "mean": sum(latencies) / len(latencies),
            "p95": sorted(latencies)[math.ceil(len(latencies) * .95) - 1],
        },
        "implementation_sha256": implementation,
        "limitations": [
            "Observed-reference recall is not independently judged case relevance "
            "or answer accuracy.",
            "Empty references are unknown; unseen targets remain in overall recall denominators.",
            "Union candidate coverage is an upper bound, not guaranteed fused Top10 recall.",
            "Hybrid uses two candidate pools (up to twice the single-route retrieval effort), "
            "but only the same final case_k cases enter shared voting.",
            "Cached latency covers fusion/voting only, not encoding, retrieval, "
            "loading or validation.",
            "Bootstrap intervals omit missing-label and sampling bias; "
            "dev comparisons are exploratory.",
            "Test should be evaluated only after development choices are frozen.",
        ],
    }
    if any(file_hash(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Input artifacts changed during hybrid replay")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "rankings.jsonl", results)
    write_jsonl(output / "diagnosis.jsonl", diagnosis)
    report["rankings_sha256"] = file_hash(output / "rankings.jsonl")
    report["diagnosis_sha256"] = file_hash(output / "diagnosis.jsonl")
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "index", "bm25", "dense", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    report = evaluate_hybrid(args.dataset, args.index, args.bm25, args.dense, args.output,
                             split=args.split)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
