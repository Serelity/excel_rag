"""Paired BM25/dense comparison on identical queries and observed reference labels."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import closing
from pathlib import Path

from .common import VERSION, file_hash, read_jsonl, readonly, write_json, write_jsonl
from .lexical import observed_metrics, verified_manifest


def paired_delta(values: list[float], *, seed: int = 42, resamples: int = 2000) -> dict:
    if not values:
        return {"delta": None, "ci95": None, "wins": 0, "losses": 0, "ties": 0}
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples)
    )
    return {
        "delta": sum(values) / len(values),
        "ci95": [bootstrap[int(.025 * (resamples - 1))],
                 bootstrap[int(.975 * (resamples - 1))]],
        "wins": sum(v > 0 for v in values), "losses": sum(v < 0 for v in values),
        "ties": sum(v == 0 for v in values),
    }


def compare_runs(dataset: Path, bm25: Path, dense: Path, output: Path) -> dict:
    reports = [json.loads((d / "report.json").read_text(encoding="utf-8")) for d in (bm25, dense)]
    split = reports[0]["split"]
    if split not in {"dev", "test"}:
        raise ValueError("Unknown split")
    if reports[1].get("metadata_index_manifest_sha256") != reports[0]["index_manifest_sha256"]:
        raise ValueError("Both runs must use the same historical citation metadata index")
    verified_manifest(
        dataset, ("dataset.sqlite3", f"queries.{split}.jsonl", f"qrels.{split}.jsonl"),
    )
    for report, directory in zip(reports, (bm25, dense), strict=True):
        if (
            report["version"] != VERSION or report["split"] != split
            or report["dataset_manifest_sha256"] != file_hash(dataset / "manifest.json")
            or report["config"]["case_k"] != reports[0]["config"]["case_k"]
            or report["config"]["top_k"] != 10 or report["config"]["rrf_k"] != 60
        ):
            raise ValueError("Runs must share dataset, split, candidate budget and voting rules")
        if report.get("rankings_sha256") and (
            report["rankings_sha256"] != file_hash(directory / "rankings.jsonl")
        ):
            raise ValueError("Rankings artifact changed")
    runs = [read_jsonl(d / "rankings.jsonl") for d in (bm25, dense)]
    qrels = read_jsonl(dataset / f"qrels.{split}.jsonl")
    queries = read_jsonl(dataset / f"queries.{split}.jsonl")
    expected = [q["source_id"] for q in qrels]
    if not expected or [q["source_id"] for q in queries] != expected or any(
        [r["source_id"] for r in run] != expected for run in runs
    ):
        raise ValueError("Runs must contain identical complete query IDs in dataset order")
    names = ("case_bm25_vote", "dense_case_vote")
    predictions = [[r["rankings"][name] for r in run]
                   for run, name in zip(runs, names, strict=True)]
    targets = [set(q["observed_knowledge_ids"]) for q in qrels]
    differences = {k: [] for k in (1, 5, 10)}
    diagnosis, coverage = [], {name: [] for name in names}
    with closing(readonly(dataset / "dataset.sqlite3")) as db:
        known = {r[0] for r in db.execute("SELECT DISTINCT knowledge_id FROM catalog")}
        for i, gold in enumerate(targets):
            if not gold:
                continue
            row = {"source_id": expected[i], "observed_targets": sorted(gold),
                   "targets_outside_catalog": sorted(gold - known), "routes": {}}
            for run, name, prediction in zip(runs, names, predictions, strict=True):
                hits = run[i]["case_hits"]
                sids = [h["source_id"] for h in hits]
                if len(sids) != len(set(sids)) or len(sids) > reports[0]["config"]["case_k"]:
                    raise ValueError("Invalid case candidate list")
                marks = ",".join("?" for _ in sids)
                rows = db.execute(
                    "SELECT r.source_id,l.knowledge_id FROM records r "
                    "JOIN corpus c ON c.rid=r.rid LEFT JOIN corpus_links l ON l.rid=r.rid "
                    f"WHERE r.source_id IN ({marks})", sids,
                ).fetchall() if sids else []
                if {r[0] for r in rows} != set(sids):
                    raise ValueError("Candidates must belong to the frozen historical corpus")
                candidate_ids = {r[1] for r in rows if r[1] is not None}
                coverage[name].append(len(gold & candidate_ids) / len(gold))
                row["routes"][name] = {
                    "candidate_target_hits": sorted(gold & candidate_ids),
                    "top10_target_hits": sorted(gold & set(prediction[i][:10])),
                    "missed_in_candidates": sorted((gold & known) - candidate_ids),
                    "lost_in_voting": sorted((gold & candidate_ids) - set(prediction[i][:10])),
                }
            diagnosis.append(row)
            for k in differences:
                differences[k].append(
                    (len(gold & set(predictions[1][i][:k]))
                     - len(gold & set(predictions[0][i][:k]))) / len(gold)
                )
    result = {
        "version": "paired-retrieval-comparison-v1", "split": split,
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "runs": {name: {"report_sha256": file_hash(d / "report.json"),
                        "rankings_sha256": file_hash(d / "rankings.jsonl")}
                 for name, d in zip(names, (bm25, dense), strict=True)},
        "query_count": len(expected), "labeled_queries": sum(bool(t) for t in targets),
        "metrics": {name: observed_metrics(targets, prediction, known)
                    for name, prediction in zip(names, predictions, strict=True)},
        "observed_reference_candidate_coverage": {
            name: sum(values) / len(values) if values else None for name, values in coverage.items()
        },
        "dense_minus_bm25": {f"recall@{k}": paired_delta(v) for k, v in differences.items()},
        "bootstrap": {"seed": 42, "resamples": 2000, "unit": "labeled query"},
        "limitations": [
            "Reference-candidate coverage is NOT independently judged case relevance.",
            "Intervals concern sampled observed positives, not missing-label or sampling bias.",
            "Development comparisons and confidence intervals are exploratory.",
            "BM25 historical latency includes three routes; dense latency covers one route. "
            "Do not present their raw ratio as an isolated retriever speed comparison.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    write_jsonl(output / "diagnosis.jsonl", diagnosis)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "bm25", "dense", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.dataset, args.bm25, args.dense, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
