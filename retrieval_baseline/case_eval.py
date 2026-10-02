"""Private, human-judged case retrieval experiments, separate from knowledge qrels."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path

from .common import file_hash
from .reranker import POLICY_VERSION, digest, policy_context, rerank_report, select_results
from .search import _write_private_json, add_runtime_arguments, make_reranker, make_searcher

RUN_VERSION = "case-relevance-run-v1"
POOL_VERSION = "case-judgment-pool-v1"
PARTITIONS = {"calibration", "evaluation"}
FIELDS = [
    "annotation_id",
    "query_id",
    "mode",
    "query",
    "address",
    "case_content",
    "problem_grade",
    "address_grade",
    "related_event_group",
    "notes",
]


def private_open(path: Path, *, encoding: str = "utf-8"):
    return os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
        "w",
        encoding=encoding,
        newline="",
    )


def validate_queries(rows: list[dict]) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("Queries must be a nonempty JSON array")
    rows = copy.deepcopy(rows)
    ids, identities, groups = set(), set(), {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each query must be a JSON object")
        for key in ("query_id", "scenario_id", "partition", "mode", "query"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing nonempty query field: {key}")
        if row["partition"] not in PARTITIONS or row["mode"] not in {
            "problem",
            "address",
            "combined",
        }:
            raise ValueError("Invalid query partition or mode")
        if row["mode"] == "combined" and (
            not isinstance(row.get("address"), str) or not row["address"].strip()
        ):
            raise ValueError("Combined evaluation queries require an explicit address")
        if row["mode"] != "combined" and row.get("address") is not None:
            raise ValueError("Only combined queries may provide address")
        row["query"] = row["query"].strip()
        if row["mode"] == "combined":
            row["address"] = row["address"].strip()
        identity = (row["mode"], row["query"].strip(), row.get("address"))
        if row["query_id"] in ids or identity in identities:
            raise ValueError("Duplicate query ID or query definition")
        ids.add(row["query_id"])
        identities.add(identity)
        previous = groups.setdefault(row["scenario_id"], row["partition"])
        if previous != row["partition"]:
            raise ValueError("A scenario cannot cross calibration/evaluation partitions")
    return rows


def init_queries(output: Path) -> None:
    # Authored examples, not source records or held-out baseline queries. Review before collecting.
    scenarios = [
        ("cleaning", "楼道长期没人打扫", "花语馨苑"),
        ("fire-access", "车辆堵住消防通道", "翠竹新村"),
        ("waste", "垃圾堆积多天没有清运", "泰村花园"),
        ("leak", "房屋漏水物业一直不维修", "西上园小区"),
        ("noise", "夜间施工噪音影响休息", "胡姬花园"),
        ("lighting", "路灯损坏夜间出行不安全", "劳动东路"),
        ("flooding", "下雨后道路积水无法通行", "清凉新村"),
        ("vendors", "摊贩占道经营影响通行", "中天钢铁体育馆"),
        ("fees", "物业费和停车费强制捆绑收取", "凯旋商务广场"),
        ("greenery", "公共绿化被破坏无人处理", "人民路"),
    ]
    rows = []
    for number, (scenario, problem, address) in enumerate(scenarios, 1):
        for mode in ("problem", "address", "combined"):
            row = {
                "query_id": f"q{number:02d}-{mode}",
                "scenario_id": scenario,
                "partition": "calibration" if number <= 6 else "evaluation",
                "mode": mode,
                "query": address if mode == "address" else problem,
            }
            if mode == "combined":
                row["address"] = address
            rows.append(row)
    _write_private_json(output, validate_queries(rows))


def collect(args) -> dict:
    queries = validate_queries(json.loads(args.queries.read_text(encoding="utf-8-sig")))
    if args.case_k < 1:
        raise ValueError("case_k must be positive")
    if any(q["mode"] != "address" for q in queries) and args.reranker_model is None:
        raise ValueError("Collection requires --reranker-model for paired candidate rankings")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    _write_private_json(args.output / "queries.json", queries)
    reranker = None
    with make_searcher(args) as searcher, private_open(args.output / "runs.jsonl") as target:
        for number, query in enumerate(queries, 1):
            baseline = searcher.search(
                query["query"],
                mode=query["mode"],
                address=query.get("address"),
                retriever=args.retriever,
                top_k=args.case_k,
                case_k=args.case_k,
            )
            ranked = copy.deepcopy(baseline)
            if query["mode"] != "address" and baseline["results"]:
                if reranker is None:
                    reranker = make_reranker(args)
                ranked = rerank_report(baseline, reranker, top_k=args.case_k)
            target.write(
                json.dumps(
                    {"query": query, "baseline": baseline, "reranked": ranked},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            target.flush()
            print(
                f"collected_queries={number}/{len(queries)} candidates={len(baseline['results'])}",
                flush=True,
            )
    manifest = {
        "version": RUN_VERSION,
        "query_count": len(queries),
        "artifacts": {
            name: file_hash(args.output / name) for name in ("queries.json", "runs.jsonl")
        },
        "limitations": [
            "Authored pilot queries require human review; not representative sampling.",
            "No knowledge qrels or existing frozen test queries were used.",
        ],
    }
    _write_private_json(args.output / "manifest.json", manifest)
    return manifest


def load_runs(directory: Path) -> list[dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != RUN_VERSION or any(
        file_hash(directory / name) != manifest.get("artifacts", {}).get(name)
        for name in ("queries.json", "runs.jsonl")
    ):
        raise ValueError("Run manifest/artifact mismatch or incomplete collection")
    queries = validate_queries(json.loads((directory / "queries.json").read_text(encoding="utf-8")))
    with (directory / "runs.jsonl").open(encoding="utf-8") as source:
        runs = [json.loads(line) for line in source if line.strip()]
    if [run["query"] for run in runs] != queries or len(runs) != manifest["query_count"]:
        raise ValueError("Run query definitions differ from the frozen query file")
    for run in runs:
        query = run["query"]
        for name in ("baseline", "reranked"):
            report = run[name]
            rows = report["results"]
            if (
                report["mode"] != query["mode"]
                or report["query"] != query["query"]
                or report["address_query"]
                != (query["query"] if query["mode"] == "address" else query.get("address"))
                or report["result_count"] != len(rows)
                or len({r["source_id"] for r in rows}) != len(rows)
                or [r["rank"] for r in rows] != list(range(1, len(rows) + 1))
            ):
                raise ValueError("Invalid paired ranking or query identity")
        before = {row["source_id"]: row["case_content"] for row in run["baseline"]["results"]}
        after = {row["source_id"]: row["case_content"] for row in run["reranked"]["results"]}
        if before != after:
            raise ValueError("Reranking must preserve the same candidate IDs and original texts")
        if query["mode"] != "address" and after:
            scores = [row["matching"]["reranker"]["score"] for row in run["reranked"]["results"]]
            if any(not math.isfinite(v) for v in scores) or scores != sorted(scores, reverse=True):
                raise ValueError("Invalid reranker scores/order")
    return runs


def annotation_id(query_id: str, source_id: str) -> str:
    return "pair_" + digest([query_id, source_id])[:24]


def pool_rows(runs: list[dict], depth: int) -> dict[str, dict]:
    if type(depth) is not int or depth < 1:
        raise ValueError("Pool depth must be positive")
    pool = {}
    for run in runs:
        query = run["query"]
        if depth > run["baseline"]["config"]["case_k"]:
            raise ValueError("Annotation depth exceeds collected candidate budget")
        for method in ("baseline", "reranked"):
            for row in run[method]["results"][:depth]:
                key = annotation_id(query["query_id"], row["source_id"])
                item = {
                    "query": query,
                    "source_id": row["source_id"],
                    "case_content": row["case_content"],
                }
                if key in pool and pool[key] != item:
                    raise ValueError("Conflicting annotation identity")
                pool[key] = item
    return pool


def safe_cell(value: str) -> str:
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def export_labels(run_dir: Path, output: Path, *, depth: int = 10) -> dict:
    runs = load_runs(run_dir)
    pool = pool_rows(runs, depth)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    with private_open(output / "judgments.tsv", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        # Hash order hides retrieval ranks, model scores and which route proposed a pair.
        for key, item in sorted(pool.items()):
            query = item["query"]
            writer.writerow(
                {
                    "annotation_id": key,
                    "query_id": query["query_id"],
                    "mode": query["mode"],
                    "query": safe_cell(query["query"]),
                    "address": safe_cell(
                        query.get("address")
                        or (query["query"] if query["mode"] == "address" else "")
                    ),
                    "case_content": safe_cell(item["case_content"]),
                    "problem_grade": "",
                    "address_grade": "",
                }
            )
    manifest = {
        "version": POOL_VERSION,
        "run_manifest_sha256": file_hash(run_dir / "manifest.json"),
        "pool_depth": depth,
        "pairs": len(pool),
        "query_count": len(runs),
        "empty_pool_queries": [
            r["query"]["query_id"] for r in runs if not r["baseline"]["results"]
        ],
        "rubric": {"0": "不相关", "1": "部分相关/范围更宽/证据不足", "2": "直接相关"},
        "positive_grade": 2,
        "labels": "human only; blank is unjudged, never grade 0",
    }
    _write_private_json(output / "manifest.json", manifest)
    return manifest


def load_labels(run_dir: Path, labels_dir: Path, *, partition: str, required_depth: int):
    runs = load_runs(run_dir)
    manifest = json.loads((labels_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("version") != POOL_VERSION
        or manifest.get("run_manifest_sha256") != file_hash(run_dir / "manifest.json")
        or manifest.get("pool_depth", 0) < required_depth
    ):
        raise ValueError("Judgment pool provenance/depth mismatch")
    pool = pool_rows(runs, manifest["pool_depth"])
    seen, grades = set(), {}
    with (labels_dir / "judgments.tsv").open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError("Judgment columns changed; edit only grade/group/note cells")
        for row in reader:
            key = row["annotation_id"]
            if key not in pool or key in seen:
                raise ValueError("Unknown or duplicate annotation_id")
            seen.add(key)
            item = pool[key]
            query = item["query"]
            if row["query_id"] != query["query_id"] or row["mode"] != query["mode"]:
                raise ValueError("Judgment query identity was changed")
            expected_text = {
                "query": safe_cell(query["query"]),
                "address": safe_cell(
                    query.get("address") or (query["query"] if query["mode"] == "address" else "")
                ),
                "case_content": safe_cell(item["case_content"]),
            }
            if any(
                (row.get(key) or "").replace("\r\n", "\n") != text.replace("\r\n", "\n")
                for key, text in expected_text.items()
            ):
                raise ValueError(
                    "Judgment query/case text changed; edit only grade/group/note cells"
                )
            if query["partition"] != partition:
                continue  # In particular: calibration never consumes evaluation grades.
            fields = (
                {
                    "problem": ["problem_grade"],
                    "address": ["address_grade"],
                    "combined": ["problem_grade", "address_grade"],
                }
            )[query["mode"]]
            if any(row[field].strip() not in {"0", "1", "2"} for field in fields):
                raise ValueError(
                    f"Unjudged/invalid required grade: {key}; blanks are not negatives"
                )
            grades[(query["query_id"], item["source_id"])] = min(
                int(row[f].strip()) for f in fields
            )
    if seen != set(pool):
        raise ValueError("Judgment rows were deleted or the pool changed")
    selected = [r for r in runs if r["query"]["partition"] == partition]
    if not selected:
        raise ValueError("No queries in requested partition")
    return selected, grades


def query_metrics(rows: list[dict], grades: dict[str, int], *, k: int, ndcg_k: int) -> dict:
    def relevance(row):
        if row["source_id"] not in grades:
            raise ValueError("Ranked candidate is unjudged; extend the common annotation pool")
        return grades[row["source_id"]]

    top = rows[:k]
    relevant = sum(relevance(row) == 2 for row in top)
    gains = [
        (2 ** relevance(row) - 1) / math.log2(rank + 2) for rank, row in enumerate(rows[:ndcg_k])
    ]
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 2)
        for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:ndcg_k])
    )
    return {
        f"precision@{k}": relevant / k,
        f"pooled_ndcg@{ndcg_k}": sum(gains) / ideal if ideal else None,
        "returned": len(top),
        "relevant_returned": relevant,
        "judged_pool_positives": sum(grade == 2 for grade in grades.values()),
    }


def summarize(rows: list[dict], *, k: int, ndcg_k: int) -> dict:
    ndcg = [
        row[f"pooled_ndcg@{ndcg_k}"] for row in rows if row[f"pooled_ndcg@{ndcg_k}"] is not None
    ]
    returned = sum(row["returned"] for row in rows)
    return {
        "queries": len(rows),
        f"precision@{k}": sum(row[f"precision@{k}"] for row in rows) / len(rows),
        f"pooled_ndcg@{ndcg_k}": sum(ndcg) / len(ndcg) if ndcg else None,
        "ndcg_queries_with_pool_gain": len(ndcg),
        "returned_precision": sum(row["relevant_returned"] for row in rows) / returned
        if returned
        else None,
        "query_coverage": sum(row["returned"] > 0 for row in rows) / len(rows),
        "mean_returned": returned / len(rows),
        "queries_without_observed_pool_positive": sum(
            row["judged_pool_positives"] == 0 for row in rows
        ),
    }


def evaluate(
    run_dir: Path,
    labels_dir: Path,
    *,
    partition: str = "evaluation",
    k: int = 5,
    ndcg_k: int = 5,
    policy: dict | None = None,
) -> dict:
    if k < 1 or ndcg_k < 1:
        raise ValueError("Metric cutoffs must be positive")
    if policy is not None and ndcg_k > k:
        raise ValueError("Filtered nDCG cutoff must not exceed the calibrated output budget k")
    runs, grades = load_labels(
        run_dir, labels_dir, partition=partition, required_depth=max(k, ndcg_k)
    )
    metrics, paired = {}, []
    methods = ["baseline", "reranked"] + (["filtered"] if policy is not None else [])
    for run in runs:
        query = run["query"]
        truth = {sid: grade for (qid, sid), grade in grades.items() if qid == query["query_id"]}
        result = {"query_id": query["query_id"], "mode": query["mode"], "metrics": {}}
        for method in methods:
            report = run["baseline" if method == "baseline" else "reranked"]
            if method == "filtered" and query["mode"] != "address":
                report = select_results(report, top_k=k, policy=policy)
            result["metrics"][method] = query_metrics(report["results"], truth, k=k, ndcg_k=ndcg_k)
        paired.append(result)
    for mode in ["all"] + sorted({r["mode"] for r in paired}):
        subset = [row for row in paired if mode == "all" or row["mode"] == mode]
        metrics[mode] = {
            method: summarize([row["metrics"][method] for row in subset], k=k, ndcg_k=ndcg_k)
            for method in methods
        }
    return {
        "version": "case-relevance-evaluation-v1",
        "partition": partition,
        "run_manifest_sha256": file_hash(run_dir / "manifest.json"),
        "judgments_sha256": digest(sorted((qid, sid, g) for (qid, sid), g in grades.items())),
        "policy_sha256": digest(policy) if policy else None,
        "metrics_by_mode": metrics,
        "per_query": paired,
        "limitations": [
            "Pilot query results are exploratory, not population accuracy.",
            "Precision@k uses fixed k; returned_precision uses the actual returned count.",
            "nDCG ideal ranking is limited to the common judged pool; no corpus Recall claim.",
            "No observed pool positive does not prove no relevant case exists.",
            "Cases remain separate records; no automatic same-event merging.",
        ],
    }


def calibrate(
    run_dir: Path,
    labels_dir: Path,
    *,
    k: int = 5,
    target_precision: float = 0.9,
    min_results: int = 5,
    min_queries: int = 3,
) -> dict:
    if k < 1 or min_results < 1 or min_queries < 1 or not 0 < target_precision <= 1:
        raise ValueError("Invalid calibration budget, support or target precision")
    runs, grades = load_labels(run_dir, labels_dir, partition="calibration", required_depth=k)
    settings = {}
    for mode in sorted({r["query"]["mode"] for r in runs} - {"address"}):
        subset = [r for r in runs if r["query"]["mode"] == mode]
        contexts = [policy_context(r["reranked"]) for r in subset if r["reranked"]["results"]]
        if not contexts or any(context != contexts[0] for context in contexts):
            raise ValueError(
                f"No scored candidates or inconsistent calibration profiles for {mode}"
            )
        candidates = [
            (r["query"]["query_id"], row["source_id"], row["matching"]["reranker"]["score"])
            for r in subset
            for row in r["reranked"]["results"][:k]
        ]
        choices = []
        for threshold in sorted({score for _, _, score in candidates}):
            accepted = [(qid, sid) for qid, sid, score in candidates if score >= threshold]
            query_count = len({qid for qid, _ in accepted})
            relevant = sum(grades[key] == 2 for key in accepted)
            if (
                len(accepted) >= min_results
                and query_count >= min_queries
                and relevant / len(accepted) >= target_precision
            ):
                choices.append((len(accepted), -threshold, relevant, query_count))
        if not choices:
            raise ValueError(
                f"No supported threshold reaches target precision for {mode}; "
                "review more calibration labels"
            )
        count, negative_threshold, relevant, covered = max(choices)
        settings[mode] = {
            "threshold": -negative_threshold,
            "context": contexts[0],
            "calibration_queries": len(subset),
            "accepted_results": count,
            "observed_returned_precision": relevant / count,
            "query_coverage": covered / len(subset),
        }
    if not settings:
        raise ValueError("Need problem or combined calibration queries")
    return {
        "version": POLICY_VERSION,
        "max_top_k": k,
        "modes": settings,
        "target_precision": target_precision,
        "min_results": min_results,
        "min_queries": min_queries,
        "calibration_judgments_sha256": digest(
            sorted((qid, sid, g) for (qid, sid), g in grades.items())
        ),
        "limitations": [
            "Empirical calibration target, not a population precision guarantee.",
            "Evaluation labels were not consumed; evaluate the frozen policy separately.",
            "Thresholds are raw model logits, not relevance probabilities.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-queries")
    init.add_argument("--output", type=Path, required=True)
    gathering = sub.add_parser("collect")
    add_runtime_arguments(gathering)
    gathering.add_argument("--queries", type=Path, required=True)
    gathering.add_argument("--output", type=Path, required=True)
    gathering.add_argument("--case-k", type=int, default=50)
    gathering.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="hybrid")
    for name in ("export", "evaluate", "calibrate"):
        command = sub.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "export":
            command.add_argument("--depth", type=int, default=10)
        else:
            command.add_argument("--labels", type=Path, required=True)
            command.add_argument("--k", type=int, default=5)
        if name == "evaluate":
            command.add_argument("--partition", choices=sorted(PARTITIONS), default="evaluation")
            command.add_argument("--ndcg-k", type=int, default=5)
            command.add_argument("--policy", type=Path)
        if name == "calibrate":
            command.add_argument("--target-precision", type=float, default=0.9)
            command.add_argument("--min-results", type=int, default=5)
            command.add_argument("--min-queries", type=int, default=3)
    args = parser.parse_args()
    if args.command == "init-queries":
        init_queries(args.output)
        result = {"queries": 30, "output": str(args.output)}
    elif args.command == "collect":
        result = collect(args)
    elif args.command == "export":
        result = export_labels(args.run, args.output, depth=args.depth)
    elif args.command == "calibrate":
        result = calibrate(
            args.run,
            args.labels,
            k=args.k,
            target_precision=args.target_precision,
            min_results=args.min_results,
            min_queries=args.min_queries,
        )
        _write_private_json(args.output, result)
    else:
        policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else None
        result = evaluate(
            args.run,
            args.labels,
            partition=args.partition,
            k=args.k,
            ndcg_k=args.ndcg_k,
            policy=policy,
        )
        _write_private_json(args.output, result)
        result = {key: value for key, value in result.items() if key != "per_query"}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
