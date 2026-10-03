"""Reproduce private case evaluation and export a strictly aggregate public report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from .case_eval import evaluate, load_labels, load_runs, pool_rows, private_open
from .common import file_hash
from .reranker import digest, select_results

VERSION = "case-relevance-report-v1"
MODES = ("problem", "combined", "address")
METHODS = ("baseline", "reranked", "filtered")
REASONS = ("problem_mismatch", "address_conflict", "insufficient_evidence")
LIMITATIONS = [
    "探索性留出结果不代表总体准确率；同一场景的多种查询不是独立样本。",
    "Precision@k 使用固定 k 分母；returned_precision 使用实际返回数量。",
    "有用结果覆盖要求至少一条 grade=2，区别于仅要求返回结果的 query_coverage。",
    "nDCG 的理想排序仅来自共同人工池；无正增益时为 null，不报告全库 Recall。",
    "池内未观察到 grade=2 不代表全库没有相关案例；正常空候选不属于执行失败。",
    "错误原因只统计人工备注中的明确方括号标记；可重叠，未标记不视为否定判断。",
    "grade=0 或 grade=1 本身不能证明具体地点冲突、问题原因或证据不足。",
    "查询级截断总数不能定位具体候选，也不能证明排序退化由截断造成。",
    "嵌套计时不相加；冷启动与后续查询分开统计，不从一批推断线上 SLA。",
]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _cutoffs(evaluation: dict) -> tuple[int, int]:
    config = evaluation.get("metric_config")
    if config is not None:
        k, ndcg_k = config.get("k"), config.get("ndcg_k")
    else:
        metrics = evaluation.get("metrics_by_mode", {}).get("all", {}).get("baseline", {})
        precision = [key for key in metrics if re.fullmatch(r"precision@[1-9]\d*", key)]
        ndcg = [key for key in metrics if re.fullmatch(r"pooled_ndcg@[1-9]\d*", key)]
        if len(precision) != 1 or len(ndcg) != 1:
            raise ValueError("Evaluation metric cutoffs are missing or ambiguous")
        k, ndcg_k = int(precision[0].split("@")[1]), int(ndcg[0].split("@")[1])
    if any(type(value) is not int or value < 1 for value in (k, ndcg_k)):
        raise ValueError("Invalid evaluation metric cutoffs")
    return k, ndcg_k


def _number(value):
    if value is None or (
        type(value) in (int, float) and math.isfinite(value) and value >= 0
    ):
        return value
    raise ValueError("Aggregate report received a nonnumeric metric")


def _stats(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "sum": sum(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _timing_summary(runs: list[dict]) -> dict:
    # Only known numeric durations can cross the public boundary.
    fields = (
        "seconds_search", "seconds_reranker_load", "seconds_rerank",
        "seconds_query_total", "seconds_retrieval", "seconds_reranker_scoring",
        "seconds_total", "seconds_search_this_query", "seconds_scoring",
        "retrieval_total", "reranker_model_load", "reranker_scoring", "query_total",
    )
    samples = {group: {field: [] for field in fields} for group in ("cold", "warm", "unknown")}
    for run in runs:
        timing = run.get("timing_seconds", run.get("timing", run.get("timings", {})))
        cold = timing.get("first_reranker_load", timing.get("cold_start"))
        group = "cold" if cold is True else "warm" if cold is False else "unknown"
        for field in fields:
            value = timing.get(field)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                samples[group][field].append(value)
    return {
        group: {field: _stats(values) for field, values in entries.items() if values}
        for group, entries in samples.items()
    }


def _execution_summary(status: dict | None) -> dict:
    if status is None:
        return {"status": "not_recorded"}
    permitted = {
        "running", "completed", "failed", "interrupted", "success", "input_error",
        "execution_error", "no_scored_candidates", "insufficient_support", "no_qualified_threshold",
    }
    value = status.get("status")
    result = {"status": value if value in permitted else "unrecognized"}
    for field in ("planned_queries", "completed_queries", "query_count", "completed_count"):
        if type(status.get(field)) is int and status[field] >= 0:
            result[field] = status[field]
    if isinstance(status.get("modes"), dict):
        result["modes"] = {
            mode: _execution_summary(status["modes"][mode])
            for mode in MODES
            if isinstance(status["modes"].get(mode), dict)
        }
    return result


def _reason_codes(notes: str) -> list[str]:
    return sorted(set(re.findall(
        r"\[(problem_mismatch|address_conflict|insufficient_evidence)\]", notes
    )))


def _review_summary(labels_dir: Path, label_freeze: dict | None) -> dict:
    if label_freeze is None:
        return {"status": "not_formally_frozen"}
    summary = label_freeze["review"]
    return {
        "status": "frozen",
        "reviewed_pairs": _number(summary["reviewed_pairs"]),
        "pool_pairs": _number(summary["pool_pairs"]),
        "reviewer_count": len(summary["reviewers"]),
        "annotator_count": _number(summary["annotator_count"]),
        "independent_reviewer_count": _number(summary["independent_reviewer_count"]),
        "methods": {
            method: _number(summary["methods"][method])
            for method in ("independent", "delayed_self")
        },
    }


def _grade(value: str) -> int | None:
    return int(value.strip()) if value.strip() in {"0", "1", "2"} else None


def _error_summary(pairs: list[dict]) -> dict:
    assessed = sum(bool(pair["explicit_reasons"]) for pair in pairs)
    result = {
        "returned_pairs": len(pairs),
        "explicit_reason_annotated_pairs": assessed,
        "reason_counts": {
            reason: {
                "count": sum(reason in pair["explicit_reasons"] for pair in pairs),
                "denominator": assessed,
            }
            for reason in REASONS
        },
        "unclassified_non_direct_pairs": sum(
            pair["effective_grade"] < 2 and not pair["explicit_reasons"] for pair in pairs
        ),
    }
    for dimension in ("problem_grade", "address_grade"):
        judged = [pair[dimension] for pair in pairs if pair[dimension] is not None]
        result[dimension] = {
            "denominator": len(judged),
            "counts": {str(grade): judged.count(grade) for grade in range(3)},
        }
    return result


def build_report(
    run_dir: Path,
    labels_dir: Path,
    evaluation_path: Path,
    *,
    drill: bool = False,
    policy_path: Path | None = None,
    calibration_status_path: Path | None = None,
) -> tuple[dict, dict]:
    """Verify provenance and reproduce metrics before constructing either report.

    The public object is assembled from fixed keys, enumerations and numbers.
    No arbitrary input strings, notes, paths, queries or record IDs are copied.
    """
    run_dir, labels_dir, evaluation_path = map(Path, (run_dir, labels_dir, evaluation_path))
    all_runs = load_runs(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    label_freeze = None
    if not drill:
        from .case_eval import require_formal_run, verify_label_freeze

        manifest = require_formal_run(run_dir)
        label_freeze = verify_label_freeze(run_dir, labels_dir)
    collection_path = run_dir / "collection-status.json"
    collection = _read_json(collection_path) if collection_path.exists() else None
    if collection is not None and collection.get("status") != "completed":
        raise ValueError("Incomplete collection cannot produce quality metrics")

    supplied = _read_json(evaluation_path)
    k, ndcg_k = _cutoffs(supplied)
    partition = supplied.get("partition")
    if partition not in {"calibration", "evaluation"}:
        raise ValueError("Unknown evaluation partition")
    if not drill and (partition != "evaluation" or (k, ndcg_k) != (5, 5)):
        raise ValueError("Formal report requires held-out evaluation at k=5 and ndcg_k=5")
    judgments_hash = file_hash(labels_dir / "judgments.tsv")
    if supplied.get("judgments_file_sha256") not in {None, judgments_hash}:
        raise ValueError("Evaluation judgment file fingerprint mismatch")
    if not drill and supplied.get("judgments_file_sha256") != judgments_hash:
        raise ValueError("Formal evaluation must bind the exact judgment file")
    if not drill and supplied.get("label_freeze_sha256") != file_hash(
        labels_dir / "label-freeze.json"
    ):
        raise ValueError("Evaluation label freeze fingerprint mismatch")
    policy = _read_json(Path(policy_path)) if policy_path is not None else supplied.get("policy")
    if supplied.get("policy_sha256") != (digest(policy) if policy else None):
        raise ValueError("Evaluation policy missing or fingerprint mismatch")
    expected = evaluate(
        run_dir, labels_dir, partition=partition, k=k, ndcg_k=ndcg_k, policy=policy
    )
    for field in (
        "version", "partition", "run_manifest_sha256", "judgments_sha256", "policy_sha256",
        "metrics_by_mode", "per_query",
    ):
        if supplied.get(field) != expected.get(field):
            raise ValueError(f"Evaluation provenance or recomputed metrics mismatch: {field}")
    selected, grades = load_labels(
        run_dir, labels_dir, partition=partition, required_depth=max(k, ndcg_k)
    )
    if any(run["query"]["mode"] not in MODES for run in selected):
        raise ValueError("Unsupported report mode")
    pool_manifest = _read_json(labels_dir / "manifest.json")
    pool = pool_rows(all_runs, pool_manifest["pool_depth"])
    annotations = {}
    with (labels_dir / "judgments.tsv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            item = pool[row["annotation_id"]]
            annotations[(item["query"]["query_id"], item["source_id"])] = row

    methods = ["baseline", "reranked"] + (["filtered"] if policy is not None else [])
    paired = {row["query_id"]: row for row in expected["per_query"]}
    scenarios, diagnostics = {}, []
    for run in selected:
        query = run["query"]
        qid = query["query_id"]
        stats = run["reranked"].get("reranking", {}).get("stats", {})
        truncated = stats.get("truncated_pairs")
        diagnostic = {
            "query_id": qid, "scenario_id": query["scenario_id"], "mode": query["mode"],
            "query": query["query"], "address": query.get("address"),
            "empty_candidates": not run["baseline"]["results"],
            "no_observed_grade2": not any(
                grade == 2 for (item_qid, _), grade in grades.items() if item_qid == qid
            ),
            "observations": {
                "candidate_count": len(run["baseline"]["results"]),
                "query_truncated_pairs": _number(truncated),
                "oom_retries": _number(stats.get("oom_retries")),
            },
            "hypotheses": (
                ["存在查询级截断；具体候选是否截断及其对排序的影响待验证。"]
                if type(truncated) in (int, float) and truncated > 0 else []
            ),
            "methods": {},
        }
        for method in methods:
            report = run["baseline" if method == "baseline" else "reranked"]
            if method == "filtered" and query["mode"] != "address":
                report = select_results(report, top_k=k, policy=policy)
            pairs = []
            for row in report["results"][:k]:
                label = annotations[(qid, row["source_id"])]
                reasons = _reason_codes(label.get("notes", ""))
                pairs.append({
                    "source_id": row["source_id"], "rank": row["rank"],
                    "case_content": row["case_content"],
                    "problem_grade": _grade(label["problem_grade"]),
                    "address_grade": _grade(label["address_grade"]),
                    "effective_grade": grades[(qid, row["source_id"])],
                    "notes": label.get("notes", ""), "explicit_reasons": reasons,
                    "evidence_level": "human_explicit_note" if reasons else "grade_only",
                })
            metric = paired[qid]["metrics"][method]
            diagnostic["methods"][method] = {
                "metrics": metric, "returned_pairs": pairs,
                "has_useful_result": metric["relevant_returned"] > 0,
                "empty_returned": not pairs,
            }
        before = diagnostic["methods"]["baseline"]["metrics"]
        diagnostic["paired_changes"] = {
            method: {
                "relevant_returned_delta": (
                    diagnostic["methods"][method]["metrics"]["relevant_returned"]
                    - before["relevant_returned"]
                ),
                "returned_delta": (
                    diagnostic["methods"][method]["metrics"]["returned"] - before["returned"]
                ),
                f"precision@{k}_delta": (
                    diagnostic["methods"][method]["metrics"][f"precision@{k}"]
                    - before[f"precision@{k}"]
                ),
                f"pooled_ndcg@{ndcg_k}_delta": (
                    diagnostic["methods"][method]["metrics"][f"pooled_ndcg@{ndcg_k}"]
                    - before[f"pooled_ndcg@{ndcg_k}"]
                    if before[f"pooled_ndcg@{ndcg_k}"] is not None else None
                ),
            }
            for method in methods if method != "baseline"
        }
        diagnostics.append(diagnostic)
        scenarios.setdefault(query["scenario_id"], []).append(diagnostic)

    metrics = {}
    for mode in ("all", *MODES):
        subset = [item for item in diagnostics if mode == "all" or item["mode"] == mode]
        if not subset:
            continue
        metrics[mode] = {}
        for method in methods:
            original = expected["metrics_by_mode"][mode][method]
            allowed = (
                "queries", f"precision@{k}", f"pooled_ndcg@{ndcg_k}",
                "ndcg_queries_with_pool_gain", "returned_precision", "query_coverage",
                "mean_returned", "queries_without_observed_pool_positive",
            )
            aggregate = {key: _number(original[key]) for key in allowed}
            values = [item["methods"][method]["metrics"] for item in subset]
            useful = sum(value["relevant_returned"] > 0 for value in values)
            aggregate.update({
                "query_denominator": len(subset), "precision_denominator": len(subset) * k,
                "returned_pairs": sum(value["returned"] for value in values),
                "relevant_returned_pairs": sum(value["relevant_returned"] for value in values),
                "useful_queries": useful, "useful_result_coverage": useful / len(subset),
                "empty_candidates": sum(item["empty_candidates"] for item in subset),
                "empty_returned_queries": sum(value["returned"] == 0 for value in values),
                "no_observed_grade2": sum(item["no_observed_grade2"] for item in subset),
                "errors": _error_summary([
                    pair for item in subset for pair in item["methods"][method]["returned_pairs"]
                ]),
            })
            metrics[mode][method] = aggregate

    calibration = None
    if calibration_status_path is not None:
        calibration = _read_json(Path(calibration_status_path))
        if calibration.get("run_manifest_sha256") != file_hash(run_dir / "manifest.json"):
            raise ValueError("Calibration status belongs to a different run")
    public = {
        "version": VERSION,
        "experiment_kind": "drill" if drill else "formal",
        "partition": partition, "metric_config": {"k": k, "ndcg_k": ndcg_k},
        "counts": {"queries": len(selected), "independent_scenarios": len(scenarios)},
        "metrics_by_mode": metrics,
        "human_review": _review_summary(labels_dir, label_freeze),
        "execution": {
            "collection": _execution_summary(collection),
            "calibration": _execution_summary(calibration),
            "quality_metrics_require_completed_collection": True,
        },
        "timings": {
            "queries_by_model_load": _timing_summary(selected),
            "whole_run": {
                key: _number(manifest.get("timing_seconds", {}).get(key))
                for key in ("searcher_initialization", "batch_total")
            },
        },
        "reranking": {
            key: {
                "observed_queries": sum(
                    item["observations"][key] is not None for item in diagnostics
                ),
                "total": sum(item["observations"][key] or 0 for item in diagnostics),
            }
            for key in ("query_truncated_pairs", "oom_retries")
        },
        "limitations": list(LIMITATIONS),
    }
    if drill:
        public["limitations"].append("演练产物不得作为正式模型或真实业务效果的验收证据。")
    elif not public["human_review"]["independent_reviewer_count"]:
        if public["human_review"]["methods"]["delayed_self"]:
            public["limitations"].append(
                "本实验为单人标注与隔日复核，不提供双人一致性证据。"
                if public["human_review"]["annotator_count"] == 1
                else "本实验采用标注者隔日自审，未记录独立第二人复核，不提供双人一致性证据。"
            )
        else:
            public["limitations"].append("没有独立第二人复核记录，不提供双人一致性证据。")
    if supplied.get("judgments_file_sha256") is None:
        public["limitations"].append(
            "旧版演练评估未绑定完整标签文件；已核对有效评分指纹，分项与备注取当前标签。"
        )
    private = {
        "version": VERSION, "experiment_kind": public["experiment_kind"],
        "provenance": {
            "run_manifest_sha256": file_hash(run_dir / "manifest.json"),
            "evaluation_file_sha256": file_hash(evaluation_path),
            "judgments_file_sha256": judgments_hash,
            "judgments_sha256": expected["judgments_sha256"],
            "policy_sha256": expected["policy_sha256"],
            "label_freeze": label_freeze,
        },
        "execution": {"collection": collection, "calibration": calibration},
        "scenarios": [
            {"scenario_id": scenario, "queries": entries} for scenario, entries in scenarios.items()
        ],
        "limitations": list(LIMITATIONS),
    }
    return public, private


def public_markdown(public: dict) -> str:
    k = public["metric_config"]["k"]
    ndcg_k = public["metric_config"]["ndcg_k"]
    lines = [
        "# 案例检索聚合报告", "",
        f"实验类型：{public['experiment_kind']}；分区：{public['partition']}。",
        f"查询数：{public['counts']['queries']}；独立场景数："
        f"{public['counts']['independent_scenarios']}。", "",
        "| 模式 | 路线 | 查询数 | Precision | pooled nDCG | nDCG有效分母 | "
        "返回精度 | 返回对数 | 直接相关对数 | 有用覆盖 | 空候选查询 | 池内无直接正例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def display(value):
        return "null" if value is None else f"{value:.4f}" if type(value) is float else str(value)

    for mode, methods in public["metrics_by_mode"].items():
        for method, values in methods.items():
            cells = [mode, method] + [display(values[key]) for key in (
                "queries", f"precision@{k}", f"pooled_ndcg@{ndcg_k}",
                "ndcg_queries_with_pool_gain", "returned_precision", "returned_pairs",
                "relevant_returned_pairs", "useful_result_coverage", "empty_candidates",
                "no_observed_grade2",
            )]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"Precision 截止位：{k}；nDCG 截止位：{ndcg_k}。", ""]
    review = public["human_review"]
    if review["status"] == "frozen":
        lines += [
            f"人工复核：{review['reviewed_pairs']}/{review['pool_pairs']} 对；"
            f"独立复核者 {review['independent_reviewer_count']} 人；"
            f"隔日自审 {review['methods']['delayed_self']} 对。", "",
        ]
    lines += [f"- {text}" for text in public["limitations"]]
    lines += ["", "聚合错误分项、计时有效样本量和执行状态见同目录 report.json。", ""]
    return "\n".join(lines)


def private_markdown(private: dict) -> str:
    lines = ["# 私有逐场景诊断", "", "含查询与业务文本，仅供内部复核。", ""]
    for scenario in private["scenarios"]:
        lines += [f"## 场景 {scenario['scenario_id']}", ""]
        for query in scenario["queries"]:
            lines += [f"### {query['mode']} / {query['query_id']}", "",
                      f"查询：{query['query']}", "",
                      "```json", json.dumps(query, ensure_ascii=False, indent=2, allow_nan=False),
                      "```", ""]
    return "\n".join(lines)


def write_report(output: Path, public: dict, private: dict) -> None:
    """Create a fresh bundle; never replace earlier report artifacts."""
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name in ("public", "private"):
        (output / name).mkdir(mode=0o700)
    for relative, content in (
        ("public/report.json", json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False)),
        ("public/report.md", public_markdown(public)),
        ("private/diagnostics.json", json.dumps(private, ensure_ascii=False, indent=2,
                                               allow_nan=False)),
        ("private/diagnostics.md", private_markdown(private)),
    ):
        with private_open(output / relative) as target:
            target.write(content + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "labels", "evaluation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, help="Private policy for legacy filtered evaluations"
    )
    parser.add_argument("--calibration-status", type=Path)
    parser.add_argument("--drill", action="store_true", help="Explicitly allow nonformal artifacts")
    args = parser.parse_args()
    public, private = build_report(
        args.run, args.labels, args.evaluation, drill=args.drill, policy_path=args.policy,
        calibration_status_path=args.calibration_status,
    )
    write_report(args.output, public, private)
    print(json.dumps({"status": "completed", "experiment_kind": public["experiment_kind"],
                      "queries": public["counts"]["queries"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
