from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from retrieval_baseline.case_eval import FIELDS, RUN_VERSION, calibrate, evaluate, export_labels
from retrieval_baseline.case_report import build_report, public_markdown, write_report
from retrieval_baseline.common import file_hash
from retrieval_baseline.reranker import rerank_report
from retrieval_baseline.tests.test_case_eval import raw_report
from retrieval_baseline.tests.test_reranker import FakeReranker


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _rewrite_labels(path: Path, edit):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    for row in rows:
        edit(row)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def report_bundle(tmp_path):
    run_dir, labels = tmp_path / "run-private-sentinel", tmp_path / "labels-private-sentinel"
    run_dir.mkdir()
    queries = [
        {"query_id": "query-private-" + mode, "scenario_id": "scenario-private-" + mode,
         "partition": "evaluation", "mode": mode, "query": "秘密查询原文-" + mode}
        for mode in ("problem", "combined", "address")
    ]
    queries[1]["address"] = "秘密住宅地址"
    for original in list(queries[:2]):
        query = copy.deepcopy(original)
        query.update(query_id=original["query_id"] + "-calibration",
                     scenario_id=original["scenario_id"] + "-calibration",
                     query=original["query"] + "-校准", partition="calibration")
        queries.append(query)
    runs = []
    for query in queries:
        raw = raw_report(query)
        raw["address_query"] = (
            query["query"] if query["mode"] == "address" else query.get("address")
        )
        if query["mode"] == "address":
            raw["results"] = []
            raw["result_count"] = 0
        else:
            for index, row in enumerate(raw["results"]):
                row["case_content"] = "秘密投诉正文" + str(index)
                row["source_id"] = "source-private-" + str(index)
        ranked = (copy.deepcopy(raw) if query["mode"] == "address" else rerank_report(
            raw, FakeReranker({"秘密投诉正文0": -1, "秘密投诉正文1": 3, "秘密投诉正文2": 2}),
            top_k=50,
        ))
        if query["mode"] != "address":
            ranked["reranking"]["stats"].update({"truncated_pairs": 1, "oom_retries": 0})
        runs.append({"query": query, "baseline": raw, "reranked": ranked,
                     "timing_seconds": {"retrieval_total": 0.1, "query_total": 0.3,
                                        "first_reranker_load": query["mode"] == "problem"}})
    _write_json(run_dir / "queries.json", queries)
    (run_dir / "runs.jsonl").write_text(
        "".join(json.dumps(run, ensure_ascii=False) + "\n" for run in runs), encoding="utf-8"
    )
    _write_json(run_dir / "manifest.json", {
        "version": RUN_VERSION, "query_count": len(queries),
        "artifacts": {name: file_hash(run_dir / name) for name in ("queries.json", "runs.jsonl")},
        "timing_seconds": {"searcher_initialization": 1.2, "batch_total": 2.1},
        "limitations": ["秘密原文不能因输入限制说明而进入公开报告"],
    })
    export_labels(run_dir, labels, depth=10)

    def grade(row):
        calibration = row["query_id"].endswith("-calibration")
        row["problem_grade"] = (
            "2" if (row["mode"] == "problem" or calibration)
            and row["case_content"].endswith("1") else "0"
        )
        if row["mode"] == "combined":
            row["address_grade"] = "2" if calibration else "0"
        row["notes"] = "私有标注备注"
        if row["mode"] == "problem" and row["case_content"].endswith("0"):
            row["notes"] += " [problem_mismatch] [insufficient_evidence]"

    _rewrite_labels(labels / "judgments.tsv", grade)
    evaluation = tmp_path / "evaluation-private.json"
    result = evaluate(run_dir, labels)
    result["judgments_file_sha256"] = file_hash(labels / "judgments.tsv")
    _write_json(evaluation, result)
    return run_dir, labels, evaluation


def test_report_preserves_denominators_useful_coverage_and_no_positive(report_bundle):
    public, private = build_report(*report_bundle, drill=True)
    assert public["counts"] == {"queries": 3, "independent_scenarios": 3}
    problem = public["metrics_by_mode"]["problem"]["baseline"]
    assert problem["precision@5"] == pytest.approx(1 / 5)
    assert problem["precision_denominator"] == 5
    assert problem["returned_precision"] == pytest.approx(1 / 3)
    assert problem["useful_result_coverage"] == 1
    overall = public["metrics_by_mode"]["all"]["baseline"]
    assert overall["precision_denominator"] == 15
    assert overall["query_denominator"] == 3
    assert overall["returned_pairs"] == 6
    assert overall["useful_queries"] == 1
    assert overall["useful_result_coverage"] == pytest.approx(1 / 3)
    assert overall["ndcg_queries_with_pool_gain"] == 1
    assert overall["no_observed_grade2"] == 2
    assert overall["empty_candidates"] == 1
    combined = public["metrics_by_mode"]["combined"]["baseline"]
    assert combined["pooled_ndcg@5"] is None
    assert combined["ndcg_queries_with_pool_gain"] == 0
    assert combined["query_coverage"] == 1
    assert combined["useful_result_coverage"] == 0
    empty = public["metrics_by_mode"]["address"]["baseline"]
    assert empty["returned_precision"] is None
    assert empty["empty_candidates"] == 1
    assert len(private["scenarios"]) == 3


def test_public_export_excludes_all_private_strings_and_raw_identifiers(report_bundle, tmp_path):
    public, private = build_report(*report_bundle, drill=True)
    serialized = json.dumps(public, ensure_ascii=False) + public_markdown(public)
    for secret in (
        "秘密", "私有标注备注", "query-private-", "scenario-private-", "source-private-",
        "query_id", "scenario_id", "source_id", "case_content", "notes", str(tmp_path),
    ):
        assert secret not in serialized
    private_json = json.dumps(private, ensure_ascii=False)
    assert "秘密投诉正文" in private_json
    assert "私有标注备注" in private_json
    output = tmp_path / "report"
    write_report(output, public, private)
    assert (output / "public/report.json").is_file()
    assert (output / "public/report.md").is_file()
    assert (output / "private/diagnostics.json").is_file()
    with pytest.raises(FileExistsError):
        write_report(output, public, private)


def test_errors_require_explicit_evidence_and_truncation_stays_query_level(report_bundle):
    public, private = build_report(*report_bundle, drill=True)
    errors = public["metrics_by_mode"]["combined"]["baseline"]["errors"]
    assert errors["address_grade"]["counts"]["0"] == 3
    assert errors["reason_counts"]["address_conflict"] == {"count": 0, "denominator": 0}
    assert errors["unclassified_non_direct_pairs"] == 3
    tagged = public["metrics_by_mode"]["problem"]["baseline"]["errors"]
    assert tagged["reason_counts"]["problem_mismatch"] == {"count": 1, "denominator": 1}
    assert tagged["reason_counts"]["insufficient_evidence"]["count"] == 1
    query = private["scenarios"][0]["queries"][0]
    assert query["observations"]["query_truncated_pairs"] == 1
    assert query["hypotheses"]
    assert "truncated" not in query["methods"]["baseline"]["returned_pairs"][0]
    assert public["reranking"]["query_truncated_pairs"] == {"observed_queries": 2, "total": 2}
    cold = public["timings"]["queries_by_model_load"]["cold"]
    assert cold["retrieval_total"]["samples"] == 1
    assert public["timings"]["whole_run"]["batch_total"] == 2.1


@pytest.mark.parametrize("changed", ["run_manifest_sha256", "judgments_sha256", "metrics"])
def test_stale_or_tampered_evaluation_is_rejected(report_bundle, changed):
    _, _, evaluation = report_bundle
    value = json.loads(evaluation.read_text(encoding="utf-8"))
    if changed == "metrics":
        value["metrics_by_mode"]["problem"]["baseline"]["precision@5"] = 1
    else:
        value[changed] = "wrong-provenance"
    _write_json(evaluation, value)
    with pytest.raises(ValueError, match="mismatch"):
        build_report(*report_bundle, drill=True)


def test_changed_label_notes_rejected_without_auto_repair(report_bundle):
    _, labels, _ = report_bundle
    path = labels / "judgments.tsv"
    _rewrite_labels(path, lambda row: row.update(notes="changed-private-note"))
    changed_hash = file_hash(path)
    with pytest.raises(ValueError, match="judgment file fingerprint mismatch"):
        build_report(*report_bundle, drill=True)
    assert file_hash(path) == changed_hash


def test_incomplete_collection_cannot_be_reported(report_bundle):
    run_dir, _, _ = report_bundle
    _write_json(run_dir / "collection-status.json", {"status": "running"})
    with pytest.raises(ValueError, match="[Ii]ncomplete|completed"):
        build_report(*report_bundle, drill=True)


def test_filtered_report_recomputes_policy_and_pairs_changes(report_bundle):
    run_dir, labels, evaluation = report_bundle
    policy = calibrate(run_dir, labels, min_results=1, min_queries=1)
    result = evaluate(run_dir, labels, policy=policy)
    result["policy"] = policy
    result["judgments_file_sha256"] = file_hash(labels / "judgments.tsv")
    _write_json(evaluation, result)
    public, private = build_report(*report_bundle, drill=True)
    filtered = public["metrics_by_mode"]["problem"]["filtered"]
    assert filtered["returned_pairs"] == 1
    assert filtered["returned_precision"] == 1
    assert filtered["precision@5"] == pytest.approx(1 / 5)
    query = private["scenarios"][0]["queries"][0]
    assert query["paired_changes"]["filtered"]["returned_delta"] == -2
    result["policy"]["modes"]["problem"]["threshold"] = 4
    _write_json(evaluation, result)
    with pytest.raises(ValueError, match="policy"):
        build_report(*report_bundle, drill=True)


def test_explicit_diagnostic_codes_do_not_modify_original_labels(report_bundle):
    _, labels, _ = report_bundle
    before = (labels / "judgments.tsv").read_bytes()
    build_report(*report_bundle, drill=True)
    assert (labels / "judgments.tsv").read_bytes() == before


def test_calibration_failure_is_separate_from_quality_counts(report_bundle, tmp_path):
    run_dir, _, _ = report_bundle
    status = tmp_path / "status.json"
    _write_json(status, {
        "status": "no_qualified_threshold",
        "run_manifest_sha256": file_hash(run_dir / "manifest.json"),
        "failed_query_id": "query-private-failure", "error": "秘密业务正文",
        "modes": {"problem": {"status": "insufficient_support"}},
    })
    public, _ = build_report(*report_bundle, drill=True, calibration_status_path=status)
    assert public["execution"]["calibration"]["status"] == "no_qualified_threshold"
    assert "filtered" not in public["metrics_by_mode"]["all"]
    assert "秘密" not in json.dumps(public, ensure_ascii=False)
    assert public["metrics_by_mode"]["all"]["baseline"]["query_denominator"] == 3
