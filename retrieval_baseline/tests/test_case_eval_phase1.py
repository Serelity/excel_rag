"""Synthetic end-to-end formal workflow; these fixtures are never business gold labels."""

from __future__ import annotations

import copy
import csv
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrieval_baseline import case_eval, case_lifecycle, case_report, phase1
from retrieval_baseline.tests import test_phase1 as phase1_fixtures
from retrieval_baseline.tests.test_phase1 import write_json, write_rows
from retrieval_baseline.tests.test_reranker import FakeReranker

frozen = phase1_fixtures.frozen
prepared = phase1_fixtures.prepared

DOCUMENTS = [
    ("SYNTHETIC_PRIVATE_NEGATIVE_A", -5, 0),
    ("=SYNTHETIC_PRIVATE_NEGATIVE_B", -4, 0),
    ("SYNTHETIC_PRIVATE_PARTIAL\n中文第二行\t保留制表符", 0, 1),
    ("SYNTHETIC_PRIVATE_POSITIVE_A", 4, 2),
    ("SYNTHETIC_PRIVATE_NEGATIVE_C", -3, 0),
    ("SYNTHETIC_PRIVATE_POSITIVE_B", 5, 2),
]


class SyntheticSearcher:
    def __init__(self):
        self.calls = []
        self.empty_identity = None
        self.failure_at = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def search(self, query, *, mode, address, retriever, top_k, case_k, max_terms):
        self.calls.append((mode, query, address))
        if len(self.calls) == self.failure_at:
            raise RuntimeError("Synthetic retrieval failure")
        rows = [
            {"source_id": f"synthetic-candidate-{index}", "source_row": index,
             "case_content": text, "rank": index,
             "matching": {"routes": {}, "address": None}}
            for index, (text, _, _) in enumerate(DOCUMENTS, 1)
        ]
        if (mode, query) == self.empty_identity:
            rows = []
        return {
            "version": "case-search-v1", "mode": mode, "query": query,
            "address_query": query if mode == "address" else address, "retriever": retriever,
            "results": rows, "result_count": len(rows),
            "result_status": "results_available" if rows else "no_candidates",
            "config": {"input_field": "case_content", "top_k": top_k, "case_k": case_k,
                       "max_terms": max_terms, "rrf_k": 60, "allow_broader": False},
            "dataset_manifest_sha256": "synthetic", "lexical_index_manifest_sha256": "lexical",
            "dense_index_manifest_sha256": "dense", "address_index_manifest_sha256": "address",
            "model_sha256": "embedding", "implementation_sha256": {"search.py": "synthetic"},
            "seconds_search_this_query": 0,
        }


@pytest.fixture
def collection(frozen, tmp_path, monkeypatch):
    args = copy.copy(frozen)
    args.freeze_manifest = frozen.output
    args.output = tmp_path / "experiment" / "run-001"
    args.drill = False
    args.reranker_model = Path("synthetic-model-only")
    searcher = SyntheticSearcher()
    reranker = FakeReranker({text: score for text, score, _ in DOCUMENTS})
    constructions = []

    def make_searcher(_args):
        constructions.append("searcher")
        return searcher

    def make_reranker(_args):
        constructions.append("reranker")
        return reranker

    monkeypatch.setattr(case_eval, "make_searcher", make_searcher)
    monkeypatch.setattr(case_eval, "make_reranker", make_reranker)
    return SimpleNamespace(args=args, root=args.output.parent, run=args.output,
                           searcher=searcher, reranker=reranker, constructions=constructions)


def read_table(labels):
    with (labels / "judgments.tsv").open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_table(labels, rows):
    with (labels / "judgments.tsv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=case_eval.FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def grade_fields(row):
    return {field: int(row[field]) for field in {
        "problem": ("problem_grade",), "address": ("address_grade",),
        "combined": ("problem_grade", "address_grade"),
    }[row["mode"]]}


def review_record(row, *, initial=None, timestamp="2026-01-01T10:00:00+08:00"):
    grades = grade_fields(row)
    return {
        "annotation_id": row["annotation_id"], "reviewer": "synthetic-reviewer",
        "reviewed_at": timestamp, "method": "independent",
        "resolution": "Synthetic fixture agreement or correction; no real human label claim",
        "initial_grades": initial or grades, "review_grades": grades, "final_grades": grades,
    }


def fill_synthetic_labels(run, labels, *, failed_mode=None):
    queries = {row["query_id"]: row for row in phase1._read(run / "queries.json")}
    rows = read_table(labels)
    truth = {case_eval.safe_cell(text): grade for text, _, grade in DOCUMENTS}
    for row in rows:
        grade = truth[row["case_content"]]
        if row["mode"] == failed_mode and queries[row["query_id"]]["partition"] == "calibration":
            grade = 0
        row["problem_grade"] = str(grade) if row["mode"] != "address" else ""
        row["address_grade"] = str(grade) if row["mode"] != "problem" else ""
        row["notes"] = "synthetic fixture only"
    write_table(labels, rows)
    # Full synthetic review covers the 20% floor and each applicable query mode.
    write_rows(labels / "review-log.jsonl", [review_record(row) for row in rows])


def complete_labels(collection, *, failed_mode=None):
    case_eval.collect(collection.args)
    labels = collection.root / "labels-001"
    case_eval.export_labels(collection.run, labels)
    fill_synthetic_labels(collection.run, labels, failed_mode=failed_mode)
    case_lifecycle.freeze_labels(collection.run, labels, annotator="synthetic-annotator")
    return labels


@pytest.fixture
def workflow(collection):
    labels = complete_labels(collection)
    attempt = collection.root / "calibration-001"
    status = case_eval.calibrate_attempt(collection.run, labels, attempt)
    assert status["status"] == "success", status
    return SimpleNamespace(**vars(collection), labels=labels, attempt=attempt,
                           policy=phase1._read(attempt / "policy.json"))


def test_complete_formal_workflow_to_aggregate_and_private_reports(workflow):
    baseline = case_eval.evaluate(workflow.run, workflow.labels)
    filtered = case_eval.evaluate(workflow.run, workflow.labels, policy=workflow.policy)
    assert len(baseline["per_query"]) == 12
    assert baseline["metrics_by_mode"]["problem"]["baseline"]["precision@5"] == 0.2
    assert baseline["metrics_by_mode"]["problem"]["reranked"]["precision@5"] == 0.4
    assert filtered["metrics_by_mode"]["problem"]["filtered"]["returned_precision"] == 1
    assert all(workflow.policy["modes"][mode]["threshold"] == 4 for mode in ("problem", "combined"))
    evaluation_path = workflow.root / "evaluation.json"
    write_json(evaluation_path, filtered)
    public, private = case_report.build_report(
        workflow.run, workflow.labels, evaluation_path,
        policy_path=workflow.attempt / "policy.json",
        calibration_status_path=workflow.attempt / "status.json")
    assert public["counts"] == {"queries": 12, "independent_scenarios": 4}
    assert "SYNTHETIC_PRIVATE" not in json.dumps(public)
    assert "SYNTHETIC_PRIVATE" in json.dumps(private)
    report_path = workflow.root / "report"
    case_report.write_report(report_path, public, private)
    assert (report_path / "public/report.md").is_file()
    assert (report_path / "private/diagnostics.json").is_file()
    assert workflow.constructions == ["searcher", "reranker"]
    runs = case_eval.load_runs(workflow.run)
    assert sum(row["timing_seconds"]["first_reranker_load"] for row in runs) == 1
    assert len(workflow.reranker.calls) == 20
    for row in runs:
        timing = row["timing_seconds"]
        assert timing["query_total"] >= timing["retrieval_total"]
        assert timing["query_total"] >= timing["reranker_model_load"]


def test_completed_workflow_is_portable_without_original_inputs_or_models(
        workflow, frozen, tmp_path, monkeypatch):
    original = case_eval.evaluate(workflow.run, workflow.labels, policy=workflow.policy)
    offline = tmp_path / "offline-experiment"
    shutil.copytree(workflow.root, offline)
    shutil.rmtree(workflow.root)
    shutil.rmtree(frozen.dataset)
    for name in phase1.ARTIFACTS:
        getattr(frozen, name).unlink()
    frozen.output.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Offline evaluation must not construct models or access live resources")

    monkeypatch.setattr(phase1, "verify_freeze", forbidden)
    monkeypatch.setattr(case_eval, "make_searcher", forbidden)
    monkeypatch.setattr(case_eval, "make_reranker", forbidden)
    run, labels, attempt = offline / "run-001", offline / "labels-001", offline / "calibration-001"
    result = case_eval.evaluate(run, labels, policy=phase1._read(attempt / "policy.json"))
    assert result == original
    path = offline / "evaluation.json"
    write_json(path, result)
    public, _ = case_report.build_report(run, labels, path, policy_path=attempt / "policy.json")
    assert public["counts"]["queries"] == 12


@pytest.mark.parametrize("change", ["missing", "input", "config", "manifest"])
def test_formal_collection_rejects_freeze_problems_before_loading_models(
        collection, frozen, change):
    if change == "missing":
        collection.args.freeze_manifest = None
    elif change == "input":
        frozen.provenance.write_bytes(frozen.provenance.read_bytes() + b"\n")
    elif change == "config":
        collection.args.case_k = 25
    else:
        manifest = phase1._read(frozen.output)
        manifest["review_status"] = "pending"
        write_json(frozen.output, manifest)
    with pytest.raises(ValueError):
        case_eval.collect(collection.args)
    assert collection.constructions == []
    status = phase1._read(collection.run / "collection-status.json")
    assert status["status"] == "failed"
    assert status["completed_queries"] == 0
    assert not (collection.run / "manifest.json").exists()


@pytest.mark.parametrize("kind", ["drill", "legacy"])
def test_legacy_and_drill_cannot_enter_formal_workflow(collection, tmp_path, kind):
    collection.args.drill = True
    collection.args.freeze_manifest = None
    case_eval.collect(collection.args)
    if kind == "legacy":
        path = collection.run / "manifest.json"
        manifest = phase1._read(path)
        manifest["version"] = case_eval.RUN_VERSION
        manifest.pop("experiment_kind")
        write_json(path, manifest)
    with pytest.raises(ValueError, match="Formal workflow"):
        case_eval.require_formal_run(collection.run)
    result = case_eval.calibrate_attempt(
        collection.run, tmp_path / "nonexistent-labels", collection.root / "calibration-rejected")
    assert result["status"] == "input_error"
    assert not (collection.root / "calibration-rejected/policy.json").exists()


def test_interrupted_collection_never_yields_formal_labels(collection):
    collection.searcher.failure_at = 2
    with pytest.raises(RuntimeError, match="Synthetic"):
        case_eval.collect(collection.args)
    status = phase1._read(collection.run / "collection-status.json")
    assert status["status"] == "failed"
    assert status["completed_queries"] == 1
    assert status["failed_query_id"]
    assert not (collection.run / "manifest.json").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        case_eval.require_formal_run(collection.run)


def test_completed_manifest_cannot_override_running_status(workflow):
    path = workflow.run / "collection-status.json"
    status = phase1._read(path)
    status["status"] = "running"
    write_json(path, status)
    with pytest.raises(ValueError, match="Incomplete collection"):
        case_eval.evaluate(workflow.run, workflow.labels)


def test_empty_result_is_successful_and_remains_in_query_denominator(collection):
    query = next(row for row in phase1._read(collection.args.queries)
                 if row["mode"] == "address" and row["partition"] == "evaluation")
    collection.searcher.empty_identity = (query["mode"], query["query"])
    labels = complete_labels(collection)
    status = phase1._read(collection.run / "collection-status.json")
    assert status["status"] == "completed"
    assert status["empty_query_ids"] == [query["query_id"]]
    report = case_eval.evaluate(collection.run, labels)
    assert report["metrics_by_mode"]["address"]["baseline"]["queries"] == 4
    assert report["metrics_by_mode"]["address"]["baseline"]["query_coverage"] == 0.75


def test_one_mode_calibration_failure_publishes_no_policy_but_allows_unfiltered_report(collection):
    labels = complete_labels(collection, failed_mode="combined")
    attempt = collection.root / "calibration-001"
    status = case_eval.calibrate_attempt(collection.run, labels, attempt)
    assert status["status"] == "failed"
    assert status["modes"]["problem"]["status"] == "success"
    assert status["modes"]["combined"]["status"] == "no_qualified_threshold"
    assert not (attempt / "policy.json").exists()
    assert set(phase1._read(attempt / "threshold-search.json")) == {"problem", "combined"}
    evaluation = case_eval.evaluate(collection.run, labels)
    assert set(evaluation["metrics_by_mode"]["all"]) == {"baseline", "reranked"}
    path = collection.root / "evaluation-unfiltered.json"
    write_json(path, evaluation)
    public, _ = case_report.build_report(
        collection.run, labels, path, calibration_status_path=attempt / "status.json")
    assert public["execution"]["calibration"]["status"] == "failed"


def revised_labels(workflow, *, partition, notes_only=False):
    labels = workflow.root / ("labels-notes" if notes_only else "labels-" + partition)
    case_lifecycle.revise_labels(
        workflow.run, workflow.labels, labels, reason="Synthetic correction")
    queries = {row["query_id"]: row for row in phase1._read(workflow.run / "queries.json")}
    rows = read_table(labels)
    row = next(row for row in rows if queries[row["query_id"]]["partition"] == partition
               and row["mode"] == "problem" and row["problem_grade"] == "2")
    initial = grade_fields(row)
    if not notes_only:
        row["problem_grade"] = "1"
    row["notes"] = "Synthetic reviewed correction"
    write_table(labels, rows)
    if not notes_only:
        old_freeze = phase1._read(workflow.labels / "label-freeze.json")
        reviewed = (
            datetime.fromisoformat(old_freeze["frozen_at"]) + timedelta(seconds=1)).isoformat()
        reviews = phase1._rows(labels / "review-log.jsonl")
        reviews = [review_record(row, initial=initial, timestamp=reviewed)
                   if old["annotation_id"] == row["annotation_id"] else old for old in reviews]
        write_rows(labels / "review-log.jsonl", reviews)
    case_lifecycle.freeze_labels(workflow.run, labels, annotator="synthetic-annotator")
    return labels


def test_calibration_label_revision_rejects_old_policy(workflow):
    labels = revised_labels(workflow, partition="calibration")
    revision = phase1._read(labels / "label-freeze.json")["revision"]
    assert revision["changed_effective_partitions"] == ["calibration"]
    assert revision["policy_action"] == "recalibrate"
    with pytest.raises(ValueError, match="Stale policy"):
        case_eval.evaluate(workflow.run, labels, policy=workflow.policy)
    assert len(case_eval.evaluate(workflow.run, labels)["per_query"]) == 12


def test_evaluation_only_revision_reuses_frozen_policy_and_moves_portably(workflow, tmp_path):
    labels = revised_labels(workflow, partition="evaluation")
    policy_bytes = (workflow.attempt / "policy.json").read_bytes()
    result = case_eval.evaluate(workflow.run, labels, policy=workflow.policy)
    revision = phase1._read(labels / "label-freeze.json")["revision"]
    assert revision["policy_action"] == "keep_thresholds"
    assert (workflow.attempt / "policy.json").read_bytes() == policy_bytes
    assert result["policy"] == workflow.policy
    moved = tmp_path / "moved-revision-chain"
    shutil.copytree(workflow.root, moved)
    shutil.rmtree(workflow.root)
    repeated = case_eval.evaluate(
        moved / "run-001", moved / labels.name,
        policy=phase1._read(moved / "calibration-001/policy.json"))
    assert repeated == result


def test_notes_only_revision_keeps_effective_fingerprint_and_policy(workflow):
    labels = revised_labels(workflow, partition="evaluation", notes_only=True)
    frozen = phase1._read(labels / "label-freeze.json")
    assert frozen["revision"]["changed_effective_partitions"] == []
    result = case_eval.evaluate(workflow.run, labels, policy=workflow.policy)
    assert result["policy"] == workflow.policy


def test_modified_calibration_attempt_is_rejected(workflow):
    path = workflow.attempt / "threshold-search.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Calibration attempt"):
        case_eval.evaluate(workflow.run, workflow.labels, policy=workflow.policy)


def test_incomplete_gold_cannot_be_frozen_and_calibration_records_input_error(collection):
    case_eval.collect(collection.args)
    labels = collection.root / "labels-001"
    case_eval.export_labels(collection.run, labels)
    fill_synthetic_labels(collection.run, labels)
    rows = read_table(labels)
    row = next(row for row in rows if row["mode"] == "combined")
    row["address_grade"] = ""
    write_table(labels, rows)
    with pytest.raises(ValueError, match="Unjudged"):
        case_lifecycle.freeze_labels(collection.run, labels, annotator="synthetic-annotator")
    attempt = collection.root / "calibration-001"
    assert case_eval.calibrate_attempt(collection.run, labels, attempt)["status"] == "input_error"
    assert not (attempt / "policy.json").exists()


def test_tsv_roundtrip_preserves_formula_protection_and_multiline_text(workflow):
    rows = read_table(workflow.labels)
    texts = {row["case_content"] for row in rows}
    assert "'=SYNTHETIC_PRIVATE_NEGATIVE_B" in texts
    assert "SYNTHETIC_PRIVATE_PARTIAL\n中文第二行\t保留制表符" in texts
    assert case_lifecycle.verify_label_freeze(workflow.run, workflow.labels)
