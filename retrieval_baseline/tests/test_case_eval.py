from __future__ import annotations

import copy
import csv
import json
from types import SimpleNamespace

import pytest

from retrieval_baseline import case_eval
from retrieval_baseline.case_eval import (
    FIELDS,
    RUN_VERSION,
    calibrate,
    evaluate,
    export_labels,
    init_queries,
    load_labels,
    load_runs,
    query_metrics,
    validate_queries,
)
from retrieval_baseline.common import file_hash
from retrieval_baseline.reranker import rerank_report
from retrieval_baseline.tests.test_reranker import FakeReranker


def raw_report(query):
    rows = [
        {
            "source_id": sid,
            "rank": i,
            "source_row": i,
            "case_content": text,
            "matching": {"routes": {}, "address": None},
        }
        for i, (sid, text) in enumerate(
            [
                ("9000000000000000001", "消防通道堵塞"),
                ("9000000000000000002", "楼道清洁无人负责\u2028希望处理"),
                ("9000000000000000003", "=保洁长期不打扫"),
            ],
            1,
        )
    ]
    return {
        "version": "case-search-v1",
        "mode": query["mode"],
        "query": query["query"],
        "address_query": query.get("address"),
        "retriever": "hybrid",
        "results": rows,
        "result_count": len(rows),
        "config": {
            "input_field": "case_content",
            "top_k": 50,
            "case_k": 50,
            "max_terms": 32,
            "rrf_k": 60,
            "allow_broader": False,
        },
        "dataset_manifest_sha256": "dataset",
        "lexical_index_manifest_sha256": "lexical",
        "dense_index_manifest_sha256": "dense",
        "address_index_manifest_sha256": None,
        "model_sha256": "embedding",
        "implementation_sha256": {"search.py": "test"},
    }


@pytest.fixture
def bundle(tmp_path, request):
    run_dir, labels = tmp_path / "run", tmp_path / "labels"
    run_dir.mkdir()
    queries = [
        {
            "query_id": name,
            "scenario_id": name,
            "partition": partition,
            "mode": getattr(request, "param", "problem"),
            "query": name + "楼道保洁",
        }
        for name, partition in [
            ("dev-a", "calibration"),
            ("dev-b", "calibration"),
            ("eval-a", "evaluation"),
        ]
    ]
    for query in queries:
        if query["mode"] == "combined":
            query["address"] = "示例小区"
    runs = []
    model = FakeReranker(
        {"消防通道堵塞": -2, "楼道清洁无人负责\u2028希望处理": 4, "=保洁长期不打扫": 2}
    )
    for query in queries:
        raw = raw_report(query)
        runs.append(
            {"query": query, "baseline": raw, "reranked": rerank_report(raw, model, top_k=50)}
        )
    (run_dir / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    with (run_dir / "runs.jsonl").open("w", encoding="utf-8") as target:
        for run in runs:
            target.write(json.dumps(run, ensure_ascii=False) + "\n")
    manifest = {
        "version": RUN_VERSION,
        "query_count": len(queries),
        "artifacts": {name: file_hash(run_dir / name) for name in ("queries.json", "runs.jsonl")},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    export_labels(run_dir, labels, depth=10)
    return run_dir, labels


def fill_labels(labels_dir, *, fill_evaluation=False, flip_evaluation=False):
    path = labels_dir / "judgments.tsv"
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    for row in rows:
        if row["query_id"].startswith("eval") and not fill_evaluation:
            continue
        grade = 0 if "消防" in row["case_content"] else 2
        if row["query_id"].startswith("eval") and flip_evaluation:
            grade = 2 - grade
        row["problem_grade"] = str(grade)
        if row["mode"] == "combined":
            row["address_grade"] = "2"
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_initial_queries_are_reviewable_and_partitioned_by_whole_scenario(tmp_path):
    path = tmp_path / "queries.json"
    init_queries(path)
    rows = validate_queries(json.loads(path.read_text(encoding="utf-8")))
    assert len(rows) == 30
    assert sum(row["partition"] == "calibration" for row in rows) == 18
    for scenario in {row["scenario_id"] for row in rows}:
        assert len({row["partition"] for row in rows if row["scenario_id"] == scenario}) == 1
    rows[1]["partition"] = "evaluation"
    with pytest.raises(ValueError, match="scenario"):
        validate_queries(rows)
    with pytest.raises(FileExistsError):
        init_queries(path)


def test_blind_export_preserves_unicode_and_avoids_excel_numeric_ids(bundle):
    run_dir, labels = bundle
    with (labels / "judgments.tsv").open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        rows = list(reader)
        assert reader.fieldnames == FIELDS
    assert len(rows) == 9
    assert all(row["annotation_id"].startswith("pair_") for row in rows)
    assert all(row["problem_grade"] == "" for row in rows)
    assert any("\u2028" in row["case_content"] for row in rows)
    assert any(row["case_content"].startswith("'=") for row in rows)
    assert not any("score" in f or "rank" in f or f == "source_id" for f in FIELDS)
    assert len(load_runs(run_dir)) == 3


def test_blank_grades_fail_and_evaluation_grades_do_not_affect_calibration(bundle):
    run_dir, labels = bundle
    with pytest.raises(ValueError, match="Unjudged"):
        calibrate(run_dir, labels, min_results=2, min_queries=2)
    fill_labels(labels)
    first = calibrate(run_dir, labels, min_results=2, min_queries=2)
    assert first["modes"]["problem"]["threshold"] == 2
    assert first["modes"]["problem"]["accepted_results"] == 4
    with pytest.raises(ValueError, match="Unjudged"):
        evaluate(run_dir, labels)
    fill_labels(labels, fill_evaluation=True, flip_evaluation=True)
    second = calibrate(run_dir, labels, min_results=2, min_queries=2)
    assert first == second


def test_paired_evaluation_uses_fixed_denominator_and_pool_ndcg(bundle):
    run_dir, labels = bundle
    fill_labels(labels, fill_evaluation=True)
    policy = calibrate(run_dir, labels, min_results=2, min_queries=2)
    result = evaluate(run_dir, labels, policy=policy)
    metrics = result["metrics_by_mode"]["problem"]
    assert metrics["baseline"]["precision@5"] == pytest.approx(2 / 5)
    assert metrics["reranked"]["precision@5"] == pytest.approx(2 / 5)
    assert metrics["filtered"]["precision@5"] == pytest.approx(2 / 5)
    assert metrics["baseline"]["returned_precision"] == pytest.approx(2 / 3)
    assert metrics["filtered"]["returned_precision"] == 1
    assert metrics["filtered"]["mean_returned"] == 2
    assert metrics["baseline"]["pooled_ndcg@5"] < 1
    assert metrics["reranked"]["pooled_ndcg@5"] == 1
    assert result["partition"] == "evaluation"
    assert len(result["per_query"]) == 1
    with pytest.raises(ValueError, match="output budget"):
        evaluate(run_dir, labels, policy=policy, ndcg_k=10)


def test_metrics_do_not_turn_unjudged_candidates_into_negatives():
    with pytest.raises(ValueError, match="unjudged"):
        query_metrics([{"source_id": "missing"}], {}, k=5, ndcg_k=10)
    values = query_metrics([], {}, k=5, ndcg_k=10)
    assert values["precision@5"] == 0
    assert values["pooled_ndcg@10"] is None


def test_query_whitespace_is_frozen_as_actually_searched():
    original = [
        {
            "query_id": "q1",
            "scenario_id": "s1",
            "partition": "calibration",
            "mode": "combined",
            "query": "  楼道保洁\n",
            "address": " 示例小区 ",
        }
    ]
    normalized = validate_queries(original)
    assert normalized[0]["query"] == "楼道保洁"
    assert normalized[0]["address"] == "示例小区"
    assert original[0]["query"] == "  楼道保洁\n"


@pytest.mark.parametrize("field", ["query", "case_content"])
def test_modified_annotation_text_is_rejected(bundle, field):
    run_dir, labels = bundle
    fill_labels(labels, fill_evaluation=True)
    path = labels / "judgments.tsv"
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    rows[0][field] += "人工改写"
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="text changed"):
        evaluate(run_dir, labels)


@pytest.mark.parametrize("bundle", ["combined"], indirect=True)
def test_combined_requires_problem_and_address_to_be_directly_relevant(bundle):
    run_dir, labels = bundle
    fill_labels(labels, fill_evaluation=True)
    path = labels / "judgments.tsv"
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    for row in rows:
        if "清洁" in row["case_content"]:
            row["address_grade"] = "1"
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    result = evaluate(run_dir, labels)
    metrics = result["metrics_by_mode"]["combined"]["reranked"]
    assert metrics["precision@5"] == pytest.approx(1 / 5)
    assert metrics["returned_precision"] == pytest.approx(1 / 3)


def test_no_supported_threshold_cannot_succeed_by_returning_nothing(bundle):
    run_dir, labels = bundle
    fill_labels(labels)
    with pytest.raises(ValueError, match="No supported threshold"):
        calibrate(run_dir, labels, min_results=100, min_queries=2)


def test_tampered_run_or_wrong_label_pool_is_rejected(bundle):
    run_dir, labels = bundle
    with (run_dir / "runs.jsonl").open("a", encoding="utf-8") as target:
        target.write("{}\n")
    with pytest.raises(ValueError, match="artifact"):
        load_labels(run_dir, labels, partition="calibration", required_depth=5)


def test_collect_calls_retrieval_once_per_query_and_reuses_reranker(bundle, tmp_path, monkeypatch):
    run_dir, _ = bundle
    runs = load_runs(run_dir)
    calls = []

    class Searcher:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def search(self, query, **kwargs):
            calls.append(query)
            return copy.deepcopy(next(r["baseline"] for r in runs if r["query"]["query"] == query))

    models = []

    def factory(args):
        model = FakeReranker()
        models.append(model)
        return model

    monkeypatch.setattr(case_eval, "make_searcher", lambda args: Searcher())
    monkeypatch.setattr(case_eval, "make_reranker", factory)
    args = SimpleNamespace(
        queries=run_dir / "queries.json",
        output=tmp_path / "collected",
        case_k=50,
        retriever="hybrid",
        reranker_model=tmp_path / "model",
        drill=True,
    )
    case_eval.collect(args)
    assert len(calls) == 3
    assert len(models) == 1 and len(models[0].calls) == 3
    assert len(load_runs(args.output)) == 3
