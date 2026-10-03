from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from retrieval_baseline import case_eval
from retrieval_baseline.tests.test_case_eval import bundle as _bundle
from retrieval_baseline.tests.test_case_eval import fill_labels
from retrieval_baseline.tests.test_reranker import FakeReranker

bundle = _bundle


def runtime(bundle, tmp_path):
    return SimpleNamespace(
        queries=bundle[0] / "queries.json",
        output=tmp_path / "collected",
        retriever="hybrid",
        case_k=50,
        reranker_model=tmp_path / "model",
        drill=True,
    )


def test_collection_timing_attributes_cold_load_and_scoring(bundle, tmp_path, monkeypatch):
    rows = case_eval.load_runs(bundle[0])
    clock = [0.0]
    monkeypatch.setattr(case_eval.time, "perf_counter", lambda: clock[0])

    class Searcher:
        def __init__(self):
            clock[0] += 7

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def search(self, query, **unused):
            clock[0] += 5
            return copy.deepcopy(next(r["baseline"] for r in rows if r["query"]["query"] == query))

    class Reranker(FakeReranker):
        def score(self, query, documents):
            clock[0] += 3
            return super().score(query, documents)

    def model(args):
        clock[0] += 11
        return Reranker()

    monkeypatch.setattr(case_eval, "make_searcher", lambda args: Searcher())
    monkeypatch.setattr(case_eval, "make_reranker", model)
    args = runtime(bundle, tmp_path)
    manifest = case_eval.collect(args)
    collected = case_eval.load_runs(args.output)
    first, second, third = [r["timing_seconds"] for r in collected]
    assert first == {
        "retrieval_total": 5,
        "reranker_model_load": 11,
        "reranker_scoring": 3,
        "first_reranker_load": True,
        "query_total": 19,
    }
    assert (
        second
        == third
        == {
            "retrieval_total": 5,
            "reranker_model_load": 0,
            "reranker_scoring": 3,
            "first_reranker_load": False,
            "query_total": 8,
        }
    )
    assert manifest["timing_seconds"] == {"searcher_initialization": 7, "batch_total": 42}
    with pytest.raises(ValueError, match="Formal workflow"):
        case_eval.require_formal_run(args.output)


@pytest.mark.parametrize(
    "error_type,expected", [(RuntimeError, "failed"), (KeyboardInterrupt, "interrupted")]
)
def test_partial_collection_is_not_an_empty_query(
    bundle, tmp_path, monkeypatch, error_type, expected
):
    rows = case_eval.load_runs(bundle[0])
    calls = []

    class Searcher:
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def search(self, query, **unused):
            calls.append(query)
            if len(calls) == 2:
                raise error_type("PRIVATE BUSINESS TEXT MUST NOT ENTER STATUS")
            return copy.deepcopy(rows[0]["baseline"])

    monkeypatch.setattr(case_eval, "make_searcher", lambda args: Searcher())
    monkeypatch.setattr(case_eval, "make_reranker", lambda args: FakeReranker())
    args = runtime(bundle, tmp_path)
    with pytest.raises(error_type):
        case_eval.collect(args)
    raw = (args.output / "collection-status.json").read_text()
    status = json.loads(raw)
    assert status["status"] == expected
    assert status["completed_queries"] == 1 and status["empty_query_ids"] == []
    assert status["failed_query_id"] == rows[1]["query"]["query_id"]
    assert "PRIVATE BUSINESS TEXT" not in raw
    assert not (args.output / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        case_eval.load_runs(args.output)


def test_calibration_failure_leaves_evidence_and_allows_unfiltered(bundle, tmp_path):
    run, labels = bundle
    fill_labels(labels, fill_evaluation=True)
    output = tmp_path / "calibration-001"
    status = case_eval.calibrate_attempt(run, labels, output, drill=True, min_results=100)
    assert status["status"] == "failed"
    assert status["modes"]["problem"]["status"] == "insufficient_support"
    assert (output / "threshold-search.json").is_file()
    assert not (output / "policy.json").exists()
    assert case_eval.evaluate(run, labels)["metrics_by_mode"]["problem"]["baseline"]


def test_input_error_attempt_is_recorded_without_a_policy(bundle, tmp_path):
    run, labels = bundle
    status = case_eval.calibrate_attempt(run, labels, tmp_path / "attempt", drill=True)
    assert status["status"] == "input_error"
    assert status["error_code"] == "ValueError"
    assert not (tmp_path / "attempt" / "policy.json").exists()


def test_unfrozen_default_collection_is_rejected_before_model_init(bundle, tmp_path, monkeypatch):
    args = runtime(bundle, tmp_path)
    args.drill = False
    monkeypatch.setattr(case_eval, "make_searcher", lambda args: pytest.fail("must not load model"))
    with pytest.raises(ValueError, match="freeze-manifest"):
        case_eval.collect(args)
    status = json.loads((args.output / "collection-status.json").read_text())
    assert status["status"] == "failed" and status["stage"] == "validation"
    assert not (args.output / "runs.jsonl").exists()
