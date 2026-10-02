from __future__ import annotations

import copy

import pytest

from retrieval_baseline.reranker import (
    POLICY_VERSION,
    policy_context,
    rerank_report,
    select_results,
)
from retrieval_baseline.search import CaseSearcher
from retrieval_baseline.tests import test_search as search_fixtures

search_data = search_fixtures.search_data
address_index = search_fixtures.address_index


class FakeReranker:
    def __init__(self, scores=None):
        self.scores = scores
        self.calls = []
        self.identity = {
            "profile": "test-only",
            "model": {"sha256": "test-model"},
            "max_length": 1024,
            "dtype": "float32",
            "batch_size": 4,
            "device": "cpu",
        }

    def score(self, query, documents):
        self.calls.append((query, documents))
        values = self.scores or {text: float(i) for i, text in enumerate(documents)}
        return [values[text] for text in documents], {"pairs": len(documents)}


def policy_for(report, threshold=0.5, max_top_k=5):
    return {
        "version": POLICY_VERSION,
        "max_top_k": max_top_k,
        "modes": {
            report["mode"]: {
                "threshold": threshold,
                "context": copy.deepcopy(policy_context(report)),
            }
        },
    }


def test_reranker_sees_full_pool_before_top_k_and_only_query_case_content(search_data):
    query = "垃圾清运"
    model = FakeReranker()
    with CaseSearcher(*search_data, reranker=model) as searcher:
        baseline = searcher.search(query, retriever="bm25", top_k=50)
        ranked = searcher.search(query, retriever="bm25", rerank=True, top_k=1)
    assert len(baseline["results"]) > 1
    assert model.calls == [(query, [row["case_content"] for row in baseline["results"]])]
    assert ranked["results"][0]["source_id"] == baseline["results"][-1]["source_id"]
    assert ranked["reranking"]["scored_candidates"] == len(baseline["results"])
    assert ranked["results"][0]["matching"]["reranker"]["retrieval_rank"] > 1
    assert ranked["timing_seconds"]["reranker_scoring"] >= 0


def test_rerank_preserves_address_constraint_and_skips_empty_pool(search_data, address_index):
    model = FakeReranker()
    calls = []

    def factory():
        calls.append(1)
        return model

    with CaseSearcher(
        *search_data, address_index=address_index, reranker_factory=factory
    ) as searcher:
        empty = searcher.search(
            "垃圾清运", mode="combined", retriever="bm25", address="不存在的小区", rerank=True
        )
        assert empty["result_status"] == "no_candidates"
        assert not calls
        ranked = searcher.search(
            "垃圾清运", mode="combined", retriever="bm25", address="幸福小区3号楼", rerank=True
        )
    assert [row["source_id"] for row in ranked["results"]] == ["old-unreferenced"]
    assert calls == [1]
    assert len(model.calls[0][1]) == 1
    assert ranked["results"][0]["matching"]["address"]["broader_match"] is False


def test_filter_allows_fewer_or_zero_and_never_mutates_baseline(search_data):
    with CaseSearcher(*search_data) as searcher:
        baseline = searcher.search("垃圾清运", retriever="bm25", top_k=50)
    original = copy.deepcopy(baseline)
    ranked = rerank_report(baseline, FakeReranker(), top_k=50)
    kept = select_results(ranked, top_k=5, policy=policy_for(ranked, threshold=1.5))
    assert len(kept["results"]) == 1
    assert kept["result_status"] == "fewer_than_requested"
    assert kept["relevance_filter"]["rejected_candidates"] == 2
    assert kept["results"][0]["rank"] == 1
    empty = select_results(ranked, top_k=5, policy=policy_for(ranked, threshold=99))
    assert empty["results"] == []
    assert empty["result_status"] == "below_relevance_threshold"
    assert baseline == original
    assert len(ranked["results"]) == 3


@pytest.mark.parametrize("change", ["model", "mode", "dataset", "budget", "nan"])
def test_policy_profile_or_budget_mismatch_is_rejected(search_data, change):
    with CaseSearcher(*search_data) as searcher:
        baseline = searcher.search("垃圾清运", retriever="bm25", top_k=50)
    ranked = rerank_report(baseline, FakeReranker(), top_k=50)
    policy = policy_for(ranked)
    setting = policy["modes"]["problem"]
    if change == "model":
        setting["context"]["reranker_profile"]["model"] = {"sha256": "other"}
    elif change == "mode":
        policy["modes"] = {}
    elif change == "dataset":
        setting["context"]["dataset"] = "other"
    elif change == "nan":
        setting["threshold"] = float("nan")
    with pytest.raises(ValueError):
        select_results(ranked, top_k=6 if change == "budget" else 5, policy=policy)


def test_nonfinite_model_output_rejected(search_data):
    with CaseSearcher(*search_data) as searcher:
        report = searcher.search("垃圾清运", retriever="bm25", top_k=50)
    model = FakeReranker({r["case_content"]: float("nan") for r in report["results"]})
    with pytest.raises(ValueError, match="non-finite"):
        rerank_report(report, model, top_k=5)


def test_address_only_cannot_apply_problem_reranking(search_data, address_index):
    with CaseSearcher(*search_data, address_index=address_index) as searcher:
        with pytest.raises(ValueError, match="Address-only"):
            searcher.search("幸福小区", mode="address", rerank=True)
        with pytest.raises(ValueError, match="requires reranking"):
            searcher.search("垃圾清运", retriever="bm25", relevance_policy={})
