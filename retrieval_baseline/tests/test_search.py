from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrieval_baseline.dataset import build_dataset
from retrieval_baseline.lexical import build_index
from retrieval_baseline.tests.test_baseline import record, ref, write_source


@pytest.fixture
def search_data(tmp_path):
    source, dataset, lexical = [tmp_path / n for n in ("source.tsv", "dataset", "lexical")]
    rows = [
        record("old-drain", "道路排水故障", refs=[ref("drain", "只存在于知识标题的医保办理")]),
        record("old-unreferenced", "常州市武进区幸福小区3号楼，垃圾清运不及时，楼道无人打扫。"),
        record("old-other-building", "常州市武进区幸福小区4号楼，垃圾清运不及时。"),
        record("old-other-site", "常州市武进区幸福花园3号楼，垃圾清运不及时。"),
        record("old-school", "办理入学手续"),
        record("future-dev", "未来投诉：幸福小区3号楼垃圾清运问题", date="2026-01-03 00:00:00"),
        record("future-test", "未来测试：幸福小区3号楼垃圾清运问题", date="2026-02-03 00:00:00"),
    ]
    write_source(source, rows)
    build_dataset(source, dataset)
    build_index(dataset, lexical)
    return dataset, lexical


@pytest.fixture
def address_index(search_data, tmp_path):
    from retrieval_baseline.address import build_address_index

    output = tmp_path / "address"
    build_address_index(search_data[0], output)
    return output


def dense_fixture(search_data, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("faiss")
    pytest.importorskip("filelock")
    from retrieval_baseline.dense import build_dense
    from retrieval_baseline.tests.test_dense import TestEncoder

    output, encoder = tmp_path / "dense", TestEncoder()
    build_dense(search_data[0], output, encoder, shard_size=2)
    encoder.seen.clear()
    return output, encoder


def assert_case_results(response, expected_mode, expected_retriever):
    assert response["version"] == "case-search-v1"
    assert response["mode"] == expected_mode
    assert response["retriever"] == expected_retriever
    results = response["results"]
    assert len({r["source_id"] for r in results}) == len(results)
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    for result in results:
        assert isinstance(result["source_row"], int)
        assert result["case_content"]
        assert result["call_time"].startswith("2025-")
        assert "routes" in result["matching"]
        if expected_mode in {"address", "combined"}:
            assert "address" in result["matching"]
    return results


def test_new_query_returns_original_cases_without_knowledge_labels(search_data, monkeypatch):
    from retrieval_baseline.search import CaseSearcher

    real_open, real_read_text = Path.open, Path.read_text

    def reject_labels(path, *args, **kwargs):
        assert not path.name.startswith(("qrels.", "queries.")), "online retrieval read eval data"
        return real_open(path, *args, **kwargs)

    def reject_label_text(path, *args, **kwargs):
        assert not path.name.startswith(("qrels.", "queries.")), "online retrieval read eval data"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_labels)
    monkeypatch.setattr(Path, "read_text", reject_label_text)
    with CaseSearcher(*search_data) as searcher:
        response = searcher.search("垃圾清运无人打扫的新投诉", retriever="bm25")
    results = assert_case_results(response, "problem", "bm25")
    ids = {r["source_id"] for r in results}
    assert "old-unreferenced" in ids
    assert not any(sid.startswith("future-") for sid in ids)
    found = next(r for r in results if r["source_id"] == "old-unreferenced")
    assert found["case_content"] == "常州市武进区幸福小区3号楼，垃圾清运不及时，楼道无人打扫。"
    assert found["matching"]["routes"]["bm25"]["rank"] >= 1


def test_bm25_has_no_arbitrary_fallback_or_knowledge_title_route(search_data):
    from retrieval_baseline.search import CaseSearcher

    with CaseSearcher(*search_data) as searcher:
        assert searcher.search("unmatchedwordxyz", retriever="bm25")["results"] == []
        assert searcher.search("医保", retriever="bm25")["results"] == []
        for invalid in ("", "   ", "\n\t"):
            with pytest.raises(ValueError):
                searcher.search(invalid, retriever="bm25")


def test_metadata_validation_uses_historical_membership_index(search_data):
    from retrieval_baseline.search import CaseSearcher

    with CaseSearcher(*search_data) as searcher:
        statements = []
        searcher.source.set_trace_callback(statements.append)
        assert searcher.search("垃圾清运", retriever="bm25")["results"]
        searcher.source.set_trace_callback(None)
        queries = [statement for statement in statements if "JOIN corpus c" in statement]
        assert queries
        for query in queries:
            plan = searcher.source.execute("EXPLAIN QUERY PLAN " + query).fetchall()
            assert any("SEARCH c" in row[3] and "corpus_rid" in row[3] for row in plan)


@pytest.mark.parametrize("configuration", [
    {"top_k": 0}, {"case_k": 0}, {"max_terms": 0}, {"top_k": 3, "case_k": 2},
    {"mode": "unsupported"}, {"retriever": "unsupported"},
])
def test_invalid_search_configuration_is_rejected(search_data, configuration):
    from retrieval_baseline.search import CaseSearcher

    with CaseSearcher(*search_data) as searcher, pytest.raises(ValueError):
        searcher.search("垃圾", **{"retriever": "bm25", **configuration})


@pytest.mark.parametrize("artifact", ["dataset", "lexical"])
def test_changed_index_or_dataset_is_rejected(search_data, artifact):
    from retrieval_baseline.search import CaseSearcher

    directory = search_data[0 if artifact == "dataset" else 1]
    path = directory / ("dataset.sqlite3" if artifact == "dataset" else "index.sqlite3")
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        with CaseSearcher(*search_data):
            pass


def test_address_mode_never_encodes_and_keeps_building_constraint(search_data, address_index):
    from retrieval_baseline.search import CaseSearcher

    class NeverEncode:
        def encode(self, texts):
            raise AssertionError("address search must not load or query a semantic model")

    with CaseSearcher(*search_data, address_index=address_index, encoder=NeverEncode()) as searcher:
        response = searcher.search("幸福小区3号楼", mode="address")
        unknown = searcher.search("不存在的小区9号楼", mode="address")
    results = assert_case_results(response, "address", "address_surface")
    assert [r["source_id"] for r in results] == ["old-unreferenced"]
    assert results[0]["matching"]["address"]
    assert unknown["results"] == []


def test_combined_explicit_and_parsed_address_are_hard_constraints(search_data, address_index):
    from retrieval_baseline.search import CaseSearcher

    with CaseSearcher(*search_data, address_index=address_index) as searcher:
        explicit = searcher.search("垃圾清运", mode="combined", retriever="bm25",
                                   address="幸福小区3号楼")
        parsed = searcher.search("幸福小区3号楼，垃圾清运不及时。", mode="combined",
                                 retriever="bm25")
        unknown = searcher.search("垃圾清运", mode="combined", retriever="bm25",
                                  address="不存在的小区9号楼")
        with pytest.raises(ValueError, match="address"):
            searcher.search("垃圾清运", mode="combined", retriever="bm25")
    for response in (explicit, parsed):
        assert [r["source_id"] for r in response["results"]] == ["old-unreferenced"]
        assert response["results"][0]["matching"]["address"]
    assert unknown["results"] == []


def test_address_relaxation_is_explicit_and_never_merges_different_sites(
    search_data, address_index,
):
    from retrieval_baseline.search import CaseSearcher

    with CaseSearcher(*search_data, address_index=address_index) as searcher:
        strict = searcher.search("幸福小区99号楼", mode="address")
        relaxed = searcher.search("幸福小区99号楼", mode="address", allow_broader=True)
        exact = searcher.search("幸福小区3号楼", mode="address", allow_broader=True)
    assert strict["results"] == []
    assert {r["source_id"] for r in relaxed["results"]} == {
        "old-unreferenced", "old-other-building",
    }
    assert all(r["matching"]["address"]["broader_match"] for r in relaxed["results"])
    assert [r["source_id"] for r in exact["results"]] == ["old-unreferenced"]


def test_combined_no_address_matches_never_calls_encoder_factory(
    search_data, address_index, tmp_path,
):
    from retrieval_baseline.search import CaseSearcher

    calls = []

    def unexpected_factory():
        calls.append(1)
        raise AssertionError("No address candidates means no model startup")

    with CaseSearcher(*search_data, dense_index=tmp_path / "not-loaded-dense",
                      encoder_factory=unexpected_factory, address_index=address_index) as searcher:
        response = searcher.search("垃圾清运", mode="combined", retriever="hybrid",
                                   address="不存在的小区9号楼")
    assert response["results"] == []
    assert response["address_candidate_count"] == 0
    assert response["dense_index_manifest_sha256"] is None
    assert response["address_index_manifest_sha256"]
    assert calls == []


@pytest.mark.parametrize("retriever", ["bm25", "dense", "hybrid"])
def test_combined_searches_within_address_pool_before_case_cutoff(tmp_path, retriever):
    from retrieval_baseline.address import build_address_index
    from retrieval_baseline.search import CaseSearcher

    source, dataset, lexical, address = [tmp_path / n for n in (
        "source.tsv", "dataset", "lexical", "address",
    )]
    rows = [record(f"a-global-{i}", f"垃圾清运垃圾清运垃圾清运第{i}处",
                   refs=[ref("trash", "垃圾清运")]) for i in range(8)]
    rows.append(record("z-local", "幸福小区3号楼，保洁长期不到位，需要清运垃圾。"))
    write_source(source, rows)
    build_dataset(source, dataset)
    build_index(dataset, lexical)
    build_address_index(dataset, address)
    kwargs = {"address_index": address}
    if retriever != "bm25":
        dense, encoder = dense_fixture((dataset, lexical), tmp_path)
        kwargs.update(dense_index=dense, encoder=encoder)
    with CaseSearcher(dataset, lexical, **kwargs) as searcher:
        global_response = searcher.search("垃圾清运", retriever=retriever, top_k=1, case_k=1)
        local_response = searcher.search("垃圾清运", mode="combined", retriever=retriever,
                                         address="幸福小区3号楼", top_k=1, case_k=1)
    assert global_response["results"][0]["source_id"].startswith("a-global-")
    assert [r["source_id"] for r in local_response["results"]] == ["z-local"]


def test_dense_encodes_only_raw_query_and_has_no_future_candidates(search_data, tmp_path):
    from retrieval_baseline.search import CaseSearcher

    dense, encoder = dense_fixture(search_data, tmp_path)
    query = "新投诉\u2028垃圾无人打扫\u2029需要处理"
    with CaseSearcher(*search_data, dense_index=dense, encoder=encoder, threads=1) as searcher:
        response = searcher.search(query, retriever="dense")
    assert encoder.seen == [query]
    results = assert_case_results(response, "problem", "dense")
    assert "old-unreferenced" in {r["source_id"] for r in results}
    assert all(not r["source_id"].startswith("future-") for r in results)


def test_hybrid_uses_case_rrf_without_duplicate_results(search_data, tmp_path):
    from retrieval_baseline.search import CaseSearcher

    dense, encoder = dense_fixture(search_data, tmp_path)
    with CaseSearcher(*search_data, dense_index=dense, encoder=encoder, threads=1) as searcher:
        bm25 = searcher.search("垃圾清运", retriever="bm25", top_k=5, case_k=5)["results"]
        semantic = searcher.search("垃圾清运", retriever="dense", top_k=5, case_k=5)["results"]
        response = searcher.search("垃圾清运", retriever="hybrid", top_k=5, case_k=5)
    scores = {}
    for ranking in (bm25, semantic):
        for rank, row in enumerate(ranking, 1):
            sid = row["source_id"]
            scores[sid] = scores.get(sid, 0) + 1 / (60 + rank)
    expected = sorted(scores, key=lambda sid: (-scores[sid], sid))[:5]
    results = assert_case_results(response, "problem", "hybrid")
    assert [r["source_id"] for r in results] == expected
    assert len(results) <= 5
    shared = set(r["source_id"] for r in bm25) & set(r["source_id"] for r in semantic)
    assert shared
    for row in results:
        if row["source_id"] in shared:
            assert set(row["matching"]["routes"]) == {"bm25", "dense"}
        assert row["matching"]["rrf_score"] == pytest.approx(scores[row["source_id"]])


def test_dense_identity_or_content_corruption_is_rejected(search_data, tmp_path):
    from retrieval_baseline.search import CaseSearcher

    dense, encoder = dense_fixture(search_data, tmp_path)
    encoder.identity["max_length"] = 32
    with pytest.raises(ValueError):
        with CaseSearcher(*search_data, dense_index=dense, encoder=encoder) as searcher:
            searcher.search("垃圾", retriever="dense")
    encoder.identity["max_length"] = 8192
    with (dense / "index.faiss").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        with CaseSearcher(*search_data, dense_index=dense, encoder=encoder) as searcher:
            searcher.search("垃圾", retriever="dense")


def cli_arguments(search_data):
    return ["case-search", "--dataset", str(search_data[0]), "--index", str(search_data[1]),
            "--retriever", "bm25"]


def never_load_model(monkeypatch):
    class NeverLoad:
        def __init__(self, *args, **kwargs):
            raise AssertionError("BM25/address CLI must not instantiate an embedding model")

    monkeypatch.setitem(sys.modules, "retrieval_baseline.encoder",
                        SimpleNamespace(BGEEncoder=NeverLoad))


def test_cli_query_file_preserves_unicode_and_private_output(
    search_data, tmp_path, monkeypatch, capsys,
):
    from retrieval_baseline.search import main

    query = "垃圾清运\u2028private-query-marker\u2029楼道无人打扫"
    source, output = tmp_path / "query.txt", tmp_path / "private" / "result.json"
    source.write_text(query, encoding="utf-8-sig")
    never_load_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", cli_arguments(search_data) + [
        "--query-file", str(source), "--output", str(output),
    ])
    main()
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert set(summary) == {"output", "result_count"}
    assert summary["output"] == str(output)
    assert summary["result_count"] >= 1
    assert "private-query-marker" not in captured.out
    assert "case_content" not in captured.out
    assert captured.err == ""
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["query"] == query
    assert_case_results(report, "problem", "bm25")
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_bytes() == original


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions are not Windows ACLs")
def test_cli_output_stays_private_with_permissive_umask(
    search_data, tmp_path, monkeypatch, capsys,
):
    from retrieval_baseline.search import main

    output = tmp_path / "private-permissions" / "result.json"
    never_load_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", cli_arguments(search_data) + [
        "--query", "垃圾清运", "--output", str(output),
    ])
    previous_umask = os.umask(0)
    try:
        main()
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert json.loads(capsys.readouterr().out)["result_count"] >= 1


def test_cli_interactive_reuses_searcher_and_emits_one_json_per_success(
    search_data, monkeypatch, capsys,
):
    from retrieval_baseline import search

    real_searcher, constructions = search.CaseSearcher, []

    def counting_searcher(*args, **kwargs):
        constructions.append(1)
        return real_searcher(*args, **kwargs)

    monkeypatch.setattr(search, "CaseSearcher", counting_searcher)
    never_load_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", cli_arguments(search_data) + ["--interactive"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("垃圾清运\n\n道路排水故障\n"))
    search.main()
    captured = capsys.readouterr()
    reports = [json.loads(line) for line in captured.out.splitlines()]
    assert len(reports) == 2
    assert [r["query"] for r in reports] == ["垃圾清运", "道路排水故障"]
    assert len(constructions) == 1
    assert captured.err == ""


def test_cli_interactive_error_does_not_echo_failed_private_query(
    search_data, address_index, monkeypatch, capsys,
):
    from retrieval_baseline.search import main

    never_load_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", cli_arguments(search_data) + [
        "--interactive", "--mode", "combined", "--address-index", str(address_index),
    ])
    secret = "private-query-marker垃圾清运"
    valid = "幸福小区3号楼，垃圾清运不及时。"
    monkeypatch.setattr(sys, "stdin", io.StringIO(secret + "\n" + valid + "\n"))
    main()
    captured = capsys.readouterr()
    reports = [json.loads(line) for line in captured.out.splitlines()]
    assert len(reports) == 1
    assert reports[0]["query"] == valid
    errors = [json.loads(line) for line in captured.err.splitlines()]
    assert len(errors) == 1
    assert "address" in errors[0]["error"]
    assert secret not in captured.out + captured.err
