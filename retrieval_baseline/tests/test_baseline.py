from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from retrieval_baseline.common import read_jsonl, tokens
from retrieval_baseline.dataset import Groups, build_dataset, references
from retrieval_baseline.lexical import build_index, evaluate, observed_metrics


def record(sid, text, *, order=None, date="2025-08-01 10:00:00", refs=None, **extra):
    return {
        "id": sid, "order_id": order or sid, "case_content": text,
        "call_time": date, "delete_flag": "0", "order_invalid_type": "",
        "case_accord_type_one_name": "ignored-metadata",
        "knowledge_quote": json.dumps(refs or [], ensure_ascii=False), **extra,
    }


def ref(key, title):
    return {"type": 0, "value": key, "label": title}


def write_source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_transitive_group_link_does_not_leak_through_different_order_ids():
    g = Groups()
    a = g.add("order-a", "text-a", 1)
    b = g.add("order-a", "text-b", 1)
    c = g.add("order-c", "text-b", 2)
    assert g.find(a) == g.find(b) == g.find(c)
    assert g.periods[g.find(a)] == 3


def test_dataset_and_search_exclude_future_titles_and_related_cases(tmp_path):
    source = tmp_path / "source.tsv"
    dataset, index, output = [tmp_path / name for name in ("dataset", "index", "report")]
    rows = [
        record("old-1", "道路积水，希望排水", refs=[ref("drain", "道路排水办理")]),
        record("old-2", "垃圾清运不及时", refs=[ref("rubbish", "垃圾清运")]),
        # Corpus duplicates do not create duplicate retrieval documents.
        record("old-3", "道路积水，希望排水", refs=[ref("drain", "道路排水流程")]),
        # Transitive order/text links span periods and must all be excluded.
        record("bridge-1", "同一历史事项", order="bridge"),
        record("bridge-2", "相同事项的另一正文", order="bridge"),
        record("bridge-3", "相同事项的另一正文", date="2026-01-05 10:00:00"),
        record("dev-1", "路面积水排水咨询", date="2026-01-05 10:00:00",
               refs=[ref("drain", "不能泄漏的未来改名"), ref("new", "未来独有标题")]),
        record("dev-unlabeled", "路边噪音", date="2026-01-06 10:00:00"),
        record("test-1", "垃圾清运办理咨询", date="2026-02-01 10:00:00",
               refs=[ref("rubbish", "另一个未来改名")]),
        record("bad-tab", "损坏\t被吞入的字段"),
        record("deleted", "已删除正文", delete_flag="1"),
        record("missing-time", "没有可用时间", date=""),
    ]
    write_source(source, rows)
    manifest = build_dataset(source, dataset, queries_per_split=20)
    assert manifest["counts"]["records_cross_period"] == 3
    assert manifest["counts"]["records_missing_time"] == 1
    assert manifest["counts"]["corpus_unique_texts"] == 2
    assert manifest["counts"]["catalog_ids"] == 2
    assert manifest["samples"]["dev"]["unlabeled_queries"] == 1
    assert manifest["samples"]["dev"]["unseen_targets"] == 1
    assert all(value == 0 for value in manifest["split_audit"].values())
    catalog = (dataset / "catalog.jsonl").read_text(encoding="utf-8")
    assert "未来" not in catalog
    assert "道路排水流程" in catalog
    with sqlite3.connect(dataset / "dataset.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM records WHERE split='corpus'").fetchone()[0] == 3
    build_index(dataset, index)
    report = evaluate(dataset, index, output)
    for route in ("title_bm25", "case_bm25_vote", "title_case_rrf"):
        metrics = report["metrics"][route]
        assert metrics["observed_recall@10"] == .5  # unseen target stays in the denominator
        assert metrics["known_target_recall@10"] == 1
        assert metrics["unlabeled_queries"] == 1
        assert metrics["unseen_target_count"] == 1
    rankings = read_jsonl(output / "rankings.jsonl")
    assert all(hit["source_id"].startswith("old-") for r in rankings for hit in r["case_hits"])
    assert not (dataset / "report.test.json").exists()
    # File fingerprints prevent mixing edited queries with old qrels.
    with (dataset / "queries.dev.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="artifact changed"):
        evaluate(dataset, index, tmp_path / "tampered")


def test_metrics_have_no_silent_unknown_target_filter_or_empty_negative():
    scores = observed_metrics([{"a", "b"}, {"c"}, set()], [["a"], [], ["a"]], {"a", "b"})
    assert scores["observed_recall@10"] == .25
    assert scores["target_catalog_coverage"] == 2 / 3
    assert scores["catalog_recall_ceiling"] == .5
    assert scores["known_target_recall@10"] == .5
    assert scores["known_target_query_count"] == 1
    assert scores["unlabeled_queries"] == 1
    assert observed_metrics([set()], [[]], set())["observed_recall@10"] is None
    with pytest.raises(ValueError, match="duplicate knowledge"):
        observed_metrics([{"a"}], [["a", "a"]], {"a"})


def test_json_and_unicode_boundaries_are_preserved(tmp_path):
    source, dataset = tmp_path / "source.tsv", tmp_path / "dataset"
    write_source(source, [
        record("one", "道路\u2028积水", refs=[ref("one", "排水")]),
        record("two", "查询\u2029道路排水", date="2026-01-01 00:00:00"),
    ])
    build_dataset(source, dataset)
    assert read_jsonl(dataset / "queries.dev.jsonl")[0]["case_content"] == "查询\u2029道路排水"
    assert references("NULL") == ([], None)
    assert references("broken")[1] == "invalid_reference_json"
    assert references('[{"type":true,"value":1,"label":"名称"}]')[1] == "invalid_reference_item"
    assert tokens("道路积水 [PHONE] ABC") == ["道路", "路积", "积水", "abc"]
    with pytest.raises(FileExistsError):
        build_dataset(source, dataset)
