from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import pytest

from retrieval_baseline.common import VERSION, file_hash, read_jsonl, readonly, write_jsonl
from retrieval_baseline.dataset import build_dataset
from retrieval_baseline.hybrid import evaluate_hybrid, fuse_cases
from retrieval_baseline.lexical import build_index, evaluate, observed_metrics, vote_cases
from retrieval_baseline.tests.test_baseline import record, ref, write_source


def hit(sid: str, rank: int, source_row: int = 1) -> dict:
    return {"source_id": sid, "source_row": source_row, "rank": rank}


def rewrite_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def rewrite_run(directory: Path, rows: list[dict], *, update_hash: bool = True) -> None:
    path = directory / "rankings.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    if update_hash:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if "rankings_sha256" in report:
            report["rankings_sha256"] = file_hash(path)
        rewrite_json(directory / "report.json", report)


def make_artifacts(tmp_path, rows, *, case_k=3, dense_candidates=None):
    source = tmp_path / "source.tsv"
    dataset, index, bm25, dense = [tmp_path / name for name in
                                   ("dataset", "index", "bm25", "dense")]
    write_source(source, rows)
    build_dataset(source, dataset)
    build_index(dataset, index)
    evaluate(dataset, index, bm25, case_k=case_k)
    queries = read_jsonl(dataset / "queries.dev.jsonl")
    qrels = read_jsonl(dataset / "qrels.dev.jsonl")
    dense.mkdir()
    with readonly(index / "index.sqlite3") as db:
        candidates = {r[1]: r for r in db.execute(
            "SELECT docid,source_id,source_row,group_id FROM cases"
        )}
        dense_rows = []
        for query in queries:
            ids = ((dense_candidates or {}).get(query["source_id"])
                   or sorted(candidates)[:case_k])
            ranking, hits, support = vote_cases(db, [candidates[sid] for sid in ids], top_k=10)
            dense_rows.append({
                "source_id": query["source_id"], "rankings": {"dense_case_vote": ranking},
                "case_hits": hits, "supporting_cases": support,
            })
        known = {r[0] for r in db.execute("SELECT knowledge_id FROM titles")}
    write_jsonl(dense / "rankings.jsonl", dense_rows)
    rewrite_json(dense / "report.json", {
        "version": VERSION, "dense_version": "raw-dense-v1", "split": "dev",
        "query_count": len(queries),
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "index_manifest_sha256": "synthetic-dense-index",
        "metadata_index_manifest_sha256": file_hash(index / "manifest.json"),
        "config": {"case_k": case_k, "top_k": 10, "rrf_k": 60, "faiss_threads": 1},
        "rankings_sha256": file_hash(dense / "rankings.jsonl"),
        "metrics": {"dense_case_vote": observed_metrics(
            [set(q["observed_knowledge_ids"]) for q in qrels],
            [r["rankings"]["dense_case_vote"] for r in dense_rows], known,
        )},
    })
    return dataset, index, bm25, dense


@pytest.fixture
def artifacts(tmp_path):
    return make_artifacts(tmp_path, [
        record("old-a", "道路积水希望排水", refs=[ref("drain", "排水办理")]),
        record("old-b", "垃圾清运不及时", refs=[ref("trash", "垃圾清运")]),
        record("old-c", "路灯照明故障", refs=[ref("lights", "照明维修")]),
        record("old-d", "道路破损希望修复", refs=[ref("repair", "道路修复")]),
        record("old-e", "排水管道破损", refs=[ref("drain", "排水咨询")]),
        record("old-f", "临时垃圾存放", refs=[ref("trash", "垃圾转运")]),
        record("dev-a", "道路积水清运垃圾", date="2026-01-02 10:00:00",
               refs=[ref("drain", "不能用于检索的未来标题"), ref("unseen", "目录外知识")]),
        record("dev-unknown", "路灯道路照明", date="2026-01-03 10:00:00"),
        record("test-a", "垃圾清运进度", date="2026-02-03 10:00:00",
               refs=[ref("trash", "未来改名")]),
    ], dense_candidates={"dev-a": ["old-b", "old-c", "old-a"],
                         "dev-unknown": ["old-a", "old-d", "old-c"]})


def replace_candidates(artifacts, *, lexical_ids, dense_ids, case_k):
    dataset, index, bm25, dense = artifacts
    qrels = read_jsonl(dataset / "qrels.dev.jsonl")
    with readonly(index / "index.sqlite3") as db:
        metadata = {r[1]: r for r in db.execute(
            "SELECT docid,source_id,source_row,group_id FROM cases"
        )}
        known = {r[0] for r in db.execute("SELECT knowledge_id FROM titles")}
        for directory, route, candidates in (
            (bm25, "case_bm25_vote", lexical_ids), (dense, "dense_case_vote", dense_ids),
        ):
            rows = read_jsonl(directory / "rankings.jsonl")
            for row in rows:
                ranking, hits, support = vote_cases(
                    db, [metadata[sid] for sid in candidates], top_k=10,
                )
                row["case_hits"] = hits
                row["rankings"][route] = ranking
                row["supporting_cases"] = support
            rewrite_run(directory, rows)
            path = directory / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["config"]["case_k"] = case_k
            report["metrics"][route] = observed_metrics(
                [set(q["observed_knowledge_ids"]) for q in qrels],
                [r["rankings"][route] for r in rows], known,
            )
            rewrite_json(path, report)


def run_hybrid(artifacts, output: Path, **kwargs):
    return evaluate_hybrid(*artifacts, output, **kwargs)


def test_fusion_deduplicates_across_routes_keeps_budget_and_exact_scores():
    bm25 = [hit("shared", 1, 20), hit("lexical", 2, 30), hit("third", 3, 40)]
    dense = [hit("semantic", 1, 50), hit("shared", 2, 20), hit("fourth", 3, 60)]
    fused = fuse_cases(bm25, dense, case_k=3)
    assert len(fused) == 3
    assert fused[0]["source_id"] == "shared"
    assert fused[0]["source_row"] == 20
    assert fused[0]["route_ranks"] == {"case_bm25_vote": 1, "dense_case_vote": 2}
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1]["source_id"] == "semantic"
    assert fused[1]["rrf_score"] == pytest.approx(1 / 61)
    assert [row["rank"] for row in fused] == [1, 2, 3]
    assert len({row["source_id"] for row in fused}) == 3
    assert bm25[0] == hit("shared", 1, 20)  # Input traces remain unchanged.


def test_fusion_ties_are_deterministic_and_route_scores_are_symmetric():
    left, right = [hit("z", 1, 2), hit("a", 2, 3)], [hit("a", 1, 3), hit("z", 2, 2)]
    original = fuse_cases(left, right, case_k=2)
    swapped = fuse_cases(right, left, case_k=2)
    assert [row["source_id"] for row in original] == ["a", "z"]
    assert [row["source_id"] for row in original] == [row["source_id"] for row in swapped]
    assert [row["rrf_score"] for row in original] == [row["rrf_score"] for row in swapped]
    assert fuse_cases([], [], case_k=2) == []


@pytest.mark.parametrize("left,right", [
    ([hit("same", 1), hit("same", 2)], []),
    ([hit("a", 2)], []),
    ([hit("a", 1)], [hit("a", 1, 999)]),
])
def test_fusion_rejects_duplicate_rank_or_inconsistent_source_traces(left, right):
    with pytest.raises(ValueError):
        fuse_cases(left, right, case_k=2)


def test_hybrid_replays_shared_voting_saves_provenance_and_leaves_test_untouched(
    artifacts, tmp_path,
):
    dataset, index, bm25, dense = artifacts
    output = tmp_path / "hybrid"
    report = run_hybrid(artifacts, output)
    rows = read_jsonl(output / "rankings.jsonl")
    queries = read_jsonl(dataset / "queries.dev.jsonl")
    lexical_rows = read_jsonl(bm25 / "rankings.jsonl")
    dense_rows = read_jsonl(dense / "rankings.jsonl")
    assert report["query_count"] == len(rows) == 2
    assert report["split"] == "dev"
    assert report["dataset_manifest_sha256"] == file_hash(dataset / "manifest.json")
    assert [r["source_id"] for r in rows] == [q["source_id"] for q in queries]
    with readonly(index / "index.sqlite3") as db:
        metadata = {r[1]: r for r in db.execute(
            "SELECT docid,source_id,source_row,group_id FROM cases"
        )}
        for row, left, right in zip(rows, lexical_rows, dense_rows, strict=True):
            expected = fuse_cases(left["case_hits"], right["case_hits"], case_k=3)
            assert [h["source_id"] for h in row["case_hits"]] == [
                h["source_id"] for h in expected
            ]
            ranking, _, _ = vote_cases(
                db, [metadata[h["source_id"]] for h in expected], top_k=10,
            )
            assert row["rankings"]["hybrid_case_rrf_vote"] == ranking
            assert len(row["case_hits"]) <= 3
    assert (output / "diagnosis.jsonl").is_file()
    diagnoses = read_jsonl(output / "diagnosis.jsonl")
    assert len(diagnoses) == 1
    assert diagnoses[0]["targets_outside_catalog"] == ["0:unseen"]
    assert "0:unseen" not in diagnoses[0]["lost_in_fusion_cutoff"]
    assert "0:unseen" not in diagnoses[0]["lost_in_voting"]
    assert set(report["metrics"]) == {
        "case_bm25_vote", "dense_case_vote", "hybrid_case_rrf_vote",
    }
    assert report["metrics"]["hybrid_case_rrf_vote"]["unlabeled_queries"] == 1
    assert report["metrics"]["hybrid_case_rrf_vote"]["observed_recall@10"] == .5
    assert set(report["observed_reference_candidate_coverage"]) == {
        "case_bm25_vote", "dense_case_vote", "route_union", "hybrid_case_rrf_vote",
    }
    assert not (output / "query-vectors.npy").exists()
    assert not (dataset / "report.test.json").exists()
    with pytest.raises(FileExistsError):
        run_hybrid(artifacts, output)


@pytest.mark.parametrize("field,value", [
    ("dataset_manifest_sha256", "unrelated-dataset"),
    ("metadata_index_manifest_sha256", "unrelated-index"),
    ("split", "test"),
])
def test_hybrid_rejects_mixed_run_provenance(artifacts, tmp_path, field, value):
    report_path = artifacts[3] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    rewrite_json(report_path, report)
    with pytest.raises(ValueError):
        run_hybrid(artifacts, tmp_path / "rejected")


def test_dense_ranking_hash_is_required_and_validated(artifacts, tmp_path):
    dense = artifacts[3]
    report_path = dense / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("rankings_sha256")
    rewrite_json(report_path, report)
    with pytest.raises(ValueError):
        run_hybrid(artifacts, tmp_path / "missing-hash")
    report["rankings_sha256"] = "broken-hash"
    rewrite_json(report_path, report)
    with pytest.raises(ValueError):
        run_hybrid(artifacts, tmp_path / "changed-rankings")


def test_hybrid_rejects_query_order_corruption_even_with_updated_file_hash(
    artifacts, tmp_path,
):
    dense = artifacts[3]
    rows = read_jsonl(dense / "rankings.jsonl")
    rewrite_run(dense, list(reversed(rows)))
    with pytest.raises(ValueError):
        run_hybrid(artifacts, tmp_path / "rejected")


@pytest.mark.parametrize("alteration", ["future", "wrong-source-row", "bad-rank"])
def test_all_candidates_are_checked_including_unlabeled_queries(
    artifacts, tmp_path, alteration,
):
    dense = artifacts[3]
    rows = read_jsonl(dense / "rankings.jsonl")
    row = next(r for r in rows if r["source_id"] == "dev-unknown")
    candidate = row["case_hits"][0]
    if alteration == "future":
        candidate["source_id"] = "test-a"
    elif alteration == "wrong-source-row":
        candidate["source_row"] += 10000
    else:
        candidate["rank"] = 5
    rewrite_run(dense, rows)
    with pytest.raises(ValueError):
        run_hybrid(artifacts, tmp_path / "rejected")


def test_hybrid_rejects_predictions_not_reproduced_by_shared_case_voting(
    artifacts, tmp_path,
):
    dense = artifacts[3]
    rows = read_jsonl(dense / "rankings.jsonl")
    row = next(r for r in rows if r["source_id"] == "dev-unknown")
    row["rankings"]["dense_case_vote"] = ["0:not-retrieved"]
    rewrite_run(dense, rows)
    with pytest.raises(ValueError):
        run_hybrid(artifacts, tmp_path / "rejected")


def test_fusion_is_label_independent_even_when_evaluation_targets_change(artifacts, tmp_path):
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    run_hybrid(artifacts, before_dir)
    before = read_jsonl(before_dir / "rankings.jsonl")
    dataset, index, bm25, dense = artifacts
    qrels_path = dataset / "qrels.dev.jsonl"
    qrels = read_jsonl(qrels_path)
    qrels[0]["observed_knowledge_ids"] = ["0:lights"]
    qrels[1]["observed_knowledge_ids"] = ["0:trash"]
    qrels_path.write_text("".join(json.dumps(q) + "\n" for q in qrels), encoding="utf-8")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["qrels.dev.jsonl"] = file_hash(qrels_path)
    rewrite_json(manifest_path, manifest)
    dataset_hash = file_hash(manifest_path)
    index_manifest_path = index / "manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["dataset_manifest_sha256"] = dataset_hash
    rewrite_json(index_manifest_path, index_manifest)
    index_hash = file_hash(index_manifest_path)
    for directory, key in ((bm25, "index_manifest_sha256"),
                           (dense, "metadata_index_manifest_sha256")):
        path = directory / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["dataset_manifest_sha256"] = dataset_hash
        report[key] = index_hash
        rewrite_json(path, report)
    run_hybrid(artifacts, after_dir)
    after = read_jsonl(after_dir / "rankings.jsonl")
    for original, changed in zip(before, after, strict=True):
        assert original["case_hits"] == changed["case_hits"]
        assert original["rankings"] == changed["rankings"]


@pytest.mark.parametrize("changed_artifact", ["dataset", "index"])
def test_hybrid_rejects_edited_dataset_and_lexical_index(
    artifacts, tmp_path, changed_artifact,
):
    dataset, index, _, _ = artifacts
    if changed_artifact == "dataset":
        with (dataset / "queries.dev.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
    else:
        with sqlite3.connect(index / "index.sqlite3") as db:
            db.execute("UPDATE cases SET source_row=source_row+1 WHERE source_id='old-a'")
    with pytest.raises(ValueError):
        run_hybrid(artifacts, tmp_path / "changed-artifact")


def test_hybrid_rejects_different_candidate_budget(artifacts, tmp_path):
    path = artifacts[3] / "report.json"
    report = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    report["config"]["case_k"] += 1
    rewrite_json(path, report)
    with pytest.raises(ValueError):
        run_hybrid(artifacts, tmp_path / "rejected")


def assert_target_partition(report, diagnoses):
    stages = ["targets_outside_catalog", "missed_in_union", "lost_in_fusion_cutoff",
              "lost_in_voting", "top10_target_hits"]
    for row in diagnoses:
        stage_ids = [sid for stage in stages for sid in row[stage]]
        assert len(stage_ids) == len(set(stage_ids))
        assert set(stage_ids) == set(row["observed_targets"])
    counts = report["target_stage_counts"]
    assert counts["observed_targets"] == sum(counts[stage] for stage in stages)
    for stage in stages + ["observed_targets"]:
        assert counts[stage] == sum(len(row[stage]) for row in diagnoses)


def test_union_coverage_does_not_hide_final_budget_cutoff_loss(artifacts, tmp_path):
    replace_candidates(artifacts, lexical_ids=["old-e"], dense_ids=["old-c"], case_k=1)
    output = tmp_path / "cutoff"
    report = run_hybrid(artifacts, output)
    diagnosis = read_jsonl(output / "diagnosis.jsonl")
    rows = read_jsonl(output / "rankings.jsonl")
    assert all(row["case_hits"][0]["source_id"] == "old-c" for row in rows)
    assert diagnosis[0]["union_candidate_target_hits"] == ["0:drain"]
    assert diagnosis[0]["fused_candidate_target_hits"] == []
    assert diagnosis[0]["lost_in_fusion_cutoff"] == ["0:drain"]
    assert diagnosis[0]["lost_in_voting"] == []
    coverage = report["observed_reference_candidate_coverage"]
    assert coverage["route_union"] == .5
    assert coverage["hybrid_case_rrf_vote"] == 0
    assert report["metrics"]["hybrid_case_rrf_vote"]["observed_recall@10"] == 0
    assert_target_partition(report, diagnosis)


def test_correlated_group_is_capped_after_hybrid_fusion(tmp_path):
    artifacts = make_artifacts(tmp_path, [
        record("old-a", "道路第一处积水", order="same-issue", refs=[
            ref("z-repeated", "反复引用"), ref("filler-a", "第一项"),
        ]),
        record("old-b", "道路第二处积水", order="same-issue", refs=[
            ref("z-repeated", "反复引用"), ref("filler-b", "第二项"),
        ]),
        record("old-c", "另一处独立的积水", refs=[ref("a-independent", "独立引用")]),
        record("dev-a", "道路积水咨询", date="2026-01-02 10:00:00",
               refs=[ref("a-independent", "目标")]),
    ])
    replace_candidates(artifacts, lexical_ids=["old-a", "old-b", "old-c"],
                       dense_ids=["old-a", "old-b", "old-c"], case_k=3)
    output = tmp_path / "group-cap"
    run_hybrid(artifacts, output)
    row = read_jsonl(output / "rankings.jsonl")[0]
    # Without the shared group cap the repeated reference would win: 1/122+1/124 > 1/63.
    assert 1 / 122 + 1 / 124 > 1 / 63
    assert row["rankings"]["hybrid_case_rrf_vote"][0] == "0:a-independent"
    assert row["supporting_cases"]["0:z-repeated"] == ["old-a", "old-b"]


def test_reference_present_in_fused_cases_can_still_be_lost_in_top10_voting(tmp_path):
    many_refs = [ref(f"knowledge-{i:02}", f"知识项{i}") for i in range(11)]
    artifacts = make_artifacts(tmp_path, [
        record("old-a", "道路积水历史事项", refs=many_refs),
        record("old-b", "不同的垃圾清运事项", refs=[ref("not-retrieved", "未召回项")]),
        record("dev-a", "道路积水如何处理", date="2026-01-02 10:00:00", refs=[
            ref("knowledge-00", "命中项"), ref("knowledge-10", "投票丢失项"),
            ref("not-retrieved", "候选缺失项"), ref("unseen", "目录缺失项"),
        ]),
    ], case_k=1)
    replace_candidates(artifacts, lexical_ids=["old-a"], dense_ids=["old-a"], case_k=1)
    output = tmp_path / "vote-loss"
    report = run_hybrid(artifacts, output)
    diagnosis = read_jsonl(output / "diagnosis.jsonl")
    assert diagnosis[0]["targets_outside_catalog"] == ["0:unseen"]
    assert diagnosis[0]["missed_in_union"] == ["0:not-retrieved"]
    assert diagnosis[0]["lost_in_fusion_cutoff"] == []
    assert diagnosis[0]["lost_in_voting"] == ["0:knowledge-10"]
    assert diagnosis[0]["top10_target_hits"] == ["0:knowledge-00"]
    assert report["observed_reference_candidate_coverage"]["hybrid_case_rrf_vote"] == .5
    assert report["metrics"]["hybrid_case_rrf_vote"]["observed_recall@10"] == .25
    assert_target_partition(report, diagnosis)
