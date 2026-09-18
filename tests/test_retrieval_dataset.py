import csv
import json
from datetime import date

from run_bge_retrieval import _load_corpus
from run_bm25_retrieval import _write_pyserini_collection

from pipeline.retrieval_baseline import (
    evaluate_run,
    knowledge_frequencies,
    load_qrels,
    load_query_metadata,
    popularity_rankings,
    write_trec_run,
)
from pipeline.retrieval_dataset import (
    SplitWindow,
    build_retrieval_dataset,
    parse_knowledge_quote,
)


COLUMNS = [
    "id",
    "order_id",
    "case_content",
    "case_goal",
    "area_code_city",
    "area_code_area",
    "case_accord_type_one_name",
    "case_accord_type_two_name",
    "case_accord_type_three_name",
    "call_time",
    "knowledge_quote",
    "delete_flag",
]


def knowledge(*items):
    return json.dumps(
        [
            {"type": item_type, "value": value, "label": label}
            for item_type, value, label in items
        ],
        ensure_ascii=False,
    )


def row(
    occurrence_id,
    order_id,
    content,
    goal,
    call_time,
    quote,
    *,
    category3="三级",
    delete_flag="0",
):
    return {
        "id": occurrence_id,
        "order_id": order_id,
        "case_content": content,
        "case_goal": goal,
        "area_code_city": "常州市",
        "area_code_area": "武进区",
        "case_accord_type_one_name": "一级",
        "case_accord_type_two_name": "二级",
        "case_accord_type_three_name": category3,
        "call_time": call_time,
        "knowledge_quote": quote,
        "delete_flag": delete_flag,
    }


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def test_parse_knowledge_quote_deduplicates_by_type_and_value():
    parsed = parse_knowledge_quote(
        knowledge((0, "same", "旧标题"), (2, "same", "另一类型"), (0, "same", "新标题"))
    )

    assert parsed == [
        {"knowledge_id": "0:same", "title": "新标题"},
        {"knowledge_id": "2:same", "title": "另一类型"},
    ]


def test_build_retrieval_dataset_groups_parents_and_prevents_exact_duplicate_leakage(
    tmp_path,
):
    source = tmp_path / "source.tsv"
    output = tmp_path / "retrieval"
    k1_old = knowledge((0, "k1", "知识一旧标题"))
    k1_new = knowledge((0, "k1", "知识一"))
    k2 = knowledge((2, "k2", "知识二"))
    rows = [
        row("1", "parent-train", "训练事实", "训练诉求", "2024-01-10 08:00:00", k1_old),
        row("2", "parent-train", "训练事实", "训练诉求", "2024-01-10 08:00:00", k1_new),
        row("3", "parent-dev", "开发事实", "开发诉求", "2024-02-10", k1_new),
        row("4", "parent-test", "测试事实", "测试诉求", "2024-03-10", k2),
        # Same normalized content+goal in train and test: both are excluded.
        row("5", "leak-train", "重复  正文", "同诉求", "2024-01-12", k1_new),
        row("6", "leak-test", "重复 正文", "同诉求", "2024-03-12", k1_new),
        # Conflicting goal under one parent: excluded and audited.
        row("7", "conflict", "冲突正文", "目标甲", "2024-01-13", k1_new),
        row("8", "conflict", "冲突正文", "目标乙", "2024-01-13", k1_new),
        # Empty citations are unlabeled and do not enter qrels.
        row("9", "unlabeled", "未引用正文", "未引用诉求", "2024-02-13", "[]"),
        row("10", "deleted", "删除正文", "删除诉求", "2024-02-14", k1_new, delete_flag="1"),
    ]
    write_tsv(source, rows)
    splits = (
        SplitWindow("train", date(2024, 1, 1), date(2024, 1, 31)),
        SplitWindow("dev", date(2024, 2, 1), date(2024, 2, 29)),
        SplitWindow("test", date(2024, 3, 1), date(2024, 3, 31)),
    )

    manifest = build_retrieval_dataset(source, output, splits=splits)

    assert manifest["counts"]["source_rows_read"] == 10
    assert manifest["counts"]["parent_tickets"] == 8
    assert manifest["counts"]["knowledge_items"] == 2
    assert manifest["counts"]["eligible_queries"] == {"train": 1, "dev": 1, "test": 1}
    assert manifest["counts"]["cross_split_parent_tickets_excluded"] == 2
    assert manifest["counts"]["parent_conflicts"]["goal_conflict"] == 1

    queries = read_jsonl(output / "queries.jsonl")
    assert [query["_id"] for query in queries] == [
        "order:parent-dev",
        "order:parent-test",
        "order:parent-train",
    ]
    train_query = next(query for query in queries if query["_id"] == "order:parent-train")
    assert train_query["text"] == "[事实] 训练事实\n[诉求] 训练诉求"
    assert train_query["metadata"]["occurrence_count"] == 2

    assert load_qrels(output / "qrels/train.tsv") == {
        "order:parent-train": {"0:k1": 1}
    }
    assert load_qrels(output / "qrels/dev.tsv") == {
        "order:parent-dev": {"0:k1": 1}
    }
    assert load_qrels(output / "qrels/test.tsv") == {
        "order:parent-test": {"2:k2": 1}
    }
    corpus = {record["_id"]: record for record in read_jsonl(output / "corpus.jsonl")}
    assert set(corpus) == {"0:k1", "2:k2"}
    assert corpus["0:k1"]["metadata"]["title_variant_count"] == 2
    assert len(read_jsonl(output / "parent_conflicts.jsonl")) == 1
    assert len(read_jsonl(output / "cross_split_duplicates.jsonl")) == 1


def test_popularity_baseline_and_metrics_use_training_frequencies_only(tmp_path):
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in [
                {"_id": "train-a", "text": "", "metadata": {"category": ["A"]}},
                {"_id": "train-b", "text": "", "metadata": {"category": ["B"]}},
                {"_id": "test-a", "text": "", "metadata": {"category": ["A"]}},
            ]
        ),
        encoding="utf-8",
    )
    train_qrels = {
        "train-a": {"k-category": 1, "k-global": 1},
        "train-b": {"k-global": 1},
    }
    metadata = load_query_metadata(query_path)
    global_run = popularity_rankings(
        train_qrels,
        metadata,
        {"test-a": metadata["test-a"]},
        top_k=2,
        category_aware=False,
    )
    category_run = popularity_rankings(
        train_qrels,
        metadata,
        {"test-a": metadata["test-a"]},
        top_k=2,
        category_aware=True,
    )
    assert [item[0] for item in global_run["test-a"]] == ["k-global", "k-category"]
    assert [item[0] for item in category_run["test-a"]] == ["k-category", "k-global"]

    run_path = tmp_path / "run.trec"
    write_trec_run(run_path, category_run, run_name="category")
    result = evaluate_run(
        {"test-a": {"k-category": 1, "k-new": 1}},
        {"test-a": ["k-category", "k-global"]},
        cutoffs=(1, 2),
        mrr_cutoff=2,
        ndcg_cutoff=2,
        training_frequencies=knowledge_frequencies(train_qrels),
    )
    assert result["metrics"]["hit@1"] == 1.0
    assert result["metrics"]["recall@1"] == 0.5
    assert result["metrics"]["mrr@2"] == 1.0
    assert result["frequency_buckets"]["recall@1"]["new"] == 0.0
    assert result["frequency_buckets"]["recall@1"]["tail_1_4"] == 1.0


def test_dense_and_bm25_adapters_preserve_corpus_ids_and_text(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in [
                {"_id": "0:k1", "title": "标题一", "text": "正文一"},
                {"_id": "2:k2", "title": "标题二", "text": ""},
            ]
        ),
        encoding="utf-8",
    )

    identifiers, texts = _load_corpus(corpus)
    assert identifiers == ["0:k1", "2:k2"]
    assert texts == ["标题一\n正文一", "标题二"]

    collection = tmp_path / "collection"
    assert _write_pyserini_collection(corpus, collection) == 2
    documents = read_jsonl(collection / "docs.jsonl")
    assert documents == [
        {"id": "0:k1", "contents": "标题一\n正文一"},
        {"id": "2:k2", "contents": "标题二"},
    ]


def test_builder_rejects_source_directory_as_output(tmp_path):
    source = tmp_path / "source.tsv"
    write_tsv(source, [])

    try:
        build_retrieval_dataset(source, tmp_path, overwrite=True)
    except ValueError as exc:
        assert "unsafe output directory" in str(exc)
    else:
        raise AssertionError("unsafe output directory was accepted")
