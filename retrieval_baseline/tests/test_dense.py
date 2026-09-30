from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")
pytest.importorskip("filelock")

from filelock import FileLock, Timeout  # noqa: E402

from retrieval_baseline.compare import compare_runs, paired_delta  # noqa: E402
from retrieval_baseline.dataset import build_dataset  # noqa: E402
from retrieval_baseline.dense import build_dense, evaluate_dense  # noqa: E402
from retrieval_baseline.encoder import (  # noqa: E402
    BGEEncoder,
    model_fingerprint,
    normalized_vectors,
)
from retrieval_baseline.lexical import build_index, evaluate  # noqa: E402
from retrieval_baseline.tests.test_baseline import record, ref, write_source  # noqa: E402


class TestEncoder:
    """Synthetic vectors exercise plumbing, not real BGE-M3 retrieval quality."""

    __test__ = False

    def __init__(self, fail_at=None):
        self.identity = {"profile": "test-only", "dimension": 3, "max_length": 8192}
        self.seen = []
        self.fail_at = fail_at

    def encode(self, texts):
        if self.fail_at is not None and len(self.seen) >= self.fail_at:
            raise RuntimeError("simulated interruption")
        self.seen.extend(texts)
        vectors = []
        for text in texts:
            if any(term in text for term in ("排水", "渗水")):
                vectors.append([1, 0, 0])
            elif "垃圾" in text:
                vectors.append([0, 1, 0])
            else:
                vectors.append([0, 0, 1])
        return np.array(vectors, dtype=np.float32), {"records": len(texts), "truncated_records": 0}


@pytest.fixture
def data(tmp_path):
    source, dataset = tmp_path / "source.tsv", tmp_path / "dataset"
    write_source(source, [
        record("old-a", "道路排水故障", refs=[ref("drain", "排水办理")]),
        record("old-b", "垃圾清运不及时", refs=[ref("trash", "垃圾清理")]),
        record("old-c", "办理入学手续", refs=[ref("school", "入学咨询")]),
        record("dev-a", "房顶渗水问题", date="2026-01-02 00:00:00",
               refs=[ref("drain", "不能输入模型的未来标题"), ref("unseen", "未来目录外知识")]),
        record("dev-b", "垃圾清运咨询", date="2026-01-03 00:00:00"),
        record("test-a", "道路维修问题", date="2026-02-03 00:00:00"),
    ])
    build_dataset(source, dataset)
    return dataset


def test_resume_reuses_only_committed_vectors_and_refuses_changed_identity(data, tmp_path):
    output = tmp_path / "dense"
    encoder = TestEncoder(fail_at=1)
    with pytest.raises(RuntimeError, match="interruption"):
        build_dense(data, output, encoder, shard_size=1)
    state = json.loads((output / "checkpoint.json").read_text())
    assert len(state["shards"]) == 1
    restored = TestEncoder()
    manifest = build_dense(data, output, restored, shard_size=1, resume=True)
    assert len(restored.seen) == 2
    assert encoder.seen + restored.seen == ["道路排水故障", "垃圾清运不及时", "办理入学手续"]
    assert manifest["indexed_cases"] == 3
    assert manifest["encoding_stats"]["records"] == 3
    build_dense(data, output, restored, shard_size=1, resume=True)
    assert len(restored.seen) == 2
    changed = TestEncoder()
    changed.identity["max_length"] = 512
    with pytest.raises(ValueError, match="changed"):
        build_dense(data, output, changed, shard_size=1, resume=True)
    with pytest.raises(FileExistsError):
        build_dense(data, output, TestEncoder(), shard_size=1)
    shard = output / state["shards"][0]["vectors"]
    shard.write_bytes(b"broken")
    with pytest.raises(ValueError, match="shard changed"):
        build_dense(data, output, TestEncoder(), shard_size=1, resume=True)


def test_exact_search_shared_voting_comparison_and_test_isolation(data, tmp_path):
    dense, lexical = tmp_path / "dense", tmp_path / "lexical"
    bm25, dense_report, comparison = [tmp_path / n for n in ("bm25", "dense-result", "compare")]
    encoder = TestEncoder()
    build_dense(data, dense, encoder, shard_size=2)
    build_index(data, lexical)
    evaluate(data, lexical, bm25, case_k=1)
    encoder.seen.clear()
    report = evaluate_dense(data, dense, lexical, dense_report, encoder, case_k=1, threads=1)
    assert set(encoder.seen) == {"房顶渗水问题", "垃圾清运咨询"}
    assert not any("未来" in text or "道路维修" in text for text in encoder.seen)
    metrics = report["metrics"]["dense_case_vote"]
    assert metrics["observed_recall@10"] == .5
    assert metrics["known_target_recall@10"] == 1
    assert metrics["unlabeled_queries"] == 1
    result = compare_runs(data, bm25, dense_report, comparison)
    assert result["dense_minus_bm25"]["recall@10"]["delta"] == .5
    assert result["observed_reference_candidate_coverage"]["dense_case_vote"] == .5
    diagnoses = [json.loads(line) for line in (comparison / "diagnosis.jsonl").open()]
    assert diagnoses[0]["targets_outside_catalog"] == ["0:unseen"]
    # A changed candidate budget must not be called the same experiment.
    path = bm25 / "report.json"
    altered = json.loads(path.read_text())
    altered["config"]["case_k"] = 2
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="candidate budget"):
        compare_runs(data, bm25, dense_report, tmp_path / "bad-comparison")
    with (dense / "index.faiss").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_dense(data, dense, lexical, tmp_path / "bad-index", encoder)


def test_exclusive_build_lock_and_normalization_guards(data, tmp_path):
    output = tmp_path / "dense"
    output.mkdir()
    with FileLock(str(output / ".build.lock")):
        with pytest.raises(Timeout):
            build_dense(data, output, TestEncoder())
    for invalid in (np.zeros((1, 3)), np.array([[np.nan, 0, 1]]), np.ones((1, 2))):
        with pytest.raises(ValueError):
            normalized_vectors(invalid, 1, 3)
    assert np.allclose(normalized_vectors([[3, 4, 0]], 1, 3), [[.6, .8, 0]])


def test_model_fingerprint_covers_weight_content_and_paired_statistics(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{}')
    (model / "model.safetensors").write_bytes(b"weight-a")
    original = copy.deepcopy(model_fingerprint(model))
    (model / "model.safetensors").write_bytes(b"weight-b")
    assert model_fingerprint(model)["sha256"] != original["sha256"]
    assert paired_delta([.5, .5])["ci95"] == [.5, .5]
    assert paired_delta([0, 0])["delta"] == 0
    assert paired_delta([])["delta"] is None


def test_uncommitted_shards_are_recovered_after_checkpoint_write_failure(
    data, tmp_path, monkeypatch,
):
    from retrieval_baseline import dense

    real_write = dense.atomic_json

    def interrupted_write(path, value):
        if path.name == "checkpoint.json":
            raise OSError("simulated interrupted commit")
        return real_write(path, value)

    output = tmp_path / "interrupted"
    monkeypatch.setattr(dense, "atomic_json", interrupted_write)
    with pytest.raises(OSError, match="interrupted commit"):
        build_dense(data, output, TestEncoder(), shard_size=1)
    assert (output / "shard-000000.npy").is_file()
    assert not (output / "checkpoint.json").exists()
    monkeypatch.setattr(dense, "atomic_json", real_write)
    result = build_dense(data, output, TestEncoder(), shard_size=1, resume=True)
    assert result["indexed_cases"] == 3
    assert result["encoding_stats"]["records"] == 3


def test_encoder_offline_settings_truncation_audit_and_oom_recovery(tmp_path, monkeypatch):
    from retrieval_baseline import encoder as module

    model = tmp_path / "model"
    (model / "1_Pooling").mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"xlm-roberta"}')
    (model / "modules.json").write_text('[]')
    (model / "1_Pooling/config.json").write_text('{"pooling_mode_cls_token":true}')
    calls = []

    class OOM(Exception):
        pass

    class Transformer:
        def __init__(self, path, **kwargs):
            assert kwargs["local_files_only"] is True
            assert kwargs["trust_remote_code"] is False

        def get_sentence_embedding_dimension(self):
            return 1024

        def tokenizer(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            return {"length": [100, 20]}

        def encode(self, texts, **kwargs):
            assert kwargs["prompt"] == ""
            calls.append(kwargs["batch_size"])
            if kwargs["batch_size"] > 1:
                raise OOM()
            return np.ones((len(texts), 1024), dtype=np.float16)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float16="float16", cuda=SimpleNamespace(is_available=lambda: True,
                                               OutOfMemoryError=OOM, empty_cache=lambda: None),
    ))
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=Transformer))
    monkeypatch.setattr(module.importlib.metadata, "version", lambda p: "test")
    engine = BGEEncoder(model, max_length=32, batch_size=8, device="cuda", dtype="float16")
    vectors, stats = engine.encode(["一个长文本", "一个短文本"])
    assert calls == [2, 1]
    assert stats["truncated_records"] == 1
    assert stats["oom_retries"] == 1
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
