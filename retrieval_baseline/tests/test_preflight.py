from __future__ import annotations

import ast
import builtins
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrieval_baseline import preflight
from retrieval_baseline.address import PARSER_VERSION, _implementation
from retrieval_baseline.address import VERSION as ADDRESS_VERSION
from retrieval_baseline.common import VERSION, file_hash
from retrieval_baseline.lexical import TOKENIZER


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def assets(tmp_path):
    paths = {name: tmp_path / name for name in preflight.RESOURCE_NAMES}
    for directory in paths.values():
        directory.mkdir()
    args = SimpleNamespace(**paths, threads=2, max_length=512, batch_size=3,
                           reranker_max_length=512, reranker_batch_size=2,
                           device="cpu", dtype="float32", retriever="hybrid", case_k=50)
    for name in (
        "config.json", "modules.json", "1_Pooling/config.json", "sentence_bert_config.json",
        "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
    ):
        write_json(paths["model"] / name, {})
    write_json(paths["model"] / "config.json", {"model_type": "xlm-roberta", "hidden_size": 1024})
    write_json(paths["model"] / "1_Pooling/config.json", {"pooling_mode_cls_token": True})
    (paths["model"] / "model.safetensors").write_bytes(b"synthetic embedding weight")
    write_json(paths["reranker_model"] / "config.json", {
        "model_type": "xlm-roberta", "architectures": ["XLMRobertaForSequenceClassification"],
        "id2label": {"0": "score"},
    })
    for name in ("tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"):
        write_json(paths["reranker_model"] / name, {})
    (paths["reranker_model"] / "model.safetensors").write_bytes(b"synthetic reranker weight")
    (paths["dataset"] / "dataset.sqlite3").write_bytes(b"synthetic dataset")
    write_json(paths["dataset"] / "manifest.json", {
        "version": VERSION, "counts": {"corpus_unique_texts": 2}, "input_sha256": "a" * 64,
        "artifacts": {"dataset.sqlite3": file_hash(paths["dataset"] / "dataset.sqlite3"),
                      "queries.test.jsonl": "not-to-be-read", "qrels.test.jsonl": "not-to-be-read"},
    })
    data_hash = file_hash(paths["dataset"] / "manifest.json")
    for name in ("index", "address_index"):
        (paths[name] / "index.sqlite3").write_bytes(b"synthetic index")
        manifest = {"version": VERSION, "tokenizer": TOKENIZER,
                    "dataset_manifest_sha256": data_hash, "source_input_sha256": "a" * 64,
                    "indexed_cases": 2, "index_sha256": file_hash(paths[name] / "index.sqlite3")}
        if name == "address_index":
            manifest.update(version=ADDRESS_VERSION, parser_version=PARSER_VERSION,
                            implementation_sha256=_implementation())
        write_json(paths[name] / "manifest.json", manifest)
    for name in ("index.faiss", "shard-000000.npy", "shard-000000.ids.npy"):
        (paths["dense_index"] / name).write_bytes(b"synthetic dense artifact")
    packages = dict.fromkeys(preflight.PACKAGE_NAMES, "fixture-version")
    encoder = {"profile": "bge-m3-dense-cls-v1", "dimension": 1024, "max_length": 512,
               "batch_size": 8, "device": "cuda", "dtype": "float32", "prompt": "",
               "normalized": True, "model": preflight._model_fingerprint(paths["model"]),
               "packages": {key: packages[key] for key in (
                   "torch", "transformers", "sentence-transformers", "numpy",
               )}}
    write_json(paths["dense_index"] / "manifest.json", {
        "version": "raw-dense-v1", "indexed_cases": 2,
        "index_sha256": file_hash(paths["dense_index"] / "index.faiss"),
        "config": {"dataset_manifest_sha256": data_hash, "encoder": encoder,
                   "input_field": "case_content", "faiss_version": "fixture-version",
                   "implementation_sha256": {name: file_hash(
                       Path(preflight.__file__).with_name(name))
                       for name in ("dense.py", "encoder.py")}},
        "shards": [{"vectors": "shard-000000.npy", "ids": "shard-000000.ids.npy", "count": 2,
                    "last_rid": 2,
                    "vectors_sha256": file_hash(paths["dense_index"] / "shard-000000.npy"),
                    "ids_sha256": file_hash(paths["dense_index"] / "shard-000000.ids.npy")}],
    })
    return args, packages


@pytest.fixture
def passing_environment(assets, monkeypatch):
    args, packages = assets
    monkeypatch.setattr(preflight, "_runtime_info", lambda unused: ({"packages": packages}, []))
    return args


def smoke_queries():
    return [{"mode": "problem", "query": "synthetic private issue"},
            {"mode": "address", "query": "synthetic private address"},
            {"mode": "combined", "query": "synthetic private issue",
             "address": "synthetic private address"}]


def test_static_fingerprints_need_no_ml_imports_and_never_read_heldout_files(assets, monkeypatch):
    args, _ = assets
    real_import, real_open = builtins.__import__, Path.open

    def audited_import(name, *arguments, **kwargs):
        assert name.split(".")[0] not in {"numpy", "torch", "faiss"}
        return real_import(name, *arguments, **kwargs)

    def audited_open(path, *arguments, **kwargs):
        assert not path.name.startswith(("queries.test", "qrels.test"))
        return real_open(path, *arguments, **kwargs)

    monkeypatch.setattr(builtins, "__import__", audited_import)
    monkeypatch.setattr(Path, "open", audited_open)
    result = preflight.resource_fingerprints(args)
    assert set(result) == set(preflight.RESOURCE_NAMES)
    assert result["model"]["sha256"] == preflight._model_fingerprint(args.model)["sha256"]
    assert result["dataset"]["artifacts"] == {
        "dataset.sqlite3": file_hash(args.dataset / "dataset.sqlite3"),
    }


def test_fingerprint_matches_online_encoder_without_importing_numpy(assets):
    args, _ = assets
    # Execute the production function itself, leaving its unrelated ML imports out.
    tree = ast.parse(Path(preflight.__file__).with_name("encoder.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "model_fingerprint")
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "file_hash": file_hash}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "encoder.py", "exec"), namespace)
    (args.model / "README.md").write_text("excluded")
    (args.model / ".hidden.bin").write_bytes(b"excluded")
    assert preflight._model_fingerprint(args.model) == namespace["model_fingerprint"](args.model)


@pytest.mark.parametrize("resource,filename", [
    ("dataset", "dataset.sqlite3"), ("index", "index.sqlite3"),
    ("dense_index", "index.faiss"), ("dense_index", "shard-000000.ids.npy"),
    ("address_index", "index.sqlite3"), ("model", "model.safetensors"),
])
def test_changed_assets_are_rejected(assets, resource, filename):
    args, _ = assets
    with (getattr(args, resource) / filename).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        preflight.resource_fingerprints(args)


def test_missing_reranker_weights_are_rejected(assets):
    args, _ = assets
    (args.reranker_model / "model.safetensors").unlink()
    with pytest.raises(ValueError):
        preflight.resource_fingerprints(args)


@pytest.mark.parametrize("key,value", [
    ("dataset_manifest_sha256", "wrong"), ("input_field", "extracted_text"),
    ("implementation_sha256", {}),
])
def test_dense_provenance_mismatch_is_rejected(assets, key, value):
    args, _ = assets
    path = args.dense_index / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"][key] = value
    write_json(path, manifest)
    with pytest.raises(ValueError):
        preflight.resource_fingerprints(args)


def test_resource_paths_and_effective_runtime_are_bound(assets):
    args, _ = assets
    config = preflight.runtime_configuration(args)
    assert config["dataset"] == str(args.dataset.resolve())
    assert config["case_k"] == 50 and config["max_terms"] == 32
    assert config["input_field"] == "case_content" and config["allow_broader"] is False
    args.max_length = 1024
    with pytest.raises(ValueError, match="settings mismatch"):
        preflight.resource_fingerprints(args)


def test_runtime_identity_tracks_interpreter_and_missing_packages(monkeypatch):
    def version(package):
        if package == "tokenizers":
            raise preflight.importlib.metadata.PackageNotFoundError(package)
        return "1.2.3"

    monkeypatch.setattr(preflight.importlib.metadata, "version", version)
    identity = preflight.runtime_identity()
    assert identity["python"]["executable"] == sys.executable
    assert identity["packages"]["torch"] == "1.2.3"
    assert identity["packages"]["tokenizers"] is None


def test_malformed_manifest_has_stable_valueerror(assets):
    args, _ = assets
    path = args.dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"] = []
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Malformed runtime resource manifest"):
        preflight.resource_fingerprints(args)


def test_static_only_cannot_claim_formal_readiness(passing_environment, monkeypatch):
    monkeypatch.setattr(preflight, "make_searcher", lambda _: pytest.fail("static inference"))
    report = preflight.inspect_preflight(passing_environment)
    assert report["checks"]["files"]["status"] == "passed"
    assert report["checks"]["environment"]["status"] == "passed"
    assert report["status"] == "not_ready"
    assert report["inference"]["status"] == "not_run"
    assert report["inference"]["joint_hybrid_reranker_passed"] is False
    assert report["inference_implementation_sha256"] == (
        preflight.inference_implementation_fingerprints()
    )


def install_fake_searcher(args, monkeypatch, *, empty_mode=None, fail=False):
    instances = []
    resources = preflight.resource_fingerprints(args)

    class FakeSearcher:
        def __init__(self, _):
            self.calls = []
            self.encoder = self.reranker = None
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def search(self, query, **settings):
            self.calls.append(settings)
            if fail:
                print("private query stdout leaked")
                print("private model stderr leaked", file=sys.stderr)
                raise RuntimeError("private text and secret environment value")
            mode = settings["mode"]
            count = 0 if mode == empty_mode else 2
            if settings["rerank"]:
                identities = []
                for resource, batch_key, length_key in (
                    ("model", "batch_size", "max_length"),
                    ("reranker_model", "reranker_batch_size", "reranker_max_length"),
                ):
                    parameter = SimpleNamespace(dtype="torch." + args.dtype, device=args.device)
                    identities.append(SimpleNamespace(identity={
                        "model": {"sha256": resources[resource]["sha256"]},
                        "device": args.device, "dtype": args.dtype,
                        "batch_size": getattr(args, batch_key),
                        "max_length": getattr(args, length_key),
                    }, model=SimpleNamespace(parameters=lambda value=parameter: iter([value]))))
                self.encoder, self.reranker = identities
            return {"result_count": count, "query": query,
                    "reranking": {"scored_candidates": count if settings["rerank"] else 0},
                    "route_candidate_counts": {"dense": count if mode != "address" else 0}}

    monkeypatch.setattr(preflight, "make_searcher", FakeSearcher)
    return instances


def test_joint_smoke_runs_all_modes_in_one_resident_process(passing_environment, monkeypatch):
    instances = install_fake_searcher(passing_environment, monkeypatch)
    report = preflight.inspect_preflight(passing_environment, smoke_queries())
    assert report["status"] == report["inference"]["status"] == "passed"
    assert report["inference"]["joint_hybrid_reranker_passed"] is True
    assert len(instances) == 1
    assert [row["mode"] for row in instances[0].calls] == ["problem", "address", "combined"]
    assert [row["rerank"] for row in instances[0].calls] == [True, False, True]
    assert all(row["retriever"] == "hybrid" for row in instances[0].calls)
    assert report["inference"]["actual_models"]["embedding"]["batch_size"] == 3
    assert "synthetic private" not in json.dumps(report)


@pytest.mark.parametrize("mode", ["problem", "combined"])
def test_empty_problem_candidates_do_not_count_as_joint_inference(
    passing_environment, monkeypatch, mode,
):
    install_fake_searcher(passing_environment, monkeypatch, empty_mode=mode)
    report = preflight.inspect_preflight(passing_environment, smoke_queries())
    assert report["status"] == report["inference"]["status"] == "failed"
    assert report["inference"]["joint_hybrid_reranker_passed"] is False


def test_model_change_during_smoke_invalidates_readiness(passing_environment, monkeypatch):
    install_fake_searcher(passing_environment, monkeypatch)
    run_smoke = preflight._run_smoke

    def changed_during_run(args, rows, resources, inference):
        run_smoke(args, rows, resources, inference)
        (args.reranker_model / "model.safetensors").write_bytes(b"replaced during inference")

    monkeypatch.setattr(preflight, "_run_smoke", changed_during_run)
    report = preflight.inspect_preflight(passing_environment, smoke_queries())
    assert report["status"] == report["inference"]["status"] == "failed"
    assert report["inference"]["joint_hybrid_reranker_passed"] is False


def test_inference_code_change_during_smoke_invalidates_readiness(passing_environment, monkeypatch):
    install_fake_searcher(passing_environment, monkeypatch)
    before = preflight.inference_implementation_fingerprints()
    identities = iter((before, {**before, "reranker.py": "changed-after-smoke"}))
    monkeypatch.setattr(preflight, "inference_implementation_fingerprints",
                        lambda: next(identities))
    report = preflight.inspect_preflight(passing_environment, smoke_queries())
    assert report["status"] == report["inference"]["status"] == "failed"
    assert report["inference"]["joint_hybrid_reranker_passed"] is False


def test_actual_parameter_dtype_must_match_configured_dtype():
    parameter = SimpleNamespace(dtype="torch.float32", device="cpu")
    model = SimpleNamespace(identity={"dtype": "float16"},
                            model=SimpleNamespace(parameters=lambda: iter([parameter])))
    with pytest.raises(ValueError, match="Actual model dtype mismatch"):
        preflight._loaded_model_settings(model)


def test_exceptions_and_model_console_output_cannot_leak_private_text(
    passing_environment, monkeypatch, capsys,
):
    install_fake_searcher(passing_environment, monkeypatch, fail=True)
    report = preflight.inspect_preflight(passing_environment, smoke_queries())
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert report["inference"]["error_type"] == "RuntimeError"
    assert "private" not in json.dumps(report)
    assert "secret" not in json.dumps(report)


def test_missing_modes_and_heldout_smoke_paths_are_rejected_before_inference(
    passing_environment, monkeypatch, tmp_path,
):
    monkeypatch.setattr(preflight, "make_searcher", lambda _: pytest.fail("invalid smoke ran"))
    for queries in (smoke_queries()[:2], tmp_path / "queries.test.jsonl"):
        report = preflight.inspect_preflight(passing_environment, queries)
        assert report["status"] == "failed"
        assert report["inference"]["error_type"] == "ValueError"


def test_package_mismatch_cannot_run_smoke(passing_environment, monkeypatch):
    monkeypatch.setattr(preflight, "_runtime_info", lambda _: ({"packages": {}}, []))
    monkeypatch.setattr(preflight, "make_searcher", lambda _: pytest.fail("bad environment ran"))
    report = preflight.inspect_preflight(passing_environment, smoke_queries())
    assert report["status"] == "failed"
    assert report["checks"]["environment"]["status"] == "failed"
    assert report["inference"]["status"] == "not_run"


def test_cli_writes_private_report_and_summary_only(
    passing_environment, monkeypatch, tmp_path, capsys,
):
    args = passing_environment
    output = tmp_path / "private" / "preflight.json"
    command = ["preflight", "--output", str(output), "--device", "cpu", "--dtype", "float32",
               "--max-length", "512"]
    for name in preflight.RESOURCE_NAMES:
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as stopped:
        preflight.main()
    assert stopped.value.code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "not_ready"
    assert str(tmp_path) not in json.dumps(summary)
    assert json.loads(output.read_text())["inference"]["status"] == "not_run"
    before = output.read_bytes()
    with pytest.raises(SystemExit) as stopped:
        preflight.main()
    assert stopped.value.code == 1
    assert output.read_bytes() == before
