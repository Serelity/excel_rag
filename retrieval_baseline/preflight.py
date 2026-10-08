"""Read-only asset checks and an explicit, private joint-inference preflight.

The default invocation never loads a model. A successful static check is not
formal readiness: all three modes must also run with the frozen configuration
in one process, with nonempty hybrid/reranker work in both problem modes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .address import PARSER_VERSION, _implementation
from .address import VERSION as ADDRESS_VERSION
from .common import VERSION, file_hash
from .lexical import TOKENIZER
from .search import _write_private_json, add_runtime_arguments, make_searcher

PREFLIGHT_VERSION = "case-preflight-v1"
RESOURCE_NAMES = ("dataset", "index", "dense_index", "address_index", "model", "reranker_model")
PACKAGE_NAMES = (
    "torch", "transformers", "sentence-transformers", "numpy", "faiss-cpu",
    "sentencepiece", "filelock", "tokenizers",
)
INFERENCE_FILES = (
    "preflight.py", "search.py", "reranker.py", "encoder.py", "dense.py",
    "address.py", "lexical.py", "hybrid.py", "common.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("Invalid resource JSON") from None
    _require(isinstance(value, dict), "Resource JSON must be an object")
    return value


def _model_fingerprint(path: Path) -> dict:
    """Exactly encoder.model_fingerprint's algorithm, without importing numpy.

    Kept deliberately identical to the online encoder so missing ML packages
    cannot prevent a read-only asset audit. A regression checks equivalence.
    """
    if not path.is_dir():
        raise FileNotFoundError("Local model directory is missing")
    names = sorted(
        p for p in path.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
        and p.suffix in {".json", ".txt", ".model", ".safetensors", ".bin"}
    )
    _require(bool(names), "Local model has no configuration, tokenizer or weight files")
    files = [{"name": p.relative_to(path).as_posix(), "bytes": p.stat().st_size,
              "sha256": file_hash(p)} for p in names]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"algorithm": "local-embedding-files-v1", "sha256": digest, "files": files}


def runtime_configuration(args) -> dict:
    """Canonical effective settings shared with the formal freeze gate."""
    settings = {
        name: str(Path(getattr(args, name)).resolve()) if getattr(args, name, None) else None
        for name in RESOURCE_NAMES
    }
    defaults = {
        "threads": 4, "max_length": 8192, "batch_size": 8,
        "reranker_max_length": 1024, "reranker_batch_size": 4,
        "device": "cuda", "dtype": "float16", "retriever": "hybrid",
        "case_k": 50, "max_terms": 32,
    }
    settings.update({name: getattr(args, name, default) for name, default in defaults.items()})
    settings.update(input_field="case_content", rrf_k=60, allow_broader=False)
    return settings


def runtime_identity() -> dict:
    """Cheap interpreter/package binding for preflight, freeze, and collection."""
    packages = {}
    for package in PACKAGE_NAMES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": {"executable": sys.executable, "version": platform.python_version()},
            "packages": packages}


def inference_implementation_fingerprints() -> dict[str, str]:
    """Bind only code used by the real joint preflight, not reports or documents."""
    return {name: file_hash(Path(__file__).with_name(name)) for name in INFERENCE_FILES}


def _verified_artifact(directory: Path, name: str, expected: str) -> str:
    _require(isinstance(name, str) and Path(name).name == name, "Invalid artifact filename")
    actual = file_hash(directory / name)
    _require(actual == expected, "Resource artifact hash mismatch")
    return actual


def resource_fingerprints(args) -> dict:
    """Validate runtime assets only; never read baseline query or qrel files.

    Missing files raise FileNotFoundError; incompatible manifests, file hashes,
    model profiles or representation settings raise ValueError. No ML package
    import, model load, network access, or mutation occurs here.
    """
    try:
        return _resource_fingerprints(args)
    except FileNotFoundError:
        raise
    except (KeyError, TypeError, AttributeError, UnicodeError, OSError):
        raise ValueError("Malformed runtime resource manifest") from None


def _resource_fingerprints(args) -> dict:
    config = runtime_configuration(args)
    _require(all(config[name] for name in RESOURCE_NAMES), "All six runtime resources are required")
    paths = {name: Path(config[name]) for name in RESOURCE_NAMES}
    manifests = {name: _json_object(paths[name] / "manifest.json")
                 for name in RESOURCE_NAMES[:4]}
    data, lexical, dense, address = [manifests[name] for name in RESOURCE_NAMES[:4]]
    _require(data.get("version") == VERSION, "Unsupported dataset version")
    count = data.get("counts", {}).get("corpus_unique_texts")
    _require(type(count) is int and count > 0, "Invalid historical corpus count")
    dataset_hash = file_hash(paths["dataset"] / "manifest.json")
    artifacts = {"dataset": {"dataset.sqlite3": _verified_artifact(
        paths["dataset"], "dataset.sqlite3", data.get("artifacts", {}).get("dataset.sqlite3"),
    )}}
    for name in ("index", "dense_index", "address_index"):
        manifest = manifests[name]
        relation = manifest.get("config", {}) if name == "dense_index" else manifest
        _require(relation.get("dataset_manifest_sha256") == dataset_hash,
                 "Index refers to another dataset")
        _require(manifest.get("indexed_cases") == count, "Index corpus count mismatch")
        filename = "index.faiss" if name == "dense_index" else "index.sqlite3"
        artifacts[name] = {filename: _verified_artifact(
            paths[name], filename, manifest.get("index_sha256"),
        )}
    _require(lexical.get("version") == VERSION and lexical.get("tokenizer") == TOKENIZER,
             "Unsupported lexical index profile")
    _require(address.get("version") == ADDRESS_VERSION
             and address.get("parser_version") == PARSER_VERSION
             and address.get("implementation_sha256") == _implementation(),
             "Address index implementation mismatch")
    for manifest in (lexical, address):
        _require(manifest.get("source_input_sha256") == data.get("input_sha256"),
                 "Index source fingerprint mismatch")
    dense_config = dense.get("config", {})
    _require(dense.get("version") == "raw-dense-v1"
             and dense_config.get("input_field") == "case_content"
             and dense_config.get("implementation_sha256") == {
                 name: file_hash(Path(__file__).with_name(name))
                 for name in ("dense.py", "encoder.py")
             }, "Dense index implementation or input profile mismatch")
    shards = dense.get("shards")
    _require(isinstance(shards, list) and bool(shards), "Dense shard manifest is missing")
    shard_count, seen_shards = 0, set()
    for shard in shards:
        _require(isinstance(shard, dict) and type(shard.get("count")) is int
                 and shard["count"] > 0, "Invalid dense shard manifest")
        shard_count += shard["count"]
        for kind in ("vectors", "ids"):
            name = shard.get(kind)
            _require(isinstance(name, str) and name not in seen_shards
                     and name.endswith(".npy"), "Invalid or repeated dense shard filename")
            seen_shards.add(name)
            artifacts["dense_index"][name] = _verified_artifact(
                paths["dense_index"], name, shard.get(kind + "_sha256"),
            )
    _require(shard_count == count, "Dense shard corpus count mismatch")
    from .prepare_model import verify_files as verify_encoder
    from .prepare_reranker import verify_files as verify_reranker

    verify_encoder(paths["model"])
    verify_reranker(paths["reranker_model"])
    model = _model_fingerprint(paths["model"])
    reranker = _model_fingerprint(paths["reranker_model"])
    encoder = dense_config.get("encoder", {})
    _require(isinstance(encoder, dict) and encoder.get("model") == model,
             "Dense index embedding model fingerprint mismatch")
    expected = {
        "profile": "bge-m3-dense-cls-v1", "dimension": 1024,
        "max_length": config["max_length"], "dtype": config["dtype"],
        "prompt": "", "normalized": True,
    }
    _require(all(encoder.get(key) == value for key, value in expected.items()),
             "Dense index embedding settings mismatch")
    resources = {
        name: {"path": str(paths[name]), "manifest_sha256": file_hash(
            paths[name] / "manifest.json"), "artifacts": artifacts[name]}
        for name in RESOURCE_NAMES[:4]
    }
    resources["model"] = {"path": str(paths["model"]), **model}
    resources["reranker_model"] = {"path": str(paths["reranker_model"]), **reranker}
    return resources


def _memory_info() -> dict:
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return {"total_bytes": memory.get("MemTotal"), "available_bytes": memory.get("MemAvailable")}


def _runtime_info(args) -> tuple[dict, list[str]]:
    """Inventory actual packages and devices; this does not run model inference."""
    config = runtime_configuration(args)
    runtime = {
        **runtime_identity(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "cuda": {"available": False}, "memory": _memory_info(), "disks": {},
    }
    errors = []
    if any(value is None for value in runtime["memory"].values()):
        errors.append("host_memory_probe_unavailable")
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        errors.append("unsupported_python_version")
    for package, version in runtime["packages"].items():
        if version is None:
            errors.append("missing_package:" + package)
    for key, minimum, maximum in (
        ("threads", 1, None), ("batch_size", 1, None), ("reranker_batch_size", 1, None),
        ("max_length", 16, 8192), ("reranker_max_length", 16, 8192),
        ("case_k", 1, None), ("max_terms", 1, None),
    ):
        value = config[key]
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            errors.append("invalid_setting:" + key)
    if config["retriever"] != "hybrid":
        errors.append("formal_preflight_requires_hybrid")
    if config["dtype"] not in {"float32", "float16", "bfloat16"}:
        errors.append("unsupported_dtype")
    device = config["device"]
    if not isinstance(device, str) or not (device == "cpu" or device.startswith("cuda")):
        errors.append("unsupported_device")
    elif device == "cpu" and config["dtype"] != "float32":
        errors.append("cpu_requires_float32")
    try:
        import torch

        available = torch.cuda.is_available()
        runtime["cuda"].update(available=available, torch_cuda_version=torch.version.cuda)
        if isinstance(device, str) and device.startswith("cuda"):
            if not available:
                errors.append("cuda_unavailable")
            else:
                selected = torch.device(device)
                index = (selected.index if selected.index is not None
                         else torch.cuda.current_device())
                properties = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                runtime["cuda"].update(
                    selected_device=index, device_name=properties.name,
                    total_memory_bytes=total, free_memory_bytes=free,
                )
                if config["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
                    errors.append("cuda_bfloat16_unsupported")
    except Exception:
        errors.append("torch_device_probe_failed")
    for name in RESOURCE_NAMES:
        path = config[name]
        if path is None:
            continue
        location = Path(path)
        while not location.exists() and location != location.parent:
            location = location.parent
        try:
            usage = shutil.disk_usage(location)
            runtime["disks"][name] = {"total_bytes": usage.total, "free_bytes": usage.free}
            if usage.free <= 0:
                errors.append("disk_full:" + name)
        except OSError:
            errors.append("disk_probe_failed:" + name)
    return runtime, errors


def _check_dense_packages(args, runtime: dict) -> list[str]:
    """Online dense identity compares recorded versions, not only importability."""
    config = _json_object(Path(args.dense_index) / "manifest.json").get("config", {})
    expected = config.get("encoder", {}).get("packages", {})
    errors = ["dense_package_mismatch:" + package for package in (
        "torch", "transformers", "sentence-transformers", "numpy",
    ) if expected.get(package) != runtime["packages"].get(package)]
    if config.get("faiss_version") != runtime["packages"].get("faiss-cpu"):
        errors.append("dense_package_mismatch:faiss-cpu")
    return errors


def _smoke_rows(smoke_queries: Path | list[dict]) -> list[dict]:
    if isinstance(smoke_queries, Path):
        _require(not smoke_queries.name.startswith(("qrels.test", "queries.test")),
                 "Held-out baseline files cannot be used for preflight")
        try:
            rows = json.loads(smoke_queries.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeError):
            raise ValueError("Smoke queries must be valid JSON") from None
    else:
        rows = smoke_queries
    _require(isinstance(rows, list) and bool(rows), "Smoke queries must be a nonempty array")
    for row in rows:
        _require(isinstance(row, dict) and row.get("mode") in {"problem", "address", "combined"}
                 and isinstance(row.get("query"), str) and bool(row["query"].strip()),
                 "Invalid smoke query mode or text")
        if row["mode"] == "combined":
            _require(isinstance(row.get("address"), str) and bool(row["address"].strip()),
                     "Combined smoke query requires an address")
        else:
            _require(row.get("address") is None, "Only combined smoke queries accept an address")
    _require({row["mode"] for row in rows} == {"problem", "address", "combined"},
             "Smoke queries must cover all three modes")
    return rows


def _gpu_measurement(device: str) -> dict:
    # Do not import torch for a static check. It is already loaded after a smoke run.
    torch = sys.modules.get("torch")
    try:
        if torch is not None and device.startswith("cuda") and torch.cuda.is_available():
            return {"allocated_bytes": torch.cuda.memory_allocated(device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    except Exception:
        pass
    return {}


def _loaded_model_settings(model) -> dict:
    settings = {key: model.identity.get(key) for key in (
        "profile", "device", "dtype", "batch_size", "max_length",
    )}
    parameter = next(model.model.parameters())
    settings["parameter_dtype"] = str(parameter.dtype).removeprefix("torch.")
    settings["parameter_device"] = str(parameter.device)
    _require(settings["parameter_dtype"] == settings["dtype"], "Actual model dtype mismatch")
    requested = settings["device"]
    actual = settings["parameter_device"]
    _require(actual.startswith("cuda:") if requested == "cuda" else actual == requested,
             "Actual model device mismatch")
    return settings


def _run_smoke(args, rows: list[dict], resources: dict, inference: dict) -> None:
    config = runtime_configuration(args)
    inference.update(status="failed", process_id=os.getpid(),
                     execution="same_process_case_searcher")
    started = time.perf_counter()
    modes, jointly_scored_modes = set(), set()
    with make_searcher(args) as searcher:
        for row in rows:
            mode = row["mode"]
            result = searcher.search(
                row["query"], mode=mode, address=row.get("address"), retriever="hybrid",
                top_k=config["case_k"], case_k=config["case_k"],
                max_terms=config["max_terms"], allow_broader=False, rerank=mode != "address",
            )
            modes.add(mode)
            inference["modes"] = sorted(modes)
            inference["queries_completed"] += 1
            scored = result.get("reranking", {}).get("scored_candidates", 0)
            dense_count = result.get("route_candidate_counts", {}).get("dense", 0)
            encoder_retries = (result.get("query_encoding_stats") or {}).get("oom_retries", 0)
            reranker_retries = result.get("reranking", {}).get("stats", {}).get("oom_retries", 0)
            inference["results"].append({
                "mode": mode, "result_count": result["result_count"],
                "scored_candidates": scored, "dense_candidates": dense_count,
                "embedding_effective_batch_size": 1 if dense_count else None,
                "embedding_oom_retries": encoder_retries,
                "reranker_batch_size_after_oom": min(
                    scored, max(1, config["reranker_batch_size"] // (2 ** reranker_retries)),
                ) if scored else None,
                "reranker_oom_retries": reranker_retries,
            })
            if mode != "address" and scored > 0 and dense_count > 0:
                _require(searcher.encoder is not None and searcher.reranker is not None,
                         "Joint inference did not retain both models")
                for model, key in ((searcher.encoder, "model"),
                                   (searcher.reranker, "reranker_model")):
                    _require(model.identity.get("model", {}).get("sha256")
                             == resources[key]["sha256"], "Loaded model fingerprint mismatch")
                    _require(model.identity.get("device") == config["device"]
                             and model.identity.get("dtype") == config["dtype"],
                             "Loaded model runtime mismatch")
                jointly_scored_modes.add(mode)
        _require(modes == {"problem", "address", "combined"}
                 and jointly_scored_modes == {"problem", "combined"},
                 "Both problem modes require nonempty joint hybrid and reranker inference")
        inference["actual_models"] = {
            name: _loaded_model_settings(model)
            for name, model in (("embedding", searcher.encoder), ("reranker", searcher.reranker))
        }
        inference["gpu_memory"] = _gpu_measurement(config["device"])
    inference.update(status="passed", joint_hybrid_reranker_passed=True,
                     seconds=time.perf_counter() - started)


def inspect_preflight(args, smoke_queries: Path | list[dict] | None = None) -> dict:
    """Return only metadata/counts; no queries, case content, or exception text."""
    report = {
        "schema_version": PREFLIGHT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "not_ready", "runtime_config": runtime_configuration(args),
        "checks": {}, "resources": {}, "runtime": {},
        "inference_implementation_sha256": {},
        "inference": {"status": "not_run", "joint_hybrid_reranker_passed": False,
                      "modes": [], "queries_completed": 0, "results": []},
    }
    try:
        report["resources"] = resource_fingerprints(args)
        report["checks"]["files"] = {"status": "passed"}
    except Exception as error:
        report["checks"]["files"] = {"status": "failed", "error_type": type(error).__name__}
    try:
        report["inference_implementation_sha256"] = inference_implementation_fingerprints()
        report["checks"]["implementation"] = {"status": "passed"}
    except Exception as error:
        report["checks"]["implementation"] = {
            "status": "failed", "error_type": type(error).__name__,
        }
    # Model libraries can emit unsanitized paths/text. Suppress their console output;
    # only this module's fixed-key summary is printed by the CLI.
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink), \
            contextlib.redirect_stderr(sink):
        try:
            runtime, errors = _runtime_info(args)
            report["runtime"] = runtime
            if report["checks"]["files"]["status"] == "passed":
                errors.extend(_check_dense_packages(args, runtime))
            report["checks"]["environment"] = {
                "status": "failed" if errors else "passed", "errors": errors,
            }
        except Exception as error:
            report["checks"]["environment"] = {
                "status": "failed", "error_type": type(error).__name__,
            }
        if all(check["status"] == "passed" for check in report["checks"].values()):
            if smoke_queries is not None:
                try:
                    rows = _smoke_rows(smoke_queries)
                    report["inference"]["queries_planned"] = len(rows)
                    report["inference"]["smoke_queries_sha256"] = hashlib.sha256(
                        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode(),
                    ).hexdigest()
                    _run_smoke(args, rows, report["resources"], report["inference"])
                    # Detect an asset or manifest changed while the models were running.
                    _require(resource_fingerprints(args) == report["resources"],
                             "Runtime resources changed during inference")
                    _require(inference_implementation_fingerprints()
                             == report["inference_implementation_sha256"],
                             "Inference implementation changed during preflight")
                    report["status"] = "passed"
                except Exception as error:
                    report["inference"].update(status="failed", error_type=type(error).__name__,
                                               joint_hybrid_reranker_passed=False)
                    report["status"] = "failed"
        else:
            report["status"] = "failed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-queries", type=Path,
                        help="Private JSON array with mode/query and combined address; all modes")
    parser.add_argument("--retriever", choices=("hybrid",), default="hybrid")
    parser.add_argument("--case-k", type=int, default=50)
    parser.add_argument("--max-terms", type=int, default=32)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output exists")
        report = inspect_preflight(args, args.smoke_queries)
        _write_private_json(args.output, report)
        print(json.dumps({"status": report["status"], "checks": report["checks"],
                          "inference_status": report["inference"]["status"],
                          "queries_completed": report["inference"]["queries_completed"]}))
        code = {"passed": 0, "not_ready": 2, "failed": 1}[report["status"]]
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
