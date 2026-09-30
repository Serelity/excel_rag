"""Resumable case_content encoding, exact cosine retrieval and fixed citation voting."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import faiss
import numpy as np
from filelock import FileLock

from .common import VERSION, file_hash, read_jsonl, readonly, write_json, write_jsonl
from .encoder import BGEEncoder, normalized_vectors
from .lexical import TOKENIZER, observed_metrics, verified_manifest, vote_cases

DENSE_VERSION = "raw-dense-v1"


def implementation_hashes() -> dict:
    return {name: file_hash(Path(__file__).with_name(name))
            for name in ("dense.py", "encoder.py")}


def atomic_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def atomic_array(path: Path, data: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as handle:
        np.save(handle, data, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_shard(output: Path, entry: dict, dimension: int):
    for name in ("vectors", "ids"):
        if Path(entry[name]).name != entry[name]:
            raise ValueError("Invalid shard filename")
        if file_hash(output / entry[name]) != entry[name + "_sha256"]:
            raise ValueError("Committed shard changed; restore it or use a new output directory")
    vectors = np.load(output / entry["vectors"], allow_pickle=False)
    ids = np.load(output / entry["ids"], allow_pickle=False)
    if (
        ids.dtype != np.int64 or ids.shape != (entry["count"],)
        or vectors.dtype != np.float32 or vectors.shape != (len(ids), dimension)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
        or len(ids) == 0 or (np.diff(ids) <= 0).any()
        or int(ids[-1]) != entry["last_rid"]
    ):
        raise ValueError("Invalid committed shard contents")
    return vectors, ids


def build_dense(dataset: Path, output: Path, encoder, *, shard_size: int = 512,
                resume: bool = False) -> dict:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    data = verified_manifest(dataset, ("dataset.sqlite3",))
    count = data["counts"]["corpus_unique_texts"]
    if count < 1:
        raise ValueError("Empty historical corpus")
    config = {
        "version": DENSE_VERSION, "dataset_version": VERSION,
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "input_field": "case_content", "corpus_count": count,
        "encoder": encoder.identity, "shard_size": shard_size,
        "faiss_version": faiss.__version__, "index_type": "IndexIDMap2(IndexFlatIP)",
        "implementation_sha256": implementation_hashes(),
    }
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".build.lock"), timeout=0), closing(
        readonly(dataset / "dataset.sqlite3")
    ) as source:
        started = time.perf_counter()
        run_path, state_path = output / "run.json", output / "checkpoint.json"
        if run_path.exists():
            if not resume:
                raise FileExistsError("Output exists; use --resume for the same run")
            if read_json(run_path) != config:
                raise ValueError("Dataset/model/runtime/config changed; use a new output directory")
        else:
            if any(p.name != ".build.lock" for p in output.iterdir()):
                raise FileExistsError("Refusing to reuse a nonempty output directory")
            atomic_json(run_path, config)
        state = read_json(state_path) if state_path.exists() else {"shards": [], "stats": {}}
        dimension = encoder.identity["dimension"]
        completed, last_rid = 0, -1
        for entry in state["shards"]:
            _, ids = validate_shard(output, entry, dimension)
            expected = [r[0] for r in source.execute(
                "SELECT rid FROM corpus WHERE rid>? ORDER BY rid LIMIT ?", (last_rid, len(ids)),
            )]
            if ids.tolist() != expected:
                raise ValueError("Committed document order differs from the corpus")
            completed += len(ids)
            last_rid = int(ids[-1])
        if completed > count:
            raise ValueError("Checkpoint exceeds historical corpus size")
        if (output / "manifest.json").exists():
            manifest = read_json(output / "manifest.json")
            if (
                completed != count or manifest["config"] != config
                or manifest["shards"] != state["shards"]
                or file_hash(output / "index.faiss") != manifest["index_sha256"]
            ):
                raise ValueError("Completed dense index changed")
            print(f"resumed_complete=true indexed_cases={count}", flush=True)
            return manifest
        print(f"resumed_documents={completed} corpus_documents={count}", flush=True)
        stats = Counter(state["stats"])
        while completed < count:
            rows = source.execute(
                "SELECT r.rid,r.content FROM corpus c JOIN records r ON r.rid=c.rid "
                "WHERE r.rid>? ORDER BY r.rid LIMIT ?", (last_rid, shard_size),
            ).fetchall()
            if not rows:
                raise ValueError("Corpus ended before expected count")
            vectors, batch_stats = encoder.encode([r[1] for r in rows])
            vectors = normalized_vectors(vectors, len(rows), dimension)
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            name = f"shard-{len(state['shards']):06d}"
            vector_path, id_path = output / f"{name}.npy", output / f"{name}.ids.npy"
            # An uncommitted shard from a failed attempt can be regenerated safely.
            atomic_array(vector_path, vectors)
            atomic_array(id_path, ids)
            entry = {
                "vectors": vector_path.name, "ids": id_path.name,
                "vectors_sha256": file_hash(vector_path), "ids_sha256": file_hash(id_path),
                "count": len(rows), "last_rid": int(ids[-1]),
            }
            state["shards"].append(entry)
            stats.update(batch_stats)
            state["stats"] = dict(stats)
            atomic_json(state_path, state)
            completed += len(rows)
            last_rid = int(ids[-1])
            print(f"encoded_documents={completed}/{count} "
                  f"truncated_documents={stats['truncated_records']}", flush=True)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        for entry in state["shards"]:
            vectors, ids = validate_shard(output, entry, dimension)
            index.add_with_ids(vectors, ids)
        if index.ntotal != count:
            raise ValueError("FAISS index count mismatch")
        temporary = output / "index.faiss.tmp"
        faiss.write_index(index, str(temporary))
        os.replace(temporary, output / "index.faiss")
        manifest = {
            "version": DENSE_VERSION, "config": config, "indexed_cases": count,
            "index_sha256": file_hash(output / "index.faiss"),
            "shards": state["shards"], "encoding_stats": dict(stats),
            "seconds_this_attempt": time.perf_counter() - started,
        }
        atomic_json(output / "manifest.json", manifest)
        return manifest


def verified_lexical_index(dataset: Path, lexical_index: Path) -> dict:
    manifest = read_json(lexical_index / "manifest.json")
    if (
        manifest["version"] != VERSION or manifest["tokenizer"] != TOKENIZER
        or manifest["dataset_manifest_sha256"] != file_hash(dataset / "manifest.json")
        or manifest["index_sha256"] != file_hash(lexical_index / "index.sqlite3")
    ):
        raise ValueError("Lexical metadata index provenance/hash mismatch")
    return manifest


def evaluate_dense(dataset: Path, dense_index: Path, lexical_index: Path, output: Path,
                   encoder, *, split: str = "dev", case_k: int = 50, threads: int = 4) -> dict:
    if split not in {"dev", "test"} or case_k < 1 or threads < 1:
        raise ValueError("Invalid split or retrieval configuration")
    data = verified_manifest(dataset, (f"queries.{split}.jsonl", f"qrels.{split}.jsonl"))
    verified_lexical_index(dataset, lexical_index)
    manifest = read_json(dense_index / "manifest.json")
    config = manifest["config"]
    if (
        manifest["version"] != DENSE_VERSION or config["encoder"] != encoder.identity
        or config["implementation_sha256"] != implementation_hashes()
        or config["dataset_manifest_sha256"] != file_hash(dataset / "manifest.json")
        or config["input_field"] != "case_content"
        or config["faiss_version"] != faiss.__version__
        or manifest["indexed_cases"] != data["counts"]["corpus_unique_texts"]
        or manifest["index_sha256"] != file_hash(dense_index / "index.faiss")
    ):
        raise ValueError("Dense index provenance/model/hash mismatch")
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index(str(dense_index / "index.faiss"))
    if (index.ntotal != manifest["indexed_cases"] or index.d != encoder.identity["dimension"]
            or index.metric_type != faiss.METRIC_INNER_PRODUCT):
        raise ValueError("Dense index shape/metric mismatch")
    queries = read_jsonl(dataset / f"queries.{split}.jsonl")
    qrels = read_jsonl(dataset / f"qrels.{split}.jsonl")
    if not queries or [q["source_id"] for q in queries] != [q["source_id"] for q in qrels]:
        raise ValueError("Empty or misaligned queries/qrels")
    output.mkdir(parents=True, exist_ok=False)
    with closing(readonly(lexical_index / "index.sqlite3")) as db:
        results, latencies, encoding_seconds, search_seconds = [], [], [], []
        stats = Counter()
        known = {r[0] for r in db.execute("SELECT knowledge_id FROM titles")}
        # Query vectors are saved separately for reproducibility; no label enters encoding.
        query_vectors = []
        for number, query in enumerate(queries, 1):
            started = time.perf_counter()
            vector, batch_stats = encoder.encode([query["case_content"]])
            vector = normalized_vectors(vector, 1, index.d)
            encoded = time.perf_counter()
            scores, ids = index.search(vector, min(case_k, index.ntotal))
            searched = time.perf_counter()
            ids = [int(rid) for rid in ids[0]]
            marks = ",".join("?" for _ in ids)
            metadata = {r[0]: r for r in db.execute(
                f"SELECT docid,source_id,source_row,group_id FROM cases WHERE docid IN ({marks})",
                ids,
            )}
            if len(metadata) != len(ids):
                raise ValueError("Dense candidates absent from historical metadata")
            ranked = sorted(zip(ids, scores[0], strict=True),
                            key=lambda x: (-float(x[1]), metadata[x[0]][1]))
            cases = [metadata[rid] for rid, _ in ranked]
            ranking, hits, support = vote_cases(db, cases, top_k=10)
            candidate_ids = sorted({r[0] for r in db.execute(
                f"SELECT knowledge_id FROM links WHERE docid IN ({marks})", ids,
            )})
            results.append({
                "source_id": query["source_id"], "rankings": {"dense_case_vote": ranking},
                "case_hits": hits, "supporting_cases": support,
                "candidate_knowledge_ids": candidate_ids,
                "encoding": batch_stats,
            })
            stats.update(batch_stats)
            query_vectors.append(vector[0])
            encoding_seconds.append(encoded - started)
            search_seconds.append(searched - encoded)
            latencies.append(time.perf_counter() - started)
            if number % 10 == 0:
                print(f"evaluated_queries={number}/{len(queries)}", flush=True)
        targets = [set(q["observed_knowledge_ids"]) for q in qrels]
        atomic_array(output / "query-vectors.npy", np.stack(query_vectors))
        write_jsonl(output / "rankings.jsonl", results)
        report = {
            "version": VERSION, "dense_version": DENSE_VERSION, "split": split,
            "query_count": len(queries),
            "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
            "index_manifest_sha256": file_hash(dense_index / "manifest.json"),
            "metadata_index_manifest_sha256": file_hash(lexical_index / "manifest.json"),
            "config": {"case_k": case_k, "top_k": 10, "rrf_k": 60, "faiss_threads": threads},
            "metrics": {"dense_case_vote": observed_metrics(
                targets, [r["rankings"]["dense_case_vote"] for r in results], known,
            )},
            "query_encoding_stats": dict(stats),
            "metrics_by_query_length": {
                group: observed_metrics(
                    [target for target, r in zip(targets, results, strict=True)
                     if bool(r["encoding"]["truncated_records"]) == truncated],
                    [r["rankings"]["dense_case_vote"] for r in results
                     if bool(r["encoding"]["truncated_records"]) == truncated], known,
                )
                for group, truncated in (("within_limit", False), ("truncated", True))
            },
            "latency_seconds": {
                "mean": sum(latencies) / len(latencies),
                "p95": sorted(latencies)[math.ceil(len(latencies) * .95) - 1],
                "mean_encoding": sum(encoding_seconds) / len(queries),
                "mean_exact_search": sum(search_seconds) / len(queries),
            },
            "rankings_sha256": file_hash(output / "rankings.jsonl"),
            "query_vectors_sha256": file_hash(output / "query-vectors.npy"),
            "limitations": [
                "Observed knowledge-reference recall is not case-relevance or answer accuracy.",
                "Empty references are unknown; unseen targets remain in overall denominators.",
                "Input over max_length is right-truncated, with counts reported separately.",
                "Only dense BGE-M3 representations are used; sparse/ColBERT outputs are unused.",
                "Latency includes query encoding/search/voting, not model/index loading.",
                "FAISS exact scores are sorted by source ID within returned equal-score results; "
                "equal scores at the candidate cutoff can still have ambiguous membership.",
            ],
        }
        write_json(output / "report.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "build", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                             default="float16")
        command.add_argument("--max-length", type=int, default=8192)
        command.add_argument("--batch-size", type=int, default=8)
        if name != "check":
            command.add_argument("--dataset", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        if name == "build":
            command.add_argument("--shard-size", type=int, default=512)
            command.add_argument("--resume", action="store_true")
        if name == "evaluate":
            command.add_argument("--dense-index", type=Path, required=True)
            command.add_argument("--lexical-index", type=Path, required=True)
            command.add_argument("--split", choices=("dev", "test"), default="dev")
            command.add_argument("--case-k", type=int, default=50)
            command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    encoder = BGEEncoder(args.model, max_length=args.max_length, batch_size=args.batch_size,
                         device=args.device, dtype=args.dtype)
    if args.command == "check":
        vectors, stats = encoder.encode(["道路积水，希望排水。", "咨询社会保险办理条件。"])
        result = {"check": "passed", "dimension": vectors.shape[1], "stats": stats,
                  "model_sha256": encoder.identity["model"]["sha256"]}
    elif args.command == "build":
        result = build_dense(args.dataset, args.output, encoder,
                             shard_size=args.shard_size, resume=args.resume)
    else:
        result = evaluate_dense(args.dataset, args.dense_index, args.lexical_index, args.output,
                                encoder, split=args.split, case_k=args.case_k, threads=args.threads)
    # Full model-file and shard manifests are saved on disk; progress logs stay compact.
    print(json.dumps({k: v for k, v in result.items() if k not in {"config", "shards"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
