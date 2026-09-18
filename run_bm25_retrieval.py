"""Run a Chinese Lucene BM25 baseline through pinned Pyserini."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pipeline.retrieval_baseline import iter_jsonl, load_qrels


PROJECT_ROOT = Path(__file__).resolve().parent
PYSERINI_VERSION = "1.2.0"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _corpus_contents(record: dict[str, Any]) -> str:
    title = record.get("title", "")
    text = record.get("text", "")
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError("corpus title and text must be strings")
    return "\n".join(part for part in (title.strip(), text.strip()) if part)


def _write_pyserini_collection(corpus_path: Path, collection_dir: Path) -> int:
    collection_dir.mkdir(parents=True)
    output_path = collection_dir / "docs.jsonl"
    count = 0
    seen: set[str] = set()
    with output_path.open("w", encoding="utf-8") as output:
        for record in iter_jsonl(corpus_path):
            corpus_id = record.get("_id")
            if not isinstance(corpus_id, str) or not corpus_id:
                raise ValueError("every corpus record requires a non-empty string _id")
            if corpus_id in seen:
                raise ValueError(f"duplicate corpus id {corpus_id!r}")
            contents = _corpus_contents(record)
            if not contents:
                raise ValueError(f"corpus record {corpus_id!r} has no title or text")
            output.write(
                json.dumps(
                    {"id": corpus_id, "contents": contents},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            seen.add(corpus_id)
            count += 1
    if count == 0:
        raise ValueError("corpus is empty")
    return count


def _ensure_pyserini_version() -> None:
    try:
        installed = importlib.metadata.version("pyserini")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Pyserini {PYSERINI_VERSION} is required for the BM25 baseline"
        ) from exc
    if installed != PYSERINI_VERSION:
        raise RuntimeError(
            f"Pyserini version mismatch: installed {installed}, expected {PYSERINI_VERSION}"
        )


def ensure_index(
    corpus_path: Path,
    index_dir: Path,
    *,
    threads: int,
) -> dict[str, Any]:
    corpus_sha256 = _sha256_file(corpus_path)
    manifest_path = index_dir / "civic-rag-index-manifest.json"
    expected = {
        "schema_version": "civic-rag-pyserini-index-v1",
        "pyserini_version": PYSERINI_VERSION,
        "corpus_sha256": corpus_sha256,
        "language": "zh",
        "generator": "DefaultLuceneDocumentGenerator",
    }
    if index_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError(
                f"index directory exists without a compatible manifest: {index_dir}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"index manifest mismatch for {key}; choose a new --index-dir"
                )
        return existing

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{index_dir.name}.build-", dir=index_dir.parent)
    )
    collection_dir = staging / "collection"
    staged_index = staging / "index"
    try:
        corpus_count = _write_pyserini_collection(corpus_path, collection_dir)
        command = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(collection_dir),
            "--index",
            str(staged_index),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--language",
            "zh",
            "--threads",
            str(threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        subprocess.run(command, check=True)
        manifest = {**expected, "corpus_count": corpus_count}
        (staged_index / manifest_path.name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staged_index.replace(index_dir)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _iter_query_batches(path: Path, query_ids: set[str], batch_size: int):
    identifiers: list[str] = []
    texts: list[str] = []
    found: set[str] = set()
    for record in iter_jsonl(path):
        query_id = record.get("_id")
        if query_id not in query_ids:
            continue
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"query {query_id!r} has invalid text")
        if query_id in found:
            raise ValueError(f"duplicate query id {query_id!r}")
        identifiers.append(query_id)
        texts.append(text)
        found.add(query_id)
        if len(identifiers) == batch_size:
            yield identifiers, texts
            identifiers, texts = [], []
    if identifiers:
        yield identifiers, texts
    missing = query_ids - found
    if missing:
        example = sorted(missing)[:3]
        raise ValueError(f"{len(missing)} judged queries are missing; examples={example}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pinned Pyserini BM25 retrieval.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval/title-v1",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--query-view", choices=("content", "goal", "joint"), default="joint")
    parser.add_argument("--top-k", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=512)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--k1", type=_positive_float, default=0.9)
    parser.add_argument("--b", type=_unit_float, default=0.4)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _ensure_pyserini_version()
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:
        raise RuntimeError("unable to import Pyserini LuceneSearcher") from exc

    qrels = load_qrels(args.dataset / f"qrels/{args.split}.tsv")
    if not qrels:
        raise ValueError(f"split {args.split!r} has no judged queries")
    corpus_path = args.dataset / "corpus.jsonl"
    index_dir = args.index_dir or args.dataset / "pyserini-index-zh-v1"
    manifest = ensure_index(corpus_path, index_dir, threads=args.threads)
    output_path = args.output or (
        PROJECT_ROOT
        / "run-records/retrieval-baselines"
        / f"bm25-{args.query_view}.{args.split}.trec"
    )

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_language("zh")
    searcher.set_bm25(args.k1, args.b)
    query_path = args.dataset / f"queries.{args.query_view}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_count = 0
    next_progress = 1_000
    run_name = f"pyserini-bm25-zh-{args.query_view}"
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for query_ids, query_texts in _iter_query_batches(
            query_path, set(qrels), args.batch_size
        ):
            nonempty = [
                (query_id, text)
                for query_id, text in zip(query_ids, query_texts)
                if text.strip()
            ]
            results = (
                searcher.batch_search(
                    [text for _, text in nonempty],
                    [query_id for query_id, _ in nonempty],
                    k=args.top_k,
                    threads=args.threads,
                )
                if nonempty
                else {}
            )
            for query_id in query_ids:
                for rank, hit in enumerate(results.get(query_id, ()), start=1):
                    writer.writerow(
                        [query_id, "Q0", hit.docid, rank, f"{hit.score:.10g}", run_name]
                    )
            query_count += len(query_ids)
            if query_count >= next_progress:
                print(f"retrieved_queries={query_count}", flush=True)
                next_progress += 1_000

    metadata = {
        "run_name": run_name,
        "split": args.split,
        "query_view": args.query_view,
        "pyserini_version": PYSERINI_VERSION,
        "language": "zh",
        "k1": args.k1,
        "b": args.b,
        "top_k": args.top_k,
        "query_count": query_count,
        "corpus_count": manifest["corpus_count"],
        "corpus_sha256": manifest["corpus_sha256"],
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"queries={query_count} output={output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
