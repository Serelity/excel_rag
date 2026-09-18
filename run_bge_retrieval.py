"""Run exact BGE-M3 dense retrieval against the title-level knowledge corpus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pipeline.embedding import BGEM3Embedder
from pipeline.retrieval_baseline import iter_jsonl, load_qrels


PROJECT_ROOT = Path(__file__).resolve().parent


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _corpus_text(record: dict[str, Any]) -> str:
    title = record.get("title", "")
    text = record.get("text", "")
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError("corpus title and text must be strings")
    return "\n".join(part for part in (title.strip(), text.strip()) if part)


def _load_corpus(path: Path) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for record in iter_jsonl(path):
        corpus_id = record.get("_id")
        if not isinstance(corpus_id, str) or not corpus_id:
            raise ValueError(f"{path}: every corpus record requires a non-empty string _id")
        if corpus_id in seen:
            raise ValueError(f"{path}: duplicate corpus id {corpus_id!r}")
        text = _corpus_text(record)
        if not text:
            raise ValueError(f"{path}: corpus record {corpus_id!r} has no title or text")
        identifiers.append(corpus_id)
        texts.append(text)
        seen.add(corpus_id)
    if not identifiers:
        raise ValueError(f"{path}: corpus is empty")
    return identifiers, texts


def _iter_query_batches(
    path: Path,
    query_ids: set[str],
    batch_size: int,
):
    identifiers: list[str] = []
    texts: list[str] = []
    found: set[str] = set()
    for record in iter_jsonl(path):
        query_id = record.get("_id")
        if query_id not in query_ids:
            continue
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{path}: query {query_id!r} has invalid text")
        if query_id in found:
            raise ValueError(f"{path}: duplicate query id {query_id!r}")
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
        raise ValueError(f"{path}: {len(missing)} judged queries are missing; examples={example}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exact BGE-M3 title retrieval.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval/title-v1",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--query-view", choices=("content", "goal", "joint"), default="joint")
    parser.add_argument("--model", required=True, help="Absolute local BGE-M3 model path.")
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-device", default="cuda:0")
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--top-k", type=_positive_int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("BGE retrieval requires the locked index environment") from exc

    qrels = load_qrels(args.dataset / f"qrels/{args.split}.tsv")
    if not qrels:
        raise ValueError(f"split {args.split!r} has no judged queries")
    corpus_ids, corpus_texts = _load_corpus(args.dataset / "corpus.jsonl")
    top_k = min(args.top_k, len(corpus_ids))
    output_path = args.output or (
        PROJECT_ROOT
        / "run-records/retrieval-baselines"
        / f"bge-m3-{args.query_view}.{args.split}.trec"
    )

    embedder = BGEM3Embedder(
        args.model,
        revision=args.model_revision,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=True,
        normalize_embeddings=True,
        device=args.device,
    )
    print(f"encoding_corpus={len(corpus_ids)}", flush=True)
    corpus_vectors = embedder.encode(corpus_texts)
    corpus_tensor = torch.tensor(
        corpus_vectors,
        dtype=torch.float32,
        device=args.search_device,
    ).transpose(0, 1).contiguous()

    query_path = args.dataset / f"queries.{args.query_view}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_count = 0
    next_progress = 1_000
    run_name = f"bge-m3-{args.query_view}"
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        with torch.inference_mode():
            for query_ids, query_texts in _iter_query_batches(
                query_path, set(qrels), args.batch_size
            ):
                query_vectors = embedder.encode(query_texts)
                query_tensor = torch.tensor(
                    query_vectors,
                    dtype=torch.float32,
                    device=args.search_device,
                )
                scores, positions = torch.topk(query_tensor @ corpus_tensor, k=top_k, dim=1)
                for row_index, query_id in enumerate(query_ids):
                    for rank, (position, score) in enumerate(
                        zip(positions[row_index].tolist(), scores[row_index].tolist()),
                        start=1,
                    ):
                        writer.writerow(
                            [
                                query_id,
                                "Q0",
                                corpus_ids[position],
                                rank,
                                f"{score:.10g}",
                                run_name,
                            ]
                        )
                query_count += len(query_ids)
                if query_count >= next_progress:
                    print(f"retrieved_queries={query_count}", flush=True)
                    next_progress += 1_000

    metadata = {
        "run_name": run_name,
        "split": args.split,
        "query_view": args.query_view,
        "model": args.model,
        "model_revision": args.model_revision,
        "max_length": args.max_length,
        "top_k": top_k,
        "query_count": query_count,
        "corpus_count": len(corpus_ids),
        "score": "cosine via normalized inner product",
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"queries={query_count} corpus={len(corpus_ids)} output={output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
