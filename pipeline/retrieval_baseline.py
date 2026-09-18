"""Dependency-light helpers for retrieval baselines and evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        first = next(reader, None)
        if first is None:
            return {}
        rows: Iterable[list[str]]
        if first[:3] == ["query-id", "corpus-id", "score"]:
            rows = reader
        else:
            rows = chain((first,), reader)
        for line_number, row in enumerate(rows, start=2):
            if len(row) != 3:
                raise ValueError(f"{path}: qrel row {line_number} must have three columns")
            query_id, corpus_id, raw_score = row
            score = int(raw_score)
            if score > 0:
                qrels[query_id][corpus_id] = score
    return dict(qrels)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


def load_query_metadata(
    path: str | Path,
    query_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        query_id = record.get("_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{path}: every query requires a non-empty string _id")
        if query_ids is not None and query_id not in query_ids:
            continue
        value = record.get("metadata", {})
        if not isinstance(value, dict):
            raise ValueError(f"{path}: query {query_id!r} has invalid metadata")
        metadata[query_id] = value
    return metadata


def _category_key(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    category = metadata.get("category", [])
    if not isinstance(category, list):
        return ()
    return tuple(str(value).strip() for value in category if str(value).strip())


def popularity_rankings(
    train_qrels: Mapping[str, Mapping[str, int]],
    train_metadata: Mapping[str, Mapping[str, Any]],
    target_metadata: Mapping[str, Mapping[str, Any]],
    *,
    top_k: int,
    category_aware: bool,
) -> dict[str, list[tuple[str, float]]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    global_counts: Counter[str] = Counter()
    category_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for query_id, judgments in train_qrels.items():
        category = _category_key(train_metadata.get(query_id, {}))
        for corpus_id in judgments:
            global_counts[corpus_id] += 1
            if category:
                category_counts[category][corpus_id] += 1

    global_ranking = sorted(global_counts.items(), key=lambda item: (-item[1], item[0]))
    rankings: dict[str, list[tuple[str, float]]] = {}
    for query_id, metadata in target_metadata.items():
        selected: list[tuple[str, int]] = []
        seen: set[str] = set()
        if category_aware:
            category = _category_key(metadata)
            for corpus_id, count in sorted(
                category_counts.get(category, {}).items(),
                key=lambda item: (-item[1], item[0]),
            ):
                selected.append((corpus_id, count))
                seen.add(corpus_id)
                if len(selected) >= top_k:
                    break
        if len(selected) < top_k:
            for corpus_id, count in global_ranking:
                if corpus_id in seen:
                    continue
                selected.append((corpus_id, count))
                if len(selected) >= top_k:
                    break
        rankings[query_id] = [
            (corpus_id, float(top_k - position))
            for position, (corpus_id, _) in enumerate(selected)
        ]
    return rankings


def write_trec_run(
    path: str | Path,
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    run_name: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for query_id in sorted(rankings):
            for rank, (corpus_id, score) in enumerate(rankings[query_id], start=1):
                writer.writerow([query_id, "Q0", corpus_id, rank, f"{score:.10g}", run_name])


def load_trec_run(path: str | Path) -> dict[str, list[str]]:
    ranked: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 6:
                raise ValueError(f"{path}: run row {line_number} must have six columns")
            query_id, _, corpus_id, raw_rank, _, _ = row
            ranked[query_id].append((int(raw_rank), corpus_id))

    result: dict[str, list[str]] = {}
    for query_id, values in ranked.items():
        seen: set[str] = set()
        result[query_id] = []
        for _, corpus_id in sorted(values):
            if corpus_id not in seen:
                result[query_id].append(corpus_id)
                seen.add(corpus_id)
    return result


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        ((2**relevance) - 1) / math.log2(position + 2)
        for position, relevance in enumerate(relevances)
    )


def evaluate_run(
    qrels: Mapping[str, Mapping[str, int]],
    rankings: Mapping[str, Sequence[str]],
    *,
    cutoffs: Sequence[int] = (1, 5, 10, 50),
    mrr_cutoff: int = 10,
    ndcg_cutoff: int = 10,
    training_frequencies: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if not qrels:
        raise ValueError("qrels contain no positive judgments")
    normalized_cutoffs = sorted(set(cutoffs))
    if not normalized_cutoffs or any(value <= 0 for value in normalized_cutoffs):
        raise ValueError("cutoffs must contain positive integers")

    totals = {
        f"hit@{cutoff}": 0.0 for cutoff in normalized_cutoffs
    } | {f"recall@{cutoff}": 0.0 for cutoff in normalized_cutoffs}
    reciprocal_rank = 0.0
    ndcg = 0.0
    item_hits: dict[int, Counter[str]] = {
        cutoff: Counter() for cutoff in normalized_cutoffs
    }
    item_totals: Counter[str] = Counter()
    bucket_hits: dict[int, Counter[str]] = {
        cutoff: Counter() for cutoff in normalized_cutoffs
    }
    bucket_totals: Counter[str] = Counter()

    def bucket(corpus_id: str) -> str:
        frequency = (training_frequencies or {}).get(corpus_id, 0)
        if frequency == 0:
            return "new"
        if frequency <= 4:
            return "tail_1_4"
        if frequency <= 19:
            return "tail_5_19"
        if frequency <= 99:
            return "mid_20_99"
        return "head_100_plus"

    for query_id, judgments in qrels.items():
        relevant = set(judgments)
        ranking = list(rankings.get(query_id, ()))
        for corpus_id in relevant:
            item_totals[corpus_id] += 1
            bucket_totals[bucket(corpus_id)] += 1
        for cutoff in normalized_cutoffs:
            retrieved = set(ranking[:cutoff])
            matched = relevant & retrieved
            totals[f"hit@{cutoff}"] += float(bool(matched))
            totals[f"recall@{cutoff}"] += len(matched) / len(relevant)
            for corpus_id in matched:
                item_hits[cutoff][corpus_id] += 1
                bucket_hits[cutoff][bucket(corpus_id)] += 1

        for rank, corpus_id in enumerate(ranking[:mrr_cutoff], start=1):
            if corpus_id in relevant:
                reciprocal_rank += 1.0 / rank
                break

        observed_relevance = [judgments.get(corpus_id, 0) for corpus_id in ranking[:ndcg_cutoff]]
        ideal_relevance = sorted(judgments.values(), reverse=True)[:ndcg_cutoff]
        ideal = _dcg(ideal_relevance)
        ndcg += _dcg(observed_relevance) / ideal if ideal else 0.0

    query_count = len(qrels)
    metrics = {name: value / query_count for name, value in totals.items()}
    metrics[f"mrr@{mrr_cutoff}"] = reciprocal_rank / query_count
    metrics[f"ndcg@{ndcg_cutoff}"] = ndcg / query_count

    macro_item_recall = {}
    bucket_recall = {}
    for cutoff in normalized_cutoffs:
        per_item = [
            item_hits[cutoff][corpus_id] / count
            for corpus_id, count in item_totals.items()
        ]
        macro_item_recall[f"macro_item_recall@{cutoff}"] = (
            sum(per_item) / len(per_item) if per_item else 0.0
        )
        bucket_recall[f"recall@{cutoff}"] = {
            name: bucket_hits[cutoff][name] / count
            for name, count in sorted(bucket_totals.items())
        }

    return {
        "query_count": query_count,
        "judged_knowledge_items": len(item_totals),
        "metrics": metrics,
        "macro_knowledge": macro_item_recall,
        "frequency_buckets": bucket_recall,
    }


def knowledge_frequencies(qrels: Mapping[str, Mapping[str, int]]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for judgments in qrels.values():
        frequencies.update(judgments)
    return frequencies
