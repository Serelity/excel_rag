"""Local query/case cross-encoding and calibrated, optional relevance filtering."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

from .common import file_hash

POLICY_VERSION = "case-relevance-policy-v1"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def score_profile(identity: dict) -> dict:
    return {key: value for key, value in identity.items() if key not in {"device", "batch_size"}}


def policy_context(report: dict) -> dict:
    return {
        "mode": report["mode"],
        "retriever": report["retriever"],
        "dataset": report["dataset_manifest_sha256"],
        "lexical_index": report["lexical_index_manifest_sha256"],
        "dense_index": report["dense_index_manifest_sha256"],
        "address_index": report["address_index_manifest_sha256"],
        "embedding_model": report["model_sha256"],
        "search_implementation": report["implementation_sha256"],
        "candidate_config": {
            key: report["config"][key]
            for key in (
                "input_field",
                "case_k",
                "max_terms",
                "rrf_k",
                "allow_broader",
            )
        },
        "reranker_profile": report["reranking"]["score_profile"],
    }


def policy_threshold(report: dict, policy: dict, top_k: int) -> float:
    if policy.get("version") != POLICY_VERSION:
        raise ValueError("Unsupported relevance policy")
    if type(policy.get("max_top_k")) is not int or not 0 < top_k <= policy["max_top_k"]:
        raise ValueError("Requested top_k exceeds the calibrated policy budget")
    setting = policy.get("modes", {}).get(report["mode"])
    if setting is None or setting.get("context") != policy_context(report):
        raise ValueError("Relevance policy model/index/mode/settings mismatch; recalibrate")
    threshold = setting.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(threshold)
    ):
        raise ValueError("Relevance policy threshold must be finite")
    return float(threshold)


def select_results(report: dict, *, top_k: int, policy: dict | None = None) -> dict:
    """Select a prefix of an already scored ranking; never refill rejected results."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    output = copy.deepcopy(report)
    rows = output["results"]
    threshold = policy_threshold(output, policy, top_k) if policy is not None and rows else None
    accepted = (
        rows
        if threshold is None
        else [row for row in rows if row["matching"]["reranker"]["score"] >= threshold]
    )
    output["results"] = accepted[:top_k]
    output["result_count"] = len(output["results"])
    output["config"]["top_k"] = top_k
    output["relevance_filter"] = {
        "enabled": policy is not None and bool(rows),
        "threshold": threshold,
        "skipped_reason": "no_candidates" if not rows else None,
        "policy_sha256": digest(policy) if policy is not None else None,
        "rejected_candidates": len(rows) - len(accepted),
        "score_is_probability": False,
    }
    output["result_status"] = (
        "no_candidates"
        if not rows
        else "below_relevance_threshold"
        if not accepted
        else "fewer_than_requested"
        if len(output["results"]) < top_k
        else "results_available"
    )
    for rank, row in enumerate(output["results"], 1):
        row["rank"] = rank
    return output


def rerank_report(report: dict, reranker, *, top_k: int, policy: dict | None = None) -> dict:
    if report["mode"] == "address":
        raise ValueError("Address-only queries do not use problem reranking")
    output = copy.deepcopy(report)
    rows = output["results"]
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate IDs")
    started = time.perf_counter()
    scores, stats = reranker.score(output["query"], [row["case_content"] for row in rows])
    if len(scores) != len(rows) or any(
        isinstance(value, bool) or not math.isfinite(float(value)) for value in scores
    ):
        raise ValueError("Reranker score count mismatch or non-finite score")
    for row, score in zip(rows, scores, strict=True):
        row["matching"]["reranker"] = {
            "score": float(score),
            "retrieval_rank": row["rank"],
        }
    rows.sort(key=lambda row: (-row["matching"]["reranker"]["score"], row["source_id"]))
    output["reranking"] = {
        "score_profile": score_profile(reranker.identity),
        "scored_candidates": len(rows),
        "stats": stats,
        "seconds_scoring": time.perf_counter() - started,
        "score_is_probability": False,
    }
    return select_results(output, top_k=top_k, policy=policy)


class BGEReranker:
    """BGE-reranker-v2-m3 raw logits, using only query and original case_content."""

    def __init__(
        self,
        model: Path,
        *,
        max_length: int = 1024,
        batch_size: int = 4,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        if not 16 <= max_length <= 8192 or batch_size < 1:
            raise ValueError("Require 16 <= reranker max_length <= 8192 and positive batch size")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported reranker dtype")
        if device == "cpu" and dtype != "float32":
            raise ValueError("CPU reranking requires float32")
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1"
        )
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from .prepare_reranker import verify_files

        verify_files(model)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from .encoder import model_fingerprint

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use a GPU task for reranking")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(model),
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=getattr(torch, dtype),
            )
            .to(device)
            .eval()
        )
        if self.model.config.num_labels != 1:
            raise ValueError("Expected one relevance logit per query/case pair")
        self.torch, self.device = torch, device
        self.max_length, self.batch_size = max_length, batch_size
        self.identity = {
            "profile": "bge-reranker-v2-m3-raw-logit-v1",
            "model": model_fingerprint(model),
            "max_length": max_length,
            "batch_size": batch_size,
            "device": device,
            "dtype": dtype,
            "input_fields": ["query", "case_content"],
            "truncation": "longest_first",
            "implementation_sha256": file_hash(Path(__file__)),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "tokenizers",
                )
            },
        }

    def score(self, query: str, documents: list[str]) -> tuple[list[float], dict]:
        scores, offset, batch_size = [], 0, self.batch_size
        stats = {"pairs": len(documents), "input_tokens": 0, "truncated_pairs": 0, "oom_retries": 0}
        while offset < len(documents):
            batch = documents[offset : offset + batch_size]
            queries = [query] * len(batch)
            lengths = self.tokenizer(
                queries, batch, truncation=False, padding=False, return_length=True
            )["length"]
            inputs = self.tokenizer(
                queries,
                batch,
                padding=True,
                truncation="longest_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
            try:
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                with self.torch.inference_mode():
                    logits = self.model(**inputs).logits
                    if tuple(logits.shape) != (len(batch), 1):
                        raise ValueError("Unexpected reranker output shape")
                    values = logits[:, 0].float().cpu().tolist()
                scores.extend(values)
                stats["input_tokens"] += sum(lengths)
                stats["truncated_pairs"] += sum(n > self.max_length for n in lengths)
                offset += len(batch)
            except self.torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise RuntimeError("Reranker OOM at batch size 1; lower max_length") from None
                batch_size = max(1, batch_size // 2)
                stats["oom_retries"] += 1
                inputs = None
                self.torch.cuda.empty_cache()
        return scores, stats
