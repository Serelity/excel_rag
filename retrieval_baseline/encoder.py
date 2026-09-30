"""Local-only BGE-M3 dense encoding. No extraction or metadata enters the model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from .common import file_hash


def model_fingerprint(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"Local embedding model not found: {path}")
    names = sorted(
        p for p in path.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
        and p.suffix in {".json", ".txt", ".model", ".safetensors", ".bin"}
    )
    if not names:
        raise ValueError("Local model contains no configuration/tokenizer/weight files")
    files = [{"name": p.relative_to(path).as_posix(), "bytes": p.stat().st_size,
              "sha256": file_hash(p)} for p in names]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"algorithm": "local-embedding-files-v1", "sha256": digest, "files": files}


def normalized_vectors(vectors, count: int, dimension: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise ValueError("Embedding shape mismatch or non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 1e-8).any():
        raise ValueError("Zero embedding vector")
    return np.ascontiguousarray(vectors / norms)


class BGEEncoder:
    def __init__(self, model: Path, *, max_length: int, batch_size: int,
                 device: str, dtype: str):
        if not 16 <= max_length <= 8192 or batch_size < 1:
            raise ValueError("Require 16 <= max_length <= 8192 and positive batch_size")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported dtype")
        if device == "cpu" and dtype != "float32":
            raise ValueError("Use float32 for CPU encoding")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        pooling = json.loads((model / "1_Pooling/config.json").read_text(encoding="utf-8"))
        if (
            not (model / "modules.json").is_file()
            or config.get("model_type") != "xlm-roberta"
            or not pooling.get("pooling_mode_cls_token")
            or pooling.get("pooling_mode_mean_tokens")
        ):
            raise ValueError("This profile requires a complete BGE-M3 CLS-pooling snapshot")
        import torch
        from sentence_transformers import SentenceTransformer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run on a GPU node or choose cpu/float32")
        self.model = SentenceTransformer(
            str(model), device=device, local_files_only=True, trust_remote_code=False,
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
        )
        if self.model.get_sentence_embedding_dimension() != 1024:
            raise ValueError("Expected 1024-dimensional BGE-M3 dense embeddings")
        self.model.max_seq_length = max_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.torch = torch
        self.identity = {
            "profile": "bge-m3-dense-cls-v1", "dimension": 1024,
            "max_length": max_length, "batch_size": batch_size,
            "device": device, "dtype": dtype, "prompt": "", "normalized": True,
            "model": model_fingerprint(model),
            "packages": {p: importlib.metadata.version(p) for p in (
                "torch", "transformers", "sentence-transformers", "numpy",
            )},
        }

    def encode(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        lengths = self.model.tokenizer(
            texts, truncation=False, padding=False, return_length=True,
        )["length"]
        retries, batch_size = 0, min(self.batch_size, len(texts))
        while True:
            try:
                vectors = self.model.encode(
                    texts, batch_size=batch_size, prompt="", show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True,
                )
                break
            except self.torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    raise RuntimeError(
                        "GPU OOM at batch_size=1; use a new run with shorter max_length"
                    ) from None
                self.torch.cuda.empty_cache()
                batch_size //= 2
                retries += 1
        stats = {
            "records": len(texts), "truncated_records": sum(n > self.max_length for n in lengths),
            "input_tokens": sum(lengths), "oom_retries": retries,
        }
        return normalized_vectors(vectors, len(texts), 1024), stats
