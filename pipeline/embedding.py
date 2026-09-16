"""Dense-only BGE-M3 embedding helpers.

FlagEmbedding is imported only when a model is instantiated so this module can
be imported by lightweight tooling and tests without CUDA or ML dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 512
DEFAULT_NORMALIZE_EMBEDDINGS = True


def _load_model_class():
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError(
            "BGE-M3 embedding requires FlagEmbedding. Install the locked "
            "embedding dependencies before running the indexing CLI."
        ) from exc
    return BGEM3FlagModel


def _load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Pinned Hugging Face revisions require huggingface-hub. Install "
            "the locked embedding dependencies before running the indexing CLI."
        ) from exc
    return snapshot_download


def _resolve_model_source(model_name: str, revision: str | None) -> tuple[str, str | None]:
    local_path = Path(model_name).expanduser()
    if local_path.exists() or revision is None:
        return model_name, None

    snapshot_download = _load_snapshot_download()
    try:
        snapshot_path = Path(snapshot_download(repo_id=model_name, revision=revision)).resolve()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to resolve embedding model revision {revision!r} ({type(exc).__name__})"
        ) from None

    if snapshot_path.parent.name != "snapshots" or not snapshot_path.name:
        raise RuntimeError(
            "Unable to determine the immutable commit for the downloaded model snapshot"
        )
    return str(snapshot_path), snapshot_path.name


class BGEM3Embedder:
    """Small adapter exposing only BGE-M3 dense embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        use_fp16: bool = True,
        normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
        device: str | Sequence[str] | None = None,
        model: Any | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("embedding model_name must be a non-empty string")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("embedding revision must be a non-empty string or null")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("embedding batch_size must be greater than zero")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("embedding max_length must be greater than zero")
        if not isinstance(use_fp16, bool):
            raise ValueError("embedding use_fp16 must be a boolean")
        if not isinstance(normalize_embeddings, bool):
            raise ValueError("embedding normalize_embeddings must be a boolean")

        self.model_name = model_name.strip()
        self.revision = revision.strip() if revision is not None else None
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.normalize_embeddings = normalize_embeddings
        self.loaded_model_name = self.model_name
        self.resolved_revision: str | None = None

        if model is not None:
            self._model = model
            return

        self.loaded_model_name, self.resolved_revision = _resolve_model_source(
            self.model_name,
            self.revision,
        )
        model_class = _load_model_class()
        model_kwargs: dict[str, Any] = {
            "use_fp16": use_fp16,
            "normalize_embeddings": normalize_embeddings,
        }
        if device is not None:
            # FlagEmbedding calls this option ``devices`` even for one device.
            model_kwargs["devices"] = device

        try:
            self._model = model_class(self.loaded_model_name, **model_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load BGE-M3 model from {self.model_name!r}: {type(exc).__name__}"
            ) from None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode one bounded batch and return dense vectors."""

        text_batch = list(texts)
        if not text_batch:
            return []
        if any(not isinstance(text, str) for text in text_batch):
            raise TypeError("all embedding inputs must be strings")

        result = self._model.encode(
            text_batch,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        if not isinstance(result, Mapping) or "dense_vecs" not in result:
            raise RuntimeError("BGE-M3 did not return the expected dense_vecs output")

        dense_vectors = result["dense_vecs"]
        if hasattr(dense_vectors, "tolist"):
            dense_vectors = dense_vectors.tolist()

        vectors = [list(vector) for vector in dense_vectors]
        if len(vectors) != len(text_batch):
            raise RuntimeError("BGE-M3 returned a different number of vectors than input texts")
        return vectors


def embed(
    chunks: Sequence[str | Mapping[str, Any]],
    *,
    embedder: BGEM3Embedder | None = None,
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    use_fp16: bool = True,
    normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
    device: str | Sequence[str] | None = None,
) -> list[list[float]]:
    """Embed a bounded batch of strings or chunk dictionaries."""

    texts: list[str] = []
    for position, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            texts.append(chunk)
            continue
        if not isinstance(chunk, Mapping) or not isinstance(chunk.get("text"), str):
            raise ValueError(f"chunk {position} must contain a string 'text' field")
        texts.append(chunk["text"])

    active_embedder = embedder or BGEM3Embedder(
        model_name,
        revision=revision,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
        normalize_embeddings=normalize_embeddings,
        device=device,
    )
    return active_embedder.encode(texts)
