import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from index_chunks import (
    _environment_api_key,
    _loaded_model_revision,
    _resolve_model_revision,
    _verify_normalization,
    build_parser,
    run,
)
from pipeline.embedding import BGEM3Embedder, embed
from pipeline.indexer import (
    build_embedding_contract,
    create_qdrant_client,
    index,
    iter_jsonl,
    stable_point_id,
    upsert_chunks,
)


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {"dense_vecs": [[float(len(text)), 0.0, 1.0] for text in texts]}


class FakePointStruct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeVectorParams:
    def __init__(self, *, size, distance):
        self.size = size
        self.distance = distance


class FakeDistance:
    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"
    MANHATTAN = "Manhattan"


FAKE_MODELS = SimpleNamespace(
    Distance=FakeDistance,
    VectorParams=FakeVectorParams,
    PointStruct=FakePointStruct,
)


class FakeQdrantClient:
    def __init__(
        self,
        *,
        exists=False,
        existing_payload=None,
        upsert_errors=None,
    ):
        self.exists = exists
        self.existing_payload = existing_payload
        self.upsert_errors = list(upsert_errors or [])
        self.params = FakeVectorParams(size=3, distance=FakeDistance.COSINE)
        self.events = []
        self.deleted = []
        self.created = []
        self.scroll_calls = []
        self.upsert_calls = []
        self.successful_upserts = []

    def collection_exists(self, *, collection_name):
        self.events.append(("collection_exists", collection_name))
        return self.exists

    def delete_collection(self, *, collection_name):
        self.events.append(("delete", collection_name))
        self.deleted.append(collection_name)
        self.exists = False

    def create_collection(self, *, collection_name, vectors_config):
        self.events.append(("create", collection_name))
        self.created.append((collection_name, vectors_config))
        self.params = vectors_config
        self.exists = True

    def get_collection(self, *, collection_name):
        self.events.append(("get_collection", collection_name))
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self.params)))

    def scroll(self, **kwargs):
        self.events.append(("scroll", kwargs["collection_name"]))
        self.scroll_calls.append(kwargs)
        if self.existing_payload is None:
            return [], None
        return [SimpleNamespace(payload=self.existing_payload)], None

    def upsert(self, *, collection_name, points, wait):
        call = (collection_name, points, wait)
        self.events.append(("upsert", collection_name))
        self.upsert_calls.append(call)
        if self.upsert_errors:
            error = self.upsert_errors.pop(0)
            if error is not None:
                raise error
        self.successful_upserts.append(call)


def make_chunk(number=0, *, text=None):
    source_id = f"source-{number}"
    return {
        "schema_version": "1.0.0",
        "id": source_id,
        "type": "problem",
        "text": text if text is not None else f"problem text {number}",
        "source_text_hash": hashlib.sha256(f"raw source text {number}".encode()).hexdigest(),
        "extraction_run_id": "run-20260915",
        "extraction": {
            "model": "test-extractor",
            "prompt_version": "problem-v1",
        },
        "metadata": {
            "source_id": source_id,
            "city": "Shanghai",
            "district": "Pudong",
            "category1": "environment",
            "category2": "noise",
            "category3": "construction",
            "create_time": "2026-09-15T08:00:00+08:00",
            "problem_type": "construction noise",
            "verification_status": "candidate",
            "template_version": "problem-v1",
        },
    }


def make_embedding_contract():
    return build_embedding_contract(
        model="BAAI/bge-m3",
        revision="0123456789abcdef",
        max_length=512,
        normalize_embeddings=True,
    )


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def make_loaded_embedder(
    *,
    model_revision=None,
    tokenizer_revision=None,
    normalize_embeddings=True,
):
    backend = SimpleNamespace(normalize_embeddings=normalize_embeddings)
    backend.model = SimpleNamespace(config=SimpleNamespace(_commit_hash=model_revision))
    backend.tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": tokenizer_revision})
    return SimpleNamespace(_model=backend)


def make_cli_config(*, revision="resolved-commit"):
    return {
        "data": {},
        "embedding": {
            "model": "BAAI/bge-m3",
            "revision": revision,
            "device": "cuda:0",
            "use_fp16": True,
            "normalize_embeddings": True,
            "batch_size": 17,
            "max_length": 768,
        },
        "vectorstore": {
            "url": None,
            "host": "127.0.0.1",
            "port": 6333,
            "https": False,
            "api_key_env": None,
            "timeout_seconds": 12.5,
            "collection": "problem-chunks",
            "vector_size": 3,
            "distance": "cosine",
            "upsert_batch_size": 23,
            "upsert_max_attempts": 4,
            "upsert_retry_backoff_seconds": 0.25,
        },
    }


class IndexingTests(unittest.TestCase):
    def test_empty_qdrant_api_key_is_treated_as_unset(self):
        with patch.dict(
            "index_chunks.os.environ",
            {
                "EMPTY_QDRANT_API_KEY": "",
                "BLANK_QDRANT_API_KEY": "  ",
                "SET_QDRANT_API_KEY": " secret ",
            },
            clear=True,
        ):
            self.assertIsNone(_environment_api_key(None))
            self.assertIsNone(_environment_api_key("MISSING_QDRANT_API_KEY"))
            self.assertIsNone(_environment_api_key("EMPTY_QDRANT_API_KEY"))
            self.assertIsNone(_environment_api_key("BLANK_QDRANT_API_KEY"))
            self.assertEqual(_environment_api_key("SET_QDRANT_API_KEY"), "secret")

    def test_embedder_forwards_model_identity_and_vector_settings(self):
        captured = {}

        class CapturingEmbeddingModel:
            def __init__(self, model_name, **kwargs):
                captured["model_name"] = model_name
                captured["kwargs"] = kwargs

        with tempfile.TemporaryDirectory() as directory:
            snapshot_path = Path(directory) / "models--BAAI--bge-m3" / "snapshots" / "commit-abc"
            snapshot_path.mkdir(parents=True)
            with (
                patch(
                    "pipeline.embedding._load_model_class",
                    return_value=CapturingEmbeddingModel,
                ),
                patch(
                    "pipeline.embedding._load_snapshot_download",
                    return_value=Mock(return_value=str(snapshot_path)),
                ) as load_snapshot_download,
            ):
                embedder = BGEM3Embedder(
                    " BAAI/bge-m3 ",
                    revision=" commit-abc ",
                    batch_size=16,
                    max_length=1024,
                    use_fp16=False,
                    normalize_embeddings=False,
                    device="cuda:0",
                )

        load_snapshot_download.return_value.assert_called_once_with(
            repo_id="BAAI/bge-m3",
            revision="commit-abc",
        )
        self.assertEqual(captured["model_name"], str(snapshot_path.resolve()))
        self.assertEqual(
            captured["kwargs"],
            {
                "use_fp16": False,
                "normalize_embeddings": False,
                "devices": "cuda:0",
            },
        )
        self.assertEqual(embedder.revision, "commit-abc")
        self.assertEqual(embedder.resolved_revision, "commit-abc")
        self.assertFalse(embedder.normalize_embeddings)

    def test_embedder_requests_dense_vectors_only(self):
        model = FakeEmbeddingModel()
        embedder = BGEM3Embedder(model=model, batch_size=8, max_length=128)

        vectors = embed([{"text": "alpha"}, {"text": "beta"}], embedder=embedder)

        self.assertEqual(vectors, [[5.0, 0.0, 1.0], [4.0, 0.0, 1.0]])
        _, kwargs = model.calls[0]
        self.assertTrue(kwargs["return_dense"])
        self.assertFalse(kwargs["return_sparse"])
        self.assertFalse(kwargs["return_colbert_vecs"])
        self.assertEqual(kwargs["batch_size"], 8)
        self.assertEqual(kwargs["max_length"], 128)

    def test_jsonl_streams_complete_strict_schema_record(self):
        chunk = make_chunk()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            write_jsonl(path, [chunk])

            records = iter_jsonl(path)

            self.assertNotIsInstance(records, list)
            self.assertEqual(next(records), chunk)
            with self.assertRaises(StopIteration):
                next(records)

    def test_candidate_status_is_required_and_validation_is_sanitized(self):
        secret_text = "private resident complaint that must never appear in errors"
        chunk = make_chunk(text=secret_text)
        chunk["metadata"]["verification_status"] = "verified"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-candidate.jsonl"
            write_jsonl(path, [chunk])

            with self.assertRaises(ValueError) as raised:
                list(iter_jsonl(path))

        message = str(raised.exception)
        self.assertIn("metadata.verification_status", message)
        self.assertIn("candidate", message)
        self.assertNotIn(secret_text, message)

    def test_schema_rejects_missing_extra_and_wrong_type_fields(self):
        cases = []

        missing = make_chunk()
        del missing["source_text_hash"]
        cases.append(("missing", missing, "source_text_hash"))

        extra = make_chunk()
        extra["unexpected"] = "forbidden"
        cases.append(("extra", extra, "unexpected"))

        wrong_type = make_chunk()
        wrong_type["metadata"]["city"] = 7
        cases.append(("wrong-type", wrong_type, "metadata.city"))

        mismatched_id = make_chunk()
        mismatched_id["metadata"]["source_id"] = "different-source"
        cases.append(("mismatched-id", mismatched_id, "metadata.source_id"))

        with tempfile.TemporaryDirectory() as directory:
            for name, chunk, expected_location in cases:
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.jsonl"
                    write_jsonl(path, [chunk])

                    with self.assertRaises(ValueError) as raised:
                        list(iter_jsonl(path))

                    self.assertIn(expected_location, str(raised.exception))

    def test_empty_and_invalid_preflight_have_no_qdrant_or_embedding_side_effects(self):
        valid_chunk = make_chunk(text="sensitive text before malformed input")
        invalid_schema = copy.deepcopy(valid_chunk)
        invalid_schema["metadata"]["verification_status"] = "reviewed"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "empty": "\n",
                "invalid-json": json.dumps(valid_chunk) + "\n{" + "\n",
                "invalid-schema": json.dumps(invalid_schema) + "\n",
            }
            for name, contents in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.jsonl"
                    path.write_text(contents, encoding="utf-8")
                    model = FakeEmbeddingModel()
                    embedder = BGEM3Embedder(model=model)
                    client = FakeQdrantClient(exists=True)

                    with (
                        patch(
                            "pipeline.indexer._load_qdrant_models",
                            return_value=FAKE_MODELS,
                        ) as load_models,
                        self.assertRaises(ValueError),
                    ):
                        index(
                            path,
                            embedder=embedder,
                            client=client,
                            embedding_contract=make_embedding_contract(),
                            collection_name="problems",
                            vector_size=3,
                            recreate=True,
                        )

                    load_models.assert_not_called()
                    self.assertEqual(client.events, [])
                    self.assertEqual(client.deleted, [])
                    self.assertEqual(client.created, [])
                    self.assertEqual(model.calls, [])

    def test_direct_upsert_revalidates_problem_chunk_before_embedding(self):
        invalid_chunk = make_chunk(text="private text must not be in the error")
        del invalid_chunk["extraction"]["prompt_version"]
        client = FakeQdrantClient()
        model = FakeEmbeddingModel()

        with self.assertRaises(ValueError) as raised:
            upsert_chunks(
                [invalid_chunk],
                embedder=BGEM3Embedder(model=model),
                client=client,
                embedding_contract=make_embedding_contract(),
                collection_name="problems",
                vector_size=3,
                point_factory=FakePointStruct,
            )

        self.assertIn("extraction.prompt_version", str(raised.exception))
        self.assertNotIn(invalid_chunk["text"], str(raised.exception))
        self.assertEqual(model.calls, [])
        self.assertEqual(client.events, [])

    def test_inconsistent_embedding_contract_is_rejected_before_side_effects(self):
        contract = make_embedding_contract()
        contract["revision"] = "changed-without-updating-the-fingerprint"
        client = FakeQdrantClient(exists=True)
        model = FakeEmbeddingModel()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            write_jsonl(path, [make_chunk()])

            with (
                patch(
                    "pipeline.indexer._load_qdrant_models",
                    return_value=FAKE_MODELS,
                ) as load_models,
                self.assertRaisesRegex(ValueError, "fingerprint or settings"),
            ):
                index(
                    path,
                    embedder=BGEM3Embedder(model=model),
                    client=client,
                    embedding_contract=contract,
                    collection_name="problems",
                    vector_size=3,
                    recreate=True,
                )

        load_models.assert_not_called()
        self.assertEqual(model.calls, [])
        self.assertEqual(client.events, [])

    def test_index_batches_upserts_and_preserves_the_full_payload(self):
        chunks = [make_chunk(number) for number in range(5)]
        contract = make_embedding_contract()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            write_jsonl(path, chunks)
            model = FakeEmbeddingModel()
            embedder = BGEM3Embedder(model=model, batch_size=2, max_length=128)
            client = FakeQdrantClient()

            with patch("pipeline.indexer._load_qdrant_models", return_value=FAKE_MODELS):
                indexed = index(
                    path,
                    embedder=embedder,
                    client=client,
                    embedding_contract=contract,
                    collection_name="problems",
                    vector_size=3,
                    upsert_batch_size=2,
                )

        self.assertEqual(indexed, 5)
        self.assertEqual(
            [len(call[1]) for call in client.successful_upserts],
            [2, 2, 1],
        )
        self.assertEqual([len(call[0]) for call in model.calls], [2, 2, 1])
        first_point = client.successful_upserts[0][1][0]
        expected_payload = {
            "schema_version": chunks[0]["schema_version"],
            "source_id": chunks[0]["id"],
            "type": chunks[0]["type"],
            "text": chunks[0]["text"],
            "source_text_hash": chunks[0]["source_text_hash"],
            "extraction_run_id": chunks[0]["extraction_run_id"],
            "extraction": chunks[0]["extraction"],
            "metadata": chunks[0]["metadata"],
            "embedding": contract,
        }
        self.assertEqual(first_point.id, stable_point_id(chunks[0]["id"]))
        self.assertEqual(first_point.payload, expected_payload)
        self.assertNotIn("id", first_point.payload)

    def test_valid_preflight_allows_explicit_collection_recreation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            write_jsonl(path, [make_chunk()])
            model = FakeEmbeddingModel()
            client = FakeQdrantClient(exists=True)

            with patch("pipeline.indexer._load_qdrant_models", return_value=FAKE_MODELS):
                index(
                    path,
                    embedder=BGEM3Embedder(model=model),
                    client=client,
                    embedding_contract=make_embedding_contract(),
                    collection_name="problems",
                    vector_size=3,
                    recreate=True,
                )

        self.assertEqual(client.deleted, ["problems"])
        self.assertEqual(len(client.created), 1)
        self.assertEqual(len(model.calls), 1)

    def test_existing_collection_embedding_fingerprint_mismatch_is_rejected(self):
        expected_contract = make_embedding_contract()
        stale_fingerprint = "0" * 64
        client = FakeQdrantClient(
            exists=True,
            existing_payload={"embedding": {"fingerprint": stale_fingerprint}},
        )
        model = FakeEmbeddingModel()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            write_jsonl(path, [make_chunk()])

            with (
                patch("pipeline.indexer._load_qdrant_models", return_value=FAKE_MODELS),
                self.assertRaises(ValueError) as raised,
            ):
                index(
                    path,
                    embedder=BGEM3Embedder(model=model),
                    client=client,
                    embedding_contract=expected_contract,
                    collection_name="problems",
                    vector_size=3,
                )

        message = str(raised.exception)
        self.assertIn(stale_fingerprint, message)
        self.assertIn(expected_contract["fingerprint"], message)
        self.assertEqual(model.calls, [])
        self.assertEqual(client.deleted, [])
        self.assertEqual(client.created, [])
        self.assertEqual(client.upsert_calls, [])
        self.assertEqual(
            client.scroll_calls,
            [
                {
                    "collection_name": "problems",
                    "limit": 1,
                    "with_payload": ["embedding"],
                    "with_vectors": False,
                }
            ],
        )

    def test_transient_upsert_failures_retry_with_exponential_backoff(self):
        client = FakeQdrantClient(
            upsert_errors=[TimeoutError("first failure"), ConnectionError("second failure")]
        )
        model = FakeEmbeddingModel()
        delays = []

        indexed = upsert_chunks(
            [make_chunk()],
            embedder=BGEM3Embedder(model=model),
            client=client,
            embedding_contract=make_embedding_contract(),
            collection_name="problems",
            vector_size=3,
            max_attempts=3,
            retry_backoff_seconds=0.25,
            point_factory=FakePointStruct,
            sleep=delays.append,
        )

        self.assertEqual(indexed, 1)
        self.assertEqual(len(client.upsert_calls), 3)
        self.assertEqual(len(client.successful_upserts), 1)
        self.assertEqual(delays, [0.25, 0.5])
        self.assertEqual(len(model.calls), 1)

    def test_permanent_upsert_error_is_not_retried_and_does_not_leak_text(self):
        secret_text = "resident phone 13800000000 and private complaint details"
        chunks = [make_chunk(7, text=secret_text), make_chunk(8)]
        client = FakeQdrantClient(upsert_errors=[ValueError(secret_text)])
        delays = []

        with self.assertRaises(RuntimeError) as raised:
            upsert_chunks(
                chunks,
                embedder=BGEM3Embedder(model=FakeEmbeddingModel()),
                client=client,
                embedding_contract=make_embedding_contract(),
                collection_name="problems",
                vector_size=3,
                max_attempts=3,
                retry_backoff_seconds=0.1,
                point_factory=FakePointStruct,
                sleep=delays.append,
            )

        message = str(raised.exception)
        self.assertIn("batch 1", message)
        self.assertIn("source-7", message)
        self.assertIn("source-8", message)
        self.assertIn("ValueError", message)
        self.assertNotIn(secret_text, message)
        self.assertEqual(len(client.upsert_calls), 1)
        self.assertEqual(delays, [])

    def test_exhausted_transient_error_is_bounded_and_does_not_leak_text(self):
        secret_text = "private complaint body from failed request"
        client = FakeQdrantClient(
            upsert_errors=[TimeoutError(secret_text), TimeoutError(secret_text)]
        )
        delays = []

        with self.assertRaises(RuntimeError) as raised:
            upsert_chunks(
                [make_chunk(3, text=secret_text)],
                embedder=BGEM3Embedder(model=FakeEmbeddingModel()),
                client=client,
                embedding_contract=make_embedding_contract(),
                collection_name="problems",
                vector_size=3,
                max_attempts=2,
                retry_backoff_seconds=0.1,
                point_factory=FakePointStruct,
                sleep=delays.append,
            )

        message = str(raised.exception)
        self.assertIn("batch 1", message)
        self.assertIn("source-3", message)
        self.assertIn("after 2 attempt(s)", message)
        self.assertIn("TimeoutError", message)
        self.assertNotIn(secret_text, message)
        self.assertEqual(len(client.upsert_calls), 2)
        self.assertEqual(delays, [0.1])

    def test_qdrant_url_api_key_https_and_timeout_are_forwarded(self):
        captured = {}

        class CapturingClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        api_key = "qdrant-test-secret"
        with patch(
            "pipeline.indexer._load_qdrant_client_class",
            return_value=CapturingClient,
        ):
            client = create_qdrant_client(
                host="ignored-host",
                port=9999,
                url=" https://qdrant.internal:6333 ",
                api_key=api_key,
                https=True,
                timeout=12.5,
            )

        self.assertIsInstance(client, CapturingClient)
        self.assertEqual(
            captured,
            {
                "url": "https://qdrant.internal:6333",
                "api_key": api_key,
                "https": True,
                "timeout": 12.5,
            },
        )
        self.assertNotIn("host", captured)
        self.assertNotIn("port", captured)

    def test_qdrant_https_is_optional_and_nonfinite_timeout_is_rejected(self):
        captured = {}

        class CapturingClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch(
            "pipeline.indexer._load_qdrant_client_class",
            return_value=CapturingClient,
        ):
            create_qdrant_client(host="127.0.0.1", port=6333, https=None)

        self.assertNotIn("https", captured)
        with self.assertRaises(ValueError):
            create_qdrant_client(timeout=float("nan"))

    def test_cli_wires_embedding_identity_and_qdrant_security_options(self):
        revision = "resolved-commit"
        loaded_embedder = make_loaded_embedder(
            model_revision=revision,
            tokenizer_revision=revision,
            normalize_embeddings=False,
        )
        client = SimpleNamespace(close=Mock())
        config = {
            "data": {},
            "embedding": {
                "model": "BAAI/bge-m3",
                "revision": revision,
                "device": "cuda:0",
                "use_fp16": True,
                "normalize_embeddings": False,
                "batch_size": 17,
                "max_length": 768,
            },
            "vectorstore": {
                "url": "https://qdrant.internal:6333",
                "host": "ignored-host",
                "port": 7443,
                "https": True,
                "api_key_env": "TEST_QDRANT_API_KEY",
                "timeout_seconds": 12.5,
                "collection": "problem-chunks",
                "vector_size": 3,
                "distance": "cosine",
                "upsert_batch_size": 23,
                "upsert_max_attempts": 4,
                "upsert_retry_backoff_seconds": 0.25,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "chunks.jsonl"
            write_jsonl(input_path, [make_chunk()])
            args = build_parser().parse_args(
                ["--config", "unused.yaml", "--input", str(input_path), "--no-fp16"]
            )
            with (
                patch("index_chunks.load_config", return_value=config),
                patch(
                    "index_chunks.BGEM3Embedder",
                    return_value=loaded_embedder,
                ) as embedder_class,
                patch(
                    "index_chunks.create_qdrant_client",
                    return_value=client,
                ) as client_factory,
                patch("index_chunks.index", return_value=1) as index_call,
                patch.dict(
                    "index_chunks.os.environ",
                    {"TEST_QDRANT_API_KEY": "test-secret"},
                    clear=True,
                ),
            ):
                summary = run(args)

        embedder_class.assert_called_once_with(
            "BAAI/bge-m3",
            revision=revision,
            batch_size=17,
            max_length=768,
            use_fp16=False,
            normalize_embeddings=False,
            device="cuda:0",
        )
        client_factory.assert_called_once_with(
            "ignored-host",
            7443,
            url="https://qdrant.internal:6333",
            api_key="test-secret",
            https=True,
            timeout=12.5,
        )
        index_kwargs = index_call.call_args.kwargs
        self.assertEqual(index_kwargs["max_attempts"], 4)
        self.assertEqual(index_kwargs["retry_backoff_seconds"], 0.25)
        self.assertFalse(index_kwargs["embedding_contract"]["use_fp16"])
        self.assertFalse(index_kwargs["embedding_contract"]["normalize_embeddings"])
        self.assertEqual(
            summary["embedding_fingerprint"],
            index_kwargs["embedding_contract"]["fingerprint"],
        )
        client.close.assert_called_once_with()

    def test_cli_rejects_lossy_numbers_and_non_text_values_before_model_load(self):
        cases = [
            ("embedding", "batch_size", True),
            ("embedding", "max_length", 768.5),
            ("embedding", "device", ["cuda:0"]),
            ("vectorstore", "host", False),
            ("vectorstore", "port", "6333"),
            ("vectorstore", "collection", 42),
            ("vectorstore", "vector_size", 3.5),
            ("vectorstore", "distance", True),
            ("vectorstore", "upsert_batch_size", 23.5),
            ("vectorstore", "upsert_max_attempts", False),
            ("vectorstore", "timeout_seconds", True),
            ("vectorstore", "upsert_retry_backoff_seconds", "0.25"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "chunks.jsonl"
            write_jsonl(input_path, [make_chunk()])
            args = build_parser().parse_args(
                ["--config", "unused.yaml", "--input", str(input_path)]
            )

            for section, key, invalid_value in cases:
                with self.subTest(field=f"{section}.{key}", value=invalid_value):
                    config = make_cli_config()
                    config[section][key] = invalid_value
                    with (
                        patch("index_chunks.load_config", return_value=config),
                        patch("index_chunks.BGEM3Embedder") as embedder_class,
                        self.assertRaises(ValueError) as raised,
                    ):
                        run(args)

                    self.assertIn(f"{section}.{key}", str(raised.exception))
                    embedder_class.assert_not_called()

    def test_primary_index_error_survives_client_close_error_without_leak(self):
        revision = "resolved-commit"
        primary_error = RuntimeError("sanitized primary indexing failure")
        close_secret = "private close transport details"
        client = SimpleNamespace(close=Mock(side_effect=ValueError(close_secret)))
        loaded_embedder = make_loaded_embedder(
            model_revision=revision,
            tokenizer_revision=revision,
            normalize_embeddings=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "chunks.jsonl"
            write_jsonl(input_path, [make_chunk()])
            args = build_parser().parse_args(
                ["--config", "unused.yaml", "--input", str(input_path)]
            )
            with (
                patch("index_chunks.load_config", return_value=make_cli_config()),
                patch("index_chunks.BGEM3Embedder", return_value=loaded_embedder),
                patch("index_chunks.create_qdrant_client", return_value=client),
                patch("index_chunks.index", side_effect=primary_error),
                self.assertRaises(RuntimeError) as raised,
            ):
                run(args)

        self.assertIs(raised.exception, primary_error)
        self.assertEqual(str(raised.exception), "sanitized primary indexing failure")
        self.assertNotIn(close_secret, str(raised.exception))
        client.close.assert_called_once_with()

    def test_standalone_client_close_error_is_sanitized(self):
        revision = "resolved-commit"
        close_secret = "private close transport details"
        client = SimpleNamespace(close=Mock(side_effect=ValueError(close_secret)))
        loaded_embedder = make_loaded_embedder(
            model_revision=revision,
            tokenizer_revision=revision,
            normalize_embeddings=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "chunks.jsonl"
            write_jsonl(input_path, [make_chunk()])
            args = build_parser().parse_args(
                ["--config", "unused.yaml", "--input", str(input_path)]
            )
            with (
                patch("index_chunks.load_config", return_value=make_cli_config()),
                patch("index_chunks.BGEM3Embedder", return_value=loaded_embedder),
                patch("index_chunks.create_qdrant_client", return_value=client),
                patch("index_chunks.index", return_value=1),
                self.assertRaises(RuntimeError) as raised,
            ):
                run(args)

        message = str(raised.exception)
        self.assertIn("Unable to close Qdrant client (ValueError)", message)
        self.assertNotIn(close_secret, message)
        client.close.assert_called_once_with()

    def test_loaded_model_revision_uses_model_and_tokenizer_commit_hashes(self):
        revision = "abc123"
        self.assertEqual(
            _loaded_model_revision(
                make_loaded_embedder(
                    model_revision=revision,
                    tokenizer_revision=revision,
                )
            ),
            revision,
        )
        self.assertEqual(
            _loaded_model_revision(make_loaded_embedder(model_revision=revision)),
            revision,
        )
        with self.assertRaisesRegex(RuntimeError, "different revisions"):
            _loaded_model_revision(
                make_loaded_embedder(
                    model_revision="model-commit",
                    tokenizer_revision="tokenizer-commit",
                )
            )

    def test_model_revision_resolution_detects_drift_and_requires_identity(self):
        detected = make_loaded_embedder(model_revision="loaded-commit")
        undetectable = make_loaded_embedder()
        resolved_snapshot = make_loaded_embedder(model_revision="resolved-commit")
        resolved_snapshot.resolved_revision = "resolved-commit"

        self.assertEqual(_resolve_model_revision(detected, None), "loaded-commit")
        self.assertEqual(
            _resolve_model_revision(detected, "loaded-commit"),
            "loaded-commit",
        )
        self.assertEqual(
            _resolve_model_revision(undetectable, "explicit-local-revision"),
            "explicit-local-revision",
        )
        self.assertEqual(
            _resolve_model_revision(resolved_snapshot, "release-tag"),
            "resolved-commit",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _resolve_model_revision(detected, "configured-commit")
        with self.assertRaisesRegex(ValueError, "Unable to determine"):
            _resolve_model_revision(undetectable, None)

    def test_embedding_normalization_must_match_loaded_backend(self):
        _verify_normalization(
            make_loaded_embedder(normalize_embeddings=True),
            True,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _verify_normalization(
                make_loaded_embedder(normalize_embeddings=False),
                True,
            )
        with self.assertRaisesRegex(RuntimeError, "Unable to determine"):
            _verify_normalization(
                make_loaded_embedder(normalize_embeddings=None),
                True,
            )

    def test_embedding_contract_and_point_ids_are_stable(self):
        contract = make_embedding_contract()
        self.assertEqual(contract, make_embedding_contract())
        self.assertNotEqual(
            contract["fingerprint"],
            build_embedding_contract(
                model="BAAI/bge-m3",
                revision="different-revision",
                max_length=512,
                normalize_embeddings=True,
            )["fingerprint"],
        )
        self.assertNotEqual(
            contract["fingerprint"],
            build_embedding_contract(
                model="BAAI/bge-m3",
                revision="0123456789abcdef",
                max_length=512,
                normalize_embeddings=True,
                use_fp16=False,
            )["fingerprint"],
        )
        self.assertEqual(stable_point_id("42"), stable_point_id(42))
        self.assertNotEqual(stable_point_id("42"), stable_point_id("43"))
        with self.assertRaises(ValueError):
            stable_point_id("  ")


if __name__ == "__main__":
    unittest.main()
