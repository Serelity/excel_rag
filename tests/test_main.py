import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main
from pipeline.chunker import build_problem_chunk
from pipeline.extractor import ProblemExtractionError
from schemas.problem import Problem

TEST_CODE_COMMIT = "1" * 40


@pytest.fixture(autouse=True)
def code_commit(monkeypatch):
    monkeypatch.setenv("RAG_CODE_COMMIT", TEST_CODE_COMMIT)


class FakeExtractor:
    model = "fake-model"
    model_revision = None
    model_source_repo = None
    model_artifact_fingerprint = None
    prompt_version = "test-v1"
    enable_thinking = False
    temperature = 0.0
    max_tokens = 768
    max_input_chars = 15_000
    seed = 42
    fail_ids: set[str] = {"2"}
    calls: list[str] = []
    active = 0
    peak_active = 0
    closed = False

    def __init__(self, config):
        self.model = str(config.get("model", type(self).model))
        self.model_revision = config.get("model_revision", type(self).model_revision)
        self.model_source_repo = config.get("model_source_repo", type(self).model_source_repo)
        self.model_artifact_fingerprint = config.get(
            "model_artifact_fingerprint",
            type(self).model_artifact_fingerprint,
        )
        self.enable_thinking = config.get("enable_thinking", type(self).enable_thinking)
        self.temperature = float(config.get("temperature", type(self).temperature))
        self.max_tokens = int(config.get("max_tokens", type(self).max_tokens))
        self.max_input_chars = int(config.get("max_input_chars", type(self).max_input_chars))
        self.seed = config.get("seed", type(self).seed)
        type(self).calls = []
        type(self).active = 0
        type(self).peak_active = 0
        type(self).closed = False

    async def extract(self, ticket):
        type(self).calls.append(ticket.ticket_id)
        type(self).active += 1
        type(self).peak_active = max(type(self).peak_active, type(self).active)
        try:
            await asyncio.sleep(0)
            if ticket.ticket_id in type(self).fail_ids:
                raise RuntimeError("synthetic failure")
            return Problem(
                problem_type="噪声扰民",
                category=ticket.category_path,
                symptom=["夜间噪声"],
                impact=[],
                location_type="住宅区",
                keywords=["噪声"],
            )
        finally:
            type(self).active -= 1

    async def aclose(self):
        type(self).closed = True


def make_ticket(ticket_id: str, *, content: str = "夜间有施工噪声"):
    return SimpleNamespace(
        ticket_id=ticket_id,
        content=content,
        goal="请核实处理",
        category1="环境保护",
        category2="噪声污染",
        category3="施工噪声",
        category_path=["环境保护", "噪声污染", "施工噪声"],
        city="常州市",
        district="武进区",
        create_time="2024-06-11 20:51:18",
    )


def make_chunk(ticket_id: str) -> dict:
    ticket = make_ticket(ticket_id)
    problem = Problem(
        problem_type="噪声扰民",
        category=ticket.category_path,
        symptom=["夜间噪声"],
        impact=[],
        location_type="住宅区",
        keywords=["噪声"],
    )
    return build_problem_chunk(
        ticket,
        problem,
        extraction_run_id="run_test",
        model="fake-model",
        prompt_version="test-v1",
    )


def make_config(
    source: Path,
    output: Path,
    quarantine: Path,
    *,
    model: str = "fake-model",
) -> dict:
    return {
        "data": {
            "input": str(source),
            "output": str(output),
            "quarantine": str(quarantine),
        },
        "pipeline": {
            "concurrency": 2,
            "max_attempts": 2,
            "retry_backoff_seconds": 0,
            "checkpoint_every": 1,
            "log_every": 1,
        },
        "llm": {
            "model": model,
            "temperature": 0,
            "max_tokens": 768,
            "seed": 42,
        },
    }


def run_pipeline(
    config: dict,
    *,
    limit: int | None = None,
    resume: bool = False,
    retry_failures: bool = False,
):
    return asyncio.run(
        main.run_pipeline(
            config,
            limit=limit,
            resume=resume,
            overwrite=False,
            concurrency_override=None,
            retry_failures=retry_failures,
        )
    )


def test_cli_rejects_unbounded_direct_extraction(monkeypatch) -> None:
    monkeypatch.delenv("RAG_ALLOW_FULL_EXTRACTION", raising=False)

    with pytest.raises(SystemExit) as error:
        main.cli([])

    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [["--limit", "101"], ["--limit=101"]])
def test_cli_rejects_direct_limit_above_pilot_cap(monkeypatch, arguments) -> None:
    monkeypatch.delenv("RAG_ALLOW_FULL_EXTRACTION", raising=False)
    load_config = Mock()
    monkeypatch.setattr(main, "load_config", load_config)

    with pytest.raises(SystemExit) as error:
        main.cli(arguments)

    assert error.value.code == 2
    load_config.assert_not_called()


def test_incremental_limits_reach_one_twenty_and_one_hundred(tmp_path, monkeypatch) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    tickets = [make_ticket(str(index)) for index in range(1, 151)]
    monkeypatch.setattr(FakeExtractor, "fail_ids", set())
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter(tickets))

    first = run_pipeline(config, limit=1)
    second = run_pipeline(config, limit=19, resume=True)
    third = run_pipeline(config, limit=80, resume=True)

    assert (first.submitted, second.submitted, third.submitted) == (1, 19, 80)
    assert (first.skipped, second.skipped, third.skipped) == (0, 1, 20)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 100
    assert quarantine.read_text(encoding="utf-8") == ""


def jsonl_text(*records: dict) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )


def test_pipeline_is_bounded_and_quarantines_failures(tmp_path, monkeypatch) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(FakeExtractor, "fail_ids", {"2"})
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(
        main,
        "load_tickets",
        lambda _path: map(make_ticket, ["1", "2", "3"]),
    )

    stats = run_pipeline(config)

    chunks = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    errors = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    assert stats.succeeded == 2
    assert stats.failed == 1
    assert {chunk["id"] for chunk in chunks} == {"1", "3"}
    assert errors[0]["ticket_id"] == "2"
    assert errors[0]["error_code"] == "UNEXPECTED_ERROR"
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["attempts"] == 2
    assert FakeExtractor.calls.count("2") == 2
    assert FakeExtractor.peak_active <= 2
    assert FakeExtractor.closed is True


def test_quarantine_persists_stable_error_code_without_error_message(
    tmp_path,
    monkeypatch,
) -> None:
    class CodedFailureExtractor(FakeExtractor):
        async def extract(self, ticket):
            raise ProblemExtractionError(
                "sensitive model output must not be persisted",
                code="SENSITIVE_OUTPUT",
            )

    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", CodedFailureExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter([make_ticket("1")]))

    stats = run_pipeline(config)

    serialized = quarantine.read_text(encoding="utf-8")
    failure = json.loads(serialized)
    assert stats.failed == 1
    assert failure["error_code"] == "SENSITIVE_OUTPUT"
    assert failure["error_type"] == "ProblemExtractionError"
    assert "sensitive model output" not in serialized
    assert "夜间有施工噪声" not in serialized


def test_resume_repairs_incomplete_final_line_after_valid_chunks(tmp_path) -> None:
    output = tmp_path / "chunks.jsonl"
    expected = jsonl_text(make_chunk("a"), make_chunk("b"))
    output.write_bytes(expected.encode("utf-8") + b'{"schema_version":')

    assert main.read_completed_ids(output) == {"a", "b"}
    assert output.read_text(encoding="utf-8") == expected


def test_resume_rejects_newline_terminated_bad_final_line_without_modifying_file(
    tmp_path,
) -> None:
    output = tmp_path / "chunks.jsonl"
    original = (jsonl_text(make_chunk("a")) + "not-json\n").encode()
    output.write_bytes(original)

    with pytest.raises(ValueError, match="invalid JSONL"):
        main.read_completed_ids(output)

    assert output.read_bytes() == original


def test_resume_rejects_invalid_middle_line_without_modifying_file(tmp_path) -> None:
    output = tmp_path / "chunks.jsonl"
    original = (jsonl_text(make_chunk("a")) + "not-json\n" + jsonl_text(make_chunk("b"))).encode()
    output.write_bytes(original)

    with pytest.raises(ValueError, match="invalid JSONL"):
        main.read_completed_ids(output)

    assert output.read_bytes() == original


def test_resume_rejects_incomplete_or_incorrect_chunk_contract(tmp_path) -> None:
    wrong_source = copy.deepcopy(make_chunk("wrong-source"))
    wrong_source["metadata"]["source_id"] = "different-id"
    invalid_records = {
        "incomplete": {"id": "only-an-id"},
        "incorrect": wrong_source,
    }

    for name, record in invalid_records.items():
        output = tmp_path / f"{name}.jsonl"
        original = jsonl_text(record).encode()
        output.write_bytes(original)

        with pytest.raises(ValueError, match="invalid problem chunk contract"):
            main.read_completed_ids(output)

        assert output.read_bytes() == original


def test_resume_rejects_schema_valid_but_foreign_completed_source_hash(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(FakeExtractor, "fail_ids", set())
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter([make_ticket("1")]))
    run_pipeline(config)

    record = json.loads(output.read_text(encoding="utf-8"))
    record["source_text_hash"] = "0" * 64
    tampered = jsonl_text(record).encode()
    output.write_bytes(tampered)

    with pytest.raises(ValueError, match="completed chunk source hash differs"):
        run_pipeline(config, resume=True)

    assert output.read_bytes() == tampered


@pytest.mark.parametrize("drift", ["source_hash", "model", "prompt_version"])
def test_resume_rejects_quarantine_hash_or_provenance_drift(
    tmp_path,
    monkeypatch,
    drift,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(FakeExtractor, "fail_ids", {"2"})
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter([make_ticket("2")]))
    run_pipeline(config)

    record = json.loads(quarantine.read_text(encoding="utf-8"))
    if drift == "source_hash":
        record["source_text_hash"] = "0" * 64
        expected = "quarantine source hash differs"
    else:
        record["extraction"][drift] = f"foreign-{drift}"
        expected = "quarantine extraction provenance differs"
    tampered = jsonl_text(record).encode()
    quarantine.write_bytes(tampered)

    with pytest.raises(ValueError, match=expected):
        run_pipeline(config, resume=True)

    assert quarantine.read_bytes() == tampered


def test_retry_failures_still_rejects_quarantine_source_hash_drift(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(FakeExtractor, "fail_ids", {"2"})
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter([make_ticket("2")]))
    run_pipeline(config)

    record = json.loads(quarantine.read_text(encoding="utf-8"))
    record["source_text_hash"] = "0" * 64
    quarantine.write_text(jsonl_text(record), encoding="utf-8")
    monkeypatch.setattr(FakeExtractor, "fail_ids", set())

    with pytest.raises(ValueError, match="quarantine source hash differs"):
        run_pipeline(config, resume=True, retry_failures=True)

    assert FakeExtractor.calls == []


@pytest.mark.parametrize("record_kind", ["completed", "quarantine"])
def test_full_resume_rejects_ticket_ids_absent_from_current_input(
    tmp_path,
    monkeypatch,
    record_kind,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter(()))
    run_pipeline(config)

    if record_kind == "completed":
        output.write_text(jsonl_text(make_chunk("foreign")), encoding="utf-8")
    else:
        quarantine.write_text(
            jsonl_text(
                {
                    "schema_version": "1.0.0",
                    "ticket_id": "foreign",
                    "source_text_hash": "0" * 64,
                    "stage": "llm_extraction",
                    "error_code": "EMPTY_INPUT",
                    "error_type": "ProblemExtractionError",
                    "attempts": 1,
                    "extraction_run_id": "run_test",
                    "extraction": {
                        "model": "fake-model",
                        "prompt_version": "test-v1",
                    },
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="ticket IDs absent from the input"):
        run_pipeline(config, resume=True)


def test_resume_rejects_conflicting_hashes_across_completed_and_quarantine(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter([make_ticket("2")]))
    monkeypatch.setattr(FakeExtractor, "fail_ids", {"2"})
    run_pipeline(config)
    monkeypatch.setattr(FakeExtractor, "fail_ids", set())
    run_pipeline(config, resume=True, retry_failures=True)

    quarantine_record = json.loads(quarantine.read_text(encoding="utf-8"))
    quarantine_record["source_text_hash"] = "0" * 64
    quarantine.write_text(jsonl_text(quarantine_record), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting source hashes"):
        run_pipeline(config, resume=True)

    assert FakeExtractor.calls == []


@pytest.mark.parametrize(
    "collision",
    ["input-output", "input-quarantine", "output-quarantine"],
)
def test_overwrite_rejects_path_collisions_without_modifying_input(
    tmp_path,
    monkeypatch,
    collision,
) -> None:
    source = tmp_path / "tickets.tsv"
    original = b"source-data-must-survive"
    source.write_bytes(original)
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    if collision == "input-output":
        output = source
    elif collision == "input-quarantine":
        quarantine = source
    else:
        quarantine = output
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)

    with pytest.raises(ValueError, match="must be distinct"):
        asyncio.run(
            main.run_pipeline(
                config,
                limit=None,
                resume=False,
                overwrite=True,
                concurrency_override=None,
            )
        )

    assert source.read_bytes() == original


def test_overwrite_rejects_hard_link_collision_without_modifying_input(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "tickets.tsv"
    original = b"source-data-must-survive"
    source.write_bytes(original)
    output = tmp_path / "chunks.jsonl"
    output.hardlink_to(source)
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)

    with pytest.raises(ValueError, match="same file"):
        asyncio.run(
            main.run_pipeline(
                config,
                limit=None,
                resume=False,
                overwrite=True,
                concurrency_override=None,
            )
        )

    assert source.read_bytes() == original


def test_programmatic_resume_and_overwrite_are_mutually_exclusive(tmp_path) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_bytes(b"source-data")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)

    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(
            main.run_pipeline(
                config,
                limit=None,
                resume=True,
                overwrite=True,
                concurrency_override=None,
            )
        )


def test_open_outputs_second_path_failure_does_not_truncate_existing_output(tmp_path) -> None:
    output = tmp_path / "chunks.jsonl"
    original = b"existing-output\n"
    output.write_bytes(original)
    quarantine = tmp_path / "quarantine-directory"
    quarantine.mkdir()

    with pytest.raises(OSError, match="regular file|directory"):
        main._open_outputs(
            output,
            quarantine,
            resume=False,
            overwrite=True,
        )

    assert output.read_bytes() == original


def test_overwrite_manifest_write_failure_preserves_existing_files(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_bytes(b"source-data")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    manifest = main._manifest_path(output)
    originals = {
        output: b"existing-output\n",
        quarantine: b"existing-quarantine\n",
        manifest: b"existing-manifest\n",
    }
    for path, content in originals.items():
        path.write_bytes(content)
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)

    def fail_manifest_write(_path, _manifest):
        raise OSError("synthetic manifest write failure")

    monkeypatch.setattr(main, "_write_manifest_atomic", fail_manifest_write)

    with pytest.raises(OSError, match="synthetic manifest write failure"):
        asyncio.run(
            main.run_pipeline(
                config,
                limit=None,
                resume=False,
                overwrite=True,
                concurrency_override=None,
            )
        )

    assert {path: path.read_bytes() for path in originals} == originals


def test_overwrite_output_open_failure_restores_previous_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_bytes(b"source-data")
    output = tmp_path / "chunks.jsonl"
    original_output = b"existing-output\n"
    output.write_bytes(original_output)
    quarantine = tmp_path / "quarantine-directory"
    quarantine.mkdir()
    manifest = main._manifest_path(output)
    original_manifest = b"existing-manifest\n"
    manifest.write_bytes(original_manifest)
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    observed_states = []
    open_outputs = main._open_outputs

    def observe_manifest_then_open(*args, **kwargs):
        observed_states.append(main._read_manifest(manifest)["state"])
        return open_outputs(*args, **kwargs)

    monkeypatch.setattr(main, "_open_outputs", observe_manifest_then_open)

    with pytest.raises(OSError, match="regular file|directory"):
        asyncio.run(
            main.run_pipeline(
                config,
                limit=None,
                resume=False,
                overwrite=True,
                concurrency_override=None,
            )
        )

    assert output.read_bytes() == original_output
    assert manifest.read_bytes() == original_manifest
    assert quarantine.is_dir()
    assert observed_states == [main.RUN_STATE_INITIALIZING]


def test_overwrite_final_manifest_failure_leaves_fail_closed_state(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_bytes(b"source-data")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    output.write_bytes(b"old-output\n")
    quarantine.write_bytes(b"old-quarantine\n")
    manifest = main._manifest_path(output)
    manifest.write_bytes(b"old-manifest\n")
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter(()))
    write_manifest = main._write_manifest_atomic
    observed_states = []

    def fail_ready_manifest(path, value):
        observed_states.append(value["state"])
        if value["state"] == main.RUN_STATE_READY:
            raise OSError("synthetic final manifest failure")
        write_manifest(path, value)

    monkeypatch.setattr(main, "_write_manifest_atomic", fail_ready_manifest)

    with pytest.raises(OSError, match="synthetic final manifest failure"):
        asyncio.run(
            main.run_pipeline(
                config,
                limit=None,
                resume=False,
                overwrite=True,
                concurrency_override=None,
            )
        )

    assert output.read_bytes() == b""
    assert quarantine.read_bytes() == b""
    assert main._read_manifest(manifest)["state"] == main.RUN_STATE_INITIALIZING
    assert observed_states == [main.RUN_STATE_INITIALIZING, main.RUN_STATE_READY]

    with pytest.raises(ValueError, match="manifest is not ready"):
        run_pipeline(config, resume=True)


def test_open_outputs_lock_shared_quarantine_through_hard_link(tmp_path) -> None:
    quarantine = tmp_path / "errors-a.jsonl"
    quarantine.write_text("", encoding="utf-8")
    quarantine_alias = tmp_path / "errors-b.jsonl"
    quarantine_alias.hardlink_to(quarantine)

    first_handles = main._open_outputs(
        tmp_path / "chunks-a.jsonl",
        quarantine,
        resume=True,
        overwrite=False,
    )
    try:
        with pytest.raises(RuntimeError, match="writing output file"):
            main._open_outputs(
                tmp_path / "chunks-b.jsonl",
                quarantine_alias,
                resume=True,
                overwrite=False,
            )
    finally:
        for handle in first_handles:
            handle.close()

    assert not (tmp_path / "chunks-b.jsonl").exists()


def test_run_pipeline_rejects_output_lock_conflict(tmp_path) -> None:
    source = tmp_path / "tickets.tsv"
    original = b"source-data"
    source.write_bytes(original)
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)

    with main._exclusive_output_lock(main._lock_path(output)):
        with pytest.raises(RuntimeError, match="holds the output lock"):
            run_pipeline(config)

    assert source.read_bytes() == original
    assert not output.exists()
    assert not quarantine.exists()


def test_run_pipeline_rejects_shared_quarantine_lock(tmp_path) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_bytes(b"source-data")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "shared-errors.jsonl"
    config = make_config(source, output, quarantine)
    quarantine_lock = main._lock_path(quarantine.resolve())

    with main._exclusive_output_lock(quarantine_lock):
        with pytest.raises(RuntimeError, match="holds the output lock"):
            run_pipeline(config)

    assert not output.exists()
    assert not quarantine.exists()


def test_pipeline_rejects_duplicate_input_ids(tmp_path, monkeypatch) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(FakeExtractor, "fail_ids", set())
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(
        main,
        "load_tickets",
        lambda _path: iter(
            [
                make_ticket("duplicate", content="first"),
                make_ticket("duplicate", content="second"),
            ]
        ),
    )

    with pytest.raises(ValueError, match="duplicate ticket id"):
        run_pipeline(config)

    assert output.read_text(encoding="utf-8") == ""
    assert FakeExtractor.closed is True


@pytest.mark.parametrize(
    "key",
    ["concurrency", "max_attempts", "checkpoint_every", "log_every"],
)
@pytest.mark.parametrize("value", [True, 1.0, 1.5, "1"])
def test_pipeline_integer_settings_require_actual_integers(
    tmp_path,
    key,
    value,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    config = make_config(
        source,
        tmp_path / "chunks.jsonl",
        tmp_path / "errors.jsonl",
    )
    config["pipeline"][key] = value

    with pytest.raises(ValueError, match=rf"pipeline\.{key} must be an integer"):
        run_pipeline(config)


def test_pipeline_preserves_primary_error_when_extractor_close_also_fails(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    class CloseFailingExtractor(FakeExtractor):
        fail_ids: set[str] = set()

        async def aclose(self):
            raise RuntimeError("synthetic close failure")

    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    config = make_config(
        source,
        tmp_path / "chunks.jsonl",
        tmp_path / "errors.jsonl",
    )
    monkeypatch.setattr(main, "ProblemExtractor", CloseFailingExtractor)
    monkeypatch.setattr(
        main,
        "load_tickets",
        lambda _path: map(make_ticket, ["duplicate", "duplicate"]),
    )

    with pytest.raises(ValueError, match="duplicate ticket id"):
        run_pipeline(config)

    assert "RuntimeError while preserving primary ValueError" in caplog.text
    assert "synthetic close failure" not in caplog.text


def test_pipeline_preserves_primary_base_exception_when_close_fails(
    tmp_path,
    monkeypatch,
) -> None:
    class PrimarySignal(BaseException):
        pass

    class CloseFailingExtractor(FakeExtractor):
        async def aclose(self):
            raise RuntimeError("synthetic close failure")

    def interrupted_tickets(_path):
        raise PrimarySignal

    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    config = make_config(
        source,
        tmp_path / "chunks.jsonl",
        tmp_path / "errors.jsonl",
    )
    monkeypatch.setattr(main, "ProblemExtractor", CloseFailingExtractor)
    monkeypatch.setattr(main, "load_tickets", interrupted_tickets)

    with pytest.raises(PrimarySignal):
        run_pipeline(config)


def test_pipeline_propagates_close_error_after_success(tmp_path, monkeypatch) -> None:
    class CloseFailingExtractor(FakeExtractor):
        fail_ids: set[str] = set()

        async def aclose(self):
            raise RuntimeError("synthetic close failure")

    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    config = make_config(
        source,
        tmp_path / "chunks.jsonl",
        tmp_path / "errors.jsonl",
    )
    monkeypatch.setattr(main, "ProblemExtractor", CloseFailingExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter(()))

    with pytest.raises(RuntimeError, match="synthetic close failure"):
        run_pipeline(config)


def test_resume_skips_quarantine_by_default_and_retries_only_when_requested(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(
        main,
        "load_tickets",
        lambda _path: map(make_ticket, ["1", "2"]),
    )
    monkeypatch.setattr(FakeExtractor, "fail_ids", {"2"})

    initial = run_pipeline(config)
    assert initial.succeeded == 1
    assert initial.failed == 1

    monkeypatch.setattr(FakeExtractor, "fail_ids", set())
    resumed = run_pipeline(config, resume=True)
    assert resumed.submitted == 0
    assert resumed.skipped == 2
    assert FakeExtractor.calls == []

    retried = run_pipeline(config, resume=True, retry_failures=True)
    assert retried.succeeded == 1
    assert retried.skipped == 1
    assert FakeExtractor.calls == ["2"]
    assert {record["id"] for record in map(json.loads, output.read_text().splitlines())} == {
        "1",
        "2",
    }
    assert len(quarantine.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize(
    "drift",
    [
        "input",
        "model",
        "model_revision",
        "model_source_repo",
        "model_artifact_fingerprint",
        "enable_thinking",
        "code_commit",
        "quarantine_path",
    ],
)
def test_resume_rejects_manifest_input_or_model_drift(tmp_path, monkeypatch, drift) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("original source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter(()))
    run_pipeline(config)
    original_output = output.read_bytes()
    original_quarantine = quarantine.read_bytes()
    original_manifest = main._manifest_path(output).read_bytes()

    if drift == "input":
        source.write_text("changed source", encoding="utf-8")
    elif drift == "model":
        config["llm"]["model"] = "changed-model"
    elif drift == "model_revision":
        config["llm"]["model_revision"] = "changed-model-revision"
    elif drift == "model_source_repo":
        config["llm"]["model_source_repo"] = "Qwen/changed-model-source"
    elif drift == "model_artifact_fingerprint":
        config["llm"]["model_artifact_fingerprint"] = f"sha256:{'a' * 64}"
    elif drift == "enable_thinking":
        config["llm"]["enable_thinking"] = True
    elif drift == "code_commit":
        monkeypatch.setenv("RAG_CODE_COMMIT", "2" * 40)
    else:
        changed_quarantine = tmp_path / "changed-errors.jsonl"
        config["data"]["quarantine"] = str(changed_quarantine)

    with pytest.raises(ValueError, match="Cannot resume.*differs"):
        run_pipeline(config, resume=True)

    assert output.read_bytes() == original_output
    assert quarantine.read_bytes() == original_quarantine
    assert main._manifest_path(output).read_bytes() == original_manifest
    if drift == "quarantine_path":
        assert not changed_quarantine.exists()


@pytest.mark.parametrize("commit", [None, "", "1" * 39, "A" * 40, "g" * 40])
def test_pipeline_requires_valid_lowercase_git_commit(tmp_path, monkeypatch, commit) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter(()))
    if commit is None:
        monkeypatch.delenv("RAG_CODE_COMMIT")
    else:
        monkeypatch.setenv("RAG_CODE_COMMIT", commit)

    with pytest.raises(ValueError, match="RAG_CODE_COMMIT"):
        run_pipeline(config)

    assert not output.exists()
    assert not quarantine.exists()


def test_manifest_records_code_commit(tmp_path, monkeypatch) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("test source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter(()))

    run_pipeline(config)

    manifest = json.loads(main._manifest_path(output).read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "3.0.0"
    assert manifest["code"] == {"git_commit": TEST_CODE_COMMIT}


@pytest.mark.parametrize("missing", ["completed", "quarantine"])
def test_resume_rejects_missing_result_file(tmp_path, monkeypatch, missing) -> None:
    source = tmp_path / "tickets.tsv"
    source.write_text("original source", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    quarantine = tmp_path / "errors.jsonl"
    config = make_config(source, output, quarantine)
    monkeypatch.setattr(main, "ProblemExtractor", FakeExtractor)
    monkeypatch.setattr(main, "load_tickets", lambda _path: iter(()))
    run_pipeline(config)
    removed = output if missing == "completed" else quarantine
    removed.unlink()

    with pytest.raises(ValueError, match="required result files are missing"):
        run_pipeline(config, resume=True)

    assert not removed.exists()


def test_empty_ticket_is_not_retried() -> None:
    class EmptyExtractor:
        model = "fake-model"
        prompt_version = "test-v1"

        async def extract(self, _ticket):
            raise ProblemExtractionError("should not be called")

    ticket = make_ticket("empty")
    ticket.content = ""
    ticket.goal = ""

    chunk, error, attempts = asyncio.run(
        main._extract_one(
            EmptyExtractor(),
            ticket,
            extraction_run_id="run_test",
            max_attempts=3,
            retry_backoff_seconds=0,
        )
    )

    assert chunk is None
    assert isinstance(error, ProblemExtractionError)
    assert error.code == "EMPTY_INPUT"
    assert attempts == 1
