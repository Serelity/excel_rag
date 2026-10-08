import csv
import importlib.util
import json
import sqlite3

import pytest

from semantic_extraction.case_contract import runner

_spec = importlib.util.spec_from_file_location(
    "prepare_contract_inputs", runner.ROOT / "deploy/prepare-case-contract-inputs.py",
)
prepare_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_inputs)


@pytest.fixture
def reference(tmp_path):
    texts = [f"模拟案例{i:03}：路灯不亮。" for i in range(1, 81)]
    texts[0] = '  模拟文本\r\n包含\t制表符与"引号"；Ａ\u3000B  '
    rows = [{"sample_id": f"B{i:03}", "input": {"case_content": content}}
            for i, content in enumerate(texts, 1)]
    selection = json.loads(prepare_inputs.SELECTION.read_text(encoding="utf-8"))
    selection["samples"] = [
        {"sample_id": row["sample_id"], "case_content_sha256": prepare_inputs.digest(
            row["input"]["case_content"].encode("utf-8"),
        )} for row in rows
    ]
    selection["inputs_sha256"] = prepare_inputs.digest(prepare_inputs.input_bytes(rows))
    path = tmp_path / "selection.json"
    path.write_bytes(prepare_inputs.json_bytes(selection))
    return texts, rows, path


def write_source(path, kind, texts):
    if kind == "sqlite":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE records (rid INTEGER PRIMARY KEY, content TEXT)")
            connection.executemany("INSERT INTO records(content) VALUES(?)", [(t,) for t in texts])
    elif kind == "tsv":
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["case_content", "reply"])
            writer.writerows((t, "do_not_use_this_other_field") for t in texts)
    else:
        path.write_bytes(b"".join(
            (json.dumps({"content": t, "sample_id": "wrong_id_ignore"}, ensure_ascii=False)
             + "\n").encode("utf-8") for t in texts
        ))


@pytest.mark.parametrize("kind,suffix", [
    ("sqlite", ".sqlite3"), ("tsv", ".tsv"), ("jsonl", ".jsonl"),
])
def test_reconstructs_exact_bytes_without_changing_source(tmp_path, reference, kind, suffix):
    texts, rows, selection = reference
    source = tmp_path / ("source" + suffix)
    write_source(source, kind, ["无关的模拟记录", *reversed(texts), texts[0]])
    before = source.read_bytes()
    output = tmp_path / "prepared"
    summary = prepare_inputs.reconstruct(source, output, selection_path=selection)
    assert source.read_bytes() == before
    assert (output / "inputs.jsonl").read_bytes() == prepare_inputs.input_bytes(rows)
    assert summary["sample_count"] == 80 and summary["model_run"] is False
    assert not (output / "acceptance-matrix.jsonl").exists()
    provenance = runner.load(output / "source-provenance.json")
    assert provenance["matching_occurrences"]["B001"] == 2
    assert provenance["source_record_count"] == 82
    runner.verify_artifacts(output, runner.load(output / "manifest.json"))
    # Exercise the real downstream preparation without starting a model or transport.
    prepared = runner.prepare(
        output, tmp_path / "run", sample_ids=runner.DEFAULT_SMOKE_IDS.split(","),
        model="Qwen3-30B-A3B", model_fingerprint="sha256:" + "a" * 64,
    )
    assert prepared == {"status": "prepared", "selected": 10}


def test_sqlite_reads_committed_wal_content(tmp_path, reference):
    texts, rows, selection = reference
    source = tmp_path / "wal.sqlite3"
    connection = sqlite3.connect(source)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE records (rid INTEGER PRIMARY KEY, content TEXT)")
        connection.executemany("INSERT INTO records(content) VALUES(?)", [(t,) for t in texts])
        connection.commit()
        output = tmp_path / "prepared"
        prepare_inputs.reconstruct(source, output, selection_path=selection)
        assert (output / "inputs.jsonl").read_bytes() == prepare_inputs.input_bytes(rows)
    finally:
        connection.close()


@pytest.mark.parametrize("change", ["drop", "strip", "newline"])
def test_missing_or_normalized_content_fails_without_partial_package(tmp_path, reference, change):
    texts, _, selection = reference
    if change == "drop":
        texts = texts[1:]
    elif change == "strip":
        texts[0] = texts[0].strip()
    else:
        texts[0] = texts[0].replace("\r\n", "\n")
    source = tmp_path / "source.jsonl"
    write_source(source, "jsonl", texts)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="missing_exact_samples:B001$"):
        prepare_inputs.reconstruct(source, output, selection_path=selection)
    assert not output.exists()


@pytest.mark.parametrize("field,value,error", [
    ("inputs_sha256", "0" * 64, "differ_from_reviewed"),
    ("contract_manifest_sha256", "0" * 64, "contract_manifest_mismatch"),
])
def test_fixed_reference_digest_must_match(tmp_path, reference, field, value, error):
    texts, _, selection = reference
    data = json.loads(selection.read_text(encoding="utf-8"))
    data[field] = value
    selection.write_bytes(prepare_inputs.json_bytes(data))
    source = tmp_path / "source.jsonl"
    write_source(source, "jsonl", texts)
    with pytest.raises(ValueError, match=error):
        prepare_inputs.reconstruct(source, tmp_path / "out", selection_path=selection)
    assert not (tmp_path / "out").exists()


def test_existing_output_is_never_overwritten(tmp_path, reference):
    output = tmp_path / "out"
    output.mkdir()
    (output / "inputs.jsonl").write_bytes(b"keep")
    with pytest.raises(ValueError, match="output_exists"):
        prepare_inputs.reconstruct(tmp_path / "missing.tsv", output, selection_path=reference[2])
    assert (output / "inputs.jsonl").read_bytes() == b"keep"


@pytest.mark.parametrize("content,error", [
    (b"wrong_header\nvalue\n", "expected_unique_TSV_headers"),
    (b"case_content\tcase_content\nvalue\tvalue\n", "expected_unique_TSV_headers"),
    (b"case_content\tother\nvalue\n", "invalid_TSV_width"),
])
def test_rejects_bad_tsv_without_exposing_text(tmp_path, reference, content, error):
    source = tmp_path / "source.tsv"
    source.write_bytes(content)
    with pytest.raises(ValueError, match=error):
        prepare_inputs.reconstruct(source, tmp_path / "out", selection_path=reference[2])


def test_bad_jsonl_error_does_not_echo_source(tmp_path, capsys):
    source = tmp_path / "bad.jsonl"
    source.write_bytes(b'{"content":"private_marker","content":"other"}\n')
    assert prepare_inputs.main(["--source", str(source), "--output", str(tmp_path / "out")]) == 2
    captured = capsys.readouterr()
    assert "private_marker" not in captured.err
    assert "invalid_candidates_jsonl_at_line_1" in captured.err
    assert not (tmp_path / "out").exists()


def test_source_discovery_and_explicit_source_do_not_silently_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_INPUT_PATH", raising=False)
    dataset = tmp_path / "data/retrieval-baseline-v1/dataset/dataset.sqlite3"
    dataset.parent.mkdir(parents=True)
    dataset.touch()
    assert prepare_inputs.choose_source(None, tmp_path) == dataset.resolve()
    with pytest.raises(ValueError, match="source_not_found"):
        prepare_inputs.choose_source(tmp_path / "missing.tsv", tmp_path)
    monkeypatch.setenv("RAG_INPUT_PATH", str(tmp_path / "missing.tsv"))
    with pytest.raises(ValueError, match="source_not_found"):
        prepare_inputs.choose_source(None, tmp_path)


def test_published_selection_contains_only_fixed_ids_and_digests():
    selection = prepare_inputs.load_selection(
        prepare_inputs.SELECTION, runner.ROOT / "research/specs/case-content-extraction-v1",
    )
    assert len(selection["samples"]) == 80
    assert all(set(row) == {"sample_id", "case_content_sha256"} for row in selection["samples"])
