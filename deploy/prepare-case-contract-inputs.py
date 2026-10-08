"""Recover the fixed reviewed inputs from existing server data, using CPU only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "case-contract-input-reconstruction-v1"
SPEC_VERSION = "case-content-extraction-v1"
SELECTION = ROOT / "research/specs/case-contract-run-v1/development-selection.json"
OUTPUT = ROOT / "data/case-relevance-phase1-v1/extraction-contract-v1-001"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json(value: str):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = item
        return result

    def reject_constant(_):
        raise ValueError("nonfinite_json_number")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=reject_constant)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def input_bytes(rows) -> bytes:
    # Match the reviewed preparation exactly, including LF on Windows and Linux.
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def load_selection(path: Path, contract_dir: Path) -> dict:
    selection = strict_json(path.read_text(encoding="utf-8"))
    if (selection.get("version") != VERSION
            or selection.get("spec_version") != SPEC_VERSION
            or selection.get("formal_gold") is not False):
        raise ValueError("invalid_selection_version")
    rows = selection["samples"]
    if len(rows) != 80 or [r["sample_id"] for r in rows] != [f"B{i:03}" for i in range(1, 81)]:
        raise ValueError("expected_fixed_80_sample_ids")
    hashes = [r["case_content_sha256"] for r in rows]
    hashes += [selection[k] for k in (
        "inputs_sha256", "review_manifest_sha256", "contract_manifest_sha256",
    )]
    if not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
        raise ValueError("invalid_selection_digest")
    if len({r["case_content_sha256"] for r in rows}) != 80:
        raise ValueError("duplicate_selection_content")
    contract_digest = digest((contract_dir / "manifest.json").read_bytes())
    if contract_digest != selection["contract_manifest_sha256"]:
        raise ValueError("contract_manifest_mismatch")
    contract = strict_json((contract_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in contract["artifacts"].items():
        artifact = (contract_dir / name).resolve()
        if (not artifact.is_relative_to(contract_dir.resolve())
                or digest(artifact.read_bytes()) != expected):
            raise ValueError("contract_artifact_mismatch")
    return selection


def choose_source(explicit: Path | None, root: Path = ROOT) -> Path:
    if explicit is not None:
        candidates = [explicit]
    elif os.environ.get("RAG_INPUT_PATH"):
        candidates = [Path(os.environ["RAG_INPUT_PATH"])]
    else:
        candidates = [
            root / "data/case-relevance-phase1-v1/sampling-001/candidates.jsonl",
            root / "data/retrieval-baseline-v1/dataset/dataset.sqlite3",
            root / "data/raw/t_order_master.sanitized.v1_9.tsv",
        ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise ValueError("source_not_found_use_--source_with_existing_TSV_SQLite_or_candidates_JSONL")


def source_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    formats = {".tsv": "tsv", ".jsonl": "jsonl", ".sqlite3": "sqlite",
               ".sqlite": "sqlite", ".db": "sqlite"}
    if path.suffix.lower() not in formats:
        raise ValueError("unknown_source_format_use_--format")
    return formats[path.suffix.lower()]


def read_contents(path: Path, kind: str):
    if kind == "sqlite":
        # A transaction keeps a consistent logical snapshot, including committed WAL rows.
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            for (content,) in connection.execute("SELECT content FROM records ORDER BY rid"):
                yield content
        finally:
            connection.close()
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if kind == "jsonl":
            for number, line in enumerate(handle, 1):
                try:
                    row = strict_json(line)
                    content = row["content"]
                except (ValueError, TypeError, KeyError):
                    raise ValueError(f"invalid_candidates_jsonl_at_line_{number}") from None
                yield content
        elif kind == "tsv":
            # Match the original dataset builder's quoting and raw-text handling.
            limit = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(limit)
                    break
                except OverflowError:
                    limit //= 10
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            header = reader.fieldnames
            if not header or "case_content" not in header or len(set(header)) != len(header):
                raise ValueError("expected_unique_TSV_headers_including_case_content")
            for number, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"invalid_TSV_width_at_record_{number}")
                yield row["case_content"]
        else:
            raise ValueError("unsupported_source_format")


def reconstruct(source: Path, output: Path, *, kind="auto", selection_path=SELECTION,
                contract_dir=ROOT / "research/specs/case-content-extraction-v1") -> dict:
    if output.exists():
        raise ValueError("output_exists_use_a_new_directory")
    selection_bytes = selection_path.read_bytes()
    selection = load_selection(selection_path, contract_dir)
    kind = source_format(source, kind)
    source = source.resolve()
    before = source.stat()
    wanted = {r["case_content_sha256"]: r["sample_id"] for r in selection["samples"]}
    found, occurrences = {}, Counter()
    stream_digest = hashlib.sha256()
    count = 0
    for count, content in enumerate(read_contents(source, kind), 1):
        if not isinstance(content, str):
            raise ValueError(f"non_string_content_at_record_{count}")
        raw = content.encode("utf-8")
        # This fingerprints the complete ordered text stream, not other source columns.
        stream_digest.update(len(raw).to_bytes(8, "big"))
        stream_digest.update(raw)
        key = digest(raw)
        if key in wanted:
            sample_id = wanted[key]
            if sample_id in found and found[sample_id] != content:
                raise ValueError("content_hash_collision")
            found[sample_id] = content
            occurrences[sample_id] += 1
    after = source.stat()
    if any(getattr(before, k) != getattr(after, k) for k in ("st_size", "st_mtime_ns", "st_ino")):
        raise ValueError("source_changed_during_scan")
    missing = [r["sample_id"] for r in selection["samples"] if r["sample_id"] not in found]
    if missing:
        raise ValueError("missing_exact_samples:" + ",".join(missing))
    rows = [{"sample_id": r["sample_id"], "input": {"case_content": found[r["sample_id"]]}}
            for r in selection["samples"]]
    inputs = input_bytes(rows)
    if digest(inputs) != selection["inputs_sha256"]:
        raise ValueError("reconstructed_inputs_differ_from_reviewed_inputs")
    if selection_path.read_bytes() != selection_bytes:
        raise ValueError("selection_changed_during_scan")
    summary = {
        "status": "prepared", "version": VERSION, "sample_count": len(rows),
        "inputs_sha256": digest(inputs), "matches_reviewed_inputs": True,
        "formal_gold": False, "model_run": False,
        "acceptance_matrix": "retained_locally_not_reconstructed",
        "semantic_review_status": "not_run",
    }
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": str(source), "source_format": kind,
        "source_size_bytes": before.st_size, "source_mtime_ns": before.st_mtime_ns,
        "source_record_count": count, "source_content_stream_sha256": stream_digest.hexdigest(),
        "source_fingerprint_scope": "ordered_utf8_contents_prefixed_by_8_byte_big_endian_length",
        "matching_occurrences": dict(sorted(occurrences.items())),
        "selection_sha256": digest(selection_bytes),
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "python_executable": sys.executable, "python_version": sys.version,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "text_normalization": "none", "gpu_inference": "not_run",
    }
    files = {"inputs.jsonl": inputs, "summary.json": json_bytes(summary),
             "source-provenance.json": json_bytes(provenance),
             "selection.json": selection_bytes}
    manifest = {
        "spec_version": SPEC_VERSION, "preparation_version": VERSION,
        "contract_manifest_sha256": selection["contract_manifest_sha256"],
        "review_manifest_sha256": selection["review_manifest_sha256"],
        "review_manifest_location": "local_review_archive_not_copied",
        "formal_gold": False, "model_run": False,
        "artifacts": {name: digest(value) for name, value in files.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    for name, value in files.items():
        with (output / name).open("xb") as handle:
            handle.write(value)
    # Written last: failed preparation never appears as a complete input package.
    with (output / "manifest.json").open("xb") as handle:
        handle.write(json_bytes(manifest))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="Existing raw TSV, dataset SQLite, or candidates JSONL")
    parser.add_argument("--format", choices=("auto", "tsv", "sqlite", "jsonl"), default="auto")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        summary = reconstruct(choose_source(args.source), args.output, kind=args.format)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, csv.Error) as exc:
        # Do not print source rows or parser excerpts containing private text.
        reason = type(exc).__name__
        if isinstance(exc, ValueError) and not isinstance(exc, UnicodeError):
            reason = str(exc)
        print(json.dumps({"status": "failed", "reason": reason}, ensure_ascii=False),
              file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
