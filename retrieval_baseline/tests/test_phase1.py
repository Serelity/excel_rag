from __future__ import annotations

import copy
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrieval_baseline import phase1
from retrieval_baseline.common import content_key, file_hash, stable_rank


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def refresh_dataset(dataset):
    write_json(dataset / "manifest.json", {
        "artifacts": {name: file_hash(dataset / name)
                      for name in ("dataset.sqlite3", "queries.dev.jsonl")},
    })


@pytest.fixture
def prepared(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with sqlite3.connect(dataset / "dataset.sqlite3") as db:
        db.execute("CREATE TABLE records(rid INTEGER PRIMARY KEY,source_id TEXT,source_row INTEGER,"
                   "group_id INTEGER,text_hash TEXT,content TEXT,call_time TEXT,split TEXT)")
        db.execute("INSERT INTO records VALUES(0,'corpus',1,0,?,?,'2025-01-01','corpus')",
                   (content_key("历史库中的独立问题"), "历史库中的独立问题"))
        for group in range(1, 17):
            for offset in (0, 1):
                rid = 2 * group + offset
                text = f"第{group}个测试场景的诉求记录{offset}"
                db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?)",
                           (rid, f"source-{rid}", rid + 1, group, content_key(text),
                            text, "2026-01-01", "dev"))
    write_rows(dataset / "queries.dev.jsonl", [{"source_id": "source-2"}])
    refresh_dataset(dataset)
    # Invalid files make any accidental full test-file read fail deterministically.
    (dataset / "queries.test.jsonl").write_text("DO NOT READ TEST DATA", encoding="utf-8")
    (dataset / "qrels.test.jsonl").write_text("DO NOT READ TEST LABELS", encoding="utf-8")
    exclusions = tmp_path / "source-exclusions.json"
    write_json(exclusions, {
        "schema_version": phase1.EXCLUSIONS_VERSION, "version": "test-1",
        "excluded_source_ids": ["source-5"], "excluded_group_ids": [],
        "history_complete": False, "limitations": ["Synthetic fixture only"],
        "reviewed_by": "fixture-reviewer", "reviewed_at": "2026-10-03T10:00:00+08:00",
    })
    output = tmp_path / "prepared"
    args = SimpleNamespace(dataset=dataset, exclusions=exclusions, output=output, limit=80)
    phase1.prepare_queries(args)
    return args


def review_rows(prepared):
    rows = phase1._rows(prepared.output / "candidates.jsonl")
    for index, row in enumerate(rows):
        row["review"] = {
            "reviewed_by": "fixture-reviewer", "reviewed_at": "2026-10-03T10:00:00+08:00",
            "scope_status": "in_scope", "source_review_status": "verified",
            "intent_review_status": "preserved",
            "temporal_review_status": "supported_at_acceptance",
            "query_as_of": "2026-01-01", "information_basis": "Synthetic contemporaneous evidence",
            "removed_followup_information": "Synthetic source contains none",
            "address_review_status": "supported", "near_duplicate_review_status": "independent",
            "related_event_group": f"event-{row['source_id']}", "selection_reason": "Fixture scope",
            "rewrite_notes": "Synthetic faithful rewrite", "problem": f"场景{index}楼道清洁诉求",
            "address": f"测试{index}花园", "notes": "",
        }
    return rows


def final_args(prepared, tmp_path, *, uncertain=False):
    worksheet = tmp_path / "reviewed.jsonl"
    write_rows(worksheet, review_rows(prepared))
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {
        "schema_version": phase1.PROTOCOL_VERSION, "version": "1.1", "experiment_id": "fixture",
        "scenario_scope": "Synthetic test cases", "reviewed_by": "fixture-reviewer",
        "reviewed_at": "2026-10-03T10:00:00+08:00", "selection_without_retrieval_results": True,
        "allow_temporal_uncertainty": uncertain,
        "temporal_limitation": "Exported-text exploratory experiment; no online replay claim",
    })
    return SimpleNamespace(dataset=prepared.dataset, exclusions=prepared.exclusions,
                           worksheet=worksheet, protocol=protocol,
                           sampling_plan=prepared.output / "sampling-plan.json",
                           output=tmp_path / "final")


def test_prepare_group_minimum_and_exclusions_without_review_or_test_files(prepared):
    rows = phase1._rows(prepared.output / "candidates.jsonl")
    assert len(rows) == 14
    assert all(row["rid"] % 2 == 0 for row in rows)
    assert all(row["group_id"] not in (1, 2) for row in rows)
    assert [row["source_id"] for row in rows] == sorted(
        [row["source_id"] for row in rows], key=lambda source: stable_rank(42, source))
    assert all(set(row["review"].values()) == {""} for row in rows)
    assert (prepared.output / "protocol.template.json").is_file()


def test_finalization_rejects_missing_human_review(prepared, tmp_path):
    args = final_args(prepared, tmp_path)
    args.worksheet = prepared.output / "candidates.jsonl"
    with pytest.raises(ValueError, match="human review"):
        phase1.finalize_queries(args)
    assert not args.output.exists()


def test_finalization_opaque_triplets_fixed_split_and_duplicate_skip(prepared, tmp_path):
    args = final_args(prepared, tmp_path)
    rows = phase1._rows(args.worksheet)
    rows[1]["review"]["address"] = rows[0]["review"]["address"]
    write_rows(args.worksheet, rows)
    result = phase1.finalize_queries(args)
    assert result == {"query_count": 30, "scenario_count": 10,
                      "temporal_uncertainty_count": 0, "stop_position": 11}
    queries = phase1._read(args.output / "queries.json")
    assert phase1.validate_formal_queries(queries) == queries
    audit = phase1._rows(args.output / "sampling-audit.jsonl")
    assert audit[2]["first_exclusion_reason"] == "duplicate_query_definition"
    provenance = phase1._rows(args.output / "query-provenance.jsonl")
    ordered = sorted(provenance, key=lambda row: stable_rank(43, row["source_id"]))
    assert [row["partition"] for row in ordered] == ["calibration"] * 6 + ["evaluation"] * 4
    for key, name in (("queries", "queries.json"), ("provenance", "query-provenance.jsonl"),
                      ("sampling_audit", "sampling-audit.jsonl"), ("id_map", "query-id-map.json")):
        setattr(args, key, args.output / name)
    assert phase1._verify_inputs(args)["scenario_count"] == 10


@pytest.mark.parametrize("field", ["query_id", "scenario_id", "query"])
def test_formal_query_definition_tampering(prepared, tmp_path, field):
    args = final_args(prepared, tmp_path)
    phase1.finalize_queries(args)
    queries = phase1._read(args.output / "queries.json")
    combined = next(row for row in queries if row["mode"] == "combined")
    combined[field] = "eval-1" if field != "query" else "Changed problem"
    with pytest.raises(ValueError):
        phase1.validate_formal_queries(queries)


def test_temporal_uncertainty_needs_protocol_declaration(prepared, tmp_path):
    args = final_args(prepared, tmp_path)
    rows = phase1._rows(args.worksheet)
    rows[0]["review"].update(temporal_review_status="export_text_only", query_as_of="unknown")
    write_rows(args.worksheet, rows)
    with pytest.raises(ValueError, match="protocol limitation"):
        phase1.finalize_queries(args)
    protocol = phase1._read(args.protocol)
    protocol["allow_temporal_uncertainty"] = True
    write_json(args.protocol, protocol)
    assert phase1.finalize_queries(args)["temporal_uncertainty_count"] == 1


@pytest.mark.parametrize("column", ["source_id", "group_id", "text_hash"])
def test_corpus_self_match_or_group_leakage_rejected(prepared, tmp_path, column):
    candidate = phase1._rows(prepared.output / "candidates.jsonl")[0]
    with sqlite3.connect(prepared.dataset / "dataset.sqlite3") as db:
        db.execute(f"UPDATE records SET {column}=? WHERE split='corpus'", (candidate[column],))
    refresh_dataset(prepared.dataset)
    prepared.output = tmp_path / "reprepared"
    phase1.prepare_queries(prepared)
    args = final_args(prepared, tmp_path)
    with pytest.raises(ValueError, match="overlaps the corpus"):
        phase1.finalize_queries(args)


def test_cannot_replace_failed_min_rid_with_another_group_record(prepared, tmp_path):
    args = final_args(prepared, tmp_path)
    rows = phase1._rows(args.worksheet)
    rows[0]["rid"] += 1
    write_rows(args.worksheet, rows)
    with pytest.raises(ValueError, match="fixed representatives"):
        phase1.finalize_queries(args)


def test_original_development_file_cannot_be_removed_to_hide_prior_use(prepared, tmp_path):
    write_rows(prepared.dataset / "queries.dev.jsonl", [])
    prepared.output = tmp_path / "tampered-original-dev"
    with pytest.raises(ValueError, match="Original dev query file"):
        phase1.prepare_queries(prepared)


def test_related_events_must_not_cross_scenarios(prepared, tmp_path):
    args = final_args(prepared, tmp_path)
    rows = phase1._rows(args.worksheet)
    rows[1]["review"]["related_event_group"] = rows[0]["review"]["related_event_group"]
    write_rows(args.worksheet, rows)
    assert phase1.finalize_queries(args)["stop_position"] == 11
    audit = phase1._rows(args.output / "sampling-audit.jsonl")
    assert audit[2]["first_exclusion_reason"] == "related_or_near_duplicate"


@pytest.fixture
def frozen(prepared, tmp_path, monkeypatch):
    args = final_args(prepared, tmp_path)
    phase1.finalize_queries(args)
    for key, name in (("queries", "queries.json"), ("provenance", "query-provenance.jsonl"),
                      ("sampling_audit", "sampling-audit.jsonl"), ("id_map", "query-id-map.json")):
        setattr(args, key, args.output / name)
    runtime = {"dataset": str(args.dataset), "retriever": "hybrid", "case_k": 50, "max_terms": 32}
    environment = {"python": {"executable": "synthetic", "version": "3.12"},
                   "packages": {"synthetic": "1.0"}}
    inference_implementation = {"search.py": "synthetic-search",
                                "reranker.py": "synthetic-reranker"}
    resources = {"dataset": {"path": str(args.dataset), "manifest_sha256": "synthetic"}}
    # Model and preflight internals are covered by their own tests; this fixture
    # exercises the full query/audit/freeze chain without downloading model weights.
    module = SimpleNamespace(resource_fingerprints=lambda _: resources,
                             runtime_identity=lambda: environment,
                             inference_implementation_fingerprints=lambda: inference_implementation,
                             runtime_configuration=lambda value: {
                                 "dataset": str(value.dataset), "retriever": value.retriever,
                                 "case_k": value.case_k, "max_terms": value.max_terms})
    monkeypatch.setitem(sys.modules, "retrieval_baseline.preflight", module)
    monkeypatch.setattr(phase1, "code_identity", lambda: {
        "git_head": "fixture-head", "implementation_sha256": {"search.py": "fixture-source"}})
    monkeypatch.setattr(phase1, "_verify_git_commit", lambda _: None)
    args.retriever, args.case_k, args.max_terms = "hybrid", 50, 32
    args.config, args.preflight = tmp_path / "config.json", tmp_path / "preflight.json"
    write_json(args.config, runtime)
    write_json(args.preflight, {"status": "passed", "inference": {
        "status": "passed", "joint_hybrid_reranker_passed": True},
        "runtime_config": runtime, "resources": resources, "runtime": environment,
        "inference_implementation_sha256": inference_implementation})
    args.output = tmp_path / "freeze-manifest.json"
    phase1.freeze_experiment(args)
    return args


def test_freeze_roundtrip_ignores_unrelated_files_and_binds_actual_config(frozen, tmp_path):
    (tmp_path / "unrelated.txt").write_text("Unrelated edits are allowed", encoding="utf-8")
    manifest = phase1.verify_freeze(frozen.output, frozen)
    assert manifest["review_status"] == "complete"
    assert phase1.verify_freeze(frozen.output) == manifest
    args = copy.copy(frozen)
    args.case_k = 25
    with pytest.raises(ValueError, match="runtime configuration"):
        phase1.verify_freeze(frozen.output, args)


@pytest.mark.parametrize("artifact", ["queries", "provenance", "protocol", "config", "exclusions"])
def test_frozen_input_content_changes_are_rejected(frozen, artifact):
    path = getattr(frozen, artifact)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input has changed"):
        phase1.verify_freeze(frozen.output)


def test_relevant_code_change_rejected(frozen, monkeypatch):
    monkeypatch.setattr(phase1, "code_identity", lambda: {
        "git_head": "fixture-head", "implementation_sha256": {"search.py": "changed"}})
    with pytest.raises(ValueError, match="code identity"):
        phase1.verify_freeze(frozen.output)


def test_unrelated_commit_does_not_invalidate_source_fingerprints(frozen, monkeypatch):
    monkeypatch.setattr(phase1, "code_identity", lambda: {
        "git_head": "new-unrelated-head", "implementation_sha256": {"search.py": "fixture-source"}})
    assert phase1.verify_freeze(frozen.output)["code"]["git_head"] == "fixture-head"


def test_environment_change_after_joint_preflight_is_rejected(frozen):
    sys.modules["retrieval_baseline.preflight"].runtime_identity = lambda: {
        "python": {"executable": "another-interpreter", "version": "3.12"},
        "packages": {"synthetic": "1.0"}}
    with pytest.raises(ValueError, match="interpreter or dependency"):
        phase1.verify_freeze(frozen.output)


@pytest.mark.parametrize("source_name", ["search.py", "reranker.py"])
def test_inference_source_change_after_preflight_prevents_freezing(frozen, tmp_path, source_name):
    preflight = sys.modules["retrieval_baseline.preflight"]
    changed = {**preflight.inference_implementation_fingerprints(), source_name: "modified-source"}
    preflight.inference_implementation_fingerprints = lambda: changed
    frozen.output = tmp_path / "changed-implementation-freeze.json"
    with pytest.raises(ValueError, match="Inference implementation differs from joint preflight"):
        phase1.freeze_experiment(frozen)
    assert not frozen.output.exists()


def test_preflight_without_inference_source_binding_is_rejected(frozen, tmp_path):
    preflight = phase1._read(frozen.preflight)
    del preflight["inference_implementation_sha256"]
    write_json(frozen.preflight, preflight)
    frozen.output = tmp_path / "old-preflight-freeze.json"
    with pytest.raises(ValueError, match="Inference implementation differs from joint preflight"):
        phase1.freeze_experiment(frozen)


def test_static_only_preflight_cannot_freeze(frozen, tmp_path):
    preflight = phase1._read(frozen.preflight)
    preflight["status"] = "not_ready"
    preflight["inference"] = {"status": "not_run", "joint_hybrid_reranker_passed": False}
    write_json(frozen.preflight, preflight)
    frozen.output = tmp_path / "static-freeze.json"
    with pytest.raises(ValueError, match="joint hybrid/reranker inference"):
        phase1.freeze_experiment(frozen)
    assert not frozen.output.exists()


def test_forged_audit_cannot_skip_an_eligible_candidate(frozen, tmp_path):
    audit = phase1._rows(frozen.sampling_audit)
    audit[1]["decision"] = "excluded"
    audit[1]["first_exclusion_reason"] = "out_of_scope"
    write_rows(frozen.sampling_audit, audit)
    frozen.output = tmp_path / "forged-freeze.json"
    with pytest.raises(ValueError, match="decisions/order"):
        phase1.freeze_experiment(frozen)


def test_swapping_partitions_is_rejected_even_with_valid_six_four_count(frozen, tmp_path):
    rows = phase1._rows(frozen.provenance)
    first = next(row for row in rows if row["partition"] == "calibration")
    second = next(row for row in rows if row["partition"] == "evaluation")
    first["partition"], second["partition"] = second["partition"], first["partition"]
    write_rows(frozen.provenance, rows)
    frozen.output = tmp_path / "partition-freeze.json"
    with pytest.raises(ValueError, match="partition or association"):
        phase1.freeze_experiment(frozen)


def test_pending_exclusion_review_cannot_finalize(prepared, tmp_path):
    exclusions = phase1._read(prepared.exclusions)
    exclusions["reviewed_by"] = ""
    write_json(prepared.exclusions, exclusions)
    prepared.output = tmp_path / "pending-exclusion-prepared"
    phase1.prepare_queries(prepared)
    args = final_args(prepared, tmp_path)
    with pytest.raises(ValueError, match="reviewed_by"):
        phase1.finalize_queries(args)


def test_address_syntax_validation_does_not_open_an_index():
    phase1._syntax_check("测试花园")
    with pytest.raises(ValueError, match="unsupported"):
        phase1._syntax_check("测试花园二期")


def test_source_file_hash_coverage():
    assert {"case_eval.py", "phase1.py", "preflight.py", "reranker.py", "search.py", "address.py",
            "lexical.py", "hybrid.py", "common.py", "encoder.py", "dense.py"}.issubset(
                phase1.SOURCE_FILES)
    assert Path(phase1.__file__).name == "phase1.py"


@pytest.fixture
def committed_code(tmp_path, monkeypatch):
    root = tmp_path / "code-repository"
    package = root / "retrieval_baseline"
    package.mkdir(parents=True)
    (package / "search.py").write_text("# synthetic committed search\n", encoding="utf-8")

    def git(*arguments):
        return subprocess.run(
            ["git", "-c", "user.name=Synthetic Fixture", "-c", "user.email=fixture@example.test",
             "-c", "commit.gpgsign=false", *arguments], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()

    git("init")
    git("config", "core.autocrlf", "false")
    git("add", "retrieval_baseline/search.py")
    git("commit", "-m", "Synthetic initial implementation")
    monkeypatch.setattr(phase1, "__file__", str(package / "phase1.py"))
    monkeypatch.setattr(phase1, "SOURCE_FILES", ("search.py",))
    return root, package, git


@pytest.mark.parametrize("staged", [False, True])
def test_code_identity_rejects_uncommitted_related_source(committed_code, staged):
    _, package, git = committed_code
    (package / "search.py").write_text("# uncommitted implementation\n", encoding="utf-8")
    if staged:
        git("add", "retrieval_baseline/search.py")
    with pytest.raises(ValueError, match="match the recorded Git commit"):
        phase1.code_identity()


def test_code_identity_rejects_related_file_not_in_commit(committed_code, monkeypatch):
    _, package, _ = committed_code
    (package / "new_source.py").write_text("# untracked implementation\n", encoding="utf-8")
    monkeypatch.setattr(phase1, "SOURCE_FILES", ("search.py", "new_source.py"))
    with pytest.raises(ValueError, match="match the recorded Git commit"):
        phase1.code_identity()


def test_code_identity_allows_unrelated_dirty_files_and_commits(committed_code):
    root, package, git = committed_code
    original = phase1.code_identity()
    assert original["git_head"] == git("rev-parse", "HEAD")
    assert original["implementation_sha256"] == {"search.py": file_hash(package / "search.py")}
    notes = root / "unrelated-notes.md"
    notes.write_text("Unrelated local report\n", encoding="utf-8")
    assert phase1.code_identity() == original
    git("add", "unrelated-notes.md")
    git("commit", "-m", "Synthetic unrelated documentation")
    updated = phase1.code_identity()
    assert updated["git_head"] != original["git_head"]
    assert updated["implementation_sha256"] == original["implementation_sha256"]


@pytest.fixture
def portable_run(frozen, tmp_path):
    directory = tmp_path / "server-run"
    directory.mkdir()
    artifacts = phase1.snapshot_freeze(frozen.output, directory)
    # Collection may normalize surrounding query whitespace when saving queries.
    write_json(directory / "queries.json", phase1.validate_formal_queries(
        phase1._read(frozen.queries)))
    manifest = {"experiment_kind": "formal", "freeze_manifest_sha256":
                artifacts["freeze-manifest.json"], "artifacts": {
                    **artifacts, "queries.json": file_hash(directory / "queries.json")}}
    return directory, manifest


def test_snapshot_copies_exact_original_bytes(portable_run, frozen):
    directory, manifest = portable_run
    assert (directory / "freeze-manifest.json").read_bytes() == frozen.output.read_bytes()
    for key in phase1.ARTIFACTS:
        local = directory / "frozen-inputs" / phase1._snapshot_filename(key)
        assert local.read_bytes() == getattr(frozen, key).read_bytes()
    assert phase1.verify_run_freeze(directory, manifest) == phase1._read(frozen.output)


def test_portable_run_verifies_after_move_without_original_host_or_resources(
        portable_run, frozen, tmp_path, monkeypatch):
    directory, manifest = portable_run
    moved = tmp_path / "offline-copy"
    shutil.copytree(directory, moved)
    original = phase1._read(frozen.output)
    shutil.rmtree(frozen.dataset)
    for key in phase1.ARTIFACTS:
        getattr(frozen, key).unlink()
    frozen.output.unlink()
    shutil.rmtree(directory)

    def forbid_host_access(*args, **kwargs):
        raise AssertionError("Offline scoring must not access collection-host resources")

    for name in ("verify_freeze", "runtime_config", "code_identity", "_verify_git_commit"):
        monkeypatch.setattr(phase1, name, forbid_host_access)
    preflight = sys.modules["retrieval_baseline.preflight"]
    preflight.resource_fingerprints = forbid_host_access
    preflight.runtime_identity = forbid_host_access
    assert phase1.verify_run_freeze(moved, manifest) == original


@pytest.mark.parametrize("relative", [
    "freeze-manifest.json", "frozen-inputs/manifest.json", "frozen-inputs/queries.json",
    "frozen-inputs/provenance.jsonl", "frozen-inputs/config.json", "queries.json",
])
def test_portable_snapshot_tampering_is_rejected(portable_run, relative):
    directory, manifest = portable_run
    path = directory / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mismatch|changed|differs"):
        phase1.verify_run_freeze(directory, manifest)


def rewrite_snapshot_manifest(directory, run_manifest, snapshot):
    path = directory / "frozen-inputs/manifest.json"
    write_json(path, snapshot)
    run_manifest["artifacts"]["frozen-inputs/manifest.json"] = file_hash(path)


@pytest.mark.parametrize("filename", [
    "../queries.json", "/tmp/queries.json", "C:\\private\\queries.json", "config.json",
])
def test_portable_snapshot_rejects_escaped_or_wrong_mapping(portable_run, filename):
    directory, manifest = portable_run
    snapshot = phase1._read(directory / "frozen-inputs/manifest.json")
    snapshot["artifacts"]["queries"]["filename"] = filename
    rewrite_snapshot_manifest(directory, manifest, snapshot)
    with pytest.raises(ValueError, match="mapping differs"):
        phase1.verify_run_freeze(directory, manifest)


def test_portable_snapshot_manifest_cannot_omit_inputs(portable_run):
    directory, manifest = portable_run
    snapshot = phase1._read(directory / "frozen-inputs/manifest.json")
    del snapshot["artifacts"]["provenance"]
    rewrite_snapshot_manifest(directory, manifest, snapshot)
    with pytest.raises(ValueError, match="Incomplete or mismatched"):
        phase1.verify_run_freeze(directory, manifest)


def test_portable_snapshot_digest_must_match_original_freeze(portable_run):
    directory, manifest = portable_run
    snapshot = phase1._read(directory / "frozen-inputs/manifest.json")
    snapshot["artifacts"]["protocol"]["sha256"] = "0" * 64
    rewrite_snapshot_manifest(directory, manifest, snapshot)
    with pytest.raises(ValueError, match="mapping differs"):
        phase1.verify_run_freeze(directory, manifest)


def test_portable_snapshot_rejects_missing_file(portable_run):
    directory, manifest = portable_run
    (directory / "frozen-inputs/provenance.jsonl").unlink()
    with pytest.raises(ValueError, match="Missing or escaped"):
        phase1.verify_run_freeze(directory, manifest)


def test_portable_snapshot_rejects_symlink_escape(portable_run, frozen):
    directory, manifest = portable_run
    path = directory / "frozen-inputs/queries.json"
    path.unlink()
    path.symlink_to(frozen.queries)
    with pytest.raises(ValueError, match="symlinks"):
        phase1.verify_run_freeze(directory, manifest)


def test_portable_snapshot_rejects_symlinked_parent_directory(portable_run, tmp_path):
    directory, manifest = portable_run
    snapshot = directory / "frozen-inputs"
    external = tmp_path / "outside-run-inputs"
    shutil.move(snapshot, external)
    snapshot.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        phase1.verify_run_freeze(directory, manifest)


def test_portable_run_queries_cannot_diverge_with_recomputed_hash(portable_run):
    directory, manifest = portable_run
    path = directory / "queries.json"
    queries = phase1._read(path)
    scenario = next(row["scenario_id"] for row in queries if row["mode"] == "combined")
    for row in queries:
        if row["scenario_id"] == scenario and row["mode"] in {"problem", "combined"}:
            row["query"] = "另一个完全不同的诉求"
    write_json(path, queries)
    manifest["artifacts"]["queries.json"] = file_hash(path)
    with pytest.raises(ValueError, match="differ from the portable"):
        phase1.verify_run_freeze(directory, manifest)


def test_snapshot_refuses_changed_original_input(frozen, tmp_path):
    frozen.protocol.write_bytes(frozen.protocol.read_bytes() + b"\n")
    directory = tmp_path / "failed-snapshot"
    directory.mkdir()
    with pytest.raises(ValueError, match="changed during snapshot"):
        phase1.snapshot_freeze(frozen.output, directory)
    assert not (directory / "frozen-inputs/manifest.json").exists()


def test_portable_run_rejects_drill_kind(portable_run):
    directory, manifest = portable_run
    manifest["experiment_kind"] = "drill"
    with pytest.raises(ValueError, match="Only a formal run"):
        phase1.verify_run_freeze(directory, manifest)
