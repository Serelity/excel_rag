"""Prepare human-reviewed dev scenarios and enforce the phase-one freeze contract.

No command reads the original test query/qrel files or assigns human relevance labels.
The worksheet is private and deliberately incomplete until a reviewer fills it in.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from .common import content_key, file_hash, readonly, stable_rank

FREEZE_VERSION = "phase1-freeze-v1"
SNAPSHOT_VERSION = "phase1-frozen-input-snapshot-v1"
SAMPLING_VERSION = "phase1-sampling-v1"
EXCLUSIONS_VERSION = "phase1-source-exclusions-v1"
PROTOCOL_VERSION = "phase1-protocol-v1"
MODES = ("problem", "address", "combined")
ARTIFACTS = (
    "queries", "provenance", "protocol", "config", "preflight", "exclusions",
    "sampling_audit", "id_map",
)
SOURCE_FILES = (
    "case_eval.py", "case_lifecycle.py", "phase1.py", "preflight.py", "reranker.py",
    "search.py", "address.py",
    "lexical.py", "hybrid.py", "common.py", "encoder.py", "dense.py", "dataset.py",
)
REVIEW_FIELDS = (
    "reviewed_by", "reviewed_at", "scope_status", "source_review_status",
    "intent_review_status", "temporal_review_status", "query_as_of", "information_basis",
    "removed_followup_information", "address_review_status", "near_duplicate_review_status",
    "related_event_group", "selection_reason", "rewrite_notes", "problem", "address", "notes",
)
TEMPORAL_STATUSES = {"supported_at_acceptance", "export_text_only", "rejected"}
REASON_PRIORITY = (
    ("scope_status", "out_of_scope", "out_of_scope"),
    ("temporal_review_status", "rejected", "known_followup_information"),
    ("address_review_status", "unsupported", "unsupported_address"),
    ("near_duplicate_review_status", "related_or_duplicate", "related_or_near_duplicate"),
    ("source_review_status", "rejected", "source_not_verified"),
    ("intent_review_status", "rejected", "intent_not_preserved"),
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write(path: Path, data, *, jsonl: bool = False) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "w", encoding="utf-8", newline="\n") as target:
        if jsonl:
            for row in data:
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(data, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")


def _nonempty(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing human review field: {key}")
    return value.strip()


def runtime_config(args) -> dict:
    from .preflight import runtime_configuration

    return runtime_configuration(args)


def _validate_protocol(protocol: dict) -> None:
    if protocol.get("schema_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol schema")
    for key in ("version", "experiment_id", "scenario_scope", "reviewed_by", "reviewed_at"):
        _nonempty(protocol, key)
    if protocol.get("selection_without_retrieval_results") is not True:
        raise ValueError("Protocol must confirm selection without retrieval results")
    if type(protocol.get("allow_temporal_uncertainty")) is not bool:
        raise ValueError("Protocol must declare whether temporal uncertainty is permitted")
    if protocol["allow_temporal_uncertainty"]:
        _nonempty(protocol, "temporal_limitation")


def _validate_exclusions(exclusions: dict, *, reviewed: bool = True) -> None:
    if exclusions.get("schema_version") != EXCLUSIONS_VERSION:
        raise ValueError("Unsupported source exclusions schema")
    _nonempty(exclusions, "version")
    for key in ("excluded_source_ids", "excluded_group_ids", "limitations"):
        if not isinstance(exclusions.get(key), list):
            raise ValueError(f"Exclusions require list field: {key}")
    if any(not isinstance(sid, str) or not sid.strip()
           for sid in exclusions["excluded_source_ids"]):
        raise ValueError("Excluded source IDs must be nonempty strings")
    if any(type(group) not in (str, int) for group in exclusions["excluded_group_ids"]):
        raise ValueError("Excluded group IDs must be strings or integers")
    if reviewed:
        for key in ("reviewed_by", "reviewed_at"):
            _nonempty(exclusions, key)
        if type(exclusions.get("history_complete")) is not bool:
            raise ValueError("Exclusion review must declare history_complete")
        if not exclusions["history_complete"] and not exclusions["limitations"]:
            raise ValueError("Incomplete debug history requires an explicit source limitation")


def _candidate_plan(dataset: Path, exclusions_path: Path) -> tuple[dict, list[dict]]:
    """Read dev/corpus records and only the old dev query file; never open test files."""
    exclusions = _read(exclusions_path)
    _validate_exclusions(exclusions, reviewed=False)
    old_path = dataset / "queries.dev.jsonl"
    old_queries = _rows(old_path)
    database = dataset / "dataset.sqlite3"
    manifest = _read(dataset / "manifest.json")
    if manifest.get("artifacts", {}).get("dataset.sqlite3") != file_hash(database):
        raise ValueError("Dataset manifest does not match dataset.sqlite3")
    if manifest.get("artifacts", {}).get("queries.dev.jsonl") != file_hash(old_path):
        raise ValueError("Original dev query file differs from the frozen dataset manifest")
    excluded = {str(group) for group in exclusions["excluded_group_ids"]}
    old_groups = set()
    with readonly(database) as db:
        db.row_factory = sqlite3.Row
        for row in old_queries:
            record = db.execute("SELECT group_id FROM records WHERE source_id=?",
                                (row.get("source_id"),)).fetchone()
            if record is None:
                raise ValueError("Old dev query source is missing from the dataset")
            old_groups.add(str(record["group_id"]))
        for source_id in exclusions["excluded_source_ids"]:
            record = db.execute("SELECT group_id FROM records WHERE source_id=?",
                                (source_id,)).fetchone()
            if record is None:
                raise ValueError("Excluded source ID is missing; use an auditable group ID")
            excluded.add(str(record["group_id"]))
        representatives = [dict(row) for row in db.execute(
            "SELECT r.rid,r.source_id,r.source_row,r.group_id,r.text_hash,r.content,r.call_time "
            "FROM records r JOIN (SELECT MIN(rid) rid FROM records WHERE split='dev' "
            "GROUP BY group_id) s ON r.rid=s.rid"
        )]
    rejected = []
    candidates = []
    for record in sorted(representatives, key=lambda row: stable_rank(42, row["source_id"])):
        if str(record["group_id"]) in old_groups | excluded:
            rejected.append({"source_id": record["source_id"], "group_id": record["group_id"],
                             "reason": "original_dev_debug_or_rehearsal_group"})
        else:
            if content_key(record["content"]) != record["text_hash"]:
                raise ValueError("Dataset record text hash is inconsistent")
            record["candidate_position"] = len(candidates) + 1
            candidates.append(record)
    plan = {
        "schema_version": SAMPLING_VERSION,
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "dataset_sqlite_sha256": file_hash(database),
        "old_dev_queries_sha256": file_hash(old_path),
        "exclusions_sha256": file_hash(exclusions_path),
        "representative_rule": "MIN(rid) per dev group; never substitute another record",
        "selection_seed": 42, "partition_seed": 43,
        "total_dev_groups": len(representatives), "candidate_groups": len(candidates),
        "excluded_groups": rejected,
        "candidate_order": [row["source_id"] for row in candidates],
    }
    return plan, candidates


def prepare_queries(args) -> dict:
    plan, candidates = _candidate_plan(args.dataset, args.exclusions)
    limit = getattr(args, "limit", 80)
    if type(limit) is not int or limit < 10:
        raise ValueError("Worksheet limit must be at least 10")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    exported = []
    for row in candidates[:limit]:
        row = copy.deepcopy(row)
        row["source_type"] = "dataset_dev"
        row["review"] = {key: "" for key in REVIEW_FIELDS}
        exported.append(row)
    plan["worksheet_count"] = len(exported)
    _write(args.output / "sampling-plan.json", plan)
    _write(args.output / "candidates.jsonl", exported, jsonl=True)
    _write(args.output / "protocol.template.json", {
        "schema_version": PROTOCOL_VERSION, "version": "", "experiment_id": "",
        "scenario_scope": "", "reviewed_by": "", "reviewed_at": "",
        "selection_without_retrieval_results": None, "allow_temporal_uncertainty": None,
        "temporal_limitation": "",
    })
    _write(args.output / "review-schema.json", {
        "instructions": "Humans review in candidate order; blank fields do not mean approval.",
        "scope_status": ["in_scope", "out_of_scope"],
        "source_review_status": ["verified", "rejected"],
        "intent_review_status": ["preserved", "rejected"],
        "temporal_review_status": sorted(TEMPORAL_STATUSES),
        "address_review_status": ["supported", "unsupported"],
        "near_duplicate_review_status": ["independent", "related_or_duplicate"],
        "query_as_of": "An evidenced time or the literal unknown; never infer from call_time.",
        "information_basis": "Evidence for query-time information, or explicit uncertainty.",
        "removed_followup_information": "Describe removal, or explicitly record none found.",
        "related_event_group": "Reviewed association key; shared related events share one key.",
        "problem": "Faithful rewritten complaint, without known follow-up information.",
        "address": "Supported named place/road; no retrieval result may decide eligibility.",
        "remaining_fields": "Reviewer/time, selection reason and rewrite notes are required.",
    })
    return plan


def _syntax_check(address: str) -> None:
    from .address import AddressSearcher

    class SyntaxOnly(AddressSearcher):
        def __init__(self):
            # Exercise the exact query parser without constructing an index or viewing results.
            pass

        def _matching_groups(self, entities):
            return set()

    SyntaxOnly().search(address, allow_broader=False)


def _review_reason(review: dict, protocol: dict) -> str | None:
    if not isinstance(review, dict):
        raise ValueError("Candidate requires a human review object")
    for key in ("reviewed_by", "reviewed_at", "query_as_of", "information_basis",
                "removed_followup_information", "selection_reason", "rewrite_notes"):
        _nonempty(review, key)
    allowed = {
        "scope_status": {"in_scope", "out_of_scope"},
        "source_review_status": {"verified", "rejected"},
        "intent_review_status": {"preserved", "rejected"},
        "temporal_review_status": TEMPORAL_STATUSES,
        "address_review_status": {"supported", "unsupported"},
        "near_duplicate_review_status": {"independent", "related_or_duplicate"},
    }
    for key, choices in allowed.items():
        if review.get(key) not in choices:
            raise ValueError(f"Missing or invalid human review: {key}")
    for field, rejected, reason in REASON_PRIORITY:
        if review[field] == rejected:
            return reason
    if review["temporal_review_status"] == "export_text_only":
        if not protocol["allow_temporal_uncertainty"]:
            raise ValueError("Temporal uncertainty requires a declared protocol limitation")
    for key in ("problem", "address", "related_event_group"):
        _nonempty(review, key)
    _syntax_check(review["address"].strip())
    return None


def _select(candidates: list[dict], worksheet: list[dict], protocol: dict) -> tuple[list, list]:
    selected, audit, definitions, events = [], [], set(), set()
    for position, source in enumerate(candidates):
        if position >= len(worksheet):
            raise ValueError(
                "Fewer than 10 eligible reviewed scenarios; review the next candidates")
        row = worksheet[position]
        if any(row.get(key) != value for key, value in source.items()):
            raise ValueError("Candidate order/source content differs from fixed representatives")
        review = copy.deepcopy(row.get("review"))
        reason = _review_reason(review, protocol)
        identities = set()
        if reason is None:
            problem, address = review["problem"].strip(), review["address"].strip()
            identities = {("problem", problem, None), ("address", address, None),
                          ("combined", problem, address)}
            if review["related_event_group"].strip() in events:
                reason = "related_or_near_duplicate"
            elif definitions & identities:
                reason = "duplicate_query_definition"
        entry = {"record_type": "candidate", **source, "review": review,
                 "decision": "excluded" if reason else "selected", "first_exclusion_reason": reason}
        # Source content is needed only in the original private worksheet, not the frozen audit.
        entry.pop("content")
        audit.append(entry)
        if reason is None:
            definitions.update(identities)
            events.add(review["related_event_group"].strip())
            selected.append(entry)
        if len(selected) == 10:
            return selected, audit
    raise ValueError("Fewer than 10 eligible reviewed scenarios; expand scope before retrieval")


def validate_formal_queries(rows: list[dict]) -> list[dict]:
    from .case_eval import validate_queries

    rows = validate_queries(rows)
    scenarios = {}
    for row in rows:
        for field, prefix in (("query_id", "q_"), ("scenario_id", "s_")):
            if re.fullmatch(prefix + r"[0-9a-f]{32}", row[field]) is None:
                raise ValueError("Formal query/scenario IDs must be opaque random identifiers")
        scenarios.setdefault(row["scenario_id"], []).append(row)
    if len(rows) != 30 or len(scenarios) != 10:
        raise ValueError("Formal experiment requires 10 scenarios and 30 queries")
    partitions = Counter()
    for group in scenarios.values():
        modes = {row["mode"]: row for row in group}
        if len(group) != 3 or set(modes) != set(MODES):
            raise ValueError("Every formal scenario requires exactly three modes")
        if modes["problem"]["query"] != modes["combined"]["query"]:
            raise ValueError("Combined and problem queries must have identical problem text")
        if modes["address"]["query"] != modes["combined"]["address"]:
            raise ValueError("Combined and address queries must have identical address text")
        _syntax_check(modes["address"]["query"])
        partitions[group[0]["partition"]] += 1
    if partitions != {"calibration": 6, "evaluation": 4}:
        raise ValueError("Formal partition requires six calibration and four evaluation scenarios")
    return rows


def _source_isolation(dataset: Path, provenance: list[dict]) -> None:
    fields = ("source_id", "group_id", "text_hash", "related_event_group")
    seen = {field: set() for field in fields}
    with readonly(dataset / "dataset.sqlite3") as db:
        for row in provenance:
            for field in fields:
                key = str(row[field])
                if key in seen[field]:
                    raise ValueError(f"Source/association leakage between scenarios: {field}")
                seen[field].add(key)
            overlap = db.execute(
                "SELECT 1 FROM records WHERE split='corpus' "
                "AND (source_id=? OR group_id=? OR text_hash=?) LIMIT 1",
                (row["source_id"], row["group_id"], row["text_hash"]),
            ).fetchone()
            if overlap:
                raise ValueError("Query source ID, association group or text overlaps the corpus")


def finalize_queries(args) -> dict:
    protocol = _read(args.protocol)
    _validate_protocol(protocol)
    _validate_exclusions(_read(args.exclusions))
    expected, candidates = _candidate_plan(args.dataset, args.exclusions)
    plan = _read(args.sampling_plan)
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Sampling plan changed after worksheet preparation")
    selected, audit = _select(candidates, _rows(args.worksheet), protocol)
    ordered = sorted(selected, key=lambda row: stable_rank(43, row["source_id"]))
    queries, provenance, mapping = [], [], []
    # UUIDs encode neither source IDs, selection order, partitions nor modes.
    for index, entry in enumerate(ordered):
        scenario_id = "s_" + uuid.uuid4().hex
        partition = "calibration" if index < 6 else "evaluation"
        review = entry["review"]
        source = {key: entry[key] for key in
                  ("source_id", "source_row", "rid", "group_id", "text_hash", "candidate_position")}
        source.update({"scenario_id": scenario_id, "partition": partition,
                       "source_type": "dataset_dev", "related_event_group":
                       review["related_event_group"].strip(), "review": review})
        provenance.append(source)
        for mode in MODES:
            row = {"query_id": "q_" + uuid.uuid4().hex, "scenario_id": scenario_id,
                   "partition": partition, "mode": mode,
                   "query": review["address" if mode == "address" else "problem"].strip()}
            if mode == "combined":
                row["address"] = review["address"].strip()
            queries.append(row)
            mapping.append({key: row[key] for key in
                            ("query_id", "scenario_id", "partition", "mode")})
            mapping[-1].update(source_id=entry["source_id"], group_id=entry["group_id"])
    queries.sort(key=lambda row: stable_rank(44, row["query_id"]))
    validate_formal_queries(queries)
    _source_isolation(args.dataset, provenance)
    summary = {"record_type": "summary", **expected,
               "stop_position": audit[-1]["candidate_position"], "selected_count": 10,
               "selection_counts": dict(Counter(row["first_exclusion_reason"] or "selected"
                                                for row in audit)),
               "worksheet_sha256": file_hash(args.worksheet),
               "protocol_sha256": file_hash(args.protocol)}
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    _write(args.output / "queries.json", queries)
    _write(args.output / "query-provenance.jsonl", provenance, jsonl=True)
    _write(args.output / "sampling-audit.jsonl", [summary, *audit], jsonl=True)
    _write(args.output / "query-id-map.json", {"schema_version": "phase1-query-id-map-v1",
                                             "mappings": mapping})
    return {"query_count": 30, "scenario_count": 10,
            "temporal_uncertainty_count": sum(row["review"]["temporal_review_status"]
                                              == "export_text_only" for row in provenance),
            "stop_position": summary["stop_position"]}


def _verify_inputs(args) -> dict:
    protocol = _read(args.protocol)
    _validate_protocol(protocol)
    _validate_exclusions(_read(args.exclusions))
    queries = validate_formal_queries(_read(args.queries))
    provenance = _rows(args.provenance)
    if len(provenance) != 10 or len({row.get("scenario_id") for row in provenance}) != 10:
        raise ValueError("Exactly one reviewed provenance record per scenario is required")
    plan, candidates = _candidate_plan(Path(args.dataset), Path(args.exclusions))
    audit = _rows(args.sampling_audit)
    if not audit or audit[0].get("record_type") != "summary":
        raise ValueError("Sampling audit summary is missing")
    summary = audit[0]
    if any(summary.get(key) != value for key, value in plan.items()):
        raise ValueError("Sampling audit does not match current dataset/exclusions")
    if summary.get("protocol_sha256") != file_hash(Path(args.protocol)):
        raise ValueError("Protocol changed after scenario selection")
    worksheet = []
    for position, entry in enumerate(audit[1:]):
        if position >= len(candidates):
            raise ValueError("Sampling audit exceeds available candidates")
        worksheet.append({**entry, "content": candidates[position]["content"]})
    selected, expected_audit = _select(candidates, worksheet, protocol)
    if audit[1:] != expected_audit or summary.get("stop_position") != len(expected_audit):
        raise ValueError("Sampling audit decisions/order/stopping rule differ from fixed selection")
    expected_counts = dict(Counter(row["first_exclusion_reason"] or "selected"
                                   for row in expected_audit))
    if summary.get("selected_count") != 10 or summary.get("selection_counts") != expected_counts:
        raise ValueError("Sampling audit counts differ from reviewed decisions")
    source_to_partition = {row["source_id"]: "calibration" if index < 6 else "evaluation"
                           for index, row in enumerate(sorted(
                               selected, key=lambda row: stable_rank(43, row["source_id"]))) }
    by_source = {row["source_id"]: row for row in selected}
    by_scenario = {row["scenario_id"]: row for row in provenance}
    if {row["scenario_id"] for row in queries} != set(by_scenario):
        raise ValueError("Provenance scenarios differ from query scenarios")
    for row in provenance:
        source = by_source.get(row.get("source_id"))
        if source is None or row.get("source_type") != "dataset_dev":
            raise ValueError("Provenance source is not a reviewed dev representative")
        for key in ("rid", "source_row", "group_id", "text_hash", "candidate_position", "review"):
            if row.get(key) != source[key]:
                raise ValueError(f"Provenance/source mismatch: {key}")
        expected_event = source["review"]["related_event_group"].strip()
        if (row.get("partition") != source_to_partition[row["source_id"]]
                or row.get("related_event_group") != expected_event):
            raise ValueError("Provenance partition or association differs from reviewed selection")
    mappings = []
    for query in queries:
        source = by_scenario[query["scenario_id"]]
        review = source["review"]
        expected = review["address" if query["mode"] == "address" else "problem"].strip()
        if query["query"] != expected or query["partition"] != source["partition"]:
            raise ValueError("Query differs from reviewed source or frozen partition")
        if query["mode"] == "combined" and query["address"] != review["address"].strip():
            raise ValueError("Combined address differs from reviewed source")
        mappings.append({**{key: query[key] for key in
                            ("query_id", "scenario_id", "partition", "mode")},
                         "source_id": source["source_id"], "group_id": source["group_id"]})
    id_map = _read(args.id_map)
    if (id_map.get("schema_version") != "phase1-query-id-map-v1"
            or sorted(id_map.get("mappings", []), key=lambda row: row["query_id"])
            != sorted(mappings, key=lambda row: row["query_id"])):
        raise ValueError("Private ID map differs from queries/provenance")
    _source_isolation(Path(args.dataset), provenance)
    return {"protocol_version": protocol["version"], "experiment_id": protocol["experiment_id"],
            "query_count": 30, "scenario_count": 10,
            "temporal_uncertainty_count": sum(row["review"]["temporal_review_status"]
                                              == "export_text_only" for row in provenance)}


def code_identity() -> dict:
    package = Path(__file__).resolve().parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, check=True,
                          capture_output=True, text=True).stdout.strip()
    prefix = subprocess.run(["git", "rev-parse", "--show-prefix"], cwd=package, check=True,
                            capture_output=True, text=True).stdout.strip()
    hashes = {name: file_hash(package / name) for name in SOURCE_FILES}
    for name, actual in hashes.items():
        committed = subprocess.run(["git", "show", f"{head}:{prefix}{name}"], cwd=package,
                                   capture_output=True)
        if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != actual:
            raise ValueError(
                "Formal freeze requires relevant source files to match the recorded Git commit; "
                "commit or restore related changes before freezing")
    return {"git_head": head, "implementation_sha256": hashes}


def _verify_git_commit(head: str) -> None:
    if not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head) is None:
        raise ValueError("Invalid frozen Git commit identity")
    result = subprocess.run(["git", "cat-file", "-e", head + "^{commit}"],
                            cwd=Path(__file__).resolve().parent, capture_output=True)
    if result.returncode:
        raise ValueError("Frozen Git commit is no longer available for audit")


def _freeze_state(args) -> dict:
    from .preflight import (
        inference_implementation_fingerprints,
        resource_fingerprints,
        runtime_identity,
    )

    runtime = runtime_config(args)
    if (runtime.get("retriever") != "hybrid" or runtime.get("case_k") != 50
            or runtime.get("max_terms") != 32):
        raise ValueError("Phase-one formal profile requires hybrid, case_k=50 and max_terms=32")
    if _read(args.config) != runtime:
        raise ValueError("Config file differs from actual effective runtime configuration")
    review = _verify_inputs(args)
    resources = resource_fingerprints(args)
    preflight = _read(args.preflight)
    if (preflight.get("status") != "passed"
            or preflight.get("inference", {}).get("status") != "passed"
            or preflight.get("inference", {}).get("joint_hybrid_reranker_passed") is not True):
        raise ValueError(
            "Formal freezing requires passed preflight and joint hybrid/reranker inference")
    if preflight.get("resources") != resources or preflight.get("runtime_config") != runtime:
        raise ValueError("Preflight resources/config differ from the effective formal experiment")
    inference_implementation = inference_implementation_fingerprints()
    if preflight.get("inference_implementation_sha256") != inference_implementation:
        raise ValueError("Inference implementation differs from joint preflight; rerun preflight")
    environment = runtime_identity()
    if any(preflight.get("runtime", {}).get(key) != value for key, value in environment.items()):
        raise ValueError("Python interpreter or dependency versions differ from joint preflight")
    return {**review, "runtime_config": runtime, "resources": resources, "code": code_identity(),
            "runtime_identity": environment,
            "inference_implementation_sha256": inference_implementation}


def freeze_experiment(args) -> dict:
    state = _freeze_state(args)
    manifest = {"schema_version": FREEZE_VERSION, "created_at": _now(),
                "review_status": "complete", **state,
                "artifacts": {name: {"path": str(Path(getattr(args, name)).resolve()),
                                     "sha256": file_hash(Path(getattr(args, name)))}
                              for name in ARTIFACTS}}
    args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _write(args.output, manifest)
    return manifest


def _freeze_header(manifest: dict) -> None:
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != FREEZE_VERSION
            or manifest.get("review_status") != "complete"
            or manifest.get("query_count") != 30 or manifest.get("scenario_count") != 10):
        raise ValueError("Missing or incomplete formal freeze manifest")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACTS):
        raise ValueError("Freeze manifest artifact bindings are incomplete")
    for artifact in artifacts.values():
        if (not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str)
                or not artifact["path"].strip() or not isinstance(artifact.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is None):
            raise ValueError("Invalid freeze artifact binding")
    for key in ("runtime_config", "runtime_identity", "resources", "code"):
        if not isinstance(manifest.get(key), dict) or not manifest[key]:
            raise ValueError(f"Incomplete freeze identity: {key}")
    for key in ("experiment_id", "protocol_version", "created_at"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Incomplete freeze metadata: {key}")


def verify_freeze(path: Path, args=None) -> dict:
    """Revalidate frozen content and relevant code before any model construction.

    This checks live collection assets. Completed runs use verify_run_freeze to
    inspect portable input snapshots without reopening server models or indexes.
    With collect args, also prove that the actual configuration and queries match.
    """
    manifest = _read(path)
    _freeze_header(manifest)
    artifacts = manifest.get("artifacts", {})
    for artifact in artifacts.values():
        if file_hash(Path(artifact["path"])) != artifact.get("sha256"):
            raise ValueError("Frozen experiment input has changed")
    frozen_args = SimpleNamespace(**manifest["runtime_config"])
    for name, resource in manifest.get("resources", {}).items():
        setattr(frozen_args, name, Path(resource["path"]))
    for name, artifact in artifacts.items():
        setattr(frozen_args, name, Path(artifact["path"]))
    if args is not None:
        if runtime_config(args) != manifest["runtime_config"]:
            raise ValueError("Actual effective runtime configuration differs from freeze")
        if file_hash(Path(args.queries)) != artifacts["queries"]["sha256"]:
            raise ValueError("Collect queries differ from frozen queries")
    state = _freeze_state(frozen_args)
    _verify_git_commit(manifest.get("code", {}).get("git_head"))
    if (manifest.get("code", {}).get("implementation_sha256")
            != state["code"]["implementation_sha256"]):
        raise ValueError("Frozen relevant code identity no longer matches")
    # An unrelated commit is harmless when every relevant source file is unchanged.
    if any(manifest.get(key) != value for key, value in state.items() if key != "code"):
        raise ValueError("Frozen review/resources/config/code identity no longer matches")
    return manifest


def _snapshot_filename(name: str) -> str:
    return name + (".jsonl" if name in {"provenance", "sampling_audit"} else ".json")


def snapshot_freeze(manifest_path: Path, run_dir: Path) -> dict[str, str]:
    """Preserve all freeze inputs after the caller passes the live collect gate.

    Copies exact bytes, refuses overwrites, and verifies every copied input against
    the manifest. This does not replace verify_freeze's live resource validation.
    """
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw.decode("utf-8-sig"))
    _freeze_header(manifest)
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError("Create the private run directory before snapshotting freeze inputs")
    snapshot_dir = run_dir / "frozen-inputs"
    snapshot_dir.mkdir(mode=0o700, exist_ok=False)
    freeze_path = run_dir / "freeze-manifest.json"
    with os.fdopen(os.open(freeze_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "wb") as target:
        target.write(raw)
    snapshot = {"schema_version": SNAPSHOT_VERSION,
                "freeze_manifest_sha256": hashlib.sha256(raw).hexdigest(), "artifacts": {}}
    for name in ARTIFACTS:
        binding = manifest["artifacts"][name]
        filename = _snapshot_filename(name)
        destination = snapshot_dir / filename
        with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       "wb") as target, Path(binding["path"]).open("rb") as source:
            shutil.copyfileobj(source, target)
        actual = file_hash(destination)
        if actual != binding["sha256"]:
            raise ValueError("Frozen input changed during snapshot creation")
        snapshot["artifacts"][name] = {"filename": filename, "sha256": actual}
    _write(snapshot_dir / "manifest.json", snapshot)
    return {"freeze-manifest.json": snapshot["freeze_manifest_sha256"],
            "frozen-inputs/manifest.json": file_hash(snapshot_dir / "manifest.json")}


def _local_run_file(run_dir: Path, relative: str) -> Path:
    """Only regular files inside the run are allowed, including every path component."""
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise ValueError("Invalid local snapshot path")
    root = Path(run_dir).resolve()
    target = root
    for component in Path(relative).parts:
        target = target / component
        if target.is_symlink():
            raise ValueError("Snapshot paths must not contain symlinks")
    if not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError("Missing or escaped local snapshot file")
    return target


def verify_run_freeze(run_dir: Path, run_manifest: dict) -> dict:
    """Verify a completed run's portable freeze without consulting collection hosts.

    The caller also validates collection completion and paired-run artifact hashes.
    Frozen source paths, model weights, indexes, Git, CUDA and current dependency
    versions are intentionally not consulted when recomputing stored rankings.
    """
    if run_manifest.get("experiment_kind") != "formal":
        raise ValueError("Only a formal run may use the portable freeze snapshot")
    bindings = run_manifest.get("artifacts", {})
    if not isinstance(bindings, dict):
        raise ValueError("Missing formal run artifact bindings")
    freeze_path = _local_run_file(run_dir, "freeze-manifest.json")
    freeze_sha = file_hash(freeze_path)
    if (freeze_sha != run_manifest.get("freeze_manifest_sha256")
            or freeze_sha != bindings.get("freeze-manifest.json")):
        raise ValueError("Formal run freeze provenance mismatch")
    manifest = _read(freeze_path)
    _freeze_header(manifest)
    snapshot_path = _local_run_file(run_dir, "frozen-inputs/manifest.json")
    if file_hash(snapshot_path) != bindings.get("frozen-inputs/manifest.json"):
        raise ValueError("Frozen input snapshot manifest hash mismatch")
    snapshot = _read(snapshot_path)
    if (not isinstance(snapshot, dict) or snapshot.get("schema_version") != SNAPSHOT_VERSION
            or snapshot.get("freeze_manifest_sha256") != freeze_sha
            or not isinstance(snapshot.get("artifacts"), dict)
            or set(snapshot["artifacts"]) != set(ARTIFACTS)):
        raise ValueError("Incomplete or mismatched frozen input snapshot manifest")
    paths = {}
    for name in ARTIFACTS:
        binding = snapshot["artifacts"][name]
        if (not isinstance(binding, dict) or set(binding) != {"filename", "sha256"}
                or binding.get("filename") != _snapshot_filename(name)
                or binding.get("sha256") != manifest["artifacts"][name]["sha256"]):
            raise ValueError("Frozen input snapshot mapping differs from the original freeze")
        paths[name] = _local_run_file(run_dir, "frozen-inputs/" + binding["filename"])
        if file_hash(paths[name]) != binding["sha256"]:
            raise ValueError("Frozen input snapshot content has changed")
    frozen_queries = validate_formal_queries(_read(paths["queries"]))
    run_queries = _local_run_file(run_dir, "queries.json")
    if file_hash(run_queries) != bindings.get("queries.json"):
        raise ValueError("Saved run queries hash differs from its run manifest")
    if validate_formal_queries(_read(run_queries)) != frozen_queries:
        raise ValueError("Saved run queries differ from the portable formal freeze")
    return manifest


def main() -> None:
    from .search import add_runtime_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-queries")
    finalize = sub.add_parser("finalize-queries")
    for command in (prepare, finalize):
        for name in ("dataset", "exclusions", "output"):
            command.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=80)
    for name in ("worksheet", "sampling-plan", "protocol"):
        finalize.add_argument("--" + name, type=Path, required=True)
    freeze = sub.add_parser("freeze")
    add_runtime_arguments(freeze)
    freeze.add_argument("--retriever", choices=("hybrid",), default="hybrid")
    freeze.add_argument("--case-k", type=int, default=50)
    freeze.add_argument("--max-terms", type=int, default=32)
    for name in (*ARTIFACTS, "output"):
        freeze.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    action = {"prepare-queries": prepare_queries, "finalize-queries": finalize_queries,
              "freeze": freeze_experiment}[args.command]
    result = action(args)
    print(json.dumps({"status": "completed", "output": str(args.output),
                      "query_count": result.get("query_count"),
                      "candidate_groups": result.get("candidate_groups")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
