"""Private atomic progress, human label snapshots and append-only revision records."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path

from .common import file_hash
from .reranker import digest
from .search import _write_private_json

LABEL_FREEZE_VERSION = "case-label-freeze-v1"


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_status(path: Path, value: dict) -> None:
    """Only progress files may be replaced; completed artifacts use exclusive writes."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".status-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, allow_nan=False, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def copy_private(source: Path, target: Path) -> None:
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output, source.open("rb") as handle:
        shutil.copyfileobj(handle, output)


def effective_grades(run_dir: Path, labels_dir: Path) -> dict[str, str]:
    from .case_eval import load_labels

    result = {}
    for partition in ("calibration", "evaluation"):
        _, grades = load_labels(run_dir, labels_dir, partition=partition, required_depth=5)
        result[partition] = digest(sorted((qid, sid, g) for (qid, sid), g in grades.items()))
    return result


def _timestamp(value, description: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{description} requires a timezone")
    return parsed


def _label_rows(labels_dir: Path) -> dict[str, dict]:
    with (labels_dir / "judgments.tsv").open(encoding="utf-8-sig", newline="") as source:
        return {row["annotation_id"]: row for row in csv.DictReader(source, delimiter="\t")}


def _grade_fields(mode: str) -> tuple[str, ...]:
    return {
        "problem": ("problem_grade",),
        "address": ("address_grade",),
        "combined": ("problem_grade", "address_grade"),
    }[mode]


def _row_grades(row: dict) -> dict[str, int]:
    return {field: int(row[field]) for field in _grade_fields(row["mode"])}


def _review_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise ValueError("A human review-log.jsonl is required before freezing labels")
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Human review entries must be JSON objects")
        key = row.get("annotation_id")
        if not isinstance(key, str) or key in records:
            raise ValueError("Review log has invalid or duplicate annotation identity")
        records[key] = row
    return records


def _reviews(
    run_dir: Path,
    labels_dir: Path,
    *,
    annotator: str,
    previous_labels: Path | None = None,
    previous_freeze: dict | None = None,
) -> dict:
    from .case_eval import load_runs, pool_rows

    pool_manifest = json.loads((labels_dir / "manifest.json").read_text(encoding="utf-8"))
    pool = pool_rows(load_runs(run_dir), pool_manifest["pool_depth"])
    labels = _label_rows(labels_dir)
    records = _review_rows(labels_dir / "review-log.jsonl")
    old_labels = _label_rows(previous_labels) if previous_labels is not None else {}
    old_reviews = (
        _review_rows(previous_labels / "review-log.jsonl") if previous_labels is not None else {}
    )
    changed = {
        key for key in labels
        if old_labels and _row_grades(labels[key]) != _row_grades(old_labels[key])
    }
    if not changed.issubset(records):
        raise ValueError("Every revised grade pair requires a new human review")
    reviewed, modes, reviewers, independent_reviewers = set(), set(), set(), set()
    methods = {"independent": 0, "delayed_self": 0}
    annotators_by_pair = {}
    for key, row in records.items():
        if key not in pool:
            raise ValueError("Review log has unknown annotation identity")
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise ValueError("Review log requires a human reviewer")
        reviewer = row["reviewer"].strip()
        timestamp = _timestamp(row.get("reviewed_at"), "Review timestamp")
        if row.get("method") not in methods:
            raise ValueError("Review method must be independent or delayed_self")
        actor = annotator
        # Unchanged inherited reviews retain their original annotator identity even
        # when another human performs a notes-only revision in this version.
        if previous_freeze is not None and row == old_reviews.get(key) and key not in changed:
            actor = previous_freeze["review"]["annotators_by_pair"][key]
        if row["method"] == "independent" and reviewer == actor:
            raise ValueError("An independent reviewer must differ from the annotator")
        if row["method"] == "delayed_self":
            annotated = _timestamp(row.get("annotated_at"), "Annotation timestamp")
            if reviewer != actor or (timestamp - annotated).total_seconds() < 86400:
                raise ValueError("Delayed self-review requires the same human at least 24h later")
        if not isinstance(row.get("resolution"), str) or not row["resolution"].strip():
            raise ValueError("Review log requires the agreement or adjudication outcome")
        fields = _grade_fields(pool[key]["query"]["mode"])
        for name in ("initial_grades", "review_grades", "final_grades"):
            snapshot = row.get(name)
            if (not isinstance(snapshot, dict) or set(snapshot) != set(fields)
                    or any(type(value) is not int or value not in {0, 1, 2}
                           for value in snapshot.values())):
                raise ValueError("Review grade snapshots require applicable integer grades 0/1/2")
        if row["final_grades"] != _row_grades(labels[key]):
            raise ValueError("Review final grades differ from current judgments")
        if key in changed:
            if previous_freeze is None or timestamp <= _timestamp(
                previous_freeze["frozen_at"], "Previous freeze timestamp"
            ):
                raise ValueError("A revised grade requires review after the previous freeze")
            if row["initial_grades"] != _row_grades(old_labels[key]):
                raise ValueError("A revision review must preserve the previous initial grades")
        reviewed.add(key)
        modes.add(pool[key]["query"]["mode"])
        reviewers.add(reviewer)
        methods[row["method"]] += 1
        annotators_by_pair[key] = actor
        if row["method"] == "independent":
            independent_reviewers.add(reviewer)
    required_modes = {item["query"]["mode"] for item in pool.values()}
    if len(reviewed) < math.ceil(len(pool) * 0.2) or not required_modes.issubset(modes):
        raise ValueError("Review at least 20% of pairs, covering every nonempty mode")
    return {
        "reviewed_pairs": len(reviewed), "pool_pairs": len(pool),
        "modes": sorted(modes), "reviewers": sorted(reviewers), "methods": methods,
        "annotator_count": len(set(annotators_by_pair.values()) or {annotator}),
        "independent_reviewer_count": len(independent_reviewers),
        "annotators_by_pair": annotators_by_pair,
    }


def _previous_revision(run_dir: Path, labels_dir: Path, *, visited: set[Path] | None = None):
    path = labels_dir / "revision-context.json"
    if not path.exists():
        return None, None, None
    context = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(context.get("reason"), str) or not context["reason"].strip():
        raise ValueError("A label revision needs a human correction reason")
    recorded_path = context.get("previous_labels")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ValueError("Revision context requires its preceding label snapshot")
    snapshot_name = recorded_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if snapshot_name in {"", ".", ".."}:
        raise ValueError("Invalid preceding label snapshot name")
    candidates = (labels_dir.parent / snapshot_name, Path(recorded_path))
    previous_path = next((
        path for path in candidates
        if (path / "label-freeze.json").is_file()
        and file_hash(path / "label-freeze.json") == context["previous_freeze_sha256"]
    ), None)
    if previous_path is None:
        raise ValueError("Previous label snapshot is missing or changed")
    before = verify_label_freeze(run_dir, previous_path, _visited=visited)
    if file_hash(previous_path / "label-freeze.json") != context["previous_freeze_sha256"]:
        raise ValueError("Previous label snapshot changed")
    return context, previous_path, before


def _revision_record(
    labels_dir: Path, previous_labels: Path, context: dict, before: dict,
    grades: dict, *, annotator: str, review: dict,
) -> dict:
    old_rows, new_rows = _label_rows(previous_labels), _label_rows(labels_dir)
    changed_grades = sorted(
        key for key in new_rows if _row_grades(new_rows[key]) != _row_grades(old_rows[key])
    )
    changed_rows = sorted(key for key in new_rows if new_rows[key] != old_rows[key])
    changed = [key for key in grades if grades[key] != before["effective_grades_sha256"][key]]
    return {
        "version": "case-label-revision-v1", "operator": annotator,
        "reason": context["reason"].strip(),
        "previous_labels": context["previous_labels"],
        "previous_freeze_sha256": context["previous_freeze_sha256"],
        "previous_judgments_file_sha256": before["judgments_file_sha256"],
        "judgments_file_sha256": file_hash(labels_dir / "judgments.tsv"),
        "review_log_sha256": file_hash(labels_dir / "review-log.jsonl"),
        "revision_context_sha256": file_hash(labels_dir / "revision-context.json"),
        "changed_effective_partitions": changed, "changed_grade_pairs": changed_grades,
        "changed_rows": changed_rows, "reviewers": review["reviewers"],
        "policy_action": "recalibrate" if "calibration" in changed else "keep_thresholds",
        "report_action": "recompute_all_routes" if changed else "refresh_diagnostics",
        "supersedes_label_freeze_sha256": context["previous_freeze_sha256"],
        "affected_artifacts": "Reports bound to the previous label snapshot require replacement; "
                              "a new report is a revision of the same observed samples.",
    }


def _append_revision_index(labels_dir: Path, revision: dict) -> None:
    event = {
        "version": "case-label-revision-index-v1", "recorded_at": now(),
        "labels": str(labels_dir.resolve()),
        "label_freeze_sha256": file_hash(labels_dir / "label-freeze.json"),
        "revision_record_sha256": file_hash(labels_dir / "revision-record.json"),
        "previous_freeze_sha256": revision["previous_freeze_sha256"],
    }
    path = labels_dir.parent / "label-revisions.jsonl"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as target:
            target.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
    except OSError:
        warnings.warn(
            "Label revision is frozen with its authoritative audit record; "
            "the optional global revision index could not be updated.",
            RuntimeWarning, stacklevel=2,
        )


def freeze_labels(run_dir: Path, labels_dir: Path, *, annotator: str) -> dict:
    """Record human-completed labels; never fill grades or review decisions."""
    from .case_eval import require_formal_run

    run = require_formal_run(run_dir)
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError("A human annotator is required")
    annotator = annotator.strip()
    if (labels_dir / "label-freeze.json").exists():
        raise FileExistsError("Labels already frozen; create a revision directory")
    grades = effective_grades(run_dir, labels_dir)
    context, previous_labels, before = _previous_revision(
        run_dir, labels_dir, visited={labels_dir.resolve()}
    )
    review = _reviews(
        run_dir, labels_dir, annotator=annotator,
        previous_labels=previous_labels, previous_freeze=before,
    )
    record = {
        "version": LABEL_FREEZE_VERSION, "frozen_at": now(), "annotator": annotator,
        "run_manifest_sha256": file_hash(run_dir / "manifest.json"),
        "freeze_manifest_sha256": run["freeze_manifest_sha256"],
        "pool_manifest_sha256": file_hash(labels_dir / "manifest.json"),
        "judgments_file_sha256": file_hash(labels_dir / "judgments.tsv"),
        "review_log_sha256": file_hash(labels_dir / "review-log.jsonl"),
        "effective_grades_sha256": grades, "review": review,
    }
    if before is not None:
        revision = _revision_record(
            labels_dir, previous_labels, context, before, grades, annotator=annotator, review=review
        )
        revision_path = labels_dir / "revision-record.json"
        if revision_path.exists():
            # A failed final exclusive write can be retried only with the exact same audit.
            if json.loads(revision_path.read_text(encoding="utf-8")) != revision:
                raise ValueError("Revision audit already exists for different inputs")
        else:
            _write_private_json(revision_path, revision)
        record["revision"] = revision
        record["revision_record_sha256"] = file_hash(revision_path)
    _write_private_json(labels_dir / "label-freeze.json", record)
    if before is not None:
        _append_revision_index(labels_dir, record["revision"])
    return record


def verify_label_freeze(
    run_dir: Path, labels_dir: Path, *, _visited: set[Path] | None = None
) -> dict:
    from .case_eval import require_formal_run

    visited = set() if _visited is None else set(_visited)
    if labels_dir.resolve() in visited:
        raise ValueError("Cyclic label revision ancestry is invalid")
    visited.add(labels_dir.resolve())
    run = require_formal_run(run_dir)
    record = json.loads((labels_dir / "label-freeze.json").read_text(encoding="utf-8"))
    annotator = record.get("annotator")
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError("Frozen labels require a human annotator")
    _timestamp(record.get("frozen_at"), "Freeze timestamp")
    grades = effective_grades(run_dir, labels_dir)
    expected = {
        "version": LABEL_FREEZE_VERSION,
        "run_manifest_sha256": file_hash(run_dir / "manifest.json"),
        "freeze_manifest_sha256": run["freeze_manifest_sha256"],
        "pool_manifest_sha256": file_hash(labels_dir / "manifest.json"),
        "judgments_file_sha256": file_hash(labels_dir / "judgments.tsv"),
        "review_log_sha256": file_hash(labels_dir / "review-log.jsonl"),
        "effective_grades_sha256": grades,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen labels changed; create and freeze a new label revision")
    context, previous_labels, before = _previous_revision(run_dir, labels_dir, visited=visited)
    review = _reviews(
        run_dir, labels_dir, annotator=annotator,
        previous_labels=previous_labels, previous_freeze=before,
    )
    if record.get("review") != review:
        raise ValueError("Frozen human review record mismatch")
    if before is not None:
        revision_path = labels_dir / "revision-record.json"
        revision = _revision_record(
            labels_dir, previous_labels, context, before, grades, annotator=annotator, review=review
        )
        if (record.get("revision_record_sha256") != file_hash(revision_path)
                or record.get("revision") != revision
                or json.loads(revision_path.read_text(encoding="utf-8")) != revision):
            raise ValueError("Frozen label revision audit mismatch")
    elif record.get("revision") is not None or record.get("revision_record_sha256") is not None:
        raise ValueError("Frozen label revision is missing its ancestry")
    return record


def revise_labels(run_dir: Path, labels_dir: Path, output: Path, *, reason: str) -> dict:
    """Copy an immutable label version to an unfrozen working version."""
    verify_label_freeze(run_dir, labels_dir)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A label revision needs a reason")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name in ("manifest.json", "judgments.tsv", "review-log.jsonl"):
        copy_private(labels_dir / name, output / name)
    context = {"previous_labels": str(labels_dir.resolve()),
               "previous_freeze_sha256": file_hash(labels_dir / "label-freeze.json"),
               "reason": reason.strip(), "created_at": now()}
    _write_private_json(output / "revision-context.json", context)
    return context
