from __future__ import annotations

import csv
import json
import shutil

import pytest

from retrieval_baseline import case_eval
from retrieval_baseline import case_lifecycle as lifecycle
from retrieval_baseline.common import file_hash
from retrieval_baseline.tests import test_case_eval

bundle = test_case_eval.bundle


def _labels(directory):
    with (directory / "judgments.tsv").open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def _rewrite(directory, rows):
    with (directory / "judgments.tsv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=case_eval.FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _reviews(directory):
    return [json.loads(line) for line in (directory / "review-log.jsonl").read_text().splitlines()]


def _write_reviews(directory, rows):
    (directory / "review-log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


@pytest.fixture
def lifecycle_bundle(bundle, monkeypatch):
    run, labels = bundle
    test_case_eval.fill_labels(labels, fill_evaluation=True)
    records = []
    for row in _labels(labels):
        grades = lifecycle._row_grades(row)
        records.append({
            "annotation_id": row["annotation_id"], "reviewer": "reviewer",
            "reviewed_at": "2026-10-02T10:00:00+08:00", "method": "independent",
            "resolution": "Reviewed original content and agreed with these grades.",
            "initial_grades": grades, "review_grades": grades, "final_grades": grades,
        })
    _write_reviews(labels, records)
    monkeypatch.setattr(case_eval, "require_formal_run", lambda _: {
        "freeze_manifest_sha256": "synthetic-protocol-fingerprint",
    })
    monkeypatch.setattr(lifecycle, "now", lambda: "2026-10-03T12:00:00+08:00")
    return run, labels


def test_freeze_binds_human_grade_snapshots_and_rejects_overwrite(lifecycle_bundle):
    run, labels = lifecycle_bundle
    record = lifecycle.freeze_labels(run, labels, annotator="annotator")
    assert record == lifecycle.verify_label_freeze(run, labels)
    assert record["review"]["methods"] == {"independent": 9, "delayed_self": 0}
    assert record["review"]["independent_reviewer_count"] == 1
    assert record["review"]["annotator_count"] == 1
    with pytest.raises(FileExistsError):
        lifecycle.freeze_labels(run, labels, annotator="annotator")


@pytest.mark.parametrize("mutation", ["same_human", "missing_grades", "wrong_final", "bool_grade"])
def test_review_evidence_is_checked(lifecycle_bundle, mutation):
    run, labels = lifecycle_bundle
    records = _reviews(labels)
    if mutation == "same_human":
        records[0]["reviewer"] = " annotator "
    elif mutation == "missing_grades":
        del records[0]["initial_grades"]
    elif mutation == "wrong_final":
        records[0]["final_grades"]["problem_grade"] = 1
    else:
        records[0]["initial_grades"]["problem_grade"] = True
    _write_reviews(labels, records)
    with pytest.raises(ValueError, match="independent|snapshots|final grades"):
        lifecycle.freeze_labels(run, labels, annotator="annotator")
    assert not (labels / "label-freeze.json").exists()


@pytest.mark.parametrize("valid", [True, False])
def test_delayed_self_review_requires_same_person_and_24_hours(lifecycle_bundle, valid):
    run, labels = lifecycle_bundle
    records = _reviews(labels)
    for row in records:
        row.update(method="delayed_self", reviewer="annotator",
                   annotated_at="2026-10-01T10:00:00+08:00")
    if not valid:
        records[0]["annotated_at"] = "2026-10-01T10:00:01+08:00"
    _write_reviews(labels, records)
    if valid:
        result = lifecycle.freeze_labels(run, labels, annotator="annotator")
        assert result["review"]["methods"]["delayed_self"] == 9
    else:
        with pytest.raises(ValueError, match="same human at least 24h"):
            lifecycle.freeze_labels(run, labels, annotator="annotator")


def _revision(lifecycle_bundle, tmp_path):
    run, original = lifecycle_bundle
    previous = lifecycle.freeze_labels(run, original, annotator="annotator")
    new = tmp_path / "labels-002"
    lifecycle.revise_labels(run, original, new, reason="Human correction based on source evidence.")
    return run, original, new, previous


def _change_grade(directory, *, partition="calibration", field="problem_grade"):
    rows = _labels(directory)
    row = next(row for row in rows if row["query_id"].startswith(
        "dev" if partition == "calibration" else "eval"
    ) and row[field] == "2")
    old = lifecycle._row_grades(row)
    row[field] = "1"
    _rewrite(directory, rows)
    return row["annotation_id"], old, lifecycle._row_grades(row)


def _review_correction(directory, key, initial, final, *, late=True):
    records = _reviews(directory)
    record = next(row for row in records if row["annotation_id"] == key)
    record.update({
        "initial_grades": initial, "review_grades": final, "final_grades": final,
        "resolution": "Original text supports only partial relevance; corrected the grade.",
        "reviewed_at": "2026-10-04T12:00:00+08:00" if late else "2026-10-02T12:00:00+08:00",
    })
    _write_reviews(directory, records)


def test_revision_cannot_reuse_stale_review_or_alter_prior_grade_evidence(
    lifecycle_bundle, tmp_path,
):
    run, _, new, _ = _revision(lifecycle_bundle, tmp_path)
    key, initial, final = _change_grade(new)
    with pytest.raises(ValueError, match="final grades"):
        lifecycle.freeze_labels(run, new, annotator="annotator")
    _review_correction(new, key, initial, final, late=False)
    with pytest.raises(ValueError, match="after the previous freeze"):
        lifecycle.freeze_labels(run, new, annotator="annotator")
    _review_correction(new, key, final, final)
    with pytest.raises(ValueError, match="previous initial grades"):
        lifecycle.freeze_labels(run, new, annotator="annotator")
    assert not (new / "label-freeze.json").exists()


@pytest.mark.parametrize("partition", ["calibration", "evaluation"])
def test_revision_dependency_action_tracks_which_partition_changed(
    lifecycle_bundle, tmp_path, monkeypatch, partition,
):
    run, original, new, before = _revision(lifecycle_bundle, tmp_path)
    original_bytes = (original / "judgments.tsv").read_bytes()
    key, initial, final = _change_grade(new, partition=partition)
    _review_correction(new, key, initial, final)
    monkeypatch.setattr(lifecycle, "now", lambda: "2026-10-05T12:00:00+08:00")
    record = lifecycle.freeze_labels(run, new, annotator="annotator")
    assert record["revision"]["changed_effective_partitions"] == [partition]
    assert record["revision"]["changed_grade_pairs"] == [key]
    assert record["revision"]["policy_action"] == (
        "recalibrate" if partition == "calibration" else "keep_thresholds"
    )
    assert record["revision"]["supersedes_label_freeze_sha256"] == file_hash(
        original / "label-freeze.json"
    )
    assert record["revision_record_sha256"] == file_hash(new / "revision-record.json")
    assert lifecycle.verify_label_freeze(run, new) == record
    assert (original / "judgments.tsv").read_bytes() == original_bytes
    assert (
        record["effective_grades_sha256"][partition] != before["effective_grades_sha256"][partition]
    )


def test_notes_only_revision_records_new_operator_and_keeps_thresholds(lifecycle_bundle, tmp_path):
    run, _, new, before = _revision(lifecycle_bundle, tmp_path)
    rows = _labels(new)
    rows[0]["notes"] = "Human clarification of evidence, no grade changes."
    _rewrite(new, rows)
    # The prior independent reviewer can publish this notes-only revision without
    # being mislabeled as the annotator of the inherited original reviews.
    result = lifecycle.freeze_labels(run, new, annotator="reviewer")
    assert result["effective_grades_sha256"] == before["effective_grades_sha256"]
    assert result["revision"]["policy_action"] == "keep_thresholds"
    assert result["revision"]["report_action"] == "refresh_diagnostics"
    assert result["revision"]["operator"] == "reviewer"
    assert result["revision"]["reason"]
    assert result["revision"]["changed_rows"] == [rows[0]["annotation_id"]]
    assert lifecycle.verify_label_freeze(run, new) == result


@pytest.mark.parametrize("bundle", ["combined"], indirect=True)
def test_changed_component_with_same_effective_grade_still_requires_new_review(
    lifecycle_bundle, tmp_path, monkeypatch,
):
    run, _, new, before = _revision(lifecycle_bundle, tmp_path)
    rows = _labels(new)
    row = next(row for row in rows if row["problem_grade"] == "0")
    initial = lifecycle._row_grades(row)
    row["address_grade"] = "1"
    final = lifecycle._row_grades(row)
    _rewrite(new, rows)
    with pytest.raises(ValueError, match="final grades"):
        lifecycle.freeze_labels(run, new, annotator="annotator")
    _review_correction(new, row["annotation_id"], initial, final)
    monkeypatch.setattr(lifecycle, "now", lambda: "2026-10-05T12:00:00+08:00")
    result = lifecycle.freeze_labels(run, new, annotator="annotator")
    assert result["effective_grades_sha256"] == before["effective_grades_sha256"]
    assert result["revision"]["changed_grade_pairs"] == [row["annotation_id"]]
    assert result["revision"]["policy_action"] == "keep_thresholds"


def test_global_index_failure_keeps_authoritative_audit_verifiable(
    lifecycle_bundle, tmp_path, monkeypatch,
):
    run, _, new, _ = _revision(lifecycle_bundle, tmp_path)
    real_open = lifecycle.os.open

    def failed_index(path, flags, *args, **kwargs):
        if str(path).endswith("label-revisions.jsonl"):
            raise OSError("Private OS detail that must never be printed")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(lifecycle.os, "open", failed_index)
    with pytest.warns(RuntimeWarning, match="optional global revision index"):
        record = lifecycle.freeze_labels(run, new, annotator="annotator")
    assert lifecycle.verify_label_freeze(run, new) == record
    assert (new / "revision-record.json").is_file()
    audit = json.loads((new / "revision-record.json").read_text())
    audit["reason"] = "tampered"
    (new / "revision-record.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="audit mismatch"):
        lifecycle.verify_label_freeze(run, new)


def test_revision_bundle_is_portable_without_original_absolute_paths(lifecycle_bundle, tmp_path):
    run, original, new, _ = _revision(lifecycle_bundle, tmp_path)
    lifecycle.freeze_labels(run, new, annotator="annotator")
    relocated = tmp_path / "offline-copy"
    relocated.mkdir()
    for path in (run, original, new):
        shutil.copytree(path, relocated / path.name)
    for path in (run, original, new):
        shutil.rmtree(path)
    result = lifecycle.verify_label_freeze(relocated / run.name, relocated / new.name)
    assert result["revision"]["policy_action"] == "keep_thresholds"
