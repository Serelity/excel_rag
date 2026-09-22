from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .loader import SourceRecord, load_records
from .pipeline import content_sha256
from .schema import Polarity

ANNOTATION_VERSION = "semantic-gold-v1"
ADJUDICATION_VERSION = "semantic-adjudication-v1"

CaseStatus = Literal[
    "active",
    "resolved",
    "withdrawal",
    "consultation",
    "follow_up",
    "unclear",
]
IssueScope = Literal["active", "background"]
IssueId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    ),
]
LabelText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
NoteText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]
EvidenceText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]


class GoldIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    issue_id: IssueId
    label: LabelText
    expected_polarity: Polarity
    is_current_request: bool = Field(
        description="该问题是否直接表达当前/最终诉求，而不只是仍待处理的事实。"
    )
    required_knowledge_need: LabelText = Field(
        description="该问题需要检索哪一类政策、规范或处置知识。"
    )
    evidence_quotes: list[EvidenceText] = Field(
        min_length=1,
        max_length=8,
        description="判断该问题不可缺少的连续原文片段。",
    )


class GoldPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_status: CaseStatus
    active_retrieval_issues: list[GoldIssue] = Field(max_length=12)
    background_issues: list[GoldIssue] = Field(max_length=12)
    annotation_notes: NoteText = ""

    @model_validator(mode="after")
    def issue_ids_are_unique(self) -> GoldPayload:
        issue_ids = [
            issue.issue_id
            for issue in (*self.active_retrieval_issues, *self.background_issues)
        ]
        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("issue_id must be unique within a record")
        return self


class GoldRecord(GoldPayload):
    annotation_version: Literal[ANNOTATION_VERSION] = ANNOTATION_VERSION
    source_id: str = Field(min_length=1)
    source_row: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotator_id: str = Field(min_length=1, max_length=100)


class EventMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_index: int = Field(ge=0)
    gold_scope: IssueScope
    gold_issue_id: IssueId
    match_notes: NoteText = ""


class AdjudicationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    review_status: Literal["pending", "complete"] = "pending"
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matches: list[EventMatch]
    spurious_event_indices: list[int]
    adjudication_notes: NoteText = ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            records.append(value)
    return records


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            os.chmod(temporary, 0o600)
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _require_new_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")


def _index_unique(records: Iterable[dict[str, Any]], *, path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"missing source_id at {path}:{position}")
        if source_id in indexed:
            raise ValueError(f"duplicate source_id at {path}:{position}: {source_id}")
        indexed[source_id] = record
    return indexed


def _prediction_events(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = record.get("result")
    if not isinstance(result, dict):
        raise ValueError("prediction result must be an object")
    events = result.get("active_retrieval_issues", result.get("events"))
    if not isinstance(events, list):
        raise ValueError("prediction must contain result.events or result.active_retrieval_issues")
    if not all(isinstance(event, dict) for event in events):
        raise ValueError("every predicted event must be an object")
    return events


def _span_text(value: object) -> str | None:
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) and text else None
    return None


def _event_evidence(event: Mapping[str, Any]) -> list[str]:
    evidence: list[str] = []
    trigger = _span_text(event.get("trigger"))
    if trigger:
        evidence.append(trigger)
    for field in (
        "actors",
        "objects",
        "behaviors",
        "impacts",
        "requests",
        "time_expressions",
    ):
        values = event.get(field, [])
        if isinstance(values, list):
            evidence.extend(text for value in values if (text := _span_text(value)))
    locations = event.get("locations", [])
    if isinstance(locations, list):
        for location in locations:
            if isinstance(location, dict):
                text = _span_text(location.get("evidence"))
                if text:
                    evidence.append(text)
    return list(dict.fromkeys(evidence))


def _compact_event(index: int, event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_index": index,
        "normalized_event_type": event.get("normalized_event_type"),
        "polarity": event.get("polarity"),
        "trigger": _span_text(event.get("trigger")),
        "behaviors": [
            text for value in event.get("behaviors", []) if (text := _span_text(value))
        ],
        "impacts": [
            text for value in event.get("impacts", []) if (text := _span_text(value))
        ],
        "requests": [
            text for value in event.get("requests", []) if (text := _span_text(value))
        ],
        "search_terms": event.get("search_terms", []),
    }


def _prediction_hash(record: Mapping[str, Any]) -> str:
    result = record.get("result")
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prediction_context(record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    events = _prediction_events(record)
    return {
        "schema_version": record.get("schema_version"),
        "prompt_version": provenance.get("prompt_version"),
        "events": [_compact_event(index, event) for index, event in enumerate(events)],
    }


def prepare_gold_worksheet(
    *,
    input_path: Path,
    output_path: Path,
    predictions_path: Path | None,
    limit: int,
    overwrite: bool,
) -> int:
    if limit < 1:
        raise ValueError("limit must be positive")
    _require_new_output(output_path, overwrite=overwrite)
    predictions: dict[str, dict[str, Any]] = {}
    if predictions_path is not None:
        predictions = _index_unique(_read_jsonl(predictions_path), path=predictions_path)

    worksheet: list[dict[str, Any]] = []
    for record in load_records(input_path):
        if len(worksheet) >= limit:
            break
        prediction = predictions.get(record.source_id)
        if predictions_path is not None and prediction is None:
            raise ValueError(f"prediction is missing source_id={record.source_id}")
        if prediction is not None:
            predicted_hash = prediction.get("content_sha256")
            if predicted_hash != content_sha256(record.case_content):
                raise ValueError(
                    f"prediction content hash differs for source_id={record.source_id}"
                )
        worksheet.append(
            {
                "annotation_version": ANNOTATION_VERSION,
                "source_id": record.source_id,
                "source_row": record.source_row,
                "content_sha256": content_sha256(record.case_content),
                "case_content": record.case_content,
                "prediction": _prediction_context(prediction) if prediction is not None else None,
                "review_status": "pending",
                "annotator_id": "",
                "gold": {
                    "case_status": "unclear",
                    "active_retrieval_issues": [],
                    "background_issues": [],
                    "annotation_notes": "",
                },
            }
        )
    if not worksheet:
        raise ValueError("input contains no records")
    _atomic_write_jsonl(output_path, worksheet)
    return len(worksheet)


def _source_index(path: Path) -> dict[str, SourceRecord]:
    indexed: dict[str, SourceRecord] = {}
    for record in load_records(path):
        if record.source_id in indexed:
            raise ValueError(f"input contains duplicate source_id: {record.source_id}")
        indexed[record.source_id] = record
    return indexed


def _validate_gold_evidence(gold: GoldPayload, source: SourceRecord) -> None:
    for issue in (*gold.active_retrieval_issues, *gold.background_issues):
        for quote in issue.evidence_quotes:
            if quote not in source.case_content:
                raise ValueError(
                    f"gold evidence is not an exact source quote for source_id={source.source_id}, "
                    f"issue_id={issue.issue_id}: {quote!r}"
                )


def finalize_gold(
    *,
    input_path: Path,
    worksheet_path: Path,
    output_path: Path,
    overwrite: bool,
) -> int:
    _require_new_output(output_path, overwrite=overwrite)
    sources = _source_index(input_path)
    rows = _read_jsonl(worksheet_path)
    final: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, row in enumerate(rows, start=1):
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or source_id not in sources:
            raise ValueError(f"unknown source_id at {worksheet_path}:{position}")
        if source_id in seen:
            raise ValueError(f"duplicate source_id at {worksheet_path}:{position}")
        seen.add(source_id)
        source = sources[source_id]
        if row.get("review_status") != "complete":
            raise ValueError(f"annotation is not complete for source_id={source_id}")
        annotator_id = row.get("annotator_id")
        if not isinstance(annotator_id, str) or not annotator_id.strip():
            raise ValueError(f"annotator_id is required for source_id={source_id}")
        if row.get("source_row") != source.source_row:
            raise ValueError(f"source_row differs for source_id={source_id}")
        source_hash = content_sha256(source.case_content)
        if row.get("content_sha256") != source_hash:
            raise ValueError(f"source content changed for source_id={source_id}")
        gold = GoldPayload.model_validate(row.get("gold"))
        _validate_gold_evidence(gold, source)
        record = GoldRecord(
            source_id=source_id,
            source_row=source.source_row,
            content_sha256=source_hash,
            annotator_id=annotator_id.strip(),
            **gold.model_dump(),
        )
        final.append(record.model_dump(mode="json"))
    if not final:
        raise ValueError("worksheet contains no records")
    _atomic_write_jsonl(output_path, final)
    return len(final)


def _load_gold(path: Path) -> dict[str, GoldRecord]:
    indexed: dict[str, GoldRecord] = {}
    for position, value in enumerate(_read_jsonl(path), start=1):
        record = GoldRecord.model_validate(value)
        if record.source_id in indexed:
            raise ValueError(f"duplicate source_id at {path}:{position}")
        indexed[record.source_id] = record
    if not indexed:
        raise ValueError("gold file contains no records")
    return indexed


def prepare_adjudication(
    *,
    gold_path: Path,
    predictions_path: Path,
    output_path: Path,
    overwrite: bool,
) -> int:
    _require_new_output(output_path, overwrite=overwrite)
    gold = _load_gold(gold_path)
    predictions = _index_unique(_read_jsonl(predictions_path), path=predictions_path)
    missing = set(gold) - set(predictions)
    if missing:
        raise ValueError(f"predictions are missing {len(missing)} gold records")

    rows: list[dict[str, Any]] = []
    for source_id, gold_record in gold.items():
        prediction = predictions[source_id]
        if prediction.get("source_row") != gold_record.source_row:
            raise ValueError(f"prediction source_row differs for source_id={source_id}")
        if prediction.get("content_sha256") != gold_record.content_sha256:
            raise ValueError(f"prediction content hash differs for source_id={source_id}")
        rows.append(
            {
                "adjudication_version": ADJUDICATION_VERSION,
                "source_id": source_id,
                "source_row": gold_record.source_row,
                "gold": {
                    "case_status": gold_record.case_status,
                    "active_retrieval_issues": [
                        issue.model_dump(mode="json")
                        for issue in gold_record.active_retrieval_issues
                    ],
                    "background_issues": [
                        issue.model_dump(mode="json") for issue in gold_record.background_issues
                    ],
                },
                "prediction": _prediction_context(prediction),
                "adjudication": {
                    "review_status": "pending",
                    "prediction_sha256": _prediction_hash(prediction),
                    "matches": [],
                    "spurious_event_indices": [],
                    "adjudication_notes": "",
                },
            }
        )
    _atomic_write_jsonl(output_path, rows)
    return len(rows)


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _issue_lookup(gold: GoldRecord) -> dict[tuple[IssueScope, str], GoldIssue]:
    return {
        **{("active", issue.issue_id): issue for issue in gold.active_retrieval_issues},
        **{("background", issue.issue_id): issue for issue in gold.background_issues},
    }


def _validate_adjudication(
    *,
    row: Mapping[str, Any],
    gold: GoldRecord,
    prediction: Mapping[str, Any],
) -> tuple[AdjudicationPayload, list[dict[str, Any]]]:
    if row.get("adjudication_version") != ADJUDICATION_VERSION:
        raise ValueError(f"wrong adjudication_version for source_id={gold.source_id}")
    if row.get("source_row") != gold.source_row:
        raise ValueError(f"adjudication source_row differs for source_id={gold.source_id}")
    adjudication = AdjudicationPayload.model_validate(row.get("adjudication"))
    if adjudication.review_status != "complete":
        raise ValueError(f"adjudication is not complete for source_id={gold.source_id}")
    if adjudication.prediction_sha256 != _prediction_hash(prediction):
        raise ValueError(f"adjudication is stale for source_id={gold.source_id}")

    events = _prediction_events(prediction)
    event_indices = [match.event_index for match in adjudication.matches]
    event_indices.extend(adjudication.spurious_event_indices)
    if len(event_indices) != len(set(event_indices)):
        raise ValueError(
            f"predicted event is adjudicated more than once for source_id={gold.source_id}"
        )
    expected_indices = set(range(len(events)))
    if set(event_indices) != expected_indices:
        raise ValueError(
            f"every predicted event must be matched or spurious for source_id={gold.source_id}"
        )

    issue_lookup = _issue_lookup(gold)
    issue_keys = [(match.gold_scope, match.gold_issue_id) for match in adjudication.matches]
    if len(issue_keys) != len(set(issue_keys)):
        raise ValueError(f"gold issue is matched more than once for source_id={gold.source_id}")
    unknown = set(issue_keys) - set(issue_lookup)
    if unknown:
        raise ValueError(f"adjudication references unknown gold issues: {sorted(unknown)}")
    return adjudication, events


def score(
    *,
    gold_path: Path,
    predictions_path: Path,
    adjudication_path: Path,
) -> dict[str, Any]:
    gold = _load_gold(gold_path)
    predictions = _index_unique(_read_jsonl(predictions_path), path=predictions_path)
    adjudications = _index_unique(_read_jsonl(adjudication_path), path=adjudication_path)
    for name, values in (("predictions", predictions), ("adjudications", adjudications)):
        missing = set(gold) - set(values)
        extra = set(values) - set(gold)
        if missing or extra:
            raise ValueError(
                f"{name} source IDs differ from gold: missing={len(missing)} extra={len(extra)}"
            )

    totals = {
        "records": 0,
        "gold_active_issues": 0,
        "gold_current_requests": 0,
        "predicted_issues": 0,
        "matched_active_issues": 0,
        "matched_current_requests": 0,
        "background_leaks": 0,
        "spurious_issues": 0,
        "matched_active_polarities": 0,
        "correct_active_polarities": 0,
        "gold_active_evidence_quotes": 0,
        "covered_active_evidence_quotes": 0,
        "matched_active_evidence_quotes": 0,
        "covered_matched_active_evidence_quotes": 0,
        "exact_records": 0,
        "case_status_predictions": 0,
        "correct_case_status_predictions": 0,
    }
    per_record: list[dict[str, Any]] = []

    for source_id, gold_record in gold.items():
        prediction = predictions[source_id]
        if prediction.get("source_row") != gold_record.source_row:
            raise ValueError(f"prediction source_row differs for source_id={source_id}")
        if prediction.get("content_sha256") != gold_record.content_sha256:
            raise ValueError(f"prediction content hash differs for source_id={source_id}")
        adjudication, events = _validate_adjudication(
            row=adjudications[source_id],
            gold=gold_record,
            prediction=prediction,
        )
        issue_lookup = _issue_lookup(gold_record)
        active_matches = [m for m in adjudication.matches if m.gold_scope == "active"]
        background_matches = [m for m in adjudication.matches if m.gold_scope == "background"]
        current_ids = {
            issue.issue_id
            for issue in gold_record.active_retrieval_issues
            if issue.is_current_request
        }
        current_matches = [m for m in active_matches if m.gold_issue_id in current_ids]

        active_evidence_total = sum(
            len(issue.evidence_quotes) for issue in gold_record.active_retrieval_issues
        )
        matched_evidence_total = 0
        covered_evidence = 0
        polarity_correct = 0
        for match in active_matches:
            issue = issue_lookup[(match.gold_scope, match.gold_issue_id)]
            event = events[match.event_index]
            evidence = _event_evidence(event)
            matched_evidence_total += len(issue.evidence_quotes)
            covered_evidence += sum(
                any(quote in span for span in evidence) for quote in issue.evidence_quotes
            )
            polarity_correct += int(event.get("polarity") == issue.expected_polarity)

        exact = (
            len(active_matches) == len(gold_record.active_retrieval_issues)
            and not background_matches
            and not adjudication.spurious_event_indices
        )
        result = prediction.get("result")
        predicted_status = result.get("case_status") if isinstance(result, dict) else None
        status_present = isinstance(predicted_status, str)
        status_correct = status_present and predicted_status == gold_record.case_status

        totals["records"] += 1
        totals["gold_active_issues"] += len(gold_record.active_retrieval_issues)
        totals["gold_current_requests"] += len(current_ids)
        totals["predicted_issues"] += len(events)
        totals["matched_active_issues"] += len(active_matches)
        totals["matched_current_requests"] += len(current_matches)
        totals["background_leaks"] += len(background_matches)
        totals["spurious_issues"] += len(adjudication.spurious_event_indices)
        totals["matched_active_polarities"] += len(active_matches)
        totals["correct_active_polarities"] += polarity_correct
        totals["gold_active_evidence_quotes"] += active_evidence_total
        totals["covered_active_evidence_quotes"] += covered_evidence
        totals["matched_active_evidence_quotes"] += matched_evidence_total
        totals["covered_matched_active_evidence_quotes"] += covered_evidence
        totals["exact_records"] += int(exact)
        totals["case_status_predictions"] += int(status_present)
        totals["correct_case_status_predictions"] += int(status_correct)
        per_record.append(
            {
                "source_id": source_id,
                "source_row": gold_record.source_row,
                "gold_active_issues": len(gold_record.active_retrieval_issues),
                "predicted_issues": len(events),
                "matched_active_issues": len(active_matches),
                "matched_current_requests": len(current_matches),
                "background_leaks": len(background_matches),
                "spurious_issues": len(adjudication.spurious_event_indices),
                "exact": exact,
            }
        )

    precision = _divide(totals["matched_active_issues"], totals["predicted_issues"])
    recall = _divide(totals["matched_active_issues"], totals["gold_active_issues"])
    rates = {
        "issue_precision": precision,
        "issue_recall": recall,
        "issue_f1": _f1(precision, recall),
        "current_request_recall": _divide(
            totals["matched_current_requests"], totals["gold_current_requests"]
        ),
        "background_leakage_rate": _divide(
            totals["background_leaks"], totals["predicted_issues"]
        ),
        "spurious_issue_rate": _divide(
            totals["spurious_issues"], totals["predicted_issues"]
        ),
        "active_polarity_accuracy": _divide(
            totals["correct_active_polarities"], totals["matched_active_polarities"]
        ),
        "active_evidence_quote_recall": _divide(
            totals["covered_active_evidence_quotes"],
            totals["gold_active_evidence_quotes"],
        ),
        "matched_issue_evidence_quote_recall": _divide(
            totals["covered_matched_active_evidence_quotes"],
            totals["matched_active_evidence_quotes"],
        ),
        "exact_record_rate": _divide(totals["exact_records"], totals["records"]),
        "case_status_prediction_coverage": _divide(
            totals["case_status_predictions"], totals["records"]
        ),
        "case_status_accuracy": _divide(
            totals["correct_case_status_predictions"],
            totals["case_status_predictions"],
        ),
    }
    return {"counts": totals, "rates": rates, "records": per_record}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and score human evaluation for semantic extraction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-gold", help="create a private annotation worksheet")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--predictions", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=20)
    prepare.add_argument("--overwrite", action="store_true")

    finalize = subparsers.add_parser("finalize-gold", help="validate and freeze completed gold")
    finalize.add_argument("--input", type=Path, required=True)
    finalize.add_argument("--worksheet", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--overwrite", action="store_true")

    adjudicate = subparsers.add_parser(
        "prepare-adjudication", help="create a worksheet for one candidate run"
    )
    adjudicate.add_argument("--gold", type=Path, required=True)
    adjudicate.add_argument("--predictions", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    adjudicate.add_argument("--overwrite", action="store_true")

    scorer = subparsers.add_parser("score", help="score a completed candidate adjudication")
    scorer.add_argument("--gold", type=Path, required=True)
    scorer.add_argument("--predictions", type=Path, required=True)
    scorer.add_argument("--adjudication", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare-gold":
        count = prepare_gold_worksheet(
            input_path=args.input,
            output_path=args.output,
            predictions_path=args.predictions,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        print(f"worksheet_records={count} output={args.output}")
    elif args.command == "finalize-gold":
        count = finalize_gold(
            input_path=args.input,
            worksheet_path=args.worksheet,
            output_path=args.output,
            overwrite=args.overwrite,
        )
        print(f"gold_records={count} output={args.output}")
    elif args.command == "prepare-adjudication":
        count = prepare_adjudication(
            gold_path=args.gold,
            predictions_path=args.predictions,
            output_path=args.output,
            overwrite=args.overwrite,
        )
        print(f"adjudication_records={count} output={args.output}")
    else:
        result = score(
            gold_path=args.gold,
            predictions_path=args.predictions,
            adjudication_path=args.adjudication,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
