from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _records(path: Path):
    if not path.exists():
        return
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
            yield value


def audit(output: Path, errors: Path) -> dict:
    summary = Counter()
    polarity = Counter()
    error_codes = Counter()
    for record in _records(output):
        summary["output_records"] += 1
        result = record.get("result", {})
        events = result.get("events", []) if isinstance(result, dict) else []
        if not isinstance(events, list):
            raise ValueError("result.events must be an array")
        summary["events"] += len(events)
        summary["zero_event_records"] += int(not events)
        summary["multi_event_records"] += int(len(events) > 1)
        processing = record.get("processing", {})
        if isinstance(processing, dict):
            summary["cache_hits"] += int(bool(processing.get("cache_hit")))
            summary["model_calls"] += int(processing.get("model_calls", 0))
            summary["alignment_repairs"] += int(processing.get("alignment_repairs", 0))
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
            polarity[str(event.get("polarity", "missing"))] += 1
            summary["evidence_spans"] += 1
            for field in (
                "actors",
                "objects",
                "behaviors",
                "impacts",
                "requests",
                "locations",
                "time_expressions",
            ):
                values = event.get(field, [])
                if isinstance(values, list):
                    summary["evidence_spans"] += len(values)

    for record in _records(errors):
        summary["quarantine_records"] += 1
        error_codes[str(record.get("error_code", "missing"))] += 1

    return {
        "counts": dict(sorted(summary.items())),
        "event_polarity": dict(sorted(polarity.items())),
        "quarantine_by_code": dict(sorted(error_codes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a semantic extraction run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.errors), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
