from __future__ import annotations

import json

from semantic_extraction.audit import audit


def test_audit_computes_exact_evidence_quote_rate(tmp_path) -> None:
    output = tmp_path / "output.jsonl"
    errors = tmp_path / "errors.jsonl"
    output.write_text(
        json.dumps(
            {
                "result": {"events": []},
                "processing": {
                    "proposed_evidence_quotes": 10,
                    "rejected_evidence_quotes": 2,
                    "trigger_fallbacks": 1,
                    "dropped_events": 1,
                    "polarity_repairs": 2,
                    "merged_duplicate_events": 3,
                    "truncation_retries": 1,
                    "truncation_recoveries": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    errors.write_text(
        json.dumps(
            {
                "error_code": "NO_GROUNDED_EVENTS",
                "processing": {
                    "model_calls": 2,
                    "proposed_evidence_quotes": 2,
                    "rejected_evidence_quotes": 2,
                    "trigger_fallbacks": 0,
                    "dropped_events": 1,
                    "truncation_retries": 1,
                    "truncation_recoveries": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = audit(output, errors)

    assert result["rates"]["exact_evidence_quote_rate"] == 8 / 12
    assert result["counts"]["trigger_fallbacks"] == 1
    assert result["counts"]["dropped_events"] == 2
    assert result["counts"]["polarity_repairs"] == 2
    assert result["counts"]["merged_duplicate_events"] == 3
    assert result["counts"]["model_calls"] == 2
    assert result["counts"]["truncation_retries"] == 2
    assert result["counts"]["truncation_recoveries"] == 1
