from __future__ import annotations

TABULAR_CONTAMINATION = "EMBEDDED_TABULAR_DATA"


def case_content_quality_issue(content: str) -> str | None:
    """Return a stable code when narrative text contains embedded TSV payloads."""
    if "\t" in content:
        return TABULAR_CONTAMINATION
    return None


def require_clean_case_content(content: str, *, location: str) -> None:
    issue = case_content_quality_issue(content)
    if issue is not None:
        raise ValueError(f"{issue} in case_content at {location}")
