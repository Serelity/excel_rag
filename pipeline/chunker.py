from __future__ import annotations

import hashlib
import json
from typing import Any

from schemas.chunk import CHUNK_SCHEMA_VERSION, CHUNK_TEMPLATE_VERSION, ProblemChunk
from schemas.problem import Problem
from schemas.ticket import Ticket

SOURCE_HASH_VERSION = "ticket-content-goal-json-v1"


def _join(values: list[str], separator: str = "；") -> str:
    return separator.join(value for value in values if value)


def source_text_hash(ticket: Ticket) -> str:
    """Hash loader-normalized source fields before cleaning or model redaction."""

    source = json.dumps(
        {"content": ticket.content, "goal": ticket.goal},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def build_problem_chunk(
    ticket: Ticket,
    problem: Problem,
    *,
    extraction_run_id: str,
    model: str,
    prompt_version: str,
    source_hash: str | None = None,
) -> dict[str, Any]:
    """Build a candidate retrieval chunk without copying source text into metadata."""

    lines = [f"问题：{problem.problem_type}"]
    if problem.category:
        lines.append(f"分类：{_join(problem.category, ' > ')}")
    if problem.symptom:
        lines.append(f"典型表现：{_join(problem.symptom)}")
    if problem.impact:
        lines.append(f"影响：{_join(problem.impact)}")
    if problem.location_type:
        lines.append(f"地点类型：{problem.location_type}")
    if problem.keywords:
        lines.append(f"关键词：{_join(problem.keywords)}")

    chunk = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "id": ticket.ticket_id,
        "type": "problem",
        "text": "\n".join(lines),
        "source_text_hash": source_hash or source_text_hash(ticket),
        "extraction_run_id": extraction_run_id,
        "extraction": {
            "model": model,
            "prompt_version": prompt_version,
        },
        "metadata": {
            "source_id": ticket.ticket_id,
            "city": ticket.city,
            "district": ticket.district,
            "category1": ticket.category1,
            "category2": ticket.category2,
            "category3": ticket.category3,
            "create_time": ticket.create_time,
            "problem_type": problem.problem_type,
            "verification_status": "candidate",
            "template_version": CHUNK_TEMPLATE_VERSION,
        },
    }
    return ProblemChunk.model_validate(chunk).model_dump(mode="json")
