from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CHUNK_SCHEMA_VERSION = "1.0.0"
CHUNK_TEMPLATE_VERSION = "problem-v1"

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
Sha256Text = Annotated[
    str,
    StringConstraints(pattern=r"^[a-f0-9]{64}$"),
]


class ChunkExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    model: NonEmptyText = Field(max_length=256)
    prompt_version: NonEmptyText = Field(max_length=128)


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    source_id: NonEmptyText = Field(max_length=256)
    city: str = Field(max_length=128)
    district: str = Field(max_length=128)
    category1: str = Field(max_length=256)
    category2: str = Field(max_length=256)
    category3: str = Field(max_length=256)
    create_time: str = Field(max_length=64)
    problem_type: NonEmptyText = Field(max_length=40)
    verification_status: Literal["candidate"]
    template_version: Literal[CHUNK_TEMPLATE_VERSION]


class ProblemChunk(BaseModel):
    """Runtime contract shared by extraction resume and vector indexing."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    schema_version: Literal[CHUNK_SCHEMA_VERSION]
    id: NonEmptyText = Field(max_length=256)
    type: Literal["problem"]
    text: NonEmptyText = Field(max_length=4096)
    source_text_hash: Sha256Text
    extraction_run_id: NonEmptyText = Field(max_length=128)
    extraction: ChunkExtraction
    metadata: ChunkMetadata

    @model_validator(mode="after")
    def source_ids_match(self) -> Self:
        if self.metadata.source_id != self.id:
            raise ValueError("metadata.source_id must match id")
        return self


class QuarantineRecord(BaseModel):
    """Non-sensitive failure state used to make resume behavior deterministic."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    schema_version: Literal[CHUNK_SCHEMA_VERSION]
    ticket_id: NonEmptyText = Field(max_length=256)
    source_text_hash: Sha256Text
    stage: Literal["llm_extraction"]
    error_code: NonEmptyText = Field(max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    error_type: NonEmptyText = Field(max_length=128)
    attempts: int = Field(ge=1)
    extraction_run_id: NonEmptyText = Field(max_length=128)
    extraction: ChunkExtraction
    status_code: int | None = Field(default=None, ge=100, le=599)
