from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CategoryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
]
ProblemTypeText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=40),
]
DetailText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
LocationTypeText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32),
]
KeywordText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20),
]


class LLMExtractedProblem(BaseModel):
    """Fields that may be extracted from untrusted ticket text by the LLM."""

    model_config = ConfigDict(extra="forbid")

    problem_type: ProblemTypeText
    symptom: list[DetailText] = Field(max_length=3)
    impact: list[DetailText] = Field(max_length=3)
    location_type: LocationTypeText
    keywords: list[KeywordText] = Field(max_length=6)


class Problem(LLMExtractedProblem):
    """Validated problem enriched with the source system's category path."""

    category: list[CategoryText] = Field(max_length=5)
