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

    problem_type: ProblemTypeText = Field(
        description=(
            "One concise, searchable primary problem. Preserve uncertainty for allegations; "
            "use a specific consultation label instead of unknown when the topic is clear."
        )
    )
    symptom: list[DetailText] = Field(
        max_length=3,
        description=(
            "Up to three already-observed facts copied as contiguous phrases from case_content; "
            "exclude requests, handling goals, replies, policies, and preventive measures."
        ),
    )
    impact: list[DetailText] = Field(
        max_length=3,
        description=(
            "Up to three consequences explicitly stated as contiguous phrases in case_content; "
            "never add forecasts or hazards absent from the source, and exclude allegations "
            "or requested outcomes."
        ),
    )
    location_type: LocationTypeText = Field(
        description=(
            "A generic venue type only, never a city, district, address, named road, complex, "
            "venue, company, or institution; use 行政区域 or 未知 when appropriate."
        )
    )
    keywords: list[KeywordText] = Field(
        max_length=6,
        description=(
            "Up to six generic concepts central to the problem; exclude proper-place and named "
            "organization/department/app terms, people, times, laws, and request-only actions."
        ),
    )


class Problem(LLMExtractedProblem):
    """Validated problem enriched with the source system's category path."""

    category: list[CategoryText] = Field(max_length=5)
