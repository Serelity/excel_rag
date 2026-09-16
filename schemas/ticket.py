from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

_NULL_MARKERS = frozenset({"null", "nan"})


class Ticket(BaseModel):
    """Normalized source ticket used by the extraction pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    ticket_id: str = Field(min_length=1)
    content: str
    goal: str
    category1: str
    category2: str
    category3: str
    city: str
    district: str
    create_time: str = ""

    @field_validator(
        "ticket_id",
        "content",
        "goal",
        "category1",
        "category2",
        "category3",
        "city",
        "district",
        "create_time",
        mode="before",
    )
    @classmethod
    def normalize_nullable_text(cls, value: Any) -> Any:
        if value is None:
            return ""

        if isinstance(value, str):
            text = value.strip()
            if text.casefold() in _NULL_MARKERS:
                return ""
            return text

        if isinstance(value, float) and value != value:
            return ""

        return value

    @computed_field(return_type=list[str])
    @property
    def category_path(self) -> list[str]:
        return [
            category for category in (self.category1, self.category2, self.category3) if category
        ]
