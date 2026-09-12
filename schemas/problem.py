from pydantic import BaseModel
from typing import List


class Problem(BaseModel):
    """
    定义了LLM应该输出什么，规定输出的格式
    """

    problem_type: str

    category: List[str]

    symptom: List[str]

    impact: List[str]

    location_type: str

    keywords: List[str]