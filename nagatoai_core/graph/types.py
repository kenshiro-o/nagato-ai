# Standard Library
from enum import Enum, auto
from typing import Any, Optional

# Third Party
from pydantic import BaseModel


class ComparisonType(str, Enum):
    """
    Enum representing different types of comparisons that can be performed.
    """

    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"


class PredicateJoinType(str, Enum):
    """
    Enum representing logical operations to join multiple predicates.
    """

    AND = "and"  # All conditions must be true
    OR = "or"  # At least one condition must be true


class NodeResult(BaseModel):
    """
    NodeResult is a Pydantic model that represents the result of a node execution.
    """

    class Config:
        arbitrary_types_allowed = True

    node_id: str
    result: Any
    error: Optional[Exception] = None
    step: int = 0
