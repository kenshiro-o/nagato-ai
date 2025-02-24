from pydantic import BaseModel
from typing import Any, Optional


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
