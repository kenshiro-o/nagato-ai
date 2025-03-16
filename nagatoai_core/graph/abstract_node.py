from __future__ import annotations

# Standard Library
from abc import ABC, abstractmethod
from typing import List

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.graph.types import NodeResult


class AbstractNode(BaseModel, ABC):
    id: str = Field(..., description="The unique identifier for the node.")
    name: str = Field("", description="The name of the node. This name does not have to be unique across nodes")

    # TODO - all nodes should have access to a logger
    # - should they all use the same logger?
    # - or should the logger be namedspace on the node name/id?!

    class Config:
        # Allow arbitrary types (self-referencing types in our case)
        arbitrary_types_allowed = True

    @abstractmethod
    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute the node's logic.
        This method should be implemented by concrete subclasses.
        """
        raise NotImplementedError("Subclasses must implement the execute method")
