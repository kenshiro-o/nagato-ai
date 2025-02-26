from __future__ import annotations

# Standard Library
from abc import ABC, abstractmethod
from typing import List, Union

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.types import NodeResult


class AbstractFlow(BaseModel, ABC):
    """
    Abstract base class for flows.

    A flow represents a subgraph or composite structure that orchestrates the execution
    of its component nodes and/or sub-flows.
    """

    nodes: List[Union[AbstractNode, AbstractFlow]] = Field(default_factory=list)

    # Think about how to increase the depth of a given flow
    depth: int = 0

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute the flow's logic.
        """
        raise NotImplementedError("Subclasses must implement the execute method")
