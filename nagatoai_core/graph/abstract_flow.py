from __future__ import annotations

# Standard Library
from abc import ABC, abstractmethod
from typing import List, Union

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.types import NodeResult


class AbstractFlow(AbstractNode):
    """
    Abstract base class for flows.

    A flow represents a subgraph or composite structure that orchestrates the execution
    of its component nodes and/or sub-flows.
    """

    # Think about how to increase the depth of a given flow
    depth: int = 0

    class Config:
        arbitrary_types_allowed = True
