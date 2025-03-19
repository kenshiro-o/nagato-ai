from __future__ import annotations

# Standard Library
import logging
from abc import ABC, abstractmethod
from typing import List

# Third Party
import structlog
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.graph.types import NodeResult

# Configure basic logging
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,  # Set default log level
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class AbstractNode(BaseModel, ABC):
    id: str = Field(..., description="The unique identifier for the node.")
    name: str = Field("", description="The name of the node. This name does not have to be unique across nodes")

    class Config:
        # Allow arbitrary types (self-referencing types in our case)
        arbitrary_types_allowed = True

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """
        Get a logger for the node.
        """
        node_logger = structlog.get_logger(f"{self.__class__.__name__}.{self.id}")
        node_logger.bind(node_id=self.id, node_name=self.name)

        return node_logger

    def __hash__(self) -> int:
        """Make nodes hashable by their ID to enable use in sets and as dictionary keys."""
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """Equality comparison based on node ID."""
        if isinstance(other, AbstractNode):
            # ID and instance should be the same
            return self.id == other.id and self == other
        return False

    @abstractmethod
    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute the node's logic.
        This method should be implemented by concrete subclasses.
        """
        raise NotImplementedError("Subclasses must implement the execute method")
