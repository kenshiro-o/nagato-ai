from typing import Optional, List, Type
import logging

from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.types import NodeResult


class SequentialFlow(AbstractFlow):
    """
    SequentialFlow executes its component nodes one after the other.
    """

    # TODO - What about retries at the flow level

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute each node in sequence, passing **kwargs to each node's execute method.

        For simplicity, this implementation does not automatically forward each node's output
        as input to the next node. Instead, all nodes receive the same keyword arguments.
        The result of the last node executed is returned.
        """
        last_results: List[NodeResult] = []
        for node in self.nodes:
            # The input becomes the output and so on
            inputs = node.execute(inputs)
            last_results = inputs

        # We are not interested in the intermediate results - only the last one counts
        return last_results
