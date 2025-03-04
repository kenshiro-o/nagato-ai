# Standard Library
from typing import Dict, List, Optional, Union

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.types import NodeResult


class Graph(BaseModel):
    """
    Graph represents a directed acyclic graph (DAG) of flows that are executed in sequence.

    The Graph class manages the execution flow between different AbstractFlow implementations,
    passing the results of one flow as inputs to the next.
    """

    nodes: List[AbstractFlow]
    current_node: Optional[AbstractFlow] = None
    result_map: Dict[str, List[NodeResult]] = Field(default_factory=dict)
    compiled: bool = False

    def run(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute each node in the graph in sequence, passing the results of each node
        as inputs to the next node.

        Args:
            inputs: Initial input NodeResult objects for the first node

        Returns:
            List[NodeResult]: Results from the final node in the graph
        """
        if not self.nodes:
            return inputs

        current_inputs = inputs
        results_at_this_node = []

        for node in self.nodes:
            self.current_node = node
            results_at_this_node = node.execute(current_inputs)

            # Store the results in the result map for future reference
            for result in results_at_this_node:
                if result.node_id not in self.result_map:
                    self.result_map[result.node_id] = []
                self.result_map[result.node_id].append(result)

            # Use the results of this node as input to the next node
            current_inputs = results_at_this_node

        return results_at_this_node
