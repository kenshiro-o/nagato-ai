# Standard Library
from typing import Dict, List, Optional, Union

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.types import NodeResult


class Graph(BaseModel):
    nodes: List[AbstractFlow]
    current_node: Optional[AbstractFlow] = None
    result_map: Dict[str, List[NodeResult]] = {}

    def run(self, inputs: List[NodeResult]) -> List[NodeResult]:
        results_at_this_node: List[NodeResult] = None

        for node in self.nodes:
            self.current_node = node
            results_at_this_node = node.execute(inputs)
            for result_at_this_node in results_at_this_node:
                self.result_map[result_at_this_node.node_id] = result_at_this_node

            inputs = result_at_this_node

        return results_at_this_node
