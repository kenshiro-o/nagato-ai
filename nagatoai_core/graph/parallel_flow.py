# Standard Library
import concurrent.futures
from typing import List

# Third Party
from pydantic import Field

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.types import NodeResult


class ParallelFlow(AbstractFlow):
    """
    ParallelFlow executes all component nodes in parallel on each input.

    Each input is processed by all nodes concurrently, resulting in
    a list of results with the same length as the input list.
    """

    nodes: List[AbstractNode] = Field(default_factory=list, description="The nodes that process inputs in parallel.")
    max_workers: int = Field(default=4, description="Maximum number of worker threads for parallel execution.")

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute all nodes on the inputs in parallel.

        Parameters
        ----------
        inputs : List[NodeResult]
            Input data to be processed by all nodes

        Returns
        -------
        List[NodeResult]
            Results from applying all nodes to the inputs in parallel
        """
        if not self.nodes or not inputs:
            return inputs

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each input to be processed by all nodes
            futures = {executor.submit(self._process_input, input_item): i for i, input_item in enumerate(inputs)}

            # Maintain the original input order
            ordered_results = [None] * len(inputs)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as e:
                    # If processing fails, create a NodeResult with the error
                    ordered_results[idx] = NodeResult(node_id=self.id, result=None, error=e)

            results = ordered_results

        return results

    def _process_input(self, input_item: NodeResult) -> NodeResult:
        """
        Process a single input with all nodes.

        Parameters
        ----------
        input_item : NodeResult
            The input to process

        Returns
        -------
        NodeResult
            The result after processing with all nodes
        """
        # Create a single NodeResult from the results of all nodes
        current_result = input_item

        for node in self.nodes:
            # Process the input with each node
            node_results = node.execute([current_result])
            if node_results:  # Update the current result if we got a valid result
                current_result = node_results[0]

        return current_result
