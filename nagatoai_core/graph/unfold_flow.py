# Standard Library
from typing import List

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.types import NodeResult


class UnfoldFlow(AbstractFlow):
    """
    UnfoldFlow expands lists contained in the result field of each input NodeResult
    into individual NodeResult objects, then returns the flattened list.

    This is similar to the flatten operation in functional programming:
    - It expects each input to contain a list in its result field
    - For each element in that list, it creates a new NodeResult
    - All NodeResults are combined into a single output list

    For example, if you have:
    - 2 inputs, each with result=[1, 2, 3] and result=[4, 5]

    The output will be:
    - A flat list of 5 NodeResults, each containing one of the values: 1, 2, 3, 4, 5

    This allows for subsequent nodes in the graph to process each element individually.
    """

    # Whether to preserve original NodeResult metadata (node_id, step)
    preserve_metadata: bool = True

    # Option to skip empty lists instead of raising an error
    skip_empty_lists: bool = False

    # Option to include non-list inputs by wrapping them in a list
    wrap_non_list_inputs: bool = True

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Unfold each element of each input's result list into individual NodeResults.

        Args:
            inputs: List of NodeResult objects where each result field is expected to be a list

        Returns:
            List[NodeResult]: Flattened list of NodeResults, one per element in the input lists
        """
        try:
            all_results: List[NodeResult] = []

            for input_idx, input_node in enumerate(inputs):
                input_result = input_node.result

                # Handle non-list inputs if needed
                if not isinstance(input_result, list):
                    if self.wrap_non_list_inputs:
                        # Wrap in a list to process as a single element
                        input_result = [input_result]
                    else:
                        # Raise an error for non-list inputs
                        raise ValueError(f"Input at index {input_idx} has a non-list result: {type(input_result)}")

                # Handle empty lists
                if not input_result:
                    if self.skip_empty_lists:
                        continue
                    else:
                        raise ValueError(f"Input at index {input_idx} has an empty list result")

                # Process each element in the input result list
                for element_idx, element in enumerate(input_result):
                    # Create a new NodeResult for this element
                    element_result = NodeResult(
                        node_id=(
                            f"{input_node.node_id}_element{element_idx}"
                            if self.preserve_metadata
                            else f"element{element_idx}"
                        ),
                        result=element,
                        error=input_node.error,
                        step=input_node.step if self.preserve_metadata else 0,
                    )

                    # Add the element NodeResult to our collection
                    all_results.append(element_result)

            return all_results

        except Exception as e:
            # Create an error result
            error_result = NodeResult(
                node_id=f"unfold_flow_error", result=None, error=e, step=inputs[-1].step + 1 if inputs else 0
            )
            return [error_result]
