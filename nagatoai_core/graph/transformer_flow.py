from __future__ import annotations

# Standard Library
import traceback
from typing import Callable, List

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.types import NodeResult


def combine_inputs_with_flow(inputs: List[NodeResult], flow: AbstractFlow) -> List[NodeResult]:
    """
    Executes the flow with the given inputs and combines the original inputs with the flow results.

    This is a common functor pattern that sequences the execution of a flow and combines
    its results with the original inputs. In functional programming, this pattern is similar
    to a "sequence" operation where effects are combined in order.

    Args:
        inputs: A list of NodeResult objects to be passed to the flow
        flow: The flow to execute with the inputs

    Returns:
        A concatenated list of the original inputs followed by the flow execution results
    """
    flow_results = flow.execute(inputs)
    return inputs + flow_results


class TransformerFlow(AbstractFlow):
    """
    TransformerFlow combines node results with another flow through a transformation function.

    This flow allows for the injection of another flow midway through graph execution,
    applying a custom transformation function to combine the results of previous nodes
    with the injected flow. This pattern is similar to the applicative functor concept
    in functional programming.

    Attributes:
        flow_param: The flow to inject into the execution pipeline
        functor: The transformation function that combines inputs with the flow_param
    """

    flow_param: AbstractFlow
    functor: Callable[[List[NodeResult], AbstractFlow], List[NodeResult]]

    class Config:
        arbitrary_types_allowed = True

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute the flow by applying the functor to the inputs and flow_param.

        Args:
            inputs: A list of NodeResult objects from previous executions

        Returns:
            A list of NodeResult objects after transformation
        """
        try:
            return self.functor(inputs, self.flow_param)
        except Exception as e:
            self.logger.error(f"Error in TransformerFlow", error=e)
            self.logger.error(traceback.format_exc())
            return [NodeResult(node_id="transformer_flow_error", result=None, error=e)]
