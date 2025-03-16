# Standard Library
from typing import List

# Third Party
import pytest

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.sequential_flow import SequentialFlow
from nagatoai_core.graph.transformer_flow import TransformerFlow, combine_inputs_with_flow
from nagatoai_core.graph.types import NodeResult


class SimpleTestNode(AbstractNode):
    """Simple test node for testing."""

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """Return a NodeResult with the node_id and a simple value."""
        return [NodeResult(node_id=self.id, result=f"Result from {self.id}")]


def test_transformer_flow_basic():
    """Test basic functionality of TransformerFlow."""
    # Create a simple flow to be injected
    flow_to_inject = SequentialFlow(
        id="flow_to_inject", nodes=[SimpleTestNode(id="injected_node_1"), SimpleTestNode(id="injected_node_2")]
    )

    # Create a simple functor that combines input results with flow results
    def simple_functor(inputs: List[NodeResult], flow: AbstractFlow) -> List[NodeResult]:
        # Execute the flow
        flow_results = flow.execute([])

        # Combine input results with flow results
        combined_results = inputs + flow_results

        # Create a new result based on the combination
        return combined_results

    # Create the transformer flow
    transformer_flow = TransformerFlow(id="transformer_flow", flow_param=flow_to_inject, functor=simple_functor)

    # Create input results
    input_results = [
        NodeResult(node_id="input_1", result="Input 1 value"),
        NodeResult(node_id="input_2", result="Input 2 value"),
    ]

    # Execute the transformer flow
    results = transformer_flow.execute(input_results)

    # Verify results
    assert len(results) == 3
    assert results[0].node_id == "input_1"
    assert results[1].node_id == "input_2"
    assert results[2].node_id == "injected_node_2"


def test_transformer_flow_with_complex_functor():
    """Test TransformerFlow with a more complex transformation functor."""
    # Create a flow to inject
    flow_to_inject = SequentialFlow(id="flow_to_inject", nodes=[SimpleTestNode(id="data_processor")])

    # Create a functor that does more complex transformation
    def complex_functor(inputs: List[NodeResult], flow: AbstractFlow) -> List[NodeResult]:
        # Extract values from inputs
        input_values = [input_result.result for input_result in inputs]
        combined_input = " + ".join(input_values)

        # Execute the flow with original inputs
        flow_results = flow.execute(inputs)

        # Create a result that demonstrates transformation
        return [
            NodeResult(
                node_id="transformed_result", result=f"Transformed: {combined_input} -> {flow_results[0].result}"
            )
        ]

    # Create the transformer flow
    transformer_flow = TransformerFlow(id="transformer_flow", flow_param=flow_to_inject, functor=complex_functor)

    # Create input results
    input_results = [
        NodeResult(node_id="data_1", result="First data"),
        NodeResult(node_id="data_2", result="Second data"),
    ]

    # Execute the transformer flow
    results = transformer_flow.execute(input_results)

    # Verify results
    assert len(results) == 1
    assert results[0].node_id == "transformed_result"
    assert "Transformed: First data + Second data -> Result from data_processor" in results[0].result


def test_transformer_flow_error_handling():
    """Test error handling in TransformerFlow."""
    # Create a flow that will be used
    flow_to_inject = SequentialFlow(id="flow_to_inject", nodes=[SimpleTestNode(id="error_node")])

    # Create a functor that might raise an exception
    def error_functor(inputs: List[NodeResult], flow: AbstractFlow) -> List[NodeResult]:
        if not inputs:
            raise ValueError("Inputs cannot be empty")

        flow_results = flow.execute(inputs)
        return [NodeResult(node_id="error_handler", result="Processed successfully")]

    # Create the transformer flow
    transformer_flow = TransformerFlow(id="transformer_flow", flow_param=flow_to_inject, functor=error_functor)

    # Test with empty inputs (should raise exception)
    results = transformer_flow.execute([])
    assert len(results) == 1
    assert results[0].node_id == "transformer_flow_error"
    assert results[0].error is not None

    # Test with valid inputs (should succeed)
    input_results = [NodeResult(node_id="valid_input", result="Valid data")]
    results = transformer_flow.execute(input_results)

    assert len(results) == 1
    assert results[0].node_id == "error_handler"
    assert results[0].result == "Processed successfully"


def test_combine_inputs_with_flow_utility():
    """Test the combine_inputs_with_flow utility function."""
    # Create a test flow
    test_flow = SequentialFlow(
        id="test_flow", nodes=[SimpleTestNode(id="flow_node_1"), SimpleTestNode(id="flow_node_2")]
    )

    # Create input results
    input_results = [
        NodeResult(node_id="input_1", result="Input 1 value"),
        NodeResult(node_id="input_2", result="Input 2 value"),
    ]

    # Use the utility function
    combined_results = combine_inputs_with_flow(input_results, test_flow)

    # Verify results
    assert len(combined_results) == 3  # 2 inputs + 1 flow result

    # Check that the original inputs are preserved at the beginning
    assert combined_results[0].node_id == "input_1"
    assert combined_results[1].node_id == "input_2"

    # Check that the flow results are appended
    assert combined_results[2].node_id == "flow_node_2"

    # Test with a TransformerFlow using the utility function
    transformer_flow = TransformerFlow(id="transformer_flow", flow_param=test_flow, functor=combine_inputs_with_flow)

    # Execute the transformer flow
    results = transformer_flow.execute(input_results)

    # Verify same results as direct call
    assert len(results) == 3
    assert [r.node_id for r in results] == ["input_1", "input_2", "flow_node_2"]
