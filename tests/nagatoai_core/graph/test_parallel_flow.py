# Standard Library
from typing import List

# Third Party
import pytest
import concurrent.futures
from unittest.mock import patch, call

# Nagato AI
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.parallel_flow import ParallelFlow
from nagatoai_core.graph.types import NodeResult


class AddingNode(AbstractNode):
    """A node that adds a specific value to its input"""

    add_value: int = 1

    def execute_one(self, input_item: NodeResult) -> NodeResult:
        """Process a single input by adding the add_value"""
        if not isinstance(input_item.result, (int, float)):
            return NodeResult(node_id=self.id, result=None, error=ValueError("Input is not a number"))

        return NodeResult(node_id=self.id, result=input_item.result + self.add_value, error=None)

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """Process all inputs"""
        results = []
        for inp in inputs:
            results.append(self.execute_one(inp))
        return results


class MultiplyNode(AbstractNode):
    """A node that multiplies its input by a specific value"""

    multiply_value: int = 2

    def execute_one(self, input_item: NodeResult) -> NodeResult:
        """Process a single input by multiplying by multiply_value"""
        if not isinstance(input_item.result, (int, float)):
            return NodeResult(node_id=self.id, result=None, error=ValueError("Input is not a number"))

        return NodeResult(node_id=self.id, result=input_item.result * self.multiply_value, error=None)

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """Process all inputs"""
        results = []
        for inp in inputs:
            results.append(self.execute_one(inp))
        return results


def test_parallel_flow_empty_nodes():
    """Test a parallel flow with no nodes"""
    flow = ParallelFlow(id="empty_flow")

    inputs = [NodeResult(node_id="input", result=5)]
    results = flow.execute(inputs)

    # With no nodes, the flow should return the inputs unchanged
    assert results == inputs


def test_parallel_flow_empty_inputs():
    """Test a parallel flow with empty inputs"""
    flow = ParallelFlow(
        id="flow_with_nodes",
        nodes=[
            AddingNode(id="adder", add_value=10),
            MultiplyNode(id="multiplier", multiply_value=3),
        ],
    )

    results = flow.execute([])

    # With no inputs, the flow should return an empty list
    assert results == []


def test_parallel_flow_with_multiple_nodes():
    """Test a parallel flow with multiple nodes processing inputs sequentially"""
    flow = ParallelFlow(
        id="test_flow",
        nodes=[
            AddingNode(id="adder", add_value=2),
            MultiplyNode(id="multiplier", multiply_value=3),
        ],
    )

    # Start with input 5, add 2 to get 7, then multiply by 3 to get 21
    results = flow.execute([NodeResult(node_id="input", result=5)])

    assert len(results) == 1
    assert results[0].result == 21  # (5 + 2) * 3 = 21


def test_parallel_flow_with_multiple_inputs():
    """Test a parallel flow processing multiple inputs"""
    flow = ParallelFlow(
        id="multi_input_flow",
        nodes=[
            AddingNode(id="adder", add_value=1),
            MultiplyNode(id="multiplier", multiply_value=2),
        ],
    )

    inputs = [
        NodeResult(node_id="input1", result=2),
        NodeResult(node_id="input2", result=3),
        NodeResult(node_id="input3", result=4),
    ]

    results = flow.execute(inputs)

    # Each input should be processed independently
    assert len(results) == 3
    assert results[0].result == 6  # (2 + 1) * 2 = 6
    assert results[1].result == 8  # (3 + 1) * 2 = 8
    assert results[2].result == 10  # (4 + 1) * 2 = 10


def test_parallel_flow_error_handling():
    """Test error handling in parallel flow"""
    flow = ParallelFlow(
        id="error_flow",
        nodes=[
            AddingNode(id="adder", add_value=1),
        ],
    )

    # Non-numeric input should cause an error
    inputs = [
        NodeResult(node_id="input1", result="not a number"),
    ]

    results = flow.execute(inputs)

    assert len(results) == 1
    assert results[0].error is not None
    assert isinstance(results[0].error, ValueError)


def test_parallel_flow_parallelism():
    """Test that the flow actually executes in parallel"""

    with patch("concurrent.futures.ThreadPoolExecutor", wraps=concurrent.futures.ThreadPoolExecutor) as mock_executor:
        flow = ParallelFlow(
            id="parallel_test_flow",
            nodes=[
                AddingNode(id="adder1", add_value=1),
                AddingNode(id="adder2", add_value=2),
            ],
        )

        inputs = [
            NodeResult(node_id="input1", result=1),
            NodeResult(node_id="input2", result=2),
            NodeResult(node_id="input3", result=3),
        ]

        results = flow.execute(inputs)

        # Verify ThreadPoolExecutor was used
        assert mock_executor.called

        # Check results are correct
        assert len(results) == 3
        assert results[0].result == 4  # 1 + 1 + 2 = 4
        assert results[1].result == 5  # 2 + 1 + 2 = 5
        assert results[2].result == 6  # 3 + 1 + 2 = 6


def test_parallel_flow_with_custom_workers():
    """Test that the flow uses the specified number of worker threads"""

    with patch("concurrent.futures.ThreadPoolExecutor", wraps=concurrent.futures.ThreadPoolExecutor) as mock_executor:
        # Create flow with custom number of workers
        flow = ParallelFlow(
            id="custom_workers_flow",
            nodes=[
                AddingNode(id="adder", add_value=1),
            ],
            max_workers=8,  # Custom number of workers
        )

        inputs = [
            NodeResult(node_id="input1", result=1),
            NodeResult(node_id="input2", result=2),
        ]

        flow.execute(inputs)

        # Verify ThreadPoolExecutor was called with the custom max_workers
        mock_executor.assert_called_once_with(max_workers=8)


def test_parallel_flow_default_workers():
    """Test that the flow uses the default number of worker threads (4)"""

    with patch("concurrent.futures.ThreadPoolExecutor", wraps=concurrent.futures.ThreadPoolExecutor) as mock_executor:
        # Create flow with default workers
        flow = ParallelFlow(
            id="default_workers_flow",
            nodes=[
                AddingNode(id="adder", add_value=1),
            ],
            # No max_workers specified, should use default of 4
        )

        inputs = [
            NodeResult(node_id="input1", result=1),
            NodeResult(node_id="input2", result=2),
        ]

        flow.execute(inputs)

        # Verify ThreadPoolExecutor was called with the default max_workers=4
        mock_executor.assert_called_once_with(max_workers=4)


def test_parallel_flow_with_nested_flow():
    """Test a parallel flow containing another flow"""
    inner_flow = ParallelFlow(
        id="inner_flow",
        nodes=[
            AddingNode(id="inner_adder", add_value=3),
        ],
    )

    outer_flow = ParallelFlow(
        id="outer_flow",
        nodes=[
            AddingNode(id="outer_adder", add_value=2),
            inner_flow,
        ],
    )

    # Input 5 -> add 2 -> 7 -> add 3 -> 10
    results = outer_flow.execute([NodeResult(node_id="input", result=5)])

    assert len(results) == 1
    assert results[0].result == 10  # 5 + 2 + 3 = 10


def test_parallel_flow_preserves_input_order():
    """Test that the flow preserves the order of inputs in the results"""
    flow = ParallelFlow(
        id="order_test_flow",
        nodes=[
            AddingNode(id="adder", add_value=1),
        ],
    )

    inputs = [
        NodeResult(node_id="input1", result=100),
        NodeResult(node_id="input2", result=200),
        NodeResult(node_id="input3", result=300),
    ]

    results = flow.execute(inputs)

    # The order of results should match the order of inputs
    assert len(results) == 3
    assert results[0].result == 101
    assert results[1].result == 201
    assert results[2].result == 301
