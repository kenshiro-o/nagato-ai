# Standard Library
from typing import List

# Third Party
import pytest

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.types import NodeResult


class TestFlow(AbstractFlow):
    """A simple test flow that adds a prefix to input results"""

    id: str
    prefix: str = "prefix_"

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        results = []
        for input_node in inputs:
            # If input is a string, add prefix, otherwise keep as is
            if isinstance(input_node.result, str):
                result_value = f"{self.prefix}{input_node.result}"
            else:
                result_value = input_node.result

            results.append(NodeResult(node_id=f"{self.id}", result=result_value, step=input_node.step + 1))
        return results


class MultiplyFlow(AbstractFlow):
    """A test flow that multiplies numeric inputs by a factor"""

    id: str
    factor: int = 2

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        results = []
        for input_node in inputs:
            # If input is a number, multiply it, otherwise keep as is
            if isinstance(input_node.result, (int, float)):
                result_value = input_node.result * self.factor
            else:
                result_value = input_node.result

            results.append(NodeResult(node_id=f"{self.id}", result=result_value, step=input_node.step + 1))
        return results


class SplitFlow(AbstractFlow):
    """A test flow that splits strings and returns multiple results"""

    id: str

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        results = []
        for input_node in inputs:
            # If input is a string, split it and create multiple results
            if isinstance(input_node.result, str) and "," in input_node.result:
                parts = input_node.result.split(",")
                for i, part in enumerate(parts):
                    results.append(
                        NodeResult(node_id=f"{self.id}_part{i}", result=part.strip(), step=input_node.step + 1)
                    )
            else:
                # Simply pass through non-splittable inputs
                results.append(
                    NodeResult(node_id=f"{self.id}_passthrough", result=input_node.result, step=input_node.step + 1)
                )
        return results


def test_empty_graph():
    """Test that an empty graph returns the inputs unchanged"""
    graph = Graph(nodes=[])
    inputs = [NodeResult(node_id="input1", result="test")]

    results = graph.run(inputs)

    assert results == inputs
    assert len(graph.result_map) == 0


def test_single_node_graph():
    """Test a graph with a single node"""
    test_flow = TestFlow(id="test_flow")
    graph = Graph(nodes=[test_flow])

    inputs = [NodeResult(node_id="input1", result="hello")]
    results = graph.run(inputs)

    assert len(results) == 1
    assert results[0].node_id == "test_flow"
    assert results[0].result == "prefix_hello"
    assert results[0].step == 1

    # Check result_map
    assert "test_flow" in graph.result_map
    assert len(graph.result_map["test_flow"]) == 1
    assert graph.result_map["test_flow"][0].result == "prefix_hello"


def test_multi_node_sequential_graph():
    """Test a graph with multiple nodes executed sequentially"""
    test_flow1 = TestFlow(id="test_flow1", prefix="first_")
    test_flow2 = TestFlow(id="test_flow2", prefix="second_")

    graph = Graph(nodes=[test_flow1, test_flow2])

    inputs = [NodeResult(node_id="input1", result="hello")]
    results = graph.run(inputs)

    # Final result should have both prefixes applied
    assert len(results) == 1
    assert results[0].node_id == "test_flow2"
    assert results[0].result == "second_first_hello"
    assert results[0].step == 2

    # Check result_map contains both nodes' results
    assert "test_flow1" in graph.result_map
    assert "test_flow2" in graph.result_map
    assert graph.result_map["test_flow1"][0].result == "first_hello"
    assert graph.result_map["test_flow2"][0].result == "second_first_hello"


def test_mixed_node_types_graph():
    """Test a graph with different types of nodes"""
    multiply_flow = MultiplyFlow(id="multiply_flow", factor=3)
    test_flow = TestFlow(id="test_flow", prefix="prefixed_")

    graph = Graph(nodes=[multiply_flow, test_flow])

    # Send mixed input types (number and string)
    inputs = [NodeResult(node_id="input1", result=5), NodeResult(node_id="input2", result="hello")]

    results = graph.run(inputs)

    # Results should have 2 elements, one for each input
    assert len(results) == 2

    # For numeric input: multiply then prefix (which keeps it as is)
    numeric_result = next(r for r in results if isinstance(r.result, (int, float)))
    assert numeric_result.result == 15  # 5 * 3

    # For string input: keep as is then prefix
    string_result = next(r for r in results if isinstance(r.result, str))
    assert string_result.result == "prefixed_hello"


def test_graph_with_expanding_node():
    """Test a graph where one node expands the number of results"""
    # First node splits comma-separated strings into multiple results
    split_flow = SplitFlow(id="split_flow")
    # Second node adds a prefix to each part
    test_flow = TestFlow(id="test_flow", prefix="item_")

    graph = Graph(nodes=[split_flow, test_flow])

    inputs = [NodeResult(node_id="input1", result="apple, banana, cherry")]
    results = graph.run(inputs)

    # Should have 3 results (one for each split part)
    assert len(results) == 3

    # Check that all parts were prefixed
    result_values = [r.result for r in results]
    assert "item_apple" in result_values
    assert "item_banana" in result_values
    assert "item_cherry" in result_values

    # Check result_map contains all intermediate results
    assert "split_flow_part0" in graph.result_map
    assert "split_flow_part1" in graph.result_map
    assert "split_flow_part2" in graph.result_map
    assert "test_flow" in graph.result_map
    assert len(graph.result_map["test_flow"]) == 3


def test_graph_with_multiple_inputs():
    """Test a graph that processes multiple inputs simultaneously"""
    test_flow = TestFlow(id="test_flow", prefix="processed_")
    graph = Graph(nodes=[test_flow])

    inputs = [
        NodeResult(node_id="input1", result="first"),
        NodeResult(node_id="input2", result="second"),
        NodeResult(node_id="input3", result="third"),
    ]

    results = graph.run(inputs)

    # Should have 3 results (one for each input)
    assert len(results) == 3

    # Check all inputs were processed
    result_values = [r.result for r in results]
    assert "processed_first" in result_values
    assert "processed_second" in result_values
    assert "processed_third" in result_values


def test_graph_current_node_tracking():
    """Test that the graph correctly tracks the current_node during execution"""
    test_flow1 = TestFlow(id="test_flow1")
    test_flow2 = TestFlow(id="test_flow2")
    test_flow3 = TestFlow(id="test_flow3")

    graph = Graph(nodes=[test_flow1, test_flow2, test_flow3])

    # Initially current_node should be None
    assert graph.current_node is None

    inputs = [NodeResult(node_id="input1", result="test")]
    graph.run(inputs)

    # After execution, current_node should be the last node
    assert graph.current_node == test_flow3


def test_graph_with_error_handling():
    """Test how the graph handles errors from nodes"""

    class ErrorFlow(AbstractFlow):
        id: str
        should_error: bool = True

        def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
            if self.should_error:
                return [
                    NodeResult(
                        node_id=f"{self.id}_error",
                        result=None,
                        error=ValueError("Intentional test error"),
                        step=inputs[0].step + 1 if inputs else 0,
                    )
                ]
            return inputs

    error_flow = ErrorFlow(id="error_flow")
    test_flow = TestFlow(id="test_flow")

    # Graph with error node followed by a regular node
    graph = Graph(nodes=[error_flow, test_flow])

    inputs = [NodeResult(node_id="input1", result="test")]
    results = graph.run(inputs)

    # The error from the first node should be passed through
    assert len(results) == 1
    assert results[0].node_id == "test_flow"
    assert results[0].error is None  # The test_flow should have processed the NodeResult even with an error

    # Result map should contain the error result
    assert "error_flow_error" in graph.result_map
    assert graph.result_map["error_flow_error"][0].error is not None
