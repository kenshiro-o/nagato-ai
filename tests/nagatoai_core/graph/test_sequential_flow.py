# Standard Library
from typing import List

# Third Party
import pytest

# Nagato AI
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.sequential_flow import SequentialFlow
from nagatoai_core.graph.types import NodeResult


class MinimalisticNode(AbstractNode):
    """A simple node that increments its input by 1"""

    def execute_one(self, input: NodeResult) -> NodeResult:
        res_value = input.result
        if not isinstance(res_value, int):
            return NodeResult(node_id=self.id, result=None, error=ValueError("not a number"))

        return NodeResult(node_id=self.id, result=res_value + 1, error=None)

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        results: List[NodeResult] = []
        for inp in inputs:
            results.append(self.execute_one(inp))

        return results


def test_sequential_flow_with_three_nodes():
    """Test a sequential flow containing three incrementing nodes"""
    # Create a sequential flow with three nodes
    flow = SequentialFlow(
        nodes=[
            MinimalisticNode(id="node_1", parents=[], descendants=[]),
            MinimalisticNode(id="node_2", parents=[], descendants=[]),
            MinimalisticNode(id="node_3", parents=[], descendants=[]),
        ]
    )

    # Execute the flow starting with an initial input of 0
    results = flow.execute([NodeResult(node_id="some_id", result=0)])

    # After three increments, expect the value to be 3
    assert isinstance(results, list)
    assert len(results) == 1

    result_item = results[0]
    assert isinstance(result_item, NodeResult)
    assert result_item.node_id is not None
    assert result_item.result == 3


def test_sequential_flow_with_nested_two_nodes():
    """Test a nested sequential flow containing two incrementing nodes"""
    # Create an inner sequential flow with two nodes
    inner_flow = SequentialFlow(
        nodes=[
            MinimalisticNode(id="node_1", parents=[], descendants=[]),
            MinimalisticNode(id="node_2", parents=[], descendants=[]),
        ]
    )

    # Create a outer sequential flow containing the inner flow
    flow = SequentialFlow(nodes=[inner_flow])

    # Execute the flow starting with an initial input of 0
    results = flow.execute([NodeResult(node_id="some_id", result=0)])

    # After three increments, expect the value to be 3
    assert isinstance(results, list)
    assert len(results) == 1

    result_item = results[0]
    assert isinstance(result_item, NodeResult)
    assert result_item.node_id is not None
    assert result_item.result == 2
