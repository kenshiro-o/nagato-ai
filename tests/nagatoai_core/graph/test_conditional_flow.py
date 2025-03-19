# Standard Library
from typing import Any, List, Optional

# Third Party
import pytest

# Nagato AI
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.conditional_flow import ConditionalFlow
from nagatoai_core.graph.types import ComparisonType, NodeResult, PredicateJoinType


class TestNode(AbstractNode):
    """A simple test node that returns a predetermined value"""

    return_value: Any

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        return [NodeResult(node_id=self.id, result=self.return_value)]


class ModifierNode(AbstractNode):
    """A node that applies a transformation to its input"""

    transform: str  # "double", "square", or "add_prefix"

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        if not inputs:
            return [NodeResult(node_id=self.id, result=None, error=ValueError("No inputs provided"))]

        input_value = inputs[0].result

        if self.transform == "double":
            if isinstance(input_value, (int, float)):
                result = input_value * 2
            else:
                result = None
                error = ValueError(f"Cannot double non-numeric value: {input_value}")
                return [NodeResult(node_id=self.id, result=result, error=error)]
        elif self.transform == "square":
            if isinstance(input_value, (int, float)):
                result = input_value**2
            else:
                result = None
                error = ValueError(f"Cannot square non-numeric value: {input_value}")
                return [NodeResult(node_id=self.id, result=result, error=error)]
        elif self.transform == "add_prefix":
            if isinstance(input_value, str):
                result = f"prefix_{input_value}"
            else:
                result = None
                error = ValueError(f"Cannot add prefix to non-string value: {input_value}")
                return [NodeResult(node_id=self.id, result=result, error=error)]
        else:
            result = None
            error = ValueError(f"Unknown transform: {self.transform}")
            return [NodeResult(node_id=self.id, result=result, error=error)]

        return [NodeResult(node_id=self.id, result=result)]


def test_conditional_flow_equal_true_path():
    """Test conditional flow with EQUAL comparison that follows positive path"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        comparison_type=ComparisonType.EQUAL,
        comparison_value=10,
    )

    # Input equals comparison value, should take positive path
    result = flow.execute([NodeResult(node_id="input", result=10)])

    assert len(result) == 1
    assert result[0].node_id == "positive_node"
    assert result[0].result == "positive result"


def test_conditional_flow_equal_false_path():
    """Test conditional flow with EQUAL comparison that follows negative path"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        comparison_type=ComparisonType.EQUAL,
        comparison_value=10,
    )

    # Input doesn't equal comparison value, should take negative path
    result = flow.execute([NodeResult(node_id="input", result=5)])

    assert len(result) == 1
    assert result[0].node_id == "negative_node"
    assert result[0].result == "negative result"


def test_conditional_flow_greater_than():
    """Test conditional flow with GREATER_THAN comparison"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        comparison_type=ComparisonType.GREATER_THAN,
        comparison_value=10,
    )

    # Input greater than comparison value, should take positive path
    result = flow.execute([NodeResult(node_id="input", result=15)])
    assert len(result) == 1
    assert result[0].node_id == "positive_node"

    # Input less than comparison value, should take negative path
    result = flow.execute([NodeResult(node_id="input", result=5)])
    assert len(result) == 1
    assert result[0].node_id == "negative_node"


def test_conditional_flow_with_custom_comparison():
    """Test conditional flow with a custom comparison function"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    # Custom function to check if a string starts with a specific prefix
    def starts_with_prefix(value, prefix):
        if isinstance(value, str):
            return value.startswith(prefix)
        return False

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        comparison_value="test",
        custom_comparison_fn=starts_with_prefix,
    )

    # String starts with "test", should take positive path
    result = flow.execute([NodeResult(node_id="input", result="test_string")])
    assert len(result) == 1
    assert result[0].node_id == "positive_node"

    # String doesn't start with "test", should take negative path
    result = flow.execute([NodeResult(node_id="input", result="other_string")])
    assert len(result) == 1
    assert result[0].node_id == "negative_node"


def test_conditional_flow_with_attribute_access():
    """Test conditional flow accessing an attribute of the input"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        comparison_type=ComparisonType.EQUAL,
        comparison_value="expected",
        input_attribute="status",
    )

    # Input with status "expected", should take positive path
    result = flow.execute([NodeResult(node_id="input", result={"status": "expected"})])
    assert len(result) == 1
    assert result[0].node_id == "positive_node"

    # Input with status "other", should take negative path
    result = flow.execute([NodeResult(node_id="input", result={"status": "other"})])
    assert len(result) == 1
    assert result[0].node_id == "negative_node"


def test_conditional_flow_with_only_positive_path():
    """Test conditional flow with only a positive path defined"""
    positive_node = TestNode(id="positive_node", return_value="positive result")

    flow = ConditionalFlow(
        id="test_flow", positive_path=positive_node, comparison_type=ComparisonType.EQUAL, comparison_value=10
    )

    # Input equals comparison value, should take positive path
    result = flow.execute([NodeResult(node_id="input", result=10)])
    assert len(result) == 1
    assert result[0].node_id == "positive_node"

    # Input doesn't equal comparison value, should return original inputs
    original_input = NodeResult(node_id="input", result=5)
    result = flow.execute([original_input])
    assert len(result) == 1
    assert result[0].node_id == "input"
    assert result[0].result == 5


def test_conditional_flow_with_nested_flows():
    """Test conditional flow with nested flows"""
    # Define nodes for different transformations
    double_node = ModifierNode(id="double_node", transform="double")
    square_node = ModifierNode(id="square_node", transform="square")

    # Create a nested conditional flow structure
    inner_flow = ConditionalFlow(
        id="inner_flow",
        positive_path=square_node,
        negative_path=double_node,
        comparison_type=ComparisonType.GREATER_THAN,
        comparison_value=5,
    )

    outer_flow = ConditionalFlow(
        id="outer_flow",
        positive_path=inner_flow,
        negative_path=TestNode(id="default_node", return_value=0),
        comparison_type=ComparisonType.GREATER_THAN,
        comparison_value=0,
    )

    # Test with value > 0 and > 5: should square (value = 10 -> result = 100)
    result = outer_flow.execute([NodeResult(node_id="input", result=10)])
    assert len(result) == 1
    assert result[0].node_id == "square_node"
    assert result[0].result == 100

    # Test with value > 0 but < 5: should double (value = 3 -> result = 6)
    result = outer_flow.execute([NodeResult(node_id="input", result=3)])
    assert len(result) == 1
    assert result[0].node_id == "double_node"
    assert result[0].result == 6

    # Test with value <= 0: should return default value (0)
    result = outer_flow.execute([NodeResult(node_id="input", result=-2)])
    assert len(result) == 1
    assert result[0].node_id == "default_node"
    assert result[0].result == 0


def test_conditional_flow_error_handling():
    """Test conditional flow error handling"""
    flow = ConditionalFlow(
        id="test_flow",
        positive_path=TestNode(id="positive_node", return_value="positive result"),
        comparison_type=ComparisonType.EQUAL,
        comparison_value=10,
        input_attribute="non_existent",  # This will cause an error
    )

    result = flow.execute([NodeResult(node_id="input", result=10)])
    assert len(result) == 1
    assert result[0].node_id == "conditional_flow_error"
    assert result[0].error is not None
    assert "does not have attribute" in str(result[0].error)


def test_conditional_flow_with_broadcasting_and():
    """Test conditional flow with broadcasting comparison using AND join"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        broadcast_comparison=True,
        comparison_type=ComparisonType.GREATER_THAN,
        comparison_value=5,
        predicate_join_type=PredicateJoinType.AND,
    )

    # When all inputs pass the condition, should take positive path
    result = flow.execute(
        [
            NodeResult(node_id="input1", result=10),
            NodeResult(node_id="input2", result=7),
            NodeResult(node_id="input3", result=8),
        ]
    )
    assert len(result) == 1
    assert result[0].node_id == "positive_node"

    # When any input fails the condition, should take negative path
    result = flow.execute(
        [
            NodeResult(node_id="input1", result=10),
            NodeResult(node_id="input2", result=3),  # This fails
            NodeResult(node_id="input3", result=8),
        ]
    )
    assert len(result) == 1
    assert result[0].node_id == "negative_node"


def test_conditional_flow_with_broadcasting_or():
    """Test conditional flow with broadcasting comparison using OR join"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        broadcast_comparison=True,
        comparison_type=ComparisonType.GREATER_THAN,
        comparison_value=5,
        predicate_join_type=PredicateJoinType.OR,
    )

    # When any input passes the condition, should take positive path
    result = flow.execute(
        [
            NodeResult(node_id="input1", result=3),
            NodeResult(node_id="input2", result=4),
            NodeResult(node_id="input3", result=6),  # Only this passes
        ]
    )
    assert len(result) == 1
    assert result[0].node_id == "positive_node"

    # When all inputs fail the condition, should take negative path
    result = flow.execute(
        [
            NodeResult(node_id="input1", result=3),
            NodeResult(node_id="input2", result=4),
            NodeResult(node_id="input3", result=5),  # Equal to, not greater than
        ]
    )
    assert len(result) == 1
    assert result[0].node_id == "negative_node"


def test_conditional_flow_with_attribute_and_broadcasting():
    """Test conditional flow with attribute access and broadcasting"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        broadcast_comparison=True,
        comparison_type=ComparisonType.EQUAL,
        comparison_value="active",
        input_attribute="status",
        predicate_join_type=PredicateJoinType.AND,
    )

    # When all inputs have status=active, should take positive path
    result = flow.execute(
        [
            NodeResult(node_id="input1", result={"status": "active", "value": 1}),
            NodeResult(node_id="input2", result={"status": "active", "value": 2}),
            NodeResult(node_id="input3", result={"status": "active", "value": 3}),
        ]
    )
    assert len(result) == 1
    assert result[0].node_id == "positive_node"

    # When any input doesn't have status=active, should take negative path
    result = flow.execute(
        [
            NodeResult(node_id="input1", result={"status": "active", "value": 1}),
            NodeResult(node_id="input2", result={"status": "inactive", "value": 2}),  # This fails
            NodeResult(node_id="input3", result={"status": "active", "value": 3}),
        ]
    )
    assert len(result) == 1
    assert result[0].node_id == "negative_node"


def test_conditional_flow_with_custom_comparison_broadcasting():
    """Test conditional flow with custom comparison function and broadcasting"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    # Custom function to check if a number is even
    def is_even(value, _):
        if isinstance(value, (int, float)):
            return value % 2 == 0
        return False

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        broadcast_comparison=True,
        custom_comparison_fn=is_even,
        comparison_value=None,  # Not used by the custom function
        predicate_join_type=PredicateJoinType.AND,
    )

    # When all inputs are even, should take positive path
    result = flow.execute(
        [
            NodeResult(node_id="input1", result=2),
            NodeResult(node_id="input2", result=4),
            NodeResult(node_id="input3", result=6),
        ]
    )
    assert len(result) == 1
    assert result[0].node_id == "positive_node"

    # When any input is odd, should take negative path
    result = flow.execute(
        [
            NodeResult(node_id="input1", result=2),
            NodeResult(node_id="input2", result=3),  # This is odd
            NodeResult(node_id="input3", result=6),
        ]
    )
    assert len(result) == 1
    assert result[0].node_id == "negative_node"


def test_conditional_flow_empty_inputs_with_broadcasting():
    """Test conditional flow with empty inputs and broadcasting"""
    positive_node = TestNode(id="positive_node", return_value="positive result")
    negative_node = TestNode(id="negative_node", return_value="negative result")

    flow = ConditionalFlow(
        id="test_flow",
        positive_path=positive_node,
        negative_path=negative_node,
        broadcast_comparison=True,
        comparison_type=ComparisonType.EQUAL,
        comparison_value=10,
    )

    # When inputs is empty, the condition should evaluate to false
    result = flow.execute([])

    # Should get an error for empty inputs
    assert len(result) == 1
    assert result[0].node_id == "conditional_flow_error"
    assert "No inputs provided for comparison" in str(result[0].error)
