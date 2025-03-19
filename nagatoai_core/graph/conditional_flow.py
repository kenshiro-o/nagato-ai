# Standard Library
from typing import Any, Callable, List, Optional, Union

# Third Party
from pydantic import root_validator

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.types import ComparisonType, NodeResult, PredicateJoinType


class ConditionalFlow(AbstractFlow):
    """
    ConditionalFlow executes one of two paths based on a condition evaluation.

    It evaluates input values against a condition and executes either the positive_path
    if the condition is met, or the negative_path if the condition is not met.

    The flow can operate in two modes:
    1. Single input mode: Evaluates one specific input against the condition
    2. Broadcasting mode: Evaluates all inputs against the condition, then joins the results
       using the specified predicate join type (AND/OR)
    """

    # Whether to evaluate all inputs (True) or just a single input (False)
    broadcast_comparison: bool = False

    # The index of the input to use for comparison when not broadcasting
    input_index: int = 0

    # The attribute of the input to use for comparison (if None, uses the entire result)
    input_attribute: Optional[str] = None

    # The comparison value to check against
    comparison_value: Any = None

    # Type of comparison to perform
    comparison_type: ComparisonType = ComparisonType.EQUAL

    # Custom comparison function (overrides comparison_type if provided)
    custom_comparison_fn: Optional[Callable[[Any, Any], bool]] = None

    # How to join multiple comparison results when broadcasting (AND/OR)
    predicate_join_type: PredicateJoinType = PredicateJoinType.AND

    # The node or flow to execute if the condition is true
    positive_path: Union[AbstractNode, AbstractFlow]

    # The node or flow to execute if the condition is false (optional)
    negative_path: Optional[Union[AbstractNode, AbstractFlow]] = None

    @root_validator(skip_on_failure=True)
    def validate_paths(cls, values):
        """Validate that at least one path (positive or negative) is provided."""
        if not values.get("positive_path") and not values.get("negative_path"):
            raise ValueError("At least one of positive_path or negative_path must be provided")
        return values

    def _extract_value_from_result(self, result: Any) -> Any:
        """Extract a value from a result object based on input_attribute."""
        # If an attribute is specified, extract it from the result
        if self.input_attribute:
            if hasattr(result, self.input_attribute):
                return getattr(result, self.input_attribute)
            elif isinstance(result, dict) and self.input_attribute in result:
                return result[self.input_attribute]
            else:
                raise ValueError(f"Result value does not have attribute/key '{self.input_attribute}'")
        return result

    def _get_comparison_values(self, inputs: List[NodeResult]) -> List[Any]:
        """Extract the values to compare from the inputs."""
        if not inputs:
            raise ValueError("No inputs provided for comparison")

        if self.broadcast_comparison:
            # Return values from all inputs
            return [self._extract_value_from_result(input_node.result) for input_node in inputs]
        else:
            # Return value from the specified input only
            if self.input_index >= len(inputs):
                raise ValueError(f"Input index {self.input_index} is out of range for inputs of length {len(inputs)}")

            return [self._extract_value_from_result(inputs[self.input_index].result)]

    def _compare_single_value(self, input_value: Any) -> bool:
        """Compare a single input value using the specified comparison type or custom function."""
        # Use custom comparison function if provided
        if self.custom_comparison_fn:
            return self.custom_comparison_fn(input_value, self.comparison_value)

        # Otherwise use the specified comparison type
        if self.comparison_type == ComparisonType.EQUAL:
            return input_value == self.comparison_value
        elif self.comparison_type == ComparisonType.NOT_EQUAL:
            return input_value != self.comparison_value
        elif self.comparison_type == ComparisonType.GREATER_THAN:
            return input_value > self.comparison_value
        elif self.comparison_type == ComparisonType.LESS_THAN:
            return input_value < self.comparison_value
        elif self.comparison_type == ComparisonType.GREATER_THAN_OR_EQUAL:
            return input_value >= self.comparison_value
        elif self.comparison_type == ComparisonType.LESS_THAN_OR_EQUAL:
            return input_value <= self.comparison_value
        elif self.comparison_type == ComparisonType.CONTAINS:
            return self.comparison_value in input_value
        elif self.comparison_type == ComparisonType.NOT_CONTAINS:
            return self.comparison_value not in input_value
        elif self.comparison_type == ComparisonType.IS_EMPTY:
            return not input_value
        elif self.comparison_type == ComparisonType.IS_NOT_EMPTY:
            return bool(input_value)
        else:
            raise ValueError(f"Unsupported comparison type: {self.comparison_type}")

    def _evaluate_condition(self, input_values: List[Any]) -> bool:
        """
        Evaluate the condition based on all input values and join them according to predicate_join_type.
        """
        # Compare each input value
        results = [self._compare_single_value(value) for value in input_values]

        # No results means no inputs, so return False
        if not results:
            return False

        # Join the results based on the predicate join type
        if self.predicate_join_type == PredicateJoinType.AND:
            return all(results)  # All must be True
        elif self.predicate_join_type == PredicateJoinType.OR:
            return any(results)  # At least one must be True
        else:
            raise ValueError(f"Unsupported predicate join type: {self.predicate_join_type}")

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute either positive_path or negative_path based on condition evaluation.

        Args:
            inputs: List of NodeResult objects to be used for condition evaluation

        Returns:
            List[NodeResult]: Results from the executed path
        """
        try:
            # Get values to compare from the inputs
            input_values = self._get_comparison_values(inputs)

            # Evaluate the condition with all collected values
            condition_result = self._evaluate_condition(input_values)

            # Execute the appropriate path based on the condition result
            if condition_result and self.positive_path:
                return self.positive_path.execute(inputs)
            elif not condition_result and self.negative_path:
                return self.negative_path.execute(inputs)
            else:
                # If the condition is false and there's no negative path (or vice versa),
                # return the original inputs
                return inputs

        except Exception as e:
            # Create an error result
            error_result = NodeResult(
                node_id=f"conditional_flow_error", result=None, error=e, step=inputs[-1].step + 1 if inputs else 0
            )
            return [error_result]
