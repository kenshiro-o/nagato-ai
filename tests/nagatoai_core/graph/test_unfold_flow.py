# Nagato AI
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.graph.unfold_flow import UnfoldFlow


def test_unfold_flow_with_numeric_lists():
    """Test UnfoldFlow with lists of numeric values"""
    # Create the unfold flow
    unfold_flow = UnfoldFlow()

    # Input with two lists of numbers
    inputs = [NodeResult(node_id="input1", result=[1, 2, 3]), NodeResult(node_id="input2", result=[4, 5])]

    # Execute the flow
    results = unfold_flow.execute(inputs)

    # Verify results - elements from the input lists should be preserved
    assert len(results) == 5  # Total of 5 elements in the input lists

    # Check each result value
    result_values = [r.result for r in results]
    assert 1 in result_values
    assert 2 in result_values
    assert 3 in result_values
    assert 4 in result_values
    assert 5 in result_values

    # Check node_ids - should include original input node_id
    for result in results:
        assert "input1" in result.node_id or "input2" in result.node_id


def test_unfold_flow_with_string_lists():
    """Test UnfoldFlow with lists of string values"""
    # Create the unfold flow
    unfold_flow = UnfoldFlow()

    # Input with a list of strings
    inputs = [
        NodeResult(node_id="input1", result=["hello", "world"]),
        NodeResult(node_id="input2", result=["python", "test"]),
    ]

    # Execute the flow
    results = unfold_flow.execute(inputs)

    # Verify results - string elements should be preserved
    assert len(results) == 4

    # Check each result value
    result_values = [r.result for r in results]
    assert "hello" in result_values
    assert "world" in result_values
    assert "python" in result_values
    assert "test" in result_values


def test_unfold_flow_with_empty_lists():
    """Test UnfoldFlow with empty lists and skip_empty_lists option"""
    # Create flow with skip_empty_lists=True
    unfold_flow = UnfoldFlow(skip_empty_lists=True)

    # Input with one empty list and one non-empty list
    inputs = [NodeResult(node_id="input1", result=[]), NodeResult(node_id="input2", result=[4, 5])]

    # Execute the flow
    results = unfold_flow.execute(inputs)

    # Verify results - should only process the non-empty list
    assert len(results) == 2
    assert 4 in [r.result for r in results]
    assert 5 in [r.result for r in results]

    # Flow with skip_empty_lists=False (default) should raise an error
    unfold_flow = UnfoldFlow(skip_empty_lists=False)

    # Execution should fail with an error about the empty list
    result = unfold_flow.execute(inputs)
    assert len(result) == 1
    assert result[0].node_id == "unfold_flow_error"
    assert result[0].error is not None
    assert "empty list" in str(result[0].error)


def test_unfold_flow_with_non_list_inputs():
    """Test UnfoldFlow with non-list inputs and wrap_non_list_inputs option"""
    # Create flow with wrap_non_list_inputs=True
    unfold_flow = UnfoldFlow(wrap_non_list_inputs=True)

    # Input with one list and one non-list value
    inputs = [NodeResult(node_id="input1", result=[1, 2]), NodeResult(node_id="input2", result=3)]  # Not a list

    # Execute the flow
    results = unfold_flow.execute(inputs)

    # Verify results - should wrap the non-list value and process all
    assert len(results) == 3
    result_values = [r.result for r in results]
    assert 1 in result_values
    assert 2 in result_values
    assert 3 in result_values

    # Flow with wrap_non_list_inputs=False should raise an error
    unfold_flow = UnfoldFlow(wrap_non_list_inputs=False)

    # Execution should fail with an error about the non-list input
    result = unfold_flow.execute(inputs)
    assert len(result) == 1
    assert result[0].node_id == "unfold_flow_error"
    assert result[0].error is not None
    assert "non-list result" in str(result[0].error)


def test_unfold_flow_with_preserve_metadata():
    """Test UnfoldFlow with preserve_metadata option"""
    # Create a flow with preserve_metadata=True
    unfold_flow = UnfoldFlow(preserve_metadata=True)

    # Input with a list and explicit step value
    inputs = [NodeResult(node_id="input1", result=[1, 2, 3], step=5)]

    # Execute the flow
    results = unfold_flow.execute(inputs)

    # Verify results - node_ids should include original id
    assert len(results) == 3

    # Check that the node IDs have preserved the original input ID plus element info
    node_ids = [r.node_id for r in results]
    for i in range(3):
        assert "input1_element" in node_ids[i]

    # Check if step values match the input step (5)
    for result in results:
        assert result.step == 5  # Step should be preserved


def test_unfold_flow_with_nested_lists():
    """Test UnfoldFlow with nested lists - it should only unfold the top level"""
    # Create the unfold flow
    unfold_flow = UnfoldFlow()

    # Input with nested lists
    inputs = [
        NodeResult(node_id="input1", result=[[1, 2], [3, 4]]),  # Nested lists
        NodeResult(node_id="input2", result=[[5]]),  # Another nested list
    ]

    # Execute the flow
    results = unfold_flow.execute(inputs)

    # The unfold should only operate on the top level lists
    assert len(results) == 3  # 2 sublists from input1, 1 from input2

    # Check result values - should be the sublists themselves
    result_values = [r.result for r in results]
    assert [1, 2] in result_values
    assert [3, 4] in result_values
    assert [5] in result_values


def test_unfold_flow_error_handling():
    """Test UnfoldFlow error handling for general exceptions"""
    # Create the unfold flow with wrap_non_list_inputs=False to trigger an error
    unfold_flow = UnfoldFlow(wrap_non_list_inputs=False)

    # Test with invalid input (None)
    inputs = [NodeResult(node_id="input1", result=None)]

    # This should cause an error since None is not a list and wrap_non_list_inputs is False
    results = unfold_flow.execute(inputs)

    # Check error handling
    assert len(results) == 1
    assert results[0].node_id == "unfold_flow_error"
    assert results[0].error is not None
    assert "non-list result" in str(results[0].error)
