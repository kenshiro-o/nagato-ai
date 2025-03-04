# Standard Library
from typing import Any, Dict, List, Optional, Type
from unittest.mock import MagicMock, patch

# Third Party
import pytest
from pydantic import BaseModel, Field, ValidationError

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.message import Exchange, Message

# from nagatoai_core.agent.models import AgentMessage, Exchange
from nagatoai_core.graph.tool_node_with_params_conversion import ToolNodeWithParamsConversion
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.tool.abstract_tool import AbstractTool
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


class TestToolParams(BaseModel):
    """Test parameters for tool execution."""

    name: str
    count: int
    optional_param: Optional[str] = None


class TestTool(AbstractTool):
    """A simple test tool for testing the ToolNodeWithParamsConversion class."""

    # Properly annotate fields that override base class attributes
    name: str = "test_tool"
    description: str = "A test tool for unit testing"
    args_schema: Type[BaseModel] = TestToolParams

    def _run(self, params: TestToolParams) -> Dict[str, Any]:
        """Run the test tool with the given parameters."""
        return {"name": params.name, "count": params.count, "optional_param": params.optional_param, "success": True}

    # We don't need to implement schema() as OpenAIToolProvider will generate it based on args_schema


@pytest.fixture
def mock_agent() -> Agent:
    """Create a mock agent for testing."""
    agent = MagicMock(spec=Agent)

    # Set up the chat method to return a mock response
    mock_exchange = MagicMock(spec=Exchange)
    mock_agent_message = MagicMock(spec=Message)
    mock_agent_message.content = """
    <params_instance>
    {
        "name": "test_name",
        "count": 42
    }
    </params_instance>
    """
    mock_exchange.agent_response = mock_agent_message
    agent.chat.return_value = mock_exchange

    return agent


@pytest.fixture
def test_tool() -> TestTool:
    """Create a test tool for testing."""
    return TestTool()


@pytest.fixture
def test_tool_provider(test_tool: TestTool) -> OpenAIToolProvider:
    """Create a tool provider using OpenAIToolProvider."""
    return OpenAIToolProvider(
        tool=test_tool, name=test_tool.name, description=test_tool.description, args_schema=test_tool.args_schema
    )


def test_execute_with_valid_params(mock_agent: Agent, test_tool_provider: OpenAIToolProvider):
    """Test execution with valid parameters without needing conversion."""
    # Create the node with the mock agent
    node = ToolNodeWithParamsConversion(
        id="test_node", name="Test Node", agent=mock_agent, tool_provider=test_tool_provider
    )

    # Create input with valid parameters
    input_data = [NodeResult(node_id="prev_node", result={"name": "test_name", "count": 42}, step=1)]

    # Execute the node
    result = node.execute(input_data)

    # Verify the result
    assert len(result) == 1
    assert result[0].node_id == "test_node"
    assert result[0].step == 2
    assert result[0].result["name"] == "test_name"
    assert result[0].result["count"] == 42
    assert result[0].result["success"] is True

    # Verify that the agent was not used
    mock_agent.chat.assert_not_called()


def test_execute_with_invalid_params_requiring_conversion(mock_agent: Agent, test_tool_provider: OpenAIToolProvider):
    """Test execution with invalid parameters that require conversion."""
    # Create the node with the mock agent
    node = ToolNodeWithParamsConversion(
        id="test_node", name="Test Node", agent=mock_agent, tool_provider=test_tool_provider
    )

    # Create input with invalid parameters (missing required parameter)
    input_data = [
        NodeResult(
            node_id="prev_node",
            result={"person_name": "John", "number": 42},  # Different field names that need conversion
            step=1,
        )
    ]

    # Execute the node
    result = node.execute(input_data)

    # Verify the result
    assert len(result) == 1
    assert result[0].node_id == "test_node"
    assert result[0].step == 2
    assert result[0].result["name"] == "test_name"  # Converted from person_name
    assert result[0].result["count"] == 42  # Converted from number
    assert result[0].result["success"] is True

    # Verify that the agent was used for conversion
    mock_agent.chat.assert_called_once()


def test_execute_with_repeated_conversion_attempts(mock_agent, test_tool_provider):
    """Test execution with multiple conversion attempts."""
    # Configure the agent to fail on the first attempt but succeed on the second
    mock_agent.chat.side_effect = [
        ValueError("Conversion failed"),  # First attempt fails
        MagicMock(  # Second attempt succeeds
            agent_response=MagicMock(
                content="""
                <params_instance>
                {
                    "name": "test_name",
                    "count": 42
                }
                </params_instance>
                """
            )
        ),
    ]

    # Create the node with the mock agent and multiple retries
    node = ToolNodeWithParamsConversion(
        id="test_node",
        name="Test Node",
        agent=mock_agent,
        tool_provider=test_tool_provider,
        retries=2,  # Allow 2 retry attempts (3 total attempts)
    )

    # Create input with invalid parameters
    input_data = [
        NodeResult(
            node_id="prev_node",
            result={"some_name": "test", "some_count": "not_an_integer"},  # Invalid parameters
            step=1,
        )
    ]

    # Patch the _convert_params method to fail on first attempt but succeed on second
    original_convert_params = node._convert_params
    attempt_count = [0]

    def mock_convert_params(input_data, schema_dict):
        attempt_count[0] += 1
        if attempt_count[0] == 1:
            raise ValueError("First conversion attempt failed")
        else:
            return {"name": "test_name", "count": 42}

    node._convert_params = mock_convert_params

    try:
        # Execute the node
        result = node.execute(input_data)

        # Verify the result
        assert len(result) == 1
        assert result[0].node_id == "test_node"
        assert result[0].step == 2
        assert result[0].result["name"] == "test_name"
        assert result[0].result["count"] == 42
        assert result[0].result["success"] is True

        # Verify that multiple conversion attempts were made
        assert attempt_count[0] == 2
    finally:
        # Restore the original method
        node._convert_params = original_convert_params


def test_execute_with_all_conversion_attempts_failing(mock_agent, test_tool_provider):
    """Test execution when all conversion attempts fail."""
    # Configure the agent to always fail
    mock_agent.chat.side_effect = ValueError("Conversion failed")

    # Create the node with the mock agent
    node = ToolNodeWithParamsConversion(
        id="test_node",
        name="Test Node",
        agent=mock_agent,
        tool_provider=test_tool_provider,
        retries=2,  # Allow 2 retry attempts (3 total attempts)
    )

    # Create input with invalid parameters
    input_data = [NodeResult(node_id="prev_node", result={"invalid_field": "test"}, step=1)]  # Invalid parameters

    # Patch the _convert_params method to always fail
    original_convert_params = node._convert_params

    def mock_convert_params(input_data, schema_dict):
        raise ValueError("Conversion attempt failed")

    node._convert_params = mock_convert_params

    try:
        # Execute the node and expect it to return a NodeResult with an error
        result = node.execute(input_data)

        # Verify that we got an error result
        assert len(result) == 1
        assert result[0].node_id == "test_node"
        assert result[0].step == 2
        assert result[0].result is None
        assert result[0].error is not None
    finally:
        # Restore the original method
        node._convert_params = original_convert_params


def test_clear_memory_after_conversion(mock_agent, test_tool_provider):
    """Test that agent memory is cleared after conversion if configured to do so."""
    # Create the node with the mock agent and clear_memory_after_conversion=True
    node = ToolNodeWithParamsConversion(
        id="test_node",
        name="Test Node",
        agent=mock_agent,
        tool_provider=test_tool_provider,
        clear_memory_after_conversion=True,
    )

    # Create input with invalid parameters that require conversion
    input_data = [
        NodeResult(
            node_id="prev_node",
            result={"person_name": "John", "number": 42},  # Different field names that need conversion
            step=1,
        )
    ]

    # Execute the node
    node.execute(input_data)

    # Verify that the agent memory was cleared
    mock_agent.clear_memory.assert_called_once()

    # Now test with clear_memory_after_conversion=False
    mock_agent.clear_memory.reset_mock()
    node.clear_memory_after_conversion = False

    # Execute the node again
    node.execute(input_data)

    # Verify that the agent memory was not cleared
    mock_agent.clear_memory.assert_not_called()
