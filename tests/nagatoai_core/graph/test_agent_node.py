# Standard Library
from datetime import datetime, timezone
from typing import Any, List, Optional, Type
from unittest.mock import MagicMock, patch

# Third Party
import pytest
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.message import Exchange, Message, Sender, TokenStatsAndParams
from nagatoai_core.graph.agent_node import AgentNode
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.mission.task import Task
from nagatoai_core.prompt.template.prompt_template import PromptTemplate
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider


class MockOutputSchema(BaseModel):
    """Mock output schema for testing purposes."""

    result: str = Field(description="The result of the agent execution")


@pytest.fixture
def mock_agent():
    """Fixture providing a mock Agent instance."""
    agent = MagicMock(spec=Agent)
    agent.id = "mock_agent_id"

    # Setup the mock chat method to return a realistic Exchange object
    def mock_chat(**kwargs):
        return Exchange(
            chat_history=[],
            user_msg=Message(
                sender=Sender.USER, content=kwargs.get("prompt", ""), created_at=datetime.now(timezone.utc)
            ),
            agent_response=Message(
                sender=Sender.AGENT, content="This is a mock agent response", created_at=datetime.now(timezone.utc)
            ),
            token_stats_and_params=TokenStatsAndParams(
                input_tokens_used=10,
                output_tokens_used=10,
                max_tokens=kwargs.get("max_tokens", 150),
                temperature=kwargs.get("temperature", 0.7),
            ),
        )

    agent.chat.side_effect = mock_chat
    return agent


@pytest.fixture
def mock_prompt_template():
    """Fixture providing a mock PromptTemplate instance."""
    template = MagicMock(spec=PromptTemplate)
    template.generate_prompt.return_value = "Processed prompt from template with inputs"
    return template


@pytest.fixture
def mock_task():
    """Fixture providing a mock Task instance."""
    task = MagicMock(spec=Task)
    task.id = "mock_task_id"
    task.name = "Mock Task"
    return task


@pytest.fixture
def mock_tool_provider():
    """Fixture providing a mock AbstractToolProvider instance."""
    provider = MagicMock(spec=AbstractToolProvider)
    provider.name = "mock_tool"
    provider.description = "Mock tool for testing purposes"
    return provider


def test_agent_node_with_prompt_template(mock_agent, mock_prompt_template, mock_task, mock_tool_provider):
    """Test the execution of an AgentNode with a prompt template."""

    node = AgentNode(
        id="agent_node",
        agent=mock_agent,
        task=mock_task,
        prompt_template=mock_prompt_template,
        tools=[mock_tool_provider],
        temperature=0.5,
        max_tokens=100,
    )

    inputs = [NodeResult(node_id="previous_node", result="test input", step=1)]

    results = node.execute(inputs)

    # Verify prompt template was used
    mock_prompt_template.generate_prompt.assert_called_once()
    prompt_data = mock_prompt_template.generate_prompt.call_args[0][0]
    assert "inputs" in prompt_data
    assert prompt_data["inputs"] == inputs

    # Verify agent chat was called with correct parameters
    mock_agent.chat.assert_called_once()
    chat_args = mock_agent.chat.call_args[1]
    assert chat_args["task"] == mock_task
    assert chat_args["prompt"] == "Processed prompt from template with inputs"
    assert chat_args["tools"] == [mock_tool_provider]
    assert chat_args["temperature"] == 0.5
    assert chat_args["max_tokens"] == 100

    # Verify result
    assert isinstance(results, list)
    assert len(results) == 1
    result_item = results[0]
    assert isinstance(result_item, NodeResult)
    assert result_item.node_id == "agent_node"
    assert result_item.result == "This is a mock agent response"
    assert result_item.step == 2  # Incremented from input step 1


def test_agent_node_without_prompt_template(mock_agent):
    """Test the execution of an AgentNode without a prompt template."""

    node = AgentNode(id="agent_node", agent=mock_agent)

    inputs = [NodeResult(node_id="previous_node", result="test input", step=3)]

    results = node.execute(inputs)

    # Verify agent chat was called with correct parameters
    mock_agent.chat.assert_called_once()
    chat_args = mock_agent.chat.call_args[1]
    assert chat_args["prompt"] == ""  # Empty prompt when no template is provided

    # Verify result
    assert isinstance(results, list)
    assert len(results) == 1
    result_item = results[0]
    assert isinstance(result_item, NodeResult)
    assert result_item.node_id == "agent_node"
    assert result_item.result == "This is a mock agent response"
    assert result_item.step == 4  # Incremented from input step 3


def test_agent_node_with_output_schema(mock_agent):
    """Test the execution of an AgentNode with a specified output schema."""

    node = AgentNode(id="agent_node", agent=mock_agent, output_schema=MockOutputSchema)

    inputs = [NodeResult(node_id="previous_node", result="test input", step=1)]

    results = node.execute(inputs)

    # Verify output schema was passed to agent
    mock_agent.chat.assert_called_once()
    chat_args = mock_agent.chat.call_args[1]
    assert chat_args["target_output_schema"] == MockOutputSchema

    # Verify result
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].step == 2


def test_agent_node_step_increment(mock_agent):
    """Test that the AgentNode properly increments the step counter."""

    node = AgentNode(id="agent_node", agent=mock_agent)

    # Test with various step values
    for step in [0, 5, 10, 99]:
        inputs = [NodeResult(node_id="previous_node", result="test input", step=step)]
        results = node.execute(inputs)
        assert results[0].step == step + 1


def test_agent_node_parameter_passing(mock_agent):
    """Test that all parameters are correctly passed to the agent's chat method."""

    # Create node with all possible parameters
    node = AgentNode(
        id="agent_node",
        agent=mock_agent,
        task=MagicMock(spec=Task),
        tools=[MagicMock(spec=AbstractToolProvider)],
        temperature=0.33,
        max_tokens=123,
        output_schema=MockOutputSchema,
    )

    inputs = [NodeResult(node_id="previous_node", result="test input", step=1)]

    node.execute(inputs)

    # Verify all parameters were correctly passed
    mock_agent.chat.assert_called_once()
    chat_args = mock_agent.chat.call_args[1]
    assert chat_args["task"] == node.task
    assert chat_args["prompt"] == ""
    assert chat_args["tools"] == node.tools
    assert chat_args["temperature"] == 0.33
    assert chat_args["max_tokens"] == 123
    assert chat_args["target_output_schema"] == MockOutputSchema


@patch("nagatoai_core.graph.agent_node.AbstractNode.logger")
def test_agent_node_logging(mock_logger, mock_agent, mock_prompt_template):
    """Test that the AgentNode properly logs information."""

    node = AgentNode(id="agent_node", agent=mock_agent, prompt_template=mock_prompt_template)

    inputs = [NodeResult(node_id="previous_node", result="test input", step=1)]

    node.execute(inputs)

    # Verify logging calls - the AgentNode now uses debug level instead of info
    assert mock_logger.debug.call_count >= 2

    # First log should be about prompt data
    prompt_data_log = mock_logger.debug.call_args_list[0][0][0]
    assert "Prompt data" in prompt_data_log

    # Second log should be about the full prompt
    full_prompt_log = mock_logger.debug.call_args_list[1][0][0]
    assert "Full prompt" in full_prompt_log
