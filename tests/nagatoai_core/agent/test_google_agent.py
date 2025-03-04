"""Unit tests for the GoogleAgent class."""

# Standard Library
import os
from datetime import datetime, timezone
from typing import List

# Third Party
import pytest
from google import genai
from google.genai import types

# Nagato AI
from nagatoai_core.agent.google import GoogleAgent
from nagatoai_core.agent.message import Exchange, Message, Sender, TokenStatsAndParams, ToolCall, ToolResult
from nagatoai_core.tool.lib.time.time_now import TimeNowConfig, TimeNowTool
from nagatoai_core.tool.provider.google import GoogleToolProvider


@pytest.fixture
def google_client():
    """Creates a real Google generative AI client.

    Requires GOOGLE_API_KEY environment variable to be set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    return client


def test_chat_basic_prompt(google_client):
    """Test the chat method with a basic prompt without tools using real Google API."""
    agent = GoogleAgent(
        client=google_client,
        model="gemini-2.0-flash",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # Act
    result = agent.chat(task=None, prompt="What is the capital of France?", tools=[], temperature=0.7, max_tokens=100)

    # Assert
    assert isinstance(result, Exchange)
    assert result.user_msg.content == "What is the capital of France?"
    assert result.user_msg.sender == Sender.USER
    assert isinstance(result.agent_response.content, str)
    assert "Paris" in result.agent_response.content
    assert result.agent_response.sender == Sender.AGENT
    assert result.token_stats_and_params.input_tokens_used > 0
    assert result.token_stats_and_params.output_tokens_used > 0
    assert result.token_stats_and_params.temperature == 0.7
    assert result.token_stats_and_params.max_tokens == 100

    # Check that messages were formatted correctly
    # messages = result.chat_history
    # assert len(messages) == 1  # Should have one message
    # assert messages[0]["role"] == "user"
    # assert len(messages[0]["parts"]) == 2  # Should have role description and prompt
    # assert "You are a helpful AI assistant" in messages[0]["parts"][0]
    # assert messages[0]["parts"][1] == "What is the capital of France?"


def test_chat_multi_turn_conversation(google_client):
    """Test the chat method with a multi-turn conversation using real Google API."""

    agent = GoogleAgent(
        client=google_client,
        model="gemini-2.0-flash",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # Act
    result = agent.chat(
        task=None,
        prompt="What is the capital of France?",
        tools=[],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert
    assert isinstance(result, Exchange)
    assert result.user_msg.content == "What is the capital of France?"
    assert result.user_msg.sender == Sender.USER
    assert isinstance(result.agent_response.content, str)

    # Now ask what is the capital of Italy
    result = agent.chat(
        task=None,
        prompt="What is the capital of Italy?",
        tools=[],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert
    assert isinstance(result, Exchange)
    assert result.user_msg.content == "What is the capital of Italy?"
    assert result.user_msg.sender == Sender.USER
    assert isinstance(result.agent_response.content, str)
    assert "Rome" in result.agent_response.content

    # Check that the message history has two user messages and two agent messages
    chat_history: List[Exchange] = agent.history
    assert len(chat_history) == 2
    assert chat_history[0].user_msg.content == "What is the capital of France?"
    assert chat_history[0].agent_response.content.strip() == "The capital of France is Paris."
    assert chat_history[1].user_msg.content == "What is the capital of Italy?"
    assert chat_history[1].agent_response.content.strip() == "The capital of Italy is Rome."


def test_chat_with_tool(google_client):
    """Test that the agent can properly make tool calls using TimeNowTool."""

    agent = GoogleAgent(
        client=google_client,
        model="gemini-2.0-flash",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    tt = TimeNowTool()
    time_tool = GoogleToolProvider(
        name=tt.name,
        description=tt.description,
        args_schema=tt.args_schema,
        tool=tt,
    )

    # Act
    result = agent.chat(
        task=None,
        prompt="What time is it right now? Please use the TimeNowTool to check with UTC timezone. Your reply should start with the sentence 'It is currently ...'",
        tools=[time_tool],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert
    assert isinstance(result, Exchange)
    assert result.agent_response.tool_calls is not None
    assert len(result.agent_response.tool_calls) == 1
    assert result.agent_response.tool_calls[0].name == tt.name


def test_chat_then_tool_then_tool_response(google_client):
    """Test that the agent can handle a chat, followed by a tool call, and then process the tool results."""

    agent = GoogleAgent(
        client=google_client,
        model="gemini-2.0-flash",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # First chat message about France
    result = agent.chat(
        task=None,
        prompt="What is the capital of France?",
        tools=[],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert first response
    assert isinstance(result, Exchange)
    assert result.user_msg.content == "What is the capital of France?"
    assert "Paris" in result.agent_response.content

    # Set up TimeNowTool for second message
    tt = TimeNowTool()
    time_tool = GoogleToolProvider(
        name=tt.name,
        description=tt.description,
        args_schema=tt.args_schema,
        tool=tt,
    )

    # Second message requesting time
    result = agent.chat(
        task=None,
        prompt="What time is it right now? Please use the TimeNowTool to check with UTC timezone. Your reply should start with the sentence 'It is currently ...'",
        tools=[time_tool],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert tool call was made
    assert isinstance(result, Exchange)
    assert result.agent_response.tool_calls is not None
    assert len(result.agent_response.tool_calls) == 1
    assert result.agent_response.tool_calls[0].name == tt.name

    # Execute tool and send results back
    tool_result = tt._run(TimeNowConfig(use_utc_timezone=True))

    final_result = agent.send_tool_run_results(
        task=None,
        tool_results=[
            ToolResult(id=result.agent_response.tool_calls[0].id, name=tt.name, result=tool_result, error=None)
        ],
        tools=[time_tool],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert final response contains the time
    assert isinstance(final_result, Exchange)
    assert final_result.agent_response.content is not None
    assert "it is currently" in final_result.agent_response.content.lower()


def test_chat_then_tool_then_tool_response_with_error(google_client):
    """Test that the agent can handle a chat, followed by a tool call, and then process the tool results."""

    agent = GoogleAgent(
        client=google_client,
        model="gemini-2.0-flash",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # Act
    result = agent.chat(
        task=None,
        prompt="What is the capital of France?",
        tools=[],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert
    assert isinstance(result, Exchange)
    assert result.user_msg.content == "What is the capital of France?"
    assert "Paris" in result.agent_response.content

    # Set up TimeNowTool for second message
    tt = TimeNowTool()
    time_tool = GoogleToolProvider(
        name=tt.name,
        description=tt.description,
        args_schema=tt.args_schema,
        tool=tt,
    )

    # Second message requesting time
    result = agent.chat(
        task=None,
        prompt="What time is it right now? Please use the TimeNowTool to check with UTC timezone. Your reply should start with the sentence 'It is currently ...'",
        tools=[time_tool],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert tool call was made
    assert isinstance(result, Exchange)
    assert result.agent_response.tool_calls is not None
    assert len(result.agent_response.tool_calls) == 1
    assert result.agent_response.tool_calls[0].name == tt.name

    # Execute tool and send results back with error

    tool_error = "Error: Failed to get time"

    final_result = agent.send_tool_run_results(
        task=None,
        tool_results=[
            ToolResult(id=result.agent_response.tool_calls[0].id, name=tt.name, result=None, error=tool_error)
        ],
        tools=[time_tool],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert final response contains the error
    assert isinstance(final_result, Exchange)
    assert final_result.agent_response.content is not None
    assert any(keyword in final_result.agent_response.content.lower() for keyword in ["error", "unavailable"])


def test_chat_then_tool_then_tool_response_then_chat(google_client):
    """Test that the agent can handle a chat, followed by a tool call, then a tool response, and then a chat."""

    agent = GoogleAgent(
        client=google_client,
        model="gemini-2.0-flash",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # Act
    result = agent.chat(
        task=None,
        prompt="What is the capital of France?",
        tools=[],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert
    assert isinstance(result, Exchange)
    assert result.user_msg.content == "What is the capital of France?"
    assert "Paris" in result.agent_response.content

    # Set up TimeNowTool for second message
    tt = TimeNowTool()
    time_tool = GoogleToolProvider(
        name=tt.name,
        description=tt.description,
        args_schema=tt.args_schema,
        tool=tt,
    )

    # Second message requesting time
    result = agent.chat(
        task=None,
        prompt="What time is it right now? Please use the TimeNowTool to check with UTC timezone. Your reply should start with the sentence 'It is currently ...'",
        tools=[time_tool],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert tool call was made
    assert isinstance(result, Exchange)
    assert result.agent_response.tool_calls is not None
    assert len(result.agent_response.tool_calls) == 1
    assert result.agent_response.tool_calls[0].name == tt.name

    # Execute tool and send results back
    tool_result = tt._run(TimeNowConfig(use_utc_timezone=True))

    final_result = agent.send_tool_run_results(
        task=None,
        tool_results=[
            ToolResult(id=result.agent_response.tool_calls[0].id, name=tt.name, result=tool_result, error=None)
        ],
        tools=[time_tool],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert tool response was made
    assert isinstance(final_result, Exchange)
    assert final_result.agent_response.content is not None
    assert "it is currently" in final_result.agent_response.content.lower()

    # Third chat message
    result = agent.chat(
        task=None,
        prompt="What is the capital of the United States?",
        tools=[],
        temperature=0.7,
        max_tokens=100,
    )

    # Assert
    assert isinstance(result, Exchange)
    assert result.user_msg.content == "What is the capital of the United States?"
    assert "Washington, D.C." in result.agent_response.content
