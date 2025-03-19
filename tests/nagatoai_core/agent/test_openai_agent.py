"""Unit tests for the OpenAIAgent class."""

# Standard Library
import logging
import os
from datetime import datetime, timezone
from typing import List

# Third Party
import pytest
from openai import OpenAI
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.agent.message import Exchange, Message, Sender, TokenStatsAndParams, ToolCall, ToolResult
from nagatoai_core.agent.openai import OpenAIAgent
from nagatoai_core.tool.lib.time.time_now import TimeNowConfig, TimeNowTool
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


@pytest.fixture
def openai_client():
    """Creates a real OpenAI client for testing.

    Requires OPENAI_API_KEY environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    return client


def test_chat_basic_prompt(openai_client):
    """Test the chat method with a basic prompt without tools using real OpenAI API."""
    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # Act
    result = agent.chat(task=None, prompt="What is the capital of France?", tools=[], temperature=0.7, max_tokens=100)

    # Assert
    assert isinstance(result, Exchange)
    assert result.user_msg.content == "What is the capital of France?"
    assert "Paris" in result.agent_response.content


def test_chat_multi_turn_conversation(openai_client):
    """Test multi-turn conversation with the agent using real OpenAI API."""
    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # First turn
    result1 = agent.chat(task=None, prompt="What is the capital of France?", tools=[], temperature=0.7, max_tokens=100)

    # Second turn
    result2 = agent.chat(
        task=None, prompt="Tell me about a famous monument there", tools=[], temperature=0.7, max_tokens=100
    )

    # Assert
    assert result1.user_msg.content == "What is the capital of France?"
    assert "Paris" in result1.agent_response.content

    assert result2.user_msg.content == "Tell me about a famous monument there"
    assert "Eiffel" in result2.agent_response.content


def test_chat_with_tool(openai_client):
    """Test that the agent can properly make tool calls using TimeNowTool."""
    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    tt = TimeNowTool()
    time_tool = OpenAIToolProvider(
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
    assert len(result.agent_response.tool_calls) > 0
    assert result.agent_response.tool_calls[0].name == tt.name


def test_chat_then_tool_then_tool_response(openai_client):
    """Test a sequence of chat, tool call, and handling tool response using real OpenAI API."""
    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
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
    time_tool = OpenAIToolProvider(
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


def test_chat_then_tool_then_tool_response_with_error(openai_client):
    """Test tool call sequence with error handling using real OpenAI API."""
    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
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
    time_tool = OpenAIToolProvider(
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
        temperature=0,
        max_tokens=100,
    )

    # Assert final response contains the error
    assert isinstance(final_result, Exchange)
    assert final_result.agent_response.content is not None
    assert any(
        keyword in final_result.agent_response.content.lower()
        for keyword in ["error", "unavailable", "unable", "sorry"]
    )


def test_chat_then_tool_then_tool_response_then_chat(openai_client):
    """Test a complex interaction: chat, tool call, tool response, follow-up chat using real OpenAI API."""
    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
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

    tt = TimeNowTool()
    time_tool = OpenAIToolProvider(
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


def test_chat_with_structured_output(openai_client):
    """Test that the agent can return structured output using a target schema with real OpenAI API."""

    class CountryInfo(BaseModel):
        """Schema for country information."""

        name: str = Field(..., description="The name of the country")
        capital: str = Field(..., description="The capital city of the country")
        population: int = Field(..., description="The approximate population of the country")
        languages: List[str] = Field(..., description="Official languages spoken in the country")

    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # Act
    result = agent.chat(
        task=None,
        prompt="Provide information about France in a structured format.",
        tools=[],
        temperature=0.7,
        max_tokens=100,
        target_output_schema=CountryInfo,
    )

    # Assert
    assert isinstance(result, Exchange)
    assert isinstance(result.agent_response.content, CountryInfo)
    assert result.agent_response.content.name == "France"
    assert result.agent_response.content.capital == "Paris"
    assert isinstance(result.agent_response.content.population, int)
    assert len(result.agent_response.content.languages) > 0
    assert "French" in result.agent_response.content.languages


def test_clear_memory(openai_client):
    """Test that the agent can clear its memory."""
    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    # Generate some history
    agent.chat(task=None, prompt="What is the capital of France?", tools=[], temperature=0.7, max_tokens=100)

    # History should exist
    assert len(agent.history) == 1

    # Clear memory
    agent.clear_memory()

    # History should be empty
    assert len(agent.history) == 0


def test_agent_properties(openai_client):
    """Test the agent properties like maker and family."""
    agent = OpenAIAgent(
        client=openai_client,
        model="gpt-4o",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent",
    )

    assert agent.maker.lower() == "openai"
    assert agent.family.lower() == "gpt-4o"

    # Test with different model
    agent2 = OpenAIAgent(
        client=openai_client,
        model="o3-mini",
        role="assistant",
        role_description="You are a helpful AI assistant.",
        nickname="TestAgent2",
    )

    assert agent2.maker.lower() == "openai"
    assert agent2.family.lower() == "o3"
