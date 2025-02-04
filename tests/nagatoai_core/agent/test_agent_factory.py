# Standard Library
from unittest.mock import MagicMock, patch

# Third Party
import google.generativeai as genai
import pytest
from anthropic import Anthropic
from groq import Groq
from openai import OpenAI

# Nagato AI
from nagatoai_core.agent.anthropic import AnthropicAgent
from nagatoai_core.agent.deepseek import DeepSeekAgent
from nagatoai_core.agent.factory import create_agent, get_agent_tool_provider
from nagatoai_core.agent.google import GoogleAgent
from nagatoai_core.agent.groq import GroqAgent
from nagatoai_core.agent.openai import OpenAIAgent
from nagatoai_core.tool.provider.anthropic import AnthropicToolProvider
from nagatoai_core.tool.provider.google import GoogleToolProvider
from nagatoai_core.tool.provider.openai import OpenAIToolProvider

TEST_API_KEY = "test-api-key"
TEST_ROLE = "test-role"
TEST_ROLE_DESCRIPTION = "test-description"
TEST_NICKNAME = "test-nickname"


@pytest.mark.parametrize(
    "model,expected_agent_class",
    [
        ("gpt-4", OpenAIAgent),
        ("gpt-3.5-turbo", OpenAIAgent),
        ("o1", OpenAIAgent),
        ("o3", OpenAIAgent),
        ("claude-3-opus", AnthropicAgent),
        ("claude-2", AnthropicAgent),
        ("groq-mixtral-8x7b", GroqAgent),
        ("gemini-pro", GoogleAgent),
        ("deepseek-coder", DeepSeekAgent),
    ],
)
def test_create_agent_returns_correct_agent_type(model, expected_agent_class):
    # Patch all possible clients to avoid actual API calls
    with (
        patch("openai.OpenAI"),
        patch("anthropic.Anthropic"),
        patch("groq.Groq"),
        patch("google.generativeai.configure"),
        patch("google.generativeai.GenerativeModel"),
    ):

        agent = create_agent(
            api_key=TEST_API_KEY,
            model=model,
            role=TEST_ROLE,
            role_description=TEST_ROLE_DESCRIPTION,
            nickname=TEST_NICKNAME,
        )

        assert isinstance(agent, expected_agent_class)


def test_create_agent_raises_error_for_unsupported_model():
    with pytest.raises(ValueError) as exc_info:
        create_agent(
            api_key=TEST_API_KEY,
            model="unsupported-model",
            role=TEST_ROLE,
            role_description=TEST_ROLE_DESCRIPTION,
            nickname=TEST_NICKNAME,
        )

    assert "Unsupported model: unsupported-model" in str(exc_info.value)


def test_create_agent_configures_deepseek_base_url():
    with patch("nagatoai_core.agent.factory.OpenAI") as mock_openai:
        create_agent(
            api_key=TEST_API_KEY,
            model="deepseek-coder",
            role=TEST_ROLE,
            role_description=TEST_ROLE_DESCRIPTION,
            nickname=TEST_NICKNAME,
        )

        mock_openai.assert_called_once_with(api_key=TEST_API_KEY, base_url="https://api.deepseek.com")


def test_create_agent_strips_groq_prefix():
    with patch("groq.Groq") as mock_groq:
        agent = create_agent(
            api_key=TEST_API_KEY,
            model="groq-mixtral-8x7b",
            role=TEST_ROLE,
            role_description=TEST_ROLE_DESCRIPTION,
            nickname=TEST_NICKNAME,
        )

        assert agent.model == "mixtral-8x7b"


def test_create_agent_configures_gemini():
    with (
        patch("google.generativeai.configure") as mock_configure,
        patch("google.generativeai.GenerativeModel") as mock_model,
    ):

        create_agent(
            api_key=TEST_API_KEY,
            model="gemini-pro",
            role=TEST_ROLE,
            role_description=TEST_ROLE_DESCRIPTION,
            nickname=TEST_NICKNAME,
        )

        mock_configure.assert_called_once_with(api_key=TEST_API_KEY)
        mock_model.assert_called_once_with("gemini-pro")


@pytest.mark.parametrize(
    "agent_class,expected_provider",
    [
        (OpenAIAgent, OpenAIToolProvider),
        (DeepSeekAgent, OpenAIToolProvider),
        (GroqAgent, OpenAIToolProvider),
        (AnthropicAgent, AnthropicToolProvider),
        (GoogleAgent, GoogleToolProvider),
    ],
)
def test_get_agent_tool_provider_returns_correct_provider(agent_class, expected_provider):
    # Create a mock agent instance
    mock_agent = MagicMock(spec=agent_class)

    # Get the provider
    provider = get_agent_tool_provider(mock_agent)

    # Assert the correct provider is returned
    assert provider == expected_provider


def test_get_agent_tool_provider_raises_error_for_unsupported_agent():
    # Create a mock agent that doesn't match any known types
    mock_agent = MagicMock()

    with pytest.raises(ValueError) as exc_info:
        get_agent_tool_provider(mock_agent)

    assert "Unsupported agent:" in str(exc_info.value)
