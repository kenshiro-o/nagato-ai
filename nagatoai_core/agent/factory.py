# Standard Library
from typing import Type

# Third Party
from anthropic import Anthropic
from google import genai
from groq import Groq
from openai import OpenAI

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.anthropic import AnthropicAgent

# Company Libraries
from nagatoai_core.agent.deepseek import DeepSeekAgent
from nagatoai_core.agent.google import GoogleAgent
from nagatoai_core.agent.groq import GroqAgent
from nagatoai_core.agent.openai import OpenAIAgent
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider
from nagatoai_core.tool.provider.anthropic import AnthropicToolProvider
from nagatoai_core.tool.provider.google import GoogleToolProvider
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


def create_agent(api_key: str, model: str, role: str, role_description: str, nickname: str) -> Agent:
    """
    Creates an agent based on the model, role, role_description and nickname.
    :param api_key: The API key to be used by the agent.
    :param model: The model to be used by the agent.
    :param role: The role of the agent.
    :param role_description: The role description of the agent. This is essentially the system message
    :param nickname: The nickname of the agent.
    :return: The agent instance.
    """
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        client = OpenAI(api_key=api_key)
        return OpenAIAgent(client, model, role, role_description, nickname)

    if model.startswith("deepseek"):
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return DeepSeekAgent(client, model, role, role_description, nickname)

    if model.startswith("claude"):
        client = Anthropic(api_key=api_key)
        return AnthropicAgent(client, model, role, role_description, nickname)

    if model.startswith("groq-"):
        model = model.replace("groq-", "")
        client = Groq(api_key=api_key)
        return GroqAgent(client, model, role, role_description, nickname)

    if model.startswith("gemini"):
        client = genai.Client(api_key=api_key)
        return GoogleAgent(client, model, role, role_description, nickname)

    raise ValueError(f"Unsupported model: {model}")


def get_agent_tool_provider(agent: Agent) -> Type[AbstractToolProvider]:
    """
    Gets the tool provider for the agent.
    :param agent: The agent.
    :return: The tool provider for the agent.
    """
    if isinstance(agent, OpenAIAgent):
        return OpenAIToolProvider

    if isinstance(agent, DeepSeekAgent):
        return OpenAIToolProvider

    if isinstance(agent, GroqAgent):
        return OpenAIToolProvider

    if isinstance(agent, AnthropicAgent):
        return AnthropicToolProvider

    if isinstance(agent, GoogleAgent):
        return GoogleToolProvider

    raise ValueError(f"Unsupported agent: {agent}")
