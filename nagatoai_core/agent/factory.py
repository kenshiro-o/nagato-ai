from openai import OpenAI
from anthropic import Anthropic
from groq import Groq

from .agent import Agent
from .openai import OpenAIAgent
from .groq import GroqAgent
from .anthropic import AnthropicAgent


def create_agent(
    api_key: str, model: str, role: str, role_description: str, nickname: str
) -> Agent:
    """
    Creates an agent based on the model, role, role_description and nickname.
    :param api_key: The API key to be used by the agent.
    :param model: The model to be used by the agent.
    :param role: The role of the agent.
    :param role_description: The role description of the agent. This is essentially the system message
    :param nickname: The nickname of the agent.
    :return: The agent instance.
    """
    if model.startswith("gpt"):
        client = OpenAI(api_key=api_key)
        return OpenAIAgent(client, model, role, role_description, nickname)

    if model.startswith("claude"):
        client = Anthropic(api_key=api_key)
        return AnthropicAgent(client, model, role, role_description, nickname)

    if model.startswith("llama3"):
        client = Groq(api_key=api_key)
        return GroqAgent(client, model, role, role_description, nickname)

    raise ValueError(f"Unsupported model: {model}")
