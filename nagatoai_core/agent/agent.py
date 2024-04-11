from abc import ABC, abstractmethod
from typing import List, Union
import uuid

from .message import Exchange, ToolResult
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider


class Agent(ABC):
    """
    Agent represents the base class for all AI Agents.
    Any concrete AI Agent must inherit from this class and implement the generate_response method.
    """

    def __init__(
        self,
        model: str,
        role: str,
        role_description: str,
        nickname: str,
    ):
        """
        Intializes the Agent with the model, role, temperature and nickname.
        :param model: The model to be used by the agent.
        :param role: The role of the agent.
        :param role_description: The role description of the agent. This is essentially the system message
        :param nickname: The nickname of the agent.
        """
        self.agent_id = uuid.uuid4()
        self.model = model
        self.role = role
        self.role_description = role_description
        self.nickname = nickname

    @property
    def id(self) -> str:
        """
        Returns the agent's id as a string.
        """
        return str(self.agent_id)

    @abstractmethod
    def chat(
        self,
        prompt: str,
        tools: List[AbstractToolProvider],
        temperature: float,
        max_tokens: int,
    ) -> Exchange:
        """
        Generates a response for the current prompt and prompt history.
        :param prompt: The current prompt.
        :param tools: the tools available to the agent.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        :return: Exchange object containing the user message and the agent response.
        """
        pass

    @abstractmethod
    def send_tool_run_results(
        self, tool_results: List[ToolResult], temperature: float, max_tokens: int
    ) -> Exchange:
        """
        Returns the results of the running of one or multiple tools
        :param tool_results: The results of the running of one or multiple tools
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        :return: Exchange object containing the user message and the agent response.
        """
        pass

    @property
    @abstractmethod
    def maker(self) -> str:
        """
        Returns the agent's model maker (e.g. OpenAI)
        """
        pass

    @property
    @abstractmethod
    def history(self) -> List[Exchange]:
        """
        Returns the agent's conversation history.
        """
        pass

    @property
    @abstractmethod
    def family(self) -> str:
        """
        Returns the agent's model family (e.g. GPT-4)
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns the agent's nickname
        """
        return self.nickname
