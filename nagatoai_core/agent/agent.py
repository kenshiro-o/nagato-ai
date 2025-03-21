# Standard Library
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Union

# Third Party
from pydantic import BaseModel

# Nagato AI
# Company Libraries
from nagatoai_core.common.structured_logger import StructuredLogger
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider

from .message import Exchange, ToolResult


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
        self.logger = StructuredLogger.get_logger(
            {
                "agent_id": self.agent_id,
                "model": self.model,
                "role": self.role,
                "nickname": self.nickname,
            }
        )

        self.logger.info("Agent initialized")

    @property
    def id(self) -> str:
        """
        Returns the agent's id as a string.
        """
        return str(self.agent_id)

    @abstractmethod
    def chat(
        self,
        task: Optional[Task],
        prompt: str,
        tools: List[AbstractToolProvider],
        temperature: float,
        max_tokens: int,
        target_output_schema: Optional[Union[BaseModel, List[BaseModel]]] = None,
    ) -> Exchange:
        """
        Generates a response for the current prompt and prompt history.
        :param task: The task object details of the task being run.
        :param prompt: The current prompt.
        :param tools: the tools available to the agent.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        :return: Exchange object containing the user message and the agent response.
        """
        pass

    @abstractmethod
    def send_tool_run_results(
        self,
        task: Optional[Task],
        tool_results: List[ToolResult],
        tools: List[AbstractToolProvider],
        temperature: float,
        max_tokens: int,
    ) -> Exchange:
        """
        Returns the results of the running of one or multiple tools
        :param task: The task object details of the task being run
        :param tool_results: The results of the running of one or multiple tools
        :param tools: the tools available to the agent.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        :return: Exchange object containing the user message and the agent response.
        """
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        """
        Clears the agent's memory.
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
