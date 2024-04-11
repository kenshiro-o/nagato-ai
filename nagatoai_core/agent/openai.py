from openai import OpenAI
from typing import List, Union

from .agent import Agent
from .message import Sender, Message, Exchange
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider


def extract_openai_model_family(model: str) -> str:
    """
    Extracts the model family from the OpenAI model name.
    :param model: The OpenAI model name.
    """
    family_prefixes = ["gpt-4", "gpt-3.5", "dalle" "davinci", "curie", "babbage", "ada"]
    for prefix in family_prefixes:
        if model.startswith(prefix):
            return prefix

    return model.split("-")[0]


class OpenAIAgent(Agent):
    """
    OpenAIAgent is an Agent that uses the OpenAI API under the hood.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        role: str,
        role_description: str,
        nickname: str,
    ):
        """
        Intializes the OpenAIAgent with the model, role, temperature and nickname.
        :param client: The OpenAI client to be used by the agent.
        :param model: The model to be used by the agent.
        :param role: The role of the agent.
        :param role_description: The role description of the agent. This is essentially the system message
        :param nickname: The nickname of the agent.
        """
        super().__init__(model, role, role_description, nickname)
        self.client = client
        self.exchange_history: List[Exchange] = []

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

        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.role_description,
                }
            ],
        }

        previous_messages = []
        for exchange in self.exchange_history:
            previous_messages.append(
                {"role": "user", "content": exchange.user_msg.content}
            )
            previous_messages.append(
                {"role": "assistant", "content": exchange.agent_response.content}
            )

        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
        messages = [system_message] + previous_messages + [current_message]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response_text = response.choices[0].message.content

        exchange = Exchange(
            user_msg=Message(sender=Sender.USER, content=prompt),
            agent_response=Message(sender=Sender.AGENT, content=response_text),
        )
        self.exchange_history.append(exchange)

        return exchange

    @property
    def maker(self) -> str:
        """
        Returns the agent's model maker (e.g. OpenAI)
        """
        return "OpenAI"

    @property
    def family(self) -> str:
        """
        Returns the agent's model family (e.g. GPT-4)
        """
        return extract_openai_model_family(self.model)

    @property
    def history(self) -> List[Exchange]:
        """
        Returns the agent's conversation history.
        """
        return self.exchange_history
