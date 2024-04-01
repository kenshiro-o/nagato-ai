from anthropic import Client
from typing import List, Union

from .agent import Agent
from .message import Sender, Message, Exchange
from mission.task import Task


def extract_anthropic_model_family(model: str) -> str:
    """
    Extracts the model family from the Anthropic model name.
    :param model: The Anthropic model name.
    """
    family_prefixes = ["claude", "einstein"]
    for prefix in family_prefixes:
        if model.startswith(prefix):
            return prefix

    return model.split("-")[0]


class AnthropicAgent(Agent):
    """
    AnthropicAgent is an Agent that uses the Anthropic Claude API under the hood.
    """

    def __init__(
        self,
        client: Client,
        model: str,
        role: str,
        role_description: str,
        nickname: str,
    ):
        """
        Initializes the AnthropicAgent with the model, role, temperature and nickname.
        :param client: The Anthropic Claude client to be used by the agent.
        :param model: The model to be used by the agent.
        :param role: The role of the agent.
        :param role_description: The role description of the agent. This is essentially the system message
        :param nickname: The nickname of the agent.
        """
        super().__init__(model, role, role_description, nickname)
        self.client = client
        self.exchange_history: List[Exchange] = []

    # def chat(
    #         self, prompt: str, task: Union[Task, None], temperature: float, max_tokens: int
    #     ) -> Exchange:

    def chat(
        self, prompt: str, task: Union[Task, None], temperature: float, max_tokens: int
    ) -> Exchange:
        """
        Generates a response for the current prompt and prompt history.
        :param prompt: The current prompt.
        :param task: The task to reason about.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        """
        messages = []

        for exchange in self.exchange_history:
            messages.append({"role": "user", "content": exchange.user_msg.content})
            messages.append(
                {"role": "assistant", "content": exchange.agent_response.content}
            )

        messages.append({"role": "user", "content": prompt})

        response = self.client.messages.create(
            model=self.model,
            system=self.role_description,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response_text = response.content[0].text

        exchange = Exchange(
            user_msg=Message(sender=Sender.USER, content=prompt),
            agent_response=Message(sender=Sender.AGENT, content=response_text),
        )
        self.exchange_history.append(exchange)

        return exchange

    @property
    def maker(self) -> str:
        """
        Returns the agent's model maker (e.g. Anthropic)
        """
        return "Anthropic"

    @property
    def family(self) -> str:
        """
        Returns the agent's model family (e.g. claude)
        """
        return extract_anthropic_model_family(self.model)

    @property
    def history(self) -> List[Exchange]:
        """
        Returns the agent's conversation history.
        """
        return self.exchange_history
