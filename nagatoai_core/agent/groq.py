from typing import List, Optional
import json

from groq import Groq
from openai.types.chat import ChatCompletion

from .agent import Agent
from .message import Sender, Message, Exchange, ToolResult, ToolCall
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider


def extract_groq_model_family(model: str) -> str:
    """
    Extracts the model family from the Groq model name.
    :param model: The Groq model name.
    """
    family_prefixes = [
        "llama3-8b",
        "llama3-70b",
        "llama2-70b",
        "mixtral-8x7b",
        "gemma-7b",
    ]
    for prefix in family_prefixes:
        if model.startswith(prefix):
            return prefix

    return model.split("-")[0]


class GroqAgent(Agent):
    """
    GroqAgent is an Agent that uses the Groq API under the hood.
    """

    def __init__(
        self,
        client: Groq,
        model: str,
        role: str,
        role_description: str,
        nickname: str,
    ):
        """
        Intializes the GroqAgent with the model, role, temperature and nickname.
        :param client: The Groq client to be used by the agent.
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
        previous_messages = self._build_chat_history()

        current_message = {
            "role": "user",
            "content": prompt,
        }
        messages = previous_messages + [current_message]

        response: Optional[ChatCompletion] = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
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
        # TODO - to implement
        return None

    def _build_chat_history(self) -> List:
        system_message = {
            "role": "system",
            "content": self.role_description,
        }

        messages = [system_message]

        for exchange in self.exchange_history:
            user_message = {
                "role": "user",
                "content": exchange.user_msg.content,
            }
            assistant_message = {
                "role": "assistant",
                "content": exchange.agent_response.content,
            }
            messages.extend([user_message, assistant_message])

        return messages

    @property
    def maker(self) -> str:
        return "Groq"

    @property
    def family(self) -> str:
        return extract_groq_model_family(self.model)

    @property
    def history(self) -> List[Exchange]:
        return self.exchange_history
