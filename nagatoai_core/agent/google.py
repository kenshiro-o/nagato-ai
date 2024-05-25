from typing import List

import google.generativeai as genai

from .agent import Agent
from .message import Sender, Message, Exchange, ToolResult, ToolCall
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


def extract_google_model_family(model: str) -> str:
    """
    Extracts the model family from the Google model name.
    :param model: The Google model name.
    """
    family_prefixes = [
        "gemini-1.0",
        "gemini-1.5",
        "gemini-pro-vision",
    ]
    for prefix in family_prefixes:
        if model.startswith(prefix):
            return prefix

    return model.split("-")[0]


class GoogleAgent(Agent):
    def __init__(
        self,
        client: genai.GenerativeModel,
        model: str,
        role: str,
        role_description: str,
        nickname: str,
    ):
        """
        Initializes the GoogleAgent with the model, role, temperature and nickname.
        :param client: The Google generative model client to be used by the agent.
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
        tools: List[OpenAIToolProvider],
        temperature: float,
        max_tokens: int,
    ) -> Exchange:
        """
        Generates a response for the current prompt and prompt history.
        :param prompt: The current prompt.
        :param tools: the tools available to the agent.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        :return: Exchange object containing the user message and the agent response."""
        previous_messages = self._build_chat_history()
        current_message = {
            "role": "user",
            "parts": [prompt],
        }
        messages = previous_messages + [current_message]

        # Gemini models do not support a separate "system" role...
        # so we prepend the role description to the user message to act as the system prompt
        messages[0]["parts"].insert(0, f"---\n{self.role_description}---\n")

        gen_config = genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=max_tokens, temperature=temperature
        )

        response: genai.types.GenerateContentResponse = self.client.generate_content(
            messages,
            generation_config=gen_config,
        )

        # TODO - Implement logic to handle tool call responses

        response_text = response.text
        exchange = Exchange(
            user_msg=Message(sender=Sender.USER, content=prompt),
            agent_response=Message(
                sender=Sender.AGENT, content=response_text, tool_calls=[]
            ),
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
        # @TODO Implement this method
        return None

    def _build_chat_history(self) -> List:
        """
        Builds the chat history from the exchange history.
        :return: List of messages in the chat history.
        """
        messages = []
        for i, exchange in enumerate(self.exchange_history):
            user_content = []
            if exchange.user_msg.content:

                user_content.append(
                    {
                        "role": "user",
                        "parts": [exchange.user_msg.content],
                    }
                )

            messages.extend(user_content)

            assistant_content = []
            if exchange.agent_response.content:
                assistant_content.append(
                    {
                        "role": "model",
                        "parts": [exchange.agent_response.content],
                    }
                )

            messages.extend(assistant_content)

        return messages

    @property
    def maker(self) -> str:
        """
        Returns the agent's model maker (e.g. OpenAI)
        """
        return "OpenAI"

    @property
    def family(self) -> str:
        """
        Returns the agent's model family (e.g. Gemini-Pro)
        """
        return extract_google_model_family(self.model)

    @property
    def history(self) -> List[Exchange]:
        """
        Returns the agent's conversation history.
        """
        return self.exchange_history
