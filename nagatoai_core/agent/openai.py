from typing import List, Optional
import json

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .agent import Agent
from .message import Sender, Message, Exchange, ToolResult, ToolCall
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


def extract_openai_model_family(model: str) -> str:
    """
    Extracts the model family from the OpenAI model name.
    :param model: The OpenAI model name.
    """
    family_prefixes = [
        "gpt-4o",
        "gpt-4",
        "gpt-3.5",
        "dalle" "davinci",
        "curie",
        "babbage",
        "ada",
    ]
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
        :return: Exchange object containing the user message and the agent response.
        """
        previous_messages = self._build_chat_history()

        current_message = {
            "role": "user",
            "content": prompt,
        }
        messages = previous_messages + [current_message]

        response: Optional[ChatCompletion] = None
        if len(tools) > 0:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=[tool.schema() for tool in tools],
                tool_choice="auto",
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        response_text = response.choices[0].message.content
        tool_calls: List[ToolCall] = []

        if response.choices[0].message.tool_calls:
            response_text = response_text + "\n\n" if response_text else ""
            for tool_call in response.choices[0].message.tool_calls:
                params_json = json.loads(tool_call.function.arguments)
                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        parameters=params_json,
                    )
                )
                response_text += f"Tool call requested: {tool_call.function.name} with parameters: {params_json}\n"
        else:
            # TODO - dirty hack to avoid empty responses. In the future find a more elegant solution
            if not response_text:
                response_text = "OK"

        exchange = Exchange(
            user_msg=Message(sender=Sender.USER, content=prompt),
            agent_response=Message(
                sender=Sender.AGENT, content=response_text, tool_calls=tool_calls
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
        messages = self._build_chat_history()

        final_tool_result_content = ""
        for tool_result in tool_results:
            tool_result_json = json.dumps(tool_result.result, indent=2)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result.id,
                    "name": tool_result.name,
                    "content": tool_result_json,
                }
            )

            final_tool_result_content += f"{tool_result_json}\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response_text = response.choices[0].message.content

        exchange = Exchange(
            user_msg=Message(
                sender=Sender.TOOL_RESULT,
                content=final_tool_result_content,
                tool_results=tool_results,
            ),
            agent_response=Message(
                sender=Sender.AGENT,
                content=response_text,
            ),
        )

        self.exchange_history.append(exchange)

        return exchange

    def _build_chat_history(self) -> List:
        """
        Builds the chat history for the agent.
        :return: List of chat messages, which may include tool calls and tool results.
        """
        system_message = {
            "role": "system",
            "content": self.role_description,
        }

        messages = [system_message]

        for exchange in self.exchange_history:
            user_content = []
            if exchange.user_msg.tool_results:
                for tool_result in exchange.user_msg.tool_results:
                    user_content.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.id,
                            "name": tool_result.name,
                            "content": str(tool_result.result),
                        }
                    )
            if exchange.user_msg.content:
                user_content.append(
                    {
                        "role": "user",
                        "content": exchange.user_msg.content,
                    }
                )

            messages.extend(user_content)

            assistant_content = []
            if exchange.agent_response.tool_calls:
                tool_calls_list = []
                for tool_call in exchange.agent_response.tool_calls:
                    tool_calls_list.append(
                        {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.parameters),
                            },
                            "type": "function",
                        }
                    )

                assistant_content.append(
                    {
                        "role": "assistant",
                        "tool_calls": tool_calls_list,
                    }
                )
            elif exchange.agent_response.content:
                assistant_content.append(
                    {
                        "role": "assistant",
                        "content": exchange.agent_response.content,
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
        Returns the agent's model family (e.g. GPT-4)
        """
        return extract_openai_model_family(self.model)

    @property
    def history(self) -> List[Exchange]:
        """
        Returns the agent's conversation history.
        """
        return self.exchange_history
