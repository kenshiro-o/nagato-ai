from anthropic import Client
from typing import List, Optional
import json

from datetime import datetime, timezone

from .agent import Agent
from .message import (
    Sender,
    Message,
    Exchange,
    ToolCall,
    ToolResult,
    TokenStatsAndParams,
)
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.anthropic import AnthropicToolProvider


def extract_anthropic_model_family(model: str) -> str:
    """
    Extracts the model family from the Anthropic model name.
    :param model: The Anthropic model name.
    """
    family_prefixes = [
        "claude-3",
    ]
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

    def clear_memory(self) -> None:
        """
        Clears the agent's memory.
        """
        self.exchange_history = []

    def chat(
        self,
        task: Optional[Task],
        prompt: str,
        tools: List[AnthropicToolProvider],
        temperature: float,
        max_tokens: int,
    ) -> Exchange:
        """
        Generates a response for the current prompt and prompt history.
        :param task: The task object details of the task being run.
        :param prompt: The current prompt.
        :param tools: the tools available to the agent.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        """
        messages = self._build_chat_history()
        messages.append({"role": "user", "content": prompt})

        msg_send_time = datetime.now(timezone.utc)
        if len(tools) == 0:
            response = self.client.messages.create(
                model=self.model,
                system=self.role_description,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.client.beta.tools.messages.create(
                model=self.model,
                system=self.role_description,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=[tool.schema() for tool in tools],
            )

        msg_receive_time = datetime.now(timezone.utc)

        response_text = ""
        tool_calls: List[ToolCall] = []

        for response_content in response.content:
            if response_content.type == "text":
                # TODO - dirty hack to avoid empty responses. In the future find a more elegant solution
                response_text = response_content.text if response_content.text else "OK"
            elif response_content.type == "tool_use":
                tool_id = response_content.id
                tool_name = response_content.name
                tool_input = response_content.input

                tc = ToolCall(id=tool_id, name=tool_name, parameters=tool_input)
                tool_calls.append(tc)

        exchange = Exchange(
            chat_history=messages,
            user_msg=Message(
                sender=Sender.USER, content=prompt, created_at=msg_send_time
            ),
            agent_response=Message(
                sender=Sender.AGENT,
                content=response_text,
                tool_calls=tool_calls,
                created_at=msg_receive_time,
            ),
            token_stats_and_params=TokenStatsAndParams(
                input_tokens_used=response.usage.input_tokens,
                output_tokens_used=response.usage.output_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        self.exchange_history.append(exchange)

        return exchange

    def send_tool_run_results(
        self,
        task: Optional[Task],
        tool_results: List[ToolResult],
        tools: List[AnthropicToolProvider],
        temperature: float,
        max_tokens: int,
    ) -> Exchange:
        """
        Returns the results of the running of one or multiple tools
        :param task: The task object details of the task being run.
        :param tool_results: The results of the running of one or multiple tools
        :param tools: the tools available to the agent.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        :return: Exchange object containing the user message and the agent response.
        """
        messages = self._build_chat_history()

        final_tool_result_content = ""
        for tool_result in tool_results:
            # Use string representation of the result - if it is a complex object, it will be using JSON dumps
            result_str = ""
            # If tool result is a dictionary or list, we will use JSON dumps to convert it to a string
            if isinstance(tool_result.result, dict) or isinstance(
                tool_result.result, list
            ):
                result_str = json.dumps(tool_result.result)
            else:
                result_str = str(tool_result.result)

            final_tool_result_content += result_str + "\n"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_result.id,
                            "content": result_str,
                        }
                    ],
                }
            )

        msg_send_time = datetime.now(timezone.utc)
        response = self.client.beta.tools.messages.create(
            model=self.model,
            system=self.role_description,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[tool.schema() for tool in tools],
        )
        msg_receive_time = datetime.now(timezone.utc)

        response_text = ""
        tool_calls: List[ToolCall] = []

        for response_content in response.content:
            if response_content.type == "text":
                response_text = response_content.text
            elif response_content.type == "tool_use":
                tool_id = response_content.id
                tool_name = response_content.name
                tool_input = response_content.input

                tc = ToolCall(id=tool_id, name=tool_name, parameters=tool_input)
                tool_calls.append(tc)

        if not response_text:
            # Hack to avoid empty responses
            response_text = "OK"

        exchange = Exchange(
            chat_history=messages,
            user_msg=Message(
                sender=Sender.TOOL_RESULT,
                content=final_tool_result_content,
                tool_results=tool_results,
                created_at=msg_send_time,
            ),
            agent_response=Message(
                sender=Sender.AGENT,
                content=response_text,
                tool_calls=tool_calls,
                created_at=msg_receive_time,
            ),
            token_stats_and_params=TokenStatsAndParams(
                input_tokens_used=response.usage.input_tokens,
                output_tokens_used=response.usage.output_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        self.exchange_history.append(exchange)

        return exchange

    def _build_chat_history(self) -> List:
        """
        Builds the chat history for the agent.
        :return: List of chat messages, which may include tool calls and tool results.
        """
        messages = []

        for exchange in self.exchange_history:
            user_content = []
            if exchange.user_msg.tool_results:
                for tool_result in exchange.user_msg.tool_results:
                    user_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_result.id,
                            "content": str(tool_result.result),
                        }
                    )
            if exchange.user_msg.content:
                user_content.append(
                    {
                        "type": "text",
                        "text": exchange.user_msg.content,
                    }
                )

            messages.append({"role": "user", "content": user_content})

            assistant_content = []
            if exchange.agent_response.content:
                assistant_content.append(
                    {
                        "type": "text",
                        "text": exchange.agent_response.content,
                    }
                )
            if exchange.agent_response.tool_calls:
                for tool_call in exchange.agent_response.tool_calls:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.parameters,
                        }
                    )

            messages.append({"role": "assistant", "content": assistant_content})

        return messages

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
