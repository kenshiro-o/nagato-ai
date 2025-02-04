# Standard Library
import json
from datetime import datetime, timezone
from typing import List, Optional

# Third Party
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Nagato AI
# Company Libraries
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.openai import OpenAIToolProvider

from .agent import Agent
from .message import Exchange, Message, Sender, TokenStatsAndParams, ToolCall, ToolResult

REASONER_MODEL_NAME = "deepseek-reasoner"


class DeepSeekAgent(Agent):
    """
    DeepSeekAgent is an Agent that uses the DeepSeek API under the hood.
    DeepSeek API is compatible with OpenAI's API format.
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
        Initializes the DeepSeekAgent with the role, role description and nickname.
        :param client: The OpenAI client configured with DeepSeek's base URL
        :param role: The role of the agent
        :param role_description: The role description of the agent (system message)
        :param nickname: The nickname of the agent
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
        tools: List[OpenAIToolProvider],
        temperature: float,
        max_tokens: int,
    ) -> Exchange:
        """
        Generates a response for the current prompt and prompt history.
        :param task: The task object details of the task being run
        :param prompt: The current prompt
        :param tools: The tools available to the agent
        :param temperature: The temperature parameter for generation
        :param max_tokens: The maximum number of tokens to generate
        :return: Exchange object containing the user message and agent response
        """
        previous_messages = self._build_chat_history()

        current_message = {
            "role": "user",
            "content": prompt,
        }
        messages = previous_messages + [current_message]

        msg_send_time = datetime.now(timezone.utc)

        response: Optional[ChatCompletion] = None
        # Reasoner model does not support tools nor temperature setting
        if self.model == REASONER_MODEL_NAME:
            self.logger.debug(
                "Sending request to DeepSeek Reasoner Agent",
                messages=messages,
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
        elif len(tools) > 0:
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

        msg_receive_time = datetime.now(timezone.utc)

        if self.model == REASONER_MODEL_NAME:
            chain_of_thought = response.choices[0].message.reasoning_content
            self.logger.info("Chain of thought", chain_of_thought=chain_of_thought)

        response_text = response.choices[0].message.content
        tool_calls: List[ToolCall] = []

        if response.choices[0].message.tool_calls:
            response_text = response_text + "\n\n" if response_text else ""
            for tool_call in response.choices[0].message.tool_calls:
                self.logger.debug(
                    "Tool call requested",
                    tool_call_id=tool_call.id,
                    tool_call_name=tool_call.function.name,
                    tool_call_parameters=tool_call.function.arguments,
                )

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
            # Handle empty responses
            if not response_text:
                response_text = "OK"

        exchange = Exchange(
            chat_history=messages,
            user_msg=Message(sender=Sender.USER, content=prompt, created_at=msg_send_time),
            agent_response=Message(
                sender=Sender.AGENT,
                content=response_text,
                tool_calls=tool_calls,
                created_at=msg_receive_time,
            ),
            token_stats_and_params=TokenStatsAndParams(
                input_tokens_used=response.usage.prompt_tokens,
                output_tokens_used=response.usage.completion_tokens,
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
        tools: List[OpenAIToolProvider],
        temperature: float,
        max_tokens: int,
    ) -> Exchange:
        """
        Returns the results of running one or multiple tools
        :param task: The task object details of the task being run
        :param tool_results: The results of running one or multiple tools
        :param tools: The tools available to the agent
        :param temperature: The temperature parameter for generation
        :param max_tokens: The maximum number of tokens to generate
        :return: Exchange object containing the tool results and agent response
        """
        messages = []
        try:
            messages = self._build_chat_history()

            final_tool_result_content = ""
            for tool_result in tool_results:
                tool_result_json = json.dumps(tool_result.result, indent=2)
                self.logger.debug(
                    "Tool result for DeepSeek Agent",
                    tool_result_id=tool_result.id,
                    tool_result_name=tool_result.name,
                    tool_result_json=tool_result_json,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.id,
                        "name": tool_result.name,
                        "content": tool_result_json,
                    }
                )

                final_tool_result_content += f"{tool_result_json}\n"

            msg_send_time = datetime.now(timezone.utc)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg_receive_time = datetime.now(timezone.utc)

            response_text = response.choices[0].message.content

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
                    created_at=msg_receive_time,
                ),
                token_stats_and_params=TokenStatsAndParams(
                    input_tokens_used=response.usage.prompt_tokens,
                    output_tokens_used=response.usage.completion_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            self.exchange_history.append(exchange)

            return exchange
        except Exception as e:
            # Prettify the chat history for debugging
            self.logger.error(
                "Error sending request to DeepSeek API",
                chat_history=messages,
                error=str(e),
            )
            raise e

    def _build_chat_history(self) -> List:
        """
        Builds the chat history for the agent.
        :return: List of chat messages, which may include tool calls and tool results
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
        Returns the agent's model maker (DeepSeek)
        """
        return "DeepSeek"

    @property
    def family(self) -> str:
        """
        Returns the agent's model family (deepseek)
        """
        return "deepseek"

    @property
    def history(self) -> List[Exchange]:
        """
        Returns the agent's conversation history
        """
        return self.exchange_history
