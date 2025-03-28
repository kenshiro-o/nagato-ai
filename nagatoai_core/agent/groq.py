# Standard Library
import json
from datetime import datetime, timezone
from typing import List, Optional

# Third Party
from groq import Groq
from groq.types.chat import ChatCompletion
from pydantic import BaseModel

# Nagato AI
# Company Libraries
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.openai import OpenAIToolProvider  # Groq tool calling mechanism is similar to OpenAI's

from .agent import Agent
from .message import Exchange, Message, Sender, TokenStatsAndParams, ToolCall, ToolResult


def extract_groq_model_family(model: str) -> str:
    """
    Extracts the model family from the Groq model name.
    :param model: The Groq model name.
    """
    family_prefixes = [
        "qwen-",
        "deepseek-r1",
        "llama-3.1-8b",
        "llama-3.1-70b",
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
        target_output_schema: Optional[BaseModel] = None,
    ) -> Exchange:
        """
        Generates a response for the current prompt and prompt history.
        :param task: The task object details of the task being run.
        :param prompt: The current prompt.
        :param tools: the tools available to the agent.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        :param target_output_schema: An optional Pydantic model schema for structured output.
        :return: Exchange object containing the user message and the agent response.
        """
        previous_messages = self._build_chat_history()

        if target_output_schema:
            # Groq doesn't have a native structured output option
            # So we inject the schema into the prompt
            prompt = f"""{prompt}

            If not calling a tool then return a JSON object with the schema:
            {target_output_schema.model_json_schema()}

            Do not return anything apart from the JSON object, not even ``` or ```json.
            """

        current_message = {
            "role": "user",
            "content": prompt,
        }
        messages = previous_messages + [current_message]

        response: Optional[ChatCompletion] = None

        msg_send_time = datetime.now(timezone.utc)
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
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        msg_receive_time = datetime.now(timezone.utc)

        response_text = response.choices[0].message.content
        tool_calls: List[ToolCall] = []

        self.logger.debug(f"Groq chat response", response=response_text, prompt=prompt)

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
        elif target_output_schema and response_text:
            try:
                # Parse the response as JSON and convert it to the target schema
                response_json = json.loads(response_text)
                response_text = target_output_schema(**response_json)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                # If parsing fails, keep the original text response
                pass

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
                temperature=temperature,
                max_tokens=max_tokens,
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
                temperature=temperature,
                max_tokens=max_tokens,
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
        return "Groq"

    @property
    def family(self) -> str:
        return extract_groq_model_family(self.model)

    @property
    def history(self) -> List[Exchange]:
        return self.exchange_history
