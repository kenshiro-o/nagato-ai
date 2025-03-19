# Standard Library
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

# Third Party
from google import genai
from google.genai import types
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from pydantic import BaseModel

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.message import Exchange, Message, Sender, TokenStatsAndParams, ToolCall, ToolResult

# Company Libraries
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.google import GoogleToolProvider


def extract_google_model_family(model: str) -> str:
    """
    Extracts the model family from the Google model name.
    :param model: The Google model name.
    """
    family_prefixes = [
        "gemini-1.0",
        "gemini-1.5",
        "gemini-2.0",
        "gemini-pro-vision",
    ]
    for prefix in family_prefixes:
        if model.startswith(prefix):
            return prefix

    return model.split("-")[0]


class GoogleAgent(Agent):
    def __init__(
        self,
        client: genai.Client,
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

    def clear_memory(self) -> None:
        """
        Clears the agent's memory.
        """
        self.exchange_history = []

    def _serialize_message(self, messages: List) -> Dict:
        """
        Recursively converts protobuf Structs and other non-serializable objects
        to JSON-serializable types.
        """

        def _make_json_serializable(obj):
            """
            Recursively converts protobuf Structs and other non-serializable objects
            to JSON-serializable types.
            """
            if isinstance(obj, Struct):
                return MessageToDict(obj)
            elif isinstance(obj, list):
                return [_make_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _make_json_serializable(v) for k, v in obj.items()}
            else:
                return obj

        return _make_json_serializable(messages)

    def _print_messages(self, messages: List):
        """
        Prints contents of the messages list that we submit to the Gemini model.
        """
        self.logger.debug(
            "Gemini message",
            gemini_message=json.dumps(self._serialize_message(messages), indent=2),
        )

    def chat(
        self,
        task: Optional[Task],
        prompt: str,
        tools: List[GoogleToolProvider],
        temperature: float,
        max_tokens: int,
        target_output_schema: Optional[Type[BaseModel]] = None,
    ) -> Exchange:
        """
        Generates a response for the current prompt and prompt history.
        :param task: The task object details of the task being run.
        :param prompt: The current prompt.
        :param tools: the tools available to the agent.
        :param temperature: The temperature of the agent.
        :param max_tokens: The maximum number of tokens to generate.
        :param target_output_schema: The target output schema for the agent.
        :return: Exchange object containing the user message and the agent response."""
        previous_messages = self._build_chat_history()
        current_message = types.Content(parts=[types.Part.from_text(text=prompt)], role="user")
        messages = previous_messages + [current_message]

        response: Optional[types.GenerateContentResponse] = None

        gen_config = types.GenerateContentConfig(
            system_instruction=self.role_description,
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=temperature,
            tools=[types.Tool(function_declarations=[tool.schema()]) for tool in tools] if len(tools) > 0 else None,
        )

        if target_output_schema:
            gen_config.response_mime_type = "application/json"
            gen_config.response_schema = target_output_schema.model_json_schema()

        # if len(tools) > 0:
        #     gen_config.tools = [tool.schema() for tool in tools]

        # Uncomment if you want to debug
        # self._print_messages(messages)

        self.logger.debug(
            "Gemini message",
            gemini_message=messages,
        )

        msg_send_time = datetime.now(timezone.utc)

        response = self.client.models.generate_content(
            model=self.model,
            config=gen_config,
            contents=messages,
        )

        msg_receive_time = datetime.now(timezone.utc)

        # TODO - Implement logic to handle tool call responses
        self.logger.debug(
            "Gemini response",
            gemini_response=response,
        )

        response_text: str = ""

        tool_calls: List[ToolCall] = []

        fn_call_inputs = []
        for tool in tools:
            fn_call_inputs.append(tool.schema())

        print(f"**** GEMINI INPUT MESSAGE: {messages}")
        print(f"**** GEMINI RESPONSE: {response}")

        if response.function_calls:
            for fn in response.function_calls:
                fn_name = fn.name
                # Note - Gemini models do not provide an id per tool call - so we're going to use the function name instead
                fn_id = fn.id if fn.id else fn.name
                args_dict = {k: v for k, v in fn.args.items()}
                tool_call_msg = f"Tool call requested: function {fn_name} with parameters: {args_dict}"
                print(tool_call_msg)
                # response_text += f"{tool_call_msg}\n"
            tool_calls.append(ToolCall(id=fn_id, name=fn_name, parameters=args_dict))
        else:
            # When there are tool calls, there is often no text response from the model
            if target_output_schema:
                response_text = target_output_schema(**response.parsed)
            else:
                response_text = response.text

        exchange = Exchange(
            chat_history=self._serialize_message(messages),
            user_msg=Message(sender=Sender.USER, content=prompt, created_at=msg_send_time),
            agent_response=Message(
                sender=Sender.AGENT,
                content=response_text,
                tool_calls=tool_calls,
                created_at=msg_receive_time,
            ),
            token_stats_and_params=TokenStatsAndParams(
                input_tokens_used=response.usage_metadata.prompt_token_count,
                output_tokens_used=response.usage_metadata.candidates_token_count,
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
        tools: List[GoogleToolProvider],
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
        fn_res_result_parts: List[types.Part] = []
        for tool_result in tool_results:
            tool_run_result = {
                "result": tool_result.result,
            }

            # Change the response to error if there is an error while calling the tool
            if tool_result.error:
                tool_run_result = {
                    "error": tool_result.error,
                }

            fn_res_result_parts.append(
                types.Part.from_function_response(
                    name=tool_result.name,
                    response=tool_run_result,
                )
            )

        messages.append(types.Content(parts=fn_res_result_parts, role="tool"))

        self.logger.debug(
            "Gemini message with tool results",
            gemini_message=messages,
        )

        # print(
        #     f"**** Message to send to Gemini with tool results: {json.dumps(self._serialize_message(messages), indent=2)}"
        # )

        gen_config = types.GenerateContentConfig(
            candidate_count=1, max_output_tokens=max_tokens, temperature=temperature
        )

        msg_send_time = datetime.now(timezone.utc)
        response = self.client.models.generate_content(
            model=self.model,
            config=gen_config,
            contents=messages,
        )
        msg_receive_time = datetime.now(timezone.utc)

        self.logger.debug(
            "Gemini response with tool results",
            gemini_response=response,
        )

        response_text: str = response.text

        exchange = Exchange(
            chat_history=self._serialize_message(messages),
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
                input_tokens_used=response.usage_metadata.prompt_token_count,
                output_tokens_used=response.usage_metadata.candidates_token_count,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

        self.exchange_history.append(exchange)

        return exchange

    def _build_chat_history(self) -> List:
        """
        Builds the chat history from the exchange history.
        :return: List of messages in the chat history.
        """
        messages = []
        for exchange in self.exchange_history:
            user_content = []

            if exchange.user_msg.tool_results:
                fn_call_res_parts = []
                for tool_result in exchange.user_msg.tool_results:
                    struct_response = Struct()

                    # Make sure we are sending over a dictionary
                    tool_run_result = {
                        "result": tool_result.result,
                    }
                    struct_response.update(tool_run_result)
                    fn_call_res_parts.append(
                        {
                            "function_response": {
                                "name": tool_result.name,
                                "response": struct_response,
                            }
                        }
                    )
                user_content.append(
                    {
                        "role": "function",
                        "parts": fn_call_res_parts,
                    }
                )

            if exchange.user_msg.content:
                self.logger.debug(
                    "User content",
                    user_content=exchange.user_msg.content,
                )
                user_content.append(
                    types.Content(parts=[types.Part.from_text(text=exchange.user_msg.content)], role="user")
                )

            messages.extend(user_content)

            assistant_content = []
            if exchange.agent_response.tool_calls:
                tool_call_parts = []
                for tool_call in exchange.agent_response.tool_calls:
                    struct_fn_args = Struct()
                    struct_fn_args.update(tool_call.parameters)
                    tool_call_parts.append(
                        {
                            "function_call": {
                                "name": tool_call.name,
                                "args": struct_fn_args,
                            }
                        }
                    )
                assistant_content.append(
                    {
                        "role": "model",
                        "parts": tool_call_parts,
                    }
                )

            if exchange.agent_response.content:
                assistant_content.append(
                    types.Content(parts=[types.Part.from_text(text=exchange.agent_response.content)], role="model")
                )
                # assistant_content.append(
                #     {
                #         "role": "model",
                #         "parts": [exchange.agent_response.content],
                #     }
                # )

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
