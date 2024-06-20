from typing import List, Optional
import json

import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict


from .agent import Agent
from .message import Sender, Message, Exchange, ToolResult, ToolCall
from nagatoai_core.tool.provider.google import GoogleToolProvider


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

    def _print_messages(self, messages: List):
        """
        Prints contents of the messages list that we submit to the Gemini model.
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

        print(
            f"****GEMINI MESSAGE: {json.dumps(_make_json_serializable(messages), indent=2)}"
        )

    def chat(
        self,
        prompt: str,
        tools: List[GoogleToolProvider],
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

        response: Optional[genai.types.GenerateContentResponse] = None

        gen_config = genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=max_tokens, temperature=temperature
        )

        # Uncomment if you want to debug
        # self._print_messages(messages)

        if len(tools) > 0:
            response = self.client.generate_content(
                messages,
                generation_config=gen_config,
                tools=[tool.schema() for tool in tools],
            )
        else:
            response = self.client.generate_content(
                messages,
                generation_config=gen_config,
            )

        # TODO - Implement logic to handle tool call responses

        response_text: str = ""
        tool_calls: List[ToolCall] = []

        for part in response.parts:
            if fn := part.function_call:
                fn_name = fn.name
                args_dict = {k: v for k, v in fn.args.items()}
                tool_call_msg = f"Tool call requested: function {fn_name} with parameters: {args_dict}"
                response_text += f"{tool_call_msg}\n"
                # Note - Gemini models do not provide an id per tool call - so we're going to use the function name instead
                tool_calls.append(
                    ToolCall(id=fn_name, name=fn_name, parameters=args_dict)
                )
            else:
                response_text += part.text

        exchange = Exchange(
            user_msg=Message(sender=Sender.USER, content=prompt),
            agent_response=Message(
                sender=Sender.AGENT,
                content=response_text,
                tool_calls=tool_calls,
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
        fn_res_result_parts = []
        for tool_result in tool_results:
            tool_result_json = json.dumps(tool_result.result, indent=2)

            # We must use a protobuf Struct for the tool response result
            struct_response = Struct()
            # Make sure we are sending over a dictionary
            tool_run_result = {
                "result": tool_result.result,
            }
            struct_response.update(tool_run_result)

            fn_res_result_parts.append(
                {
                    "function_response": {
                        "name": tool_result.name,
                        "response": struct_response,
                    }
                }
            )
            final_tool_result_content += f"{tool_result_json}\n"

        messages.append(
            {
                "role": "function",
                "parts": fn_res_result_parts,
            }
        )

        gen_config = genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=max_tokens, temperature=temperature
        )

        response = self.client.generate_content(
            messages,
            generation_config=gen_config,
        )

        response_text: str = response.text

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
                user_content.append(
                    {
                        "role": "user",
                        "parts": [exchange.user_msg.content],
                    }
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
