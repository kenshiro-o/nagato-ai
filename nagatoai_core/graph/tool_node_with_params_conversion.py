from __future__ import annotations

# Standard Library
import json
import logging
import traceback
from typing import Dict, List

# Third Party
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.graph.tool_node import ToolNode
from nagatoai_core.graph.types import NodeResult


class ToolNodeWithParamsConversion(ToolNode):
    """
    A node that encapsulates a tool within a graph-based system, with parameter conversion capabilities.

    If standard Pydantic validation fails for the tool parameters, this node will use an Agent to
    attempt to convert the input parameters to the required format based on the tool's schema.
    """

    # Agent that will be used for parameter conversion
    agent: Agent

    # Number of times to retry parameter conversion
    retries: int = Field(default=1, description="Number of times to attempt parameter conversion if validation fails")

    # Whether to clear agent memory after conversion
    clear_memory_after_conversion: bool = True

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def _convert_params(self, input_data: Dict, schema_dict: Dict) -> Dict:
        """
        Convert input parameters to the format required by the tool.

        This uses an agent to transform the input data based on the target schema.

        :param input_data: The input data to convert
        :param schema_dict: The target schema in dictionary format
        :return: The converted parameters as a dictionary
        """
        conv_prompt = """Given an input data in a given format, and a target JSON schema, you are able to map out what fields from the input data should correspond to the output data.
        You output an instance of target data based on the original JSON schema. Fields in the output JSON schema that have a default value, and are not overriden from the input data, should be left as such.
        For required fields in the target JSON schema for which there is no corresponding value in the input data, do your best guess at proposing realistic value in the target schema.

        ---
        For example. Given the following <input_data> and <target_schema>:
        <input_data>
        {{
            "file": "test_file.md",
            "created_date": "2024-03-24",
            "created_by": "The Creator"
        }}
        </input_data>

        <target_schema>
        {{
            "name": "get_file_data",
            "description": "Extract information about the file",
            "parameters": {{
                "type": "object",
                "properties": {{
                    "file_path": {{
                        "description": "The path to the file",
                        "type": "string",
                    }},
                    "output_format": {{
                        "description": "The output format for the report. Possible values are 'text' and 'json'. Default is 'json'",
                        "type": "string"
                    }},
                }},
                "required": ["file_path", "output_format"]
            }}
        }}
        </target_schema>

        You should output the following JSON object under the <params_instance> tag:

        <params_instance>
        {{
            "file_path": "test_file.md",
            "output_format": "json",
        }}
        </params_instance>
        As you can see, the 'output_format' field was set 'json' because while it is not present in the input data, 'json' is the default we should use for this field.
        ---
        Another example. Given the following <input_data> and a light schema which is a dictionary of fields and their meaning:
        <input_data>
        {{
            "food": ["melon", "banana", "orange", "grape", "apricot", "apple"],
            "drink": ["water", "juice", "soda", "milk", "tea", "coffee"]
        }}
        </input_data>

        <target_schema>
        {{
            "fruits": "A list of fruits",
        }}
        </target_schema>

        You should output the following JSON object under the <params_instance> tag:
        <params_instance>
        {{
            "fruits": ["melon", "banana", "orange", "grape", "apricot", "apple"]
        }}
        </params_instance>
        As you can see, the 'food' field from the <input_data> tag was set to the list of fruits in the input data because the placeholder variable in the target schema was 'fruits'.
        ---

        Now, given the following <input_data> and <target_schema>:
        <input_data>
            {input_data}
        </input_data>

        <target_schema>
            {target_schema}
        </target_schema>

        Please specify the result inside the <params_instance> tag.
        When it comes to to the value of fields that look like file names, file paths, or other unique identifiers, use information from the input to create unique names (e.g. use index number, id, suffix, number, timestamp, etc.).
        Make sure that we only have valid json inside the <params_instance> tag.
        Additionally DO NOT UNDER ANY CIRCUMSTANCES prefix the json content within the tag with  ```json or ```.
        YOU MUST ENSURE THE RESPONSE STARTS WITH THE <params_instance> TAG AND ENDS WITH THE </params_instance> TAG.
        """

        params_instance: str = ""
        raw_response: str = ""

        try:
            # Format the prompt with the input data and target schema
            full_prompt = conv_prompt.format(input_data=json.dumps(input_data), target_schema=json.dumps(schema_dict))

            self.logger.debug(f"Full prompt to submit to agent converter", prompt=full_prompt)

            # Get a response from the agent
            exchange = self.agent.chat(None, full_prompt, [], 0.6, 2000)
            raw_response = exchange.agent_response.content

            # Extract data from params_instance tag
            soup = BeautifulSoup(raw_response, "html.parser")
            params_instance = soup.find("params_instance").get_text(strip=True)

            # Parse the JSON from the agent's response
            params_json = json.loads(params_instance)

            # Clear the agent's memory if configured to do so
            if self.clear_memory_after_conversion:
                self.agent.clear_memory()

            return params_json

        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"Error converting data: {input_data}. Raw response: {raw_response}. Error: {e}")
            raise e

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Executes the tool with parameters, attempting conversion if validation fails.

        First attempts standard Pydantic validation. If that fails, uses an Agent
        to convert the parameters and tries again.

        :param inputs: List of NodeResult objects containing input parameters
        :return: List of NodeResult objects containing the tool's results
        """
        try:
            # Get the tool and its parameter schema
            tool_instance = self.tool_provider.tool
            tool_params_schema = tool_instance.args_schema

            # Merge input dictionaries into a single parameter object
            obj_params = {}
            for inp in inputs:
                if isinstance(inp.result, dict):
                    # Iterate over each key, and if the type of the value is of instance base model then use model dump
                    for key, value in inp.result.items():
                        # Dump the value if it is a BaseModel
                        if isinstance(value, BaseModel):
                            obj_params[key] = value.model_dump()
                        else:
                            obj_params[key] = value
                # Dump the value if it is a BaseModel at the top level
                elif isinstance(inp.result, BaseModel):
                    obj_params.update(inp.result.model_dump())

            self.logger.info(f"Initial params to tool", tool_params=obj_params, inputs=inputs)

            # First try: standard Pydantic validation
            try:
                tool_params = tool_params_schema(**obj_params)
                self.logger.info(f"Schema generated successfully", tool_params=tool_params)

                # Run the tool with the parameters
                res = tool_instance._run(tool_params)
                return [NodeResult(node_id=self.id, result=res, step=inputs[0].step + 1)]

            except ValidationError as validation_error:
                self.logger.warning(f"Validation error", error=validation_error)
                self.logger.info("Attempting parameter conversion with agent...")

                # Get schema information for the agent to use
                schema_dict = self.tool_provider.schema()
                if isinstance(schema_dict, str):
                    self.logger.info(f"Schema dict is a string", schema_dict=schema_dict)
                    schema_dict = json.loads(schema_dict)

                # Try with agent-based conversion, with retries
                converted_params = None
                last_error = None

                for attempt in range(self.retries + 1):
                    try:
                        self.logger.info(f"Conversion attempt", attempt=attempt + 1, total=self.retries + 1)

                        # Convert parameters using the agent
                        converted_params = self._convert_params(obj_params, schema_dict)
                        self.logger.info(f"Converted params", converted_params=converted_params)

                        # Try to validate the converted parameters
                        tool_params = tool_params_schema(**converted_params)
                        self.logger.info(f"Schema generated successfully after conversion", tool_params=tool_params)

                        # Run the tool with the converted parameters
                        res = tool_instance._run(tool_params)
                        return [NodeResult(node_id=self.id, result=res, step=inputs[0].step + 1)]

                    except Exception as e:
                        self.logger.error(f"Error in conversion attempt", attempt=attempt + 1, error=e)
                        last_error = e

                        # Continue to next retry attempt if available
                        continue

                # If we've exhausted all retries and still failed, raise the last error
                if last_error:
                    raise last_error

        except Exception as e:
            error_msg = f"An error occurred while trying to run the tool node with parameter conversion: {e}"
            self.logger.error(error_msg)
            traceback.print_exc()
            return [NodeResult(node_id=self.id, result=None, error=e, step=inputs[0].step + 1)]
