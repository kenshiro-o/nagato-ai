from typing import Any, Dict
import json
import traceback

from pydantic import BaseModel
from bs4 import BeautifulSoup

from nagatoai_core.agent.agent import Agent


class AgentParamConverter(BaseModel):
    """
    AgentParamConverterLink takes an input value and converts it to
    the target argument format by using the target schema specification
    """

    agent: Agent
    clear_memory_after_conversion: bool = True

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def convert(self, params_data: Dict) -> Any:
        """
        Convert the data to a different data type
        :param input_data: input data that will be converted via the adapter
        :return: The output data that will be converted via the adapter
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
        <params_instance>
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

        Please specify the result inside the <params_instance> tag:
        """

        try:
            input_data = params_data["input_data"]
            target_schema = params_data["target_schema"]

            full_prompt = conv_prompt.format(
                input_data=input_data, target_schema=target_schema
            )

            exchange = self.agent.chat(None, full_prompt, [], 0.6, 2000)
            resp = exchange.agent_response.content

            # print(f"**** Full prompt: {full_prompt}")
            # print(f"**** Response: {resp}")

            # Extract data from params_instance tag
            soup = BeautifulSoup(resp, "html.parser")
            params_instance = soup.find("params_instance").get_text(strip=True)

            params_json = json.loads(params_instance)

            if self.clear_memory_after_conversion:
                self.agent.clear_memory()

            return params_json
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error converting data {params_data}:  {e}")
            raise e
