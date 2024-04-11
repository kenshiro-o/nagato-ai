from abc import ABC, abstractmethod
from typing import Type, Dict

from pydantic import BaseModel

from nagatoai_core.tool.abstract_tool import AbstractTool
from .abstract_tool import TOOL_METADATA_EXCLUDE_FROM_SCHEMA


def get_json_schema_type(python_type):
    """
    Converts a Python type to a JSON schema type.
    """
    type_mappings = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    return type_mappings.get(python_type, "string")


class AbstractToolSchemaGenerator(BaseModel, ABC):
    """
    AbstractToolSchemaGenerator represents the base class for all Tool schema generators
    """

    @abstractmethod
    def generate_schema(self, tool: Type[AbstractTool]) -> Dict:
        """
        Generates the schema for the tool.
        :return: The schema for the tool.
        """
        pass


class AnthropicToolSchemaGenerator(AbstractToolSchemaGenerator):
    """
    AnthropicToolSchemaGenerator represents the schema generator for Anthropic tools.
    """

    def generate_schema(self, tool: Type[AbstractTool]) -> Dict:
        """
        Generates the schema for the tool.
        :return: The schema for the tool.
        """
        schema = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

        for field_name, field in tool.args_schema.model_fields.items():
            # print(
            #     f"field_name: {field_name} | field: {field} | annotation: {field.annotation} | metadata: {field.metadata}"
            # )
            if field.json_schema_extra and field.json_schema_extra.get(
                TOOL_METADATA_EXCLUDE_FROM_SCHEMA, False
            ):
                continue

            field_type_name = ""
            field_type = field.annotation
            if hasattr(field_type, "__name__"):
                field_type_name = field_type.__name__
            else:
                field_type_name = str(field_type).replace("typing.", "")

            field_type_name = get_json_schema_type(field_type)

            schema["input_schema"]["properties"][field_name] = {
                "type": field_type_name,
                "description": field.description,
            }
            if field.is_required():
                schema["input_schema"]["required"].append(field_name)

        return schema
