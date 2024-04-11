from typing import Dict

from .abstract_tool_provider import AbstractToolProvider, get_json_schema_type
from ..abstract_tool import TOOL_METADATA_EXCLUDE_FROM_SCHEMA


class AnthropicToolProvider(AbstractToolProvider):
    """
    AnthropicToolProvider is a wrapper for all tools that are compabitible with Anthropic tool calling
    functionality.
    """

    def _generate_schema(self) -> Dict:
        """
        Generates the schema for the tool.
        """
        tool = self.tool

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

    def schema(self) -> Dict:
        """
        Returns the schema for the the tool.
        """
        return self._generate_schema()