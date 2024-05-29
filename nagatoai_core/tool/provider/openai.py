from typing import Dict, Type, Union

from datetime import datetime

from .abstract_tool_provider import AbstractToolProvider, get_json_schema_type
from ..abstract_tool import TOOL_METADATA_EXCLUDE_FROM_SCHEMA


class OpenAIToolProvider(AbstractToolProvider):
    """
    OpenAIToolProvider is a wrapper for all tools that are compabitible with OpenAI function calling
    functionality.
    """

    def _get_field_schema(self, field_type: Type) -> Dict:
        if hasattr(field_type, "__origin__"):
            if field_type.__origin__ == list:
                return {
                    "type": "array",
                    "items": self._get_field_schema(field_type.__args__[0]),
                }
            elif field_type.__origin__ == Union and type(None) in field_type.__args__:
                non_none_type = next(
                    t for t in field_type.__args__ if t is not type(None)
                )
                union_type = self._get_field_schema(non_none_type)
                return {
                    # "type": self._get_field_schema(non_none_type)["type"],
                    "nullable": True,
                    **union_type,
                }

        if field_type == datetime:
            return {"type": "string", "format": "date-time"}

        return {"type": get_json_schema_type(field_type)}

    def _generate_schema(self) -> Dict:
        """
        Generates the schema for the tool.
        """
        tool = self.tool

        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

        for field_name, field in tool.args_schema.model_fields.items():
            if field.json_schema_extra and field.json_schema_extra.get(
                TOOL_METADATA_EXCLUDE_FROM_SCHEMA, False
            ):
                continue

            field_schema = self._get_field_schema(field.annotation)
            # field_type_name = ""
            # field_type = field.annotation
            # if hasattr(field_type, "__name__"):
            #     field_type_name = field_type.__name__
            # else:
            #     field_type_name = str(field_type).replace("typing.", "")

            # field_type_name = get_json_schema_type(field_type)

            schema["function"]["parameters"]["properties"][field_name] = {
                # "type": field_type_name,
                "description": field.description,
                **field_schema,
            }
            if field.is_required():
                schema["function"]["parameters"]["required"].append(field_name)

        return schema

    def schema(self) -> Dict:
        """
        Returns the schema for the the tool.
        """
        return self._generate_schema()
