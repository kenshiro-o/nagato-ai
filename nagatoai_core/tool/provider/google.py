# Standard Library
from typing import Dict, Type, Union

# Third Party
from google.genai import types

from ..abstract_tool import TOOL_METADATA_EXCLUDE_FROM_SCHEMA
from .abstract_tool_provider import AbstractToolProvider, get_json_schema_type


class GoogleToolProvider(AbstractToolProvider):
    """
    GoogleToolProvider is a wrapper for all tools that are compatible with Google's function calling
    functionality.
    """

    def _get_field_schema(self, field_type: Type) -> types.Schema:
        if hasattr(field_type, "__origin__"):
            if field_type.__origin__ == list:
                return types.Schema(
                    type="ARRAY",
                    items=self._get_field_schema(field_type.__args__[0]),
                )
            elif field_type.__origin__ == Union and type(None) in field_type.__args__:
                non_none_type = next(t for t in field_type.__args__ if t is not type(None))
                union_schema = self._get_field_schema(non_none_type)
                union_schema.nullable = True
                return union_schema

        return types.Schema(type=get_json_schema_type(field_type).upper())

    def _generate_schema(self) -> types.FunctionDeclaration:
        """
        Generates the schema for the tool as a FunctionDeclaration.
        """
        tool = self.tool
        properties = {}
        required = []

        for field_name, field in tool.args_schema.model_fields.items():
            if field.json_schema_extra and field.json_schema_extra.get(TOOL_METADATA_EXCLUDE_FROM_SCHEMA, False):
                continue

            properties[field_name] = types.Schema(
                type=get_json_schema_type(field.annotation).upper(),
                description=field.description,
            )

            if field.is_required():
                required.append(field_name)

        return types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=types.Schema(
                type="OBJECT",
                properties=properties,
                required=required,
            ),
        )

    def schema(self) -> types.FunctionDeclaration:
        """
        Returns the schema for the tool as a FunctionDeclaration.
        """
        return self._generate_schema()
