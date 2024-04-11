from abc import abstractmethod
from typing import Any, Type
import json

from pydantic import BaseModel
from ..abstract_tool import AbstractTool


def get_json_schema_type(python_type: Type) -> str:
    """
    Returns the JSON schema type for the given Python type.
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


class AbstractToolProvider(AbstractTool, BaseModel):
    """
    AbstractToolProvider represents the base class for all tool providers.
    It primarily defines a schema method that returns the schema for the tool.
    """

    tool: AbstractTool

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the tool with the given arguments.
        :param args: The arguments to run the tool with.
        :param kwargs: The arguments to run the tool with.
        :return: The result of running the tool.
        """
        return self.tool._run(*args, **kwargs)

    @abstractmethod
    def schema(self) -> Any:
        """
        Returns the schema for the the tool.
        """
        pass

    def __str__(self) -> str:
        """
        Returns the string representation of the tool provider.
        """
        schema = self.schema()
        return json.dumps(schema, indent=2)
